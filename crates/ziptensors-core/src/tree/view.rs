//! File: tree/view.rs
//! Purpose: the read-only view of a tree with per-file containers: which file is plain, which is a
//! per-file container and under what name it restores, which names collide. The single owner of
//! container recognition and name mapping for `compress_tree`, `decompress_tree` and the overlay
//! mount.
//!
//! Main responsibilities:
//! - walk the tree through the shared `write::inputs::walk`, so the link and partial-download rules
//!   are those of every walk;
//! - recognize a container by its magic, never by its extension, and read its index only;
//! - map a per-file container to the path its member restores to, and find what is already there;
//! - list the names two entries would both claim ([`Conflict`]).
//!
//! Key structures: [`TreeView`], [`ViewEntry`], [`PerFileEntry`], [`Conflict`], [`FileStamp`],
//! [`ViewOptions`].
//!
//! Key functions: [`scan_view`], and the crate-private [`inspect_file`] for a single path.
//!
//! Implementation notes: whether the restore path of a container is taken is asked of the file
//! system (`symlink_metadata`), not looked up among the walked names, so a file system that ignores
//! case finds the file a restore would replace. Each container is opened with one thread and
//! dropped after its index is read.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::error::Error;
use crate::format::{MAGIC, check_restore_name};
use crate::io::{FileSource, RandomAccess, TEMP_FILE_SUFFIX};
use crate::read::{Container, OpenOptions};
use crate::write::inputs::{WalkKind, walk};

/// How [`scan_view`] opens containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ViewOptions {
    /// Budget on the sum of member lengths of each container, in bytes, as
    /// [`OpenOptions::max_total_decoded`]. A container over it is [`ViewEntry::Broken`]. The
    /// default is `u64::MAX`, no budget: a caller scanning untrusted files must set one.
    pub max_total_decoded: u64,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self { max_total_decoded: u64::MAX }
    }
}

/// Length and modification time of a file, to tell later whether it changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStamp {
    /// Length in bytes.
    pub len: u64,
    /// Modification time.
    pub modified: SystemTime,
}

impl FileStamp {
    /// The stamp of `path` itself, a link not followed.
    ///
    /// # Errors
    /// [`Error::Io`] with op `"metadata"`.
    pub fn of(path: &Path) -> Result<Self, Error> {
        let failed = |source| Error::Io { op: "metadata", path: path.to_path_buf(), source };
        let meta = fs::symlink_metadata(path).map_err(failed)?;
        Ok(Self { len: meta.len(), modified: meta.modified().map_err(failed)? })
    }
}

/// Why a container is not a per-file container.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ForeignReason {
    /// It holds this many members, not one: a pack.
    MemberCount(usize),
    /// Its one member's name is not a single safe file name (it has `/` or `\`, is `.`, or fails
    /// [`check_restore_name`]), so it cannot be restored beside the container.
    NotOneSafeName(String),
}

/// Why a path was left out without being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SkipReason {
    /// A symbolic link; links are not followed unless asked.
    Symlink,
    /// A followed link that points to a directory, which is never entered.
    SymlinkToDirectory,
    /// A followed link whose target does not exist.
    DanglingSymlink,
    /// A FIFO, socket or device, or a followed link to one.
    Special,
    /// A temporary file of an interrupted write (`*.ziptensors-tmp`). Never removed automatically,
    /// since it cannot be proven to be ours and unused.
    StaleTemp,
    /// A partial download (`*.incomplete`).
    Incomplete,
    /// A name that is not valid UTF-8; a directory of such a name is not entered.
    NonUtf8,
}

/// A per-file container: one member with a safe single-component name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerFileEntry {
    /// The container file.
    pub container: PathBuf,
    /// Its name relative to the view's base, `/`-separated.
    pub container_rel: String,
    /// The path its member restores to: the container's directory joined with the member name.
    pub presented: PathBuf,
    /// `presented` relative to the view's base, `/`-separated.
    pub rel_path: String,
    /// The member name, which is authoritative: a container need not be named after it.
    pub member_name: String,
    /// The member's length in bytes.
    pub member_len: u64,
    /// The member's stated BLAKE3.
    pub blake3: [u8; 32],
    /// The container file's stamp when it was read.
    pub stamp: FileStamp,
}

/// One path of a [`TreeView`]. Relative names are `/`-separated and relative to the view's base:
/// the root, or the root's directory when the root is not a directory.
#[derive(Debug)]
#[non_exhaustive]
pub enum ViewEntry {
    /// A directory below the root.
    Directory {
        /// The directory.
        path: PathBuf,
        /// Its relative name.
        rel_path: String,
        /// Its modification time.
        modified: SystemTime,
    },
    /// A regular file that is not a container.
    Plain {
        /// The file.
        path: PathBuf,
        /// Its relative name.
        rel_path: String,
        /// Its stamp when it was looked at.
        stamp: FileStamp,
    },
    /// A per-file container whose restore path is free.
    PerFile(PerFileEntry),
    /// A per-file container whose restore path is taken by a file or link: the pair a `keep` run
    /// leaves, or a different file of that name.
    Shadowed(PerFileEntry),
    /// A container that is not a per-file container.
    Foreign {
        /// The container.
        path: PathBuf,
        /// Its relative name.
        rel_path: String,
        /// Why it is not a per-file container.
        reason: ForeignReason,
    },
    /// A file that cannot be read, or that starts with the container magic and does not open.
    Broken {
        /// The file.
        path: PathBuf,
        /// Its relative name.
        rel_path: String,
        /// Why it cannot be read or opened.
        error: Error,
    },
    /// A path left out without being read.
    Skipped {
        /// The path.
        path: PathBuf,
        /// Its relative name, shown lossily for [`SkipReason::NonUtf8`].
        rel_path: String,
        /// Why it was left out.
        reason: SkipReason,
    },
}

impl ViewEntry {
    /// The path of the entry itself; for a container, the container file.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::Directory { path, .. }
            | Self::Plain { path, .. }
            | Self::Foreign { path, .. }
            | Self::Broken { path, .. }
            | Self::Skipped { path, .. } => path,
            Self::PerFile(entry) | Self::Shadowed(entry) => &entry.container,
        }
    }
}

/// Why names collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConflictReason {
    /// Several containers of one directory restore to the same name.
    SameName,
    /// A container restores to the name of a directory.
    Directory,
    /// A container restores to the name of a container file.
    ContainerFile,
}

/// A name claimed twice: restoring every container would write one path twice or over something
/// that must stay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    /// The contested path.
    pub presented: PathBuf,
    /// The containers that restore to it.
    pub containers: Vec<PathBuf>,
    /// Why this is a conflict.
    pub reason: ConflictReason,
}

impl Conflict {
    /// A sentence naming every file involved, for [`Error::TreeConflict`].
    pub(crate) fn describe(&self) -> String {
        let containers: Vec<String> =
            self.containers.iter().map(|path| path.display().to_string()).collect();
        let what = match self.reason {
            ConflictReason::SameName => "several containers restore to",
            ConflictReason::Directory => "a container restores to the directory",
            ConflictReason::ContainerFile => "a container restores to the container file",
        };
        format!("{what} {} ({})", self.presented.display(), containers.join(", "))
    }
}

/// Everything [`scan_view`] found under a root.
#[derive(Debug)]
pub struct TreeView {
    /// The root that was given: a directory, or a single file.
    pub root: PathBuf,
    /// One entry per path below the root (or the root itself if it is not a directory), sorted by
    /// relative name.
    pub entries: Vec<ViewEntry>,
    /// Names claimed twice. A consumer that writes or mounts refuses the whole view while this is
    /// not empty.
    pub conflicts: Vec<Conflict>,
}

/// Walks `root` (a directory, or a single file) and classifies every path: directories, plain
/// files, per-file containers with the path each restores to and whether that path is taken,
/// other containers, unreadable files, and paths left out (links, special files, temporary and
/// partial files, names that are not UTF-8). Reads directories, metadata, the first bytes of each
/// regular file and the index of each container; writes nothing.
///
/// # Errors
/// [`Error::Io`] if the root, a directory or an entry's type cannot be read: the walk stops, so a
/// view never silently misses part of the tree. A single file that cannot be read or opened is a
/// [`ViewEntry::Broken`] entry, not an error.
pub fn scan_view(root: &Path, options: &ViewOptions) -> Result<TreeView, Error> {
    tracing::info!(root = %root.display(), "scanning tree");
    let mut entries = Vec::new();
    walk(root, &mut |entry| {
        let path = entry.path.to_path_buf();
        let rel_path = entry.rel.to_owned();
        let skipped = |reason| ViewEntry::Skipped {
            path: entry.path.to_path_buf(),
            rel_path: entry.rel.to_owned(),
            reason,
        };
        let classified = match entry.kind {
            WalkKind::Dir => {
                let meta = fs::symlink_metadata(&path).and_then(|meta| meta.modified());
                let modified = meta.map_err(|source| Error::Io {
                    op: "metadata",
                    path: path.clone(),
                    source,
                })?;
                ViewEntry::Directory { path, rel_path, modified }
            }
            WalkKind::File => inspect_file(path, rel_path, *options),
            WalkKind::Symlink => skipped(SkipReason::Symlink),
            WalkKind::Incomplete => skipped(SkipReason::Incomplete),
            WalkKind::NonUtf8 => skipped(SkipReason::NonUtf8),
            WalkKind::Special => skipped(SkipReason::Special),
        };
        entries.push(classified);
        Ok(())
    })?;
    entries.sort_by(|a, b| rel_of(a).cmp(rel_of(b)));
    let conflicts = resolve_presented(&mut entries);
    let view = TreeView { root: root.to_path_buf(), entries, conflicts };
    tracing::info!(
        root = %root.display(),
        entries = view.entries.len(),
        conflicts = view.conflicts.len(),
        "tree scanned"
    );
    Ok(view)
}

/// The relative name of an entry, for sorting.
fn rel_of(entry: &ViewEntry) -> &str {
    match entry {
        ViewEntry::Directory { rel_path, .. }
        | ViewEntry::Plain { rel_path, .. }
        | ViewEntry::Foreign { rel_path, .. }
        | ViewEntry::Broken { rel_path, .. }
        | ViewEntry::Skipped { rel_path, .. } => rel_path,
        ViewEntry::PerFile(entry) | ViewEntry::Shadowed(entry) => &entry.container_rel,
    }
}

/// Classifies the regular file at `path`: a temporary file of a write is skipped by name, then the
/// magic decides between plain and container, and a container's index decides whether it is a
/// per-file container. A per-file container comes back as [`ViewEntry::PerFile`]; whether its
/// restore path is taken is decided by the caller. `rel_path` is the file's relative name.
pub(crate) fn inspect_file(path: PathBuf, rel_path: String, options: ViewOptions) -> ViewEntry {
    if rel_path.ends_with(TEMP_FILE_SUFFIX) {
        return ViewEntry::Skipped { path, rel_path, reason: SkipReason::StaleTemp };
    }
    let stamp = match FileStamp::of(&path) {
        Ok(stamp) => stamp,
        Err(error) => return ViewEntry::Broken { path, rel_path, error },
    };
    match has_magic(&path) {
        Ok(false) => return ViewEntry::Plain { path, rel_path, stamp },
        Ok(true) => {}
        Err(error) => return ViewEntry::Broken { path, rel_path, error },
    }
    let open = OpenOptions {
        threads: 1,
        max_total_decoded: options.max_total_decoded,
        ..OpenOptions::default()
    };
    let container = match Container::open(&path, &open) {
        Ok(container) => container,
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "a file with the container magic does not open");
            return ViewEntry::Broken { path, rel_path, error };
        }
    };
    let members = container.members();
    let [info] = members else {
        let reason = ForeignReason::MemberCount(members.len());
        return ViewEntry::Foreign { path, rel_path, reason };
    };
    if !is_one_safe_name(&info.name) {
        let reason = ForeignReason::NotOneSafeName(info.name.clone());
        return ViewEntry::Foreign { path, rel_path, reason };
    }
    let member = match container.single() {
        Ok(member) => member,
        Err(error) => return ViewEntry::Broken { path, rel_path, error },
    };
    let presented = path.with_file_name(&info.name);
    let rel_presented = match rel_path.rsplit_once('/') {
        Some((dir, _)) => format!("{dir}/{}", info.name),
        None => info.name.clone(),
    };
    ViewEntry::PerFile(PerFileEntry {
        container: path,
        container_rel: rel_path,
        presented,
        rel_path: rel_presented,
        member_name: info.name.clone(),
        member_len: member.len(),
        blake3: member.blake3(),
        stamp,
    })
}

/// Whether `name` can be joined to a container's directory as one file name: a restore accepts it
/// and it has no separator and is not `.`.
fn is_one_safe_name(name: &str) -> bool {
    check_restore_name(name).is_ok() && !name.contains(['/', '\\']) && name != "."
}

/// Whether the file at `path` starts with the container magic.
///
/// # Errors
/// [`Error::Io`] if it cannot be opened or read.
fn has_magic(path: &Path) -> Result<bool, Error> {
    let source = FileSource::open(path)?;
    // Lossless: the magic is 8 bytes.
    let magic_len = MAGIC.len() as u64;
    if source.len() < magic_len {
        return Ok(false);
    }
    let mut head = [0; MAGIC.len()];
    source.read_exact_at(0, &mut head)?;
    Ok(head == MAGIC)
}

/// Decides for every per-file container whether its restore path is free ([`ViewEntry::PerFile`])
/// or taken ([`ViewEntry::Shadowed`]), and lists the conflicts: several containers of one
/// directory restoring to one name, or a container restoring to a directory or a container file.
fn resolve_presented(entries: &mut [ViewEntry]) -> Vec<Conflict> {
    let mut container_files = HashSet::new();
    let mut claims: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    for entry in entries.iter() {
        match entry {
            ViewEntry::PerFile(per_file) => {
                container_files.insert(per_file.container.clone());
                claims.entry(per_file.presented.clone()).or_default().push(per_file.container.clone());
            }
            ViewEntry::Foreign { path, .. } | ViewEntry::Broken { path, .. } => {
                container_files.insert(path.clone());
            }
            ViewEntry::Directory { .. }
            | ViewEntry::Plain { .. }
            | ViewEntry::Shadowed(_)
            | ViewEntry::Skipped { .. } => {}
        }
    }
    let mut conflicts = Vec::new();
    for (presented, containers) in &claims {
        if containers.len() > 1 {
            let containers = containers.clone();
            conflicts.push(Conflict { presented: presented.clone(), containers, reason: ConflictReason::SameName });
        }
    }
    for entry in entries.iter_mut() {
        let ViewEntry::PerFile(per_file) = entry else { continue };
        let occupied = match fs::symlink_metadata(&per_file.presented) {
            Ok(meta) => Some(meta),
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Err(source) => {
                let error = Error::Io { op: "metadata", path: per_file.presented.clone(), source };
                tracing::error!(path = %per_file.presented.display(), %error, "restore path cannot be looked at");
                *entry = ViewEntry::Broken {
                    path: per_file.container.clone(),
                    rel_path: per_file.container_rel.clone(),
                    error,
                };
                continue;
            }
        };
        let Some(meta) = occupied else { continue };
        let reason = if meta.is_dir() {
            Some(ConflictReason::Directory)
        } else if container_files.contains(&per_file.presented) {
            Some(ConflictReason::ContainerFile)
        } else {
            None
        };
        if let Some(reason) = reason {
            let containers = vec![per_file.container.clone()];
            conflicts.push(Conflict { presented: per_file.presented.clone(), containers, reason });
        } else {
            let shadowed = per_file.clone();
            *entry = ViewEntry::Shadowed(shadowed);
        }
    }
    conflicts.sort_by(|a, b| a.presented.cmp(&b.presented));
    conflicts
}
