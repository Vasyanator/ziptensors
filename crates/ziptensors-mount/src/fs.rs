//! File: fs.rs
//! Purpose: [`MountFs`], the backend-neutral filesystem a backend serves: attributes, lookups,
//! listings, opens and reads, over a [`Tree`] and either one container (a core [`ReadCache`]) or a
//! directory tree with per-file containers (an overlay, `overlay.rs`).
//!
//! Main responsibilities:
//! - build the tree of a container under a [`NamePolicy`] and a cached handle per member
//!   ([`MountFs::new`]), or the tree of an overlay's snapshot ([`MountFs::overlay`]);
//! - answer every call a read-only filesystem needs, with [`FsError`] for failures;
//! - clamp reads at the end of a file, as a filesystem read does, and log read failures.
//!
//! Key structures: [`MountFs`], [`Attr`], [`Access`], `Backing`.
//!
//! Implementation notes: no OS type appears here, so a FUSE and a `WinFsp` backend share it. Member
//! reads go through `CachedMember::read_at`, because backends see reads far smaller than a chunk
//! (FUSE: 128 KiB) and the cache is the one place that knows window geometry. `Node::File::member`
//! indexes the files of the backing; times are kept per node. All methods take `&self` and are safe
//! from the backend's dispatch threads.

use std::fs;
use std::io;
use std::path::Path;
use std::time::SystemTime;

use ziptensors_core::tree::{ViewOptions, scan_view};
use ziptensors_core::{CacheStats, CachedMember, ReadCache};

use crate::error::{FsError, MountError};
use crate::overlay::{self, Overlay, OverlayOptions};
use crate::policy::NamePolicy;
use crate::tree::{DirEntry, Node, NodeId, NodeKind, Tree};

/// The attributes of a node that do not depend on the OS. A backend adds owner and permission bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attr {
    /// The node.
    pub id: NodeId,
    /// Directory or file.
    pub kind: NodeKind,
    /// Bytes of a file; 0 for a directory.
    pub size: u64,
    /// Hard links: 1 for a file, 2 plus the subdirectories for a directory (POSIX convention).
    pub links: u32,
    /// Modification time of the node: the container file's for a container mount; for an overlay that
    /// of the file or directory on disk (a member shows its container's).
    pub modified: SystemTime,
}

/// What an open asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Reading only.
    Read,
    /// Writing, alone or with reading, or truncation.
    Write,
}

/// A read-only filesystem over one container or one directory tree. `Send + Sync`; every method takes
/// `&self`.
#[derive(Debug)]
pub struct MountFs {
    tree: Tree,
    /// Where the bytes of each file come from; `Node::File::member` indexes its files.
    backing: Backing,
    /// Modification time of each node: node `id` is at `modified[id - 1]`.
    modified: Vec<SystemTime>,
}

/// The source of a [`MountFs`]'s file bytes.
#[derive(Debug)]
enum Backing {
    /// One container read through one cache.
    Container {
        /// The cache every member reads through.
        cache: ReadCache,
        /// One handle per member, in file order.
        members: Vec<CachedMember>,
    },
    /// A directory tree: files on disk and per-file containers, opened lazily. Boxed: it is several
    /// times the size of the other variant.
    Overlay(Box<Overlay>),
}

impl MountFs {
    /// The filesystem of the container behind `cache`, with names checked by `policy`. `modified` is
    /// the time every node reports, normally the container file's.
    ///
    /// # Errors
    /// [`MountError::Names`] naming every member `policy` cannot show (see [`Tree::build`]), and
    /// [`MountError::Member`] if the core refuses a member handle.
    pub fn new<P>(cache: ReadCache, policy: P, modified: SystemTime) -> Result<Self, MountError>
    where
        P: NamePolicy + 'static,
    {
        let infos = cache.container().members();
        let tree = Tree::build(infos.iter().map(|info| (info.name.as_str(), info.len)), policy)?;
        let members = (0..infos.len())
            .map(|index| cache.member_at(index).map_err(|source| MountError::Member { index, source }))
            .collect::<Result<Vec<_>, _>>()?;
        tracing::info!(
            members = members.len(),
            nodes = tree.node_count(),
            policy = tree.policy(),
            "mount tree built"
        );
        let modified = vec![modified; tree.node_count()];
        Ok(Self { tree, backing: Backing::Container { cache, members }, modified })
    }

    /// The read-only overlay of directory `root`: every directory below it, every regular file as it
    /// is, each per-file container under its member's name with the member's bytes, and a container
    /// whose original is still beside it (a `--keep` pair) hidden behind that original, which is read
    /// directly. Containers of several members are shown as files. Links, special files, temporary
    /// and partial files and names that are not UTF-8 are left out. The tree is a snapshot taken now:
    /// a file changed on disk later fails its reads with [`FsError::Io`]. Nodes report the times of
    /// their files and directories on disk. Containers and files open on first read (see
    /// [`OverlayOptions`]); this call reads directories, metadata, the first bytes of each file and
    /// the index of each container.
    ///
    /// # Errors
    /// - [`MountError::OverlayOption`] for an option out of range;
    /// - [`MountError::OverlayRoot`] if `root` cannot be resolved or is not a directory;
    /// - [`MountError::Scan`] if a directory or an entry's type cannot be read;
    /// - [`MountError::Overlay`] naming every damaged or unreadable file and every name claimed twice;
    /// - [`MountError::Names`] naming every path `policy` cannot show.
    pub fn overlay<P>(root: &Path, policy: P, options: &OverlayOptions) -> Result<Self, MountError>
    where
        P: NamePolicy + 'static,
    {
        let root_error = |source| MountError::OverlayRoot { root: root.to_path_buf(), source };
        // The walk does not follow a link at the root; a mount of a linked directory shows its target.
        let resolved = fs::canonicalize(root).map_err(root_error)?;
        let meta = fs::metadata(&resolved).map_err(root_error)?;
        if !meta.is_dir() {
            return Err(root_error(io::Error::new(io::ErrorKind::NotADirectory, "not a directory")));
        }
        let root_modified = meta.modified().map_err(root_error)?;
        // Options are checked before the scan, which may be long.
        options.container_cache()?;
        let mut view_options = ViewOptions::default();
        view_options.max_total_decoded = options.max_total_decoded;
        let view = scan_view(&resolved, &view_options)
            .map_err(|source| MountError::Scan { root: resolved.clone(), source })?;
        let planned = overlay::plan(view)?;
        let tree = Tree::build_with_dirs(
            planned.files.iter().map(|file| (file.rel_path.as_str(), file.len)),
            planned.dirs.iter().map(|(rel_path, _)| rel_path.as_str()),
            policy,
        )?;
        let mut modified = vec![root_modified; tree.node_count()];
        let times = planned
            .dirs
            .iter()
            .map(|(rel_path, time)| (rel_path.as_str(), *time))
            .chain(planned.files.iter().map(|file| (file.rel_path.as_str(), file.modified())));
        for (rel_path, time) in times {
            let id = rel_path.split('/').try_fold(NodeId::ROOT, |dir, part| tree.lookup(dir, part));
            // Every path was just inserted, so a missing one is only possible if the tree dropped it.
            let slot = id.and_then(|id| usize::try_from(id.get().checked_sub(1)?).ok()).and_then(|at| modified.get_mut(at));
            if let Some(slot) = slot {
                *slot = time;
            }
        }
        let files = planned.files.len();
        let overlay = Overlay::new(planned.files, options)?;
        tracing::info!(
            root = %resolved.display(),
            files,
            nodes = tree.node_count(),
            policy = tree.policy(),
            "overlay tree built"
        );
        Ok(Self { tree, backing: Backing::Overlay(Box::new(overlay)), modified })
    }

    /// The tree, for a backend that walks it itself.
    #[must_use]
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// The attributes of node `id`.
    ///
    /// # Errors
    /// [`FsError::NotFound`] if there is no such node.
    pub fn attr(&self, id: NodeId) -> Result<Attr, FsError> {
        let node = self.tree.node(id).ok_or(FsError::NotFound)?;
        let (size, links) = match node {
            Node::Dir { entries, .. } => {
                let subdirs = entries.iter().filter(|entry| entry.kind == NodeKind::Dir).count();
                (0, u32::try_from(subdirs).unwrap_or(u32::MAX).saturating_add(2))
            }
            Node::File { len, .. } => (*len, 1),
        };
        let modified = self.node_modified(id).ok_or(FsError::NotFound)?;
        Ok(Attr { id, kind: node.kind(), size, links, modified })
    }

    /// The modification time of node `id`, `None` if there is no such node.
    fn node_modified(&self, id: NodeId) -> Option<SystemTime> {
        let index = usize::try_from(id.get().checked_sub(1)?).ok()?;
        self.modified.get(index).copied()
    }

    /// The attributes of the entry `name` of directory `parent`.
    ///
    /// # Errors
    /// [`FsError::NotDir`] if `parent` is a file, [`FsError::NotFound`] if it does not exist or has no
    /// such entry.
    pub fn lookup(&self, parent: NodeId, name: &str) -> Result<Attr, FsError> {
        match self.tree.node(parent).ok_or(FsError::NotFound)? {
            Node::Dir { .. } => {}
            Node::File { .. } => return Err(FsError::NotDir),
        }
        self.attr(self.tree.lookup(parent, name).ok_or(FsError::NotFound)?)
    }

    /// The parent of node `id`; the root's is the root.
    ///
    /// # Errors
    /// [`FsError::NotFound`] if there is no such node.
    pub fn parent(&self, id: NodeId) -> Result<NodeId, FsError> {
        self.tree.node(id).map(Node::parent).ok_or(FsError::NotFound)
    }

    /// The entries of directory `id`, sorted by name, without `.` and `..`.
    ///
    /// # Errors
    /// [`FsError::NotFound`] if there is no such node, [`FsError::NotDir`] for a file.
    pub fn read_dir(&self, id: NodeId) -> Result<&[DirEntry], FsError> {
        match self.tree.node(id).ok_or(FsError::NotFound)? {
            Node::Dir { entries, .. } => Ok(entries),
            Node::File { .. } => Err(FsError::NotDir),
        }
    }

    /// Whether file `id` may be opened for `access`. Opens hold no state: every read names its node.
    ///
    /// # Errors
    /// [`FsError::NotFound`], [`FsError::IsDir`] for a directory, [`FsError::ReadOnly`] for any write.
    pub fn open(&self, id: NodeId, access: Access) -> Result<(), FsError> {
        match self.tree.node(id).ok_or(FsError::NotFound)? {
            Node::Dir { .. } => Err(FsError::IsDir),
            Node::File { .. } => match access {
                Access::Read => Ok(()),
                Access::Write => Err(FsError::ReadOnly),
            },
        }
    }

    /// Up to `size` bytes of file `id` from byte `offset`: fewer only at the end of the file, none at
    /// or past it, as a filesystem read returns.
    ///
    /// # Errors
    /// [`FsError::NotFound`], [`FsError::IsDir`], and [`FsError::Io`] when the core fails to read;
    /// that failure is logged with the member, the range and its `ErrorCode`. In an overlay also when
    /// the file on disk changed since the mount, or cannot be opened (logged with its path).
    pub fn read(&self, id: NodeId, offset: u64, size: usize) -> Result<Vec<u8>, FsError> {
        let (member, len) = match self.tree.node(id).ok_or(FsError::NotFound)? {
            Node::Dir { .. } => return Err(FsError::IsDir),
            Node::File { member, len, .. } => (*member, *len),
        };
        let available = len.saturating_sub(offset);
        // `take` is at most `size`, so it fits in `usize`.
        let take = usize::try_from(available).map_or(size, |available| available.min(size));
        if take == 0 {
            // At or past the end: the core refuses an offset past the end even for no bytes.
            return Ok(Vec::new());
        }
        let mut bytes = vec![0; take];
        match &self.backing {
            Backing::Container { members, .. } => {
                let handle = members.get(member).ok_or(FsError::NotFound)?;
                if let Err(error) = handle.read_at(offset, &mut bytes) {
                    tracing::error!(
                        member = handle.member().name().unwrap_or("<unnamed>"),
                        offset,
                        len = take,
                        code = ?error.code(),
                        %error,
                        "mount read failed"
                    );
                    return Err(FsError::Io);
                }
            }
            Backing::Overlay(overlay) => overlay.read(member, offset, &mut bytes)?,
        }
        Ok(bytes)
    }

    /// Number of files and the sum of their lengths, for a backend's volume information.
    #[must_use]
    pub fn totals(&self) -> (u64, u64) {
        (self.tree.file_count(), self.tree.total_len())
    }

    /// Counters of the read cache.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        match &self.backing {
            Backing::Container { cache, .. } => cache.stats(),
            Backing::Overlay(overlay) => overlay.stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use std::fs;
    use std::io::{ErrorKind, Write};
    use std::path::Path;
    use std::sync::Arc;
    use std::thread;

    use ziptensors_core::format::MAGIC;
    use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};

    use super::{Access, Backing, MountFs};
    use crate::error::{FsError, MountError, OverlayProblem};
    use crate::overlay::{Overlay, OverlayOptions};
    use crate::policy::{NameProblem, PosixPolicy, WindowsPolicy};
    use crate::testkit::{Fixture, OverlayFixture, Scratch, per_file};
    use crate::tree::{NodeId, NodeKind};

    const STAMP: Duration = Duration::from_secs(1_700_000_000);

    /// The filesystem of `fixture`, with the smallest cache the container allows, so reads evict.
    fn mount(fixture: &Fixture) -> MountFs {
        let container = Container::open(&fixture.container, &OpenOptions { threads: 2, ..OpenOptions::default() })
            .expect("open container");
        let options = CacheOptions { max_bytes: 2 << 20, readahead_windows: 2 };
        let cache = ReadCache::new(container, options).expect("cache");
        MountFs::new(cache, PosixPolicy, SystemTime::UNIX_EPOCH + STAMP).expect("mountable")
    }

    /// The node of `path` under the root, `/` separated.
    fn find(fs: &MountFs, path: &str) -> NodeId {
        path.split('/').fold(NodeId::ROOT, |dir, part| fs.lookup(dir, part).expect("entry").id)
    }

    #[test]
    fn every_file_reads_back_whole_and_in_small_pieces() {
        let scratch = Scratch::new("fs-read");
        let fixture = Fixture::standard(&scratch);
        let fs = mount(&fixture);
        for (name, content) in &fixture.files {
            let id = find(&fs, name);
            let attr = fs.attr(id).expect("attr");
            assert_eq!(attr.kind, NodeKind::File);
            assert_eq!(attr.size, u64::try_from(content.len()).expect("small"), "{name}");
            assert_eq!(attr.links, 1);
            assert_eq!(fs.open(id, Access::Read), Ok(()));
            assert_eq!(&fs.read(id, 0, content.len()).expect("whole"), content, "{name}");
            // 128 KiB pieces, as FUSE asks, plus an odd size that straddles windows.
            for piece in [128 << 10, 77_777] {
                let mut joined = Vec::with_capacity(content.len());
                let mut offset = 0_u64;
                loop {
                    let bytes = fs.read(id, offset, piece).expect("piece");
                    if bytes.is_empty() {
                        break;
                    }
                    offset += u64::try_from(bytes.len()).expect("small");
                    joined.extend_from_slice(&bytes);
                }
                assert_eq!(&joined, content, "{name} in pieces of {piece}");
            }
        }
        let stats = fs.stats();
        assert!(stats.hits > 0 && stats.misses > 0, "{stats:?}");
        assert!(stats.evictions > 0, "a 2 MiB cache under a 3 MiB file evicts: {stats:?}");
        assert_eq!(fs.totals(), (fixture.files.len().try_into().expect("small"), fixture.total_len()));
    }

    #[test]
    fn reads_clamp_at_the_end_of_a_file() {
        let scratch = Scratch::new("fs-clamp");
        let fixture = Fixture::standard(&scratch);
        let fs = mount(&fixture);
        let (name, content) = &fixture.files[0];
        let id = find(&fs, name);
        let len = u64::try_from(content.len()).expect("small");
        assert_eq!(fs.read(id, len - 3, 100).expect("tail"), content[content.len() - 3..]);
        assert_eq!(fs.read(id, len, 100).expect("at end"), Vec::<u8>::new());
        assert_eq!(fs.read(id, u64::MAX, 100).expect("far past end"), Vec::<u8>::new());
        assert_eq!(fs.read(id, 0, 0).expect("nothing"), Vec::<u8>::new());
        let empty = find(&fs, "empty.bin");
        assert_eq!(fs.attr(empty).map(|attr| attr.size), Ok(0));
        assert_eq!(fs.read(empty, 0, 4096).expect("empty file"), Vec::<u8>::new());
    }

    #[test]
    fn directories_list_sorted_and_refuse_file_calls() {
        let scratch = Scratch::new("fs-dirs");
        let fixture = Fixture::standard(&scratch);
        let fs = mount(&fixture);
        let root: Vec<_> = fs.read_dir(NodeId::ROOT).expect("root").iter().map(|e| e.name.as_str()).collect();
        assert_eq!(root, ["config.json", "empty.bin", "models"]);
        let models = find(&fs, "models");
        let attr = fs.attr(models).expect("attr");
        assert_eq!((attr.kind, attr.size, attr.links), (NodeKind::Dir, 0, 3));
        assert_eq!(attr.modified, SystemTime::UNIX_EPOCH + STAMP);
        assert_eq!(fs.parent(find(&fs, "models/unet")), Ok(models));
        assert_eq!(fs.parent(NodeId::ROOT), Ok(NodeId::ROOT));
        assert_eq!(fs.open(models, Access::Read), Err(FsError::IsDir));
        assert_eq!(fs.read(models, 0, 10), Err(FsError::IsDir));
        let file = find(&fs, "config.json");
        assert_eq!(fs.read_dir(file).map(<[_]>::len), Err(FsError::NotDir));
        assert_eq!(fs.lookup(file, "x"), Err(FsError::NotDir));
        assert_eq!(fs.open(file, Access::Write), Err(FsError::ReadOnly));
        assert_eq!(fs.lookup(NodeId::ROOT, "missing"), Err(FsError::NotFound));
        let unknown = NodeId::from_raw(10_000);
        assert_eq!(fs.attr(unknown), Err(FsError::NotFound));
        assert_eq!(fs.read(unknown, 0, 1), Err(FsError::NotFound));
        assert_eq!(fs.open(unknown, Access::Read), Err(FsError::NotFound));
        assert_eq!(fs.read_dir(unknown).map(<[_]>::len), Err(FsError::NotFound));
    }

    #[test]
    fn a_container_with_conflicting_names_does_not_mount() {
        let scratch = Scratch::new("fs-conflict");
        let fixture = Fixture::build(
            &scratch,
            &[("Model.bin", 1000), ("model.bin", 2000), ("dir", 10), ("dir/x.bin", 10)],
        );
        let open = || {
            let container = Container::open(&fixture.container, &OpenOptions::default()).expect("open");
            ReadCache::new(container, CacheOptions::default()).expect("cache")
        };
        let error = MountFs::new(open(), PosixPolicy, SystemTime::UNIX_EPOCH).expect_err("conflict");
        let MountError::Names(tree) = error else { panic!("not a name error: {error:?}") };
        assert_eq!(tree.bad.len(), 1);
        assert_eq!(tree.bad[0].member, "dir");
        // The same container under Windows rules also has a case collision.
        let error = MountFs::new(open(), WindowsPolicy::default(), SystemTime::UNIX_EPOCH).expect_err("conflict");
        let MountError::Names(tree) = error else { panic!("not a name error: {error:?}") };
        let problems: Vec<_> = tree.bad.iter().map(|bad| (bad.member.as_str(), &bad.problem)).collect();
        assert_eq!(
            problems,
            [
                ("dir", &NameProblem::FileIsDirectory { under: "dir/x.bin".to_owned() }),
                ("model.bin", &NameProblem::Collision { with: "Model.bin".to_owned() }),
            ]
        );
    }

    /// The overlay of `root` under POSIX rules with `options`.
    fn overlay(root: &Path, options: &OverlayOptions) -> MountFs {
        MountFs::overlay(root, PosixPolicy, options).expect("mountable overlay")
    }

    /// The overlay backing of `fs`.
    fn backing(fs: &MountFs) -> &Overlay {
        match &fs.backing {
            Backing::Overlay(overlay) => overlay,
            Backing::Container { .. } => panic!("not an overlay"),
        }
    }

    /// Every file and every directory below `id`, `/`-separated with `prefix`, in listing order.
    fn listing(fs: &MountFs, id: NodeId, prefix: &str, files: &mut Vec<String>, dirs: &mut Vec<String>) {
        for entry in fs.read_dir(id).expect("a directory") {
            let path = format!("{prefix}{}", entry.name);
            match entry.kind {
                NodeKind::Dir => {
                    dirs.push(path.clone());
                    listing(fs, entry.id, &format!("{path}/"), files, dirs);
                }
                NodeKind::File => files.push(path),
            }
        }
    }

    /// The node of `path`, or the error of the first part that is missing.
    fn try_find(fs: &MountFs, path: &str) -> Result<NodeId, FsError> {
        path.split('/').try_fold(NodeId::ROOT, |dir, part| fs.lookup(dir, part).map(|attr| attr.id))
    }

    /// Reads all of file `id` in pieces of `piece` bytes.
    fn read_all(fs: &MountFs, id: NodeId, piece: usize) -> Result<Vec<u8>, FsError> {
        let mut joined = Vec::new();
        loop {
            let bytes = fs.read(id, u64::try_from(joined.len()).expect("small"), piece)?;
            if bytes.is_empty() {
                return Ok(joined);
            }
            joined.extend_from_slice(&bytes);
        }
    }

    #[test]
    fn an_overlay_shows_members_under_their_names_and_other_files_as_they_are() {
        let scratch = Scratch::new("overlay-read");
        let fixture = OverlayFixture::standard(&scratch);
        let fs = overlay(&fixture.root, &OverlayOptions::default());
        let (mut files, mut dirs) = (Vec::new(), Vec::new());
        listing(&fs, NodeId::ROOT, "", &mut files, &mut dirs);
        files.sort();
        dirs.sort();
        let shown: Vec<&str> = fixture.shown.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(files, shown);
        assert_eq!(dirs, fixture.dirs);
        for (name, content) in &fixture.shown {
            let id = find(&fs, name);
            let attr = fs.attr(id).expect("attr");
            assert_eq!((attr.kind, attr.size, attr.links), (NodeKind::File, u64::try_from(content.len()).expect("small"), 1), "{name}");
            assert_eq!(fs.open(id, Access::Read), Ok(()));
            assert_eq!(fs.open(id, Access::Write), Err(FsError::ReadOnly));
            assert_eq!(&fs.read(id, 0, content.len()).expect("whole"), content, "{name}");
            assert_eq!(&read_all(&fs, id, 128 << 10).expect("pieces"), content, "{name} in 128 KiB pieces");
            assert_eq!(&read_all(&fs, id, 77_777).expect("pieces"), content, "{name} in odd pieces");
        }
        // The containers are shown as their members only; the kept pair's container and links not at all.
        for name in fixture.containers.iter().chain(&fixture.hidden) {
            assert_eq!(try_find(&fs, name), Err(FsError::NotFound), "{name}");
        }
        assert_eq!(fixture.content("vae/model.safetensors"), fs.read(find(&fs, "vae/model.safetensors"), 0, 1 << 30).expect("pair"));
        // Times are those on disk: a member shows its container's.
        let time = |path: &Path| fs::metadata(path).and_then(|meta| meta.modified()).expect("time");
        let modified = |name: &str| fs.attr(try_find(&fs, name).expect("node")).expect("attr").modified;
        assert_eq!(modified("unet"), time(&fixture.root.join("unet")));
        assert_eq!(modified("config.json"), time(&fixture.root.join("config.json")));
        let member = "unet/diffusion_pytorch_model.safetensors";
        assert_eq!(modified(member), time(&fixture.root.join("unet/diffusion_pytorch_model.ziptensors")));
        assert_eq!(fs.attr(NodeId::ROOT).expect("root").modified, time(&fixture.root));
        let total: u64 = fixture.shown.iter().map(|(_, content)| u64::try_from(content.len()).expect("small")).sum();
        assert_eq!(fs.totals(), (u64::try_from(fixture.shown.len()).expect("small"), total));
        let stats = fs.stats();
        assert!(stats.misses > 0 && stats.hits > 0, "{stats:?}");
    }

    #[test]
    fn an_overlay_of_a_tree_with_a_broken_container_does_not_mount() {
        let scratch = Scratch::new("overlay-broken");
        let root = scratch.path().join("tree");
        fs::create_dir(&root).expect("root");
        fs::write(root.join("config.json"), b"{}").expect("write");
        let mut damaged = MAGIC.to_vec();
        damaged.extend_from_slice(&[0x5a; 200]);
        fs::write(root.join("bad.ziptensors"), &damaged).expect("write");
        let error = MountFs::overlay(&root, PosixPolicy, &OverlayOptions::default()).expect_err("broken");
        let MountError::Overlay { problems, .. } = &error else { panic!("not an overlay error: {error:?}") };
        assert_eq!(problems.len(), 1, "{problems:?}");
        let OverlayProblem::Broken { path, .. } = &problems[0] else { panic!("not broken: {problems:?}") };
        assert!(path.ends_with("bad.ziptensors"), "{}", path.display());
        assert!(error.to_string().contains("bad.ziptensors"), "{error}");
    }

    #[test]
    fn an_overlay_where_two_containers_present_one_name_does_not_mount() {
        let scratch = Scratch::new("overlay-conflict");
        let root = scratch.path().join("tree");
        fs::create_dir(&root).expect("root");
        fs::write(root.join("a.bin"), b"some bytes of a").expect("write");
        let container = per_file(&root.join("a.bin"), false);
        fs::copy(&container, root.join("b.ziptensors")).expect("copy");
        let error = MountFs::overlay(&root, PosixPolicy, &OverlayOptions::default()).expect_err("conflict");
        let MountError::Overlay { problems, .. } = &error else { panic!("not an overlay error: {error:?}") };
        let [OverlayProblem::Conflict(conflict)] = problems.as_slice() else { panic!("not one conflict: {problems:?}") };
        assert!(conflict.presented.ends_with("a.bin"));
        assert_eq!(conflict.containers.len(), 2);
        assert!(error.to_string().contains("several containers present"), "{error}");
    }

    #[test]
    fn a_file_changed_after_the_mount_fails_its_reads() {
        let scratch = Scratch::new("overlay-changed");
        let fixture = OverlayFixture::standard(&scratch);
        let fs = overlay(&fixture.root, &OverlayOptions::default());
        // A plain file already open, changed in place without a change of length.
        let readme = find(&fs, "notes/readme.txt");
        assert_eq!(fs.read(readme, 0, 10).expect("before"), fixture.content("notes/readme.txt")[..10]);
        let path = fixture.root.join("notes/readme.txt");
        let before = fs::metadata(&path).and_then(|meta| meta.modified()).expect("time");
        let mut file = fs::OpenOptions::new().write(true).open(&path).expect("open for writing");
        file.write_all(b"CHANGED").expect("write at the start");
        file.set_modified(before + Duration::from_secs(5)).expect("set time");
        drop(file);
        assert_eq!(fs.read(readme, 0, 10), Err(FsError::Io));
        // A plain file not yet opened, replaced by a longer one.
        fs::write(fixture.root.join("config.json"), b"{\"model\":\"something else\"}\n").expect("rewrite");
        assert_eq!(fs.read(find(&fs, "config.json"), 0, 4), Err(FsError::Io));
        // A container already open whose file changed.
        let weights = find(&fs, "weights.bin");
        assert_eq!(fs.read(weights, 0, 10).expect("before"), fixture.content("weights.bin")[..10]);
        let container = fixture.root.join("weights.ziptensors");
        let time = fs::metadata(&container).and_then(|meta| meta.modified()).expect("time");
        let file = fs::File::options().write(true).open(&container).expect("open container");
        file.set_modified(time + Duration::from_secs(5)).expect("set time");
        drop(file);
        assert_eq!(fs.read(weights, 0, 10), Err(FsError::Io));
        // Back to the snapshot's time, the unchanged container reads again.
        let file = fs::File::options().write(true).open(&container).expect("open container");
        file.set_modified(time).expect("set time");
        drop(file);
        assert_eq!(fs.read(weights, 0, 10).expect("after"), fixture.content("weights.bin")[..10]);
        // A removed container.
        fs::remove_file(fixture.root.join("unet/diffusion_pytorch_model.ziptensors")).expect("remove");
        assert_eq!(fs.read(find(&fs, "unet/diffusion_pytorch_model.safetensors"), 0, 10), Err(FsError::Io));
    }

    #[test]
    fn an_overlay_keeps_at_most_its_cap_of_open_handles() {
        let scratch = Scratch::new("overlay-cap");
        let fixture = OverlayFixture::standard(&scratch);
        let mut options =
            OverlayOptions { max_open_containers: 1, max_open_files: 1, threads: 1, ..OverlayOptions::default() };
        let fs = overlay(&fixture.root, &options);
        let members = ["unet/diffusion_pytorch_model.safetensors", "text_encoder/deep/nested/model.safetensors", "weights.bin"];
        let plains = ["config.json", "notes/readme.txt"];
        for round in 0..3_usize {
            for name in members.iter().chain(&plains) {
                let content = fixture.content(name);
                // Within every file, so that each read reaches its file.
                let at = (round * 1000) % content.len();
                let end = (at + 4096).min(content.len());
                let got = fs.read(find(&fs, name), u64::try_from(at).expect("small"), 4096).expect("read");
                assert_eq!(got, content[at..end], "{name} round {round}");
            }
        }
        let containers = backing(&fs).container_pool();
        assert_eq!((containers.peak, containers.opened), (1, 9), "every switch reopens");
        let plain = backing(&fs).plain_pool();
        assert_eq!((plain.peak, plain.opened), (1, 6));
        let stats = fs.stats();
        assert!(stats.misses >= 9, "closed caches still count: {stats:?}");

        // Concurrent readers of three containers under a cap of two.
        options.max_open_containers = 2;
        let fs = Arc::new(overlay(&fixture.root, &options));
        let readers: Vec<_> = (0..6_usize)
            .map(|reader| {
                let fs = Arc::clone(&fs);
                let name = members[reader % members.len()];
                let content = fixture.content(name).to_vec();
                thread::spawn(move || {
                    let id = find(&fs, name);
                    assert_eq!(read_all(&fs, id, 64 << 10).expect("read"), content, "{name}");
                })
            })
            .collect();
        for reader in readers {
            reader.join().expect("reader");
        }
        assert!(backing(&fs).container_pool().peak <= 2);
    }

    #[test]
    fn an_overlay_refuses_bad_options_and_roots() {
        let scratch = Scratch::new("overlay-options");
        let root = scratch.path().join("tree");
        fs::create_dir(&root).expect("root");
        let refused = |options: &OverlayOptions| match MountFs::overlay(&root, PosixPolicy, options) {
            Err(MountError::OverlayOption { option, .. }) => option,
            other => panic!("not an option error: {other:?}"),
        };
        let options = OverlayOptions { max_open_containers: 0, ..OverlayOptions::default() };
        assert_eq!(refused(&options), "max_open_containers");
        let options = OverlayOptions { max_open_files: 0, ..OverlayOptions::default() };
        assert_eq!(refused(&options), "max_open_files");
        let cache = CacheOptions { max_bytes: 20 << 20, readahead_windows: 0 };
        let mut options = OverlayOptions { cache, max_open_containers: 2, ..OverlayOptions::default() };
        assert_eq!(refused(&options), "cache.max_bytes");
        options.max_open_containers = 1;
        assert!(MountFs::overlay(&root, PosixPolicy, &options).is_ok(), "a 20 MiB share holds a window");

        let file = scratch.path().join("file");
        fs::write(&file, b"x").expect("write");
        for (path, kind) in [(file, ErrorKind::NotADirectory), (scratch.path().join("missing"), ErrorKind::NotFound)] {
            match MountFs::overlay(&path, PosixPolicy, &OverlayOptions::default()) {
                Err(MountError::OverlayRoot { source, .. }) => assert_eq!(source.kind(), kind, "{}", path.display()),
                other => panic!("not a root error: {other:?}"),
            }
        }
        // An empty directory is an empty root.
        let fs = overlay(&root, &OverlayOptions::default());
        assert_eq!(fs.read_dir(NodeId::ROOT).map(<[_]>::len), Ok(0));
    }
}
