//! File: write/inputs.rs
//! Purpose: turning files and directory trees into a list of `(path, member name)` to add, and
//! refusing an output that such a walk would pick up.
//!
//! Main responsibilities:
//! - the one walk of a tree, [`walk`], shared with the `tree` module: no link followed, entries in
//!   name order, links, `*.incomplete` files, names that are not UTF-8 and special files reported as
//!   such for each caller to decide;
//! - walk each root for [`collect_inputs`], in sorted order of member names per root;
//! - skip symbolic links and `*.incomplete` files with a log line, and count them;
//! - check every member name the way `PackBuilder` will (restore-name rules, unique), so that a bad
//!   name fails before an output is started, since a pack can run for hours; [`check_member_names`]
//!   does it for any list of names, such as the members of a merge.
//!
//! Key structures: [`Inputs`], the crate-private `WalkEntry` and `WalkKind`.
//!
//! Key functions: [`collect_inputs`], [`check_output`], [`check_member_names`], the crate-private
//! [`walk`].
//!
//! Implementation notes: a symbolic link is skipped here and not left to `PackBuilder::add_file`,
//! which checks the name first, so a link with an unsafe name would stop the walk. For
//! [`collect_inputs`] every other failure stops it, a name that is not UTF-8 and a special file
//! included, in particular every I/O failure, so a caller never silently misses a file.
//! Nothing here knows about packing: a walk is a list of files with names, usable by any frontend
//! that writes one member per file.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs::{self, FileType};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::format::check_restore_name;

/// Name suffix of partial downloads, which are not model files yet.
const INCOMPLETE_SUFFIX: &str = ".incomplete";

/// What [`collect_inputs`] found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inputs {
    /// Files to add as `(path, member name)`: per root in the order given, then sorted by member
    /// name. A member name is relative to the root it was found under and `/` separated; a root that
    /// is a file is named after itself. Every name passed [`check_restore_name`] and is unique.
    pub files: Vec<(PathBuf, String)>,
    /// Symbolic links left out, at any depth or as a root. Links are neither followed nor stored,
    /// spec section 2.
    pub skipped_symlinks: u64,
    /// `*.incomplete` files left out: partial downloads.
    pub skipped_incomplete: u64,
}

/// Walks every root and returns the files to add with their member names. Reads directories and
/// metadata only, never file contents. An empty result is not an error: the caller decides whether
/// there was something to do.
///
/// # Errors
/// - [`Error::Io`] if a root, a directory or an entry cannot be read (op `"metadata"` or
///   `"read directory"`), or an entry is neither a regular file, a directory nor a link (op
///   `"read input"`, kind `InvalidInput`).
/// - [`Error::UnsafeMemberName`] if a name is not valid UTF-8 (the name is shown lossily) or fails
///   [`check_restore_name`].
/// - [`Error::DuplicateMember`] if two files give one member name, as `PackBuilder::add_file` would
///   report it; an error log names the files.
pub fn collect_inputs(roots: &[PathBuf]) -> Result<Inputs, Error> {
    let mut inputs = Inputs::default();
    for root in roots {
        add_root(&mut inputs, root)?;
    }
    check_names(&inputs.files)?;
    tracing::info!(
        roots = roots.len(),
        files = inputs.files.len(),
        symlinks_skipped = inputs.skipped_symlinks,
        incomplete_skipped = inputs.skipped_incomplete,
        "inputs collected"
    );
    Ok(inputs)
}

/// Refuses an output a walk of `roots` would pick up: as a file of its own, or at a path under a
/// directory that is walked, whether it exists yet or not, since the next run over the same tree would
/// take the last result in as an input. A name the walk skips and a link, which it skips too, are
/// fine. So is an output whose directory cannot be resolved, which the writer then reports itself.
///
/// # Errors
/// [`Error::OutputIsInput`] naming `output` (the root is logged), or [`Error::Io`] with op
/// `"metadata"` or `"resolve path"` if a root cannot be read.
pub fn check_output(output: &Path, roots: &[PathBuf]) -> Result<(), Error> {
    let Some(name) = output.file_name() else { return Ok(()) };
    let skipped = name.to_str().is_some_and(|name| name.ends_with(INCOMPLETE_SUFFIX));
    // An error here means there is nothing at `output` yet, or nothing readable: either way not a
    // link the walk would skip, so the path checks below decide.
    let is_link = fs::symlink_metadata(output).is_ok_and(|meta| meta.is_symlink());
    if skipped || is_link {
        return Ok(());
    }
    let parent = output.parent().filter(|parent| !parent.as_os_str().is_empty());
    // A directory that does not resolve cannot be under a walked root, and the writer reports why it
    // cannot create the output there with its own error, so it is not reported twice.
    let Ok(parent) = fs::canonicalize(parent.unwrap_or(Path::new("."))) else { return Ok(()) };
    let target = parent.join(name);
    for root in roots {
        let meta = metadata(root)?;
        // The walk does not enter a root that is a link.
        if meta.is_symlink() {
            continue;
        }
        let canonical = fs::canonicalize(root).map_err(|source| Error::Io {
            op: "resolve path",
            path: root.clone(),
            source,
        })?;
        let walked = if meta.is_dir() { target.starts_with(&canonical) } else { target == canonical };
        if walked {
            tracing::error!(
                output = %output.display(),
                root = %root.display(),
                "the output is inside an input, which would take it in as an input"
            );
            return Err(Error::OutputIsInput { path: output.to_path_buf() });
        }
    }
    Ok(())
}

/// `symlink_metadata` of `path`, never following a link.
fn metadata(path: &Path) -> Result<fs::Metadata, Error> {
    fs::symlink_metadata(path).map_err(|source| Error::Io {
        op: "metadata",
        path: path.to_path_buf(),
        source,
    })
}

/// Logs and counts a symbolic link.
fn skip_symlink(path: &Path, inputs: &mut Inputs) {
    tracing::info!(path = %path.display(), "skipping symbolic link");
    inputs.skipped_symlinks += 1;
}

/// What [`walk`] found at one path. The walk decides these by file type and name only, never by
/// content, and never follows a link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WalkKind {
    /// A regular file.
    File,
    /// A directory. The walk enters it after reporting it.
    Dir,
    /// A symbolic link, to anything or nothing. Never followed.
    Symlink,
    /// A name ending with `.incomplete` that is not a directory or a link: a partial download.
    Incomplete,
    /// A name that is not valid UTF-8. A directory of such a name is not entered.
    NonUtf8,
    /// Neither a regular file, a directory nor a link: a FIFO, a socket, a device.
    Special,
}

/// One path the walk found.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WalkEntry<'a> {
    /// The path as found: the root joined with the entry's names.
    pub(crate) path: &'a Path,
    /// `/`-separated name relative to the walked root; a root that is not a directory is named after
    /// itself. Shown lossily for [`WalkKind::NonUtf8`], so it is for display only in that case.
    pub(crate) rel: &'a str,
    /// What the entry is.
    pub(crate) kind: WalkKind,
}

/// Walks `root` without following links and hands every entry below it to `on_entry`: a root
/// that is a directory is not reported itself, any other root is reported as one entry. Entries of
/// a directory are visited in byte order of their names, depth first; a caller that needs one order
/// over the whole tree sorts by `rel`. The one owner of the walk rules shared by
/// [`collect_inputs`] and `tree`: links are never entered, `*.incomplete` files and names that are
/// not UTF-8 are reported as such. Reads directories and metadata only, never file contents.
///
/// # Errors
/// [`Error::Io`] with op `"metadata"` if the root or an entry's type cannot be read, or `"read
/// directory"` if a directory cannot be listed, and the first error `on_entry` returns. Each stops
/// the walk.
pub(crate) fn walk(
    root: &Path,
    on_entry: &mut dyn FnMut(WalkEntry<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    let meta = metadata(root)?;
    let mut pending = Vec::new();
    if meta.is_dir() {
        pending.push((root.to_path_buf(), String::new()));
    } else {
        visit(root, meta.file_type(), "", &mut pending, on_entry)?;
    }
    while let Some((dir, prefix)) = pending.pop() {
        let unreadable = |source| Error::Io { op: "read directory", path: dir.clone(), source };
        let mut entries = Vec::new();
        for entry in fs::read_dir(&dir).map_err(unreadable)? {
            let entry = entry.map_err(unreadable)?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|source| Error::Io {
                op: "metadata",
                path: path.clone(),
                source,
            })?;
            entries.push((path, file_type));
        }
        // `read_dir` has no order of its own; a fixed one makes logs, reports and the first error
        // of a walk reproducible.
        entries.sort_unstable_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));
        // Pushed in reverse, so subdirectories are entered in name order.
        let first_dir = pending.len();
        for (path, file_type) in &entries {
            visit(path, *file_type, &prefix, &mut pending, on_entry)?;
        }
        pending[first_dir..].reverse();
    }
    Ok(())
}

/// Classifies one entry, reports it, and queues a directory in `pending`. `prefix` is the relative
/// name of the parent, empty at the root.
fn visit(
    path: &Path,
    file_type: FileType,
    prefix: &str,
    pending: &mut Vec<(PathBuf, String)>,
    on_entry: &mut dyn FnMut(WalkEntry<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    let joined =
        |name: &str| if prefix.is_empty() { name.to_owned() } else { format!("{prefix}/{name}") };
    // A link is reported before its name is looked at: a link with a name that is not UTF-8 is still
    // a link, which every caller skips or follows deliberately.
    if file_type.is_symlink() {
        let name = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy();
        return on_entry(WalkEntry { path, rel: &joined(&name), kind: WalkKind::Symlink });
    }
    let Some(name) = path.file_name().and_then(OsStr::to_str) else {
        let lossy = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy();
        return on_entry(WalkEntry { path, rel: &joined(&lossy), kind: WalkKind::NonUtf8 });
    };
    let rel = joined(name);
    let kind = if file_type.is_dir() {
        WalkKind::Dir
    } else if name.ends_with(INCOMPLETE_SUFFIX) {
        WalkKind::Incomplete
    } else if file_type.is_file() {
        WalkKind::File
    } else {
        WalkKind::Special
    };
    on_entry(WalkEntry { path, rel: &rel, kind })?;
    if kind == WalkKind::Dir {
        pending.push((path.to_path_buf(), rel));
    }
    Ok(())
}

/// Adds the file or directory tree at `root` to `inputs`, sorted by member name among themselves.
/// Names are not checked here: [`check_names`] runs over all roots.
///
/// # Errors
/// As [`collect_inputs`], apart from the name checks: [`Error::UnsafeMemberName`] for a name that
/// is not UTF-8, [`Error::Io`] for an entry that is neither a file, a directory nor a link.
fn add_root(inputs: &mut Inputs, root: &Path) -> Result<(), Error> {
    let first = inputs.files.len();
    walk(root, &mut |entry| {
        match entry.kind {
            WalkKind::Dir => {}
            WalkKind::File => inputs.files.push((entry.path.to_path_buf(), entry.rel.to_owned())),
            WalkKind::Symlink => skip_symlink(entry.path, inputs),
            WalkKind::Incomplete => {
                tracing::info!(path = %entry.path.display(), "skipping incomplete file");
                inputs.skipped_incomplete += 1;
            }
            WalkKind::NonUtf8 => {
                return Err(Error::UnsafeMemberName { name: entry.rel.to_owned() });
            }
            WalkKind::Special => {
                return Err(Error::Io {
                    op: "read input",
                    path: entry.path.to_path_buf(),
                    source: io::Error::new(
                        ErrorKind::InvalidInput,
                        "not a regular file, a directory or a link",
                    ),
                });
            }
        }
        Ok(())
    })?;
    // Member names and not walk order: the order of members is part of what makes an output
    // reproducible.
    inputs.files[first..].sort_unstable_by(|a, b| a.1.cmp(&b.1));
    Ok(())
}

/// Checks the member names of a write request the way `PackBuilder` will for an added file or a
/// copied member (the restore-name rules, and unique), so that a bad one fails before the output is
/// started and not hours in. A frontend calls it over every name it is about to add, for example the
/// members of all the containers of a merge. Reads nothing.
///
/// # Errors
/// [`Error::UnsafeMemberName`] for the first name a restore would refuse, or
/// [`Error::DuplicateMember`] for the first name given twice.
pub fn check_member_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<(), Error> {
    let mut seen = HashSet::new();
    for name in names {
        check_restore_name(name)?;
        if !seen.insert(name) {
            return Err(Error::DuplicateMember { name: name.to_owned() });
        }
    }
    Ok(())
}

/// [`check_member_names`] over a walk, with a log line naming the files behind a duplicate, which
/// the error does not carry.
///
/// # Errors
/// As [`check_member_names`].
fn check_names(files: &[(PathBuf, String)]) -> Result<(), Error> {
    check_member_names(files.iter().map(|(_, member)| member.as_str())).inspect_err(|error| {
        if let Error::DuplicateMember { name } = error {
            let paths = files.iter().filter(|(_, member)| member == name);
            let paths: Vec<_> = paths.map(|(path, _)| path.display().to_string()).collect();
            tracing::error!(member = %name, files = ?paths, "two inputs give one member name");
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Inputs, WalkKind, add_root, check_member_names, check_names, check_output, collect_inputs, walk,
    };
    use crate::error::Error;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::{env, fs, process};

    static NEXT: AtomicU32 = AtomicU32::new(0);

    /// A directory under the system temp directory, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!("ziptensors-inputs-{}-{id}", process::id()));
            fs::create_dir_all(&path).expect("create scratch");
            Self(path)
        }

        fn file(&self, relative: &str) {
            let path = self.0.join(relative);
            let parent = path.parent().expect("a file under the scratch directory");
            fs::create_dir_all(parent).expect("create parent");
            fs::write(path, b"x").expect("write file");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                eprintln!("cannot remove scratch: {error}");
            }
        }
    }

    fn file(member: &str) -> (PathBuf, String) {
        (PathBuf::from(member), member.to_owned())
    }

    fn members(inputs: &Inputs) -> Vec<&str> {
        inputs.files.iter().map(|(_, member)| member.as_str()).collect()
    }

    #[test]
    fn members_are_sorted_relative_to_the_root_with_forward_slashes() {
        let scratch = Scratch::new();
        for name in ["z.bin", "a/c.bin", "a/b/d.bin", "a.bin"] {
            scratch.file(name);
        }
        let inputs = collect_inputs(std::slice::from_ref(&scratch.0)).expect("walk");
        assert_eq!(members(&inputs), ["a.bin", "a/b/d.bin", "a/c.bin", "z.bin"]);
        assert_eq!(inputs.files[0].0, scratch.0.join("a.bin"));
    }

    #[test]
    fn a_file_root_is_named_after_itself_and_incomplete_files_are_counted() {
        let scratch = Scratch::new();
        scratch.file("dir/x.bin");
        scratch.file("dir/x.bin.incomplete");
        let mut inputs = Inputs::default();
        add_root(&mut inputs, &scratch.0.join("dir/x.bin")).expect("walk file");
        add_root(&mut inputs, &scratch.0.join("dir")).expect("walk directory");
        assert_eq!(members(&inputs), ["x.bin", "x.bin"]);
        assert_eq!(inputs.skipped_incomplete, 1);

        // The same two roots through the public walk: the name is given twice.
        let roots = [scratch.0.join("dir/x.bin"), scratch.0.join("dir")];
        let error = collect_inputs(&roots).expect_err("duplicate");
        assert!(matches!(&error, Error::DuplicateMember { name } if name == "x.bin"), "{error:?}");
    }

    #[test]
    fn an_output_the_walk_would_pick_up_is_refused() {
        let scratch = Scratch::new();
        scratch.file("tree/deep/x.bin");
        scratch.file("other/y.bin");
        let tree = scratch.0.join("tree");
        let inside = |path: &str, roots: &[PathBuf]| check_output(&scratch.0.join(path), roots);
        let refused = |result: Result<(), Error>| {
            assert!(matches!(result, Err(Error::OutputIsInput { .. })), "{result:?}");
        };
        let roots = [tree.clone()];
        // Whether the output exists yet does not matter: the next run would pick it up.
        refused(inside("tree/out.pack", &roots));
        refused(inside("tree/deep/x.bin", &roots));
        refused(inside("tree/deep/../out.pack", &roots));
        refused(inside("tree/out.pack", &[scratch.0.join("other"), tree.clone()]));
        refused(inside("other/y.bin", &[scratch.0.join("other/y.bin")]));

        inside("out.pack", &roots).expect("beside the tree");
        inside("other/out.pack", &roots).expect("in another tree");
        inside("tree/out.incomplete", &roots).expect("a name the walk skips");
        inside("missing/out.pack", &roots).expect("for the writer to report");
    }

    #[cfg(unix)]
    #[test]
    fn an_output_through_a_link_is_compared_where_it_really_is() {
        let scratch = Scratch::new();
        scratch.file("tree/x.bin");
        let link = scratch.0.join("link");
        std::os::unix::fs::symlink(scratch.0.join("tree"), &link).expect("links");
        let roots = [scratch.0.join("tree")];
        let error = check_output(&link.join("out.pack"), &roots).expect_err("inside the tree");
        assert!(matches!(error, Error::OutputIsInput { .. }), "{error:?}");
        check_output(&scratch.0.join("out.pack"), &[link]).expect("a linked root is not walked");
    }

    #[cfg(unix)]
    #[test]
    fn links_are_skipped_and_counted_at_any_depth() {
        let scratch = Scratch::new();
        scratch.file("tree/x.bin");
        std::os::unix::fs::symlink(scratch.0.join("tree/x.bin"), scratch.0.join("tree/y.bin"))
            .expect("links");
        std::os::unix::fs::symlink(scratch.0.join("tree"), scratch.0.join("root-link"))
            .expect("links");
        let roots = [scratch.0.join("tree"), scratch.0.join("root-link")];
        let inputs = collect_inputs(&roots).expect("walk");
        assert_eq!(members(&inputs), ["x.bin"]);
        assert_eq!(inputs.skipped_symlinks, 2);
    }

    #[test]
    fn the_walk_reports_directories_and_files_in_name_order() {
        let scratch = Scratch::new();
        for name in ["b/z.bin", "b/a.bin", "a.bin", "c.bin.incomplete"] {
            scratch.file(name);
        }
        let mut seen = Vec::new();
        walk(&scratch.0, &mut |entry| {
            seen.push((entry.rel.to_owned(), entry.kind));
            Ok(())
        })
        .expect("walk");
        let expected = [
            ("a.bin", WalkKind::File),
            ("b", WalkKind::Dir),
            ("c.bin.incomplete", WalkKind::Incomplete),
            ("b/a.bin", WalkKind::File),
            ("b/z.bin", WalkKind::File),
        ];
        let expected: Vec<_> = expected.iter().map(|(rel, kind)| ((*rel).to_owned(), *kind)).collect();
        assert_eq!(seen, expected);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_name_that_is_not_utf8_stops_collect_inputs() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let scratch = Scratch::new();
        scratch.file("ok.bin");
        fs::write(scratch.0.join(OsStr::from_bytes(b"bad\xff.bin")), b"x").expect("odd name");
        let error = collect_inputs(std::slice::from_ref(&scratch.0)).expect_err("not UTF-8");
        assert!(matches!(&error, Error::UnsafeMemberName { name } if name.starts_with("bad")), "{error:?}");
    }

    #[test]
    fn a_missing_root_is_an_error() {
        let scratch = Scratch::new();
        let absent = scratch.0.join("absent");
        let error = collect_inputs(std::slice::from_ref(&absent)).expect_err("missing root");
        let named = matches!(&error, Error::Io { op: "metadata", path, .. } if *path == absent);
        assert!(named, "{error:?}");
        let error = check_output(&scratch.0.join("out.pack"), &[absent]).expect_err("missing root");
        assert!(matches!(error, Error::Io { op: "metadata", .. }), "{error:?}");
    }

    #[test]
    fn names_are_checked_before_anything_starts() {
        assert!(check_names(&[file("a"), file("b/c")]).is_ok());
        let error = check_names(&[file("a"), file("a")]).expect_err("duplicate");
        assert!(matches!(&error, Error::DuplicateMember { name } if name == "a"), "{error:?}");
        let error = check_names(&[file("a:b")]).expect_err("unsafe");
        assert!(matches!(error, Error::UnsafeMemberName { .. }), "{error:?}");
    }

    #[test]
    fn member_names_of_a_request_are_unique_and_restorable() {
        check_member_names(["a", "b/c", "B/c"]).expect("distinct and safe");
        check_member_names([]).expect("nothing to add");
        let error = check_member_names(["x", "y", "x"]).expect_err("duplicate");
        assert!(matches!(&error, Error::DuplicateMember { name } if name == "x"), "{error:?}");
        let error = check_member_names(["../up", "../up"]).expect_err("unsafe first");
        assert!(matches!(error, Error::UnsafeMemberName { .. }), "{error:?}");
    }
}
