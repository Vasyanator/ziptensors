//! File: io/temp.rs
//! Purpose: the one owner of "temporary file beside the output, sync, close, rename, sync the
//! directory", the replace protocol of spec invariant 12, for the writer and for a restore alike.
//! Which protocol a writer ends with is decided by `write::commit`; this file owns only this one.
//!
//! Key structures: [`TempFile`].
//!
//! Key functions: [`same_file`]. Key constant: [`TEMP_FILE_SUFFIX`], the public name of these files.
//!
//! Implementation notes: the file sits beside the output, so the rename stays on one filesystem. The
//! handle is closed before the rename, as renaming an open file is not portable. The path lives in a
//! guard declared after the handle, so on every failure path, panics included, the handle closes first
//! and the file is then removed. A failure after the rename is [`Error::CommittedNotDurable`], never
//! [`Error::Io`], so that `Io` from a commit always means the output is untouched.

use std::fs::{self, File};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::error::Error;

/// Suffix that ends the name of every temporary file a write or a restore creates beside its output,
/// as `<output name>.<process id>-<counter>.ziptensors-tmp`. Part of the commit contract: such a file
/// is removed unless committed, so one found later was left by a process that died mid-write and can
/// be deleted. Nothing else this library creates ends with it.
pub const TEMP_FILE_SUFFIX: &str = ".ziptensors-tmp";

/// Distinguishes the temporary files of one process.
static NEXT_TEMP: AtomicU32 = AtomicU32::new(0);

/// A path that is removed on drop, unless it was renamed away.
#[derive(Debug)]
struct RemoveOnDrop {
    path: PathBuf,
    armed: bool,
}

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // The original failure is what the caller gets; a leftover is only worth a log line.
        if let Err(error) = fs::remove_file(&self.path) {
            tracing::warn!(path = %self.path.display(), %error, "cannot remove temporary file");
        }
    }
}

/// A new file beside an output, removed on drop unless committed.
#[derive(Debug)]
pub(crate) struct TempFile {
    /// Declared before `guard`, so that it closes first.
    file: File,
    guard: RemoveOnDrop,
}

impl TempFile {
    /// Creates a file with a new name beside `output`, opened for writing.
    ///
    /// # Errors
    /// Returns [`Error::Io`] with `op` `"create temporary file"` if `output` has no file name or the
    /// file cannot be created.
    pub(crate) fn beside(output: &Path) -> Result<Self, Error> {
        const OP: &str = "create temporary file";
        let name = output.file_name().ok_or_else(|| Error::Io {
            op: OP,
            path: output.to_path_buf(),
            source: io::Error::new(ErrorKind::InvalidInput, "the output path has no file name"),
        })?;
        loop {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let mut temp_name = name.to_os_string();
            temp_name.push(format!(".{}-{id}{TEMP_FILE_SUFFIX}", std::process::id()));
            let path = output.with_file_name(temp_name);
            match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    tracing::debug!(path = %path.display(), "created temporary file");
                    return Ok(Self { file, guard: RemoveOnDrop { path, armed: true } });
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
                Err(source) => return Err(Error::Io { op: OP, path, source }),
            }
        }
    }

    /// Path of the temporary file.
    pub(crate) fn path(&self) -> &Path {
        &self.guard.path
    }

    /// The open file and its path, borrowed together.
    pub(crate) fn parts_mut(&mut self) -> (&mut File, &Path) {
        (&mut self.file, &self.guard.path)
    }

    /// Syncs the file, closes it, renames it to `output`, replacing a file there, and on Unix syncs
    /// the directory of `output`, so that the rename survives a power loss.
    ///
    /// # Errors
    /// - [`Error::Io`] with `op` `"sync"` or `"rename"`. The temporary file is then removed, and what
    ///   was at `output` is untouched.
    /// - [`Error::CommittedNotDurable`] if the directory cannot be synced. The rename has happened:
    ///   the new file is at `output` and no temporary file remains, but a power loss may still bring
    ///   back what was there before. Never returned on Windows, where no directory is synced.
    pub(crate) fn commit(self, output: &Path) -> Result<(), Error> {
        self.commit_then(output, sync_directory_of)
    }

    /// [`TempFile::commit`] with `sync_directory` as the step after the rename, given the output
    /// path. The seam exists so tests can make that step fail; production passes
    /// [`sync_directory_of`] only.
    fn commit_then(
        self,
        output: &Path,
        sync_directory: fn(&Path) -> io::Result<()>,
    ) -> Result<(), Error> {
        let Self { file, mut guard } = self;
        file.sync_all().map_err(|source| Error::Io {
            op: "sync",
            path: guard.path.clone(),
            source,
        })?;
        drop(file);
        fs::rename(&guard.path, output).map_err(|source| Error::Io {
            op: "rename",
            path: output.to_path_buf(),
            source,
        })?;
        guard.armed = false;
        // From here on the output is replaced: a failure must not read as "nothing happened".
        sync_directory(output).map_err(|source| {
            tracing::error!(
                output = %output.display(),
                %source,
                "output committed, but its directory cannot be synced: the rename may not survive a power loss"
            );
            Error::CommittedNotDurable { path: output.to_path_buf(), source }
        })
    }
}

/// The directory that holds `path`.
#[cfg(unix)]
fn directory_of(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        Some(_) | None => Path::new("."),
    }
}

/// Makes the rename to `output` durable: until its directory is synced, a power loss can bring back
/// the old entry. A filesystem with no such operation refuses it as invalid or unsupported, which is
/// no failure of the commit, as the file itself is synced and in place.
///
/// # Errors
/// The OS error of opening or syncing the directory.
#[cfg(unix)]
fn sync_directory_of(output: &Path) -> io::Result<()> {
    let directory = directory_of(output);
    match File::open(directory)?.sync_all() {
        Ok(()) => Ok(()),
        Err(error) if matches!(error.kind(), ErrorKind::InvalidInput | ErrorKind::Unsupported) => {
            tracing::debug!(path = %directory.display(), %error, "directory cannot be synced here");
            Ok(())
        }
        Err(source) => Err(source),
    }
}

/// Windows has no directory handle to sync, and its rename is journaled by the filesystem, so there
/// is nothing to do and nothing can fail.
#[cfg(not(unix))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the signature is the post-rename step of `TempFile::commit_then` on every platform"
)]
fn sync_directory_of(_output: &Path) -> io::Result<()> {
    Ok(())
}

/// Whether `first` and `second` are one file: the same canonical path, which sees through symbolic
/// links, or on Unix the same device and inode, which also sees hard links. A path that does not exist
/// is no file, so nothing is the same as it.
///
/// # Errors
/// Returns [`Error::Io`] if a path exists but cannot be resolved.
pub(crate) fn same_file(first: &Path, second: &Path) -> Result<bool, Error> {
    let resolved = |path: &Path| match fs::canonicalize(path) {
        Ok(canonical) => Ok(Some(canonical)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(source) => Err(Error::Io { op: "resolve path", path: path.to_path_buf(), source }),
    };
    let (Some(first), Some(second)) = (resolved(first)?, resolved(second)?) else {
        return Ok(false);
    };
    let same = first == second;
    // The stable API has no file identity on Windows, so there the canonical path decides alone.
    #[cfg(unix)]
    let same = same || same_inode(&first, &second)?;
    Ok(same)
}

/// Whether two existing paths are one inode.
#[cfg(unix)]
fn same_inode(first: &Path, second: &Path) -> Result<bool, Error> {
    use std::os::unix::fs::MetadataExt;
    let identity = |path: &Path| {
        fs::metadata(path)
            .map(|meta| (meta.dev(), meta.ino()))
            .map_err(|source| Error::Io { op: "metadata", path: path.to_path_buf(), source })
    };
    Ok(identity(first)? == identity(second)?)
}


#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::directory_of;
    use super::{TEMP_FILE_SUFFIX, TempFile, same_file};
    use crate::error::Error;
    use std::fs;
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};

    /// A unique scratch directory, removed on drop.
    struct Dir(PathBuf);

    impl Dir {
        fn new(tag: &str) -> Self {
            let name = format!("ziptensors-temp-{tag}-{}", std::process::id());
            let path = std::env::temp_dir().join(name);
            fs::create_dir_all(&path).expect("scratch directory");
            Self(path)
        }

        fn entries(&self) -> Vec<String> {
            let entries = fs::read_dir(&self.0).expect("readable");
            entries
                .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
                .collect()
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                eprintln!("cannot remove {}: {error}", self.0.display());
            }
        }
    }

    #[test]
    fn a_dropped_file_leaves_nothing() {
        let dir = Dir::new("drop");
        let temp = TempFile::beside(&dir.0.join("out")).expect("creates");
        assert_eq!(dir.entries().len(), 1);
        assert_eq!(temp.path().parent(), Some(dir.0.as_path()));
        let name = temp.path().file_name().expect("a file name").to_string_lossy().into_owned();
        assert!(name.starts_with("out.") && name.ends_with(TEMP_FILE_SUFFIX), "{name}");
        drop(temp);
        assert_eq!(dir.entries(), [""; 0]);
    }

    #[test]
    fn a_commit_replaces_the_output_and_leaves_one_file() {
        let dir = Dir::new("commit");
        let output = dir.0.join("out");
        fs::write(&output, b"old").expect("writes");
        let mut temp = TempFile::beside(&output).expect("creates");
        temp.parts_mut().0.write_all(b"new").expect("writes");
        temp.commit(&output).expect("commits");
        assert_eq!(dir.entries(), ["out"]);
        assert_eq!(fs::read(&output).expect("reads"), b"new");
    }

    #[test]
    fn a_failed_directory_sync_reports_the_new_file_as_committed() {
        fn fail(_output: &Path) -> io::Result<()> {
            Err(io::Error::other("injected directory sync failure"))
        }
        let dir = Dir::new("dirsync");
        let output = dir.0.join("out");
        fs::write(&output, b"old").expect("writes");
        let mut temp = TempFile::beside(&output).expect("creates");
        temp.parts_mut().0.write_all(b"new").expect("writes");
        let error = temp.commit_then(&output, fail).expect_err("the directory sync fails");
        match &error {
            Error::CommittedNotDurable { path, source } => {
                assert_eq!(path, &output);
                assert_eq!(source.to_string(), "injected directory sync failure");
            }
            other => panic!("expected CommittedNotDurable, got {other:?}"),
        }
        assert_eq!(fs::read(&output).expect("reads"), b"new");
        assert_eq!(dir.entries(), ["out"]);
    }

    #[test]
    fn a_failed_rename_removes_the_file_and_keeps_the_target() {
        let dir = Dir::new("rename");
        let output = dir.0.join("out");
        fs::create_dir(&output).expect("a directory cannot be replaced by a file");
        fs::write(output.join("kept"), b"x").expect("writes");
        let temp = TempFile::beside(&output).expect("creates");
        let error = temp.commit(&output).expect_err("the target is a directory");
        assert!(matches!(error, Error::Io { op: "rename", .. }), "{error:?}");
        assert_eq!(dir.entries(), ["out"]);
        assert!(output.join("kept").exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_bare_file_name_lives_in_the_current_directory() {
        assert_eq!(directory_of(Path::new("out")), Path::new("."));
        assert_eq!(directory_of(Path::new("a/out")), Path::new("a"));
        assert_eq!(directory_of(Path::new("/out")), Path::new("/"));
    }

    #[test]
    fn same_file_sees_through_paths_and_not_between_files() {
        let dir = Dir::new("same");
        let (first, second) = (dir.0.join("first"), dir.0.join("second"));
        fs::write(&first, b"x").expect("writes");
        fs::write(&second, b"x").expect("writes");
        assert!(same_file(&first, &first).expect("resolves"));
        assert!(same_file(&first, &dir.0.join(".").join("first")).expect("resolves"));
        assert!(!same_file(&first, &second).expect("resolves"));
        assert!(!same_file(&first, &dir.0.join("missing")).expect("resolves"));
    }

    #[cfg(unix)]
    #[test]
    fn same_file_sees_links() {
        let dir = Dir::new("links");
        let file = dir.0.join("file");
        fs::write(&file, b"x").expect("writes");
        let (soft, hard) = (dir.0.join("soft"), dir.0.join("hard"));
        std::os::unix::fs::symlink(&file, &soft).expect("links");
        fs::hard_link(&file, &hard).expect("links");
        assert!(same_file(&file, &soft).expect("resolves"));
        assert!(same_file(&file, &hard).expect("resolves"));
    }

    #[test]
    fn an_output_path_with_no_file_name_is_refused() {
        let error = TempFile::beside(Path::new("/")).expect_err("no file name");
        assert!(matches!(error, Error::Io { op: "create temporary file", .. }), "{error:?}");
    }
}
