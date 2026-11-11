//! File: write/commit.rs
//! Purpose: the writer's side of the commit protocols of spec invariant 12 (a)/(b): where the bytes
//! of a container go, and how the file is ended.
//!
//! Main responsibilities:
//! - [`Sink`], the append position over one of three targets: a temporary file beside the output
//!   (replace protocol), the end of an existing container (append protocol), or an existing
//!   container rewritten from its header on (in place). Rollback, and the header rewrite that sets
//!   feature bits.
//! - [`Ending`] and [`Sink::finish`], the one owner of how a file is ended: the rename of a temporary
//!   file, by `io::TempFile`; a footer written after durable data and index; one final cut of a file
//!   rewritten in place. [`Sink::abandon`] is the failure ending of each target, and [`Sink::fail`]
//!   the error a caller reports after it ([`Error::TrailingBytes`], [`Error::DamagedInPlace`]).
//! - The moves of an in-place rewrite ([`Sink::copy_within`]), and its seal: the first write zeroes
//!   the footer of the container it rewrites.
//! - [`store_index`] and [`write_index`], the stored index and its footer.
//! - [`verify_output`], the full decode through the real reader, over any source.
//!
//! Key structures: [`Sink`], [`Ending`], [`StoredIndex`], [`SourceHash`]; under `cfg(test)`
//! [`SinkFault`], the crash injection seam.
//!
//! Key functions: [`verify_output`], [`store_index`], [`write_index`].
//!
//! Implementation notes: verification opens the source as any reader does, with default limits and no
//! special path: a file that would need raised limits to open is refused here and not found out by its
//! user. A rollback of a temporary file truncates, since bytes past the logical end must not survive
//! into the index offset. An append never writes below its frontier, the length of the container it
//! extends, so cutting back to the frontier restores that container byte for byte. An append works on
//! a handle its caller opened (never the path again), writes positionally, and hands out clones of
//! that handle for verification, so what is verified and written is the file the caller validated. An in-place
//! rewrite works the same way on its caller's handle. It never truncates before its end: what lies
//! past the position is dead data by the edit's plan, and the one final cut removes it. A move reads
//! a piece before it writes it and never writes past its source, and a rollback never goes below the
//! moved data, so no write lands on a kept byte not yet copied.

use std::fs::File;
use std::io::{self, ErrorKind, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::error::Error;
use crate::format::{FOOTER_LEN, Footer, HEADER_LEN, Header, IndexBuilder};
use crate::io::{FileSource, RandomAccess, TempFile};
use crate::read::{Container, OpenOptions};

use super::report::{ClassTable, VerifyReport, unreferenced_table};

/// zstd level of the stored index.
const INDEX_LEVEL: i32 = 3;

/// Where the bytes of a [`Sink`] go.
#[derive(Debug)]
enum Target {
    /// A new temporary file beside the output, renamed over it at the end.
    Temp(TempFile),
    /// The end of an existing container.
    Append(AppendFile),
    /// An existing container, rewritten from the end of its header on.
    InPlace(InPlaceFile),
}

/// An existing container rewritten in place: its kept chunks moved towards the header, new data
/// after them, then index and footer, and one final cut. Nothing is ever truncated before that cut.
#[derive(Debug)]
struct InPlaceFile {
    /// Opened for reading and writing by the caller, written positionally.
    file: File,
    /// A clone of `file`, for the reads of [`Sink::copy_within`]; its length is the container's.
    reader: FileSource,
    path: PathBuf,
    /// Offset of the container's footer, zeroed by the first write ([`Sink::seal`]).
    footer_at: u64,
    /// The first write happened: from here on the file is damaged until the rewrite completes.
    sealed: bool,
    /// End of the moved data: a rollback below it would let new data overwrite kept chunks.
    floor: u64,
}

/// An existing container opened for an append. Unless settled, dropping it cuts the file back to
/// `frontier`, so that an append abandoned on any path, panics included, leaves the old container.
#[derive(Debug)]
struct AppendFile {
    file: File,
    path: PathBuf,
    /// Length of the container before the append: nothing below it is ever written.
    frontier: u64,
    /// The append ended, or its cut back was already attempted; drop does nothing.
    settled: bool,
}

impl Drop for AppendFile {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        tracing::warn!(
            path = %self.path.display(),
            frontier = self.frontier,
            "append abandoned: cutting the file back to its previous container"
        );
        if let Err(source) = cut_back(&self.file, self.frontier, None) {
            tracing::error!(
                path = %self.path.display(),
                valid_len = self.frontier,
                %source,
                "cannot cut an abandoned append back; `ziptensors recover` repairs the file"
            );
        }
    }
}

/// Writes all of `bytes` at `off`, independent of the file cursor.
#[cfg(unix)]
fn write_all_at_os(file: &File, off: u64, bytes: &[u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(bytes, off)
}

/// Writes all of `bytes` at `off`. `seek_write` moves the cursor, which no append write relies on.
#[cfg(windows)]
fn write_all_at_os(file: &File, off: u64, bytes: &[u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut pos = off;
    let mut done = 0;
    while done < bytes.len() {
        match file.seek_write(&bytes[done..], pos) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                done += n;
                // n <= bytes.len(), and a file cannot reach 2^64 bytes.
                pos += u64::try_from(n).map_err(io::Error::other)?;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Cuts `file` to `len` if it is longer, and syncs it. `injected` replaces the cut by a failure,
/// the test seam of [`SinkFault`]; production passes `None`.
fn cut_back(file: &File, len: u64, injected: Option<io::Error>) -> io::Result<()> {
    if file.metadata()?.len() == len {
        return Ok(());
    }
    match injected {
        Some(error) => Err(error),
        None => file.set_len(len),
    }?;
    file.sync_all()
}

/// How [`Sink::finish`] ends the file. Each ending belongs to one target.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Ending<'a> {
    /// Replace protocol, for [`Sink::create`]: sync, close, rename the temporary file over this
    /// output, sync its directory. Everything, footer included, is written before.
    Rename(&'a Path),
    /// Append protocol, for [`Sink::append_to`]: data and stored index are written before; the sink
    /// syncs them, writes this footer, and syncs again.
    AppendFooter {
        /// The footer of the new index, from [`store_index`].
        footer: [u8; 64],
    },
    /// In-place rewrite, for [`Sink::in_place`]: everything, footer included, is written before and
    /// ends at `final_len`; the sink syncs, cuts the file to `final_len` once, and syncs again.
    InPlace {
        /// Length of the new container, which must be the sink's position.
        final_len: u64,
    },
}

/// What [`SinkFault`] does once the sink has written `after_bytes` bytes.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SinkFaultMode {
    /// The write that crosses the threshold writes the bytes below it, then fails with
    /// [`Error::Io`]: the normal failure path, rollback and cut back included.
    FailWrite,
    /// As `FailWrite`, and every later operation fails too, cut back and sync included, without
    /// touching the file: a process that died mid-write, whose bytes stay on disk.
    Die,
    /// Every truncation fails from the threshold on: rollback of a temporary file or an append, the
    /// cut back of an append, the final cut of an in-place rewrite. Writes are not affected.
    FailSetLen,
}

/// Crash injection for tests of the commit protocols (plan D12). Bytes are counted over the sink's
/// writes from its creation, the header of a temporary file included.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SinkFault {
    /// Bytes written before the fault takes effect.
    pub(crate) after_bytes: u64,
    /// What happens then.
    pub(crate) mode: SinkFaultMode,
}

/// The state of the fault seam of one sink.
#[cfg(test)]
#[derive(Debug, Default)]
struct FaultState {
    fault: Option<SinkFault>,
    written: u64,
    dead: bool,
}

#[cfg(test)]
impl FaultState {
    /// How many of the next `len` bytes may be written before an injected write failure, or `None`
    /// if the write is not affected.
    fn allowance(&self, len: usize) -> Option<usize> {
        let fault = self.fault?;
        if !matches!(fault.mode, SinkFaultMode::FailWrite | SinkFaultMode::Die) {
            return None;
        }
        let left = fault.after_bytes.saturating_sub(self.written);
        let left = usize::try_from(left).unwrap_or(usize::MAX);
        (len > left).then_some(left)
    }

    /// The injected failure of a truncation, if one is due.
    fn set_len_error(&self) -> Option<io::Error> {
        let fault = self.fault?;
        let due = self.dead
            || (fault.mode == SinkFaultMode::FailSetLen && self.written >= fault.after_bytes);
        due.then(|| io::Error::other("injected truncation failure"))
    }
}

/// The output of a writer: an open file and a position that can move back.
///
/// After a failed write the file holds an unknown tail, so the sink refuses all further work; only
/// its failure ending, [`Sink::abandon`] or a drop, still acts on the file.
#[derive(Debug)]
pub(crate) struct Sink {
    target: Target,
    /// Logical end of the file.
    position: u64,
    /// A write or a rollback failed.
    broken: bool,
    #[cfg(test)]
    fault: FaultState,
}

impl Sink {
    /// The sink over `target`, at `position`.
    fn over(target: Target, position: u64) -> Self {
        Self {
            target,
            position,
            broken: false,
            #[cfg(test)]
            fault: FaultState::default(),
        }
    }

    /// Creates a temporary file beside `output` and writes the header.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if `output` has no file name, or the file cannot be created or written.
    pub(crate) fn create(output: &Path) -> Result<Self, Error> {
        let temp = TempFile::beside(output)?;
        let mut sink = Self::over(Target::Temp(temp), 0);
        sink.write(&Header::current().to_bytes())?;
        Ok(sink)
    }

    /// Appends a new generation to the container open as `file`, opened for writing from `path`:
    /// the position and the frontier are its length, and no byte below the frontier is ever
    /// written. The caller has validated the container, through a handle on this same file (a
    /// clone of `file`, never a second open of `path`, which may name another file by now); its
    /// header stays as it is. Writes are positional, so readers on a clone of `file` may read
    /// between them: on Windows such a read moves the cursor the handles share.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be measured, and [`Error::NotZiptensors`] if it is
    /// too short for a header and a footer.
    pub(crate) fn append_to(file: File, path: &Path) -> Result<Self, Error> {
        let frontier = file
            .metadata()
            .map_err(|source| Error::Io { op: "metadata", path: path.to_path_buf(), source })?
            .len();
        if frontier < HEADER_LEN + FOOTER_LEN {
            return Err(Error::NotZiptensors);
        }
        tracing::info!(path = %path.display(), frontier, "appending to container");
        let append = AppendFile { file, path: path.to_path_buf(), frontier, settled: false };
        Ok(Self::over(Target::Append(append), frontier))
    }

    /// Rewrites in place the container open as `file`, opened for reading and writing from `path`,
    /// positioned right after its header. As for [`Sink::append_to`], the caller validated the
    /// container through a handle on this same file and never through `path` again, and writes are
    /// positional. The file keeps its length until [`Ending::InPlace`]: a rollback never truncates
    /// it. The first write of any kind is preceded by [`Sink::seal`], which zeroes the container's
    /// footer: from then on the file is damaged until the rewrite completes, and the failure ending
    /// ([`Sink::fail`]) reports [`Error::DamagedInPlace`].
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be measured or its handle cloned, and
    /// [`Error::NotZiptensors`] if it is too short for a header and a footer.
    pub(crate) fn in_place(file: File, path: &Path) -> Result<Self, Error> {
        let io_error = |op, source| Error::Io { op, path: path.to_path_buf(), source };
        let reader = file.try_clone().map_err(|source| io_error("clone handle", source))?;
        let reader = FileSource::from_file(reader, path)?;
        let len = reader.len();
        if len < HEADER_LEN + FOOTER_LEN {
            return Err(Error::NotZiptensors);
        }
        tracing::info!(path = %path.display(), len, "rewriting container in place");
        let in_place = InPlaceFile {
            file,
            reader,
            path: path.to_path_buf(),
            footer_at: len - FOOTER_LEN,
            sealed: false,
            floor: HEADER_LEN,
        };
        Ok(Self::over(Target::InPlace(in_place), HEADER_LEN))
    }

    /// Arms the crash injection seam.
    #[cfg(test)]
    pub(crate) fn set_fault(&mut self, fault: SinkFault) {
        self.fault.fault = Some(fault);
    }

    /// Path of the file written: the temporary file, or the container appended to or rewritten.
    pub(crate) fn path(&self) -> &Path {
        match &self.target {
            Target::Temp(temp) => temp.path(),
            Target::Append(append) => &append.path,
            Target::InPlace(in_place) => &in_place.path,
        }
    }

    /// Logical end of the file, where the next write lands.
    pub(crate) fn position(&self) -> u64 {
        self.position
    }

    /// The file, unless an earlier failure made it unusable.
    fn file(&mut self) -> Result<&mut File, Error> {
        if self.broken {
            return Err(Error::Io {
                op: "write",
                path: self.path().to_path_buf(),
                source: io::Error::other("the output is unusable after an earlier write failure"),
            });
        }
        Ok(match &mut self.target {
            Target::Temp(temp) => temp.parts_mut().0,
            Target::Append(append) => &mut append.file,
            Target::InPlace(in_place) => &mut in_place.file,
        })
    }

    /// The injected failure of a truncation, if the test seam has one due.
    #[cfg(test)]
    fn injected_set_len_error(&self) -> Option<io::Error> {
        self.fault.set_len_error()
    }

    /// Outside tests no failure is ever injected.
    #[cfg(not(test))]
    #[expect(clippy::unused_self, reason = "the signature of the test version, which reads the seam")]
    fn injected_set_len_error(&self) -> Option<io::Error> {
        None
    }

    /// Fails if a write failed before, so that a caller stops before it does more work.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the sink is broken.
    pub(crate) fn check(&mut self) -> Result<(), Error> {
        self.file().map(|_| ())
    }

    /// Marks the sink unusable. For a failure that left the pack's records out of step with the file.
    pub(crate) fn poison(&mut self) {
        self.broken = true;
    }

    /// Appends `bytes` at the logical end. The first write of an in-place rewrite seals it first.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the write fails; the sink then refuses everything else.
    pub(crate) fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.seal()?;
        let at = self.position;
        self.put(at, bytes)?;
        // Lossless: `usize` is at most 64 bits on every supported target. A file cannot reach 2^64.
        self.position += bytes.len() as u64;
        Ok(())
    }

    /// Writes `bytes` at `at` through the test seam, without moving the position.
    fn put(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        #[cfg(test)]
        if let Some(allowed) = self.fault.allowance(bytes.len()) {
            // A crash mid-write leaves the bytes before it on disk.
            self.put_all(at, &bytes[..allowed])?;
            return Err(self.inject_write_failure());
        }
        self.put_all(at, bytes)
    }

    /// Before the first write of an in-place rewrite: zeroes the container's footer and syncs, so
    /// that from the first moved byte on, the file never ends in the footer of a generation whose
    /// chunks may have moved. Without it a torn rewrite could still open, and `recover`, which takes
    /// the last valid footer of a file, would stop at that one rather than at the footer this
    /// rewrite writes. Nothing for other targets, or once done.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the write or the sync fails; the sink then refuses everything else.
    fn seal(&mut self) -> Result<(), Error> {
        let Target::InPlace(in_place) = &mut self.target else {
            return Ok(());
        };
        if in_place.sealed {
            return Ok(());
        }
        // Set before the write: a failed write may have changed the footer too.
        in_place.sealed = true;
        let footer_at = in_place.footer_at;
        tracing::warn!(
            path = %in_place.path.display(),
            "rewriting in place: an interruption from now on damages the container"
        );
        self.put(footer_at, &[0; 64])?;
        self.sync()
    }

    /// Copies the `len` bytes at `src` to `dst`, which must be the position, forward in pieces of
    /// at most `buffer.len()` bytes, and moves the position past them: one move of an in-place
    /// rewrite. `dst <= src` is required. A piece is read whole before it is written, and the write
    /// position never passes the read position, so every write lands on bytes already read: those
    /// of this move, or bytes before it that no later move reads (a plan's moves are in increasing
    /// `src` order). With `src == dst` the bytes are already in place and nothing is read or
    /// written. Moves the floor of rollbacks to the new position.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the sink is no in-place rewrite, `dst` is not the position or lies
    /// past `src`, the buffer is empty while bytes must move, or a read or write fails; the sink
    /// then refuses everything else. [`Error::OutOfBounds`] if the range is outside the container.
    pub(crate) fn copy_within(
        &mut self,
        src: u64,
        dst: u64,
        len: u64,
        buffer: &mut [u8],
    ) -> Result<(), Error> {
        let position = self.position;
        if !matches!(self.target, Target::InPlace(_)) {
            return Err(self.wrong_ending("in-place"));
        }
        if dst != position || src < dst || (src > dst && buffer.is_empty()) {
            self.broken = true;
            return Err(Error::Io {
                op: "move",
                path: self.path().to_path_buf(),
                source: io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "a move of {len} bytes from {src} to {dst} at position {position} is not \
                         a forward move"
                    ),
                ),
            });
        }
        let end = src.checked_add(len).ok_or(Error::OutOfBounds {
            offset: src,
            len,
            file_len: u64::MAX,
        })?;
        if src > position {
            let mut at = src;
            while at < end {
                let piece = usize::try_from(end - at).map_or(buffer.len(), |left| left.min(buffer.len()));
                let piece = &mut buffer[..piece];
                self.read_source(at, piece)?;
                self.write(piece)?;
                // Lossless: `usize` is at most 64 bits on every supported target.
                at += piece.len() as u64;
            }
        } else {
            self.position = end;
        }
        if let Target::InPlace(in_place) = &mut self.target {
            in_place.floor = self.position;
        }
        Ok(())
    }

    /// Reads `buf` at `at` from the container an in-place rewrite works on, through its handle.
    fn read_source(&mut self, at: u64, buf: &mut [u8]) -> Result<(), Error> {
        self.check()?;
        let Target::InPlace(in_place) = &self.target else {
            return Err(self.wrong_ending("in-place"));
        };
        let result = in_place.reader.read_exact_at(at, buf);
        if result.is_err() {
            self.broken = true;
        }
        result
    }

    /// The file as it is on disk now, read through the sink's own handle and never through its
    /// path: what the commit verifies is the file it writes. Only for an append or an in-place
    /// rewrite.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the sink is broken, is a temporary file, or its handle cannot be
    /// cloned or measured.
    pub(crate) fn written_source(&mut self) -> Result<FileSource, Error> {
        if matches!(self.target, Target::Temp(_)) {
            // A temporary file writes at its cursor; a read through a shared handle would move it
            // on Windows. It is read by its path, which only this process knows.
            return Err(self.wrong_ending("append or in-place"));
        }
        let path = self.path().to_path_buf();
        let file = self.file()?.try_clone();
        let file = file.map_err(|source| Error::Io { op: "clone handle", path: path.clone(), source })?;
        FileSource::from_file(file, &path)
    }

    /// Writes all of `bytes` at `at`, without the test seam: positionally for an append and an
    /// in-place rewrite, whatever the cursor of the handle they share with readers; at the cursor
    /// for a temporary file, which has its handle to itself and whose cursor is always `at`.
    fn put_all(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        let positional = !matches!(self.target, Target::Temp(_));
        let file = self.file()?;
        let result = if positional { write_all_at_os(file, at, bytes) } else { file.write_all(bytes) };
        result.map_err(|source| {
            self.broken = true;
            Error::Io { op: "write", path: self.path().to_path_buf(), source }
        })?;
        #[cfg(test)]
        {
            // Lossless: `usize` is at most 64 bits on every supported target.
            self.fault.written += bytes.len() as u64;
        }
        Ok(())
    }

    /// Breaks the sink as the armed fault says and returns the error of the failed write.
    #[cfg(test)]
    fn inject_write_failure(&mut self) -> Error {
        self.broken = true;
        if self.fault.fault.is_some_and(|fault| fault.mode == SinkFaultMode::Die) {
            self.fault.dead = true;
            // A dead process cuts nothing back.
            if let Target::Append(append) = &mut self.target {
                append.settled = true;
            }
        }
        Error::Io {
            op: "write",
            path: self.path().to_path_buf(),
            source: io::Error::other("injected write failure"),
        }
    }

    /// Moves the logical end back to `position`. A temporary file and an append are cut there, so
    /// nothing written after it survives; an in-place rewrite only moves its position, since the
    /// bytes past it are dead data the final cut removes.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the file cannot be cut or the position moved, if an append is asked
    /// to go below its frontier, which would cut the container it extends, or an in-place rewrite
    /// below the end of its moved data, which new data would then overwrite; the sink then refuses
    /// everything else.
    pub(crate) fn rollback(&mut self, position: u64) -> Result<(), Error> {
        if position == self.position {
            return Ok(());
        }
        let limit = match &self.target {
            Target::Temp(_) => None,
            Target::Append(append) => Some(("the append frontier", append.frontier)),
            Target::InPlace(in_place) => Some(("the end of the moved data", in_place.floor)),
        };
        if let Some((limit_name, limit)) = limit
            && position < limit
        {
            self.broken = true;
            tracing::error!(
                path = %self.path().display(),
                position,
                limit,
                limit_name,
                "rollback below the sink's limit refused"
            );
            return Err(Error::Io {
                op: "roll back",
                path: self.path().to_path_buf(),
                source: io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("position {position} is below {limit_name} {limit}"),
                ),
            });
        }
        if matches!(self.target, Target::InPlace(_)) {
            // Positional writes: nothing to cut and no cursor to move.
            self.check()?;
            self.position = position;
            return Ok(());
        }
        let injected = self.injected_set_len_error();
        let file = self.file()?;
        let cut = match injected {
            Some(error) => Err(error),
            None => file.set_len(position),
        };
        let result = cut.and_then(|()| file.seek(SeekFrom::Start(position)));
        result.map_err(|source| {
            self.broken = true;
            Error::Io { op: "roll back", path: self.path().to_path_buf(), source }
        })?;
        self.position = position;
        Ok(())
    }

    /// Writes `header` over the one at the start of the file, leaving the logical end where it is. For
    /// feature bits known only once the blobs are in, spec section 7a; the commit's verification then
    /// reads the header like any reader. An in-place rewrite is sealed first and writes it through
    /// the test seam.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if a seek or the write fails; the sink then refuses everything else. An
    /// append refuses with [`Error::Io`] and writes nothing: its header is below the frontier.
    pub(crate) fn patch_header(&mut self, header: &Header) -> Result<(), Error> {
        if matches!(self.target, Target::Append(_)) {
            return Err(Error::Io {
                op: "write header",
                path: self.path().to_path_buf(),
                source: io::Error::new(
                    ErrorKind::InvalidInput,
                    "an append keeps the header of the container it extends",
                ),
            });
        }
        if matches!(self.target, Target::InPlace(_)) {
            self.seal()?;
            return self.put(0, &header.to_bytes());
        }
        let end = self.position;
        let file = self.file()?;
        let result = file
            .seek(SeekFrom::Start(0))
            .and_then(|_| file.write_all(&header.to_bytes()))
            .and_then(|()| file.seek(SeekFrom::Start(end)));
        result.map_err(|source| {
            self.broken = true;
            Error::Io { op: "write header", path: self.path().to_path_buf(), source }
        })?;
        Ok(())
    }

    /// Syncs the file.
    fn sync(&mut self) -> Result<(), Error> {
        let result = self.file()?.sync_all();
        result.map_err(|source| {
            self.broken = true;
            Error::Io { op: "sync", path: self.path().to_path_buf(), source }
        })
    }

    /// The error of an ending that does not belong to the sink's target, a caller's bug.
    fn wrong_ending(&self, ending: &str) -> Error {
        Error::Io {
            op: "commit",
            path: self.path().to_path_buf(),
            source: io::Error::new(
                ErrorKind::InvalidInput,
                format!("the {ending} ending does not belong to this output"),
            ),
        }
    }

    /// Commits a temporary file to `output`: [`Sink::finish`] with [`Ending::Rename`].
    ///
    /// # Errors
    /// As [`Sink::finish`].
    pub(crate) fn commit(self, output: &Path) -> Result<(), Error> {
        self.finish(Ending::Rename(output))
    }

    /// Ends the file as `ending` says. Each ending belongs to one target; another is refused.
    ///
    /// - [`Ending::Rename`]: [`TempFile::commit`], spec invariant 12 (a), replace protocol.
    /// - [`Ending::AppendFooter`]: sync of data and index, the footer, sync: append protocol. A
    ///   failure before the second sync returns cuts the file back to the frontier, which restores
    ///   the previous container byte for byte.
    /// - [`Ending::InPlace`]: sync, one cut to `final_len` if the file is longer, sync.
    ///
    /// # Errors
    /// - [`Error::Io`] if the sink is broken, the ending does not belong to it, or a sync, write or
    ///   rename fails. A temporary file is then removed; an append is cut back to its frontier.
    /// - [`Error::CommittedNotDurable`] if a temporary file was renamed to its output but the
    ///   directory could not be synced.
    /// - [`Error::TrailingBytes`] if an append failed and its partial tail could not be cut, with the
    ///   frontier as `valid_len`: the previous container is intact before it. Or if an in-place
    ///   rewrite could not be cut to `final_len`, with `final_len` as `valid_len`: the new container
    ///   is complete and durable before it.
    /// - [`Error::DamagedInPlace`] for any other failure of an in-place ending: the rewrite began.
    pub(crate) fn finish(mut self, ending: Ending<'_>) -> Result<(), Error> {
        match ending {
            Ending::Rename(output) => {
                self.check()?;
                match self.target {
                    Target::Temp(temp) => temp.commit(output),
                    Target::Append(_) | Target::InPlace(_) => Err(self.wrong_ending("rename")),
                }
            }
            Ending::AppendFooter { footer } => match self.append_footer(&footer) {
                Ok(()) => {
                    if let Target::Append(append) = &mut self.target {
                        append.settled = true;
                    }
                    tracing::info!(
                        path = %self.path().display(),
                        len = self.position,
                        "appended a new generation"
                    );
                    Ok(())
                }
                Err(error) => {
                    tracing::error!(path = %self.path().display(), %error, "append failed");
                    self.cut_back_append()?;
                    Err(error)
                }
            },
            Ending::InPlace { final_len } => match self.finish_in_place(final_len) {
                Err(error @ Error::TrailingBytes { .. }) => Err(error),
                Err(error) => Err(self.fail(error)),
                Ok(()) => Ok(()),
            },
        }
    }

    /// The append protocol after data and index: sync, footer, sync.
    fn append_footer(&mut self, footer: &[u8; 64]) -> Result<(), Error> {
        if !matches!(self.target, Target::Append(_)) {
            return Err(self.wrong_ending("append"));
        }
        // The footer is written only over durable data and index, so a reader that finds it finds
        // everything it points at.
        self.sync()?;
        self.write(footer)?;
        self.sync()
    }

    /// Cuts an append back to its frontier, once; nothing for other targets or a dead test sink.
    ///
    /// # Errors
    /// Returns [`Error::TrailingBytes`] with the frontier as `valid_len` if the cut or its sync fails.
    fn cut_back_append(&mut self) -> Result<(), Error> {
        let injected = self.injected_set_len_error();
        let Target::Append(append) = &mut self.target else {
            return Ok(());
        };
        if append.settled {
            return Ok(());
        }
        append.settled = true;
        let (path, valid_len) = (append.path.clone(), append.frontier);
        match cut_back(&append.file, valid_len, injected) {
            Ok(()) => {
                tracing::warn!(
                    path = %path.display(),
                    valid_len,
                    "append cut back to the previous container"
                );
                Ok(())
            }
            Err(source) => {
                tracing::error!(
                    path = %path.display(),
                    valid_len,
                    %source,
                    "cannot cut a failed append back; the old container is intact below valid_len"
                );
                Err(Error::TrailingBytes { path, valid_len })
            }
        }
    }

    /// The end of an in-place rewrite: sync, one cut to `final_len`, sync.
    fn finish_in_place(&mut self, final_len: u64) -> Result<(), Error> {
        if !matches!(self.target, Target::InPlace(_)) {
            return Err(self.wrong_ending("in-place"));
        }
        if final_len != self.position {
            self.broken = true;
            return Err(Error::Io {
                op: "commit",
                path: self.path().to_path_buf(),
                source: io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "in-place ending at {final_len}, the written bytes end at {}",
                        self.position
                    ),
                ),
            });
        }
        // The new container is durable before anything past it is cut, so a failed cut leaves a
        // complete container followed by dead bytes.
        self.sync()?;
        let injected = self.injected_set_len_error();
        let path = self.path().to_path_buf();
        let file = self.file()?;
        let metadata = file.metadata();
        let len = metadata
            .map_err(|source| Error::Io { op: "metadata", path: path.clone(), source })?
            .len();
        if len > final_len {
            let cut = match injected {
                Some(error) => Err(error),
                None => file.set_len(final_len),
            };
            if let Err(source) = cut {
                tracing::error!(
                    path = %path.display(),
                    valid_len = final_len,
                    len,
                    %source,
                    "cannot cut the rewritten container to its length"
                );
                return Err(Error::TrailingBytes { path, valid_len: final_len });
            }
            self.sync()?;
        }
        tracing::info!(
            path = %path.display(),
            len = final_len,
            previous_len = len,
            "rewrote container in place"
        );
        Ok(())
    }

    /// The failure ending: a temporary file is removed; an append is cut back to its frontier; an
    /// in-place rewrite is left as it is, since its bytes are what a recovery can still read.
    ///
    /// # Errors
    /// Returns [`Error::TrailingBytes`] if an append cannot be cut back, with the frontier as
    /// `valid_len`.
    pub(crate) fn abandon(mut self) -> Result<(), Error> {
        match &self.target {
            Target::Temp(_) => Ok(()),
            Target::Append(_) => self.cut_back_append(),
            Target::InPlace(in_place) => {
                tracing::warn!(
                    path = %in_place.path.display(),
                    sealed = in_place.sealed,
                    "in-place rewrite abandoned; the file is left as it is"
                );
                Ok(())
            }
        }
    }

    /// The failure ending after `error`, which it returns as the caller must report it: as
    /// [`Sink::abandon`], then [`Error::TrailingBytes`] if an append could not be cut back, and
    /// [`Error::DamagedInPlace`] wrapping `error` if an in-place rewrite was sealed (its first write
    /// happened, so the container is damaged). `error` itself otherwise: a temporary file, an
    /// append cut back, or an in-place rewrite that never wrote.
    pub(crate) fn fail(self, error: Error) -> Error {
        if let Target::InPlace(in_place) = &self.target
            && in_place.sealed
        {
            let path = in_place.path.clone();
            tracing::error!(
                path = %path.display(),
                %error,
                "the in-place rewrite failed after it began: the container is damaged; restore it \
                 from a backup or its original inputs"
            );
            // Nothing to cut or remove: the bytes are left for whatever can still be read.
            return match error {
                Error::DamagedInPlace { .. } => error,
                cause => Error::DamagedInPlace { path, cause: Box::new(cause) },
            };
        }
        match self.abandon() {
            Ok(()) => error,
            Err(trailing) => {
                tracing::error!(%error, "write failed and the output could not be restored");
                trailing
            }
        }
    }
}

/// An index in its stored form, placed at a known offset.
#[derive(Debug)]
pub(crate) struct StoredIndex {
    /// The zstd frame of the serialized index.
    pub(crate) stored: Vec<u8>,
    /// The footer pointing at it.
    pub(crate) footer: [u8; 64],
}

/// Serializes and compresses `index`, and builds the footer for it at `index_offset` with
/// `generation`. Writes nothing, so that a caller can verify the result before it is on disk.
///
/// # Errors
/// Returns [`Error::Zstd`] if the index cannot be compressed.
pub(crate) fn store_index(
    index: &IndexBuilder,
    index_offset: u64,
    generation: u64,
) -> Result<StoredIndex, Error> {
    let raw = index.serialize();
    let stored = zstd::bulk::compress(&raw, INDEX_LEVEL)
        .map_err(|error| Error::Zstd { detail: error.to_string() })?;
    // Lossless: `usize` is at most 64 bits on every supported target.
    let stored_len = stored.len() as u64;
    let footer = Footer::for_index(index_offset, stored_len, &raw, generation).to_bytes();
    tracing::debug!(raw = raw.len(), stored = stored_len, generation, "stored index");
    Ok(StoredIndex { stored, footer })
}

/// Serializes the index and appends it and its footer, with `generation`
/// ([`crate::format::GENERATION_INITIAL`] for a fresh file, the previous value + 1 for an edit), to a
/// file that is not yet visible. Returns the stored length of the index.
///
/// # Errors
/// Returns [`Error::Zstd`] if the index cannot be compressed and [`Error::Io`] if a write fails.
pub(crate) fn write_index(
    sink: &mut Sink,
    index: &IndexBuilder,
    generation: u64,
) -> Result<u64, Error> {
    let index_offset = sink.position();
    let StoredIndex { stored, footer } = store_index(index, index_offset, generation)?;
    sink.write(&stored)?;
    sink.write(&footer)?;
    // Lossless: `usize` is at most 64 bits on every supported target.
    let stored_len = stored.len() as u64;
    debug_assert_eq!(sink.position(), index_offset + stored_len + FOOTER_LEN);
    debug_assert!(index_offset >= HEADER_LEN);
    tracing::debug!(stored = stored_len, "wrote index");
    Ok(stored_len)
}

/// The hash a member must decode to, taken from its source and not from the file being verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceHash {
    /// Member name.
    pub(crate) name: String,
    /// BLAKE3 of the source content.
    pub(crate) blake3: [u8; 32],
}

/// Opens `source` through the real reader, with its default limits, and proves every member: the
/// record states the source hash, and the whole member decodes to it. So the index is re-parsed and
/// validated here, which `IndexBuilder` does not do. `source` is the file as it will be once
/// committed: a temporary file, or a view of a file with its new index still in memory
/// ([`crate::io::SpliceSource`]). `path` names it in logs.
///
/// Costs one sequential read of each member, however they share blobs. Also returns the accounting
/// of the blobs no member reaches, from the same index, by [`unreferenced_table`].
///
/// # Errors
/// Returns any error of [`Container::from_source`] or of a read, and [`Error::VerifyFailed`] if a
/// hash differs. [`Error::MalformedIndex`] for `MEMB` if the source does not hold exactly `sources`,
/// in order.
pub(crate) fn verify_output(
    source: Arc<dyn RandomAccess>,
    path: &Path,
    threads: usize,
    sources: &[SourceHash],
) -> Result<(VerifyReport, ClassTable), Error> {
    let started = Instant::now();
    let options = OpenOptions { threads, ..OpenOptions::default() };
    let container = Container::from_source(source, &options)?;
    let members = container.members();
    if members.len() != sources.len() {
        return Err(Error::MalformedIndex {
            section: "MEMB",
            detail: format!("{} members written, {} expected", members.len(), sources.len()),
        });
    }
    let mut bytes = 0_u64;
    for (info, source) in members.iter().zip(sources) {
        if info.name != source.name {
            return Err(Error::MalformedIndex {
                section: "MEMB",
                detail: format!("member {:?} written, {:?} expected", info.name, source.name),
            });
        }
        if info.blake3 != source.blake3 {
            return Err(Error::VerifyFailed {
                member: info.name.clone(),
                expected: blake3::Hash::from_bytes(source.blake3),
                actual: blake3::Hash::from_bytes(info.blake3),
            });
        }
        container.member(&info.name)?.verify()?;
        bytes += info.len;
    }
    let elapsed = started.elapsed();
    tracing::info!(path = %path.display(), bytes, ?elapsed, "verified output");
    let unreferenced = unreferenced_table(container.index())?;
    Ok((VerifyReport { bytes, elapsed }, unreferenced))
}

#[cfg(test)]
mod tests {
    use super::{Ending, Sink, SinkFault, SinkFaultMode, SourceHash, store_index, verify_output};
    use crate::chain::FEATURE_DELTA;
    use crate::error::Error;
    use crate::format::{FOOTER_LEN, Header, IndexBuilder, parse_footer, parse_header};
    use crate::io::{FileSource, RandomAccess, SpliceSource};
    use crate::read::{Container, OpenOptions};
    use crate::types::Mode;
    use crate::write::{EncodeOptions, encode_file};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// The container at `path`, opened for reading and writing as the edit opens it.
    fn open_rw(path: &Path) -> fs::File {
        fs::OpenOptions::new().read(true).write(true).open(path).expect("opens")
    }

    /// An append sink over the container at `path`.
    fn append(path: &Path) -> Result<Sink, Error> {
        Sink::append_to(open_rw(path), path)
    }

    /// An in-place sink over the container at `path`.
    fn in_place(path: &Path) -> Result<Sink, Error> {
        Sink::in_place(open_rw(path), path)
    }

    /// A unique scratch directory, removed on drop.
    struct Dir(PathBuf);

    impl Dir {
        fn new(tag: &str) -> Self {
            let name = format!("ziptensors-commit-{tag}-{}", std::process::id());
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

    /// A file of `len` bytes `0, 1, 2, …` at `dir/name`, standing for an existing container.
    fn existing(dir: &Dir, name: &str, len: usize) -> (PathBuf, Vec<u8>) {
        let path = dir.0.join(name);
        let bytes: Vec<u8> = (0..len).map(|at| u8::try_from(at % 251).expect("below 251")).collect();
        fs::write(&path, &bytes).expect("writes");
        (path, bytes)
    }

    /// A real container of one opaque member, written through the replace protocol.
    fn packed(dir: &Dir) -> PathBuf {
        let input = dir.0.join("input.bin");
        let content: Vec<u8> =
            (0..200_000_u32).map(|at| at.wrapping_mul(2_654_435_761).to_le_bytes()[3]).collect();
        fs::write(&input, content).expect("input");
        let output = dir.0.join("packed.ziptensors");
        let options = EncodeOptions { mode: Mode::Max, threads: 1 };
        encode_file(&input, &output, &options).expect("packs");
        output
    }

    fn fault(after_bytes: u64, mode: SinkFaultMode) -> SinkFault {
        SinkFault { after_bytes, mode }
    }

    #[test]
    fn a_rollback_cuts_the_file_and_moves_the_position() {
        let dir = Dir::new("rollback");
        let mut sink = Sink::create(&dir.0.join("out")).expect("creates");
        assert_eq!(sink.position(), 64);
        sink.write(&[1; 100]).expect("writes");
        sink.rollback(64 + 10).expect("rolls back");
        sink.write(&[2; 5]).expect("writes");
        let bytes = fs::read(sink.path()).expect("reads");
        assert_eq!(bytes.len(), 64 + 15);
        assert_eq!(&bytes[64 + 9..64 + 11], [1, 2]);
    }

    #[test]
    fn a_patched_header_keeps_the_end_and_the_data() {
        let dir = Dir::new("patch");
        let mut sink = Sink::create(&dir.0.join("out")).expect("creates");
        sink.write(&[1; 10]).expect("writes");
        let header = Header { required_features: FEATURE_DELTA, ..Header::current() };
        sink.patch_header(&header).expect("patches");
        assert_eq!(sink.position(), 74);
        sink.write(&[2; 3]).expect("writes");
        let bytes = fs::read(sink.path()).expect("reads");
        assert_eq!(bytes.len(), 77);
        assert_eq!(&bytes[64..], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]);
        let first: [u8; 64] = bytes[..64].try_into().expect("64 bytes");
        assert_eq!(parse_header(&first).expect("valid"), header);
    }

    #[test]
    fn a_poisoned_sink_refuses_work_and_still_cleans_up() {
        let dir = Dir::new("poison");
        let output = dir.0.join("out");
        let mut sink = Sink::create(&output).expect("creates");
        sink.poison();
        assert!(sink.check().is_err());
        assert!(sink.write(b"x").is_err());
        assert!(sink.commit(&output).is_err());
        assert_eq!(dir.entries(), [""; 0]);
    }

    #[test]
    fn an_append_starts_at_the_end_and_refuses_a_rollback_below_the_frontier() {
        let dir = Dir::new("frontier");
        let (path, before) = existing(&dir, "c", 200);
        let mut sink = append(&path).expect("opens");
        assert_eq!(sink.position(), 200);
        sink.write(&[7; 30]).expect("writes");
        sink.rollback(210).expect("above the frontier");
        assert_eq!(fs::read(&path).expect("reads").len(), 210, "a rollback cuts an append");
        sink.rollback(200).expect("down to the frontier");
        sink.write(&[8; 5]).expect("writes");
        let error = sink.rollback(199).expect_err("below the frontier");
        assert!(matches!(error, Error::Io { op: "roll back", .. }), "{error:?}");
        assert!(sink.write(b"x").is_err(), "a refused rollback breaks the sink");
        let after = fs::read(&path).expect("reads");
        assert_eq!(after[..200], before[..], "nothing below the frontier changed");
        sink.abandon().expect("cuts back");
        assert_eq!(fs::read(&path).expect("reads"), before);
    }

    #[test]
    fn an_append_keeps_the_header() {
        let dir = Dir::new("appendheader");
        let (path, before) = existing(&dir, "c", 200);
        let mut sink = append(&path).expect("opens");
        let error = sink.patch_header(&Header::current()).expect_err("below the frontier");
        assert!(matches!(error, Error::Io { op: "write header", .. }), "{error:?}");
        drop(sink);
        assert_eq!(fs::read(&path).expect("reads"), before);
    }

    #[test]
    fn a_dropped_append_is_cut_back() {
        let dir = Dir::new("appenddrop");
        let (path, before) = existing(&dir, "c", 200);
        let mut sink = append(&path).expect("opens");
        sink.write(&[7; 300]).expect("writes");
        drop(sink);
        assert_eq!(fs::read(&path).expect("reads"), before);
    }

    #[test]
    fn an_append_writes_at_its_position_while_a_shared_handle_reads() {
        let dir = Dir::new("appendshared");
        let (path, before) = existing(&dir, "c", 200);
        let file = fs::OpenOptions::new().read(true).write(true).open(&path).expect("opens");
        let reader = FileSource::from_file(file.try_clone().expect("clones"), &path).expect("reads");
        let mut sink = Sink::append_to(file, &path).expect("opens");
        sink.write(&[7; 30]).expect("writes");
        // On Windows this read moves the cursor the two handles share.
        let mut head = [0_u8; 10];
        reader.read_exact_at(0, &mut head).expect("reads");
        sink.write(&[8; 5]).expect("writes");
        let appended = sink.written_source().expect("an append");
        assert_eq!(appended.len(), 235, "the source of an append is its handle, as written");
        appended.read_exact_at(0, &mut head).expect("reads");
        sink.write(&[9; 4]).expect("writes");
        sink.finish(Ending::AppendFooter { footer: [6; 64] }).expect("ends");
        let after = fs::read(&path).expect("reads");
        let tail = [[7_u8; 30].as_slice(), &[8; 5], &[9; 4], &[6; 64]].concat();
        assert_eq!(after[..200], before[..]);
        assert_eq!(after[200..], tail[..]);
    }

    #[test]
    fn a_temporary_file_hands_out_no_handle() {
        let dir = Dir::new("appendsource");
        let mut sink = Sink::create(&dir.0.join("out")).expect("creates");
        let error = sink.written_source().expect_err("a temporary file");
        assert!(matches!(error, Error::Io { op: "commit", .. }), "{error:?}");
    }

    #[test]
    fn a_file_too_short_for_a_container_is_refused() {
        let dir = Dir::new("short");
        let (path, _) = existing(&dir, "c", 127);
        assert!(matches!(append(&path), Err(Error::NotZiptensors)));
        let (path, _) = existing(&dir, "d", 127);
        assert!(matches!(in_place(&path), Err(Error::NotZiptensors)));
    }

    #[test]
    fn an_append_footer_follows_its_data() {
        let dir = Dir::new("appendok");
        let (path, before) = existing(&dir, "c", 200);
        let mut sink = append(&path).expect("opens");
        sink.write(&[7; 30]).expect("writes");
        sink.finish(Ending::AppendFooter { footer: [9; 64] }).expect("ends");
        let after = fs::read(&path).expect("reads");
        assert_eq!(after.len(), 200 + 30 + 64);
        assert_eq!(after[..200], before[..]);
        assert_eq!(after[230..], [9; 64]);
    }

    #[test]
    fn a_failed_footer_write_cuts_the_append_back() {
        let dir = Dir::new("appendfail");
        let (path, before) = existing(&dir, "c", 200);
        let mut sink = append(&path).expect("opens");
        sink.set_fault(fault(30 + 10, SinkFaultMode::FailWrite));
        sink.write(&[7; 30]).expect("writes");
        let error = sink.finish(Ending::AppendFooter { footer: [9; 64] }).expect_err("fails");
        assert!(matches!(error, Error::Io { op: "write", .. }), "{error:?}");
        assert_eq!(fs::read(&path).expect("reads"), before);
    }

    #[test]
    fn an_append_that_cannot_be_cut_back_reports_trailing_bytes() {
        let dir = Dir::new("appendtrail");
        let (path, before) = existing(&dir, "c", 200);
        let mut sink = append(&path).expect("opens");
        sink.set_fault(fault(0, SinkFaultMode::FailSetLen));
        sink.write(&[7; 30]).expect("writes");
        let error = sink.abandon().expect_err("the cut fails");
        assert!(
            matches!(&error, Error::TrailingBytes { path: at, valid_len: 200 } if *at == path),
            "{error:?}"
        );
        let after = fs::read(&path).expect("reads");
        assert_eq!(after.len(), 230);
        assert_eq!(after[..200], before[..]);
    }

    #[test]
    fn a_dead_append_leaves_its_torn_tail() {
        let dir = Dir::new("appenddie");
        let (path, before) = existing(&dir, "c", 200);
        let mut sink = append(&path).expect("opens");
        sink.set_fault(fault(25, SinkFaultMode::Die));
        sink.write(&[7; 20]).expect("writes");
        let error = sink.write(&[8; 20]).expect_err("dies");
        assert!(matches!(error, Error::Io { op: "write", .. }), "{error:?}");
        assert!(sink.rollback(200).is_err(), "a dead sink does nothing more");
        let error = sink.finish(Ending::AppendFooter { footer: [9; 64] }).expect_err("dead");
        assert!(matches!(error, Error::Io { .. }), "{error:?}");
        let after = fs::read(&path).expect("reads");
        assert_eq!(after.len(), 225, "the bytes before the death stay");
        assert_eq!(after[..200], before[..]);
        assert_eq!(after[200..], [[7; 20].as_slice(), &[8; 5]].concat());
    }

    #[test]
    fn an_in_place_rollback_keeps_the_bytes_and_the_final_cut_removes_them() {
        let dir = Dir::new("inplace");
        let (path, before) = existing(&dir, "c", 300);
        let mut sink = in_place(&path).expect("opens");
        assert_eq!(sink.position(), 64);
        sink.write(&[7; 50]).expect("writes");
        sink.rollback(64 + 20).expect("moves back");
        let during = fs::read(&path).expect("reads");
        assert_eq!(during.len(), 300, "an in-place rollback never truncates");
        assert_eq!(during[64..114], [7; 50]);
        assert_eq!(during[114..236], before[114..236]);
        assert_eq!(during[236..], [0; 64], "the first write zeroed the old footer");
        sink.write(&[8; 10]).expect("writes");
        let final_len = sink.position();
        sink.finish(Ending::InPlace { final_len }).expect("ends");
        let after = fs::read(&path).expect("reads");
        assert_eq!(after.len(), 94);
        assert_eq!(after[..64], before[..64]);
        assert_eq!(after[64..], [[7; 20].as_slice(), &[8; 10]].concat());
    }

    #[test]
    fn an_in_place_ending_that_cannot_cut_reports_trailing_bytes() {
        let dir = Dir::new("inplacetrail");
        let (path, _) = existing(&dir, "c", 300);
        let mut sink = in_place(&path).expect("opens");
        sink.set_fault(fault(0, SinkFaultMode::FailSetLen));
        sink.write(&[7; 50]).expect("writes");
        let error = sink.finish(Ending::InPlace { final_len: 114 }).expect_err("the cut fails");
        assert!(matches!(error, Error::TrailingBytes { valid_len: 114, .. }), "{error:?}");
        let after = fs::read(&path).expect("reads");
        assert_eq!(after.len(), 300);
        assert_eq!(after[64..114], [7; 50]);
    }

    #[test]
    fn an_in_place_ending_must_match_the_written_end_and_its_target() {
        let dir = Dir::new("inplacewrong");
        let (path, before) = existing(&dir, "c", 300);
        let sink = in_place(&path).expect("opens");
        let error = sink.finish(Ending::InPlace { final_len: 65 }).expect_err("mismatch");
        assert!(matches!(error, Error::Io { op: "commit", .. }), "{error:?}");
        assert_eq!(fs::read(&path).expect("reads"), before, "nothing was cut");
        let sink = in_place(&path).expect("opens");
        let error = sink.finish(Ending::AppendFooter { footer: [0; 64] }).expect_err("wrong ending");
        assert!(matches!(error, Error::Io { op: "commit", .. }), "{error:?}");
        let sink = Sink::create(&dir.0.join("out")).expect("creates");
        let error = sink.finish(Ending::InPlace { final_len: 64 }).expect_err("wrong ending");
        assert!(matches!(error, Error::Io { op: "commit", .. }), "{error:?}");
        assert_eq!(dir.entries(), ["c"], "the temporary file is gone");
    }

    #[test]
    fn a_move_copies_forward_in_pieces_and_sets_the_rollback_floor() {
        let dir = Dir::new("inplacemove");
        let (path, before) = existing(&dir, "c", 400);
        let mut sink = in_place(&path).expect("opens");
        // Bytes already in place: nothing is written, so nothing is sealed.
        sink.copy_within(64, 64, 16, &mut []).expect("in place");
        assert_eq!(sink.position(), 80);
        assert_eq!(fs::read(&path).expect("reads"), before, "nothing written");
        // A move that overlaps itself, through a buffer smaller than the overlap.
        let mut buffer = [0_u8; 7];
        let error = sink.copy_within(100, 81, 90, &mut buffer).expect_err("not the position");
        assert!(matches!(error, Error::Io { op: "move", .. }), "{error:?}");
        let mut sink = in_place(&path).expect("opens again");
        sink.copy_within(64, 64, 16, &mut []).expect("in place");
        sink.copy_within(100, 80, 90, &mut buffer).expect("moves");
        assert_eq!(sink.position(), 170);
        let after = fs::read(&path).expect("reads");
        assert_eq!(after[..80], before[..80]);
        assert_eq!(after[80..170], before[100..190], "a forward memmove");
        assert_eq!(after[170..336], before[170..336]);
        assert_eq!(after[336..], [0; 64], "sealed by the first write");
        let error = sink.rollback(169).expect_err("below the moved data");
        assert!(matches!(error, Error::Io { op: "roll back", .. }), "{error:?}");
        let error = sink.fail(Error::NotZiptensors);
        let Error::DamagedInPlace { path: at, cause } = error else { panic!("{error:?}") };
        assert_eq!(at, path);
        assert!(matches!(*cause, Error::NotZiptensors), "{cause:?}");
    }

    #[test]
    fn a_move_backwards_is_refused_and_an_unsealed_failure_is_no_damage() {
        let dir = Dir::new("inplaceback");
        let (path, before) = existing(&dir, "c", 300);
        let mut sink = in_place(&path).expect("opens");
        sink.copy_within(64, 64, 40, &mut []).expect("in place");
        let error = sink.copy_within(90, 104, 10, &mut [0; 8]).expect_err("a move backwards");
        assert!(matches!(error, Error::Io { op: "move", .. }), "{error:?}");
        let error = sink.fail(Error::NotZiptensors);
        assert!(matches!(error, Error::NotZiptensors), "{error:?}");
        assert_eq!(fs::read(&path).expect("reads"), before, "nothing was written");
        let mut sink = Sink::create(&dir.0.join("out")).expect("creates");
        let error = sink.copy_within(100, 64, 1, &mut [0; 8]).expect_err("no in-place sink");
        assert!(matches!(error, Error::Io { op: "commit", .. }), "{error:?}");
    }

    #[test]
    fn a_death_while_sealing_leaves_a_damaged_file() {
        let dir = Dir::new("inplaceseal");
        let (path, before) = existing(&dir, "c", 300);
        let mut sink = in_place(&path).expect("opens");
        sink.set_fault(fault(10, SinkFaultMode::Die));
        let error = sink.write(&[7; 5]).expect_err("dies in the seal");
        let error = sink.fail(error);
        assert!(matches!(error, Error::DamagedInPlace { .. }), "{error:?}");
        let after = fs::read(&path).expect("reads");
        assert_eq!(after[..236], before[..236], "only the footer was touched");
        assert_eq!(after[236..246], [0; 10]);
        assert_eq!(after[246..], before[246..]);
    }

    #[test]
    fn a_temporary_file_rollback_can_fail_by_the_seam() {
        let dir = Dir::new("tempseam");
        let mut sink = Sink::create(&dir.0.join("out")).expect("creates");
        sink.set_fault(fault(0, SinkFaultMode::FailSetLen));
        sink.write(&[1; 10]).expect("writes");
        let error = sink.rollback(64).expect_err("the cut fails");
        assert!(matches!(error, Error::Io { op: "roll back", .. }), "{error:?}");
        assert!(sink.check().is_err());
    }

    #[test]
    fn a_seed_of_a_packed_index_has_the_records_of_its_source() {
        let dir = Dir::new("seed");
        let path = packed(&dir);
        let container = Container::open(&path, &OpenOptions::default()).expect("opens");
        let index = container.index();
        let seeded = IndexBuilder::seed(index, Ok).expect("seeds");
        let limits = crate::format::Limits::default();
        let reparsed = crate::format::Index::parse(&seeded.serialize(), &limits).expect("parses");
        assert_ne!(index.chunks(), []);
        assert_eq!(reparsed.steps(), index.steps());
        assert_eq!(reparsed.chains(), index.chains());
        assert_eq!(reparsed.blobs(), index.blobs());
        assert_eq!(reparsed.chunks(), index.chunks());
        assert_eq!(reparsed.params(), index.params());
    }

    /// Adds the records of every member of `index` to `builder`, as an edit that keeps them does.
    fn carry_members(index: &crate::format::Index, builder: &mut IndexBuilder) {
        for member in index.members() {
            let name = builder.add_string(index.string(member.name).expect("utf-8")).expect("fits");
            let extents = index.member_extents(member).expect("in range");
            assert!(index.member_tensors(member).expect("in range").is_empty(), "opaque member");
            let first_extent = extents.iter().map(|extent| builder.add_extent(*extent)).min();
            let record = crate::format::MemberRecord {
                name,
                first_extent: first_extent.unwrap_or(0),
                first_tensor: 0,
                ..*member
            };
            builder.add_member(record).expect("fits");
        }
    }

    #[test]
    fn an_appended_generation_is_verified_through_a_splice_and_then_opens() {
        let dir = Dir::new("appendgen");
        let path = packed(&dir);
        let before = fs::read(&path).expect("reads");
        let container = Container::open(&path, &OpenOptions::default()).expect("opens");
        let mut builder = IndexBuilder::seed(container.index(), Ok).expect("seeds");
        carry_members(container.index(), &mut builder);
        let sources: Vec<SourceHash> = container
            .members()
            .iter()
            .map(|info| SourceHash { name: info.name.clone(), blake3: info.blake3 })
            .collect();
        drop(container);

        let mut sink = append(&path).expect("opens");
        let index_offset = sink.position();
        let stored = store_index(&builder, index_offset, 2).expect("stores");
        // The new generation is proven before any byte of it is on disk.
        let base: Arc<dyn RandomAccess> = Arc::new(FileSource::open(&path).expect("opens"));
        let tail = [stored.stored.as_slice(), &stored.footer].concat();
        let view = SpliceSource::new(base, index_offset, tail).expect("fits");
        verify_output(Arc::new(view), &path, 1, &sources).expect("verifies");
        assert_eq!(fs::read(&path).expect("reads"), before, "verification wrote nothing");

        sink.write(&stored.stored).expect("writes");
        sink.finish(Ending::AppendFooter { footer: stored.footer }).expect("ends");
        let after = fs::read(&path).expect("reads");
        assert_eq!(after[..before.len()], before[..], "the previous generation is untouched");
        let footer: [u8; 64] = after[after.len() - 64..].try_into().expect("64 bytes");
        assert_eq!(parse_footer(&footer).expect("valid").generation, 2);
        let reopened = Container::open(&path, &OpenOptions::default()).expect("opens");
        for info in reopened.members() {
            reopened.member(&info.name).expect("exists").verify().expect("verifies");
        }
        let expected_len = index_offset + stored.stored.len() as u64 + FOOTER_LEN;
        assert_eq!(after.len() as u64, expected_len);
    }

    #[test]
    fn verification_through_a_view_refuses_a_wrong_hash() {
        let dir = Dir::new("verifyview");
        let path = packed(&dir);
        let source: Arc<dyn RandomAccess> = Arc::new(FileSource::open(&path).expect("opens"));
        let wrong = [SourceHash { name: "input.bin".into(), blake3: [0; 32] }];
        let error = verify_output(source, Path::new("view"), 1, &wrong).expect_err("wrong hash");
        assert!(matches!(error, Error::VerifyFailed { .. }), "{error:?}");
    }
}
