//! File: write/builder.rs
//! Purpose: the writer's public API: [`encode_file`] and [`PackBuilder`].
//!
//! Main responsibilities: checks on what is added, the records of each member, the checks of inputs,
//! the base of a file stored with delta blobs, and the order of the commit: header features, index,
//! verify, re-check inputs, sync, rename.
//!
//! Key structures: [`EncodeOptions`], [`PackBuilder`].
//!
//! Key functions: [`encode_file`].
//!
//! Implementation notes: a file that fails to add leaves the pack usable, with at most unreferenced
//! complete blobs, because a member's records are added only after all its bytes are in. A failed
//! write leaves the sink broken, and everything after it fails. Compression state is dropped before
//! verification, so the reader's own memory bound does not stack on the writer's. Symbolic links are
//! refused here and not skipped: skipping is a decision of a frontend, spec section 2. Of each member
//! added from a file the builder keeps its path, its stamp and its extents, so that a later file can
//! name it as a delta base; its tensor table is scanned again from the stamped file when that
//! happens, rather than kept for every member.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::SystemTime;

use crate::chain::FEATURE_DELTA;
use crate::error::Error;
use crate::format::{
    ExtentRecord, Header, IndexBuilder, MemberRecord, TensorRecord, check_restore_name,
};
use crate::io::{FileSource, RandomAccess, same_file};
use crate::profile::{normalize, scan};
use crate::read::Container;
use crate::types::{Dtype, Mode, Profile, RangePlan};

use super::commit::{Sink, SourceHash, verify_output, write_index};
use super::dedup::BlobTable;
use super::ingest::{Compressor, DeltaInput, IngestContext, ingest};
use super::merge::{CopiedSource, CopyTarget, copy_member};
use super::pairing::{BaseCatalog, BaseRange};
use super::policy::tries_delta;
use super::report::{ClassTable, EncodeReport, MemberReport};

/// How to write a container. There is no `Default`, since the two entry points have different defaults
/// by spec section 11: see [`EncodeOptions::for_encode`] and [`EncodeOptions::for_pack`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Which transforms to try. Never changes what a reader restores.
    pub mode: Mode,
    /// Compression threads, also the threads of the final verification. 0 means one per logical CPU;
    /// 1 runs compression inline on the calling thread. The output bytes do not depend on this.
    pub threads: usize,
}

impl EncodeOptions {
    /// The defaults of `encode`: [`Mode::Balanced`], all CPUs.
    #[must_use]
    pub fn for_encode() -> Self {
        Self { mode: Mode::Balanced, threads: 0 }
    }

    /// The defaults of `pack`: [`Mode::Max`], all CPUs.
    #[must_use]
    pub fn for_pack() -> Self {
        Self { mode: Mode::Max, threads: 0 }
    }

    /// The thread count in force.
    fn thread_count(&self) -> usize {
        if self.threads != 0 {
            return self.threads;
        }
        match thread::available_parallelism() {
            Ok(count) => count.get(),
            Err(error) => {
                tracing::warn!(%error, "CPU count is unknown, writing on one thread");
                1
            }
        }
    }
}

/// What an input looked like before it was read, to compare at commit.
#[derive(Debug)]
struct InputStamp {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
}

impl InputStamp {
    /// Takes the stamp of `path`, which must not be a symbolic link.
    fn of(path: &Path) -> Result<Self, Error> {
        let io_error = |op, source| Error::Io { op, path: path.to_path_buf(), source };
        let meta = fs::symlink_metadata(path).map_err(|e| io_error("metadata", e))?;
        if meta.is_symlink() {
            return Err(Error::SymlinkInput { path: path.to_path_buf() });
        }
        let modified = meta.modified().map_err(|e| io_error("metadata", e))?;
        Ok(Self { path: path.to_path_buf(), len: meta.len(), modified })
    }

    /// Checks that the file still looks the same.
    ///
    /// # Errors
    /// Returns [`Error::InputChanged`] if it does not, or is gone, and [`Error::Io`] if it cannot be
    /// looked at.
    fn check(&self) -> Result<(), Error> {
        let changed = || Error::InputChanged { path: self.path.clone() };
        let meta = match fs::symlink_metadata(&self.path) {
            Ok(meta) => meta,
            Err(error) if error.kind() == ErrorKind::NotFound => return Err(changed()),
            Err(source) => {
                return Err(Error::Io { op: "metadata", path: self.path.clone(), source });
            }
        };
        let modified = meta.modified().map_err(|source| Error::Io {
            op: "metadata",
            path: self.path.clone(),
            source,
        })?;
        if meta.len() != self.len || modified != self.modified {
            return Err(changed());
        }
        Ok(())
    }
}

/// A member added from a file, kept so that a later file can name it as its delta base.
#[derive(Debug)]
struct FileMember {
    /// The source file.
    path: PathBuf,
    /// Its stamp in [`PackBuilder::inputs`]. Always in range: the stamp is pushed before the member
    /// is recorded here, and stamps are never removed.
    stamp: usize,
    /// The member's extents, with blob ids of this pack.
    extents: Vec<ExtentRecord>,
}

/// The base of a file being added: its reopened source and the base range of each typed range.
#[derive(Debug)]
struct PreparedBase {
    source: FileSource,
    bases: Vec<Option<BaseRange>>,
    /// The base's stamp in [`PackBuilder::inputs`].
    stamp: usize,
}

/// One row of a member's tensor table, however it was obtained.
struct TensorRow<'a> {
    name: &'a str,
    offset: u64,
    len: u64,
    shape: &'a [u64],
    dtype: Dtype,
}

/// The parts of a member record that do not depend on where it came from.
struct MemberParts<'a> {
    name: &'a str,
    len: u64,
    blake3: [u8; 32],
    profile: Profile,
}

/// Builds a container of any number of members. Nothing appears at the output until
/// [`PackBuilder::finish`] has proven the whole file, and then only as one rename. Dropping a builder
/// removes its temporary file.
#[derive(Debug)]
pub struct PackBuilder {
    output: PathBuf,
    mode: Mode,
    threads: usize,
    sink: Sink,
    index: IndexBuilder,
    compressor: Compressor,
    blobs: BlobTable,
    next_blob: u32,
    extent_total: u64,
    tensor_total: u32,
    names: HashSet<String>,
    inputs: Vec<InputStamp>,
    /// Members added from files, by name.
    files: HashMap<String, FileMember>,
    sources: Vec<SourceHash>,
    copied: Vec<CopiedSource>,
    members: Vec<MemberReport>,
    classes: ClassTable,
    /// Test only: runs on the temporary file just before verification.
    #[cfg(test)]
    pub(super) fault: Option<fn(&Path)>,
}

impl PackBuilder {
    /// Starts a container that will end up at `output`. Creates a temporary file in the same directory
    /// and starts the compression threads.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the temporary file or the threads cannot be created, and [`Error::Zstd`]
    /// if a zstd context cannot be.
    pub fn new(output: &Path, options: &EncodeOptions) -> Result<Self, Error> {
        let threads = options.thread_count();
        let mode = options.mode;
        tracing::info!(output = %output.display(), ?mode, threads, "starting container");
        let sink = Sink::create(output)?;
        Ok(Self {
            output: output.to_path_buf(),
            mode,
            threads,
            sink,
            index: IndexBuilder::new(),
            compressor: Compressor::new(threads, output)?,
            blobs: BlobTable::default(),
            next_blob: 0,
            extent_total: 0,
            tensor_total: 0,
            names: HashSet::new(),
            inputs: Vec::new(),
            files: HashMap::new(),
            sources: Vec::new(),
            copied: Vec::new(),
            members: Vec::new(),
            classes: ClassTable::default(),
            #[cfg(test)]
            fault: None,
        })
    }

    /// Checks a new member name: the rules of a restore, and unique in this pack.
    fn check_name(&mut self, member: &str) -> Result<(), Error> {
        self.sink.check()?;
        check_restore_name(member)?;
        if self.names.contains(member) {
            return Err(Error::MalformedIndex {
                section: "MEMB",
                detail: format!("member name {member:?} is used twice"),
            });
        }
        Ok(())
    }

    /// Adds the file at `path` as the member `member`, reading it once in order. Recognizes the format
    /// by content, and stores bytes outside what it recognized as skeleton.
    ///
    /// On `Err` the member is not added and the builder stays usable, unless the output itself failed.
    /// Everything then fails, down to [`PackBuilder::finish`].
    ///
    /// # Errors
    /// - [`Error::UnsafeMemberName`] if a restore would refuse `member`.
    /// - [`Error::MalformedIndex`] for `MEMB` if this pack has a member called that.
    /// - [`Error::OutputIsInput`] if `path` is the output of this builder.
    /// - [`Error::SymlinkInput`] if `path` is a symbolic link. It is not followed, and a frontend
    ///   decides whether to skip it.
    /// - [`Error::Io`] for any failed file operation.
    /// - [`Error::InputChanged`] if the file changed while it was read. This compares length and
    ///   modification time, so an edit that keeps both commits the bytes that were read.
    /// - [`Error::Zstd`] if compression fails.
    pub fn add_file(&mut self, path: &Path, member: &str) -> Result<MemberReport, Error> {
        self.add_from_file(path, member, None)
    }

    /// Adds the file at `path` as the member `member`, as [`PackBuilder::add_file`] does, and stores
    /// its tensors that pair with tensors of `base_member` as delta blobs where a trial finds that
    /// smaller, spec section 7a.
    ///
    /// `base_member` must be a member this builder added from a file, by this method or by
    /// [`PackBuilder::add_file`], and its file must still be as it was then: its bytes are read again.
    /// A tensor pairs with a base tensor of the same name, or of the same name after dropping a
    /// publication prefix such as `model.diffusion_model.` and reading a norm's `scale` as `weight`,
    /// and in both cases of the same length and dtype. A tensor equal to its base becomes an extent
    /// over the base's blob, by identity, and no delta blob. Only [`Mode::Max`] stores delta blobs;
    /// under the other modes this is [`PackBuilder::add_file`] after the check of `base_member`.
    ///
    /// # Errors
    /// Every error of [`PackBuilder::add_file`], and:
    /// - [`Error::MemberNotFound`] if this builder added no member called `base_member` from a file.
    /// - [`Error::InputChanged`] if the base file differs from when it was added, or changes during
    ///   the read.
    pub fn add_file_delta(
        &mut self,
        path: &Path,
        member: &str,
        base_member: &str,
    ) -> Result<MemberReport, Error> {
        self.add_from_file(path, member, Some(base_member))
    }

    /// Reopens the file of `base_member` and pairs the typed ranges of `plan` with its tensors.
    /// `None` if the mode stores no delta blobs.
    fn prepare_base(
        &self,
        base_member: &str,
        plan: &RangePlan,
    ) -> Result<Option<PreparedBase>, Error> {
        let base = self
            .files
            .get(base_member)
            .ok_or_else(|| Error::MemberNotFound { name: base_member.to_owned() })?;
        if !tries_delta(self.mode) {
            tracing::debug!(base_member, mode = ?self.mode, "this mode stores no delta blobs");
            return Ok(None);
        }
        // The extents of the base describe the bytes it had when it was added.
        let stamp = &self.inputs[base.stamp];
        stamp.check()?;
        let source = FileSource::open(&base.path)?;
        if source.len() != stamp.len {
            return Err(Error::InputChanged { path: base.path.clone() });
        }
        let base_plan = normalize(scan(&source)?, source.len());
        let catalog = BaseCatalog::new(&base_plan.tensors, &base.extents);
        let bases = catalog.bases_of(plan);
        tracing::info!(
            base_member,
            base_tensors = catalog.len(),
            ranges = bases.len(),
            paired = bases.iter().flatten().count(),
            "paired tensors with the base"
        );
        Ok(Some(PreparedBase { source, bases, stamp: base.stamp }))
    }

    /// See [`PackBuilder::add_file_delta`]; `base_member` is `None` for [`PackBuilder::add_file`].
    fn add_from_file(
        &mut self,
        path: &Path,
        member: &str,
        base_member: Option<&str>,
    ) -> Result<MemberReport, Error> {
        self.check_name(member)?;
        if let Some(base_member) = base_member
            && !self.files.contains_key(base_member)
        {
            return Err(Error::MemberNotFound { name: base_member.to_owned() });
        }
        let stamp = InputStamp::of(path)?;
        if same_file(path, &self.output)? {
            return Err(Error::OutputIsInput { path: self.output.clone() });
        }
        let source = FileSource::open(path)?;
        if source.len() != stamp.len {
            return Err(Error::InputChanged { path: path.to_path_buf() });
        }
        tracing::info!(path = %path.display(), member, base_member, len = stamp.len, "adding file");
        let plan = normalize(scan(&source)?, source.len());
        let base = base_member.map(|name| self.prepare_base(name, &plan)).transpose()?.flatten();
        let context = IngestContext {
            compressor: &self.compressor,
            mode: self.mode,
            blobs: &mut self.blobs,
            next_blob: &mut self.next_blob,
            delta: base
                .as_ref()
                .map(|base| DeltaInput { source: &base.source, bases: &base.bases }),
        };
        let ingested = ingest(&source, &plan.ranges, context, &mut self.sink, &mut self.index)?;
        stamp.check()?;
        if let Some(base) = &base {
            self.inputs[base.stamp].check()?;
        }
        let rows = plan.tensors.iter().map(|tensor| TensorRow {
            name: &tensor.name,
            offset: tensor.offset,
            len: tensor.len,
            shape: &tensor.shape,
            dtype: tensor.dtype,
        });
        let parts = MemberParts {
            name: member,
            len: source.len(),
            blake3: ingested.blake3,
            profile: plan.profile,
        };
        let report = MemberReport {
            name: member.to_owned(),
            len: source.len(),
            blake3: ingested.blake3,
            profile: plan.profile,
            tensor_count: 0,
            extent_count: 0,
            blobs_added: ingested.blobs_added,
            blobs_reused: ingested.blobs_reused,
            widen_restreams: ingested.widen_restreams,
            delta_blobs: ingested.delta_blobs,
            compressed_bytes: ingested.compressed_bytes,
            copied_bytes: 0,
            layers: ingested.table.total(),
        };
        let file = FileMember {
            path: path.to_path_buf(),
            stamp: self.inputs.len(),
            extents: ingested.extents.clone(),
        };
        self.inputs.push(stamp);
        let added = self.add_member_records(&parts, &ingested.extents, rows, report, &ingested.table)?;
        self.files.insert(member.to_owned(), file);
        Ok(added)
    }

    /// Copies the member `member` of `src` into this pack under the same name, with no decoding and no
    /// recompression: its stored chunks are copied with their positions rebased. Blobs this pack has
    /// already are reused, by identity or because they came from `src` before. A delta blob brings
    /// its base blob, unless this pack has that base already with the same kind and `chunk_size`.
    ///
    /// The commit decodes copied members like any other, against the hash `src` states.
    ///
    /// # Errors
    /// - [`Error::MemberNotFound`] if `src` has no such member.
    /// - [`Error::UnsafeMemberName`] if a restore would refuse the name.
    /// - [`Error::MalformedIndex`] for `MEMB` if this pack has a member called that.
    /// - [`Error::Io`] if a read or a write fails.
    pub fn add_member_from(&mut self, src: &Container, member: &str) -> Result<MemberReport, Error> {
        self.check_name(member)?;
        let handle = src.member(member)?;
        let record = *handle.record();
        tracing::info!(member, len = record.file_len, "copying member");
        let at = self.copied.iter().position(|known| known.container.same_as(src));
        let at = at.unwrap_or_else(|| {
            self.copied.push(CopiedSource { container: src.clone(), blobs: HashMap::new() });
            self.copied.len() - 1
        });
        let mut target = CopyTarget {
            sink: &mut self.sink,
            index: &mut self.index,
            blobs: &mut self.blobs,
            next_blob: &mut self.next_blob,
            copied: &mut self.copied[at].blobs,
        };
        let copied = copy_member(src, &record, &mut target)?;
        let parts = MemberParts {
            name: member,
            len: record.file_len,
            blake3: record.blake3,
            profile: record.profile,
        };
        let report = MemberReport {
            name: member.to_owned(),
            len: record.file_len,
            blake3: record.blake3,
            profile: record.profile,
            tensor_count: 0,
            extent_count: 0,
            blobs_added: copied.blobs_added,
            blobs_reused: copied.blobs_reused,
            widen_restreams: 0,
            delta_blobs: copied.delta_blobs,
            compressed_bytes: 0,
            copied_bytes: copied.copied_bytes,
            layers: copied.table.total(),
        };
        let index = src.index();
        let mut rows = Vec::with_capacity(usize::try_from(record.tensor_count).unwrap_or(0));
        for tensor in index.member_tensors(&record)? {
            rows.push(TensorRow {
                name: index.string(tensor.name)?,
                offset: tensor.virt_offset,
                len: tensor.virt_len,
                shape: index.tensor_shape(tensor)?,
                dtype: tensor.dtype,
            });
        }
        self.add_member_records(&parts, &copied.extents, rows.into_iter(), report, &copied.table)
    }

    /// Adds the records of a member whose bytes are all in the file, and finishes its report.
    fn add_member_records<'t>(
        &mut self,
        parts: &MemberParts<'_>,
        extents: &[crate::format::ExtentRecord],
        rows: impl Iterator<Item = TensorRow<'t>>,
        report: MemberReport,
        table: &ClassTable,
    ) -> Result<MemberReport, Error> {
        let result = self.write_member_records(parts, extents, rows, report, table);
        if result.is_err() {
            // The index holds a part of a member, which no later member can repair.
            self.sink.poison();
        }
        result
    }

    /// See [`PackBuilder::add_member_records`].
    fn write_member_records<'t>(
        &mut self,
        parts: &MemberParts<'_>,
        extents: &[crate::format::ExtentRecord],
        rows: impl Iterator<Item = TensorRow<'t>>,
        mut report: MemberReport,
        table: &ClassTable,
    ) -> Result<MemberReport, Error> {
        let too_many = |section| Error::MalformedIndex {
            section,
            detail: "too many records for a u32 count".to_owned(),
        };
        let first_extent = self.extent_total;
        for extent in extents {
            self.extent_total = self.index.add_extent(*extent) + 1;
        }
        let extent_count = u32::try_from(extents.len()).map_err(|_| too_many("EXTN"))?;
        let first_tensor = self.tensor_total;
        let mut tensor_count = 0_u32;
        for row in rows {
            let name = self.index.add_string(row.name)?;
            let shape_off = self.index.add_shape(row.shape)?;
            let ndim = u16::try_from(row.shape.len()).map_err(|_| too_many("SHAP"))?;
            self.tensor_total = 1 + self.index.add_tensor(TensorRecord {
                name,
                virt_offset: row.offset,
                virt_len: row.len,
                shape_off,
                ndim,
                dtype: row.dtype,
            })?;
            tensor_count += 1;
        }
        let name = self.index.add_string(parts.name)?;
        self.index.add_member(MemberRecord {
            name,
            file_len: parts.len,
            blake3: parts.blake3,
            first_extent,
            extent_count,
            first_tensor,
            tensor_count,
            profile: parts.profile,
        })?;
        self.names.insert(parts.name.to_owned());
        self.sources.push(SourceHash { name: parts.name.to_owned(), blake3: parts.blake3 });
        self.classes.merge(table);
        report.extent_count = extent_count;
        report.tensor_count = tensor_count;
        self.members.push(report.clone());
        tracing::info!(
            member = parts.name,
            len = parts.len,
            blobs_added = report.blobs_added,
            blobs_reused = report.blobs_reused,
            delta_blobs = report.delta_blobs,
            stored = report.layers.stored,
            "added member"
        );
        Ok(report)
    }

    /// Finishes the container: sets required feature bit 0 in the header if the pack holds a delta
    /// blob, writes the index, decodes every member through the real reader against the hash of its
    /// source, re-checks the inputs, syncs, and renames the file to the output, replacing what is
    /// there. There is no way to skip the decode, spec invariant 12.
    ///
    /// On failure there is no new file at the output and the temporary file is gone, with one
    /// exception: [`Error::CommittedNotDurable`] means the proven file is at the output, but its
    /// directory could not be synced, so a power loss may bring back what was there before.
    ///
    /// # Errors
    /// - [`Error::InputChanged`] if an input differs in length or modification time from when it was
    ///   read, or is gone.
    /// - [`Error::VerifyFailed`], [`Error::ChunkCorrupt`], [`Error::ChecksumMismatch`],
    ///   [`Error::MalformedIndex`], [`Error::BudgetExceeded`] if the file does not prove out. The last
    ///   means a reader with default limits would refuse it, say an index over 256 MiB.
    /// - [`Error::Io`], [`Error::Zstd`] if a file operation or zstd fails.
    /// - [`Error::CommittedNotDurable`] if the rename happened and the directory sync after it failed,
    ///   on Unix. The report is lost; the file at the output is complete and proven.
    pub fn finish(self) -> Result<EncodeReport, Error> {
        let Self {
            output,
            threads,
            mut sink,
            index,
            compressor,
            blobs,
            inputs,
            files,
            sources,
            copied,
            members,
            mut classes,
            #[cfg(test)]
            fault,
            ..
        } = self;
        sink.check()?;
        // Bit 0 is set exactly when a kind 1 blob exists, spec section 7a, and a blob once recorded is
        // in the index, referenced or not.
        let delta_blobs = blobs.delta_count();
        // Their memory must not stack on the reader's bound during verification.
        drop(compressor);
        drop(blobs);
        drop(files);
        drop(copied);
        check_inputs(&inputs)?;
        if delta_blobs > 0 {
            let header = Header { required_features: FEATURE_DELTA, ..Header::current() };
            sink.patch_header(&header)?;
        }
        let index_stored_len = write_index(&mut sink, &index)?;
        drop(index);
        #[cfg(test)]
        if let Some(fault) = fault {
            fault(sink.path());
        }
        let (verify, unreferenced) = verify_output(sink.path(), threads, &sources)?;
        // Blobs of failed adds are in the data region too.
        classes.merge(&unreferenced);
        check_inputs(&inputs)?;
        let output_len = sink.position();
        sink.commit(&output)?;
        tracing::info!(
            output = %output.display(),
            output_len,
            members = members.len(),
            delta_blobs,
            verify_bytes = verify.bytes,
            verify_elapsed = ?verify.elapsed,
            "committed container"
        );
        Ok(EncodeReport {
            output_len,
            index_stored_len,
            total: classes.total(),
            classes: classes.classes(),
            members,
            verify,
        })
    }
}

/// Compares every input with its stamp.
fn check_inputs(inputs: &[InputStamp]) -> Result<(), Error> {
    inputs.iter().try_for_each(InputStamp::check)
}

/// Encodes the file at `input` as a container of one member at `output`, named after the file.
///
/// # Errors
/// [`Error::UnsafeMemberName`] if the file name is not valid UTF-8 or a restore would refuse it, and
/// every error of [`PackBuilder::new`], [`PackBuilder::add_file`] and [`PackBuilder::finish`].
pub fn encode_file(
    input: &Path,
    output: &Path,
    options: &EncodeOptions,
) -> Result<EncodeReport, Error> {
    let name = input.file_name().and_then(OsStr::to_str);
    let name = name.ok_or_else(|| Error::UnsafeMemberName { name: input.display().to_string() })?;
    let mut builder = PackBuilder::new(output, options)?;
    builder.add_file(input, name)?;
    builder.finish()
}
