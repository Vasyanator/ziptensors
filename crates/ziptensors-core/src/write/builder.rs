//! File: write/builder.rs
//! Purpose: the writer's public API: [`encode_file`] and [`PackBuilder`].
//!
//! Main responsibilities: checks on what is added, the records of each member, the checks of inputs
//! by kind ([`SourceStamp`]: a file is looked at again, a source that is no file is compared once with
//! what was read), the base of an input stored with delta blobs ([`DeltaSpec`]), and the order of the
//! commit by protocol ([`Commit`]): replace (header features, index, verify, re-check inputs, sync,
//! rename) or append (data after an existing container, verify through a splice view, index, sync,
//! footer, sync) or in place (header, verify through a splice view, index, footer, one cut). A builder
//! over an existing container ([`Seed`]) carries that container's storage records into its index and
//! its members over them with no byte copied, for the append and in-place strategies of the `edit`
//! module: all records with their ids for an append, the kept blobs renumbered in order over their
//! moved chunks for an in-place rewrite. A name the pack introduces must be one a restore accepts; a member an edit keeps
//! keeps its source name ([`NameRule`]).
//!
//! Key structures: [`EncodeOptions`], [`PackBuilder`], [`SourceStamp`], [`DeltaSpec`], [`Seed`],
//! [`Commit`], [`NameRule`].
//!
//! Key functions: [`encode_file`].
//!
//! Implementation notes: a file that fails to add leaves the pack usable, with at most unreferenced
//! complete blobs, because a member's records are added only after all its bytes are in. A failed
//! write leaves the sink broken, and everything after it fails. Compression state is dropped before
//! verification, so the reader's own memory bound does not stack on the writer's. Symbolic links are
//! refused here and not skipped: skipping is a decision of a frontend, spec section 2. Every input goes
//! through the crate-private `PackBuilder::add_source`, a file by way of `add_file`, which adds the file
//! checks first. Of each member added from an input the builder keeps its stamp, its extents and where
//! its bytes are (a file's path, reopened when needed, or the source itself), so that a later input can
//! name it as a delta base; its tensor table is scanned again from those bytes when that happens,
//! rather than kept for every member. Every failure after the sink exists ends through
//! `Sink::fail`, so that an append that cannot be cut back reports `Error::TrailingBytes`, and an
//! in-place rewrite that began writing `Error::DamagedInPlace`.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::io::ErrorKind;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::SystemTime;

use crate::chain::{BLOB_KIND_DELTA, FEATURE_DELTA};
use crate::error::Error;
use crate::format::{
    BlobRecord, ChunkRecord, ExtentRecord, FOOTER_LEN, GENERATION_INITIAL, Header, Index, IndexBuilder,
    MemberRecord, TensorRecord, check_restore_name,
};
use crate::io::{FileSource, RandomAccess, SpliceSource, same_file};
use crate::profile::{normalize, scan};
use crate::read::Container;
use crate::types::{Dtype, Mode, Profile, RangePlan};

use super::commit::{Ending, Sink, SourceHash, StoredIndex, store_index, verify_output, write_index};
use super::dedup::{BlobShape, BlobTable};
use super::ingest::{Compressor, DeltaInput, IngestContext, ingest};
use super::merge::{CopiedSource, CopyTarget, copy_member};
use super::pairing::{BaseCatalog, BaseRange};
use super::policy::tries_delta;
use super::report::{ClassTable, EncodeReport, MemberReport, VerifyReport, class_of};

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

/// What a file looked like before it was read, to compare after the read and at commit.
#[derive(Debug)]
pub(crate) struct InputStamp {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
}

impl InputStamp {
    /// Takes the stamp of `path`, which must not be a symbolic link.
    ///
    /// # Errors
    /// [`Error::SymlinkInput`] if `path` is a symbolic link, [`Error::Io`] if it cannot be looked at.
    pub(crate) fn of(path: &Path) -> Result<Self, Error> {
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

/// What an input looked like before it was read, by kind of input. The builder checks it right after
/// the read, again at commit, and before the input is read again as a delta base.
#[derive(Debug)]
pub(crate) enum SourceStamp {
    /// A file, looked at again on the file system at every check.
    File(InputStamp),
    /// A source that is no file, such as a member of an open container: its length and the BLAKE3
    /// of its content, as its owner states them. Its bytes cannot change under the builder (a
    /// container's bytes are fixed while it is open), so the one check compares what the read
    /// produced with these values; at commit there is nothing to look at again, and the commit's
    /// decode proves the pack against what was read.
    Member {
        /// Length in bytes.
        len: u64,
        /// BLAKE3 of the content.
        blake3: [u8; 32],
    },
}

impl SourceStamp {
    /// Checks the input after it was read whole as `member`: `read_len` bytes hashing to `read_blake3`.
    ///
    /// # Errors
    /// For a file, every error of `InputStamp::check`. For a source that is no file,
    /// [`Error::VerifyFailed`] naming `member` if the length or the hash differs from the stamp: the
    /// source does not hold what its owner states.
    fn check_read(&self, member: &str, read_len: u64, read_blake3: &[u8; 32]) -> Result<(), Error> {
        match self {
            Self::File(stamp) => stamp.check(),
            Self::Member { len, blake3 } => {
                if *len == read_len && blake3 == read_blake3 {
                    return Ok(());
                }
                Err(Error::VerifyFailed {
                    member: member.to_owned(),
                    expected: blake3::Hash::from(*blake3),
                    actual: blake3::Hash::from(*read_blake3),
                })
            }
        }
    }

    /// Checks the input at commit, and before it is read again as a delta base.
    ///
    /// # Errors
    /// For a file, every error of `InputStamp::check`. A source that is no file cannot fail here.
    fn check(&self) -> Result<(), Error> {
        match self {
            Self::File(stamp) => stamp.check(),
            // Fixed bytes: checked once against what was read, in `check_read`.
            Self::Member { .. } => Ok(()),
        }
    }
}

/// The base an input is stored against with delta blobs, spec section 7a.
pub(crate) enum DeltaSpec<'a> {
    /// A member this builder added, from a file or from a source, by name.
    Added(&'a str),
    /// A base that is no member of this builder, such as a kept member of the container an edit
    /// appends to. It has no stamp: the commit's decode proves every member anyway.
    Member {
        /// Its member name, for messages.
        name: &'a str,
        /// Its bytes, at the offsets of `catalog`.
        source: Arc<dyn RandomAccess>,
        /// Its tensors mapped onto blobs of this pack, see [`BaseCatalog::from_member`].
        catalog: &'a BaseCatalog,
    },
}

impl DeltaSpec<'_> {
    /// The base's member name.
    fn name(&self) -> &str {
        match self {
            Self::Added(name) | Self::Member { name, .. } => name,
        }
    }
}

impl fmt::Debug for DeltaSpec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Added(name) => f.debug_tuple("Added").field(name).finish(),
            Self::Member { name, source, catalog } => f
                .debug_struct("Member")
                .field("name", name)
                .field("source_len", &source.len())
                .field("base_tensors", &catalog.len())
                .finish(),
        }
    }
}

/// Where the bytes of an added member are read again when a later input names it as its base.
enum AddedSource {
    /// A file, reopened by path, so that the builder holds no open handle per member.
    File(PathBuf),
    /// A source that is no file, kept as it was given until [`PackBuilder::finish`] drops it.
    Shared(Arc<dyn RandomAccess>),
}

impl fmt::Debug for AddedSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
            Self::Shared(source) => f.debug_struct("Shared").field("len", &source.len()).finish(),
        }
    }
}

/// A member added from a file or a source, kept so that a later input can name it as its delta base.
#[derive(Debug)]
struct AddedMember {
    /// Where its bytes are.
    source: AddedSource,
    /// Its stamp in [`PackBuilder::inputs`]. Always in range: the stamp is pushed before the member
    /// is recorded here, and stamps are never removed.
    stamp: usize,
    /// The member's extents, with blob ids of this pack.
    extents: Vec<ExtentRecord>,
}

/// The base of an input being added: its bytes and the base range of each typed range.
struct PreparedBase {
    source: Arc<dyn RandomAccess>,
    bases: Vec<Option<BaseRange>>,
    /// The base's stamp in [`PackBuilder::inputs`]; `None` for a base that is no member of this
    /// builder, which has none.
    stamp: Option<usize>,
}

impl fmt::Debug for PreparedBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedBase")
            .field("source_len", &self.source.len())
            .field("bases", &self.bases)
            .field("stamp", &self.stamp)
            .finish()
    }
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

/// Which rules a member name must pass besides being unique in the pack (spec invariant 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameRule {
    /// A name the pack introduces: a restore must accept it ([`check_restore_name`], spec
    /// invariant 11), so that no pack this crate writes needs a refusal at restore.
    New,
    /// The name of a member of a valid container that the result keeps under it. Invariant 11
    /// binds a restore, not a container, so a kept name is not held to it: an edit that keeps a
    /// legal member such as `../x` restores nothing.
    Kept,
}

/// How [`PackBuilder::finish_with`] makes the result visible, spec invariant 12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Commit {
    /// Protocol (a): the temporary file of [`PackBuilder::new`] is renamed over the output.
    Replace,
    /// Protocol (b): the builder writes after an existing container, through [`Sink::append_to`]
    /// and with that container's [`Seed`]; the new generation is verified before its index is on
    /// disk.
    Append,
    /// In-place rewrite (outside spec invariant 12, which it gives up for space): the builder
    /// writes after the moved data of an existing container, through [`Sink::in_place`] and with
    /// [`Seed::in_place`]. The header is rewritten, the result verified before its index is on
    /// disk, then index, footer and one final cut ([`Ending::InPlace`]).
    InPlace,
}

/// The storage records of an existing container, carried into the index of a builder that writes a
/// new generation over it: for an append (spec invariant 12 (b), [`Seed::of`]) every `CHST`, `CHAN`,
/// `BLOB`, `CHNK` and `PARM` record with its id and its file offset unchanged; for an in-place
/// rewrite ([`Seed::in_place`]) the kept blobs only, renumbered in order, over their moved chunks.
/// Every carried blob is in [`BlobTable`] with its identity and shape, so that new inputs
/// deduplicate against it and name its blobs as delta bases.
#[derive(Debug)]
pub(crate) struct Seed {
    index: IndexBuilder,
    blobs: BlobTable,
    next_blob: u32,
    carried: Carried,
}

/// What a seeded builder keeps of the container it extends.
#[derive(Debug)]
struct Carried {
    /// The container, whose members [`PackBuilder::carry_member`] carries over. Only its index is
    /// read: under [`Seed::in_place`] its chunk offsets are stale once data moved.
    container: Container,
    /// Its header, which an append keeps: it decides whether a delta blob may be added.
    header: Header,
    /// Per blob id of the container: whether a carried member reached it already, so that its
    /// stored bytes count once in the report, as `layer_report` counts them.
    reached: Vec<bool>,
    /// Per blob id of the container, its id in this pack, `None` for a blob it dropped; `None` as
    /// a whole when every blob keeps its id ([`Seed::of`]).
    ids: Option<Vec<Option<u32>>>,
}

impl Carried {
    /// The extents of a carried member, `extents` of the container, with blob ids of this pack.
    ///
    /// # Errors
    /// [`Error::MalformedIndex`] for `EXTN` if an extent reaches a blob the seed dropped, which
    /// means the caller kept a member without the blobs it needs.
    fn extents(&self, extents: &[ExtentRecord]) -> Result<Vec<ExtentRecord>, Error> {
        let Some(ids) = &self.ids else {
            return Ok(extents.to_vec());
        };
        extents
            .iter()
            .map(|extent| {
                let id = usize::try_from(extent.blob_id).ok().and_then(|at| ids.get(at));
                let id = id.copied().flatten().ok_or_else(|| Error::MalformedIndex {
                    section: "EXTN",
                    detail: format!("blob {} of a carried member was dropped", extent.blob_id),
                })?;
                Ok(ExtentRecord { blob_id: id, ..*extent })
            })
            .collect()
    }
}

impl Seed {
    /// The seed of `container`, whose file starts with `header`. Blob ids, chunk ids and chunk
    /// offsets of the new index are those of the container.
    ///
    /// # Errors
    /// [`Error::MalformedIndex`] if a record of the container cannot be carried with its id, which
    /// validation at open rules out.
    pub(crate) fn of(container: &Container, header: Header) -> Result<Self, Error> {
        let index = container.index();
        // Offsets stay: an append writes nothing below the end of the container.
        let builder = IndexBuilder::seed(index, Ok)?;
        let mut blobs = BlobTable::default();
        let mut next_blob = 0_u32;
        for blob in index.blobs() {
            let shape =
                BlobShape { kind: blob.kind, chunk_size: blob.chunk_size, raw_len: blob.raw_len };
            blobs.record(next_blob, blob.blake3, shape)?;
            next_blob = next_blob.checked_add(1).ok_or_else(|| Error::MalformedIndex {
                section: "BLOB",
                detail: "too many records for a u32 id".to_owned(),
            })?;
        }
        tracing::debug!(blobs = next_blob, chunks = index.chunks().len(), "seeded from a container");
        let reached = vec![false; index.blobs().len()];
        let carried = Carried { container: container.clone(), header, reached, ids: None };
        Ok(Self { index: builder, blobs, next_blob, carried })
    }

    /// The seed of `container`, whose file starts with `header`, for an in-place rewrite that keeps
    /// only the blobs `kept` (ids in increasing order; every blob a kept member reaches, and the
    /// base of every kept delta blob) and moves their chunks: [`IndexBuilder::seed_subset`] with
    /// `rebase` giving each chunk record's new offset. Kept blobs get ids `0..kept.len()` in their
    /// old order, so a base stays below its delta blob; members carried over get their extents
    /// renumbered to match.
    ///
    /// # Errors
    /// Any error of `rebase` and of [`IndexBuilder::seed_subset`].
    pub(crate) fn in_place(
        container: &Container,
        header: Header,
        kept: &[u32],
        rebase: impl FnMut(&ChunkRecord) -> Result<u64, Error>,
    ) -> Result<Self, Error> {
        let index = container.index();
        let (builder, ids) = IndexBuilder::seed_subset(index, kept, rebase)?;
        let mut blobs = BlobTable::default();
        let mut next_blob = 0_u32;
        for &old in kept {
            // `seed_subset` resolved every id of `kept`.
            let blob = usize::try_from(old).ok().and_then(|at| index.blobs().get(at));
            let blob = blob.ok_or_else(|| Error::MalformedIndex {
                section: "BLOB",
                detail: format!("kept blob {old} does not exist"),
            })?;
            let shape =
                BlobShape { kind: blob.kind, chunk_size: blob.chunk_size, raw_len: blob.raw_len };
            blobs.record(next_blob, blob.blake3, shape)?;
            next_blob = next_blob.checked_add(1).ok_or_else(|| Error::MalformedIndex {
                section: "BLOB",
                detail: "too many records for a u32 id".to_owned(),
            })?;
        }
        tracing::debug!(
            blobs = next_blob,
            dropped = index.blobs().len() - kept.len(),
            "seeded from a container to rewrite in place"
        );
        let reached = vec![false; index.blobs().len()];
        let carried = Carried { container: container.clone(), header, reached, ids: Some(ids) };
        Ok(Self { index: builder, blobs, next_blob, carried })
    }
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
    inputs: Vec<SourceStamp>,
    /// Members added from files or sources, by name, for a later input to name as its base.
    added: HashMap<String, AddedMember>,
    sources: Vec<SourceHash>,
    copied: Vec<CopiedSource>,
    members: Vec<MemberReport>,
    classes: ClassTable,
    /// The container this builder extends, for a builder made with a [`Seed`].
    carried: Option<Carried>,
    /// Test only: runs on the written file just before verification.
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
        let sink = Sink::create(output)?;
        Self::with_sink(sink, None, options, output)
    }

    /// A builder over `sink`, which must be fresh: a temporary file from [`Sink::create`] with no
    /// `seed`, or a container opened by [`Sink::append_to`] with the `seed` of that container as it
    /// was opened. `output` is the path inputs are checked against ([`Error::OutputIsInput`]) and that
    /// [`Commit::Replace`] renames to. Starts the compression threads.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if the threads cannot be created, and [`Error::Zstd`] if a zstd context
    /// cannot be. `sink` is dropped then, which removes a temporary file and cuts an append back.
    pub(crate) fn with_sink(
        sink: Sink,
        seed: Option<Seed>,
        options: &EncodeOptions,
        output: &Path,
    ) -> Result<Self, Error> {
        let threads = options.thread_count();
        let mode = options.mode;
        let seeded = seed.is_some();
        tracing::info!(output = %output.display(), ?mode, threads, seeded, "starting container");
        let (index, blobs, next_blob, carried) = match seed {
            Some(seed) => (seed.index, seed.blobs, seed.next_blob, Some(seed.carried)),
            None => (IndexBuilder::new(), BlobTable::default(), 0, None),
        };
        Ok(Self {
            output: output.to_path_buf(),
            mode,
            threads,
            sink,
            index,
            compressor: Compressor::new(threads, output)?,
            blobs,
            next_blob,
            extent_total: 0,
            tensor_total: 0,
            names: HashSet::new(),
            inputs: Vec::new(),
            added: HashMap::new(),
            sources: Vec::new(),
            copied: Vec::new(),
            members: Vec::new(),
            classes: ClassTable::default(),
            carried,
            #[cfg(test)]
            fault: None,
        })
    }

    /// Arms the corruption seam: `fault` runs on the written file just before verification.
    #[cfg(test)]
    pub(crate) fn set_fault(&mut self, fault: fn(&Path)) {
        self.fault = Some(fault);
    }

    /// Checks a member name under `rule`, after the sink: the rules of a restore for a
    /// [`NameRule::New`] name, and unique in this pack for every name.
    fn check_name(&mut self, member: &str, rule: NameRule) -> Result<(), Error> {
        self.sink.check()?;
        match rule {
            NameRule::New => check_restore_name(member)?,
            NameRule::Kept => {}
        }
        if self.names.contains(member) {
            return Err(Error::DuplicateMember { name: member.to_owned() });
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
    /// - [`Error::DuplicateMember`] if this pack has a member called that.
    /// - [`Error::OutputIsInput`] if `path` is the output of this builder.
    /// - [`Error::SymlinkInput`] if `path` is a symbolic link. It is not followed, and a frontend
    ///   decides whether to skip it.
    /// - [`Error::Io`] for any failed file operation.
    /// - [`Error::InputChanged`] if the file changed while it was read. This compares length and
    ///   modification time, so an edit that keeps both commits the bytes that were read.
    /// - [`Error::Zstd`] if compression fails.
    pub fn add_file(&mut self, path: &Path, member: &str) -> Result<MemberReport, Error> {
        self.add_from_file(path, member, NameRule::New, None)
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
    /// - [`Error::MemberNotFound`] if this builder added no member called `base_member` from a file
    ///   (or, inside the crate, through `add_source`).
    /// - [`Error::InputChanged`] if the base file differs from when it was added, or changes during
    ///   the read.
    pub fn add_file_delta(
        &mut self,
        path: &Path,
        member: &str,
        base_member: &str,
    ) -> Result<MemberReport, Error> {
        self.add_from_file(path, member, NameRule::New, Some(&DeltaSpec::Added(base_member)))
    }

    /// Checks what every add checks before it looks at its input: the name under `rule`, and that a
    /// base named by `delta` is a member this builder added.
    fn check_add(
        &mut self,
        member: &str,
        rule: NameRule,
        delta: Option<&DeltaSpec<'_>>,
    ) -> Result<(), Error> {
        self.check_name(member, rule)?;
        if let Some(DeltaSpec::Added(base_member)) = delta
            && !self.added.contains_key(*base_member)
        {
            return Err(Error::MemberNotFound { name: (*base_member).to_owned() });
        }
        Ok(())
    }

    /// Pairs the typed ranges of `plan` with the tensors of the base `delta` names, reopening its
    /// file if it was added from one. `None` if the mode stores no delta blobs.
    fn prepare_base(
        &self,
        delta: &DeltaSpec<'_>,
        plan: &RangePlan,
    ) -> Result<Option<PreparedBase>, Error> {
        if let DeltaSpec::Added(base_member) = delta
            && !self.added.contains_key(*base_member)
        {
            return Err(Error::MemberNotFound { name: (*base_member).to_owned() });
        }
        if !tries_delta(self.mode) {
            let base_member = delta.name();
            tracing::debug!(base_member, mode = ?self.mode, "this mode stores no delta blobs");
            return Ok(None);
        }
        let (source, bases, base_tensors, stamp) = match delta {
            DeltaSpec::Added(base_member) => {
                let base = self
                    .added
                    .get(*base_member)
                    .ok_or_else(|| Error::MemberNotFound { name: (*base_member).to_owned() })?;
                // The extents of the base describe the bytes it had when it was added.
                let stamp = &self.inputs[base.stamp];
                stamp.check()?;
                let source: Arc<dyn RandomAccess> = match &base.source {
                    AddedSource::File(path) => {
                        let source = FileSource::open(path)?;
                        if let SourceStamp::File(file) = stamp
                            && source.len() != file.len
                        {
                            return Err(Error::InputChanged { path: path.clone() });
                        }
                        Arc::new(source)
                    }
                    AddedSource::Shared(source) => Arc::clone(source),
                };
                let base_plan = normalize(scan(&*source)?, source.len());
                let catalog = BaseCatalog::from_plan(&base_plan, &base.extents);
                (source, catalog.bases_of(plan), catalog.len(), Some(base.stamp))
            }
            DeltaSpec::Member { source, catalog, .. } => {
                (Arc::clone(source), catalog.bases_of(plan), catalog.len(), None)
            }
        };
        tracing::info!(
            base_member = delta.name(),
            base_tensors,
            ranges = bases.len(),
            paired = bases.iter().flatten().count(),
            "paired tensors with the base"
        );
        Ok(Some(PreparedBase { source, bases, stamp }))
    }

    /// [`PackBuilder::add_file`] with an optional base of any kind: [`PackBuilder::add_file_delta`]
    /// for [`DeltaSpec::Added`], `None` for [`PackBuilder::add_file`]. `rule` is
    /// [`NameRule::Kept`] only for a file that takes the place of a member of a container under that
    /// member's name (an edit's `Replace`).
    ///
    /// # Errors
    /// Every error of [`PackBuilder::add_file_delta`]; [`Error::UnsafeMemberName`] only under
    /// [`NameRule::New`].
    pub(crate) fn add_from_file(
        &mut self,
        path: &Path,
        member: &str,
        rule: NameRule,
        delta: Option<&DeltaSpec<'_>>,
    ) -> Result<MemberReport, Error> {
        // Before the file is looked at, so that a bad name or base is reported first, whatever the
        // file; `add_source` repeats the structural part of this check, which changes nothing.
        self.check_add(member, rule, delta)?;
        let stamp = InputStamp::of(path)?;
        if same_file(path, &self.output)? {
            return Err(Error::OutputIsInput { path: self.output.clone() });
        }
        let source = FileSource::open(path)?;
        self.add_source(Arc::new(source), member, SourceStamp::File(stamp), delta)
    }

    /// Adds the bytes of `source` as the member `member`, as [`PackBuilder::add_file`] does with a
    /// file, and with `delta` as [`PackBuilder::add_file_delta`] does with a base. `stamp` says what
    /// the input looked like before it was read (see [`SourceStamp`] for what each kind checks and
    /// when). A member added here can serve as the base of a later input; a source that is no file is
    /// then kept until [`PackBuilder::finish`].
    ///
    /// `source` must be the input `stamp` describes: for a file, the file at the stamp's path, opened
    /// after the stamp was taken. No symbolic-link or output check is made here; a file comes through
    /// [`PackBuilder::add_file`], which makes them. `member` is checked as a [`NameRule::Kept`] name
    /// (unique only): a new name comes through `add_from_file`, which checks it against the restore
    /// rules first, and a member of a container re-encoded by a repack keeps the name it has.
    ///
    /// # Errors
    /// Every error of [`PackBuilder::add_file_delta`] except the file checks above and
    /// [`Error::UnsafeMemberName`], any error a read of `source` returns, and
    /// [`Error::VerifyFailed`] if a source that is no file does not hold the length and hash of its
    /// stamp.
    pub(crate) fn add_source(
        &mut self,
        source: Arc<dyn RandomAccess>,
        member: &str,
        stamp: SourceStamp,
        delta: Option<&DeltaSpec<'_>>,
    ) -> Result<MemberReport, Error> {
        self.check_add(member, NameRule::Kept, delta)?;
        let len = source.len();
        if let SourceStamp::File(file) = &stamp
            && len != file.len
        {
            return Err(Error::InputChanged { path: file.path.clone() });
        }
        // Names the input in messages: a source that is no file goes by its member name.
        let label = match &stamp {
            SourceStamp::File(file) => file.path.clone(),
            SourceStamp::Member { .. } => PathBuf::from(member),
        };
        let base_member = delta.map(DeltaSpec::name);
        tracing::info!(path = %label.display(), member, base_member, len, "adding file");
        let plan = normalize(scan(&*source)?, len);
        let base = delta.map(|delta| self.prepare_base(delta, &plan)).transpose()?.flatten();
        let context = IngestContext {
            compressor: &self.compressor,
            mode: self.mode,
            blobs: &mut self.blobs,
            next_blob: &mut self.next_blob,
            delta: base
                .as_ref()
                .map(|base| DeltaInput { source: &*base.source, bases: &base.bases }),
        };
        let ingested =
            ingest(&*source, &label, &plan.ranges, context, &mut self.sink, &mut self.index)?;
        stamp.check_read(member, len, &ingested.blake3)?;
        if let Some(base_stamp) = base.as_ref().and_then(|base| base.stamp) {
            self.inputs[base_stamp].check()?;
        }
        let rows = plan.tensors.iter().map(|tensor| TensorRow {
            name: &tensor.name,
            offset: tensor.offset,
            len: tensor.len,
            shape: &tensor.shape,
            dtype: tensor.dtype,
        });
        let parts = MemberParts { name: member, len, blake3: ingested.blake3, profile: plan.profile };
        let report = MemberReport {
            name: member.to_owned(),
            len,
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
        let added_source = match &stamp {
            // The file is reopened if it is ever named as a base; the handle closes here.
            SourceStamp::File(file) => AddedSource::File(file.path.clone()),
            SourceStamp::Member { .. } => AddedSource::Shared(source),
        };
        let added = AddedMember {
            source: added_source,
            stamp: self.inputs.len(),
            extents: ingested.extents.clone(),
        };
        self.inputs.push(stamp);
        let report = self.add_member_records(&parts, &ingested.extents, rows, report, &ingested.table)?;
        self.added.insert(member.to_owned(), added);
        Ok(report)
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
    /// - [`Error::DuplicateMember`] if this pack has a member called that.
    /// - [`Error::Io`] if a read or a write fails.
    pub fn add_member_from(&mut self, src: &Container, member: &str) -> Result<MemberReport, Error> {
        self.copy_member_from(src, member, NameRule::New, None)
    }

    /// [`PackBuilder::add_member_from`] for a member that an edit of `src` keeps: its name is a
    /// [`NameRule::Kept`] name, so a restore need not accept it. With `base_source`, the copied
    /// member can then serve as the base of a later input by its name ([`DeltaSpec::Added`]), read
    /// from `base_source`, which must hold its bytes (a view of the member itself) and is kept until
    /// [`PackBuilder::finish_with`] drops it.
    ///
    /// # Errors
    /// Every error of [`PackBuilder::add_member_from`] but [`Error::UnsafeMemberName`].
    pub(crate) fn keep_member_from(
        &mut self,
        src: &Container,
        member: &str,
        base_source: Option<Arc<dyn RandomAccess>>,
    ) -> Result<MemberReport, Error> {
        self.copy_member_from(src, member, NameRule::Kept, base_source)
    }

    /// See [`PackBuilder::keep_member_from`]; `base_source` is `None` for a plain copy.
    fn copy_member_from(
        &mut self,
        src: &Container,
        member: &str,
        rule: NameRule,
        base_source: Option<Arc<dyn RandomAccess>>,
    ) -> Result<MemberReport, Error> {
        self.check_name(member, rule)?;
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
        let report =
            self.add_member_records(&parts, &copied.extents, rows.into_iter(), report, &copied.table)?;
        if let Some(source) = base_source {
            // A container's bytes are fixed while it is open, so this stamp never fails a check;
            // the commit's decode proves whatever is stored against the base.
            let added = AddedMember {
                source: AddedSource::Shared(source),
                stamp: self.inputs.len(),
                extents: copied.extents,
            };
            self.inputs.push(SourceStamp::Member { len: record.file_len, blake3: record.blake3 });
            self.added.insert(member.to_owned(), added);
        }
        Ok(report)
    }

    /// Carries the member `member` of the container this builder extends into the new index: its
    /// records, over the blobs that are already in the file, with no byte read or written. Only for a
    /// builder made with a [`Seed`]. The commit decodes it like any other member.
    ///
    /// # Errors
    /// - [`Error::EditRefused`] if the builder has no seed.
    /// - [`Error::MemberNotFound`] if the container has no such member.
    /// - [`Error::DuplicateMember`] as [`PackBuilder::add_member_from`]. The name is a
    ///   [`NameRule::Kept`] name: a restore need not accept it.
    pub(crate) fn carry_member(&mut self, member: &str) -> Result<MemberReport, Error> {
        self.check_name(member, NameRule::Kept)?;
        let carried = self.carried.as_mut().ok_or_else(|| Error::EditRefused {
            detail: "only a builder over an existing container carries its members".to_owned(),
        })?;
        // A second handle, so that the records below can be borrowed while the pack changes.
        let container = carried.container.clone();
        let index = container.index();
        let record = *container.member(member)?.record();
        let source_extents = index.member_extents(&record)?;
        let extents = carried.extents(source_extents)?;
        let table = carried_table(index, source_extents, &mut carried.reached)?;
        tracing::info!(member, len = record.file_len, extents = extents.len(), "carrying member");
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
            blobs_added: 0,
            blobs_reused: record.extent_count,
            widen_restreams: 0,
            delta_blobs: 0,
            compressed_bytes: 0,
            copied_bytes: 0,
            layers: table.total(),
        };
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
        self.add_member_records(&parts, &extents, rows.into_iter(), report, &table)
    }

    /// One move of an in-place rewrite, before any member is added: [`Sink::copy_within`] of `len`
    /// bytes from `src` to `dst` through `buffer`. For a builder over [`Sink::in_place`] with
    /// [`Seed::in_place`], whose chunk offsets already name the destinations.
    ///
    /// # Errors
    /// Every error of [`Sink::copy_within`]; the builder is then unusable, and the caller ends it
    /// with [`PackBuilder::fail`].
    pub(crate) fn move_data(
        &mut self,
        src: u64,
        dst: u64,
        len: u64,
        buffer: &mut [u8],
    ) -> Result<(), Error> {
        self.sink.copy_within(src, dst, len, buffer)
    }

    /// Where the next byte of the pack lands.
    pub(crate) fn position(&self) -> u64 {
        self.sink.position()
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
    /// there. There is no way to skip the decode, spec invariant 12 (a).
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
        self.finish_with(Commit::Replace, GENERATION_INITIAL)
    }

    /// Finishes the container by protocol `commit`, with `generation` in its footer. The decode of
    /// every member against its source hash precedes visibility under every protocol.
    ///
    /// - [`Commit::Replace`]: as [`PackBuilder::finish`].
    /// - [`Commit::Append`]: for a builder over [`Sink::append_to`] with a [`Seed`]. The header stays;
    ///   the new generation is verified through [`SpliceSource`] (the file up to the new data, then the
    ///   stored index and footer in memory) before any byte of its index is written; then index, sync,
    ///   footer, sync ([`Ending::AppendFooter`]). Any failure cuts the file back to the container it
    ///   extends, byte for byte.
    /// - [`Commit::InPlace`]: for a builder over [`Sink::in_place`] with [`Seed::in_place`], after
    ///   the moves. The header is rewritten with bit 0 as the pack's blobs say; the result is
    ///   verified through [`SpliceSource`] as for an append; then index and footer, sync, one cut to
    ///   the end of the footer, sync ([`Ending::InPlace`]). The file is left as it is on failure.
    ///
    /// # Errors
    /// Every error of [`PackBuilder::finish`], and:
    /// - [`Error::EditRefused`] for an append whose pack holds a delta blob while the header it keeps
    ///   has required bit 0 clear, or that has no seed; the edit refuses that before any write.
    /// - [`Error::TrailingBytes`] if an append failed and could not be cut back: the previous
    ///   container is intact in the first `valid_len` bytes. Or if an in-place rewrite could not be
    ///   cut to its end: the new container is complete and durable in the first `valid_len` bytes.
    /// - [`Error::DamagedInPlace`] for any other failure of an in-place rewrite that had begun to
    ///   write.
    pub(crate) fn finish_with(self, commit: Commit, generation: u64) -> Result<EncodeReport, Error> {
        let Self {
            output,
            threads,
            mut sink,
            index,
            compressor,
            blobs,
            inputs,
            added,
            sources,
            copied,
            members,
            mut classes,
            carried,
            #[cfg(test)]
            fault,
            ..
        } = self;
        // Bit 0 is set exactly when a kind 1 blob exists, spec section 7a, and a blob once recorded is
        // in the index, referenced or not.
        let delta_blobs = blobs.delta_count();
        // Their memory must not stack on the reader's bound during verification. Sources of added
        // members and copied or extended containers also hold handles, which must not outlive the
        // rename.
        drop(compressor);
        drop(blobs);
        drop(added);
        drop(copied);
        let kept_header = carried.map(|carried| carried.header);
        let staged = StagedCommit {
            commit,
            generation,
            threads,
            delta_blobs,
            kept_header,
            sources: &sources,
            inputs: &inputs,
            #[cfg(test)]
            fault,
        };
        let staged = match staged.run(&mut sink, index) {
            Ok(staged) => staged,
            Err(error) => return Err(sink.fail(error)),
        };
        // Blobs of failed adds, and under an append the blobs of members it left out, are in the data
        // region too.
        classes.merge(&staged.unreferenced);
        let output_len = match staged.ending {
            StagedEnding::Rename => {
                let output_len = sink.position();
                sink.commit(&output)?;
                output_len
            }
            StagedEnding::AppendFooter(footer) => {
                let output_len = sink.position() + FOOTER_LEN;
                sink.finish(Ending::AppendFooter { footer })?;
                output_len
            }
            StagedEnding::InPlace => {
                let final_len = sink.position();
                sink.finish(Ending::InPlace { final_len })?;
                final_len
            }
        };
        tracing::info!(
            output = %output.display(),
            output_len,
            ?commit,
            generation,
            members = members.len(),
            delta_blobs,
            verify_bytes = staged.verify.bytes,
            verify_elapsed = ?staged.verify.elapsed,
            "committed container"
        );
        Ok(EncodeReport {
            output_len,
            index_stored_len: staged.index_stored_len,
            total: classes.total(),
            classes: classes.classes(),
            members,
            verify: staged.verify,
        })
    }

    /// Gives up on the pack after `error`, for a caller whose own work failed between adds:
    /// [`Sink::fail`]. A temporary file is removed; an append is cut back to the container it
    /// extends; an in-place rewrite is left as it is. Returns what the caller must report: `error`,
    /// [`Error::TrailingBytes`] if an append could not be cut back, or [`Error::DamagedInPlace`]
    /// wrapping `error` if an in-place rewrite had begun writing.
    pub(crate) fn fail(self, error: Error) -> Error {
        let Self { sink, .. } = self;
        sink.fail(error)
    }
}

/// What [`PackBuilder::finish_with`] needs for the part of a commit that may still be undone.
struct StagedCommit<'a> {
    commit: Commit,
    generation: u64,
    threads: usize,
    delta_blobs: u32,
    /// The header of the container a seeded builder extends.
    kept_header: Option<Header>,
    sources: &'a [SourceHash],
    inputs: &'a [SourceStamp],
    #[cfg(test)]
    fault: Option<fn(&Path)>,
}

/// The ending [`StagedCommit::run`] leaves to [`Sink::finish`].
enum StagedEnding {
    /// Replace: everything is written; the temporary file is renamed.
    Rename,
    /// Append: data and index are written; this footer is still to be written.
    AppendFooter([u8; 64]),
    /// In place: everything is written; the file is cut at the position.
    InPlace,
}

/// What is left for the ending after [`StagedCommit::run`].
struct Staged {
    ending: StagedEnding,
    index_stored_len: u64,
    verify: VerifyReport,
    unreferenced: ClassTable,
}

impl StagedCommit<'_> {
    /// Everything of a commit before its ending: header bits, index, the full decode, inputs again.
    /// Replace writes the index and footer and verifies the file; append verifies through a splice
    /// view first and then writes the index, leaving the footer to [`Ending::AppendFooter`].
    /// `index` is dropped once stored, so that it does not stack on the reader's memory bound.
    fn run(self, sink: &mut Sink, index: IndexBuilder) -> Result<Staged, Error> {
        sink.check()?;
        check_inputs(self.inputs)?;
        match self.commit {
            Commit::Replace => {
                if self.delta_blobs > 0 {
                    let header = Header { required_features: FEATURE_DELTA, ..Header::current() };
                    sink.patch_header(&header)?;
                }
                let index_stored_len = write_index(sink, &index, self.generation)?;
                drop(index);
                self.run_fault(sink.path());
                let written = Arc::new(FileSource::open(sink.path())?);
                let (verify, unreferenced) =
                    verify_output(written, sink.path(), self.threads, self.sources)?;
                check_inputs(self.inputs)?;
                Ok(Staged { ending: StagedEnding::Rename, index_stored_len, verify, unreferenced })
            }
            Commit::Append => {
                let header = self.kept_header.ok_or_else(|| Error::EditRefused {
                    detail: "an append needs the records of the container it extends".to_owned(),
                })?;
                // The header is below the frontier: a delta blob is legal only if it says so already.
                if self.delta_blobs > 0 && header.required_features & FEATURE_DELTA == 0 {
                    return Err(Error::EditRefused {
                        detail: "a delta blob cannot be appended to a container without required \
                                 feature bit 0"
                            .to_owned(),
                    });
                }
                let index_offset = sink.position();
                let StoredIndex { stored, footer } =
                    store_index(&index, index_offset, self.generation)?;
                drop(index);
                self.run_fault(sink.path());
                // The file as it will be: the data on disk, then index and footer from memory. Read
                // through the sink's handle: the path may name another file by now.
                let (verify, unreferenced) = self.verify_spliced(sink, index_offset, &stored, &footer)?;
                check_inputs(self.inputs)?;
                sink.write(&stored)?;
                // Lossless: `usize` is at most 64 bits on every supported target.
                let index_stored_len = stored.len() as u64;
                let ending = StagedEnding::AppendFooter(footer);
                Ok(Staged { ending, index_stored_len, verify, unreferenced })
            }
            Commit::InPlace => {
                // The header is the source's until here; bit 0 must now say whether a delta blob
                // is left, spec section 7a, both ways.
                let required_features = if self.delta_blobs > 0 { FEATURE_DELTA } else { 0 };
                sink.patch_header(&Header { required_features, ..Header::current() })?;
                let index_offset = sink.position();
                let StoredIndex { stored, footer } =
                    store_index(&index, index_offset, self.generation)?;
                drop(index);
                self.run_fault(sink.path());
                // Verification precedes visibility, as for an append: no index is written that
                // was not proven against the moved and the new data.
                let (verify, unreferenced) = self.verify_spliced(sink, index_offset, &stored, &footer)?;
                check_inputs(self.inputs)?;
                sink.write(&stored)?;
                sink.write(&footer)?;
                // Lossless: `usize` is at most 64 bits on every supported target.
                let index_stored_len = stored.len() as u64;
                Ok(Staged { ending: StagedEnding::InPlace, index_stored_len, verify, unreferenced })
            }
        }
    }

    /// Verifies the file as it will be once `stored` and `footer` are written at `index_offset`:
    /// the data on disk, read through the sink's own handle (the path may name another file by
    /// now), then index and footer from memory ([`SpliceSource`]).
    fn verify_spliced(
        &self,
        sink: &mut Sink,
        index_offset: u64,
        stored: &[u8],
        footer: &[u8; 64],
    ) -> Result<(VerifyReport, ClassTable), Error> {
        let written: Arc<dyn RandomAccess> = Arc::new(sink.written_source()?);
        let tail = [stored, footer.as_slice()].concat();
        let view = Arc::new(SpliceSource::new(written, index_offset, tail)?);
        verify_output(view, sink.path(), self.threads, self.sources)
    }

    /// Runs the test seam on the written file, if one is armed.
    #[cfg(test)]
    fn run_fault(&self, path: &Path) {
        if let Some(fault) = self.fault {
            fault(path);
        }
    }

    /// Outside tests there is no seam.
    #[cfg(not(test))]
    #[expect(clippy::unused_self, reason = "the signature of the test version, which reads the seam")]
    fn run_fault(&self, _: &Path) {}
}

/// Stored bytes of the chunks of `blob`.
fn stored_of(index: &Index, blob: &BlobRecord) -> Result<u64, Error> {
    let chunks = index.blob_chunks(blob)?;
    Ok(chunks.iter().map(|chunk| u64::from(chunk.stored_len)).sum())
}

/// Marks blob `id` reached; whether it was not before.
fn first_reach(reached: &mut [bool], id: u32) -> Result<bool, Error> {
    let slot = usize::try_from(id).ok().and_then(|at| reached.get_mut(at));
    let slot = slot.ok_or_else(|| Error::MalformedIndex {
        section: "BLOB",
        detail: format!("blob {id} does not exist"),
    })?;
    Ok(!std::mem::replace(slot, true))
}

/// The accounting of a carried member over `extents` of `index`: the stored bytes of a blob, and of
/// the base of a delta blob, count with the first carried member that reaches it, as `layer_report`
/// walks members.
fn carried_table(
    index: &Index,
    extents: &[ExtentRecord],
    reached: &mut [bool],
) -> Result<ClassTable, Error> {
    let mut table = ClassTable::default();
    for extent in extents {
        let blob = usize::try_from(extent.blob_id).ok().and_then(|id| index.blobs().get(id));
        let blob = blob.ok_or_else(|| Error::MalformedIndex {
            section: "EXTN",
            detail: format!("blob {} does not exist", extent.blob_id),
        })?;
        let view = extent.view()?;
        let newly_stored = if first_reach(reached, extent.blob_id)? {
            if blob.kind == BLOB_KIND_DELTA {
                let base = index.delta_params(blob)?.base_blob;
                if first_reach(reached, base)? {
                    let base_record = usize::try_from(base).ok().and_then(|id| index.blobs().get(id));
                    let base_record = base_record.ok_or_else(|| Error::MalformedIndex {
                        section: "BLOB",
                        detail: format!("base blob {base} does not exist"),
                    })?;
                    table.add_base(class_of(blob, view), stored_of(index, base_record)?);
                }
            }
            Some(stored_of(index, blob)?)
        } else {
            None
        };
        table.add_extent(blob, view, extent.virt_len, newly_stored);
    }
    Ok(table)
}

/// Compares every input with its stamp, see [`SourceStamp::check`].
fn check_inputs(inputs: &[SourceStamp]) -> Result<(), Error> {
    inputs.iter().try_for_each(SourceStamp::check)
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
