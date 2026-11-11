//! File: read/container.rs
//! Purpose: opening a container: spec section 9.1 up to and including the decode budget.
//!
//! Main responsibilities:
//! - [`Container`], the shared handle, with the validated index and the decode engine;
//! - loading the stored index: one zstd frame, of exactly the declared length;
//! - [`OpenOptions`] and [`MemberInfo`].
//!
//! Key structures: [`Container`], [`OpenOptions`], [`MemberInfo`], `Shared`.
//!
//! Key functions: [`Container::open`], [`Container::from_source`].
//!
//! Implementation notes: nothing below the index is read here. The tensor table is built once for all
//! `TENS` records and members slice it, because members may share tensor ranges and a copy per member
//! would multiply two counts from the file.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use zstd::bulk::Decompressor;
use zstd::zstd_safe;

use crate::chain::Chain;
use crate::error::Error;
use crate::format::{
    DEFAULT_MAX_DECODE_AMPLIFICATION, DEFAULT_MAX_INDEX_BYTES, FOOTER_LEN, Footer, HEADER_LEN, Header,
    Index, Limits, check_decode_budget, parse_footer, parse_header, validate,
};
use crate::io::{FileSource, RandomAccess};
use crate::types::{Profile, TensorInfo, Verify};

use super::batch::{Engine, to_usize};
use super::member::Member;

/// First bytes of a zstd frame. A skippable frame has another magic, and this rules it out.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// How to open a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenOptions {
    /// Decode threads, and the size of the scratch pool. 0 means one per logical CPU; 1 means no
    /// pool, with everything inline on the calling thread.
    pub threads: usize,
    /// Cap on the decompressed index in bytes. Values over 1 GiB act as 1 GiB.
    pub max_index_bytes: u64,
    /// Budget on the sum of member lengths of the whole container, in bytes. A pack over it is
    /// refused at open, before any data is read. The default is `u64::MAX`, no budget: a caller
    /// opening untrusted input must set one.
    pub max_total_decoded: u64,
    /// Bound on the container's decode work as a multiple of the sum of member lengths, on top of one
    /// floor of 256 MiB. See [`Limits::max_decode_amplification`]. Raise it to open a file whose small extents point
    /// into large chunks, and pay that decoding on every read of them.
    pub max_decode_amplification: u64,
    /// What is checked per chunk.
    pub verify: Verify,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            threads: 0,
            max_index_bytes: DEFAULT_MAX_INDEX_BYTES,
            max_total_decoded: u64::MAX,
            max_decode_amplification: DEFAULT_MAX_DECODE_AMPLIFICATION,
            verify: Verify::Chunk,
        }
    }
}

impl OpenOptions {
    /// The bounds these options put on parsing and validation.
    fn limits(&self) -> Limits {
        Limits {
            max_index_bytes: self.max_index_bytes,
            max_total_decoded: self.max_total_decoded,
            max_decode_amplification: self.max_decode_amplification,
            ..Limits::default()
        }
    }

    /// The thread count in force.
    fn thread_count(&self) -> usize {
        if self.threads != 0 {
            return self.threads;
        }
        match thread::available_parallelism() {
            Ok(count) => count.get(),
            Err(error) => {
                tracing::warn!(%error, "CPU count is unknown, reading on one thread");
                1
            }
        }
    }
}

/// What a container says about one member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberInfo {
    /// Path string, `/` separated. It comes from the file: check it with
    /// [`crate::format::check_restore_name`] before joining it to a directory.
    pub name: String,
    /// Length of the virtual file in bytes.
    pub len: u64,
    /// BLAKE3 of the whole content.
    pub blake3: [u8; 32],
    /// Recognized source format, a hint.
    pub profile: Profile,
    /// Number of tensors in the table.
    pub tensor_count: u32,
}

/// The state all handles of a container share. Immutable after open, apart from the scratch pool.
#[derive(Debug)]
pub(super) struct Shared {
    pub(super) source: Source,
    /// The file behind the source, if it was opened from a path.
    pub(super) path: Option<PathBuf>,
    pub(super) index: Index,
    /// Chains by `chain_id`, from [`validate`].
    pub(super) chains: Vec<Chain>,
    pub(super) members: Vec<MemberInfo>,
    /// Member ids sorted by name, for lookups.
    by_name: Vec<u32>,
    /// One row per `TENS` record.
    pub(super) tensors: Vec<TensorInfo>,
    pub(super) engine: Engine,
}

/// The byte source. A newtype, since `dyn RandomAccess` is not `Debug`.
pub(super) struct Source(Arc<dyn RandomAccess>);

impl fmt::Debug for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Source").field("len", &self.0.len()).finish()
    }
}

impl Source {
    /// See [`RandomAccess::read_exact_at`].
    pub(super) fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
        self.0.read_exact_at(offset, buf)
    }
}

/// An open container. Cloning is cheap and shares everything. All methods take `&self` and may be
/// called from any number of threads.
#[derive(Debug, Clone)]
pub struct Container {
    pub(super) shared: Arc<Shared>,
}

impl Container {
    /// Opens the container at `path`. The file is recognized by its magic, not by its name.
    ///
    /// # Errors
    /// See [`Container::from_source`], plus [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path, options: &OpenOptions) -> Result<Self, Error> {
        let source = FileSource::open(path)?;
        Self::with_source(Arc::new(source), options, path, Some(path.to_path_buf()))
    }

    /// Opens a container held by any [`RandomAccess`] source. Follows spec section 9.1 through the
    /// decode budget. Reads the header, the footer and the stored index, and no data.
    ///
    /// # Errors
    /// - [`Error::NotZiptensors`] if the source is too short for a header or has the wrong magic.
    /// - [`Error::UnsupportedVersion`], [`Error::UnsupportedFeature`], [`Error::UnknownCodec`] for
    ///   what this version does not implement.
    /// - [`Error::BudgetExceeded`] for an index over its cap, a stored index over the largest size
    ///   zstd gives that index, too much name work, members over `max_total_decoded`, or decode
    ///   work over its bound.
    /// - [`Error::MalformedIndex`] for anything that breaks spec invariants 1 to 9.
    /// - [`Error::Zstd`] if zstd cannot decode the stored index.
    /// - [`Error::Io`] if the source fails, or the thread pool cannot be started.
    pub fn from_source(
        source: Arc<dyn RandomAccess>,
        options: &OpenOptions,
    ) -> Result<Self, Error> {
        Self::with_source(source, options, Path::new("<source>"), None)
    }

    /// [`Container::from_source`], with `path` for messages, and `file` for a file on disk.
    fn with_source(
        source: Arc<dyn RandomAccess>,
        options: &OpenOptions,
        path: &Path,
        file: Option<PathBuf>,
    ) -> Result<Self, Error> {
        let limits = options.limits();
        let file_len = source.len();
        tracing::debug!(path = %path.display(), file_len, "opening container");
        let (header, footer, raw) = read_index(source.as_ref(), &limits)?;
        let index = Index::parse(&raw, &limits)?;
        drop(raw);
        let chains = validate(&index, &header, &footer, file_len, &limits)?;
        let member_count = u32::try_from(index.members().len())
            .map_err(|_| super::batch::broken("too many members for a u32 id"))?;
        check_decode_budget(&index, 0..member_count, &limits)?;

        let members = member_infos(&index)?;
        let mut by_name: Vec<u32> = (0..member_count).collect();
        by_name.sort_unstable_by(|a, b| {
            let name = |id: &u32| members.get(*id as usize).map(|member| member.name.as_str());
            name(a).cmp(&name(b))
        });
        let tensors = tensor_table(&index)?;
        let largest_chunk = index.blobs().iter().map(|blob| blob.chunk_size).max().unwrap_or(0);
        let threads = options.thread_count();
        let engine = Engine::new(threads, largest_chunk, options.verify, path)?;
        tracing::info!(
            path = %path.display(),
            members = members.len(),
            blobs = index.blobs().len(),
            threads,
            largest_chunk,
            "opened container"
        );
        let shared = Shared {
            source: Source(source),
            path: file,
            index,
            chains,
            members,
            by_name,
            tensors,
            engine,
        };
        Ok(Self { shared: Arc::new(shared) })
    }

    /// The members, in file order.
    #[must_use]
    pub fn members(&self) -> &[MemberInfo] {
        &self.shared.members
    }

    /// The member called `name`.
    ///
    /// # Errors
    /// Returns [`Error::MemberNotFound`] if there is none.
    pub fn member(&self, name: &str) -> Result<Member, Error> {
        Member::new(self.clone(), self.member_id(name)?)
    }

    /// The file-order id of the member called `name`, for handles that key state by member.
    ///
    /// # Errors
    /// Returns [`Error::MemberNotFound`] if there is none.
    pub(super) fn member_id(&self, name: &str) -> Result<u32, Error> {
        let shared = &self.shared;
        let found = shared.by_name.binary_search_by(|id| {
            let member = shared.members.get(*id as usize);
            member.map(|member| member.name.as_str()).cmp(&Some(name))
        });
        let not_found = || Error::MemberNotFound { name: name.to_owned() };
        let at = found.map_err(|_| not_found())?;
        shared.by_name.get(at).copied().ok_or_else(not_found)
    }

    /// The validated index. For `write::merge`, which copies records and stored chunks; everything
    /// else reads through [`Member`].
    pub(crate) fn index(&self) -> &Index {
        &self.shared.index
    }

    /// Reads `buf.len()` raw bytes of the container at `offset`, with no decoding. For `write::merge`.
    pub(crate) fn read_stored(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
        self.shared.source.read_exact_at(offset, buf)
    }

    /// Whether `other` is a handle on the same opened container.
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// The only member of a single-file container.
    ///
    /// # Errors
    /// Returns [`Error::MemberNotFound`] unless the container holds exactly one member.
    pub fn single(&self) -> Result<Member, Error> {
        match self.shared.members.len() {
            1 => Member::new(self.clone(), 0),
            count => {
                Err(Error::MemberNotFound { name: format!("the single member of {count}") })
            }
        }
    }
}

/// Reads and checks the header and footer, then loads the index: steps 1 and 2 of spec section 9.1.
/// Returns the header, the footer and the decompressed index, whose length and hash are checked.
fn read_index(
    source: &dyn RandomAccess,
    limits: &Limits,
) -> Result<(Header, Footer, Vec<u8>), Error> {
    let file_len = source.len();
    if file_len < HEADER_LEN {
        return Err(Error::NotZiptensors);
    }
    let mut header = [0; 64];
    source.read_exact_at(0, &mut header)?;
    let header = parse_header(&header)?;
    let footer_offset = file_len.checked_sub(FOOTER_LEN).filter(|at| *at >= HEADER_LEN);
    let footer_offset = footer_offset.ok_or_else(|| Error::MalformedIndex {
        section: "footer",
        detail: format!("a file of {file_len} bytes has no room for a footer"),
    })?;
    let mut footer = [0; 64];
    source.read_exact_at(footer_offset, &mut footer)?;
    let footer = parse_footer(&footer)?;
    footer.check_layout(file_len, limits)?;

    let raw_len = to_usize(footer.index_raw_len)?;
    // The file really may be this long, so the length of the stored index needs a bound of its own
    // before it sizes an allocation. No zstd encoder exceeds this one.
    let stored_limit = zstd_safe::compress_bound(raw_len);
    let stored_len = to_usize(footer.index_stored_len)?;
    if stored_len > stored_limit {
        return Err(Error::BudgetExceeded {
            what: "stored index",
            declared: footer.index_stored_len,
            // Lossless: `usize` is at most 64 bits on every supported target.
            limit: stored_limit as u64,
        });
    }
    let mut stored = vec![0; stored_len];
    source.read_exact_at(footer.index_offset, &mut stored)?;
    let raw = decompress_index(&stored, raw_len, footer.index_offset)?;
    footer.check_index(&raw)?;
    Ok((header, footer, raw))
}

/// Decompresses a stored index, read from file offset `offset`, into a new buffer of exactly
/// `raw_len` bytes.
///
/// # Errors
/// Returns [`Error::MalformedIndex`] for section `"index"` unless `stored` is exactly one frame that
/// declares `raw_len` bytes and yields that many, and [`Error::Zstd`] if zstd cannot decode it; its
/// detail names the stored index, its length and offset, since zstd's own text ("Data corruption
/// detected") says nothing of where.
fn decompress_index(stored: &[u8], raw_len: usize, offset: u64) -> Result<Vec<u8>, Error> {
    let malformed = |detail: String| Error::MalformedIndex { section: "index", detail };
    let context_error =
        |error: std::io::Error| Error::Zstd { detail: format!("no decompression context: {error}") };
    let decode_error = |error: std::io::Error| Error::Zstd {
        detail: format!(
            "the stored index ({} bytes at offset {offset}) does not decode: {error}",
            stored.len()
        ),
    };
    if !stored.starts_with(&ZSTD_MAGIC) {
        return Err(malformed("the stored index does not start with a zstd frame".to_owned()));
    }
    let frame_len = zstd_safe::find_frame_compressed_size(stored)
        .map_err(|code| malformed(format!("zstd frame: {}", zstd_safe::get_error_name(code))))?;
    if frame_len != stored.len() {
        return Err(malformed(format!(
            "the stored index is {} bytes, its first frame {frame_len}",
            stored.len()
        )));
    }
    let declared = zstd_safe::get_frame_content_size(stored)
        .map_err(|error| malformed(format!("zstd frame: {error}")))?;
    // Lossless: `usize` is at most 64 bits on every supported target.
    if declared != Some(raw_len as u64) {
        return Err(malformed(format!(
            "the frame declares {declared:?} bytes, the footer {raw_len}"
        )));
    }
    let mut raw = vec![0; raw_len];
    let written = Decompressor::new()
        .map_err(context_error)?
        .decompress_to_buffer(stored, raw.as_mut_slice())
        .map_err(decode_error)?;
    if written != raw_len {
        return Err(malformed(format!("decompressed {written} bytes, expected {raw_len}")));
    }
    Ok(raw)
}

/// Builds [`MemberInfo`] for every member. Costs at most the name budget.
fn member_infos(index: &Index) -> Result<Vec<MemberInfo>, Error> {
    index
        .members()
        .iter()
        .map(|member| {
            Ok(MemberInfo {
                name: index.string(member.name)?.to_owned(),
                len: member.file_len,
                blake3: member.blake3,
                profile: member.profile,
                tensor_count: member.tensor_count,
            })
        })
        .collect()
}

/// Builds the tensor table, one row per record.
fn tensor_table(index: &Index) -> Result<Vec<TensorInfo>, Error> {
    index
        .tensors()
        .iter()
        .map(|tensor| {
            Ok(TensorInfo {
                name: index.string(tensor.name)?.to_owned(),
                offset: tensor.virt_offset,
                len: tensor.virt_len,
                shape: index.tensor_shape(tensor)?.to_vec(),
                dtype: tensor.dtype,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::decompress_index;
    use crate::error::Error;

    #[test]
    fn an_index_zstd_cannot_decode_is_named_with_its_place() {
        // Compressible, so the frame holds compressed blocks whose damage zstd reports.
        let raw: Vec<u8> = (0..4096_u32).flat_map(|i| (i % 251).to_le_bytes()).collect();
        let stored = zstd::bulk::compress(&raw, 3).expect("compresses");
        assert_eq!(decompress_index(&stored, raw.len(), 77).expect("decodes"), raw);
        let place = format!("the stored index ({} bytes at offset 77) does not decode: ", stored.len());
        let mut zstd_failures = 0;
        // Damage each byte after the frame header in turn: whatever zstd says is either a
        // malformed frame or a Zstd error naming the stored index; it never decodes to `raw`.
        for at in 6..stored.len() {
            let mut damaged = stored.clone();
            damaged[at] ^= 0x5A;
            match decompress_index(&damaged, raw.len(), 77) {
                Err(Error::Zstd { detail }) => {
                    assert!(detail.starts_with(&place), "{detail}");
                    zstd_failures += 1;
                }
                Err(Error::MalformedIndex { section: "index", .. }) => {}
                // Damage zstd does not see is left to the index hash, checked after this.
                Ok(decoded) => assert_ne!(decoded, raw, "byte {at}"),
                Err(other) => panic!("byte {at}: {other:?}"),
            }
        }
        assert!(zstd_failures > 0, "no damage reached the decoder");
    }
}
