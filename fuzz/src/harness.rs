//! File: fuzz/src/harness.rs
//! Purpose: what the fuzz targets run, one function per target, plus the seeded mutator that runs the
//! same functions on stable.
//!
//! Main responsibilities:
//! - [`open_and_read`], [`decode_chunk`] and [`profile_scan`]: feed untrusted bytes to one public
//!   surface and panic on anything that is not a typed error of the right class;
//! - [`assemble`], [`assemble_with`] and [`to_fixed_up_input`]: the input form that reaches past the
//!   file's hashes;
//! - [`sample`]: a small valid container touching every chain, view and stream codec;
//!   [`delta_sample`]: the same for delta blobs (spec section 7a);
//! - [`Prng`] and [`mutate`]: seeded mutation for `crates/ziptensors-core/tests/malicious.rs`.
//!
//! Key structures: [`Bytes`], [`Sample`], [`Prng`].
//!
//! Implementation notes: both `fuzz/src/lib.rs` and `tests/malicious.rs` compile this one file, the
//! second through `#[path]`, so stable CI and the fuzzer run the same oracles. It may use only
//! `ziptensors_core`, `zstd`, `blake3`, `xxhash_rust` and `std`, which both have. A panic here is a
//! finding by construction, so the asserts are the point and not a shortcut around error handling.
//!
//! Arbitrary bytes alone stop at the header hash, because a fuzzer cannot guess an XXH3. So the first
//! input byte selects the form: raw file bytes, or the data region and the raw index, from which a
//! file with valid hashes is assembled. The second form is what reaches validation and decoding;
//! bit 1 of its form byte sets required feature bit 0 in the assembled header, which a file with
//! delta blobs needs. Chunk hashes stay as the input has them, which is why every read also runs
//! with `Verify::Off`.

use std::cell::RefCell;
use std::sync::Arc;

use xxhash_rust::xxh3::xxh3_64;
use ziptensors_core::chain::{BLOB_KIND_CHUNKED, BLOB_KIND_DELTA, FEATURE_DELTA};
use ziptensors_core::codec::{
    StreamPolicy, Workspace, decode_chunk_into, decode_delta_chunk_into, encode_chunk,
    encode_delta_chunk, widen_into,
};
use ziptensors_core::format::{
    BlobRecord, ChunkRecord, DeltaParams, ExtentRecord, Footer, HEADER_LEN, Header, IndexBuilder,
    MemberRecord, TensorRecord, check_restore_name, parse_footer, parse_header,
};
use ziptensors_core::profile::{MIN_RANGE_BYTES, normalize, scan, width_trial};
use ziptensors_core::{
    Chain, Container, DeltaOp, Dtype, ElementWidth, Error, ErrorCode, FieldTable, Layout, Member,
    OpenOptions, Profile, RandomAccess, ReadRequest, Step, Verify, View,
};

/// Index cap of the harness. Far under the default, so one input costs milliseconds, not a 256 MiB
/// allocation. The cap itself is exercised in `tests/malicious.rs`.
const MAX_INDEX_BYTES: u64 = 4 << 20;
/// `max_total_decoded` of the harness, in bytes.
const MAX_TOTAL_DECODED: u64 = 64 << 20;
/// Bytes of a member that are read. A member at the budget costs no more than a small one.
const READ_WINDOW: u64 = 1 << 20;
/// Members exercised per file.
const MEMBERS_READ: usize = 4;
/// Tensors exercised per member.
const TENSORS_READ: usize = 8;
/// Largest `raw_len` of [`decode_chunk`] is this many chain blocks.
const MAX_CHUNK_BLOCKS: usize = 512;
/// Bytes of the guard band on either side of a chunk destination.
const GUARD_LEN: usize = 64;
/// First guard band value.
const GUARD_BYTE: u8 = 0xA5;
/// Value of the destination in the first decode of [`decode_chunk`].
const FILL_FIRST: u8 = 0x00;
/// Value of the destination in the second decode, so a result that reads it shows.
const FILL_SECOND: u8 = 0xFF;
/// Input form tag of [`open_and_read`]: the rest is `u32` data length, data region, raw index.
const FORM_FIXED_UP: u8 = 1;
/// Bit of the form byte of the fixed-up form that sets required feature bit 0, delta blobs.
const FORM_DELTA_FEATURE: u8 = 2;
/// Bit of the form byte of [`decode_chunk`] that makes the base of a delta chain zero, so the residual
/// is the canonical bytes themselves; clear, the base is those bytes with every 16th byte changed.
const FORM_ZERO_BASE: u8 = 0x10;
/// Every this many bytes, the base of a delta chain in [`decode_chunk`] differs from the raw bytes.
const BASE_EDIT_EVERY: usize = 16;

/// A byte vector as a source.
#[derive(Debug)]
pub struct Bytes(pub Vec<u8>);

impl RandomAccess for Bytes {
    fn len(&self) -> u64 {
        to_u64(self.0.len())
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let len = to_u64(buf.len());
        let file_len = self.len();
        let outside = Error::OutOfBounds { offset: off, len, file_len };
        let start = usize::try_from(off).map_err(|_| outside)?;
        let outside = Error::OutOfBounds { offset: off, len, file_len };
        let end = start.checked_add(buf.len()).ok_or(outside)?;
        let outside = Error::OutOfBounds { offset: off, len, file_len };
        buf.copy_from_slice(self.0.get(start..end).ok_or(outside)?);
        Ok(())
    }
}

/// A slice length as `u64`. Lossless: `usize` is at most 64 bits on every supported target.
fn to_u64(len: usize) -> u64 {
    len as u64
}

/// A length that is at most `READ_WINDOW`, as `usize`.
fn window_len(len: u64) -> usize {
    usize::try_from(len.min(READ_WINDOW)).expect("the window fits a length")
}

/// Whether `error` is one that opening a file may give. `Io` is out: reads of the header, footer and
/// index are bounds-checked first, and this source has no other way to fail.
fn open_error(error: &Error) -> bool {
    matches!(
        error.code(),
        ErrorCode::NotZiptensors
            | ErrorCode::UnsupportedVersion
            | ErrorCode::UnsupportedFeature
            | ErrorCode::UnknownCodec
            | ErrorCode::MalformedIndex
            | ErrorCode::BudgetExceeded
            | ErrorCode::Zstd
    )
}

/// Whether `error` is one that reading a validated file may give. `MalformedIndex` is out: at this
/// stage it means a state `validate` claims to rule out, and `Io` or `OutOfBounds` that a chunk
/// reached past the file.
fn read_error(error: &Error) -> bool {
    matches!(
        error.code(),
        ErrorCode::ChunkCorrupt
            | ErrorCode::ChecksumMismatch
            | ErrorCode::UnknownCodec
            | ErrorCode::Zstd
            | ErrorCode::VerifyFailed
    )
}

/// Builds a file with valid header, footer and index hashes around `data` and `raw_index`, and no
/// feature bits.
///
/// # Panics
/// As [`assemble_with`].
#[must_use]
pub fn assemble(data: &[u8], raw_index: &[u8]) -> Vec<u8> {
    assemble_with(data, raw_index, 0)
}

/// As [`assemble`], with `required_features` in the header: [`FEATURE_DELTA`] for an index with
/// delta blobs.
///
/// # Panics
/// If zstd cannot compress, which is an environment failure and not a finding about the input.
#[must_use]
pub fn assemble_with(data: &[u8], raw_index: &[u8], required_features: u64) -> Vec<u8> {
    let stored = zstd::bulk::compress(raw_index, 1).expect("zstd compresses bytes in memory");
    let header = Header { required_features, ..Header::current() };
    let mut file = header.to_bytes().to_vec();
    file.extend_from_slice(data);
    let index_offset = to_u64(file.len());
    file.extend_from_slice(&stored);
    let footer = Footer::for_index(index_offset, to_u64(stored.len()), raw_index);
    file.extend_from_slice(&footer.to_bytes());
    file
}

/// The data region and the raw index of a valid `file`.
///
/// # Panics
/// If `file` is not valid, since this splits fixtures and never untrusted input.
#[must_use]
pub fn split_file(file: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let tail = file.len() - 64;
    let footer: &[u8; 64] = file[tail..].try_into().expect("64 bytes");
    let footer = parse_footer(footer).expect("a valid footer");
    let index_offset = usize::try_from(footer.index_offset).expect("a small file");
    let raw_len = usize::try_from(footer.index_raw_len).expect("a small file");
    let raw = zstd::bulk::decompress(&file[index_offset..tail], raw_len).expect("a valid index");
    (file[64..index_offset].to_vec(), raw)
}

/// The required feature bits in the header of a valid `file`.
///
/// # Panics
/// If `file` has no valid header, since this reads fixtures and never untrusted input.
#[must_use]
pub fn required_features(file: &[u8]) -> u64 {
    let header: &[u8; 64] = file.first_chunk().expect("a header");
    parse_header(header).expect("a valid header").required_features
}

/// `file` in the fixed-up form of [`open_and_read`], with the form bit of its delta feature.
///
/// # Panics
/// As [`split_file`].
#[must_use]
pub fn to_fixed_up_input(file: &[u8]) -> Vec<u8> {
    let (data, raw_index) = split_file(file);
    let delta = required_features(file) & FEATURE_DELTA != 0;
    let mut input = vec![if delta { FORM_FIXED_UP | FORM_DELTA_FEATURE } else { FORM_FIXED_UP }];
    let len = u32::try_from(data.len()).expect("a small file");
    input.extend_from_slice(&len.to_le_bytes());
    input.extend_from_slice(&data);
    input.extend_from_slice(&raw_index);
    input
}

/// Target `open_and_read`: any bytes, as a file, opened and read in every way a caller can.
///
/// # Panics
/// On any behaviour outside the contract of `Container` and `Member`, which is a finding.
pub fn open_and_read(input: &[u8]) {
    let Some((form, rest)) = input.split_first() else {
        return;
    };
    if form & 1 == FORM_FIXED_UP {
        let Some((len, rest)) = rest.split_first_chunk::<4>() else {
            return;
        };
        let at = usize::try_from(u32::from_le_bytes(*len)).map_or(rest.len(), |at| at.min(rest.len()));
        let (data, raw_index) = rest.split_at(at);
        let features = if form & FORM_DELTA_FEATURE == 0 { 0 } else { FEATURE_DELTA };
        exercise(assemble_with(data, raw_index, features));
    } else {
        exercise(rest.to_vec());
    }
}

/// Opens `file` under both verify settings and reads what opens.
fn exercise(file: Vec<u8>) {
    let source = Arc::new(Bytes(file));
    let mut checked = None;
    for verify in [Verify::Chunk, Verify::Off] {
        let options = OpenOptions {
            threads: 1,
            max_index_bytes: MAX_INDEX_BYTES,
            max_total_decoded: MAX_TOTAL_DECODED,
            verify,
            ..OpenOptions::default()
        };
        // The source is shared, so this is the coercion and no copy.
        let opened = Container::from_source(Arc::<Bytes>::clone(&source), &options);
        match opened {
            Ok(container) => {
                exercise_container(&container);
                checked = Some(container.members().len());
            }
            Err(error) => {
                assert!(open_error(&error), "open gave {error:?}");
                assert_eq!(checked, None, "verify settings disagree about opening: {error:?}");
            }
        }
    }
}

/// Reads the first members of `container`. Every `Err` must be a read error.
fn exercise_container(container: &Container) {
    for (id, info) in container.members().iter().enumerate().take(MEMBERS_READ) {
        // A check of a hostile name is what a frontend does before a restore.
        if let Err(error) = check_restore_name(&info.name) {
            assert!(matches!(error, Error::UnsafeMemberName { .. }), "member {id}: {error:?}");
        }
        let member = container.member(&info.name);
        let member = member.unwrap_or_else(|error| panic!("listed member {id}: {error:?}"));
        assert_eq!(member.len(), info.len);
        exercise_member(&member);
    }
}

/// Reads one member by every route and requires that the routes agree.
fn exercise_member(member: &Member) {
    let past_end = member.read_at(member.len(), &mut [0]);
    assert!(matches!(past_end, Err(Error::OutOfBounds { .. })), "{past_end:?}");
    let mut whole = vec![0; window_len(member.len())];
    let read = member.read_at(0, &mut whole);
    if let Err(error) = &read {
        assert!(read_error(error), "read_at gave {error:?}");
    }
    let parts_agree = compare_parts(member, read.is_ok().then_some(whole.as_slice()));
    exercise_tensors(member, read.is_ok().then_some(whole.as_slice()));
    if member.len() <= READ_WINDOW {
        match member.verify() {
            Ok(()) => {
                assert!(read.is_ok() && parts_agree, "verify accepted what reads refuse");
                assert_eq!(blake3::hash(&whole).as_bytes(), &member.blake3(), "verify accepted");
            }
            Err(error) => assert!(read_error(&error), "verify gave {error:?}"),
        }
    }
}

/// Reads the window of `member` in three pieces, which takes the partial chunk paths, and through
/// `read_many`. Compares with `whole` where that exists. Returns whether the pieces all read.
fn compare_parts(member: &Member, whole: Option<&[u8]>) -> bool {
    let len = window_len(member.len());
    let (first, second) = (len / 3, len / 3 * 2);
    let mut joined = vec![0; len];
    let mut all_read = true;
    for (start, end) in [(0, first), (first, second), (second, len)] {
        if let Err(error) = member.read_at(to_u64(start), &mut joined[start..end]) {
            assert!(read_error(&error), "partial read_at gave {error:?}");
            all_read = false;
        }
    }
    if all_read && let Some(whole) = whole {
        assert_eq!(whole, joined, "whole and partial reads differ");
    }
    let mut many = vec![0; len];
    let (head, tail) = many.split_at_mut(second);
    let (head, middle) = head.split_at_mut(first);
    let mut requests = [
        ReadRequest { offset: to_u64(second), dst: tail },
        ReadRequest { offset: 0, dst: head },
        ReadRequest { offset: to_u64(first), dst: middle },
    ];
    match member.read_many(&mut requests) {
        Ok(()) => {
            assert!(all_read, "read_many accepted what read_at refuses");
            assert_eq!(many, joined, "read_many and read_at differ");
        }
        Err(error) => assert!(read_error(&error), "read_many gave {error:?}"),
    }
    all_read
}

/// Reads the first tensors of `member` that fit the window. A tensor is a range: it must give the
/// bytes `whole` holds there, and an error must be a typed one.
fn exercise_tensors(member: &Member, whole: Option<&[u8]>) {
    for tensor in member.tensors().iter().take(TENSORS_READ) {
        let missized = member.decode_tensor_into(&tensor.name, &mut [0; 1]);
        if tensor.len != 1 {
            assert!(matches!(missized, Err(Error::DestinationSize { .. })), "{missized:?}");
        }
        if tensor.len > READ_WINDOW {
            continue;
        }
        let mut bytes = vec![0; window_len(tensor.len)];
        match member.decode_tensor_into(&tensor.name, &mut bytes) {
            Ok(()) => {
                let start = usize::try_from(tensor.offset).ok();
                let at = start.and_then(|start| whole?.get(start..start + bytes.len()));
                if let Some(expected) = at {
                    assert_eq!(expected, bytes, "tensor {} differs", tensor.name);
                }
            }
            Err(error) => assert!(read_error(&error), "tensor gave {error:?}"),
        }
    }
}

/// The chains [`decode_chunk`] picks from. Every step and layout of spec section 5.2 appears. The
/// first [`PLAIN_CHAINS`] have no delta step; chain `PLAIN_CHAINS + i` is chain `i` after
/// `delta_xor` of width 2, 4 or 8 by `i % 3`, so every pairing of a delta width with a later step
/// occurs.
#[must_use]
pub fn chain_table() -> Vec<Chain> {
    use ElementWidth::{Two as W2, Four as W4, Eight as W8};
    let layouts = [
        Layout::Whole,
        Layout::Planes(W2),
        Layout::Planes(W4),
        Layout::Planes(W8),
        Layout::Fields(FieldTable::Q80),
        Layout::Fields(FieldTable::Q4K),
        Layout::Fields(FieldTable::Q5K),
        Layout::Fields(FieldTable::Q6K),
    ];
    let mut chains = Vec::new();
    for layout in layouts {
        chains.push(Chain::new(&[Step::Layout(layout)]).expect("a valid chain"));
        for width in [W2, W4, W8] {
            let steps = [Step::Rotl1(width), Step::Layout(layout)];
            chains.push(Chain::new(&steps).expect("a valid chain"));
        }
    }
    let deltas: Vec<Chain> = chains
        .iter()
        .enumerate()
        .map(|(at, chain)| {
            let delta = Step::Delta(DeltaOp::Xor([W2, W4, W8][at % 3]));
            let steps: Vec<Step> = std::iter::once(delta).chain(chain.steps()).collect();
            Chain::new(&steps).expect("a valid chain")
        })
        .collect();
    chains.extend(deltas);
    chains
}

/// Chains without a delta step at the start of [`chain_table`], which is twice this long.
pub const PLAIN_CHAINS: usize = 32;

/// The base bytes a delta chain of [`decode_chunk`] combines with: zero if `form` has
/// [`FORM_ZERO_BASE`], otherwise `source` cycled to `raw_len` with every [`BASE_EDIT_EVERY`]th byte
/// changed. For the roundtrip forms `source` is the body the raw bytes are cycled from, so the
/// residual is sparse; for a hostile stored body it is just some base.
fn delta_base(source: &[u8], raw_len: usize, form: u8) -> Vec<u8> {
    if form & FORM_ZERO_BASE != 0 || source.is_empty() {
        return vec![0; raw_len];
    }
    let cycled = source.iter().copied().cycle().take(raw_len);
    cycled.enumerate().map(|(at, byte)| if at % BASE_EDIT_EVERY == 5 { byte ^ 0x5A } else { byte }).collect()
}

/// Canonical bytes of which one byte in each `width` is constant and the rest noise. Under
/// `EntropyOrStore` and `planes(width)` exactly one stream compresses, which is the in-place route.
#[must_use]
pub fn one_plane_body(seed: u64, len: usize, width: usize) -> Vec<u8> {
    let mut rng = Prng(seed);
    (0..len)
        .map(|at| if at % width == width - 1 { 7 } else { rng.next_u64().to_le_bytes()[2] })
        .collect()
}

/// The policies [`decode_chunk`] picks from.
pub const POLICIES: [StreamPolicy; 4] = [
    StreamPolicy::Store,
    StreamPolicy::Entropy,
    StreamPolicy::EntropyOrStore,
    StreamPolicy::Zstd { level: 3 },
];

thread_local! {
    /// A workspace that lives across inputs, so that its buffers hold what earlier inputs left.
    static USED_WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::new());
}

/// Decodes `stored` into a guarded destination of `raw_len` bytes filled with `fill`, combining it
/// with `base` if the chain has a delta step. Returns the bytes on success.
fn guarded_decode(
    chain: &Chain,
    stored: &[u8],
    (raw_len, base): (usize, Option<&[u8]>),
    fill: u8,
    verify: Option<u64>,
    workspace: &mut Workspace,
) -> Result<Vec<u8>, Error> {
    let mut buffer = vec![GUARD_BYTE; raw_len + 2 * GUARD_LEN];
    let (head, rest) = buffer.split_at_mut(GUARD_LEN);
    let (dst, tail) = rest.split_at_mut(raw_len);
    dst.fill(fill);
    let result = match base {
        Some(base) => decode_delta_chunk_into(chain, stored, base, dst, workspace, verify),
        None => decode_chunk_into(chain, stored, dst, workspace, verify),
    };
    assert!(head.iter().chain(tail.iter()).all(|byte| *byte == GUARD_BYTE), "a guard band changed");
    result.map(|()| dst.to_vec())
}

/// Requires the contract of `decode_chunk_into`, or of `decode_delta_chunk_into` with `base` for a
/// chain with a delta step, on hostile `stored`. The output must depend neither on what the
/// destination held, which is what would show a read ahead of the write head in the in-place route
/// (risk R1), nor on what the workspace did before, and a hash must be enforced exactly. A delta
/// decode must equal the residual decoded by the rest of the chain, XOR the base.
fn check_hostile_chunk(chain: &Chain, stored: &[u8], raw_len: usize, base: Option<&[u8]>) {
    let shape = (raw_len, base);
    USED_WORKSPACE.with_borrow_mut(|used| {
        let first = guarded_decode(chain, stored, shape, FILL_FIRST, None, used);
        let fresh = &mut Workspace::new();
        let second = guarded_decode(chain, stored, shape, FILL_SECOND, None, fresh);
        if let Some(base) = base {
            check_delta_decode(chain, stored, base, first.as_ref(), fresh);
        }
        match (first, second) {
            (Ok(first), Ok(second)) => {
                assert_eq!(first, second, "the output depends on old state");
                let hash = xxh3_64(&first);
                let same = guarded_decode(chain, stored, shape, FILL_FIRST, Some(hash), used);
                assert_eq!(same.ok().as_ref(), Some(&first), "the right hash was refused");
                let wrong = guarded_decode(chain, stored, shape, FILL_FIRST, Some(hash ^ 1), used);
                assert!(matches!(wrong, Err(Error::ChecksumMismatch { .. })), "{wrong:?}");
            }
            (Err(first), Err(second)) => {
                assert_eq!(first.code(), second.code(), "errors depend on old state");
                let allowed = [ErrorCode::ChunkCorrupt, ErrorCode::UnknownCodec, ErrorCode::Zstd];
                assert!(allowed.contains(&first.code()), "decode gave {first:?}");
            }
            (first, second) => panic!("old state decides: {first:?} and {second:?}"),
        }
    });
}

/// The delta part of [`check_hostile_chunk`]: `delta` is the result of decoding `stored` with `base`.
/// It must be the residual, decoded by the chain without its delta step, XOR `base`, or the same
/// error. The plain entry point must refuse the chain, and a base of the wrong length is refused.
fn check_delta_decode(
    chain: &Chain,
    stored: &[u8],
    base: &[u8],
    delta: Result<&Vec<u8>, &Error>,
    workspace: &mut Workspace,
) {
    let raw_len = base.len();
    let residual = guarded_decode(&chain.residual(), stored, (raw_len, None), FILL_SECOND, None, workspace);
    match (delta, residual) {
        (Ok(delta), Ok(residual)) => {
            let combined: Vec<u8> = residual.iter().zip(base).map(|(byte, base)| byte ^ base).collect();
            assert_eq!(*delta, combined, "a delta decode is not the residual XOR the base");
        }
        (Err(delta), Err(residual)) => assert_eq!(delta.code(), residual.code(), "{delta:?}"),
        (delta, residual) => panic!("only the delta step decides: {delta:?} and {residual:?}"),
    }
    let plain = guarded_decode(chain, stored, (raw_len, None), FILL_FIRST, None, workspace);
    assert!(matches!(plain, Err(Error::ChunkCorrupt { .. })), "{plain:?}");
    let longer = [base, &[0]].concat();
    let missized = guarded_decode(chain, stored, (raw_len, Some(&longer)), FILL_FIRST, None, workspace);
    assert!(matches!(missized, Err(Error::DestinationSize { .. })), "{missized:?}");
}

/// Target `decode_chunk`. Input: chain byte, form byte, two length bytes, body. Form 0 decodes the
/// body as a stored chunk. Form 1 roundtrips the body as canonical bytes through `encode_chunk`.
/// Form 2 does that, patches one byte of the stored chunk from the end of the body and decodes it
/// as hostile, which reaches deep into valid zstd frames. A chain with a delta step goes through
/// `encode_delta_chunk` and `decode_delta_chunk_into` with the base of [`delta_base`].
///
/// # Panics
/// On any behaviour outside the contract of `decode_chunk_into` and `decode_delta_chunk_into`, which
/// is a finding.
pub fn decode_chunk(input: &[u8]) {
    let Some(([chain_byte, form, low, high], body)) = input.split_first_chunk::<4>() else {
        return;
    };
    let chains = chain_table();
    let chain = &chains[usize::from(*chain_byte) % chains.len()];
    let block = usize::try_from(chain.block_bytes()).expect("a block fits a length");
    let blocks = usize::from(u16::from_le_bytes([*low, *high])) % (MAX_CHUNK_BLOCKS + 1);
    let raw_len = blocks * block;
    let policy = POLICIES[usize::from(*form >> 2) % POLICIES.len()];
    let base = chain.delta().map(|_| delta_base(body, raw_len, *form));
    let base = base.as_deref();
    match form & 3 {
        0 => check_hostile_chunk(chain, body, raw_len, base),
        shape => {
            if body.is_empty() {
                return;
            }
            let raw: Vec<u8> = body.iter().copied().cycle().take(raw_len).collect();
            let mut stored = Vec::new();
            let back = USED_WORKSPACE.with_borrow_mut(|used| {
                let encoded = match base {
                    Some(base) => {
                        let plain = encode_chunk(&raw, chain, &policy, used, &mut stored);
                        assert!(matches!(plain, Err(Error::ChunkCorrupt { .. })), "{plain:?}");
                        encode_delta_chunk(&raw, base, chain, &policy, used, &mut stored)
                    }
                    None => encode_chunk(&raw, chain, &policy, used, &mut stored),
                };
                let meta = encoded.unwrap_or_else(|error| panic!("encoding gave {error:?}"));
                assert_eq!(usize::try_from(meta.stored_len).ok(), Some(stored.len()));
                let hash = Some(meta.xxh3_decoded);
                guarded_decode(chain, &stored, (raw_len, base), FILL_SECOND, hash, used)
            });
            assert_eq!(back.ok().as_ref(), Some(&raw), "a roundtrip changed the bytes");
            if let (2, Some(([at_low, at_high], [.., xor]))) = (shape, body.split_first_chunk::<2>())
                && !stored.is_empty()
            {
                let at = usize::from(u16::from_le_bytes([*at_low, *at_high])) % stored.len();
                stored[at] ^= *xor;
                check_hostile_chunk(chain, &stored, raw_len, base);
            }
        }
    }
}

/// Target `profile_scan`: any bytes as a source file. Malformed content is never an error, and
/// `normalize` must make any plan safe to hand to a writer.
///
/// # Panics
/// On any behaviour outside the contract of `scan` and `normalize`, which is a finding.
pub fn profile_scan(input: &[u8]) {
    let source = Bytes(input.to_vec());
    let plan = scan(&source).unwrap_or_else(|error| panic!("scan gave {error:?}"));
    let file_len = to_u64(input.len());
    let once = normalize(plan, file_len);
    let mut end = 0;
    for range in &once.ranges {
        assert!(range.offset >= end, "ranges overlap or are unsorted: {range:?}");
        end = range.offset.checked_add(range.len).expect("a range inside the file");
        assert!(end <= file_len, "a range leaves the file: {range:?}");
        assert!(range.len >= MIN_RANGE_BYTES, "a short range survived: {range:?}");
    }
    for tensor in &once.tensors {
        let tensor_end = tensor.offset.checked_add(tensor.len);
        assert!(tensor_end.is_some_and(|end| end <= file_len), "a row leaves: {tensor:?}");
        assert!(tensor.shape.len() <= 16, "too many dimensions: {tensor:?}");
    }
    assert_eq!(normalize(once.clone(), file_len), once, "normalize is not idempotent");
    let width = width_trial(input);
    assert!(matches!(width, 1 | 2 | 4 | 8), "width trial gave {width}");
}

/// A valid container and what it holds.
#[derive(Debug)]
pub struct Sample {
    /// The file.
    pub file: Vec<u8>,
    /// Each member's name and content.
    pub members: Vec<(String, Vec<u8>)>,
}

/// One blob of [`sample`]: chain, policy, chunk size, canonical length in bytes.
type BlobPlan = (Chain, StreamPolicy, u32, usize);

/// One extent of [`sample`]: blob id, blob offset, length in the member, view id.
type Part = (u32, u64, u64, u16);

/// Bytes with a low-entropy high byte in each `width`, which compress plane by plane.
fn sample_bytes(rng: &mut Prng, len: usize, width: usize) -> Vec<u8> {
    (0..len)
        .map(|at| {
            let value = rng.next_u64().to_le_bytes()[5];
            if at % width == width - 1 { value & 1 } else { value }
        })
        .collect()
}

/// The container [`sample`] is building.
#[derive(Debug, Default)]
struct SampleBuilder {
    data: Vec<u8>,
    index: IndexBuilder,
    members: Vec<(String, Vec<u8>)>,
    next_extent: u64,
    next_tensor: u32,
    /// Every chunk record added, in `CHNK` order, so a blob can reuse another's.
    chunks: Vec<ChunkRecord>,
}

impl SampleBuilder {
    /// Encodes `bytes` under `plan` and returns the blob id.
    fn add_blob(&mut self, bytes: &[u8], plan: BlobPlan) -> Result<u32, Error> {
        let (first_chunk, chunk_count) = self.add_chunks(bytes, None, plan)?;
        let chain_id = self.index.add_chain(plan.0)?;
        let record = blob_of(bytes, plan.2, chain_id, (first_chunk, chunk_count));
        self.index.add_blob(record)
    }

    /// Encodes `bytes` as a delta blob over `base`, the canonical bytes of blob `base_id` from
    /// `base_offset` on, under `plan`, whose chain has a delta step. Returns the blob id.
    fn add_delta_blob(&mut self, bytes: &[u8], (base_id, base, base_offset): (u32, &[u8], u64), plan: BlobPlan) -> Result<u32, Error> {
        let start = usize::try_from(base_offset).expect("a small offset");
        let aligned = &base[start..start + bytes.len()];
        let chunks = self.add_chunks(bytes, Some(aligned), plan)?;
        let chain_id = self.index.add_chain(plan.0)?;
        let params = DeltaParams { base_blob: base_id, base_offset };
        self.add_delta_record(blob_of(bytes, plan.2, chain_id, chunks), params)
    }

    /// Adds a delta blob whose chunk records are those of blob `base_id`, of `chunks` chunk records
    /// from `first_chunk`, with the hashes of zero bytes: its residual is the base's own stored bytes,
    /// decoded by `chain` without the delta step, so with the base's chain it is zero after combining.
    /// `base` is the base's canonical bytes.
    fn add_shared_delta(&mut self, (base_id, base): (u32, &[u8]), (first_chunk, chunks): (u64, u32), chain: Chain, chunk_size: u32) -> Result<u32, Error> {
        let zeros = vec![0; base.len()];
        let start = usize::try_from(first_chunk).expect("a small index");
        let records = self.chunks[start..start + usize::try_from(chunks).expect("few chunks")].to_vec();
        let mut first = None;
        let raw = zeros.chunks(usize::try_from(chunk_size).expect("a small chunk"));
        for (record, raw) in records.into_iter().zip(raw) {
            let record = ChunkRecord { xxh3_decoded: xxh3_64(raw), ..record };
            self.chunks.push(record);
            first.get_or_insert(self.index.add_chunk(record));
        }
        let chain_id = self.index.add_chain(chain)?;
        let record = blob_of(&zeros, chunk_size, chain_id, (first.unwrap_or(0), chunks));
        self.add_delta_record(record, DeltaParams { base_blob: base_id, base_offset: 0 })
    }

    /// Adds `record` as a delta blob with `params` in `PARM`.
    fn add_delta_record(&mut self, record: BlobRecord, params: DeltaParams) -> Result<u32, Error> {
        let (params_off, params_len) = self.index.add_params(&params.encode())?;
        self.index.add_blob(BlobRecord { kind: BLOB_KIND_DELTA, params_off, params_len, ..record })
    }

    /// Encodes `bytes` chunk by chunk under `plan`, each chunk against the same range of `base` if
    /// given, appends them to the data region, and returns the first chunk id and the count.
    fn add_chunks(&mut self, bytes: &[u8], base: Option<&[u8]>, plan: BlobPlan) -> Result<(u64, u32), Error> {
        let (chain, policy, chunk_size, _) = plan;
        let chunk_len = usize::try_from(chunk_size).expect("a small chunk");
        let mut workspace = Workspace::new();
        let mut first_chunk = None;
        let mut chunk_count = 0;
        for (at, raw) in bytes.chunks(chunk_len).enumerate() {
            let mut stored = Vec::new();
            let meta = match base {
                Some(base) => {
                    let base = &base[at * chunk_len..at * chunk_len + raw.len()];
                    encode_delta_chunk(raw, base, &chain, &policy, &mut workspace, &mut stored)?
                }
                None => encode_chunk(raw, &chain, &policy, &mut workspace, &mut stored)?,
            };
            let record = ChunkRecord {
                file_offset: HEADER_LEN + to_u64(self.data.len()),
                xxh3_decoded: meta.xxh3_decoded,
                stored_len: meta.stored_len,
            };
            self.chunks.push(record);
            first_chunk.get_or_insert(self.index.add_chunk(record));
            chunk_count += 1;
            self.data.extend_from_slice(&stored);
        }
        Ok((first_chunk.unwrap_or(0), chunk_count))
    }

    /// Adds a member over `parts`, whose content is `content`, with one tensor if it is long enough.
    fn add_member(&mut self, name: &str, parts: &[Part], content: Vec<u8>) -> Result<(), Error> {
        let first_extent = self.next_extent;
        let mut virt_offset = 0;
        for (blob_id, blob_offset, virt_len, view_id) in parts {
            let record = ExtentRecord {
                virt_offset,
                virt_len: *virt_len,
                blob_id: *blob_id,
                view_id: *view_id,
                blob_offset: *blob_offset,
            };
            self.next_extent = self.index.add_extent(record) + 1;
            virt_offset += virt_len;
        }
        let first_tensor = self.next_tensor;
        let tensor_count = u32::from(virt_offset >= 8);
        if tensor_count == 1 {
            let tensor_name = self.index.add_string("t0")?;
            let shape_off = self.index.add_shape(&[2, 2])?;
            let record = TensorRecord {
                name: tensor_name,
                virt_offset: 4,
                virt_len: 4,
                shape_off,
                ndim: 2,
                dtype: Dtype::U8,
            };
            self.next_tensor = self.index.add_tensor(record)? + 1;
        }
        let member_name = self.index.add_string(name)?;
        self.index.add_member(MemberRecord {
            name: member_name,
            file_len: virt_offset,
            blake3: *blake3::hash(&content).as_bytes(),
            first_extent,
            extent_count: u32::try_from(parts.len()).expect("few extents"),
            first_tensor,
            tensor_count,
            profile: Profile::Opaque,
        })?;
        self.members.push((name.to_owned(), content));
        Ok(())
    }

    /// Adds every blob and member of [`sample`].
    fn fill(&mut self, seed: u64) -> Result<(), Error> {
        use ElementWidth::{Two as W2, Four as W4};
        let planes = |width| Layout::Planes(width);
        let rotated = Chain::new(&[Step::Rotl1(W2), Step::Layout(planes(W2))])?;
        let plans: [BlobPlan; 6] = [
            (Chain::new(&[Step::Layout(Layout::Whole)])?, StreamPolicy::Zstd { level: 3 }, 1024, 3000),
            (Chain::new(&[Step::Layout(planes(W2))])?, StreamPolicy::Entropy, 1024, 2600),
            (Chain::new(&[Step::Layout(planes(W4))])?, StreamPolicy::EntropyOrStore, 512, 1600),
            (rotated, StreamPolicy::Entropy, 512, 1200),
            (
                Chain::new(&[Step::Layout(Layout::Fields(FieldTable::Q80))])?,
                StreamPolicy::Entropy,
                340,
                850,
            ),
            (Chain::new(&[Step::Layout(Layout::Whole)])?, StreamPolicy::Store, 256, 600),
        ];
        let mut rng = Prng(seed);
        let mut blobs = Vec::new();
        for plan in plans {
            let block = usize::try_from(plan.0.block_bytes()).expect("a small block");
            let bytes = sample_bytes(&mut rng, plan.3, block);
            blobs.push((self.add_blob(&bytes, plan)?, bytes));
        }
        for (at, (id, bytes)) in blobs.iter().enumerate() {
            let part = (*id, 0, to_u64(bytes.len()), 0);
            self.add_member(&format!("m{at}"), &[part], bytes.clone())?;
        }
        let (id, bytes) = &blobs[1];
        let split = [&bytes[1500..2000], &bytes[100..1300]].concat();
        self.add_member("dir/split", &[(*id, 1500, 500, 0), (*id, 100, 1200, 0)], split)?;
        let mut widened = vec![0; 800];
        widen_into(View::F16ToF32, &bytes[..400], &mut widened)?;
        self.add_member("wide", &[(*id, 0, 800, View::F16ToF32.to_wire())], widened)?;
        self.add_member("empty", &[], Vec::new())
    }
}

/// A kind 0 record of a blob of canonical `bytes` in chunks `(first_chunk, chunk_count)`.
fn blob_of(bytes: &[u8], chunk_size: u32, chain_id: u32, (first_chunk, chunk_count): (u64, u32)) -> BlobRecord {
    BlobRecord {
        blake3: *blake3::hash(bytes).as_bytes(),
        raw_len: to_u64(bytes.len()),
        first_chunk,
        chunk_count,
        chunk_size,
        chain_id,
        kind: BLOB_KIND_CHUNKED,
        dtype_hint: Dtype::U8,
        params_off: 0,
        params_len: 0,
    }
}

/// `base` with a few bytes changed in their low bits, as a finetune changes a tensor.
fn finetune(rng: &mut Prng, base: &[u8]) -> Vec<u8> {
    base.iter().map(|byte| if rng.below(13) == 0 { byte ^ (1 << rng.below(3)) } else { *byte }).collect()
}

impl SampleBuilder {
    /// Adds every blob and member of [`delta_sample`].
    fn fill_delta(&mut self, seed: u64) -> Result<(), Error> {
        use ElementWidth::{Two as W2, Four as W4, Eight as W8};
        let delta = |width, rest: &[Step]| {
            let steps: Vec<Step> = std::iter::once(Step::Delta(DeltaOp::Xor(width))).chain(rest.iter().copied()).collect();
            Chain::new(&steps)
        };
        let planes = Step::Layout(Layout::Planes(W2));
        let mut rng = Prng(seed);
        let base_bytes = sample_bytes(&mut rng, 1800, 2);
        let base_plan: BlobPlan = (Chain::new(&[planes])?, StreamPolicy::Entropy, 512, 1800);
        let first_base_chunk = to_u64(self.chunks.len());
        let base = self.add_blob(&base_bytes, base_plan)?;
        let tuned = finetune(&mut rng, &base_bytes);
        let plan = (delta(W2, &[planes])?, StreamPolicy::Entropy, 512, tuned.len());
        let tuned_id = self.add_delta_blob(&tuned, (base, &base_bytes, 0), plan)?;
        let rotated = finetune(&mut rng, &base_bytes[512..1536]);
        let steps = [Step::Rotl1(W2), Step::Layout(Layout::Whole)];
        let plan = (delta(W4, &steps)?, StreamPolicy::Zstd { level: 3 }, 512, rotated.len());
        let rotated_id = self.add_delta_blob(&rotated, (base, &base_bytes, 512), plan)?;
        let shared = (first_base_chunk, 4);
        let zero_id = self.add_shared_delta((base, &base_bytes), shared, delta(W2, &[planes])?, 512)?;
        let tail = finetune(&mut rng, &base_bytes[1024..]);
        let plan = (delta(W8, &[Step::Layout(Layout::Whole)])?, StreamPolicy::Store, 512, tail.len());
        let tail_id = self.add_delta_blob(&tail, (base, &base_bytes, 1024), plan)?;

        let mut widened = vec![0; 800];
        widen_into(View::F16ToF32, &rotated[..400], &mut widened)?;
        let mixed = [&base_bytes[..300], &tuned[300..1000], &widened, &tail[8..208]].concat();
        let parts = [(base, 0, 300, 0), (tuned_id, 300, 700, 0), (rotated_id, 0, 800, View::F16ToF32.to_wire()), (tail_id, 8, 200, 0)];
        self.add_member("mixed", &parts, mixed)?;
        self.add_member("zero", &[(zero_id, 0, 1800, 0)], vec![0; 1800])?;
        self.add_member("tuned", &[(tuned_id, 0, 1800, 0)], tuned)?;
        let parts = [(rotated_id, 512, 512, 0), (tail_id, 0, to_u64(tail.len()), 0)];
        self.add_member("parts", &parts, [&rotated[512..], tail.as_slice()].concat())?;
        self.add_member("base", &[(base, 0, 1800, 0)], base_bytes)
    }
}

/// A valid container of a few KiB with delta blobs (spec section 7a) over one `planes(2)` base of
/// four chunks, the last one short: a finetune over all of it, one with `rotl1` at `base_offset` 512
/// read partly and widened, one of width 8 over the tail, and one whose chunk records are the base's
/// own, which makes its target zero. The first member mixes all of them with the base, for the four
/// members a fuzz input reads. Required feature bit 0 is set. Deterministic in `seed`.
///
/// # Panics
/// If building it fails, which is a bug of this function or of the encoder.
#[must_use]
pub fn delta_sample(seed: u64) -> Sample {
    let mut builder = SampleBuilder::default();
    builder.fill_delta(seed).expect("a delta sample builds");
    let file = assemble_with(&builder.data, &builder.index.serialize(), FEATURE_DELTA);
    Sample { file, members: builder.members }
}

/// A valid container of a few KiB, with one member per kind of chunk, a widened member, a member of
/// several extents over one blob, an empty member, and tensors. Deterministic in `seed`.
///
/// # Panics
/// If building it fails, which is a bug of this function or of the encoder.
#[must_use]
pub fn sample(seed: u64) -> Sample {
    let mut builder = SampleBuilder::default();
    builder.fill(seed).expect("a sample builds");
    Sample { file: assemble(&builder.data, &builder.index.serialize()), members: builder.members }
}

/// Splitmix64: small, seedable and the same everywhere.
#[derive(Debug, Clone)]
pub struct Prng(pub u64);

impl Prng {
    /// The next value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound`, `bound` not 0.
    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next_u64() % to_u64(bound)).expect("under a length")
    }
}

/// The values that break length checks.
const INTERESTING: [u64; 10] = [
    0,
    1,
    0x7F,
    0x80,
    0xFF,
    0x8000,
    0xFFFF_FFFF,
    0x8000_0000_0000_0000,
    u64::MAX,
    u64::MAX - 63,
];

/// Applies one to four mutations to `bytes` in place. Deterministic in `rng`.
pub fn mutate(rng: &mut Prng, bytes: &mut Vec<u8>) {
    for _ in 0..=rng.below(4) {
        if bytes.is_empty() {
            bytes.push(0);
        }
        let at = rng.below(bytes.len());
        match rng.below(6) {
            0 => bytes[at] ^= 1 << rng.below(8),
            1 => bytes[at] = rng.next_u64().to_le_bytes()[0],
            2 => {
                let value = INTERESTING[rng.below(INTERESTING.len())].to_le_bytes();
                let width = [2, 4, 8][rng.below(3)];
                let end = (at + width).min(bytes.len());
                bytes[at..end].copy_from_slice(&value[..end - at]);
            }
            3 => {
                let from = rng.below(bytes.len());
                let len = rng.below(64).min(bytes.len() - from).min(bytes.len() - at);
                bytes.copy_within(from..from + len, at);
            }
            4 => bytes.truncate(at),
            _ => {
                let extra = rng.below(32);
                let fill = rng.next_u64().to_le_bytes()[1];
                bytes.splice(at..at, std::iter::repeat_n(fill, extra));
            }
        }
    }
}
