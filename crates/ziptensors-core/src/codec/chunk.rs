//! File: codec/chunk.rs
//! Purpose: the stored form of a chunk: prologue, streams, and the whole of spec sections 6.1 to 6.3.
//!
//! Main responsibilities: encode canonical bytes to one self-contained stored chunk, and decode a
//! stored chunk, which is untrusted, into a buffer of exactly its raw length.
//!
//! Key structures: [`ChunkMeta`].
//!
//! Key functions: [`encode_chunk`], [`decode_chunk_into`], [`encode_delta_chunk`],
//! [`decode_delta_chunk_into`], [`locate_error`].
//!
//! Implementation notes: all stream lengths derive from the chain and the raw length, never from the
//! file. A chain with a delta step has its own pair of entry points, which take the base bytes: the
//! residual goes through the rest of the chain exactly like canonical bytes of a plain chunk, so every
//! route below serves it, and the plain entry points refuse such a chain so that no caller can skip
//! the base. Decoding picks one of three routes, all with identical output (section 9.2):
//! - no compressed stream, or a `whole` layout: `dst` is filled directly;
//! - a `planes` chunk with one compressed stream: that stream decodes into the tail of `dst`, and the
//!   stored planes merge around it in place, with no scratch (proof in `kernels_x86.rs`);
//! - anything else: compressed streams decode into scratch, stored streams are read where they lie.
//!
//! The in-place tail is written by zstd through a slice of `dst`, so it cannot leave `dst`, and the
//! stored bytes are a shared borrow, so they cannot overlap `dst` (asserted in debug builds).

use xxhash_rust::xxh3::xxh3_64;

use crate::chain::{Chain, FieldTable, Layout, STREAM_STORE, STREAM_ZSTD};
use crate::error::Error;

use super::delta;
use super::kernels::{self, Rotation};
use super::stream::{StreamPolicy, ZstdContexts, codec_id, corrupt};
use super::workspace::{Workspace, sized};

/// [`crate::chain::MAX_CHUNK_SIZE`] as a slice length. The cast is lossless: the value is 8 MiB and
/// `try_from` is not available in a `const`.
#[allow(clippy::cast_possible_truncation, reason = "8 MiB fits usize on every supported target")]
pub const MAX_CHUNK_BYTES: usize = crate::chain::MAX_CHUNK_SIZE as usize;
/// Bytes of one stream word.
const WORD_BYTES: usize = 4;
/// Largest stream count of any layout, as `Layout::stream_count` documents.
const MAX_STREAMS: usize = 8;
/// Bits of a stream word below the codec.
const CODEC_SHIFT: u32 = 24;
/// Mask of `stored_len` in a stream word.
const STORED_LEN_MASK: u32 = (1 << CODEC_SHIFT) - 1;

/// What the index needs about an encoded chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkMeta {
    /// Stored bytes including the prologue: `CHNK.stored_len`.
    pub stored_len: u32,
    /// XXH3 of the canonical bytes: `CHNK.xxh3_decoded`.
    pub xxh3_decoded: u64,
}

/// One stream of a parsed chunk.
#[derive(Debug, Clone, Copy)]
struct StoredStream<'a> {
    /// Stream codec, validated.
    codec: u8,
    /// The stored bytes of this stream.
    bytes: &'a [u8],
    /// Decoded length.
    raw_len: usize,
}

/// Encodes `raw`, the canonical bytes of one chunk, and appends the stored chunk to `out`.
///
/// `raw` must be a multiple of the chain block and at most [`MAX_CHUNK_BYTES`]. The bytes appended
/// depend only on `raw`, `chain` and `policy`, not on `ws` or what it did before.
///
/// # Errors
/// Returns [`Error::BudgetExceeded`] if `raw` is over [`MAX_CHUNK_BYTES`], [`Error::ChunkCorrupt`] if
/// `raw` is not a multiple of the chain block or `chain` has a delta step (see
/// [`encode_delta_chunk`]), and [`Error::Zstd`] if zstd fails. On error `out` keeps its previous
/// contents.
pub fn encode_chunk(
    raw: &[u8],
    chain: &Chain,
    policy: &StreamPolicy,
    ws: &mut Workspace,
    out: &mut Vec<u8>,
) -> Result<ChunkMeta, Error> {
    if chain.delta().is_some() {
        return Err(corrupt("a chain with a delta step needs a base: use encode_delta_chunk"));
    }
    check_raw_len(raw.len(), *chain)?;
    let stored_len = append_chunk(raw, *chain, *policy, ws, out)?;
    chunk_meta(stored_len, raw)
}

/// Encodes `raw`, the canonical bytes of one chunk of a delta blob, against `base`, the base bytes of
/// spec section 7a, and appends the stored chunk to `out`.
///
/// `chain` must have a delta step. The residual is encoded by the rest of the chain into
/// [`Workspace`] scratch of the chunk's length, and `xxh3_decoded` is of `raw`, the bytes after
/// combining. The bytes appended depend only on `raw`, `base`, `chain` and `policy`.
///
/// # Errors
/// As [`encode_chunk`], with [`Error::ChunkCorrupt`] if `chain` has no delta step, and
/// [`Error::DestinationSize`] if `base` differs in length from `raw`. On error `out` keeps its
/// previous contents.
pub fn encode_delta_chunk(
    raw: &[u8],
    base: &[u8],
    chain: &Chain,
    policy: &StreamPolicy,
    ws: &mut Workspace,
    out: &mut Vec<u8>,
) -> Result<ChunkMeta, Error> {
    let op = chain.delta().ok_or_else(|| corrupt("encode_delta_chunk needs a delta step"))?;
    check_raw_len(raw.len(), *chain)?;
    // Taken out for the call, so that the rest of the workspace can be borrowed with it.
    let mut residual = std::mem::take(&mut ws.residual);
    let result = (|| {
        let target = sized(&mut residual, raw.len());
        delta::residual(op, raw, base, target)?;
        append_chunk(target, chain.residual(), *policy, ws, out)
    })();
    ws.residual = residual;
    chunk_meta(result?, raw)
}

/// Checks the raw length of a chunk to encode against the 8 MiB limit and the chain block.
fn check_raw_len(len: usize, chain: Chain) -> Result<(), Error> {
    if len > MAX_CHUNK_BYTES {
        return Err(Error::BudgetExceeded {
            what: "chunk raw length",
            declared: len as u64,
            limit: MAX_CHUNK_BYTES as u64,
        });
    }
    let block = chain_block(chain);
    if !len.is_multiple_of(block) {
        return Err(corrupt(format!("raw length {len} is not a multiple of the chain block {block}")));
    }
    Ok(())
}

/// Appends the streams of `input` to `out` and returns how many bytes it appended, or leaves `out`
/// as it was on error. `chain` has no delta step.
fn append_chunk(
    input: &[u8],
    chain: Chain,
    policy: StreamPolicy,
    ws: &mut Workspace,
    out: &mut Vec<u8>,
) -> Result<usize, Error> {
    let start = out.len();
    write_streams(input, chain, policy, ws, out).inspect_err(|_| out.truncate(start))?;
    Ok(out.len() - start)
}

/// The metadata of a chunk of `stored_len` stored bytes whose canonical bytes are `raw`.
fn chunk_meta(stored_len: usize, raw: &[u8]) -> Result<ChunkMeta, Error> {
    Ok(ChunkMeta {
        // Lossless: at most 8 words plus 8 MiB, since no stream exceeds its raw length.
        stored_len: u32::try_from(stored_len).map_err(|_| corrupt("stored chunk over 4 GiB"))?,
        xxh3_decoded: xxh3_64(raw),
    })
}

/// Appends the prologue and streams of `raw` to `out`.
fn write_streams(
    raw: &[u8],
    chain: Chain,
    policy: StreamPolicy,
    ws: &mut Workspace,
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    let layout = chain.layout();
    let input: &[u8] = match chain.rotl1() {
        Some(width) => {
            let rotated = sized(&mut ws.rotated, raw.len());
            rotated.copy_from_slice(raw);
            kernels::rotate(ws.isa, Rotation::Left, width, rotated);
            rotated
        }
        None => raw,
    };
    let prologue_start = out.len();
    let stream_count = layout.stream_count();
    out.resize(prologue_start + WORD_BYTES * stream_count, 0);
    for k in 0..stream_count {
        let stream: &[u8] = if layout == Layout::Whole {
            input
        } else {
            let (block, offset, len) = stream_span(layout, k);
            let target = sized(&mut ws.stream, input.len() / block * len);
            kernels::gather_field(input, block, offset, len, target);
            target
        };
        let keep = ws.zstd.compress(policy, stream, &mut ws.compressed)?;
        let bytes = if keep { ws.compressed.as_slice() } else { stream };
        let word = stream_word(codec_id(keep), bytes.len())?;
        let slot = prologue_start + WORD_BYTES * k;
        out[slot..slot + WORD_BYTES].copy_from_slice(&word.to_le_bytes());
        out.extend_from_slice(bytes);
    }
    Ok(())
}

/// Builds a stream word. A stream never exceeds its raw length, so `stored_len` fits 24 bits.
fn stream_word(codec: u8, stored_len: usize) -> Result<u32, Error> {
    u32::try_from(stored_len)
        .ok()
        .filter(|len| *len <= STORED_LEN_MASK)
        .map(|len| (u32::from(codec) << CODEC_SHIFT) | len)
        .ok_or_else(|| corrupt(format!("stream of {stored_len} bytes does not fit a stream word")))
}

/// The chain block as a length. `u32` to `usize` is lossless on every supported target.
fn chain_block(chain: Chain) -> usize {
    chain.block_bytes() as usize
}

/// Stream `k` as `(block, offset, len)`: `len` bytes at `offset` of every `block` bytes. `k` is below
/// the layout's stream count. Table values are small, so the `u32` conversions are lossless.
fn stream_span(layout: Layout, k: usize) -> (usize, usize, usize) {
    match layout {
        Layout::Whole => (1, 0, 1),
        Layout::Planes(width) => (width.bytes() as usize, k, 1),
        Layout::Fields(table) => {
            let block = table.block_bytes() as usize;
            let field = table.fields().get(k);
            (block, field.map_or(0, |f| f.offset as usize), field.map_or(0, |f| f.len as usize))
        }
    }
}

/// Decodes a stored chunk into `dst`, whose length is the chunk's raw length, and which must be a
/// multiple of the chain block and at most [`MAX_CHUNK_BYTES`].
///
/// `stored` is untrusted: any bytes give `Ok` or a typed error, never a panic, and memory use does not
/// depend on them. `verify` is the expected XXH3 of the canonical bytes, `None` to skip the check,
/// which the caller may do by choice only: all lengths are still enforced.
///
/// On error the contents of `dst` are unspecified. [`Error::ChunkCorrupt`] and
/// [`Error::ChecksumMismatch`] carry blob and chunk 0, as this layer does not know them; see
/// [`locate_error`].
///
/// # Errors
/// - [`Error::DestinationSize`]: `dst` cannot be a raw length of this chain.
/// - [`Error::UnknownCodec`]: a stream word names an unknown stream codec.
/// - [`Error::ChunkCorrupt`]: bad prologue, or a stream breaks spec section 6.1, or `chain` has a
///   delta step (see [`decode_delta_chunk_into`]).
/// - [`Error::ChecksumMismatch`]: the hash differs from `verify`.
/// - [`Error::Zstd`]: a zstd context cannot be created.
pub fn decode_chunk_into(
    chain: &Chain,
    stored: &[u8],
    dst: &mut [u8],
    ws: &mut Workspace,
    verify: Option<u64>,
) -> Result<(), Error> {
    if chain.delta().is_some() {
        return Err(corrupt("a chain with a delta step needs its base: use decode_delta_chunk_into"));
    }
    decode_streams(*chain, stored, dst, ws)?;
    check_hash(dst, verify)
}

/// Decodes a stored chunk of a delta blob into `dst`, combining it with `base`, the base bytes of
/// spec section 7a: the first `dst.len()` canonical bytes of the base chunk, already decoded and, if
/// the caller verifies, checked against that chunk's own hash.
///
/// `chain` must have a delta step. The residual decodes into `dst` by the routes of
/// [`decode_chunk_into`], so the in-place route applies; then `base` is combined into it, and
/// `verify` is checked on the result. All of [`decode_chunk_into`]'s guarantees hold.
///
/// # Errors
/// As [`decode_chunk_into`], with [`Error::ChunkCorrupt`] if `chain` has no delta step, and
/// [`Error::DestinationSize`] if `base` differs in length from `dst`.
pub fn decode_delta_chunk_into(
    chain: &Chain,
    stored: &[u8],
    base: &[u8],
    dst: &mut [u8],
    ws: &mut Workspace,
    verify: Option<u64>,
) -> Result<(), Error> {
    let op = chain.delta().ok_or_else(|| corrupt("decode_delta_chunk_into needs a delta step"))?;
    check_destination(*chain, dst.len())?;
    if base.len() != dst.len() {
        // Lossless: `usize` is at most 64 bits on every supported target.
        let (expected, actual) = (dst.len() as u64, base.len() as u64);
        return Err(Error::DestinationSize { expected, actual });
    }
    decode_streams(chain.residual(), stored, dst, ws)?;
    delta::combine(op, dst, base)?;
    check_hash(dst, verify)
}

/// Checks that `len` can be a raw length of a chunk of `chain`: at most [`MAX_CHUNK_BYTES`] and a
/// multiple of the chain block.
fn check_destination(chain: Chain, len: usize) -> Result<(), Error> {
    let block = chain_block(chain);
    if len > MAX_CHUNK_BYTES || !len.is_multiple_of(block) {
        let expected = len.min(MAX_CHUNK_BYTES) / block * block;
        return Err(Error::DestinationSize { expected: expected as u64, actual: len as u64 });
    }
    Ok(())
}

/// Steps 1 to 5 of spec section 6.2: the stored streams of a chain without a delta step back into
/// `dst`, with no hash check.
fn decode_streams(
    chain: Chain,
    stored: &[u8],
    dst: &mut [u8],
    ws: &mut Workspace,
) -> Result<(), Error> {
    check_destination(chain, dst.len())?;
    debug_assert!(disjoint(stored, dst), "stored chunk overlaps its destination");
    let layout = chain.layout();
    let streams = parse_streams(layout, stored, dst.len())?;
    let streams = &streams[..layout.stream_count()];
    match layout {
        Layout::Whole => decode_whole(streams, dst, &mut ws.zstd)?,
        Layout::Planes(_) => decode_planes(streams, dst, ws)?,
        Layout::Fields(table) => decode_fields(table, streams, dst, ws)?,
    }
    if let Some(width) = chain.rotl1() {
        kernels::rotate(ws.isa, Rotation::Right, width, dst);
    }
    Ok(())
}

/// The last step of spec section 6.2: `dst` hashes to `verify`, unless that is `None`.
fn check_hash(dst: &[u8], verify: Option<u64>) -> Result<(), Error> {
    if let Some(expected) = verify {
        let actual = xxh3_64(dst);
        if actual != expected {
            return Err(Error::ChecksumMismatch { blob: 0, chunk: 0, expected, actual });
        }
    }
    Ok(())
}

/// Sets the location of an error from [`decode_chunk_into`], for a caller that knows it. Other errors
/// pass through unchanged.
#[must_use]
pub fn locate_error(error: Error, blob: u32, chunk: u32) -> Error {
    match error {
        Error::ChunkCorrupt { detail, .. } => Error::ChunkCorrupt { blob, chunk, detail },
        Error::ChecksumMismatch { expected, actual, .. } => {
            Error::ChecksumMismatch { blob, chunk, expected, actual }
        }
        other => other,
    }
}

/// Whether two byte ranges do not overlap.
fn disjoint(a: &[u8], b: &[u8]) -> bool {
    let (a, b) = (a.as_ptr_range(), b.as_ptr_range());
    a.end <= b.start || b.end <= a.start
}

/// Parses the prologue of `stored` and cuts it into streams: steps 1 and 2 of spec section 6.2, plus
/// the per-stream rules of section 6.1. The first `layout.stream_count()` entries are valid.
fn parse_streams(
    layout: Layout,
    stored: &[u8],
    raw_len: usize,
) -> Result<[StoredStream<'_>; MAX_STREAMS], Error> {
    let stream_count = layout.stream_count();
    let prologue_len = WORD_BYTES * stream_count;
    let (prologue, mut body) = stored.split_at_checked(prologue_len).ok_or_else(|| {
        corrupt(format!("{} bytes cannot hold a {prologue_len} byte prologue", stored.len()))
    })?;
    let (words, _) = prologue.as_chunks::<WORD_BYTES>();
    let declared: u64 = words
        .iter()
        .map(|word| u64::from(u32::from_le_bytes(*word) & STORED_LEN_MASK))
        .sum();
    if declared != body.len() as u64 {
        return Err(corrupt(format!(
            "stream words declare {declared} bytes, the chunk has {} after its prologue",
            body.len()
        )));
    }
    let mut streams = [StoredStream { codec: STREAM_STORE, bytes: &[], raw_len: 0 }; MAX_STREAMS];
    for (k, (slot, word)) in streams.iter_mut().zip(words).enumerate() {
        let word = u32::from_le_bytes(*word);
        let (block, _, len) = stream_span(layout, k);
        let stream_raw_len = raw_len / block * len;
        // A shift of 24 leaves 8 bits, so the conversion cannot fail.
        let codec = (word >> CODEC_SHIFT).to_le_bytes()[0];
        // Lossless: the mask leaves 24 bits.
        let stored_len = (word & STORED_LEN_MASK) as usize;
        match codec {
            STREAM_STORE if stored_len != stream_raw_len => {
                return Err(corrupt(format!(
                    "store stream {k} has {stored_len} bytes, its raw length is {stream_raw_len}"
                )));
            }
            STREAM_ZSTD if stored_len > stream_raw_len => {
                return Err(corrupt(format!(
                    "zstd stream {k} has {stored_len} bytes, over its raw length {stream_raw_len}"
                )));
            }
            STREAM_STORE | STREAM_ZSTD => {}
            other => return Err(Error::UnknownCodec { id: u16::from(other) }),
        }
        // Cannot fail: the sum of the lengths equals `body.len()`, checked above.
        let (bytes, rest) = body.split_at_checked(stored_len).ok_or_else(|| corrupt("short body"))?;
        body = rest;
        *slot = StoredStream { codec, bytes, raw_len: stream_raw_len };
    }
    Ok(streams)
}

/// Decodes one stream into exactly `dst`, which has the stream's raw length.
fn decode_stream(
    stream: &StoredStream<'_>,
    dst: &mut [u8],
    zstd: &mut ZstdContexts,
) -> Result<(), Error> {
    if stream.codec == STREAM_ZSTD {
        zstd.decode_zstd(stream.bytes, dst)
    } else {
        // Lengths are equal: `parse_streams` requires a store stream to have its raw length.
        dst.copy_from_slice(stream.bytes);
        Ok(())
    }
}

/// Route for `whole`: the one stream is the chunk.
fn decode_whole(
    streams: &[StoredStream<'_>],
    dst: &mut [u8],
    zstd: &mut ZstdContexts,
) -> Result<(), Error> {
    match streams {
        [stream] => decode_stream(stream, dst, zstd),
        _ => Err(corrupt("whole layout without exactly one stream")),
    }
}

/// Route for `planes`.
fn decode_planes(
    streams: &[StoredStream<'_>],
    dst: &mut [u8],
    ws: &mut Workspace,
) -> Result<(), Error> {
    let mut compressed = streams.iter().enumerate().filter(|(_, s)| s.codec == STREAM_ZSTD);
    if let (true, Some((plane, stream)), None) = (ws.in_place, compressed.next(), compressed.next())
    {
        let tail = dst.len() - stream.raw_len;
        ws.zstd.decode_zstd(stream.bytes, &mut dst[tail..])?;
        let mut others: [&[u8]; MAX_STREAMS] = [&[]; MAX_STREAMS];
        let stored = streams.iter().filter(|s| s.codec != STREAM_ZSTD);
        for (slot, other) in others.iter_mut().zip(stored) {
            *slot = other.bytes;
        }
        kernels::merge_in_place(ws.isa, plane, &others[..streams.len() - 1], dst);
        return Ok(());
    }
    let mut views: [&[u8]; MAX_STREAMS] = [&[]; MAX_STREAMS];
    decode_to_scratch(streams, &mut ws.scratch, &mut ws.zstd, &mut views)?;
    kernels::merge_planes(ws.isa, &views[..streams.len()], dst);
    Ok(())
}

/// Route for `fields`.
fn decode_fields(
    table: FieldTable,
    streams: &[StoredStream<'_>],
    dst: &mut [u8],
    ws: &mut Workspace,
) -> Result<(), Error> {
    let mut views: [&[u8]; MAX_STREAMS] = [&[]; MAX_STREAMS];
    decode_to_scratch(streams, &mut ws.scratch, &mut ws.zstd, &mut views)?;
    kernels::merge_fields(table, &views[..streams.len()], dst);
    Ok(())
}

/// Decodes the compressed streams into `scratch` and sets `views` to every stream's raw bytes. Stored
/// streams are not copied. `scratch` needs at most the raw length of the chunk.
fn decode_to_scratch<'a>(
    streams: &[StoredStream<'a>],
    scratch: &'a mut Vec<u8>,
    zstd: &mut ZstdContexts,
    views: &mut [&'a [u8]; MAX_STREAMS],
) -> Result<(), Error> {
    let needed = streams.iter().filter(|s| s.codec == STREAM_ZSTD).map(|s| s.raw_len).sum();
    let mut free = sized(scratch, needed);
    for (view, stream) in views.iter_mut().zip(streams) {
        if stream.codec == STREAM_ZSTD {
            let (target, rest) = free.split_at_mut(stream.raw_len);
            zstd.decode_zstd(stream.bytes, target)?;
            *view = target;
            free = rest;
        } else {
            *view = stream.bytes;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::kernels::Isa;
    use super::super::test_support::{
        GUARD_BYTES, Rng, assert_guards, available_isas, guarded, guarded_dst,
    };
    use super::*;
    use crate::chain::{ElementWidth, Step};

    const POLICIES: [StreamPolicy; 4] = [
        StreamPolicy::Store,
        StreamPolicy::Entropy,
        StreamPolicy::EntropyOrStore,
        StreamPolicy::Zstd { level: 3 },
    ];
    const WIDTHS: [ElementWidth; 3] = [ElementWidth::Two, ElementWidth::Four, ElementWidth::Eight];
    const TABLES: [FieldTable; 4] =
        [FieldTable::Q80, FieldTable::Q4K, FieldTable::Q5K, FieldTable::Q6K];

    fn layouts() -> Vec<Layout> {
        let mut all = vec![Layout::Whole];
        all.extend(WIDTHS.map(Layout::Planes));
        all.extend(TABLES.map(Layout::Fields));
        all
    }

    /// Every chain: each layout, bare and after each rotation.
    fn chains() -> Vec<Chain> {
        let mut all = Vec::new();
        for layout in layouts() {
            all.push(Chain::new(&[Step::Layout(layout)]).expect("valid"));
            for width in WIDTHS {
                all.push(Chain::new(&[Step::Rotl1(width), Step::Layout(layout)]).expect("valid"));
            }
        }
        all
    }

    fn planes_chain(width: ElementWidth) -> Chain {
        Chain::new(&[Step::Layout(Layout::Planes(width))]).expect("valid")
    }

    fn whole_chain() -> Chain {
        Chain::new(&[Step::Layout(Layout::Whole)]).expect("valid")
    }

    /// Data with skewed and uniform byte positions, so policies differ per stream.
    fn mixed_data(rng: &mut Rng, len: usize) -> Vec<u8> {
        let mut data = rng.bytes(len);
        for (i, byte) in data.iter_mut().enumerate() {
            if i % 3 == 0 {
                *byte &= 0x03;
            }
        }
        data
    }

    fn encode(raw: &[u8], chain: Chain, policy: StreamPolicy) -> Vec<u8> {
        let mut out = Vec::new();
        let meta = encode_chunk(raw, &chain, &policy, &mut Workspace::new(), &mut out).expect("encode");
        assert_eq!(meta.stored_len as usize, out.len());
        assert_eq!(meta.xxh3_decoded, xxh3_64(raw));
        out
    }

    fn workspace(isa: Isa, in_place: bool) -> Workspace {
        let mut ws = Workspace::new();
        ws.isa = isa;
        ws.in_place = in_place;
        ws
    }

    fn words(stored: &[u8], count: usize) -> Vec<(u8, usize)> {
        stored[..count * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| {
                let word = u32::from_le_bytes(*word);
                (word.to_le_bytes()[3], (word & 0x00FF_FFFF) as usize)
            })
            .collect()
    }

    fn set_word(stored: &mut [u8], k: usize, codec: u8, len: usize) {
        let word = (u32::from(codec) << 24) | u32::try_from(len).expect("small");
        stored[k * 4..k * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }

    /// Checks the routes this machine has: every kernel, in-place on and off, inside a guard band.
    fn assert_all_routes_decode(chain: Chain, stored: &[u8], raw: &[u8], context: &str) {
        for isa in available_isas() {
            for in_place in [true, false] {
                let mut ws = workspace(isa, in_place);
                for misalign in [0, 7] {
                    let mut buffer = guarded(raw.len(), misalign);
                    let dst = guarded_dst(&mut buffer, raw.len(), misalign);
                    decode_chunk_into(&chain, stored, dst, &mut ws, Some(xxh3_64(raw)))
                        .unwrap_or_else(|e| panic!("{context} {isa:?} {in_place}: {e}"));
                    assert_guards(&buffer, raw.len(), misalign, context);
                    let start = GUARD_BYTES + misalign;
                    assert_eq!(&buffer[start..start + raw.len()], raw, "{context} {isa:?}");
                }
            }
        }
    }

    #[test]
    fn every_chain_and_policy_round_trips() {
        let mut rng = Rng(10);
        for chain in chains() {
            let block = chain.block_bytes() as usize;
            for blocks in [0, 1, 2, 3, 37] {
                let raw = mixed_data(&mut rng, block * blocks);
                for policy in POLICIES {
                    let stored = encode(&raw, chain, policy);
                    let context = format!("{chain:?} {policy:?} {blocks}");
                    assert!(stored.len() <= 4 * chain.stream_count() + raw.len(), "{context}");
                    assert_all_routes_decode(chain, &stored, &raw, &context);
                }
            }
        }
    }

    #[test]
    fn large_chunks_round_trip_with_several_compressed_streams() {
        let mut rng = Rng(11);
        for chain in chains() {
            let block = chain.block_bytes() as usize;
            let raw = mixed_data(&mut rng, (300_000 / block + 1) * block);
            for policy in [StreamPolicy::EntropyOrStore, StreamPolicy::Zstd { level: 3 }] {
                let stored = encode(&raw, chain, policy);
                assert_all_routes_decode(chain, &stored, &raw, "large");
            }
        }
    }

    #[test]
    fn the_largest_chunk_round_trips_and_its_stored_length_fits_the_word() {
        let raw = Rng(12).bytes(MAX_CHUNK_BYTES);
        let chain = whole_chain();
        let stored = encode(&raw, chain, StreamPolicy::Store);
        assert_eq!(words(&stored, 1), [(STREAM_STORE, MAX_CHUNK_BYTES)]);
        let mut dst = vec![0; MAX_CHUNK_BYTES];
        decode_chunk_into(&chain, &stored, &mut dst, &mut Workspace::new(), Some(xxh3_64(&raw)))
            .expect("decode");
        assert_eq!(dst, raw);
    }

    /// The window of any level stays within the format at the largest chunk.
    #[test]
    fn the_highest_level_stays_within_the_window_limit() {
        let raw = vec![7; MAX_CHUNK_BYTES];
        let chain = whole_chain();
        let stored = encode(&raw, chain, StreamPolicy::Zstd { level: 22 });
        assert_eq!(words(&stored, 1)[0].0, STREAM_ZSTD);
        assert_all_routes_decode(chain, &stored, &raw, "level 22");
    }

    /// One compressed plane at every index: the in-place route, against the scratch route.
    #[test]
    fn one_compressed_plane_at_every_index_takes_both_routes_with_identical_bytes() {
        let mut rng = Rng(13);
        for width in WIDTHS {
            let w = width.bytes() as usize;
            let chain = planes_chain(width);
            for plane in 0..w {
                for n in [0, 1, 31, 32, 33, 64, 100, 4096, 70_001] {
                    let mut raw = rng.bytes(n * w);
                    for element in raw.chunks_exact_mut(w) {
                        element[plane] &= 0x01;
                    }
                    let stored = encode(&raw, chain, StreamPolicy::EntropyOrStore);
                    if n >= 4096 {
                        let compressed: Vec<usize> = words(&stored, w)
                            .iter()
                            .enumerate()
                            .filter(|(_, (codec, _))| *codec == STREAM_ZSTD)
                            .map(|(k, _)| k)
                            .collect();
                        assert_eq!(compressed, [plane], "the fixture must compress one plane");
                    }
                    assert_all_routes_decode(chain, &stored, &raw, "one plane");
                }
            }
        }
    }

    #[test]
    fn rotation_applies_after_an_in_place_merge() {
        let mut rng = Rng(14);
        let chain = Chain::new(&[
            Step::Rotl1(ElementWidth::Two),
            Step::Layout(Layout::Planes(ElementWidth::Two)),
        ])
        .expect("valid");
        // Bytes that have a skewed high plane once rotated: small exponent-like values.
        let mut raw = rng.bytes(2 * 50_000);
        for element in raw.as_chunks_mut::<2>().0 {
            element[1] &= 0x01;
            element[0] &= 0x7F;
        }
        let stored = encode(&raw, chain, StreamPolicy::EntropyOrStore);
        assert!(words(&stored, 2).iter().any(|(codec, _)| *codec == STREAM_ZSTD));
        assert_all_routes_decode(chain, &stored, &raw, "rotated");
    }

    #[test]
    fn encoded_bytes_do_not_depend_on_workspace_history_or_kernels() {
        let mut rng = Rng(15);
        let first = mixed_data(&mut rng, 40_000);
        let second = mixed_data(&mut rng, 80_000);
        let chain = Chain::new(&[
            Step::Rotl1(ElementWidth::Four),
            Step::Layout(Layout::Planes(ElementWidth::Four)),
        ])
        .expect("valid");
        let expected = encode(&second, chain, StreamPolicy::EntropyOrStore);
        for isa in available_isas() {
            let mut ws = workspace(isa, true);
            let mut out = Vec::new();
            for raw in [&first, &second] {
                out.clear();
                encode_chunk(raw, &chain, &StreamPolicy::EntropyOrStore, &mut ws, &mut out)
                    .expect("encode");
            }
            assert_eq!(out, expected, "{isa:?}");
        }
    }

    #[test]
    fn stored_form_matches_the_spec_byte_for_byte() {
        // Elements 0x8001, 0x0102 rotate to 0x0003, 0x0204, and split into planes 03 04 and 00 02.
        let chain = Chain::new(&[
            Step::Rotl1(ElementWidth::Two),
            Step::Layout(Layout::Planes(ElementWidth::Two)),
        ])
        .expect("valid");
        let stored = encode(&[0x01, 0x80, 0x02, 0x01], chain, StreamPolicy::Store);
        assert_eq!(stored, [2, 0, 0, 0, 2, 0, 0, 0, 0x03, 0x04, 0x00, 0x02]);

        // Q8_0 streams: byte 0, byte 1, then 32 bytes.
        let raw: Vec<u8> = (0..34).collect();
        let chain = Chain::new(&[Step::Layout(Layout::Fields(FieldTable::Q80))]).expect("valid");
        let stored = encode(&raw, chain, StreamPolicy::Store);
        assert_eq!(&stored[..12], [1, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0]);
        assert_eq!(&stored[12..], raw);
    }

    #[test]
    fn a_compressed_stream_word_has_codec_one_in_the_high_byte() {
        let raw = vec![0; 8192];
        let stored = encode(&raw, whole_chain(), StreamPolicy::Entropy);
        let body = stored.len() - 4;
        assert_eq!(words(&stored, 1), [(STREAM_ZSTD, body)]);
        assert_eq!(stored[3], 1);
    }

    #[test]
    fn output_is_appended_and_kept_on_error() {
        let chain = planes_chain(ElementWidth::Two);
        let mut ws = Workspace::new();
        let mut out = vec![9, 9, 9];
        let raw = [1, 2, 3, 4];
        let meta = encode_chunk(&raw, &chain, &StreamPolicy::Store, &mut ws, &mut out).expect("encode");
        assert_eq!(meta.stored_len, 12);
        assert_eq!(&out[..3], [9, 9, 9]);
        assert_eq!(&out[3..], encode(&raw, chain, StreamPolicy::Store));
        let before = out.clone();
        encode_chunk(&[1, 2, 3], &chain, &StreamPolicy::Store, &mut ws, &mut out)
            .expect_err("not a multiple of the block");
        assert_eq!(out, before);
    }

    #[test]
    fn encode_rejects_bad_lengths() {
        let mut ws = Workspace::new();
        let mut out = Vec::new();
        let oversized = vec![0; MAX_CHUNK_BYTES + 1];
        let err = encode_chunk(&oversized, &whole_chain(), &StreamPolicy::Store, &mut ws, &mut out)
            .expect_err("over the limit");
        assert!(matches!(err, Error::BudgetExceeded { .. }), "{err:?}");
        let err = encode_chunk(
            &[0; 3],
            &planes_chain(ElementWidth::Four),
            &StreamPolicy::Store,
            &mut ws,
            &mut out,
        )
        .expect_err("not a multiple");
        assert!(matches!(err, Error::ChunkCorrupt { .. }), "{err:?}");
    }

    #[test]
    fn decode_rejects_a_destination_that_cannot_be_a_raw_length() {
        let chain = planes_chain(ElementWidth::Four);
        let stored = encode(&[0; 8], chain, StreamPolicy::Store);
        let mut ws = Workspace::new();
        let err = decode_chunk_into(&chain, &stored, &mut [0; 7], &mut ws, None).expect_err("size");
        assert!(matches!(err, Error::DestinationSize { expected: 4, actual: 7 }), "{err:?}");
        let mut oversized = vec![0; MAX_CHUNK_BYTES + 4];
        let err = decode_chunk_into(&chain, &stored, &mut oversized, &mut ws, None).expect_err("size");
        assert!(matches!(err, Error::DestinationSize { .. }), "{err:?}");
        // A valid length, but not this chunk's: the store streams do not match it.
        let err = decode_chunk_into(&chain, &stored, &mut [0; 12], &mut ws, None).expect_err("size");
        assert!(matches!(err, Error::ChunkCorrupt { .. }), "{err:?}");
    }

    fn corrupted(chain: Chain, stored: &[u8], raw_len: usize) -> Error {
        let mut dst = vec![0; raw_len];
        decode_chunk_into(&chain, stored, &mut dst, &mut Workspace::new(), None)
            .expect_err("must be rejected")
    }

    #[test]
    fn stream_word_sum_mismatch_is_rejected() {
        let chain = planes_chain(ElementWidth::Two);
        let raw = [1, 2, 3, 4, 5, 6, 7, 8];
        let stored = encode(&raw, chain, StreamPolicy::Store);

        let mut longer = stored.clone();
        longer.push(0);
        assert!(matches!(corrupted(chain, &longer, 8), Error::ChunkCorrupt { .. }));

        let mut grown = stored.clone();
        set_word(&mut grown, 1, STREAM_STORE, 5);
        let err = corrupted(chain, &grown, 8);
        assert!(err.to_string().contains("declare"), "{err}");

        let mut shrunk = stored;
        set_word(&mut shrunk, 0, STREAM_STORE, 3);
        let err = corrupted(chain, &shrunk, 8);
        assert!(err.to_string().contains("declare"), "{err}");
    }

    #[test]
    fn a_sum_that_matches_with_wrong_per_stream_lengths_is_rejected() {
        let chain = planes_chain(ElementWidth::Two);
        let stored = encode(&[1, 2, 3, 4, 5, 6, 7, 8], chain, StreamPolicy::Store);
        let mut swapped = stored.clone();
        set_word(&mut swapped, 0, STREAM_STORE, 5);
        set_word(&mut swapped, 1, STREAM_STORE, 3);
        let err = corrupted(chain, &swapped, 8);
        assert!(err.to_string().contains("raw length"), "{err}");

        let mut over = stored;
        set_word(&mut over, 0, STREAM_ZSTD, 5);
        set_word(&mut over, 1, STREAM_ZSTD, 3);
        let err = corrupted(chain, &over, 8);
        assert!(err.to_string().contains("over its raw length"), "{err}");
    }

    #[test]
    fn unknown_stream_codecs_are_rejected_by_id() {
        let chain = whole_chain();
        let stored = encode(&[1, 2, 3], chain, StreamPolicy::Store);
        for codec in [2_u8, 3, 0x80, 0xFF] {
            let mut bad = stored.clone();
            set_word(&mut bad, 0, codec, 3);
            let err = corrupted(chain, &bad, 3);
            assert!(matches!(err, Error::UnknownCodec { id } if id == u16::from(codec)), "{err:?}");
        }
    }

    #[test]
    fn every_truncation_is_rejected() {
        let mut rng = Rng(16);
        for chain in chains() {
            let block = chain.block_bytes() as usize;
            let raw = mixed_data(&mut rng, block * (1 + 1024 / block));
            for policy in POLICIES {
                let stored = encode(&raw, chain, policy);
                let hash = Some(xxh3_64(&raw));
                let step = stored.len() / 97 + 1;
                let near_end = stored.len().saturating_sub(9)..stored.len();
                for len in (0..stored.len()).step_by(step).chain(near_end) {
                    let mut dst = vec![0; raw.len()];
                    let result =
                        decode_chunk_into(&chain, &stored[..len], &mut dst, &mut Workspace::new(), hash);
                    assert!(result.is_err(), "{chain:?} {policy:?} truncated to {len}");
                }
            }
        }
    }

    /// Byte ranges of `stored` that hold zstd frames.
    fn zstd_ranges(stored: &[u8], stream_count: usize) -> Vec<std::ops::Range<usize>> {
        let mut position = 4 * stream_count;
        let mut ranges = Vec::new();
        for (codec, len) in words(stored, stream_count) {
            if codec == STREAM_ZSTD {
                ranges.push(position..position + len);
            }
            position += len;
        }
        ranges
    }

    /// With the hash on, every single bit flip in a chunk fails, at every position, with one
    /// exception. zstd entropy tables have bits with no effect on the output, such as padding, and
    /// the format adds no checksum over them. A flip there must decode to the original bytes, which
    /// is what the hash proves: such a stored byte is damaged but the data is not.
    #[test]
    fn every_single_bit_flip_is_rejected_or_harmless_when_verifying() {
        let mut rng = Rng(17);
        let chains = [
            whole_chain(),
            planes_chain(ElementWidth::Two),
            Chain::new(&[
                Step::Rotl1(ElementWidth::Four),
                Step::Layout(Layout::Planes(ElementWidth::Four)),
            ])
            .expect("valid"),
            Chain::new(&[Step::Layout(Layout::Fields(FieldTable::Q4K))]).expect("valid"),
        ];
        let (mut flips, mut harmless) = (0_usize, 0_usize);
        for chain in chains {
            let raw = mixed_data(&mut rng, 1440);
            for policy in [StreamPolicy::Entropy, StreamPolicy::EntropyOrStore, StreamPolicy::Store] {
                let stored = encode(&raw, chain, policy);
                let frames = zstd_ranges(&stored, chain.stream_count());
                let hash = Some(xxh3_64(&raw));
                let mut ws = Workspace::new();
                let mut dst = vec![0; raw.len()];
                for bit in 0..stored.len() * 8 {
                    let mut bad = stored.clone();
                    bad[bit / 8] ^= 1 << (bit % 8);
                    flips += 1;
                    if decode_chunk_into(&chain, &bad, &mut dst, &mut ws, hash).is_ok() {
                        harmless += 1;
                        let in_frame = frames.iter().any(|range| range.contains(&(bit / 8)));
                        assert!(in_frame, "{chain:?} {policy:?} bit {bit} is outside any frame");
                        assert_eq!(dst, raw, "{chain:?} {policy:?} bit {bit}");
                    }
                }
            }
        }
        assert!(harmless * 200 < flips, "{harmless} of {flips} flips went unnoticed");
    }

    #[test]
    fn a_wrong_hash_reports_both_values() {
        let chain = whole_chain();
        let stored = encode(&[1, 2, 3], chain, StreamPolicy::Store);
        let mut dst = [0; 3];
        let err = decode_chunk_into(&chain, &stored, &mut dst, &mut Workspace::new(), Some(5))
            .expect_err("wrong hash");
        assert!(
            matches!(err, Error::ChecksumMismatch { expected: 5, actual, .. } if actual == xxh3_64(&[1, 2, 3])),
            "{err:?}"
        );
    }

    /// Without the hash nothing proves the content, so a flip may decode. It must still not panic,
    /// not leave `dst`, and on `Ok` the length rules hold by construction. Random damage over all
    /// chains, on the workspace that handled earlier damage.
    #[test]
    fn hostile_bytes_never_panic_or_escape() {
        let mut rng = Rng(18);
        for chain in chains() {
            let block = chain.block_bytes() as usize;
            let raw = mixed_data(&mut rng, block * (1 + 4096 / block));
            let stored = encode(&raw, chain, StreamPolicy::EntropyOrStore);
            for isa in available_isas() {
                let mut ws = workspace(isa, true);
                let mut buffer = guarded(raw.len(), 0);
                for round in 0..300 {
                    let mut bad = stored.clone();
                    match round % 3 {
                        0 => {
                            for _ in 0..=round % 16 {
                                let position = rng.below(bad.len());
                                bad[position] ^= rng.byte();
                            }
                        }
                        1 => {
                            // Damage aimed at the prologue.
                            let position = rng.below(4 * chain.stream_count());
                            bad[position] = rng.byte();
                        }
                        _ => {
                            let keep = rng.below(bad.len());
                            bad.truncate(keep);
                            let extra = rng.below(64);
                            let tail = rng.bytes(extra);
                            bad.extend_from_slice(&tail);
                        }
                    }
                    let dst = guarded_dst(&mut buffer, raw.len(), 0);
                    let result = decode_chunk_into(&chain, &bad, dst, &mut ws, None);
                    if let Err(error) = result {
                        assert!(
                            matches!(error, Error::ChunkCorrupt { .. } | Error::UnknownCodec { .. }),
                            "{error:?}"
                        );
                    }
                    assert_guards(&buffer, raw.len(), 0, "hostile");
                }
                // The workspace is still good after all of that.
                let dst = guarded_dst(&mut buffer, raw.len(), 0);
                decode_chunk_into(&chain, &stored, dst, &mut ws, Some(xxh3_64(&raw))).expect("clean");
            }
        }
    }

    #[test]
    fn plain_garbage_is_rejected() {
        let mut rng = Rng(19);
        for chain in chains() {
            let block = chain.block_bytes() as usize;
            for size in [0, 1, 3, 4, 31, 32, 33, 4097] {
                let garbage = rng.bytes(size);
                let mut dst = vec![0; block * 3];
                let result =
                    decode_chunk_into(&chain, &garbage, &mut dst, &mut Workspace::new(), None);
                assert!(result.is_err(), "{chain:?} {size}");
            }
        }
    }

    #[test]
    fn scratch_stays_within_the_chunk_whatever_the_file_says() {
        let chain = planes_chain(ElementWidth::Eight);
        let raw = mixed_data(&mut Rng(20), 8 * 4096);
        let stored = encode(&raw, chain, StreamPolicy::Entropy);
        let mut ws = workspace(available_isas()[0], false);
        let mut dst = vec![0; raw.len()];
        decode_chunk_into(&chain, &stored, &mut dst, &mut ws, None).expect("decode");
        assert!(ws.scratch.len() <= raw.len(), "{}", ws.scratch.len());
    }

    #[test]
    fn in_place_uses_no_scratch() {
        let chain = planes_chain(ElementWidth::Two);
        let mut raw = Rng(21).bytes(2 * 8192);
        for element in raw.as_chunks_mut::<2>().0 {
            element[1] &= 0x01;
        }
        let stored = encode(&raw, chain, StreamPolicy::EntropyOrStore);
        let mut ws = Workspace::new();
        let mut dst = vec![0; raw.len()];
        decode_chunk_into(&chain, &stored, &mut dst, &mut ws, None).expect("decode");
        assert_eq!(ws.scratch.len(), 0);
        assert_eq!(dst, raw);
    }

    #[test]
    fn locate_error_fills_only_chunk_errors() {
        let located = locate_error(corrupt("x"), 3, 4);
        assert!(matches!(located, Error::ChunkCorrupt { blob: 3, chunk: 4, .. }), "{located:?}");
        let located = locate_error(
            Error::ChecksumMismatch { blob: 0, chunk: 0, expected: 1, actual: 2 },
            5,
            6,
        );
        assert!(
            matches!(
                located,
                Error::ChecksumMismatch { blob: 5, chunk: 6, expected: 1, actual: 2 }
            ),
            "{located:?}"
        );
        let other = locate_error(Error::UnknownCodec { id: 2 }, 5, 6);
        assert!(matches!(other, Error::UnknownCodec { id: 2 }), "{other:?}");
    }

    #[test]
    fn disjoint_detects_overlap() {
        let bytes = [0; 8];
        assert!(disjoint(&bytes[..4], &bytes[4..]));
        assert!(!disjoint(&bytes[..5], &bytes[4..]));
        assert!(!disjoint(&bytes, &bytes));
    }

    /// `chain` with `delta_xor` of `width` in front.
    fn with_delta(chain: Chain, width: ElementWidth) -> Chain {
        let steps: Vec<Step> = [Step::Delta(crate::chain::DeltaOp::Xor(width))]
            .into_iter()
            .chain(chain.steps())
            .collect();
        Chain::new(&steps).expect("a delta step first is valid")
    }

    fn encode_delta(raw: &[u8], base: &[u8], chain: Chain, policy: StreamPolicy) -> Vec<u8> {
        let mut out = Vec::new();
        let mut ws = Workspace::new();
        let meta = encode_delta_chunk(raw, base, &chain, &policy, &mut ws, &mut out).expect("encode");
        assert_eq!(meta.stored_len as usize, out.len());
        assert_eq!(meta.xxh3_decoded, xxh3_64(raw));
        out
    }

    /// A target that differs from `base` in a few low bits, as a finetune differs from its base.
    fn finetuned(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
        let mut target = base.to_vec();
        for byte in target.iter_mut().step_by(5) {
            *byte ^= rng.byte() & 0x03;
        }
        target
    }

    /// [`assert_all_routes_decode`] for a delta chunk.
    fn assert_all_routes_decode_delta(chain: Chain, stored: &[u8], base: &[u8], raw: &[u8]) {
        for isa in available_isas() {
            for in_place in [true, false] {
                let mut ws = workspace(isa, in_place);
                for misalign in [0, 7] {
                    let mut buffer = guarded(raw.len(), misalign);
                    let dst = guarded_dst(&mut buffer, raw.len(), misalign);
                    decode_delta_chunk_into(&chain, stored, base, dst, &mut ws, Some(xxh3_64(raw)))
                        .unwrap_or_else(|e| panic!("{chain:?} {isa:?} {in_place}: {e}"));
                    assert_guards(&buffer, raw.len(), misalign, "delta");
                    let start = GUARD_BYTES + misalign;
                    assert_eq!(&buffer[start..start + raw.len()], raw, "{chain:?} {isa:?}");
                }
            }
        }
    }

    #[test]
    fn every_chain_with_a_delta_step_round_trips() {
        let mut rng = Rng(40);
        for plain in chains() {
            for width in WIDTHS {
                let chain = with_delta(plain, width);
                let block = chain.block_bytes() as usize;
                for blocks in [0, 1, 3, 37] {
                    let base = mixed_data(&mut rng, block * blocks);
                    let raw = finetuned(&mut rng, &base);
                    for policy in POLICIES {
                        let stored = encode_delta(&raw, &base, chain, policy);
                        assert_all_routes_decode_delta(chain, &stored, &base, &raw);
                    }
                }
            }
        }
    }

    #[test]
    fn a_delta_chunk_stores_the_residual_through_the_rest_of_the_chain() {
        let mut rng = Rng(41);
        let chain = with_delta(planes_chain(ElementWidth::Two), ElementWidth::Two);
        let base = rng.bytes(4096);
        let raw = finetuned(&mut rng, &base);
        let mut residual_bytes = vec![0; raw.len()];
        let op = crate::chain::DeltaOp::Xor(ElementWidth::Two);
        delta::residual(op, &raw, &base, &mut residual_bytes).expect("equal lengths");
        let policy = StreamPolicy::EntropyOrStore;
        let stored = encode_delta(&raw, &base, chain, policy);
        assert_eq!(stored, encode(&residual_bytes, chain.residual(), policy));
        // A residual of sparse low bits compresses far below the target alone.
        assert!(stored.len() < encode(&raw, chain.residual(), policy).len() / 2);
    }

    #[test]
    fn a_delta_chunk_with_one_compressed_plane_decodes_in_place() {
        let chain = with_delta(planes_chain(ElementWidth::Two), ElementWidth::Two);
        let mut rng = Rng(42);
        let base = rng.bytes(2 * 8192);
        let mut raw = base.clone();
        // The low byte of the residual is noise, the high byte one random bit: one compressible plane.
        for element in raw.as_chunks_mut::<2>().0 {
            element[0] ^= rng.byte();
            element[1] ^= rng.byte() & 0x01;
        }
        let stored = encode_delta(&raw, &base, chain, StreamPolicy::EntropyOrStore);
        let compressed = words(&stored, 2).iter().filter(|(codec, _)| *codec == STREAM_ZSTD).count();
        assert_eq!(compressed, 1, "the fixture must compress one plane");
        let mut ws = Workspace::new();
        let mut dst = vec![0; raw.len()];
        decode_delta_chunk_into(&chain, &stored, &base, &mut dst, &mut ws, Some(xxh3_64(&raw)))
            .expect("decode");
        assert_eq!(ws.scratch.len(), 0);
        assert_eq!(dst, raw);
    }

    #[test]
    fn a_wrong_base_is_a_checksum_mismatch_or_ignored_when_not_verifying() {
        let chain = with_delta(whole_chain(), ElementWidth::Four);
        let mut rng = Rng(43);
        let base = rng.bytes(256);
        let raw = finetuned(&mut rng, &base);
        let stored = encode_delta(&raw, &base, chain, StreamPolicy::Zstd { level: 3 });
        let mut wrong = base.clone();
        wrong[100] ^= 0x10;
        let mut dst = vec![0; raw.len()];
        let mut ws = Workspace::new();
        let error = decode_delta_chunk_into(&chain, &stored, &wrong, &mut dst, &mut ws, Some(xxh3_64(&raw)))
            .expect_err("a wrong base");
        assert!(matches!(error, Error::ChecksumMismatch { .. }), "{error:?}");
        decode_delta_chunk_into(&chain, &stored, &wrong, &mut dst, &mut ws, None).expect("lengths hold");
        assert_eq!(dst[100], raw[100] ^ 0x10);
    }

    #[test]
    fn delta_lengths_are_checked_before_anything_is_written() {
        let chain = with_delta(whole_chain(), ElementWidth::Four);
        let mut ws = Workspace::new();
        let mut out = vec![9];
        let error = encode_delta_chunk(&[1; 8], &[1; 4], &chain, &StreamPolicy::Store, &mut ws, &mut out)
            .expect_err("a short base");
        assert!(matches!(error, Error::DestinationSize { expected: 8, actual: 4 }), "{error:?}");
        let error = encode_delta_chunk(&[1; 6], &[1; 6], &chain, &StreamPolicy::Store, &mut ws, &mut out)
            .expect_err("not a multiple of the delta width");
        assert!(matches!(error, Error::ChunkCorrupt { .. }), "{error:?}");
        assert_eq!(out, [9]);
        let stored = encode_delta(&[1; 8], &[2; 8], chain, StreamPolicy::Store);
        let mut dst = [0; 8];
        let error = decode_delta_chunk_into(&chain, &stored, &[2; 12], &mut dst, &mut ws, None)
            .expect_err("a long base");
        assert!(matches!(error, Error::DestinationSize { expected: 8, actual: 12 }), "{error:?}");
        let error = decode_delta_chunk_into(&chain, &stored, &[2; 6], &mut dst[..6], &mut ws, None)
            .expect_err("not a multiple of the delta width");
        assert!(matches!(error, Error::DestinationSize { expected: 4, actual: 6 }), "{error:?}");
        assert_eq!(dst, [0; 8]);
    }

    /// Expects the refusal of a chain given to the wrong entry point.
    fn wrong_entry_point<T: std::fmt::Debug>(result: Result<T, Error>) {
        let error = result.expect_err("the wrong entry point");
        assert!(matches!(error, Error::ChunkCorrupt { .. }), "{error:?}");
    }

    #[test]
    fn plain_and_delta_entry_points_refuse_each_other_s_chains() {
        let plain = whole_chain();
        let chain = with_delta(plain, ElementWidth::Two);
        let mut ws = Workspace::new();
        let mut out = Vec::new();
        wrong_entry_point(encode_chunk(&[1; 4], &chain, &StreamPolicy::Store, &mut ws, &mut out));
        let policy = StreamPolicy::Store;
        wrong_entry_point(encode_delta_chunk(&[1; 4], &[1; 4], &plain, &policy, &mut ws, &mut out));
        assert_eq!(out, []);
        let stored = encode(&[1; 4], plain, StreamPolicy::Store);
        let mut dst = [0; 4];
        wrong_entry_point(decode_chunk_into(&chain, &stored, &mut dst, &mut ws, None));
        wrong_entry_point(decode_delta_chunk_into(&plain, &stored, &[0; 4], &mut dst, &mut ws, None));
    }

    #[test]
    fn hostile_delta_chunks_never_panic() {
        let chain = with_delta(planes_chain(ElementWidth::Four), ElementWidth::Four);
        let mut rng = Rng(44);
        let base = rng.bytes(4096);
        let raw = finetuned(&mut rng, &base);
        let stored = encode_delta(&raw, &base, chain, StreamPolicy::EntropyOrStore);
        let mut ws = Workspace::new();
        let mut dst = vec![0; raw.len()];
        for _ in 0..2000 {
            let mut bytes = stored.clone();
            let at = rng.below(bytes.len());
            bytes[at] ^= rng.byte() | 1;
            let cut = rng.below(bytes.len() + 1);
            for verify in [Some(xxh3_64(&raw)), None] {
                // Any outcome but a panic, and a verified `Ok` is the right bytes.
                if decode_delta_chunk_into(&chain, &bytes[..cut], &base, &mut dst, &mut ws, verify).is_ok()
                    && verify.is_some()
                {
                    assert_eq!(dst, raw);
                }
            }
        }
    }
}
