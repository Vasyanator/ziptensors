//! File: codec/stream.rs
//! Purpose: one stream of a chunk: choosing store or zstd when encoding, and decoding to an exact
//! length.
//!
//! Main responsibilities: the writer's per-stream decision, the zstd parameters of spec section 11,
//! and the frame checks of spec section 6.1.
//!
//! Key structures: [`StreamPolicy`], [`ZstdContexts`].
//!
//! Key functions: [`ZstdContexts::compress`], [`ZstdContexts::decode_zstd`], [`corrupt`].
//!
//! Implementation notes: decoding is one-pass into a buffer of the exact raw length, so zstd cannot
//! write past it. `WindowLogMax` limits only zstd's streaming API, where a window is allocated, and
//! has no effect here, so [`check_frame`] enforces the window limit from the frame header itself. It
//! also refuses anything else section 6.1 excludes. The checks run on the file's bytes before zstd.

use zstd::bulk::{Compressor, Decompressor};
use zstd::zstd_safe::{self, CParameter, DParameter, Strategy};

use crate::chain::{STREAM_STORE, STREAM_ZSTD};
use crate::error::Error;

/// Compression level of the entropy parameters.
const ENTROPY_LEVEL: i32 = 1;
/// Largest zstd window a stream may declare, as a log: spec section 6.1.
pub(super) const WINDOW_LOG_MAX: u32 = 24;
/// A stream is stored when it would keep more than this many thousandths of its size: 0.5% gain.
const KEEP_PER_MILLE: u64 = 995;
/// First bytes of a zstd frame.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// How a writer codes each stream of a chunk: the cell of spec section 11's table it is filling.
/// A reader cannot tell these apart, it sees only a stream codec per stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPolicy {
    /// Every stream is stored.
    Store,
    /// The `h` parameters, kept whenever the frame is smaller than the raw stream.
    Entropy,
    /// The `h` parameters, stored unless the gain is at least 0.5%, because a stored stream costs
    /// nothing to decode. This is "per-stream store or h".
    EntropyOrStore,
    /// zstd at this level, with zstd's own parameters for it, kept whenever the frame is smaller than
    /// the raw stream. Any level is valid: zstd sizes the window to the input, and an input of at most
    /// `MAX_CHUNK_BYTES` keeps it under the format's limit.
    Zstd {
        /// zstd compression level.
        level: i32,
    },
}

/// Returns the error for a chunk that cannot be decoded. The location is filled in by
/// [`super::locate_error`], since this layer does not know it.
pub(super) fn corrupt(detail: impl Into<String>) -> Error {
    Error::ChunkCorrupt { blob: 0, chunk: 0, detail: detail.into() }
}

/// Maps a failure outside chunk decoding, such as creating a context.
fn zstd_error(source: &std::io::Error) -> Error {
    Error::Zstd { detail: source.to_string() }
}

/// zstd contexts, created on first use and reused, so setup is not paid per chunk.
#[derive(Default)]
pub(super) struct ZstdContexts {
    decompressor: Option<Decompressor<'static>>,
    entropy: Option<Compressor<'static>>,
    level: Option<(i32, Compressor<'static>)>,
}

impl std::fmt::Debug for ZstdContexts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZstdContexts")
            .field("decompressor", &self.decompressor.is_some())
            .field("entropy", &self.entropy.is_some())
            .field("level", &self.level.as_ref().map(|(level, _)| *level))
            .finish()
    }
}

impl ZstdContexts {
    /// Compresses `raw` as `policy` says into `compressed`, replacing its contents. Returns whether
    /// the frame should be kept; `false` means the stream is stored and `compressed` is meaningless.
    ///
    /// # Errors
    /// Returns [`Error::Zstd`] if zstd fails.
    pub(super) fn compress(
        &mut self,
        policy: StreamPolicy,
        raw: &[u8],
        compressed: &mut Vec<u8>,
    ) -> Result<bool, Error> {
        // An empty stream has no raw bytes to gain on; its frame could only be larger.
        if policy == StreamPolicy::Store || raw.is_empty() {
            return Ok(false);
        }
        compressed.clear();
        // Exact growth, and the old buffer freed first: `reserve` may double, and would hold both for
        // the move. Workspaces retain this buffer, and the writer's memory bound counts on its size.
        let bound = zstd_safe::compress_bound(raw.len());
        if compressed.capacity() < bound {
            *compressed = Vec::new();
            compressed.reserve_exact(bound);
        }
        let context = self.compressor(policy)?;
        let len = context.compress_to_buffer(raw, compressed).map_err(|e| zstd_error(&e))?;
        Ok(keeps_frame(policy, len, raw.len()))
    }

    /// Returns the compressor of `policy`, creating it if needed. The caller has excluded
    /// [`StreamPolicy::Store`], which maps to the entropy context only so this stays total.
    fn compressor(&mut self, policy: StreamPolicy) -> Result<&mut Compressor<'static>, Error> {
        if let StreamPolicy::Zstd { level } = policy {
            if self.level.as_ref().is_none_or(|(current, _)| *current != level) {
                self.level = Some((level, new_compressor(level)?));
            }
            if let Some((_, context)) = &mut self.level {
                return Ok(context);
            }
        }
        if self.entropy.is_none() {
            self.entropy = Some(new_entropy_compressor()?);
        }
        self.entropy.as_mut().ok_or_else(|| Error::Zstd { detail: "no context".to_owned() })
    }

    /// Decodes the zstd stream `src` into exactly `dst`.
    ///
    /// # Errors
    /// Returns [`Error::ChunkCorrupt`] if the frame breaks spec section 6.1 or does not decode to
    /// exactly `dst.len()` bytes, and [`Error::Zstd`] if a context cannot be created.
    pub(super) fn decode_zstd(&mut self, src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
        check_frame(src, dst.len())?;
        let context = match &mut self.decompressor {
            Some(context) => context,
            empty => empty.insert(new_decompressor()?),
        };
        let written = context
            .decompress_to_buffer(src, dst)
            .map_err(|e| corrupt(format!("zstd stream: {e}")))?;
        if written != dst.len() {
            return Err(corrupt(format!(
                "zstd stream decoded to {written} bytes, expected {}",
                dst.len()
            )));
        }
        Ok(())
    }
}

/// Whether a frame of `frame_len` bytes is worth keeping over `raw_len` stored bytes.
fn keeps_frame(policy: StreamPolicy, frame_len: usize, raw_len: usize) -> bool {
    match policy {
        StreamPolicy::Store => false,
        StreamPolicy::Entropy | StreamPolicy::Zstd { .. } => frame_len < raw_len,
        // `usize` to `u64` is lossless, and lengths are below 2^24 here, so products cannot wrap.
        StreamPolicy::EntropyOrStore => frame_len as u64 * 1000 <= raw_len as u64 * KEEP_PER_MILLE,
    }
}

/// Creates a compressor at `level` that writes the frames spec section 6.1 asks for.
fn new_compressor(level: i32) -> Result<Compressor<'static>, Error> {
    let mut context = Compressor::new(level).map_err(|e| zstd_error(&e))?;
    for parameter in [
        CParameter::ContentSizeFlag(true),
        CParameter::ChecksumFlag(false),
        CParameter::DictIdFlag(false),
    ] {
        context.set_parameter(parameter).map_err(|e| zstd_error(&e))?;
    }
    Ok(context)
}

/// Creates a compressor with the `h` parameters of spec section 11.
fn new_entropy_compressor() -> Result<Compressor<'static>, Error> {
    let mut context = new_compressor(ENTROPY_LEVEL)?;
    for parameter in [
        CParameter::WindowLog(17),
        CParameter::HashLog(6),
        CParameter::ChainLog(6),
        CParameter::SearchLog(1),
        CParameter::MinMatch(7),
        CParameter::TargetLength(0),
        CParameter::Strategy(Strategy::ZSTD_fast),
    ] {
        context.set_parameter(parameter).map_err(|e| zstd_error(&e))?;
    }
    Ok(context)
}

/// Creates the decompressor. `WindowLogMax` is set as the format's limit for any future streaming
/// use; one-pass decoding ignores it, which is why [`check_frame`] exists.
fn new_decompressor() -> Result<Decompressor<'static>, Error> {
    let mut context = Decompressor::new().map_err(|e| zstd_error(&e))?;
    context.set_parameter(DParameter::WindowLogMax(WINDOW_LOG_MAX)).map_err(|e| zstd_error(&e))?;
    Ok(context)
}

/// Checks that `src` is exactly one zstd frame of the form spec section 6.1 allows: content size flag
/// set and equal to `raw_len`, no checksum, no dictionary, window of at most 2^24 bytes.
///
/// Frame layout: RFC 8878 section 3.1.1. Parsed here because `ZSTD_getFrameHeader` is outside zstd's
/// stable API, which is all this crate links.
///
/// # Errors
/// Returns [`Error::ChunkCorrupt`] naming the first rule broken.
fn check_frame(src: &[u8], raw_len: usize) -> Result<(), Error> {
    let [m0, m1, m2, m3, descriptor, rest @ ..] = src else {
        return Err(corrupt("zstd stream is too short for a frame header"));
    };
    if [*m0, *m1, *m2, *m3] != ZSTD_MAGIC {
        return Err(corrupt("zstd stream does not start with a frame"));
    }
    let content_size_flag = descriptor >> 6;
    let single_segment = descriptor & 0x20 != 0;
    // Bit 4 is "unused" and zstd ignores it, but no writer sets it, and accepting it would let one
    // stored byte change with no effect on the output.
    if descriptor & 0x18 != 0 {
        return Err(corrupt("zstd frame sets a reserved bit"));
    }
    if descriptor & 0x04 != 0 {
        return Err(corrupt("zstd frame has a checksum"));
    }
    if descriptor & 0x03 != 0 {
        return Err(corrupt("zstd frame names a dictionary"));
    }
    let (window, rest) = if single_segment {
        // The window is the content size, which must equal `raw_len`, itself within the limit.
        (None, rest)
    } else {
        let [window_descriptor, rest @ ..] = rest else {
            return Err(corrupt("zstd frame header is truncated"));
        };
        (Some(window_size(*window_descriptor)), rest)
    };
    if window.is_some_and(|size| size > 1 << WINDOW_LOG_MAX) {
        return Err(corrupt("zstd frame window is over 2^24 bytes"));
    }
    let declared = match (content_size_flag, single_segment) {
        (0, false) => return Err(corrupt("zstd frame has no content size")),
        (0, true) => le_field::<1>(rest),
        (1, _) => le_field::<2>(rest).map(|v| v + 256),
        (2, _) => le_field::<4>(rest),
        (_, _) => le_field::<8>(rest),
    }
    .ok_or_else(|| corrupt("zstd frame header is truncated"))?;
    if declared != raw_len as u64 {
        return Err(corrupt(format!(
            "zstd frame declares {declared} bytes, the stream has {raw_len}"
        )));
    }
    // Walks block headers only. A second frame or a trailing skippable frame fails here, which also
    // puts every window in the stream under the check above.
    let frame_len = zstd_safe::find_frame_compressed_size(src)
        .map_err(|code| corrupt(format!("zstd frame: {}", zstd_safe::get_error_name(code))))?;
    if frame_len != src.len() {
        return Err(corrupt(format!(
            "zstd stream has {} bytes after its frame",
            src.len() - frame_len
        )));
    }
    Ok(())
}

/// Window size in bytes from a frame's window descriptor: RFC 8878 section 3.1.1.1.2.
fn window_size(descriptor: u8) -> u64 {
    let base = 1_u64 << (10 + u32::from(descriptor >> 3));
    base + base / 8 * u64::from(descriptor & 7)
}

/// Reads a `N` byte little-endian field from the start of `bytes`, `N` at most 8.
fn le_field<const N: usize>(bytes: &[u8]) -> Option<u64> {
    let (field, _) = bytes.split_first_chunk::<N>()?;
    let mut value = [0; 8];
    value[..N].copy_from_slice(field);
    Some(u64::from_le_bytes(value))
}

/// The stream codec of a kept or stored stream, for the prologue word.
pub(super) fn codec_id(compressed: bool) -> u8 {
    if compressed { STREAM_ZSTD } else { STREAM_STORE }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::Rng;
    use super::*;

    /// Low entropy bytes: `h` reliably compresses these.
    fn skewed(len: usize) -> Vec<u8> {
        Rng(7).bytes(len).iter().map(|b| b & 0x03).collect()
    }

    fn frame(context: &mut Compressor<'static>, data: &[u8]) -> Vec<u8> {
        context.compress(data).expect("compress")
    }

    /// A frame by hand: header fields, then one raw block with the last-block bit set.
    fn handmade(descriptor: u8, window: Option<u8>, size_field: &[u8], data: &[u8]) -> Vec<u8> {
        let mut out = ZSTD_MAGIC.to_vec();
        out.push(descriptor);
        out.extend(window);
        out.extend_from_slice(size_field);
        let block_header = (u32::try_from(data.len()).expect("small") << 3) | 1;
        out.extend_from_slice(&block_header.to_le_bytes()[..3]);
        out.extend_from_slice(data);
        out
    }

    fn message(result: Result<(), Error>) -> String {
        match result.expect_err("must be rejected") {
            Error::ChunkCorrupt { detail, .. } => detail,
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn entropy_frames_satisfy_the_format() {
        let mut contexts = ZstdContexts::default();
        let mut compressed = Vec::new();
        for len in [1, 200, 127 * 1024, 128 * 1024, 300_000, 1 << 20] {
            let raw = skewed(len);
            assert!(contexts.compress(StreamPolicy::Entropy, &raw, &mut compressed).expect("ok") || len < 16);
            if compressed.len() < len {
                check_frame(&compressed, len).expect("valid frame");
            }
        }
    }

    #[test]
    fn roundtrip_through_each_compressing_policy() {
        let raw = skewed(70_000);
        let mut contexts = ZstdContexts::default();
        let mut compressed = Vec::new();
        for policy in [
            StreamPolicy::Entropy,
            StreamPolicy::EntropyOrStore,
            StreamPolicy::Zstd { level: 3 },
            StreamPolicy::Zstd { level: -5 },
        ] {
            assert!(contexts.compress(policy, &raw, &mut compressed).expect("ok"), "{policy:?}");
            let mut out = vec![0; raw.len()];
            contexts.decode_zstd(&compressed, &mut out).expect("decode");
            assert_eq!(out, raw, "{policy:?}");
        }
    }

    #[test]
    fn the_output_buffer_grows_to_exactly_the_bound_it_needs() {
        let mut contexts = ZstdContexts::default();
        let mut compressed = Vec::new();
        for len in [700, 1000, 1 << 16, 1 << 20] {
            let raw = skewed(len);
            contexts.compress(StreamPolicy::Entropy, &raw, &mut compressed).expect("ok");
            assert_eq!(compressed.capacity(), zstd_safe::compress_bound(len));
        }
        let raw = skewed(16);
        contexts.compress(StreamPolicy::Entropy, &raw, &mut compressed).expect("ok");
        assert_eq!(compressed.capacity(), zstd_safe::compress_bound(1 << 20), "it never shrinks");
    }

    #[test]
    fn store_and_empty_never_compress() {
        let mut contexts = ZstdContexts::default();
        let mut compressed = Vec::new();
        let raw = skewed(1000);
        assert!(!contexts.compress(StreamPolicy::Store, &raw, &mut compressed).expect("ok"));
        for policy in [StreamPolicy::Entropy, StreamPolicy::EntropyOrStore] {
            assert!(!contexts.compress(policy, &[], &mut compressed).expect("ok"));
        }
    }

    #[test]
    fn incompressible_data_is_stored_under_every_policy() {
        let raw = Rng(8).bytes(64 * 1024);
        let mut contexts = ZstdContexts::default();
        let mut compressed = Vec::new();
        for policy in
            [StreamPolicy::Entropy, StreamPolicy::EntropyOrStore, StreamPolicy::Zstd { level: 3 }]
        {
            assert!(!contexts.compress(policy, &raw, &mut compressed).expect("ok"), "{policy:?}");
        }
    }

    #[test]
    fn the_half_percent_rule_has_an_exact_boundary() {
        let policy = StreamPolicy::EntropyOrStore;
        assert!(keeps_frame(policy, 995, 1000), "a gain of exactly 0.5% is kept");
        assert!(!keeps_frame(policy, 996, 1000), "under 0.5% is stored");
        assert!(keeps_frame(StreamPolicy::Entropy, 999, 1000));
        assert!(!keeps_frame(StreamPolicy::Entropy, 1000, 1000));
        assert!(!keeps_frame(StreamPolicy::Zstd { level: 3 }, 1000, 1000));
        assert!(!keeps_frame(StreamPolicy::Store, 1, 1000));
    }

    #[test]
    fn a_context_is_reused_and_the_level_can_change() {
        let raw = skewed(10_000);
        let (mut reused, mut fresh) = (ZstdContexts::default(), ZstdContexts::default());
        let (mut a, mut b) = (Vec::new(), Vec::new());
        reused.compress(StreamPolicy::Zstd { level: 9 }, &raw, &mut a).expect("ok");
        reused.compress(StreamPolicy::Entropy, &raw, &mut a).expect("ok");
        reused.compress(StreamPolicy::Zstd { level: 3 }, &raw, &mut a).expect("ok");
        fresh.compress(StreamPolicy::Zstd { level: 3 }, &raw, &mut b).expect("ok");
        assert_eq!(a, b, "bytes must not depend on earlier work of the context");
    }

    #[test]
    fn a_handmade_frame_passes_as_control() {
        let data = [9; 300];
        // Window descriptor 0: 1 KiB. Content size flag 1: two bytes, biased by 256.
        let good = handmade(0x40, Some(0), &44_u16.to_le_bytes(), &data);
        check_frame(&good, 300).expect("valid frame");
        let mut out = [0; 300];
        ZstdContexts::default().decode_zstd(&good, &mut out).expect("decode");
        assert_eq!(out, data);
        // Exactly the limit passes.
        let limit = handmade(0x80, Some(14 << 3), &300_u32.to_le_bytes(), &data);
        check_frame(&limit, 300).expect("a window of 2^24 is allowed");
        // Single segment, one byte of size.
        check_frame(&handmade(0x20, None, &[200], &[1; 200]), 200).expect("single segment");
    }

    #[test]
    fn a_window_over_the_limit_is_rejected_although_one_pass_zstd_would_decode_it() {
        let data = [9; 300];
        let over_log = handmade(0x80, Some(15 << 3), &300_u32.to_le_bytes(), &data);
        let over_mantissa = handmade(0x80, Some((14 << 3) | 1), &300_u32.to_le_bytes(), &data);
        for bad in [&over_log, &over_mantissa] {
            assert!(message(check_frame(bad, 300)).contains("window"));
            let mut out = [0; 300];
            let err = ZstdContexts::default().decode_zstd(bad, &mut out).expect_err("rejected");
            assert!(matches!(err, Error::ChunkCorrupt { .. }), "{err:?}");
        }
    }

    #[test]
    fn frames_outside_the_format_are_rejected() {
        let data = [9; 300];
        let size = 300_u32.to_le_bytes();
        let cases: [(Vec<u8>, &str); 7] = [
            (handmade(0x84, Some(0), &size, &data), "checksum"),
            (handmade(0x81, Some(0), &size, &data), "dictionary"),
            (handmade(0x88, Some(0), &size, &data), "reserved"),
            (handmade(0x90, Some(0), &size, &data), "reserved"),
            (handmade(0x00, Some(0), &[], &data), "content size"),
            (handmade(0x80, Some(0), &299_u32.to_le_bytes(), &data), "declares"),
            (vec![0; 40], "start"),
        ];
        for (bad, expected) in cases {
            let text = message(check_frame(&bad, 300));
            assert!(text.contains(expected), "{text}");
        }
        assert!(message(check_frame(&[0x28, 0xB5], 300)).contains("short"));
        assert!(message(check_frame(&ZSTD_MAGIC, 300)).contains("short"));
        assert!(message(check_frame(&[0x28, 0xB5, 0x2F, 0xFD, 0x80], 300)).contains("truncated"));
        let cut = handmade(0x80, Some(0), &size, &data);
        assert!(message(check_frame(&cut[..7], 300)).contains("truncated"));
        assert!(message(check_frame(&cut[..cut.len() - 1], 300)).contains("zstd frame"));
    }

    #[test]
    fn a_compressor_with_a_checksum_is_refused() {
        let raw = skewed(5000);
        let mut context = Compressor::new(1).expect("context");
        context.set_parameter(CParameter::ChecksumFlag(true)).expect("parameter");
        assert!(message(check_frame(&frame(&mut context, &raw), 5000)).contains("checksum"));
        let mut context = Compressor::new(1).expect("context");
        context.set_parameter(CParameter::ContentSizeFlag(false)).expect("parameter");
        let text = message(check_frame(&frame(&mut context, &raw), 5000));
        assert!(text.contains("content size"), "{text}");
    }

    #[test]
    fn trailing_frames_are_rejected() {
        let raw = skewed(5000);
        let mut context = new_compressor(1).expect("context");
        let one = frame(&mut context, &raw);
        check_frame(&one, 5000).expect("one frame");

        let mut two = one.clone();
        two.extend_from_slice(&one);
        assert!(message(check_frame(&two, 5000)).contains("after its frame"));

        let mut skippable = one;
        skippable.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 0, 0, 0, 0]);
        assert!(message(check_frame(&skippable, 5000)).contains("after its frame"));
    }

    #[test]
    fn output_length_is_exact_whatever_the_frame_says() {
        let raw = skewed(5000);
        let mut context = new_compressor(1).expect("context");
        let good = frame(&mut context, &raw);
        let mut contexts = ZstdContexts::default();
        for len in [4999, 5001] {
            let mut out = vec![0; len];
            let err = contexts.decode_zstd(&good, &mut out).expect_err("wrong length");
            assert!(matches!(err, Error::ChunkCorrupt { .. }), "{err:?}");
        }
    }

    #[test]
    fn a_frame_that_lies_about_its_content_is_rejected() {
        // Declares 300 bytes, carries 299.
        let lying = handmade(0x80, Some(0), &300_u32.to_le_bytes(), &[1; 299]);
        let mut out = [0; 300];
        let err = ZstdContexts::default().decode_zstd(&lying, &mut out).expect_err("short");
        assert!(matches!(err, Error::ChunkCorrupt { .. }), "{err:?}");
        // Declares 300 bytes, carries 301.
        let lying = handmade(0x80, Some(0), &300_u32.to_le_bytes(), &[1; 301]);
        let err = ZstdContexts::default().decode_zstd(&lying, &mut out).expect_err("long");
        assert!(matches!(err, Error::ChunkCorrupt { .. }), "{err:?}");
    }
}
