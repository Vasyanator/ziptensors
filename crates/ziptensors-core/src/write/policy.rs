//! File: write/policy.rs
//! Purpose: the mode table of spec section 11, as code: what a writer does with each class of bytes.
//!
//! Main responsibilities: map a dtype and a mode to a chain, a stream policy and a chunk size, and say
//! which optional work a mode adds: widening views, the `rotl1` trial, blob identity, the delta trial.
//!
//! Key structures: [`Encoding`].
//!
//! Key functions: [`encoding_of`], [`skeleton_encoding`], [`narrow_dtype`], [`with_rotl1`],
//! [`widen_candidates`], [`computes_identity`], [`tries_delta`], [`delta_candidate`], [`with_delta`].
//!
//! Implementation notes: nothing here is visible to a reader, so a change is a change of ratio and
//! speed only, and belongs together with spec section 11. The table has no row for `F64`: it is treated
//! as a float without a view, because it has the plane statistics of one. Every integer stored, and so
//! is an id this build does not know: the table says "unknown integer", and a wrong guess costs only
//! ratio. Nibble data reaches the writer as `U8` from every parser, and is stored either way. A
//! block-quantized type without a field table, such as `Q4_0`, is stored too: the cell an unknown id
//! had before it was registered.

use crate::chain::{Chain, DeltaOp, ElementWidth, FieldTable, Layout, Step, View};
use crate::codec::StreamPolicy;
use crate::error::Error;
use crate::types::{Dtype, Mode};

/// Target chunk size: 1 MiB, spec section 11.
const CHUNK_TARGET: u32 = 1 << 20;
/// zstd level of skeleton and opaque bytes, spec section 11.
const SKELETON_LEVEL: i32 = 3;
/// `dtype_hint` of a skeleton blob: id 0, "no dtype was determined".
pub(super) const SKELETON_DTYPE: Dtype = Dtype::Unknown(0);

/// How the chunks of one blob are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Encoding {
    /// Transform chain of the blob.
    pub(super) chain: Chain,
    /// Choice of stream codec, per stream.
    pub(super) stream: StreamPolicy,
    /// `chunk_size` of the blob, a multiple of the chain block.
    pub(super) chunk_size: u32,
}

impl Encoding {
    /// An encoding with the chunk size `chain` implies.
    fn new(chain: Chain, stream: StreamPolicy) -> Self {
        Self { chain, stream, chunk_size: chunk_size_of(chain) }
    }

    /// Most stored bytes of a chunk of `raw_len` bytes: the rule of spec section 6.1.
    pub(super) fn stored_bound(&self, raw_len: usize) -> usize {
        raw_len + 4 * self.chain.stream_count()
    }
}

/// 1 MiB rounded down to a multiple of the chain block, spec section 11.
fn chunk_size_of(chain: Chain) -> u32 {
    let block = chain.block_bytes();
    CHUNK_TARGET / block * block
}

/// The encoding of a chain of one layout step.
fn plain(layout: Layout, stream: StreamPolicy) -> Result<Encoding, Error> {
    Ok(Encoding::new(Chain::new(&[Step::Layout(layout)])?, stream))
}

/// The encoding of `hint` under `mode`: the row of the table and its cell.
///
/// # Errors
/// Returns what [`Chain::new`] returns, which the shape of the chains here rules out.
pub(super) fn encoding_of(hint: Dtype, mode: Mode) -> Result<Encoding, Error> {
    let planes = |width| plain(Layout::Planes(width), StreamPolicy::EntropyOrStore);
    let fields = |table| match mode {
        Mode::Fast => plain(Layout::Whole, StreamPolicy::Store),
        Mode::Balanced | Mode::Max => plain(Layout::Fields(table), StreamPolicy::Entropy),
    };
    match hint {
        Dtype::F16 | Dtype::Bf16 => planes(ElementWidth::Two),
        Dtype::F32 => planes(ElementWidth::Four),
        Dtype::F64 => planes(ElementWidth::Eight),
        Dtype::F8E4M3 | Dtype::F8E5M2 => plain(Layout::Whole, StreamPolicy::Entropy),
        Dtype::Q80 => fields(FieldTable::Q80),
        Dtype::Q4K => fields(FieldTable::Q4K),
        Dtype::Q5K => fields(FieldTable::Q5K),
        Dtype::Q6K => fields(FieldTable::Q6K),
        // Block types without a field table: no profile hands them out as a range today (`gguf.rs`
        // keeps their bytes in the skeleton), and the cell is the one an unknown id had before
        // they were registered, so registering them changed no output.
        Dtype::Q40
        | Dtype::Q41
        | Dtype::Q50
        | Dtype::Q51
        | Dtype::Q81
        | Dtype::Q2K
        | Dtype::Q3K
        | Dtype::Q8K
        | Dtype::Iq2Xxs
        | Dtype::Iq2Xs
        | Dtype::Iq3Xxs
        | Dtype::Iq1S
        | Dtype::Iq4Nl
        | Dtype::Iq3S
        | Dtype::Iq2S
        | Dtype::Iq4Xs
        | Dtype::Iq1M
        | Dtype::Tq10
        | Dtype::Tq20
        | Dtype::Mxfp4
        | Dtype::Bool
        | Dtype::U8
        | Dtype::I8
        | Dtype::U16
        | Dtype::I16
        | Dtype::U32
        | Dtype::I32
        | Dtype::U64
        | Dtype::I64
        | Dtype::Nibble
        | Dtype::Unknown(_) => plain(Layout::Whole, StreamPolicy::Store),
    }
}

/// Skeleton fragments and opaque files: `whole`, zstd level 3.
///
/// # Errors
/// As [`encoding_of`].
pub(super) fn skeleton_encoding() -> Result<Encoding, Error> {
    plain(Layout::Whole, StreamPolicy::Zstd { level: SKELETON_LEVEL })
}

/// The dtype of the canonical bytes behind `view`, for a source of dtype `wide`.
#[must_use]
pub(super) fn narrow_dtype(view: View, wide: Dtype) -> Dtype {
    match view {
        View::Identity => wide,
        View::F16ToF32 => Dtype::F16,
        View::Bf16ToF32 => Dtype::Bf16,
    }
}

/// The views to try on a range of `hint` under `mode`, in order of preference. Both canonical forms are
/// 2 bytes per element, so neither is smaller, and a fixed order is what stays deterministic with no
/// trial. `f16` goes first because it stores smaller in aggregate, see `MODULE_README.md`.
#[must_use]
pub(super) fn widen_candidates(hint: Dtype, mode: Mode) -> &'static [View] {
    match (hint, mode) {
        (Dtype::F32, Mode::Balanced | Mode::Max) => &[View::F16ToF32, View::Bf16ToF32],
        _ => &[],
    }
}

/// `encoding` with a `rotl1` step, if `mode` tries one on such a blob: `max`, on planes of floats.
///
/// # Errors
/// As [`encoding_of`].
pub(super) fn with_rotl1(encoding: Encoding, mode: Mode) -> Result<Option<Encoding>, Error> {
    match (mode, encoding.chain.layout()) {
        (Mode::Max, Layout::Planes(width)) => {
            let chain = Chain::new(&[Step::Rotl1(width), Step::Layout(Layout::Planes(width))])?;
            Ok(Some(Encoding::new(chain, encoding.stream)))
        }
        _ => Ok(None),
    }
}

/// Whether `mode` computes blob identity. Under the others `BLOB.blake3` stays all zero.
#[must_use]
pub(super) fn computes_identity(mode: Mode) -> bool {
    match mode {
        Mode::Fast | Mode::Balanced => false,
        Mode::Max => true,
    }
}

/// The delta operation `mode` tries on a range of `hint` that has a base, spec section 11: `delta_xor`
/// of the float width, in `max` only. FP8, quantized and integer data would be valid under
/// `delta_xor` too, but are unmeasured, so they get none.
#[must_use]
pub(super) fn delta_candidate(hint: Dtype, mode: Mode) -> Option<DeltaOp> {
    if !tries_delta(mode) {
        return None;
    }
    match hint {
        Dtype::F16 | Dtype::Bf16 => Some(DeltaOp::Xor(ElementWidth::Two)),
        Dtype::F32 => Some(DeltaOp::Xor(ElementWidth::Four)),
        Dtype::F64 => Some(DeltaOp::Xor(ElementWidth::Eight)),
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::Q80
        | Dtype::Q4K
        | Dtype::Q5K
        | Dtype::Q6K
        | Dtype::Q40
        | Dtype::Q41
        | Dtype::Q50
        | Dtype::Q51
        | Dtype::Q81
        | Dtype::Q2K
        | Dtype::Q3K
        | Dtype::Q8K
        | Dtype::Iq2Xxs
        | Dtype::Iq2Xs
        | Dtype::Iq3Xxs
        | Dtype::Iq1S
        | Dtype::Iq4Nl
        | Dtype::Iq3S
        | Dtype::Iq2S
        | Dtype::Iq4Xs
        | Dtype::Iq1M
        | Dtype::Tq10
        | Dtype::Tq20
        | Dtype::Mxfp4
        | Dtype::Bool
        | Dtype::U8
        | Dtype::I8
        | Dtype::U16
        | Dtype::I16
        | Dtype::U32
        | Dtype::I32
        | Dtype::U64
        | Dtype::I64
        | Dtype::Nibble
        | Dtype::Unknown(_) => None,
    }
}

/// Whether `mode` stores ranges as delta blobs against a base the caller names: `max` only.
#[must_use]
pub(super) fn tries_delta(mode: Mode) -> bool {
    match mode {
        Mode::Fast | Mode::Balanced => false,
        Mode::Max => true,
    }
}

/// `encoding` with the delta step `op` in front of its chain, the encoding of a delta blob: the rest of
/// the chain and the stream policy encode the residual. The chunk size follows the new chain block,
/// which the caller compares with the base's.
///
/// # Errors
/// As [`encoding_of`]; also if `encoding` already has a delta step.
pub(super) fn with_delta(encoding: Encoding, op: DeltaOp) -> Result<Encoding, Error> {
    let steps: Vec<Step> = [Step::Delta(op)].into_iter().chain(encoding.chain.steps()).collect();
    Ok(Encoding::new(Chain::new(&steps)?, encoding.stream))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [Mode; 3] = [Mode::Fast, Mode::Balanced, Mode::Max];

    fn encoded(dtype: Dtype, mode: Mode) -> Encoding {
        encoding_of(dtype, mode).expect("valid chain")
    }

    #[test]
    fn floats_are_planes_of_store_or_entropy_in_every_mode() {
        let cases = [
            (Dtype::F16, ElementWidth::Two),
            (Dtype::Bf16, ElementWidth::Two),
            (Dtype::F32, ElementWidth::Four),
            (Dtype::F64, ElementWidth::Eight),
        ];
        for mode in MODES {
            for (dtype, width) in cases {
                let encoding = encoded(dtype, mode);
                assert_eq!(encoding.chain.layout(), Layout::Planes(width));
                assert_eq!(encoding.chain.rotl1(), None);
                assert_eq!(encoding.stream, StreamPolicy::EntropyOrStore);
                assert_eq!(encoding.chunk_size, 1 << 20);
            }
        }
    }

    #[test]
    fn fp8_is_whole_entropy_in_every_mode() {
        for mode in MODES {
            for dtype in [Dtype::F8E4M3, Dtype::F8E5M2] {
                let encoding = encoded(dtype, mode);
                assert_eq!(encoding.chain.layout(), Layout::Whole);
                assert_eq!(encoding.stream, StreamPolicy::Entropy);
            }
        }
    }

    #[test]
    fn quantized_is_stored_in_fast_and_fields_otherwise() {
        let tables = [
            (Dtype::Q80, FieldTable::Q80),
            (Dtype::Q4K, FieldTable::Q4K),
            (Dtype::Q5K, FieldTable::Q5K),
            (Dtype::Q6K, FieldTable::Q6K),
        ];
        for (dtype, table) in tables {
            let fast = encoded(dtype, Mode::Fast);
            assert_eq!(fast.chain.layout(), Layout::Whole);
            assert_eq!(fast.stream, StreamPolicy::Store);
            for mode in [Mode::Balanced, Mode::Max] {
                let encoding = encoded(dtype, mode);
                assert_eq!(encoding.chain.layout(), Layout::Fields(table));
                assert_eq!(encoding.stream, StreamPolicy::Entropy);
            }
        }
    }

    #[test]
    fn chunk_size_is_a_multiple_of_the_block_under_one_mib() {
        assert_eq!(encoded(Dtype::Q80, Mode::Max).chunk_size, 1_048_560);
        for mode in MODES {
            for dtype in [Dtype::Q80, Dtype::Q4K, Dtype::Q5K, Dtype::Q6K, Dtype::F64, Dtype::U8] {
                let encoding = encoded(dtype, mode);
                assert_eq!(encoding.chunk_size % encoding.chain.block_bytes(), 0);
                assert!(encoding.chunk_size <= 1 << 20);
                assert!(encoding.chunk_size + encoding.chain.block_bytes() > 1 << 20);
            }
        }
    }

    #[test]
    fn nibble_integers_unsplit_blocks_and_unknown_ids_are_stored_in_every_mode() {
        let stored = [
            Dtype::Bool,
            Dtype::U8,
            Dtype::I8,
            Dtype::U16,
            Dtype::I16,
            Dtype::U32,
            Dtype::I32,
            Dtype::U64,
            Dtype::I64,
            Dtype::Nibble,
            Dtype::Unknown(0x7777),
            Dtype::Q40,
            Dtype::Q2K,
            Dtype::Iq4Xs,
            Dtype::Mxfp4,
        ];
        for mode in MODES {
            for dtype in stored {
                let encoding = encoded(dtype, mode);
                assert_eq!(encoding.chain.layout(), Layout::Whole);
                assert_eq!(encoding.stream, StreamPolicy::Store);
            }
        }
    }

    #[test]
    fn skeleton_is_whole_zstd_three() {
        let encoding = skeleton_encoding().expect("valid chain");
        assert_eq!(encoding.chain.layout(), Layout::Whole);
        assert_eq!(encoding.stream, StreamPolicy::Zstd { level: 3 });
    }

    #[test]
    fn widening_starts_in_balanced_and_only_on_f32() {
        assert_eq!(widen_candidates(Dtype::F32, Mode::Fast), []);
        for mode in [Mode::Balanced, Mode::Max] {
            assert_eq!(widen_candidates(Dtype::F32, mode), [View::F16ToF32, View::Bf16ToF32]);
            for dtype in [Dtype::F16, Dtype::Bf16, Dtype::F64, Dtype::U32] {
                assert_eq!(widen_candidates(dtype, mode), []);
            }
        }
    }

    #[test]
    fn a_view_names_the_dtype_of_its_canonical_bytes() {
        assert_eq!(narrow_dtype(View::F16ToF32, Dtype::F32), Dtype::F16);
        assert_eq!(narrow_dtype(View::Identity, Dtype::F32), Dtype::F32);
    }

    #[test]
    fn the_rotation_trial_exists_in_max_on_planes_only() {
        let planes = encoded(Dtype::F32, Mode::Max);
        let rotated = with_rotl1(planes, Mode::Max).expect("valid chain").expect("max tries it");
        assert_eq!(rotated.chain.rotl1(), Some(ElementWidth::Four));
        assert_eq!(rotated.chain.layout(), planes.chain.layout());
        assert_eq!(with_rotl1(planes, Mode::Balanced).expect("valid chain"), None);
        let whole = encoded(Dtype::F8E4M3, Mode::Max);
        assert_eq!(with_rotl1(whole, Mode::Max).expect("valid chain"), None);
    }

    #[test]
    fn identity_is_computed_in_max_only() {
        assert!(!computes_identity(Mode::Fast));
        assert!(!computes_identity(Mode::Balanced));
        assert!(computes_identity(Mode::Max));
    }

    #[test]
    fn delta_is_tried_in_max_on_floats_only() {
        let floats = [
            (Dtype::F16, ElementWidth::Two),
            (Dtype::Bf16, ElementWidth::Two),
            (Dtype::F32, ElementWidth::Four),
            (Dtype::F64, ElementWidth::Eight),
        ];
        for (dtype, width) in floats {
            assert_eq!(delta_candidate(dtype, Mode::Max), Some(DeltaOp::Xor(width)));
            assert_eq!(delta_candidate(dtype, Mode::Balanced), None);
            assert_eq!(delta_candidate(dtype, Mode::Fast), None);
        }
        for dtype in [Dtype::F8E4M3, Dtype::Q80, Dtype::U8, Dtype::I32, Dtype::Unknown(0x7777)] {
            assert_eq!(delta_candidate(dtype, Mode::Max), None);
        }
    }

    #[test]
    fn a_delta_encoding_keeps_the_rest_of_the_chain_and_the_chunk_size() {
        let planes = encoded(Dtype::Bf16, Mode::Max);
        let rotated = with_rotl1(planes, Mode::Max).expect("valid chain").expect("max tries it");
        let op = DeltaOp::Xor(ElementWidth::Two);
        for encoding in [planes, rotated] {
            let delta = with_delta(encoding, op).expect("valid chain");
            assert_eq!(delta.chain.delta(), Some(op));
            assert_eq!(delta.chain.residual(), encoding.chain);
            assert_eq!(delta.stream, encoding.stream);
            assert_eq!(delta.chunk_size, encoding.chunk_size);
        }
        let twice = with_delta(with_delta(planes, op).expect("valid chain"), op);
        assert!(twice.is_err(), "a chain has at most one delta step");
    }

    #[test]
    fn the_stored_bound_is_the_rule_of_the_spec() {
        let encoding = encoded(Dtype::F32, Mode::Fast);
        assert_eq!(encoding.stored_bound(100), 116);
    }
}
