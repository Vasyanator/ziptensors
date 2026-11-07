//! File: chain.rs
//! Purpose: the single owner of the transform chain model and of the spec's format id registry.
//!
//! Main responsibilities:
//! - id constants for steps, views, stream codecs, blob kinds, profiles and feature bits;
//! - typed steps and layouts, with their field tables and block sizes, and the delta operation;
//! - [`BlockGeometry`] of every ggml block-quantized type, the one table of block sizes;
//! - [`Chain`], whose construction enforces the chain rules of spec section 5.2.
//!
//! Key structures: [`RawStep`], [`Step`], [`DeltaOp`], [`Layout`], [`Chain`], [`View`],
//! [`FieldTable`], [`ElementWidth`], [`BlockGeometry`].
//!
//! Implementation notes: this is a leaf, depending only on `error`, so that `format`, `codec` and
//! `types` can share one definition. Values here must match `spec/ziptensors-format.md`; a change to
//! one requires the other. `dtype` ids are not here: they belong to `Dtype` in `types.rs`. Ids stay
//! raw `u16` values in [`RawStep`] and on the wire, and conversion to a typed value is where unknown
//! ids are rejected.

use crate::error::Error;

/// Never a valid step. In `EXTN.view` the same value means identity, see [`VIEW_IDENTITY`].
pub const STEP_INVALID: u16 = 0x0000;
/// Layout step `whole`.
pub const STEP_WHOLE: u16 = 0x0100;
/// Layout step `planes`; `param0` is the element width.
pub const STEP_PLANES: u16 = 0x0101;
/// Layout step `fields`; `param0` is the field table id.
pub const STEP_FIELDS: u16 = 0x0102;
/// Element step `rotl1`; `param0` is the element width.
pub const STEP_ROTL1: u16 = 0x0200;

/// `EXTN.view` value for no view.
pub const VIEW_IDENTITY: u16 = 0x0000;
/// View f16 to f32.
pub const VIEW_F16_TO_F32: u16 = 0x0301;
/// View bf16 to f32.
pub const VIEW_BF16_TO_F32: u16 = 0x0302;

/// Delta step `delta_xor`, version 0.3; `param0` is the element width. Valid only first in the
/// chain of a delta blob (spec section 7a).
pub const STEP_DELTA_XOR: u16 = 0x0400;
/// First id reserved for further delta steps.
pub const STEP_RESERVED_DELTA_FIRST: u16 = 0x0401;
/// Last id reserved for delta steps.
pub const STEP_RESERVED_DELTA_LAST: u16 = 0x04FF;
/// First id reserved for derived steps.
pub const STEP_RESERVED_DERIVED_FIRST: u16 = 0x0500;
/// Last id reserved for derived steps.
pub const STEP_RESERVED_DERIVED_LAST: u16 = 0x05FF;
/// First private id. Private ids are rejected by default readers.
pub const STEP_PRIVATE_FIRST: u16 = 0x8000;

/// Field table id of `Q8_0`.
pub const FIELD_TABLE_Q8_0: u16 = 1;
/// Field table id of `Q4_K`.
pub const FIELD_TABLE_Q4_K: u16 = 2;
/// Field table id of `Q5_K`.
pub const FIELD_TABLE_Q5_K: u16 = 3;
/// Field table id of `Q6_K`.
pub const FIELD_TABLE_Q6_K: u16 = 4;

/// Stream codec: the stream bytes are the raw stream.
pub const STREAM_STORE: u8 = 0;
/// Stream codec: one zstd frame.
pub const STREAM_ZSTD: u8 = 1;

/// Blob kind: chunked, the only kind before version 0.3.
pub const BLOB_KIND_CHUNKED: u8 = 0;
/// Blob kind: delta, version 0.3. Its chunks are combined with chunks of a base blob (spec
/// section 7a); its `PARM` entry names the base.
pub const BLOB_KIND_DELTA: u8 = 1;
/// Blob kind reserved for derived blobs.
pub const BLOB_KIND_RESERVED_DERIVED: u8 = 2;

/// Profile id: no structure recognized.
pub const PROFILE_OPAQUE: u8 = 0;
/// Profile id: safetensors.
pub const PROFILE_SAFETENSORS: u8 = 1;
/// Profile id: GGUF.
pub const PROFILE_GGUF: u8 = 2;
/// Profile id: ONNX.
pub const PROFILE_ONNX: u8 = 3;
/// Profile id: `PyTorch` ZIP archive.
pub const PROFILE_PT_ZIP: u8 = 4;

/// Required feature: the file has delta blobs, version 0.3. Set exactly when a blob of kind
/// [`BLOB_KIND_DELTA`] exists.
pub const FEATURE_DELTA: u64 = 1 << 0;
/// Required feature reserved for derived blobs.
pub const FEATURE_RESERVED_DERIVED: u64 = 1 << 1;
/// Required feature reserved for multi-volume containers.
pub const FEATURE_RESERVED_MULTI_VOLUME: u64 = 1 << 2;
/// The required feature bits this version implements. Any other set bit is rejected.
pub const SUPPORTED_REQUIRED_FEATURES: u64 = FEATURE_DELTA;

/// Structural limit on `CHAN.step_count`. Valid chains have 1 to 3 steps.
pub const MAX_CHAIN_STEPS: usize = 8;
/// Largest `chunk_size` of a blob and largest raw length of a chunk, 8 MiB (spec section 6.1).
pub const MAX_CHUNK_SIZE: u32 = 8 << 20;

/// One `CHST` record as stored. Any bit pattern is representable, so the format layer can hold an
/// unknown id until it is rejected by [`Step::from_raw`] or [`Chain::from_raw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawStep {
    /// Step id.
    pub codec_id: u16,
    /// First parameter; its meaning depends on `codec_id`.
    pub param0: u16,
    /// Second parameter; zero for every registered step.
    pub param1: u32,
}

/// Width in bytes of a little-endian element that `planes`, `rotl1` or a delta step acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementWidth {
    /// 2 bytes.
    Two,
    /// 4 bytes.
    Four,
    /// 8 bytes.
    Eight,
}

impl ElementWidth {
    /// Maps a `param0` value to a width. `None` for anything but 2, 4 or 8.
    #[must_use]
    pub fn from_param(param: u16) -> Option<Self> {
        match param {
            2 => Some(Self::Two),
            4 => Some(Self::Four),
            8 => Some(Self::Eight),
            _ => None,
        }
    }

    /// Returns the `param0` value of this width, which is its size in bytes.
    #[must_use]
    pub fn param(self) -> u16 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    /// Size in bytes.
    #[must_use]
    pub fn bytes(self) -> u32 {
        u32::from(self.param())
    }
}

/// Geometry of one block of a ggml block-quantized type: `elements` values packed into `bytes`
/// bytes. ggml lays a tensor row out as whole blocks, so a row of `n` elements takes
/// `n / elements * bytes` bytes and `n` must be a multiple of `elements`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockGeometry {
    /// Values one block holds. Never zero.
    pub elements: u32,
    /// Size of one block in bytes. Never zero.
    pub bytes: u32,
}

const fn block(elements: u32, bytes: u32) -> BlockGeometry {
    BlockGeometry { elements, bytes }
}

// The block structs of ggml's `ggml-common.h`, with the sizes the reference `gguf` Python package
// lists in `GGML_QUANT_SIZES`. Every block-quantized dtype of spec section 5.3 has one.

/// ggml `block_q4_0`.
pub const BLOCK_Q4_0: BlockGeometry = block(32, 18);
/// ggml `block_q4_1`.
pub const BLOCK_Q4_1: BlockGeometry = block(32, 20);
/// ggml `block_q5_0`.
pub const BLOCK_Q5_0: BlockGeometry = block(32, 22);
/// ggml `block_q5_1`.
pub const BLOCK_Q5_1: BlockGeometry = block(32, 24);
/// ggml `block_q8_0`.
pub const BLOCK_Q8_0: BlockGeometry = block(32, 34);
/// ggml `block_q8_1`.
pub const BLOCK_Q8_1: BlockGeometry = block(32, 40);
/// ggml `block_q2_K`.
pub const BLOCK_Q2_K: BlockGeometry = block(256, 84);
/// ggml `block_q3_K`.
pub const BLOCK_Q3_K: BlockGeometry = block(256, 110);
/// ggml `block_q4_K`.
pub const BLOCK_Q4_K: BlockGeometry = block(256, 144);
/// ggml `block_q5_K`.
pub const BLOCK_Q5_K: BlockGeometry = block(256, 176);
/// ggml `block_q6_K`.
pub const BLOCK_Q6_K: BlockGeometry = block(256, 210);
/// ggml `block_q8_K`.
pub const BLOCK_Q8_K: BlockGeometry = block(256, 292);
/// ggml `block_iq2_xxs`.
pub const BLOCK_IQ2_XXS: BlockGeometry = block(256, 66);
/// ggml `block_iq2_xs`.
pub const BLOCK_IQ2_XS: BlockGeometry = block(256, 74);
/// ggml `block_iq3_xxs`.
pub const BLOCK_IQ3_XXS: BlockGeometry = block(256, 98);
/// ggml `block_iq1_s`.
pub const BLOCK_IQ1_S: BlockGeometry = block(256, 50);
/// ggml `block_iq4_nl`.
pub const BLOCK_IQ4_NL: BlockGeometry = block(32, 18);
/// ggml `block_iq3_s`.
pub const BLOCK_IQ3_S: BlockGeometry = block(256, 110);
/// ggml `block_iq2_s`.
pub const BLOCK_IQ2_S: BlockGeometry = block(256, 82);
/// ggml `block_iq4_xs`.
pub const BLOCK_IQ4_XS: BlockGeometry = block(256, 136);
/// ggml `block_iq1_m`.
pub const BLOCK_IQ1_M: BlockGeometry = block(256, 56);
/// ggml `block_tq1_0`.
pub const BLOCK_TQ1_0: BlockGeometry = block(256, 54);
/// ggml `block_tq2_0`.
pub const BLOCK_TQ2_0: BlockGeometry = block(256, 66);
/// ggml `block_mxfp4`.
pub const BLOCK_MXFP4: BlockGeometry = block(32, 17);

/// One field of a quantized block: `len` bytes at byte `offset` of every block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldSpan {
    /// Start within the block, in bytes.
    pub offset: u32,
    /// Length in bytes.
    pub len: u32,
}

const fn span(offset: u32, len: u32) -> FieldSpan {
    FieldSpan { offset, len }
}

/// Spans of `Q8_0` in stream order.
const SPANS_Q8_0: [FieldSpan; 3] = [span(0, 1), span(1, 1), span(2, 32)];
/// Spans of `Q4_K` in stream order.
const SPANS_Q4_K: [FieldSpan; 6] =
    [span(0, 1), span(1, 1), span(2, 1), span(3, 1), span(4, 12), span(16, 128)];
/// Spans of `Q5_K` in stream order.
const SPANS_Q5_K: [FieldSpan; 7] =
    [span(0, 1), span(1, 1), span(2, 1), span(3, 1), span(4, 12), span(16, 32), span(48, 128)];
/// Spans of `Q6_K` in stream order.
const SPANS_Q6_K: [FieldSpan; 5] =
    [span(0, 128), span(128, 64), span(192, 16), span(208, 1), span(209, 1)];

/// A quantized block layout, from spec section 6.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldTable {
    /// ggml `block_q8_0`.
    Q80,
    /// ggml `block_q4_K`.
    Q4K,
    /// ggml `block_q5_K`.
    Q5K,
    /// ggml `block_q6_K`.
    Q6K,
}

impl FieldTable {
    /// Maps a `param0` value to a table. `None` for an unknown table id.
    #[must_use]
    pub fn from_id(id: u16) -> Option<Self> {
        match id {
            FIELD_TABLE_Q8_0 => Some(Self::Q80),
            FIELD_TABLE_Q4_K => Some(Self::Q4K),
            FIELD_TABLE_Q5_K => Some(Self::Q5K),
            FIELD_TABLE_Q6_K => Some(Self::Q6K),
            _ => None,
        }
    }

    /// Returns the table id.
    #[must_use]
    pub fn id(self) -> u16 {
        match self {
            Self::Q80 => FIELD_TABLE_Q8_0,
            Self::Q4K => FIELD_TABLE_Q4_K,
            Self::Q5K => FIELD_TABLE_Q5_K,
            Self::Q6K => FIELD_TABLE_Q6_K,
        }
    }

    /// Geometry of the ggml block the table splits.
    #[must_use]
    pub fn block(self) -> BlockGeometry {
        match self {
            Self::Q80 => BLOCK_Q8_0,
            Self::Q4K => BLOCK_Q4_K,
            Self::Q5K => BLOCK_Q5_K,
            Self::Q6K => BLOCK_Q6_K,
        }
    }

    /// Size of one block in bytes.
    #[must_use]
    pub fn block_bytes(self) -> u32 {
        self.block().bytes
    }

    /// The fields in stream order. Contiguous, and they cover the block exactly.
    #[must_use]
    pub fn fields(self) -> &'static [FieldSpan] {
        match self {
            Self::Q80 => &SPANS_Q8_0,
            Self::Q4K => &SPANS_Q4_K,
            Self::Q5K => &SPANS_Q5_K,
            Self::Q6K => &SPANS_Q6_K,
        }
    }
}

/// The step of a chain that splits bytes into streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// One stream, the input.
    Whole,
    /// One stream per byte position of an element.
    Planes(ElementWidth),
    /// One stream per field of a block.
    Fields(FieldTable),
}

impl Layout {
    /// Size in bytes of the unit this layout splits.
    #[must_use]
    pub fn block_bytes(self) -> u32 {
        match self {
            Self::Whole => 1,
            Self::Planes(width) => width.bytes(),
            Self::Fields(table) => table.block_bytes(),
        }
    }

    /// Number of streams, the `S` of a chunk prologue. At most 8.
    #[must_use]
    pub fn stream_count(self) -> usize {
        match self {
            Self::Whole => 1,
            Self::Planes(width) => usize::from(width.param()),
            Self::Fields(table) => table.fields().len(),
        }
    }
}

/// How a delta blob's canonical bytes relate to its base's, from spec section 7a. One variant per
/// registered delta step, so that a later operation is a variant and not a redesign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeltaOp {
    /// `delta_xor`: the stored residual is target XOR base, bytewise. The width only fixes the chain
    /// block, so that no element of the width straddles two chunks.
    Xor(ElementWidth),
}

impl DeltaOp {
    /// The element width of the operation, which is its contribution to the chain block.
    #[must_use]
    pub fn width(self) -> ElementWidth {
        match self {
            Self::Xor(width) => width,
        }
    }
}

/// One valid transform step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Step {
    /// Combine with the base blob's bytes; only first in the chain of a delta blob.
    Delta(DeltaOp),
    /// Rotate every little-endian element left by one bit.
    Rotl1(ElementWidth),
    /// Split into streams.
    Layout(Layout),
}

impl Step {
    /// Converts a stored step.
    ///
    /// # Errors
    /// Returns [`Error::UnknownCodec`] for an id that is not a registered step, which covers
    /// invalid, view, reserved and private ids. Returns [`Error::MalformedIndex`] for section
    /// `CHST` if the id is known and its parameters are outside the registry.
    pub fn from_raw(raw: RawStep) -> Result<Self, Error> {
        let step = match raw.codec_id {
            STEP_WHOLE if raw.param0 == 0 => Some(Self::Layout(Layout::Whole)),
            STEP_WHOLE => None,
            STEP_PLANES => {
                ElementWidth::from_param(raw.param0).map(|w| Self::Layout(Layout::Planes(w)))
            }
            STEP_FIELDS => {
                FieldTable::from_id(raw.param0).map(|t| Self::Layout(Layout::Fields(t)))
            }
            STEP_ROTL1 => ElementWidth::from_param(raw.param0).map(Self::Rotl1),
            STEP_DELTA_XOR => {
                ElementWidth::from_param(raw.param0).map(|w| Self::Delta(DeltaOp::Xor(w)))
            }
            id => return Err(Error::UnknownCodec { id }),
        };
        step.filter(|_| raw.param1 == 0).ok_or_else(|| Error::MalformedIndex {
            section: "CHST",
            detail: format!(
                "step {:#06x} has invalid parameters {} and {}",
                raw.codec_id, raw.param0, raw.param1
            ),
        })
    }

    /// Returns the stored form.
    #[must_use]
    pub fn to_raw(self) -> RawStep {
        let (codec_id, param0) = match self {
            Self::Delta(DeltaOp::Xor(width)) => (STEP_DELTA_XOR, width.param()),
            Self::Rotl1(width) => (STEP_ROTL1, width.param()),
            Self::Layout(Layout::Whole) => (STEP_WHOLE, 0),
            Self::Layout(Layout::Planes(width)) => (STEP_PLANES, width.param()),
            Self::Layout(Layout::Fields(table)) => (STEP_FIELDS, table.id()),
        };
        RawStep { codec_id, param0, param1: 0 }
    }
}

/// A validated transform chain, in encoding order: an optional delta step, an optional `rotl1`,
/// then exactly one layout (spec section 5.2).
///
/// Invalid chains cannot be represented, so holders need no further checks. Whether a delta step
/// matches the kind of the blob using the chain is a blob rule, checked by `format::validate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chain {
    delta: Option<DeltaOp>,
    rotl1: Option<ElementWidth>,
    layout: Layout,
}

impl Chain {
    /// Builds a chain from steps in encoding order.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for section `CHAN` unless `steps` is one layout, preceded
    /// by `rotl1`, by a delta step, or by a delta step then `rotl1`.
    pub fn new(steps: &[Step]) -> Result<Self, Error> {
        let (delta, rest) = match *steps {
            [Step::Delta(op), ref rest @ ..] => (Some(op), rest),
            ref rest => (None, rest),
        };
        match *rest {
            [Step::Layout(layout)] => Ok(Self { delta, rotl1: None, layout }),
            [Step::Rotl1(width), Step::Layout(layout)] => {
                Ok(Self { delta, rotl1: Some(width), layout })
            }
            _ => Err(Error::MalformedIndex {
                section: "CHAN",
                detail: format!("invalid step order: {steps:?}"),
            }),
        }
    }

    /// Builds a chain from stored records in encoding order.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for section `CHAN` if there are no steps or more than
    /// [`MAX_CHAIN_STEPS`], whatever they hold. Otherwise returns the first error of
    /// [`Step::from_raw`], then any error of [`Chain::new`]. Allocates nothing unless it fails.
    pub fn from_raw(raw: &[RawStep]) -> Result<Self, Error> {
        if raw.is_empty() || raw.len() > MAX_CHAIN_STEPS {
            return Err(Error::MalformedIndex {
                section: "CHAN",
                detail: format!("step count {} is outside 1..={MAX_CHAIN_STEPS}", raw.len()),
            });
        }
        // The filler is never read: `raw.len()` slots are overwritten and only those are passed on.
        let mut steps = [Step::Layout(Layout::Whole); MAX_CHAIN_STEPS];
        for (slot, stored) in steps.iter_mut().zip(raw) {
            *slot = Step::from_raw(*stored)?;
        }
        Self::new(&steps[..raw.len()])
    }

    /// The delta step, if the chain has one.
    #[must_use]
    pub fn delta(self) -> Option<DeltaOp> {
        self.delta
    }

    /// The same chain without its delta step: what turns a delta blob's residual into stored bytes
    /// and back. A chain without a delta step is returned unchanged.
    #[must_use]
    pub fn residual(self) -> Self {
        Self { delta: None, ..self }
    }

    /// Width of the `rotl1` step, if the chain has one.
    #[must_use]
    pub fn rotl1(self) -> Option<ElementWidth> {
        self.rotl1
    }

    /// The layout step.
    #[must_use]
    pub fn layout(self) -> Layout {
        self.layout
    }

    /// The steps in encoding order.
    pub fn steps(self) -> impl Iterator<Item = Step> {
        let delta = self.delta.map(Step::Delta).into_iter();
        delta.chain(self.rotl1.map(Step::Rotl1)).chain([Step::Layout(self.layout)])
    }

    /// The chain block of spec section 5.2: `raw_len` and `chunk_size` of a blob must be multiples
    /// of it. The least common multiple of the delta width, the `rotl1` width and the layout block.
    #[must_use]
    pub fn block_bytes(self) -> u32 {
        let widths = self.delta.map(DeltaOp::width).into_iter().chain(self.rotl1);
        widths.fold(self.layout.block_bytes(), |block, width| lcm(width.bytes(), block))
    }

    /// Number of streams, the `S` of a chunk prologue.
    #[must_use]
    pub fn stream_count(self) -> usize {
        self.layout.stream_count()
    }
}

/// Least common multiple of two non-zero block sizes. Inputs are at most 8 and 840 (an earlier lcm of
/// 8 and a field block of at most 210), so the result cannot overflow.
fn lcm(a: u32, b: u32) -> u32 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

/// How an extent presents its blob range, from spec section 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum View {
    /// The bytes as they are.
    Identity,
    /// `u16` binary16 elements presented as `u32` binary32.
    F16ToF32,
    /// `u16` bfloat16 elements presented as `u32` binary32.
    Bf16ToF32,
}

impl View {
    /// Maps an `EXTN.view` value to a view.
    ///
    /// # Errors
    /// Returns [`Error::UnknownCodec`] for any other value.
    pub fn from_wire(id: u16) -> Result<Self, Error> {
        match id {
            VIEW_IDENTITY => Ok(Self::Identity),
            VIEW_F16_TO_F32 => Ok(Self::F16ToF32),
            VIEW_BF16_TO_F32 => Ok(Self::Bf16ToF32),
            id => Err(Error::UnknownCodec { id }),
        }
    }

    /// Returns the `EXTN.view` value.
    #[must_use]
    pub fn to_wire(self) -> u16 {
        match self {
            Self::Identity => VIEW_IDENTITY,
            Self::F16ToF32 => VIEW_F16_TO_F32,
            Self::Bf16ToF32 => VIEW_BF16_TO_F32,
        }
    }

    /// Bytes in the member per byte of the blob: 1 for identity, 2 for widening.
    #[must_use]
    pub fn length_factor(self) -> u64 {
        match self {
            Self::Identity => 1,
            Self::F16ToF32 | Self::Bf16ToF32 => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_TABLES: [FieldTable; 4] =
        [FieldTable::Q80, FieldTable::Q4K, FieldTable::Q5K, FieldTable::Q6K];
    const ALL_WIDTHS: [ElementWidth; 3] =
        [ElementWidth::Two, ElementWidth::Four, ElementWidth::Eight];

    fn raw(codec_id: u16, param0: u16) -> RawStep {
        RawStep { codec_id, param0, param1: 0 }
    }

    fn all_layouts() -> Vec<Layout> {
        let planes = ALL_WIDTHS.map(Layout::Planes);
        let fields = ALL_TABLES.map(Layout::Fields);
        [Layout::Whole].into_iter().chain(planes).chain(fields).collect()
    }

    #[test]
    fn id_values_match_the_spec_registry() {
        assert_eq!(STEP_INVALID, 0x0000);
        assert_eq!(STEP_WHOLE, 0x0100);
        assert_eq!(STEP_PLANES, 0x0101);
        assert_eq!(STEP_FIELDS, 0x0102);
        assert_eq!(STEP_ROTL1, 0x0200);
        assert_eq!(STEP_DELTA_XOR, 0x0400);
        assert_eq!(VIEW_IDENTITY, 0x0000);
        assert_eq!(VIEW_F16_TO_F32, 0x0301);
        assert_eq!(VIEW_BF16_TO_F32, 0x0302);
        assert_eq!(
            (STEP_RESERVED_DELTA_FIRST, STEP_RESERVED_DELTA_LAST),
            (0x0401, 0x04FF)
        );
        assert_eq!(
            (STEP_RESERVED_DERIVED_FIRST, STEP_RESERVED_DERIVED_LAST),
            (0x0500, 0x05FF)
        );
        assert_eq!(STEP_PRIVATE_FIRST, 0x8000);
        assert_eq!(
            (FIELD_TABLE_Q8_0, FIELD_TABLE_Q4_K, FIELD_TABLE_Q5_K, FIELD_TABLE_Q6_K),
            (1, 2, 3, 4)
        );
        assert_eq!((STREAM_STORE, STREAM_ZSTD), (0, 1));
        assert_eq!(
            (BLOB_KIND_CHUNKED, BLOB_KIND_DELTA, BLOB_KIND_RESERVED_DERIVED),
            (0, 1, 2)
        );
        assert_eq!(
            (PROFILE_OPAQUE, PROFILE_SAFETENSORS, PROFILE_GGUF, PROFILE_ONNX, PROFILE_PT_ZIP),
            (0, 1, 2, 3, 4)
        );
        assert_eq!(
            (FEATURE_DELTA, FEATURE_RESERVED_DERIVED, FEATURE_RESERVED_MULTI_VOLUME),
            (1, 2, 4)
        );
        assert_eq!(SUPPORTED_REQUIRED_FEATURES, FEATURE_DELTA);
        assert_eq!(MAX_CHAIN_STEPS, 8);
    }

    #[test]
    fn field_tables_cover_their_blocks_contiguously() {
        for table in ALL_TABLES {
            let mut next = 0;
            for field in table.fields() {
                assert_eq!(field.offset, next, "{table:?}");
                next += field.len;
            }
            assert_eq!(next, table.block_bytes(), "{table:?}");
            assert_eq!(FieldTable::from_id(table.id()), Some(table));
        }
        assert_eq!(FieldTable::Q80.block(), BLOCK_Q8_0);
        assert_eq!(FieldTable::Q4K.block(), BLOCK_Q4_K);
        assert_eq!(FieldTable::Q5K.block(), BLOCK_Q5_K);
        assert_eq!(FieldTable::Q6K.block(), BLOCK_Q6_K);
        assert_eq!(FieldTable::Q80.block_bytes(), 34);
        assert_eq!(FieldTable::Q4K.block_bytes(), 144);
        assert_eq!(FieldTable::Q5K.block_bytes(), 176);
        assert_eq!(FieldTable::Q6K.block_bytes(), 210);
        assert_eq!(Layout::Fields(FieldTable::Q80).stream_count(), 3);
        assert_eq!(Layout::Fields(FieldTable::Q4K).stream_count(), 6);
        assert_eq!(Layout::Fields(FieldTable::Q5K).stream_count(), 7);
        assert_eq!(Layout::Fields(FieldTable::Q6K).stream_count(), 5);
        assert_eq!(FieldTable::from_id(0), None);
        assert_eq!(FieldTable::from_id(5), None);
    }

    #[test]
    fn layouts_alone_are_valid_chains() {
        for layout in all_layouts() {
            let chain = Chain::new(&[Step::Layout(layout)]).expect("one layout is valid");
            assert_eq!(chain.delta(), None);
            assert_eq!(chain.residual(), chain);
            assert_eq!(chain.rotl1(), None);
            assert_eq!(chain.layout(), layout);
            assert_eq!(chain.block_bytes(), layout.block_bytes());
            assert_eq!(chain.stream_count(), layout.stream_count());
        }
    }

    #[test]
    fn rotl1_then_any_layout_is_valid_and_steps_keep_encoding_order() {
        for layout in all_layouts() {
            for width in ALL_WIDTHS {
                let steps = [Step::Rotl1(width), Step::Layout(layout)];
                let chain = Chain::new(&steps).expect("rotl1 then layout is valid");
                assert_eq!(chain.rotl1(), Some(width));
                assert_eq!(chain.steps().collect::<Vec<_>>(), steps);
            }
        }
    }

    #[test]
    fn a_delta_step_first_is_valid_and_residual_drops_only_it() {
        for layout in all_layouts() {
            for width in ALL_WIDTHS {
                let delta = Step::Delta(DeltaOp::Xor(width));
                let plain = Chain::new(&[Step::Layout(layout)]).expect("valid");
                let chain = Chain::new(&[delta, Step::Layout(layout)]).expect("delta then layout");
                assert_eq!(chain.delta(), Some(DeltaOp::Xor(width)));
                assert_eq!(chain.residual(), plain);
                assert_eq!(chain.steps().collect::<Vec<_>>(), [delta, Step::Layout(layout)]);
                let steps = [delta, Step::Rotl1(width), Step::Layout(layout)];
                let rotated = Chain::new(&steps).expect("delta, rotl1, layout");
                assert_eq!(rotated.rotl1(), Some(width));
                assert_eq!(rotated.residual().delta(), None);
                assert_eq!(rotated.residual().rotl1(), Some(width));
                assert_eq!(rotated.steps().collect::<Vec<_>>(), steps);
                let stored: Vec<RawStep> = rotated.steps().map(Step::to_raw).collect();
                assert_eq!(Chain::from_raw(&stored).expect("valid"), rotated);
            }
        }
    }

    #[test]
    fn delta_steps_out_of_place_are_rejected() {
        let delta = Step::Delta(DeltaOp::Xor(ElementWidth::Two));
        let rot = Step::Rotl1(ElementWidth::Two);
        let whole = Step::Layout(Layout::Whole);
        let cases: [&[Step]; 6] = [
            &[delta],
            &[delta, delta, whole],
            &[rot, delta, whole],
            &[whole, delta],
            &[delta, rot],
            &[delta, rot, rot, whole],
        ];
        for steps in cases {
            let err = Chain::new(steps).expect_err("invalid chain");
            assert!(matches!(err, Error::MalformedIndex { section: "CHAN", .. }), "{err:?}");
        }
    }

    #[test]
    fn delta_xor_parameters_follow_the_registry() {
        for width in ALL_WIDTHS {
            let stored = raw(STEP_DELTA_XOR, width.param());
            let step = Step::Delta(DeltaOp::Xor(width));
            assert_eq!(Step::from_raw(stored).expect("valid"), step);
            assert_eq!(step.to_raw(), stored);
        }
        let cases = [
            raw(STEP_DELTA_XOR, 0),
            raw(STEP_DELTA_XOR, 1),
            raw(STEP_DELTA_XOR, 3),
            raw(STEP_DELTA_XOR, 16),
            RawStep { codec_id: STEP_DELTA_XOR, param0: 2, param1: 1 },
        ];
        for stored in cases {
            let err = Step::from_raw(stored).expect_err("invalid parameters");
            assert!(matches!(err, Error::MalformedIndex { section: "CHST", .. }), "{err:?}");
        }
    }

    #[test]
    fn the_delta_width_joins_the_chain_block() {
        let block = |steps: &[Step]| Chain::new(steps).expect("valid").block_bytes();
        let xor = |width| Step::Delta(DeltaOp::Xor(width));
        let planes = |width| Step::Layout(Layout::Planes(width));
        assert_eq!(block(&[xor(ElementWidth::Four), planes(ElementWidth::Two)]), 4);
        assert_eq!(block(&[xor(ElementWidth::Two), planes(ElementWidth::Two)]), 2);
        assert_eq!(block(&[xor(ElementWidth::Two), Step::Layout(Layout::Whole)]), 2);
        let q80 = Step::Layout(Layout::Fields(FieldTable::Q80));
        assert_eq!(block(&[xor(ElementWidth::Four), q80]), 68);
        let rot = Step::Rotl1(ElementWidth::Eight);
        let q6k = Step::Layout(Layout::Fields(FieldTable::Q6K));
        assert_eq!(block(&[xor(ElementWidth::Eight), rot, q6k]), 840);
    }

    #[test]
    fn stored_steps_round_trip() {
        for layout in all_layouts() {
            let chain = Chain::new(&[Step::Rotl1(ElementWidth::Four), Step::Layout(layout)])
                .expect("valid");
            let stored: Vec<RawStep> = chain.steps().map(Step::to_raw).collect();
            assert_eq!(Chain::from_raw(&stored).expect("valid"), chain);
        }
    }

    #[test]
    fn stored_forms_match_the_registry() {
        let cases = [
            (Step::Layout(Layout::Whole), raw(0x0100, 0)),
            (Step::Layout(Layout::Planes(ElementWidth::Two)), raw(0x0101, 2)),
            (Step::Layout(Layout::Planes(ElementWidth::Eight)), raw(0x0101, 8)),
            (Step::Layout(Layout::Fields(FieldTable::Q6K)), raw(0x0102, 4)),
            (Step::Rotl1(ElementWidth::Four), raw(0x0200, 4)),
        ];
        for (step, stored) in cases {
            assert_eq!(step.to_raw(), stored);
            assert_eq!(Step::from_raw(stored).expect("valid"), step);
        }
    }

    #[test]
    fn chain_block_is_the_least_common_multiple() {
        let block = |width, layout| {
            Chain::new(&[Step::Rotl1(width), Step::Layout(layout)]).expect("valid").block_bytes()
        };
        assert_eq!(block(ElementWidth::Eight, Layout::Whole), 8);
        assert_eq!(block(ElementWidth::Four, Layout::Planes(ElementWidth::Four)), 4);
        assert_eq!(block(ElementWidth::Two, Layout::Planes(ElementWidth::Eight)), 8);
        assert_eq!(block(ElementWidth::Eight, Layout::Planes(ElementWidth::Two)), 8);
        assert_eq!(block(ElementWidth::Two, Layout::Fields(FieldTable::Q80)), 34);
        assert_eq!(block(ElementWidth::Four, Layout::Fields(FieldTable::Q80)), 68);
        assert_eq!(block(ElementWidth::Eight, Layout::Fields(FieldTable::Q80)), 136);
        assert_eq!(block(ElementWidth::Eight, Layout::Fields(FieldTable::Q4K)), 144);
        assert_eq!(block(ElementWidth::Eight, Layout::Fields(FieldTable::Q5K)), 176);
        assert_eq!(block(ElementWidth::Eight, Layout::Fields(FieldTable::Q6K)), 840);
    }

    #[test]
    fn invalid_step_orders_are_rejected() {
        let rot = Step::Rotl1(ElementWidth::Two);
        let whole = Step::Layout(Layout::Whole);
        let cases: [&[Step]; 7] = [
            &[],
            &[rot],
            &[rot, rot],
            &[whole, rot],
            &[whole, whole],
            &[rot, whole, whole],
            &[rot, rot, whole],
        ];
        for steps in cases {
            let err = Chain::new(steps).expect_err("invalid chain");
            assert!(matches!(err, Error::MalformedIndex { section: "CHAN", .. }), "{err:?}");
        }
    }

    #[test]
    fn step_count_limits_come_before_step_contents() {
        for count in [0, MAX_CHAIN_STEPS + 1] {
            let stored = vec![raw(0x7777, 0); count];
            let err = Chain::from_raw(&stored).expect_err("count out of range");
            assert!(matches!(err, Error::MalformedIndex { section: "CHAN", .. }), "{err:?}");
        }
    }

    #[test]
    fn valid_steps_within_the_limit_but_too_many_are_rejected() {
        let stored = vec![raw(STEP_WHOLE, 0); MAX_CHAIN_STEPS];
        let err = Chain::from_raw(&stored).expect_err("too many layouts");
        assert!(matches!(err, Error::MalformedIndex { section: "CHAN", .. }), "{err:?}");
    }

    #[test]
    fn unknown_ids_are_reported_as_unknown_codec() {
        let ids = [
            STEP_INVALID,
            0x0103,
            0x0201,
            VIEW_F16_TO_F32,
            VIEW_BF16_TO_F32,
            STEP_RESERVED_DELTA_FIRST,
            STEP_RESERVED_DELTA_LAST,
            STEP_RESERVED_DERIVED_FIRST,
            STEP_RESERVED_DERIVED_LAST,
            STEP_PRIVATE_FIRST,
            u16::MAX,
        ];
        for id in ids {
            let err = Step::from_raw(raw(id, 0)).expect_err("not a step");
            assert!(matches!(err, Error::UnknownCodec { id: got } if got == id), "{err:?}");
            let err = Chain::from_raw(&[raw(STEP_WHOLE, 0), raw(id, 0)]).expect_err("not a step");
            assert!(matches!(err, Error::UnknownCodec { id: got } if got == id), "{err:?}");
        }
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let cases = [
            raw(STEP_WHOLE, 1),
            raw(STEP_PLANES, 0),
            raw(STEP_PLANES, 1),
            raw(STEP_PLANES, 3),
            raw(STEP_PLANES, 16),
            raw(STEP_FIELDS, 0),
            raw(STEP_FIELDS, 5),
            raw(STEP_ROTL1, 1),
            raw(STEP_ROTL1, 3),
            RawStep { codec_id: STEP_WHOLE, param0: 0, param1: 1 },
            RawStep { codec_id: STEP_PLANES, param0: 4, param1: 1 },
            RawStep { codec_id: STEP_FIELDS, param0: 1, param1: u32::MAX },
            RawStep { codec_id: STEP_ROTL1, param0: 2, param1: 1 },
        ];
        for stored in cases {
            let err = Step::from_raw(stored).expect_err("invalid parameters");
            assert!(matches!(err, Error::MalformedIndex { section: "CHST", .. }), "{err:?}");
        }
    }

    #[test]
    fn views_map_to_and_from_the_wire() {
        let cases = [
            (View::Identity, 0x0000, 1),
            (View::F16ToF32, 0x0301, 2),
            (View::Bf16ToF32, 0x0302, 2),
        ];
        for (view, id, factor) in cases {
            assert_eq!(view.to_wire(), id);
            assert_eq!(View::from_wire(id).expect("registered"), view);
            assert_eq!(view.length_factor(), factor);
        }
    }

    #[test]
    fn unknown_views_are_reported_as_unknown_codec() {
        for id in [STEP_WHOLE, STEP_ROTL1, 0x0300, 0x0303, STEP_PRIVATE_FIRST, u16::MAX] {
            let err = View::from_wire(id).expect_err("not a view");
            assert!(matches!(err, Error::UnknownCodec { id: got } if got == id), "{err:?}");
        }
    }
}
