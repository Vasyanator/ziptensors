//! File: types.rs
//! Purpose: plain shared types used across `ziptensors-core` and by every frontend.
//!
//! Main responsibilities:
//! - the `dtype` and `profile` id registries of the format, as enums with lossless wire mapping;
//! - per dtype: its registry name, its block geometry (from `chain`) and its field table, if any;
//! - writer modes and reader verification levels;
//! - the plan a profile parser hands to the writer.
//!
//! Key structures: [`Dtype`], [`Profile`], [`Mode`], [`Verify`], [`TypedRange`], [`TensorInfo`],
//! [`RangePlan`].
//!
//! Implementation notes: `dtype` wire ids and names here must match `spec/ziptensors-format.md`; a
//! change to one requires the other. Profile ids and quantized block geometry belong to
//! `chain.rs`. Both id spaces are hints, so an id unknown to this build is preserved rather than
//! rejected.

use crate::chain::{
    self, BlockGeometry, FieldTable, PROFILE_GGUF, PROFILE_ONNX, PROFILE_OPAQUE, PROFILE_PT_ZIP,
    PROFILE_SAFETENSORS,
};

/// Element type of a tensor or blob, as named by the `dtype` registry of the spec.
///
/// A dtype is a hint: it never decides how bytes are restored. Ids this build does not know are
/// legal and carried in [`Dtype::Unknown`]; id 0 means no dtype was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// 1-byte boolean.
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE binary16.
    F16,
    /// bfloat16.
    Bf16,
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// 8-bit float with 4 exponent and 3 mantissa bits.
    F8E4M3,
    /// 8-bit float with 5 exponent and 2 mantissa bits.
    F8E5M2,
    /// Packed 4-bit values stored in bytes.
    Nibble,
    /// ggml `block_q8_0`.
    Q80,
    /// ggml `block_q4_K`.
    Q4K,
    /// ggml `block_q5_K`.
    Q5K,
    /// ggml `block_q6_K`.
    Q6K,
    /// ggml `block_q4_0`.
    Q40,
    /// ggml `block_q4_1`.
    Q41,
    /// ggml `block_q5_0`.
    Q50,
    /// ggml `block_q5_1`.
    Q51,
    /// ggml `block_q8_1`.
    Q81,
    /// ggml `block_q2_K`.
    Q2K,
    /// ggml `block_q3_K`.
    Q3K,
    /// ggml `block_q8_K`.
    Q8K,
    /// ggml `block_iq2_xxs`.
    Iq2Xxs,
    /// ggml `block_iq2_xs`.
    Iq2Xs,
    /// ggml `block_iq3_xxs`.
    Iq3Xxs,
    /// ggml `block_iq1_s`.
    Iq1S,
    /// ggml `block_iq4_nl`.
    Iq4Nl,
    /// ggml `block_iq3_s`.
    Iq3S,
    /// ggml `block_iq2_s`.
    Iq2S,
    /// ggml `block_iq4_xs`.
    Iq4Xs,
    /// ggml `block_iq1_m`.
    Iq1M,
    /// ggml `block_tq1_0`.
    Tq10,
    /// ggml `block_tq2_0`.
    Tq20,
    /// ggml `block_mxfp4`.
    Mxfp4,
    /// An id outside the registry above, or 0. Build values with [`Dtype::from_wire`], which never
    /// wraps a registered id here.
    Unknown(u16),
}

impl Dtype {
    /// Maps a wire id to a dtype. Total: every `u16` is legal, unregistered ids become
    /// [`Dtype::Unknown`].
    #[must_use]
    pub fn from_wire(id: u16) -> Self {
        match id {
            0x0001 => Self::Bool,
            0x0002 => Self::U8,
            0x0003 => Self::I8,
            0x0004 => Self::U16,
            0x0005 => Self::I16,
            0x0006 => Self::U32,
            0x0007 => Self::I32,
            0x0008 => Self::U64,
            0x0009 => Self::I64,
            0x0010 => Self::F16,
            0x0011 => Self::Bf16,
            0x0012 => Self::F32,
            0x0013 => Self::F64,
            0x0020 => Self::F8E4M3,
            0x0021 => Self::F8E5M2,
            0x0022 => Self::Nibble,
            0x0030 => Self::Q80,
            0x0031 => Self::Q4K,
            0x0032 => Self::Q5K,
            0x0033 => Self::Q6K,
            0x0034 => Self::Q40,
            0x0035 => Self::Q41,
            0x0036 => Self::Q50,
            0x0037 => Self::Q51,
            0x0038 => Self::Q81,
            0x0039 => Self::Q2K,
            0x003A => Self::Q3K,
            0x003B => Self::Q8K,
            0x003C => Self::Iq2Xxs,
            0x003D => Self::Iq2Xs,
            0x003E => Self::Iq3Xxs,
            0x003F => Self::Iq1S,
            0x0040 => Self::Iq4Nl,
            0x0041 => Self::Iq3S,
            0x0042 => Self::Iq2S,
            0x0043 => Self::Iq4Xs,
            0x0044 => Self::Iq1M,
            0x0045 => Self::Tq10,
            0x0046 => Self::Tq20,
            0x0047 => Self::Mxfp4,
            other => Self::Unknown(other),
        }
    }

    /// Returns the wire id. For every id, `Dtype::from_wire(id).to_wire() == id`.
    #[must_use]
    pub fn to_wire(self) -> u16 {
        match self {
            Self::Bool => 0x0001,
            Self::U8 => 0x0002,
            Self::I8 => 0x0003,
            Self::U16 => 0x0004,
            Self::I16 => 0x0005,
            Self::U32 => 0x0006,
            Self::I32 => 0x0007,
            Self::U64 => 0x0008,
            Self::I64 => 0x0009,
            Self::F16 => 0x0010,
            Self::Bf16 => 0x0011,
            Self::F32 => 0x0012,
            Self::F64 => 0x0013,
            Self::F8E4M3 => 0x0020,
            Self::F8E5M2 => 0x0021,
            Self::Nibble => 0x0022,
            Self::Q80 => 0x0030,
            Self::Q4K => 0x0031,
            Self::Q5K => 0x0032,
            Self::Q6K => 0x0033,
            Self::Q40 => 0x0034,
            Self::Q41 => 0x0035,
            Self::Q50 => 0x0036,
            Self::Q51 => 0x0037,
            Self::Q81 => 0x0038,
            Self::Q2K => 0x0039,
            Self::Q3K => 0x003A,
            Self::Q8K => 0x003B,
            Self::Iq2Xxs => 0x003C,
            Self::Iq2Xs => 0x003D,
            Self::Iq3Xxs => 0x003E,
            Self::Iq1S => 0x003F,
            Self::Iq4Nl => 0x0040,
            Self::Iq3S => 0x0041,
            Self::Iq2S => 0x0042,
            Self::Iq4Xs => 0x0043,
            Self::Iq1M => 0x0044,
            Self::Tq10 => 0x0045,
            Self::Tq20 => 0x0046,
            Self::Mxfp4 => 0x0047,
            Self::Unknown(id) => id,
        }
    }

    /// Size in bytes of the smallest unit a range of this dtype can be cut at: the element, or
    /// the whole block for block-quantized types. `None` for [`Dtype::Unknown`], whose unit is not
    /// known.
    #[must_use]
    pub fn block_bytes(self) -> Option<u32> {
        match self.block_geometry() {
            Some(geometry) => Some(geometry.bytes),
            None if self == Self::Nibble => Some(1),
            None => None,
        }
    }

    /// How many elements fit in how many bytes: one element of its width for a plain type, one
    /// ggml block for a block-quantized type. The block table is owned by `chain` and matches the
    /// `GGML_QUANT_SIZES` of the reference `gguf` package.
    ///
    /// `None` for [`Dtype::Unknown`], and for [`Dtype::Nibble`], whose element count per byte the
    /// spec does not define. When it is `Some`, its `bytes` equals [`Dtype::block_bytes`].
    #[must_use]
    pub fn block_geometry(self) -> Option<BlockGeometry> {
        let element = |bytes| Some(BlockGeometry { elements: 1, bytes });
        let quantized = match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => return element(1),
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => return element(2),
            Self::U32 | Self::I32 | Self::F32 => return element(4),
            Self::U64 | Self::I64 | Self::F64 => return element(8),
            Self::Nibble | Self::Unknown(_) => return None,
            Self::Q80 => chain::BLOCK_Q8_0,
            Self::Q4K => chain::BLOCK_Q4_K,
            Self::Q5K => chain::BLOCK_Q5_K,
            Self::Q6K => chain::BLOCK_Q6_K,
            Self::Q40 => chain::BLOCK_Q4_0,
            Self::Q41 => chain::BLOCK_Q4_1,
            Self::Q50 => chain::BLOCK_Q5_0,
            Self::Q51 => chain::BLOCK_Q5_1,
            Self::Q81 => chain::BLOCK_Q8_1,
            Self::Q2K => chain::BLOCK_Q2_K,
            Self::Q3K => chain::BLOCK_Q3_K,
            Self::Q8K => chain::BLOCK_Q8_K,
            Self::Iq2Xxs => chain::BLOCK_IQ2_XXS,
            Self::Iq2Xs => chain::BLOCK_IQ2_XS,
            Self::Iq3Xxs => chain::BLOCK_IQ3_XXS,
            Self::Iq1S => chain::BLOCK_IQ1_S,
            Self::Iq4Nl => chain::BLOCK_IQ4_NL,
            Self::Iq3S => chain::BLOCK_IQ3_S,
            Self::Iq2S => chain::BLOCK_IQ2_S,
            Self::Iq4Xs => chain::BLOCK_IQ4_XS,
            Self::Iq1M => chain::BLOCK_IQ1_M,
            Self::Tq10 => chain::BLOCK_TQ1_0,
            Self::Tq20 => chain::BLOCK_TQ2_0,
            Self::Mxfp4 => chain::BLOCK_MXFP4,
        };
        Some(quantized)
    }

    /// The field table of spec section 6.3 that splits blocks of this dtype, if there is one. Only
    /// `Q8_0`, `Q4_K`, `Q5_K` and `Q6_K` have one; other block-quantized types have none.
    #[must_use]
    pub fn field_table(self) -> Option<FieldTable> {
        match self {
            Self::Q80 => Some(FieldTable::Q80),
            Self::Q4K => Some(FieldTable::Q4K),
            Self::Q5K => Some(FieldTable::Q5K),
            Self::Q6K => Some(FieldTable::Q6K),
            Self::Bool
            | Self::U8
            | Self::I8
            | Self::U16
            | Self::I16
            | Self::U32
            | Self::I32
            | Self::U64
            | Self::I64
            | Self::F16
            | Self::Bf16
            | Self::F32
            | Self::F64
            | Self::F8E4M3
            | Self::F8E5M2
            | Self::Nibble
            | Self::Q40
            | Self::Q41
            | Self::Q50
            | Self::Q51
            | Self::Q81
            | Self::Q2K
            | Self::Q3K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq4Nl
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M
            | Self::Tq10
            | Self::Tq20
            | Self::Mxfp4
            | Self::Unknown(_) => None,
        }
    }

    /// The registry name of spec section 5.3, such as `F32` or `Q4_K`. For a ggml type it equals
    /// the member name of `GGMLQuantizationType` in the reference `gguf` package. `None` for
    /// [`Dtype::Unknown`], which includes id 0.
    #[must_use]
    pub fn name(self) -> Option<&'static str> {
        let name = match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::Nibble => "NIBBLE",
            Self::Q80 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q40 => "Q4_0",
            Self::Q41 => "Q4_1",
            Self::Q50 => "Q5_0",
            Self::Q51 => "Q5_1",
            Self::Q81 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::Iq1M => "IQ1_M",
            Self::Tq10 => "TQ1_0",
            Self::Tq20 => "TQ2_0",
            Self::Mxfp4 => "MXFP4",
            Self::Unknown(_) => return None,
        };
        Some(name)
    }
}

/// Source file format a member was recognized as, by content and never by file name.
///
/// Informational in the container: unknown ids are legal and carried in [`Profile::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// No structure recognized, or the parser gave up. The file is treated as bytes.
    Opaque,
    /// safetensors.
    Safetensors,
    /// GGUF.
    Gguf,
    /// ONNX protobuf.
    Onnx,
    /// `PyTorch` ZIP archive.
    PtZip,
    /// An id outside the registry above. Build values with [`Profile::from_wire`].
    Unknown(u8),
}

impl Profile {
    /// Maps a wire id to a profile. Total: unregistered ids become [`Profile::Unknown`].
    #[must_use]
    pub fn from_wire(id: u8) -> Self {
        match id {
            PROFILE_OPAQUE => Self::Opaque,
            PROFILE_SAFETENSORS => Self::Safetensors,
            PROFILE_GGUF => Self::Gguf,
            PROFILE_ONNX => Self::Onnx,
            PROFILE_PT_ZIP => Self::PtZip,
            other => Self::Unknown(other),
        }
    }

    /// Returns the wire id. For every id, `Profile::from_wire(id).to_wire() == id`.
    #[must_use]
    pub fn to_wire(self) -> u8 {
        match self {
            Self::Opaque => PROFILE_OPAQUE,
            Self::Safetensors => PROFILE_SAFETENSORS,
            Self::Gguf => PROFILE_GGUF,
            Self::Onnx => PROFILE_ONNX,
            Self::PtZip => PROFILE_PT_ZIP,
            Self::Unknown(id) => id,
        }
    }
}

/// Writer effort. Affects which transforms are tried, never the meaning of the file. There is no
/// `Default`: `pack` defaults to [`Mode::Max`] and `encode` to [`Mode::Balanced`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Plane split and per-stream entropy coding only; quantized data is stored.
    Fast,
    /// Adds widening views and field split of quantized data.
    Balanced,
    /// Adds the bit rotation trial per blob and blob deduplication.
    Max,
}

/// What a reader checks while decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Verify {
    /// Check the XXH3-64 of every decoded chunk.
    #[default]
    Chunk,
    /// Skip chunk hashes. Exact decoded lengths are still enforced.
    Off,
}

/// One byte range of a source file that a profile parser recognized as uniform data.
///
/// A hint for the writer: the range only selects a transform, and the extents the writer emits
/// cover the file whatever the plan says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedRange {
    /// Start in the source file, in bytes.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
    /// Element type the parser believes the range holds.
    pub hint: Dtype,
}

/// One named tensor of a member: its range in the member's virtual file and its description.
///
/// Metadata only. Restore never consults it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Name as found in the source file.
    pub name: String,
    /// Start in the member, in bytes.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
    /// Dimensions, at most 16.
    pub shape: Vec<u64>,
    /// Element type.
    pub dtype: Dtype,
}

/// What a profile parser found in a source file.
///
/// `ranges` drive compression and may be dropped or trimmed by normalization. `tensors` become the
/// tensor table and stay reachable by name even when their bytes remain in the skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangePlan {
    /// The recognized format.
    pub profile: Profile,
    /// Data ranges to compress as typed blobs.
    pub ranges: Vec<TypedRange>,
    /// Tensor table of the file.
    pub tensors: Vec<TensorInfo>,
}

#[cfg(test)]
mod tests {
    use super::{Dtype, Profile, Verify};
    use crate::chain::BlockGeometry;
    use std::collections::HashSet;

    /// `GGML_QUANT_SIZES` of the reference `gguf` Python package 0.18.0 (`gguf/constants.py`), as
    /// `(GGMLQuantizationType name, block elements, block bytes)`, evaluated, for every type with a
    /// block of more than one element. Written out here so that a drift in `chain` fails a test.
    const GGUF_QUANT_SIZES: [(&str, u32, u32); 24] = [
        ("Q4_0", 32, 18),
        ("Q4_1", 32, 20),
        ("Q5_0", 32, 22),
        ("Q5_1", 32, 24),
        ("Q8_0", 32, 34),
        ("Q8_1", 32, 40),
        ("Q2_K", 256, 84),
        ("Q3_K", 256, 110),
        ("Q4_K", 256, 144),
        ("Q5_K", 256, 176),
        ("Q6_K", 256, 210),
        ("Q8_K", 256, 292),
        ("IQ2_XXS", 256, 66),
        ("IQ2_XS", 256, 74),
        ("IQ3_XXS", 256, 98),
        ("IQ1_S", 256, 50),
        ("IQ4_NL", 32, 18),
        ("IQ3_S", 256, 110),
        ("IQ2_S", 256, 82),
        ("IQ4_XS", 256, 136),
        ("IQ1_M", 256, 56),
        ("TQ1_0", 256, 54),
        ("TQ2_0", 256, 66),
        ("MXFP4", 32, 17),
    ];

    /// Every registered dtype, found by scanning the whole id space.
    fn registered() -> Vec<Dtype> {
        (0..=u16::MAX)
            .map(Dtype::from_wire)
            .filter(|dtype| !matches!(dtype, Dtype::Unknown(_)))
            .collect()
    }

    /// The registered dtype named `name`.
    fn by_name(name: &str) -> Dtype {
        registered().into_iter().find(|dtype| dtype.name() == Some(name)).expect(name)
    }

    #[test]
    fn dtype_wire_mapping_is_lossless_for_every_id() {
        for id in 0..=u16::MAX {
            assert_eq!(Dtype::from_wire(id).to_wire(), id);
        }
    }

    #[test]
    fn dtype_registered_ids_never_become_unknown() {
        let registered = (0..=u16::MAX)
            .filter(|&id| !matches!(Dtype::from_wire(id), Dtype::Unknown(_)))
            .count();
        assert_eq!(registered, 40);
        assert_eq!(Dtype::from_wire(0), Dtype::Unknown(0));
        assert_eq!(Dtype::from_wire(0x0012), Dtype::F32);
        assert_eq!(Dtype::from_wire(0x0030), Dtype::Q80);
        assert_eq!(Dtype::from_wire(0x0034), Dtype::Q40);
        assert_eq!(Dtype::from_wire(0x0047), Dtype::Mxfp4);
        assert_eq!(Dtype::from_wire(0x0048), Dtype::Unknown(0x0048));
    }

    #[test]
    fn names_and_ids_are_one_to_one() {
        let dtypes = registered();
        let names: HashSet<&str> = dtypes.iter().filter_map(|dtype| dtype.name()).collect();
        assert_eq!(names.len(), dtypes.len(), "every registered dtype has its own name");
        for dtype in dtypes {
            let name = dtype.name().expect("registered");
            assert_eq!(by_name(name).to_wire(), dtype.to_wire(), "{name}");
        }
        assert_eq!(Dtype::Unknown(0).name(), None);
        assert_eq!(Dtype::Unknown(0x7777).name(), None);
    }

    #[test]
    fn spec_registry_ids_and_names_of_ggml_types() {
        // The rows of spec section 5.3 from 0x0030 on, in id order.
        let names = [
            "Q8_0", "Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_1", "Q2_K", "Q3_K",
            "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
            "IQ1_M", "TQ1_0", "TQ2_0", "MXFP4",
        ];
        for (id, name) in (0x0030_u16..).zip(names) {
            assert_eq!(Dtype::from_wire(id).name(), Some(name), "{id:#06x}");
        }
    }

    #[test]
    fn block_geometry_matches_the_gguf_package() {
        for (name, elements, bytes) in GGUF_QUANT_SIZES {
            let dtype = by_name(name);
            assert_eq!(dtype.block_geometry(), Some(BlockGeometry { elements, bytes }), "{name}");
            assert_eq!(dtype.block_bytes(), Some(bytes), "{name}");
        }
        let element = |bytes| Some(BlockGeometry { elements: 1, bytes });
        assert_eq!(Dtype::F32.block_geometry(), element(4));
        assert_eq!(Dtype::Bf16.block_geometry(), element(2));
        assert_eq!(Dtype::F8E5M2.block_geometry(), element(1));
        assert_eq!(Dtype::I64.block_geometry(), element(8));
        assert_eq!(Dtype::Nibble.block_geometry(), None);
        assert_eq!(Dtype::Unknown(0x7777).block_geometry(), None);
    }

    #[test]
    fn geometry_agrees_with_block_bytes_for_every_id() {
        for id in 0..=u16::MAX {
            let dtype = Dtype::from_wire(id);
            if let Some(geometry) = dtype.block_geometry() {
                assert_eq!(dtype.block_bytes(), Some(geometry.bytes), "{id:#06x}");
            }
        }
    }

    #[test]
    fn only_the_four_split_types_have_a_field_table() {
        let with_table: Vec<Dtype> =
            registered().into_iter().filter(|dtype| dtype.field_table().is_some()).collect();
        assert_eq!(with_table, [Dtype::Q80, Dtype::Q4K, Dtype::Q5K, Dtype::Q6K]);
        for dtype in with_table {
            let table = dtype.field_table().expect("listed");
            assert_eq!(dtype.block_geometry(), Some(table.block()));
        }
    }

    #[test]
    fn profile_wire_mapping_is_lossless_for_every_id() {
        for id in 0..=u8::MAX {
            assert_eq!(Profile::from_wire(id).to_wire(), id);
        }
        assert_eq!(Profile::from_wire(4), Profile::PtZip);
        assert_eq!(Profile::from_wire(5), Profile::Unknown(5));
    }

    #[test]
    fn block_bytes_match_the_layouts_in_the_spec() {
        assert_eq!(Dtype::U8.block_bytes(), Some(1));
        assert_eq!(Dtype::Nibble.block_bytes(), Some(1));
        assert_eq!(Dtype::Bf16.block_bytes(), Some(2));
        assert_eq!(Dtype::F32.block_bytes(), Some(4));
        assert_eq!(Dtype::F64.block_bytes(), Some(8));
        assert_eq!(Dtype::Q80.block_bytes(), Some(34));
        assert_eq!(Dtype::Q4K.block_bytes(), Some(144));
        assert_eq!(Dtype::Q5K.block_bytes(), Some(176));
        assert_eq!(Dtype::Q6K.block_bytes(), Some(210));
        assert_eq!(Dtype::Unknown(0x7777).block_bytes(), None);
    }

    #[test]
    fn verify_defaults_to_chunk() {
        assert_eq!(Verify::default(), Verify::Chunk);
    }
}
