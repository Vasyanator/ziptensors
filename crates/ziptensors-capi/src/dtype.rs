//! File: dtype.rs
//! Purpose: the dtype wire ids as header constants, so that a C caller can read `zt_tensor_desc.dtype`.
//!
//! Implementation notes: literals, because cbindgen does not evaluate `Dtype::F32.to_wire()`. The
//! registry stays in core and the spec: a test pins each constant to `Dtype::from_wire`.

/// `zt_tensor_desc.dtype`: no dtype was determined.
pub const ZT_DTYPE_NONE: u16 = 0x0000;
/// 1-byte boolean.
pub const ZT_DTYPE_BOOL: u16 = 0x0001;
/// Unsigned 8-bit integer.
pub const ZT_DTYPE_U8: u16 = 0x0002;
/// Signed 8-bit integer.
pub const ZT_DTYPE_I8: u16 = 0x0003;
/// Unsigned 16-bit integer.
pub const ZT_DTYPE_U16: u16 = 0x0004;
/// Signed 16-bit integer.
pub const ZT_DTYPE_I16: u16 = 0x0005;
/// Unsigned 32-bit integer.
pub const ZT_DTYPE_U32: u16 = 0x0006;
/// Signed 32-bit integer.
pub const ZT_DTYPE_I32: u16 = 0x0007;
/// Unsigned 64-bit integer.
pub const ZT_DTYPE_U64: u16 = 0x0008;
/// Signed 64-bit integer.
pub const ZT_DTYPE_I64: u16 = 0x0009;
/// IEEE binary16.
pub const ZT_DTYPE_F16: u16 = 0x0010;
/// bfloat16.
pub const ZT_DTYPE_BF16: u16 = 0x0011;
/// IEEE binary32.
pub const ZT_DTYPE_F32: u16 = 0x0012;
/// IEEE binary64.
pub const ZT_DTYPE_F64: u16 = 0x0013;
/// 8-bit float with 4 exponent and 3 mantissa bits.
pub const ZT_DTYPE_F8_E4M3: u16 = 0x0020;
/// 8-bit float with 5 exponent and 2 mantissa bits.
pub const ZT_DTYPE_F8_E5M2: u16 = 0x0021;
/// Packed 4-bit values stored in bytes.
pub const ZT_DTYPE_NIBBLE: u16 = 0x0022;
/// ggml `block_q8_0`.
pub const ZT_DTYPE_Q8_0: u16 = 0x0030;
/// ggml `block_q4_K`.
pub const ZT_DTYPE_Q4_K: u16 = 0x0031;
/// ggml `block_q5_K`.
pub const ZT_DTYPE_Q5_K: u16 = 0x0032;
/// ggml `block_q6_K`.
pub const ZT_DTYPE_Q6_K: u16 = 0x0033;
/// ggml `block_q4_0`.
pub const ZT_DTYPE_Q4_0: u16 = 0x0034;
/// ggml `block_q4_1`.
pub const ZT_DTYPE_Q4_1: u16 = 0x0035;
/// ggml `block_q5_0`.
pub const ZT_DTYPE_Q5_0: u16 = 0x0036;
/// ggml `block_q5_1`.
pub const ZT_DTYPE_Q5_1: u16 = 0x0037;
/// ggml `block_q8_1`.
pub const ZT_DTYPE_Q8_1: u16 = 0x0038;
/// ggml `block_q2_K`.
pub const ZT_DTYPE_Q2_K: u16 = 0x0039;
/// ggml `block_q3_K`.
pub const ZT_DTYPE_Q3_K: u16 = 0x003A;
/// ggml `block_q8_K`.
pub const ZT_DTYPE_Q8_K: u16 = 0x003B;
/// ggml `block_iq2_xxs`.
pub const ZT_DTYPE_IQ2_XXS: u16 = 0x003C;
/// ggml `block_iq2_xs`.
pub const ZT_DTYPE_IQ2_XS: u16 = 0x003D;
/// ggml `block_iq3_xxs`.
pub const ZT_DTYPE_IQ3_XXS: u16 = 0x003E;
/// ggml `block_iq1_s`.
pub const ZT_DTYPE_IQ1_S: u16 = 0x003F;
/// ggml `block_iq4_nl`.
pub const ZT_DTYPE_IQ4_NL: u16 = 0x0040;
/// ggml `block_iq3_s`.
pub const ZT_DTYPE_IQ3_S: u16 = 0x0041;
/// ggml `block_iq2_s`.
pub const ZT_DTYPE_IQ2_S: u16 = 0x0042;
/// ggml `block_iq4_xs`.
pub const ZT_DTYPE_IQ4_XS: u16 = 0x0043;
/// ggml `block_iq1_m`.
pub const ZT_DTYPE_IQ1_M: u16 = 0x0044;
/// ggml `block_tq1_0`.
pub const ZT_DTYPE_TQ1_0: u16 = 0x0045;
/// ggml `block_tq2_0`.
pub const ZT_DTYPE_TQ2_0: u16 = 0x0046;
/// ggml `block_mxfp4`.
pub const ZT_DTYPE_MXFP4: u16 = 0x0047;
