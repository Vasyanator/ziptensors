//! File: codec/widen.rs
//! Purpose: the widening views of spec section 7 and their exact inverse for writers.
//!
//! Main responsibilities: present narrow canonical bytes as `f32` bytes, and decide whether `f32`
//! bytes can be stored narrow with no change to any bit.
//!
//! Key functions: [`widen_into`], [`narrow_exact`], [`narrow_into`].
//!
//! Implementation notes: the definitions are integer operations, so results do not depend on hardware
//! float behaviour. Narrowing runs on the writer side only and stays scalar. It accepts an element only
//! if widening it back gives the same bits, which makes the spec's eligibility rule the code.

use crate::chain::View;
use crate::error::Error;

use super::kernels::Isa;

/// Bytes of one narrow element.
const NARROW_BYTES: usize = 2;
/// Bytes of one wide element.
const WIDE_BYTES: usize = 4;

/// Writes the view of `src` to `dst`. `src` holds whole narrow elements of the blob's canonical bytes.
///
/// # Errors
/// Returns [`Error::OutOfBounds`] if a widening view is given an odd `src`, whose last element is
/// incomplete. Returns [`Error::DestinationSize`] unless `dst` has `src.len()` times the view's length
/// factor bytes.
pub fn widen_into(view: View, src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    widen_with(Isa::detect(), view, src, dst)
}

/// [`widen_into`] with chosen kernels.
pub(super) fn widen_with(isa: Isa, view: View, src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    let factor = match view {
        View::Identity => 1,
        View::F16ToF32 | View::Bf16ToF32 => {
            if !src.len().is_multiple_of(NARROW_BYTES) {
                return Err(Error::OutOfBounds {
                    offset: src.len() as u64 - 1,
                    len: NARROW_BYTES as u64,
                    file_len: src.len() as u64,
                });
            }
            view.length_factor()
        }
    };
    // `usize` to `u64` is lossless, and a slice has at most `isize::MAX` bytes, so this cannot wrap.
    let expected = src.len() as u64 * factor;
    if dst.len() as u64 != expected {
        return Err(Error::DestinationSize { expected, actual: dst.len() as u64 });
    }
    match view {
        View::Identity => dst.copy_from_slice(src),
        View::F16ToF32 => {
            let done = match isa {
                Isa::Scalar => 0,
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2(proof) => proof.widen_f16(src, dst),
            };
            widen_elements(src, dst, done, widen_f16);
        }
        View::Bf16ToF32 => {
            let done = match isa {
                Isa::Scalar => 0,
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2(proof) => proof.widen_bf16(src, dst),
            };
            widen_elements(src, dst, done, widen_bf16);
        }
    }
    Ok(())
}

/// Widens the elements from index `start` with the scalar definition.
#[inline]
fn widen_elements(src: &[u8], dst: &mut [u8], start: usize, element: impl Fn(u16) -> u32) {
    let (narrow, _) = src.as_chunks::<NARROW_BYTES>();
    let (wide, _) = dst.as_chunks_mut::<WIDE_BYTES>();
    for (out, input) in wide.iter_mut().zip(narrow).skip(start) {
        *out = element(u16::from_le_bytes(*input)).to_le_bytes();
    }
}

/// View 0x0302, bfloat16 to binary32.
fn widen_bf16(x: u16) -> u32 {
    u32::from(x) << 16
}

/// View 0x0301, binary16 to binary32: the table of spec section 7.
fn widen_f16(x: u16) -> u32 {
    let x = u32::from(x);
    let sign = (x >> 15) << 31;
    let exponent = (x >> 10) & 31;
    let fraction = x & 1023;
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let top = fraction.ilog2();
            sign | ((top + 103) << 23) | ((fraction - (1 << top)) << (23 - top))
        }
        31 => sign | (255 << 23) | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    }
}

/// The only `u16` that bfloat16 widening could map to `wide`, if any.
fn narrow_bf16(wide: u32) -> Option<u16> {
    let narrow = u16::try_from(wide >> 16).ok()?;
    (widen_bf16(narrow) == wide).then_some(narrow)
}

/// The only `u16` that binary16 widening could map to `wide`, if any. Inverse of [`widen_f16`].
fn narrow_f16(wide: u32) -> Option<u16> {
    let sign = (wide >> 31) << 15;
    let exponent = (wide >> 23) & 255;
    let fraction = wide & 0x007F_FFFF;
    let magnitude = match exponent {
        0 => 0,
        // Spec case `e = 0`, `f != 0`: `exponent = p + 103` for the top bit `p` of `f`.
        103..=112 => {
            let top = exponent - 103;
            (1 << top) | (fraction >> (23 - top))
        }
        113..=142 => ((exponent - 112) << 10) | (fraction >> 13),
        255 => (31 << 10) | (fraction >> 13),
        _ => return None,
    };
    let narrow = u16::try_from(sign | magnitude).ok()?;
    // Dropped fraction bits, and every exponent mapped loosely above, fail this comparison.
    (widen_f16(narrow) == wide).then_some(narrow)
}

/// Returns how many leading bytes of `wide` the view reproduces exactly, a multiple of the element
/// size. The whole slice is eligible for the view if and only if this equals `wide.len()`, per the
/// rule of spec section 7; a smaller value is the offset of the first element that blocks it.
///
/// Returns `None` if `wide` is not a whole number of elements of the view, so a range misaligned for
/// the view has no answer. For [`View::Identity`] every byte is accepted.
#[must_use]
pub fn narrow_exact(view: View, wide: &[u8]) -> Option<usize> {
    match view {
        View::Identity => Some(wide.len()),
        View::F16ToF32 => accepted_bytes(wide, narrow_f16),
        View::Bf16ToF32 => accepted_bytes(wide, narrow_bf16),
    }
}

/// Counts the leading bytes of whole elements that `narrow` accepts.
fn accepted_bytes(wide: &[u8], narrow: fn(u32) -> Option<u16>) -> Option<usize> {
    let (elements, rest) = wide.as_chunks::<WIDE_BYTES>();
    if !rest.is_empty() {
        return None;
    }
    let exact = elements.iter().take_while(|e| narrow(u32::from_le_bytes(**e)).is_some()).count();
    Some(exact * WIDE_BYTES)
}

/// Writes the narrow form of `wide` to `dst`, stopping at the first element that is not exact. Returns
/// the bytes of `wide` accepted, as [`narrow_exact`] does; the first `accepted / factor` bytes of `dst`
/// are written, and the view applied to them gives back the accepted bytes.
///
/// # Errors
/// Returns [`Error::OutOfBounds`] if `wide` is not a whole number of elements. Returns
/// [`Error::DestinationSize`] unless `dst` has `wide.len()` divided by the view's length factor bytes.
pub fn narrow_into(view: View, wide: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
    let factor = view.length_factor();
    let expected = wide.len() as u64 / factor;
    let check_dst = |dst: &[u8]| {
        if dst.len() as u64 == expected {
            Ok(())
        } else {
            Err(Error::DestinationSize { expected, actual: dst.len() as u64 })
        }
    };
    let narrow: fn(u32) -> Option<u16> = match view {
        View::Identity => {
            check_dst(dst)?;
            dst.copy_from_slice(wide);
            return Ok(wide.len());
        }
        View::F16ToF32 => narrow_f16,
        View::Bf16ToF32 => narrow_bf16,
    };
    let (elements, rest) = wide.as_chunks::<WIDE_BYTES>();
    if !rest.is_empty() {
        return Err(Error::OutOfBounds {
            offset: (wide.len() - rest.len()) as u64,
            len: WIDE_BYTES as u64,
            file_len: wide.len() as u64,
        });
    }
    check_dst(dst)?;
    let (outputs, _) = dst.as_chunks_mut::<NARROW_BYTES>();
    let mut accepted = 0;
    for (output, element) in outputs.iter_mut().zip(elements) {
        let Some(value) = narrow(u32::from_le_bytes(*element)) else { break };
        *output = value.to_le_bytes();
        accepted += WIDE_BYTES;
    }
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        GUARD_BYTES, Rng, assert_guards, available_isas, guarded, guarded_dst,
    };
    use super::*;

    const VIEWS: [View; 2] = [View::F16ToF32, View::Bf16ToF32];

    /// Every `u16` in order, little-endian.
    fn all_inputs() -> Vec<u8> {
        (0..=u16::MAX).flat_map(u16::to_le_bytes).collect()
    }

    /// The table of spec section 7, written out again without sharing code with the kernels.
    fn reference(view: View, x: u16) -> u32 {
        let v = u32::from(x);
        match view {
            View::Bf16ToF32 => v * 0x1_0000,
            View::F16ToF32 => {
                let (sign, exp, frac) = (v >> 15, (v >> 10) & 31, v & 1023);
                if exp == 0 && frac == 0 {
                    sign << 31
                } else if exp == 0 {
                    let top = (0..10).rev().find(|t| frac & (1 << t) != 0).expect("not zero");
                    (sign << 31) | ((top + 103) << 23) | ((frac - (1 << top)) << (23 - top))
                } else if exp == 31 {
                    (sign << 31) | (255 << 23) | (frac << 13)
                } else {
                    (sign << 31) | ((exp + 112) << 23) | (frac << 13)
                }
            }
            View::Identity => unreachable!("identity has no element function"),
        }
    }

    fn as_u32s(bytes: &[u8]) -> Vec<u32> {
        bytes.as_chunks::<4>().0.iter().map(|e| u32::from_le_bytes(*e)).collect()
    }

    fn widened(isa: Isa, view: View, src: &[u8]) -> Vec<u32> {
        let mut dst = vec![0; src.len() * 2];
        widen_with(isa, view, src, &mut dst).expect("lengths match");
        as_u32s(&dst)
    }

    /// Every input through every kernel. Shifting the start moves each element through every vector
    /// lane and, for the last ones, through the scalar tail.
    #[test]
    fn all_65536_inputs_match_the_spec_in_every_lane_and_kernel() {
        let inputs = all_inputs();
        for isa in available_isas() {
            for view in VIEWS {
                for shift in 0..9 {
                    let out = widened(isa, view, &inputs[shift * 2..]);
                    for (index, got) in out.iter().enumerate() {
                        let x = u16::try_from(index + shift).expect("in range");
                        assert_eq!(*got, reference(view, x), "{isa:?} {view:?} {x:#06x}");
                    }
                }
            }
        }
    }

    #[test]
    fn finite_inputs_keep_their_exact_value() {
        for x in 0..=u16::MAX {
            let wide = f32::from_bits(widen_f16(x));
            let (sign, e, f) = (x >> 15, (x >> 10) & 31, f64::from(x & 1023));
            if e != 31 {
                let magnitude = if e == 0 {
                    f * 2f64.powi(-24)
                } else {
                    (1.0 + f / 1024.0) * 2f64.powi(i32::from(e) - 15)
                };
                let value = if sign == 1 { -magnitude } else { magnitude };
                assert_eq!(f64::from(wide), value, "{x:#06x}");
                assert_eq!(wide.is_sign_negative(), sign == 1, "{x:#06x}");
            }
            let b = f32::from_bits(widen_bf16(x));
            assert_eq!(b.to_bits() >> 16, u32::from(x));
        }
    }

    #[test]
    fn known_points_of_binary16() {
        let cases = [
            (0x0000, 0x0000_0000),
            (0x8000, 0x8000_0000),
            (0x3C00, 0x3F80_0000),
            (0xC000, 0xC000_0000),
            (0x0001, 0x3380_0000),
            (0x8001, 0xB380_0000),
            (0x03FF, 0x387F_C000),
            (0x0400, 0x3880_0000),
            (0x7BFF, 0x477F_E000),
            (0x7C00, 0x7F80_0000),
            (0xFC00, 0xFF80_0000),
            (0x7C01, 0x7F80_2000),
            (0x7E00, 0x7FC0_0000),
            (0xFDFF, 0xFFBF_E000),
        ];
        let inputs: Vec<u8> = cases.iter().flat_map(|(x, _)| u16::to_le_bytes(*x)).collect();
        for isa in available_isas() {
            let out = widened(isa, View::F16ToF32, &inputs);
            let expected: Vec<u32> = cases.iter().map(|(_, wide)| *wide).collect();
            assert_eq!(out, expected, "{isa:?}");
        }
    }

    /// Every signalling NaN stays signalling: its payload moves up 13 bits, and bit 22 stays clear.
    /// This is the case F16C gets wrong.
    #[test]
    fn nans_keep_payload_and_signalling_bit_in_every_kernel() {
        let nans: Vec<u16> = (0..=u16::MAX).filter(|x| x & 0x7C00 == 0x7C00 && x & 0x3FF != 0).collect();
        assert_eq!(nans.len(), 2 * 1023);
        let inputs: Vec<u8> = nans.iter().flat_map(|x| x.to_le_bytes()).collect();
        for isa in available_isas() {
            for (x, wide) in nans.iter().zip(widened(isa, View::F16ToF32, &inputs)) {
                assert!(f32::from_bits(wide).is_nan(), "{isa:?} {x:#06x}");
                assert_eq!(wide & 0x007F_FFFF, u32::from(x & 0x3FF) << 13, "{isa:?} {x:#06x}");
                assert_eq!(wide & (1 << 22) != 0, x & 0x200 != 0, "{isa:?} {x:#06x}");
                assert_eq!(wide >> 31, u32::from(x >> 15), "{isa:?} {x:#06x}");
            }
        }
    }

    #[test]
    fn the_scalar_tail_stays_inside_the_guard_band() {
        let source = Rng(6).bytes(2 * 40);
        for isa in available_isas() {
            for view in VIEWS {
                for elements in 0..40 {
                    let (src, len) = (&source[..elements * 2], elements * 4);
                    let misalign = elements % 33;
                    let mut buffer = guarded(len, misalign);
                    let dst = guarded_dst(&mut buffer, len, misalign);
                    widen_with(isa, view, src, dst).expect("lengths match");
                    assert_guards(&buffer, len, misalign, "widen");
                    let start = GUARD_BYTES + misalign;
                    let out = as_u32s(&buffer[start..start + len]);
                    assert_eq!(out, widened(Isa::Scalar, view, src), "{isa:?} {view:?}");
                }
            }
        }
    }

    #[test]
    fn identity_copies() {
        let mut dst = [0; 3];
        widen_into(View::Identity, &[1, 2, 3], &mut dst).expect("lengths match");
        assert_eq!(dst, [1, 2, 3]);
    }

    #[test]
    fn wrong_lengths_are_errors() {
        for view in VIEWS {
            let err = widen_into(view, &[0; 3], &mut [0; 6]).expect_err("odd source");
            assert!(matches!(err, Error::OutOfBounds { offset: 2, len: 2, file_len: 3 }), "{err:?}");
            let err = widen_into(view, &[0; 4], &mut [0; 7]).expect_err("short destination");
            assert!(
                matches!(err, Error::DestinationSize { expected: 8, actual: 7 }),
                "{err:?}"
            );
        }
        let err = widen_into(View::Identity, &[0; 3], &mut [0; 4]).expect_err("long destination");
        assert!(matches!(err, Error::DestinationSize { expected: 3, actual: 4 }), "{err:?}");
    }

    #[test]
    fn narrowing_inverts_widening_on_all_inputs() {
        for x in 0..=u16::MAX {
            assert_eq!(narrow_f16(widen_f16(x)), Some(x), "{x:#06x}");
            assert_eq!(narrow_bf16(widen_bf16(x)), Some(x), "{x:#06x}");
        }
    }

    /// Exactly the image of the view is accepted. Probes the neighbours of every image point, where a
    /// loose inverse would slip.
    #[test]
    fn narrowing_rejects_everything_outside_the_image() {
        let image: std::collections::HashSet<u32> = (0..=u16::MAX).map(widen_f16).collect();
        assert_eq!(image.len(), 65536, "the view is injective");
        for x in 0..=u16::MAX {
            let wide = widen_f16(x);
            for bit in 0..32 {
                let probe = wide ^ (1 << bit);
                assert_eq!(narrow_f16(probe).is_some(), image.contains(&probe), "{probe:#010x}");
            }
        }
        for probe in [0x0000_0001, 0x0080_0000, 0x3300_0000, 0x4780_0000, 0x7F7F_FFFF, 0x3F80_0001] {
            assert_eq!(narrow_f16(probe), None, "{probe:#010x}");
        }
        assert_eq!(narrow_bf16(0x3F80_0001), None);
        assert_eq!(narrow_bf16(0x3F80_0000), Some(0x3F80));
    }

    #[test]
    fn narrow_exact_reports_the_exact_prefix() {
        let mut wide: Vec<u8> = [1.0_f32, -2.5, 0.0, f32::INFINITY]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(narrow_exact(View::F16ToF32, &wide), Some(16));
        wide.splice(8..12, 0.1_f32.to_le_bytes());
        assert_eq!(narrow_exact(View::F16ToF32, &wide), Some(8));
        assert_eq!(narrow_exact(View::F16ToF32, &wide[..15]), None);
        assert_eq!(narrow_exact(View::Bf16ToF32, &[0, 0, 1]), None);
        assert_eq!(narrow_exact(View::Bf16ToF32, &[]), Some(0));
        assert_eq!(narrow_exact(View::Identity, &[1, 2, 3]), Some(3));
    }

    #[test]
    fn narrow_into_round_trips_through_the_view() {
        let mut wide: Vec<u8> = (0..=u16::MAX).flat_map(|x| widen_f16(x).to_le_bytes()).collect();
        let mut narrow = vec![0; wide.len() / 2];
        assert_eq!(narrow_into(View::F16ToF32, &wide, &mut narrow).expect("sizes"), wide.len());
        assert_eq!(narrow, all_inputs());

        wide[401] ^= 0x10;
        let inexact = 0x3F80_2000_u32.to_le_bytes();
        assert_eq!(narrow_into(View::Bf16ToF32, &inexact, &mut narrow[..2]).expect("sizes"), 0);
        let accepted = narrow_into(View::F16ToF32, &wide[..800], &mut narrow[..400]).expect("sizes");
        assert_eq!(accepted, 400);

        let mut same = [0; 2];
        assert_eq!(narrow_into(View::Identity, &[7, 8], &mut same).expect("sizes"), 2);
        assert_eq!(same, [7, 8]);
    }

    #[test]
    fn narrow_into_rejects_wrong_lengths() {
        let err = narrow_into(View::F16ToF32, &[0; 6], &mut [0; 3]).expect_err("partial element");
        assert!(matches!(err, Error::OutOfBounds { offset: 4, len: 4, file_len: 6 }), "{err:?}");
        let err = narrow_into(View::F16ToF32, &[0; 8], &mut [0; 3]).expect_err("short destination");
        assert!(matches!(err, Error::DestinationSize { expected: 4, actual: 3 }), "{err:?}");
        let err = narrow_into(View::Identity, &[0; 2], &mut [0; 3]).expect_err("long destination");
        assert!(matches!(err, Error::DestinationSize { expected: 2, actual: 3 }), "{err:?}");
    }
}
