//! File: codec/delta.rs
//! Purpose: the delta operations of spec section 7a, between a delta chunk's target bytes, its
//! residual and its base bytes.
//!
//! Main responsibilities:
//! - [`residual`] turns target bytes into the residual that the rest of a chain encodes;
//! - [`combine`] turns a decoded residual back into target bytes, in place.
//!
//! Key functions: [`residual`], [`combine`].
//!
//! Implementation notes: the operations are integer definitions on bytes, so they are the same on
//! every host. `delta_xor` is bytewise; its width only fixes the chain block, and is checked here
//! only as the length rule every caller must meet. Plain byte loops: the compiler vectorizes them,
//! and a chunk's XOR costs far less than its zstd stream.

use crate::chain::DeltaOp;
use crate::error::Error;

/// Writes the residual of `target` against `base` into `out`: for `delta_xor`, `target XOR base`.
///
/// All three have the same length, a multiple of the operation's width.
///
/// # Errors
/// Returns [`Error::DestinationSize`] if `base` or `out` differs in length from `target`, or the
/// length is not a multiple of the width. `out` is then unchanged.
pub fn residual(op: DeltaOp, target: &[u8], base: &[u8], out: &mut [u8]) -> Result<(), Error> {
    check_lengths(op, target.len(), base.len())?;
    check_lengths(op, target.len(), out.len())?;
    match op {
        DeltaOp::Xor(_) => {
            for ((out, target), base) in out.iter_mut().zip(target).zip(base) {
                *out = target ^ base;
            }
        }
    }
    Ok(())
}

/// Turns `residual`, decoded, into the target bytes in place, using `base`: for `delta_xor`,
/// `residual XOR base`. The inverse of [`residual`].
///
/// # Errors
/// Returns [`Error::DestinationSize`] if `base` differs in length from `residual`, or the length is
/// not a multiple of the width. `residual` is then unchanged.
pub fn combine(op: DeltaOp, residual: &mut [u8], base: &[u8]) -> Result<(), Error> {
    check_lengths(op, residual.len(), base.len())?;
    match op {
        DeltaOp::Xor(_) => {
            for (byte, base) in residual.iter_mut().zip(base) {
                *byte ^= base;
            }
        }
    }
    Ok(())
}

/// Checks that `other` equals `len` and that `len` is a whole number of elements of `op`.
fn check_lengths(op: DeltaOp, len: usize, other: usize) -> Result<(), Error> {
    // A width is 2, 4 or 8 bytes.
    let width = op.width().bytes() as usize;
    if other != len || !len.is_multiple_of(width) {
        // Lossless: `usize` is at most 64 bits on every supported target.
        let (expected, actual) = ((len / width * width) as u64, other as u64);
        return Err(Error::DestinationSize { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::Rng;
    use super::*;
    use crate::chain::ElementWidth;

    const OPS: [DeltaOp; 3] = [
        DeltaOp::Xor(ElementWidth::Two),
        DeltaOp::Xor(ElementWidth::Four),
        DeltaOp::Xor(ElementWidth::Eight),
    ];

    #[test]
    fn combine_inverts_residual_for_every_width_and_length() {
        let mut rng = Rng(7);
        for op in OPS {
            let width = op.width().bytes() as usize;
            for elements in [0, 1, 2, 3, 31, 1000] {
                let len = elements * width;
                let (target, base) = (rng.bytes(len), rng.bytes(len));
                let mut out = vec![0; len];
                residual(op, &target, &base, &mut out).expect("equal lengths");
                combine(op, &mut out, &base).expect("equal lengths");
                assert_eq!(out, target, "{op:?} {len}");
            }
        }
    }

    #[test]
    fn delta_xor_matches_the_spec_by_hand() {
        // Spec section 7a: every target byte is the residual byte XOR the base byte.
        let target = [0x3F, 0x80, 0x00, 0xFF];
        let base = [0x3F, 0x81, 0xF0, 0x0F];
        let mut out = [0; 4];
        residual(DeltaOp::Xor(ElementWidth::Two), &target, &base, &mut out).expect("valid");
        assert_eq!(out, [0x00, 0x01, 0xF0, 0xF0]);
        combine(DeltaOp::Xor(ElementWidth::Four), &mut out, &base).expect("valid");
        assert_eq!(out, target);
    }

    #[test]
    fn equal_bytes_give_a_zero_residual() {
        let data = Rng(3).bytes(64);
        let mut out = vec![0xAA; 64];
        residual(DeltaOp::Xor(ElementWidth::Eight), &data, &data, &mut out).expect("valid");
        assert!(out.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn mismatched_lengths_are_errors_and_change_nothing() {
        let op = DeltaOp::Xor(ElementWidth::Four);
        let mut out = [7; 8];
        let error = residual(op, &[1; 8], &[1; 4], &mut out).expect_err("short base");
        assert!(matches!(error, Error::DestinationSize { expected: 8, actual: 4 }), "{error:?}");
        let error = residual(op, &[1; 8], &[1; 8], &mut [0; 12]).expect_err("long out");
        assert!(matches!(error, Error::DestinationSize { expected: 8, actual: 12 }), "{error:?}");
        assert_eq!(out, [7; 8]);
        let error = combine(op, &mut out, &[1; 12]).expect_err("long base");
        assert!(matches!(error, Error::DestinationSize { .. }), "{error:?}");
        let error = combine(op, &mut out[..6], &[1; 6]).expect_err("not whole elements");
        assert!(matches!(error, Error::DestinationSize { expected: 4, actual: 6 }), "{error:?}");
        assert_eq!(out, [7; 8]);
    }
}
