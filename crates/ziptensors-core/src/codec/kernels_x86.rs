//! File: `codec/kernels_x86.rs`
//! Purpose: runtime-detected AVX2 kernels, the only `unsafe` code in `ziptensors-core`.
//!
//! Main responsibilities: plane interleave for widths 2, 4 and 8, the same interleave inside one
//! buffer, element rotation, and both widening views.
//!
//! Key structures: [`Avx2`], a zero-size proof that the CPU has AVX2; only [`Avx2::detect`] makes one.
//!
//! Implementation notes: every method does the whole lanes and returns how much it did; the scalar
//! code in `kernels.rs` finishes the tail, so this file has no tail logic and no case of its own. All
//! methods read and write little-endian lanes, which is the only byte order on `x86_64`.
//!
//! The in-place interleave is risk R1. With `n` elements of width `w`, the compressed plane sits at
//! `tail = (w - 1) * n`. Before step `i` the write head is `w * i` and the read head is `tail + i`,
//! so `read - write = (w - 1) * (n - i)`, which is not negative for `i <= n`: the write head cannot
//! pass the read head, and they meet only at the end. A step loads all its bytes before its first
//! store, and each step only runs when `i + 32 <= n`, so its load stays inside `dst`. A load therefore
//! never sees a byte written by an earlier step, right up to the last one. [`check_heads`] asserts
//! this on every step of a debug build.

#![allow(
    unsafe_code,
    reason = "AVX2 intrinsics need raw pointers; this module is the one exception to the crate rule"
)]

use std::arch::is_x86_feature_detected;
use std::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_loadu_si128, _mm256_add_epi32, _mm256_and_si256,
    _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_castps_si256, _mm256_cmpeq_epi32,
    _mm256_cmpgt_epi32, _mm256_cvtepi32_ps, _mm256_cvtepu16_epi32, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_sll_epi16, _mm256_sll_epi32, _mm256_sll_epi64, _mm256_slli_epi32, _mm256_srl_epi16,
    _mm256_srl_epi32, _mm256_srl_epi64, _mm256_storeu_si256, _mm256_sub_epi32,
    _mm256_unpackhi_epi8, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpacklo_epi8,
    _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
};
use std::ptr;

use crate::chain::ElementWidth;

use super::kernels::Rotation;

/// Bytes in one vector, which is also the elements per step of an interleave.
const VECTOR: usize = 32;
/// Most planes an interleave takes. Unused slots of a source array hold null.
const MAX_PLANES: usize = 8;
/// Elements of `u16` per widening step.
const WIDEN_STEP: usize = 8;

/// Proof that the running CPU has AVX2. It makes the safe methods below sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Avx2 {
    _proof: (),
}

#[allow(clippy::unused_self, reason = "`self` is the runtime proof of AVX2, its only content")]
impl Avx2 {
    /// Returns the proof if the CPU has AVX2.
    pub(super) fn detect() -> Option<Self> {
        is_x86_feature_detected!("avx2").then_some(Self { _proof: () })
    }

    /// Interleaves whole steps of `planes` into `dst`, element `i` taking byte `k` from `planes[k][i]`.
    /// Returns the elements done, a multiple of 32.
    ///
    /// # Panics
    /// Unless there are 2, 4 or 8 planes of equal length `n` and `dst` has `n * planes` bytes. The
    /// callers derive all of these from one length, so file contents cannot reach this.
    pub(super) fn merge_planes(self, planes: &[&[u8]], dst: &mut [u8]) -> usize {
        let width = planes.len();
        let n = planes.first().map_or(0, |plane| plane.len());
        assert!(matches!(width, 2 | 4 | 8), "plane count {width}");
        assert!(planes.iter().all(|plane| plane.len() == n), "unequal planes");
        assert_eq!(n.checked_mul(width), Some(dst.len()), "destination length");
        let mut sources = [ptr::null::<u8>(); MAX_PLANES];
        for (source, plane) in sources.iter_mut().zip(planes) {
            *source = plane.as_ptr();
        }
        // SAFETY: the asserts give `width` sources of `n` readable bytes and `dst` of `n * width`
        // writable bytes, and `&mut dst` excludes overlap with `planes`.
        unsafe { self.weave(sources, width, dst.as_mut_ptr(), n) }
    }

    /// Interleaves whole steps into `dst` when plane `compressed` already sits in the last `n` bytes
    /// of `dst`, and `stored` holds the other planes in order. Returns the elements done.
    ///
    /// Bytes `0..(width - 1) * n` of `dst` are overwritten; the tail stays readable until consumed.
    ///
    /// # Panics
    /// Unless `stored.len() + 1` is 2, 4 or 8, `compressed` is below it, and `dst` holds that many
    /// planes of the length of each stored plane. As for [`Avx2::merge_planes`], file contents cannot
    /// reach this.
    pub(super) fn merge_in_place(
        self,
        compressed: usize,
        stored: &[&[u8]],
        dst: &mut [u8],
    ) -> usize {
        let width = stored.len() + 1;
        let n = dst.len() / width;
        assert!(matches!(width, 2 | 4 | 8), "plane count {width}");
        assert!(compressed < width, "compressed plane {compressed}");
        assert_eq!(dst.len() % width, 0, "destination length");
        assert!(stored.iter().all(|plane| plane.len() == n), "unequal planes");
        let out = dst.as_mut_ptr();
        // All accesses to `dst` go through `out`, so reads and writes of the overlap share a pointer.
        let tail = out.wrapping_add(dst.len() - n).cast_const();
        let mut sources = [ptr::null::<u8>(); MAX_PLANES];
        let mut planes = stored.iter();
        for (k, source) in sources.iter_mut().enumerate().take(width) {
            *source = if k == compressed {
                tail
            } else {
                planes.next().map_or(ptr::null(), |plane| plane.as_ptr())
            };
        }
        // SAFETY: the asserts give `width` sources of `n` readable bytes and `n * width` writable
        // bytes at `out`. The tail source overlaps the output, which `weave` allows when a source is
        // the last `n` bytes of the output: the header of this file proves the order of accesses.
        unsafe { self.weave(sources, width, out, n) }
    }

    /// Dispatches on the plane count.
    ///
    /// # Safety
    /// `sources[..width]` are readable for `n` bytes, `out` is writable for `n * width` bytes, `width`
    /// is 2, 4 or 8. A source may overlap the output only if it is the last `n` bytes of it.
    unsafe fn weave(
        self,
        sources: [*const u8; MAX_PLANES],
        width: usize,
        out: *mut u8,
        n: usize,
    ) -> usize {
        let [p0, p1, p2, p3, ..] = sources;
        match width {
            // SAFETY: `self` proves AVX2; the rest is this function's contract.
            2 => unsafe { weave2([p0, p1], out, n) },
            // SAFETY: as above.
            4 => unsafe { weave4([p0, p1, p2, p3], out, n) },
            // SAFETY: as above; `width` is 8 by this function's contract.
            _ => unsafe { weave8(sources, out, n) },
        }
    }

    /// Rotates whole vectors of `data` by one bit, as `width` byte little-endian elements. Returns
    /// the bytes done, a multiple of 32.
    pub(super) fn rotate(self, rotation: Rotation, width: ElementWidth, data: &mut [u8]) -> usize {
        let bits = 8 * i32::from(width.param());
        let left = match rotation {
            Rotation::Left => 1,
            Rotation::Right => bits - 1,
        };
        let (start, len) = (data.as_mut_ptr(), data.len());
        match width {
            // SAFETY: `self` proves AVX2, and `start` is writable and readable for `len` bytes.
            ElementWidth::Two => unsafe { rotate_lanes::<2>(start, len, left, bits - left) },
            // SAFETY: as above.
            ElementWidth::Four => unsafe { rotate_lanes::<4>(start, len, left, bits - left) },
            // SAFETY: as above.
            ElementWidth::Eight => unsafe { rotate_lanes::<8>(start, len, left, bits - left) },
        }
    }

    /// Widens whole steps of `u16` bfloat16 elements to `u32` binary32. Returns the elements done, a
    /// multiple of 8.
    ///
    /// # Panics
    /// Unless `dst` is exactly twice `src`, which is even. `widen_with` checks this first.
    pub(super) fn widen_bf16(self, src: &[u8], dst: &mut [u8]) -> usize {
        assert_widen_lengths(src, dst);
        // SAFETY: `self` proves AVX2; the lengths are asserted above and the slices are disjoint.
        unsafe { widen_bf16_lanes(src.as_ptr(), dst.as_mut_ptr(), src.len() / 2) }
    }

    /// Widens whole steps of `u16` binary16 elements to `u32` binary32. Returns the elements done, a
    /// multiple of 8.
    ///
    /// # Panics
    /// Unless `dst` is exactly twice `src`, which is even. `widen_with` checks this first.
    pub(super) fn widen_f16(self, src: &[u8], dst: &mut [u8]) -> usize {
        assert_widen_lengths(src, dst);
        // SAFETY: `self` proves AVX2; the lengths are asserted above and the slices are disjoint.
        unsafe { widen_f16_lanes(src.as_ptr(), dst.as_mut_ptr(), src.len() / 2) }
    }
}

/// Asserts the shared precondition of the widening methods.
fn assert_widen_lengths(src: &[u8], dst: &[u8]) {
    assert_eq!(src.len() % 2, 0, "source length");
    assert_eq!(src.len().checked_mul(2), Some(dst.len()), "destination length");
}

/// Asserts, in debug builds, that no read head of step `i` lies in a part of the output that has
/// been written: the R1 invariant from the file header. `sources` may lie outside the output.
#[inline]
fn check_heads(sources: &[*const u8], out: *const u8, width: usize, n: usize, i: usize) {
    if cfg!(debug_assertions) {
        let start = out.addr();
        let end = start + width * n;
        let write_head = start + width * i;
        for source in sources {
            let read_head = source.addr() + i;
            let outside = read_head < start || read_head >= end;
            debug_assert!(
                outside || read_head >= write_head,
                "read head {read_head:#x} is behind write head {write_head:#x}, i = {i}, n = {n}"
            );
        }
    }
}

/// Loads 32 bytes.
///
/// # Safety
/// `from` is readable for 32 bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load(from: *const u8) -> __m256i {
    // SAFETY: readable by contract; `loadu` has no alignment requirement.
    unsafe { _mm256_loadu_si256(from.cast()) }
}

/// Stores 32 bytes.
///
/// # Safety
/// `to` is writable for 32 bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store(to: *mut u8, value: __m256i) {
    // SAFETY: writable by contract; `storeu` has no alignment requirement.
    unsafe { _mm256_storeu_si256(to.cast(), value) }
}

/// Interleaves 2 planes, 32 elements per step. Returns the elements done.
///
/// # Safety
/// As [`Avx2::weave`], with the CPU having AVX2.
#[target_feature(enable = "avx2")]
unsafe fn weave2(sources: [*const u8; 2], out: *mut u8, n: usize) -> usize {
    let mut i = 0;
    while i + VECTOR <= n {
        check_heads(&sources, out, 2, n, i);
        // SAFETY: `i + 32 <= n`, so each source has 32 readable bytes at `i`.
        let (p0, p1) = unsafe { (load(sources[0].add(i)), load(sources[1].add(i))) };
        // Bytes pair up inside each 128-bit lane, so whole lanes then go back in element order.
        let low = _mm256_unpacklo_epi8(p0, p1);
        let high = _mm256_unpackhi_epi8(p0, p1);
        let first = _mm256_permute2x128_si256::<0x20>(low, high);
        let second = _mm256_permute2x128_si256::<0x31>(low, high);
        // SAFETY: `i + 32 <= n`, so the output has 64 writable bytes at `2 * i`.
        unsafe {
            store(out.add(2 * i), first);
            store(out.add(2 * i + VECTOR), second);
        }
        i += VECTOR;
    }
    i
}

/// Interleaves 4 planes, 32 elements per step. Returns the elements done.
///
/// # Safety
/// As [`Avx2::weave`], with the CPU having AVX2.
#[target_feature(enable = "avx2")]
unsafe fn weave4(sources: [*const u8; 4], out: *mut u8, n: usize) -> usize {
    let mut i = 0;
    while i + VECTOR <= n {
        check_heads(&sources, out, 4, n, i);
        // SAFETY: `i + 32 <= n`, so each source has 32 readable bytes at `i`.
        let (p0, p1, p2, p3) = unsafe {
            (
                load(sources[0].add(i)),
                load(sources[1].add(i)),
                load(sources[2].add(i)),
                load(sources[3].add(i)),
            )
        };
        let low01 = _mm256_unpacklo_epi8(p0, p1);
        let high01 = _mm256_unpackhi_epi8(p0, p1);
        let low23 = _mm256_unpacklo_epi8(p2, p3);
        let high23 = _mm256_unpackhi_epi8(p2, p3);
        // Each vector holds 4 elements per lane: 0-3 and 16-19, 4-7 and 20-23, and so on.
        let v0 = _mm256_unpacklo_epi16(low01, low23);
        let v1 = _mm256_unpackhi_epi16(low01, low23);
        let v2 = _mm256_unpacklo_epi16(high01, high23);
        let v3 = _mm256_unpackhi_epi16(high01, high23);
        let o0 = _mm256_permute2x128_si256::<0x20>(v0, v1);
        let o1 = _mm256_permute2x128_si256::<0x20>(v2, v3);
        let o2 = _mm256_permute2x128_si256::<0x31>(v0, v1);
        let o3 = _mm256_permute2x128_si256::<0x31>(v2, v3);
        // SAFETY: `i + 32 <= n`, so the output has 128 writable bytes at `4 * i`.
        unsafe {
            let step = out.add(4 * i);
            store(step, o0);
            store(step.add(VECTOR), o1);
            store(step.add(2 * VECTOR), o2);
            store(step.add(3 * VECTOR), o3);
        }
        i += VECTOR;
    }
    i
}

/// Interleaves 8 planes, 32 elements per step. Returns the elements done.
///
/// # Safety
/// As [`Avx2::weave`], with the CPU having AVX2.
#[target_feature(enable = "avx2")]
unsafe fn weave8(sources: [*const u8; MAX_PLANES], out: *mut u8, n: usize) -> usize {
    let mut i = 0;
    while i + VECTOR <= n {
        check_heads(&sources, out, MAX_PLANES, n, i);
        // SAFETY: `i + 32 <= n`, so each source has 32 readable bytes at `i`.
        let (p0, p1, p2, p3, p4, p5, p6, p7) = unsafe {
            (
                load(sources[0].add(i)),
                load(sources[1].add(i)),
                load(sources[2].add(i)),
                load(sources[3].add(i)),
                load(sources[4].add(i)),
                load(sources[5].add(i)),
                load(sources[6].add(i)),
                load(sources[7].add(i)),
            )
        };
        let (low01, high01) = (_mm256_unpacklo_epi8(p0, p1), _mm256_unpackhi_epi8(p0, p1));
        let (low23, high23) = (_mm256_unpacklo_epi8(p2, p3), _mm256_unpackhi_epi8(p2, p3));
        let (low45, high45) = (_mm256_unpacklo_epi8(p4, p5), _mm256_unpackhi_epi8(p4, p5));
        let (low67, high67) = (_mm256_unpacklo_epi8(p6, p7), _mm256_unpackhi_epi8(p6, p7));
        // 4 byte groups of elements 0-3, 4-7, 8-11, 12-15 in the first lane, plus 16 in the second.
        let lower = [
            _mm256_unpacklo_epi16(low01, low23),
            _mm256_unpackhi_epi16(low01, low23),
            _mm256_unpacklo_epi16(high01, high23),
            _mm256_unpackhi_epi16(high01, high23),
        ];
        let upper = [
            _mm256_unpacklo_epi16(low45, low67),
            _mm256_unpackhi_epi16(low45, low67),
            _mm256_unpacklo_epi16(high45, high67),
            _mm256_unpackhi_epi16(high45, high67),
        ];
        for (k, (&bottom, &top)) in lower.iter().zip(&upper).enumerate() {
            // Two whole elements per lane: 4 elements in `pair_low` and `pair_high` together.
            let pair_low = _mm256_unpacklo_epi32(bottom, top);
            let pair_high = _mm256_unpackhi_epi32(bottom, top);
            let first = _mm256_permute2x128_si256::<0x20>(pair_low, pair_high);
            let second = _mm256_permute2x128_si256::<0x31>(pair_low, pair_high);
            // SAFETY: `i + 32 <= n`, so the output has 256 writable bytes at `8 * i`, and `k < 4`.
            // Stores of this step start only here, after every load of the step above.
            unsafe {
                store(out.add(8 * i + k * VECTOR), first);
                store(out.add(8 * i + (k + 4) * VECTOR), second);
            }
        }
        i += VECTOR;
    }
    i
}

/// Rotates `WIDTH` byte elements left by `left` bits, in whole vectors. `right` is the element bits
/// minus `left`. A rotation right by 1 is a rotation left by bits minus 1. Returns the bytes done.
///
/// # Safety
/// `data` is readable and writable for `len` bytes, and the CPU has AVX2.
#[target_feature(enable = "avx2")]
unsafe fn rotate_lanes<const WIDTH: usize>(
    data: *mut u8,
    len: usize,
    left: i32,
    right: i32,
) -> usize {
    let (up, down): (__m128i, __m128i) = (_mm_cvtsi32_si128(left), _mm_cvtsi32_si128(right));
    let mut i = 0;
    while i + VECTOR <= len {
        // SAFETY: `i + 32 <= len`, so 32 bytes at `i` are in range.
        let x = unsafe { load(data.add(i)) };
        let y = match WIDTH {
            2 => _mm256_or_si256(_mm256_sll_epi16(x, up), _mm256_srl_epi16(x, down)),
            4 => _mm256_or_si256(_mm256_sll_epi32(x, up), _mm256_srl_epi32(x, down)),
            _ => _mm256_or_si256(_mm256_sll_epi64(x, up), _mm256_srl_epi64(x, down)),
        };
        // SAFETY: as for the load above.
        unsafe { store(data.add(i), y) };
        i += VECTOR;
    }
    i
}

/// Loads 8 `u16` elements as 8 zero-extended `u32` lanes.
///
/// # Safety
/// `from` is readable for 16 bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_u16_lanes(from: *const u8) -> __m256i {
    // SAFETY: readable by contract; `loadu` has no alignment requirement.
    _mm256_cvtepu16_epi32(unsafe { _mm_loadu_si128(from.cast()) })
}

/// Widens bfloat16 elements: `x << 16`. Returns the elements done.
///
/// # Safety
/// `src` is readable for `n * 2` bytes, `dst` is writable for `n * 4` bytes, they do not overlap, and
/// the CPU has AVX2.
#[target_feature(enable = "avx2")]
unsafe fn widen_bf16_lanes(src: *const u8, dst: *mut u8, n: usize) -> usize {
    let mut i = 0;
    while i + WIDEN_STEP <= n {
        // SAFETY: `i + 8 <= n`, so 16 bytes at `2 * i` are readable.
        let x = unsafe { load_u16_lanes(src.add(2 * i)) };
        // SAFETY: `i + 8 <= n`, so 32 bytes at `4 * i` are writable.
        unsafe { store(dst.add(4 * i), _mm256_slli_epi32::<16>(x)) };
        i += WIDEN_STEP;
    }
    i
}

/// Widens binary16 elements by spec section 7. Returns the elements done.
///
/// All four cases of the spec are computed and blended, on integers. F16C is deliberately not used:
/// it quiets signalling NaNs, which the spec's definition preserves.
///
/// # Safety
/// `src` is readable for `n * 2` bytes, `dst` is writable for `n * 4` bytes, they do not overlap, and
/// the CPU has AVX2.
#[target_feature(enable = "avx2")]
unsafe fn widen_f16_lanes(src: *const u8, dst: *mut u8, n: usize) -> usize {
    /// Exponent rebias from binary16 to binary32, in place: `(127 - 15) << 23`.
    const REBIAS: i32 = 112 << 23;
    /// `int_to_float` gives exponent `p + 127` for a value with top bit `p`; the spec wants `p + 103`.
    const SUBNORMAL_REBIAS: i32 = 24 << 23;
    let sign_bit = _mm256_set1_epi32(0x8000);
    let magnitude_bits = _mm256_set1_epi32(0x7FFF);
    let last_finite = _mm256_set1_epi32(0x7BFF);
    let first_normal = _mm256_set1_epi32(0x0400);
    let rebias = _mm256_set1_epi32(REBIAS);
    let subnormal_rebias = _mm256_set1_epi32(SUBNORMAL_REBIAS);
    let mut i = 0;
    while i + WIDEN_STEP <= n {
        // SAFETY: `i + 8 <= n`, so 16 bytes at `2 * i` are readable.
        let x = unsafe { load_u16_lanes(src.add(2 * i)) };
        let sign = _mm256_slli_epi32::<16>(_mm256_and_si256(x, sign_bit));
        let magnitude = _mm256_and_si256(x, magnitude_bits);

        // `1 <= e <= 30`: exponent and fraction move up together, then the bias changes.
        let normal = _mm256_add_epi32(_mm256_slli_epi32::<13>(magnitude), rebias);
        // `e = 31`: a second rebias takes the exponent from 143 to 255.
        let special = _mm256_add_epi32(normal, rebias);
        let is_special = _mm256_cmpgt_epi32(magnitude, last_finite);
        let wide = _mm256_blendv_epi8(normal, special, is_special);

        // `e = 0`: the fraction is an integer below 2^10. Converting it is exact, so neither the
        // rounding mode nor flush-to-zero matters, and it yields the top bit `p` and the shifted
        // fraction of the spec's formula. Zero has no top bit and stays zero.
        let converted = _mm256_castps_si256(_mm256_cvtepi32_ps(magnitude));
        let subnormal = _mm256_sub_epi32(converted, subnormal_rebias);
        let is_zero = _mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256());
        let subnormal = _mm256_andnot_si256(is_zero, subnormal);
        let is_small = _mm256_cmpgt_epi32(first_normal, magnitude);
        let wide = _mm256_blendv_epi8(wide, subnormal, is_small);

        // SAFETY: `i + 8 <= n`, so 32 bytes at `4 * i` are writable.
        unsafe { store(dst.add(4 * i), _mm256_or_si256(wide, sign)) };
        i += WIDEN_STEP;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::check_heads;

    /// Runs the head check for an in-place merge of `n` elements with the compressed plane at `tail`.
    fn walk(width: usize, n: usize, tail: usize) {
        let buffer = vec![0_u8; width * n];
        let out = buffer.as_ptr();
        for i in (0..n).step_by(32) {
            check_heads(&[out.wrapping_add(tail)], out, width, n, i);
        }
    }

    #[test]
    fn the_real_tail_keeps_the_read_head_ahead_for_every_width() {
        for width in [2, 4, 8] {
            for n in [1, 32, 33, 64, 2047] {
                walk(width, n, (width - 1) * n);
            }
        }
    }

    /// Proves the assert can fire: one byte before the real tail, the heads cross on the last step.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "is behind write head")]
    fn a_tail_placed_too_early_trips_the_assert() {
        walk(2, 96, 96 - 33);
    }

    #[test]
    fn a_source_outside_the_output_is_not_checked() {
        let (output, source) = (vec![0_u8; 128], vec![0_u8; 64]);
        check_heads(&[source.as_ptr()], output.as_ptr(), 2, 64, 32);
    }
}
