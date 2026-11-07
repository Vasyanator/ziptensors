//! File: codec/kernels.rs
//! Purpose: the byte-level transforms of spec section 6.3, and the choice of scalar or AVX2 kernels.
//!
//! Main responsibilities: gather one stream out of a chunk, merge streams back (from separate
//! buffers, or inside the destination), and rotate elements by one bit.
//!
//! Key structures: [`Isa`], [`Rotation`].
//!
//! Key functions: [`gather_field`], [`merge_planes`], [`merge_in_place`], [`merge_fields`],
//! [`rotate`].
//!
//! Implementation notes: the scalar code is complete and defines the bytes. The AVX2 kernels in
//! `kernels_x86.rs` only take whole vectors off the front and these functions finish the tail. Field
//! merge and stream gathering stay scalar: fields have no fixed stride a vector kernel could use, and
//! gathering runs next to compression, which dominates its cost. Length preconditions below are met by
//! construction in `chunk.rs`, where all lengths derive from one validated raw length.

use crate::chain::{ElementWidth, FieldTable};

#[cfg(target_arch = "x86_64")]
use super::kernels_x86::Avx2;

/// The kernels a workspace runs. All of them produce identical bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Isa {
    /// Portable code.
    Scalar,
    /// AVX2 for whole vectors, scalar for the tail.
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2),
}

impl Isa {
    /// Returns the fastest kernels this CPU can run.
    pub(super) fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if let Some(proof) = Avx2::detect() {
            return Self::Avx2(proof);
        }
        Self::Scalar
    }
}

/// Direction of a one bit element rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Rotation {
    /// Toward the high bits: encoding.
    Left,
    /// Toward the low bits: decoding.
    Right,
}

/// Writes bytes `offset..offset + len` of every `block` byte block of `src` to `dst`, in order.
/// `dst` holds `len` bytes per complete block of `src`. `offset + len` must not exceed `block`.
pub(super) fn gather_field(src: &[u8], block: usize, offset: usize, len: usize, dst: &mut [u8]) {
    if len == 1 {
        for (byte, source) in dst.iter_mut().zip(src.chunks_exact(block)) {
            *byte = source[offset];
        }
    } else {
        for (field, source) in dst.chunks_exact_mut(len).zip(src.chunks_exact(block)) {
            field.copy_from_slice(&source[offset..offset + len]);
        }
    }
}

/// Merges byte planes into `dst`: byte `k` of element `i` is byte `i` of `planes[k]`.
///
/// `planes` are 2, 4 or 8 slices of equal length `n` and `dst` has exactly `n * planes.len()` bytes.
pub(super) fn merge_planes(isa: Isa, planes: &[&[u8]], dst: &mut [u8]) {
    let done = match isa {
        Isa::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2(proof) => proof.merge_planes(planes, dst),
    };
    merge_planes_scalar(planes, dst, done);
}

/// Scalar merge of the elements from `start`. Widths 2 and 4 are the bulk of real data, and a fixed
/// width lets the compiler remove the inner loop.
fn merge_planes_scalar(planes: &[&[u8]], dst: &mut [u8], start: usize) {
    /// Merges `W` planes.
    fn fixed<const W: usize>(planes: [&[u8]; W], dst: &mut [u8], start: usize) {
        let (elements, _) = dst.as_chunks_mut::<W>();
        for (i, element) in elements.iter_mut().enumerate().skip(start) {
            for (byte, plane) in element.iter_mut().zip(planes) {
                *byte = plane[i];
            }
        }
    }
    match *planes {
        [a, b] => fixed([a, b], dst, start),
        [a, b, c, e] => fixed([a, b, c, e], dst, start),
        _ => {
            for (i, element) in dst.chunks_exact_mut(planes.len()).enumerate().skip(start) {
                for (byte, plane) in element.iter_mut().zip(planes) {
                    *byte = plane[i];
                }
            }
        }
    }
}

/// Merges planes inside `dst`, whose last `n` bytes already hold plane `compressed`. `stored` holds
/// the other planes in order, `n` bytes each, and `dst` has `n * (stored.len() + 1)` bytes.
///
/// Forward order is what makes this work: see the proof in the header of `kernels_x86.rs`. Element `i`
/// reads its tail byte before it writes, and every byte it writes is below the next tail byte.
pub(super) fn merge_in_place(isa: Isa, compressed: usize, stored: &[&[u8]], dst: &mut [u8]) {
    let done = match isa {
        Isa::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2(proof) => proof.merge_in_place(compressed, stored, dst),
    };
    let width = stored.len() + 1;
    let n = dst.len() / width;
    let tail = dst.len() - n;
    for i in done..n {
        debug_assert!(width * i <= tail + i, "write head passed read head at element {i}");
        let from_tail = dst[tail + i];
        let (below, above) = dst[width * i..width * (i + 1)].split_at_mut(compressed);
        let (own, above) = above.split_at_mut(1);
        own[0] = from_tail;
        let (lower, upper) = stored.split_at(compressed);
        for (byte, plane) in below.iter_mut().chain(above).zip(lower.iter().chain(upper)) {
            *byte = plane[i];
        }
    }
}

/// Merges field streams into `dst`, the reverse of [`gather_field`] over the fields of `table`.
/// `streams` has one slice per field, of `m * field.len` bytes for `m` blocks of `dst`.
pub(super) fn merge_fields(table: FieldTable, streams: &[&[u8]], dst: &mut [u8]) {
    // `u32` to `usize` is lossless on every supported target, and table values are small.
    let block = table.block_bytes() as usize;
    for (field, stream) in table.fields().iter().zip(streams) {
        let (offset, len) = (field.offset as usize, field.len as usize);
        if len == 1 {
            for (target, byte) in dst.chunks_exact_mut(block).zip(*stream) {
                target[offset] = *byte;
            }
        } else {
            for (target, bytes) in dst.chunks_exact_mut(block).zip(stream.chunks_exact(len)) {
                target[offset..offset + len].copy_from_slice(bytes);
            }
        }
    }
}

/// Rotates every `width` byte little-endian element of `data` by one bit. Bytes past the last whole
/// element are left alone; `chunk.rs` guarantees there are none.
pub(super) fn rotate(isa: Isa, rotation: Rotation, width: ElementWidth, data: &mut [u8]) {
    let done = match isa {
        Isa::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2(proof) => proof.rotate(rotation, width, data),
    };
    let rest = &mut data[done..];
    match (width, rotation) {
        (ElementWidth::Two, Rotation::Left) => {
            rotate_elements(rest, |e| u16::from_le_bytes(e).rotate_left(1).to_le_bytes());
        }
        (ElementWidth::Two, Rotation::Right) => {
            rotate_elements(rest, |e| u16::from_le_bytes(e).rotate_right(1).to_le_bytes());
        }
        (ElementWidth::Four, Rotation::Left) => {
            rotate_elements(rest, |e| u32::from_le_bytes(e).rotate_left(1).to_le_bytes());
        }
        (ElementWidth::Four, Rotation::Right) => {
            rotate_elements(rest, |e| u32::from_le_bytes(e).rotate_right(1).to_le_bytes());
        }
        (ElementWidth::Eight, Rotation::Left) => {
            rotate_elements(rest, |e| u64::from_le_bytes(e).rotate_left(1).to_le_bytes());
        }
        (ElementWidth::Eight, Rotation::Right) => {
            rotate_elements(rest, |e| u64::from_le_bytes(e).rotate_right(1).to_le_bytes());
        }
    }
}

/// Applies `rotate` to every whole `W` byte element.
#[inline]
fn rotate_elements<const W: usize>(data: &mut [u8], rotate: impl Fn([u8; W]) -> [u8; W]) {
    let (elements, _) = data.as_chunks_mut::<W>();
    for element in elements {
        *element = rotate(*element);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        GUARD_BYTES, Rng, assert_guards, available_isas, guarded, guarded_dst,
    };
    use super::*;

    /// Element counts of the kernel matrix, as the task sets them.
    const MATRIX_ELEMENTS: usize = 2048;
    const WIDTHS: [ElementWidth; 3] = [ElementWidth::Two, ElementWidth::Four, ElementWidth::Eight];

    /// The definition of a merge, written independently of the kernels.
    fn reference_merge(planes: &[Vec<u8>]) -> Vec<u8> {
        let width = planes.len();
        let n = planes[0].len();
        let mut out = vec![0; n * width];
        for k in 0..width {
            for i in 0..n {
                out[i * width + k] = planes[k][i];
            }
        }
        out
    }

    #[test]
    fn merge_matrix_scalar_equals_avx2_equals_reference() {
        let mut rng = Rng(1);
        for width in [2_usize, 4, 8] {
            for n in 0..MATRIX_ELEMENTS {
                // Exact allocations, so a sanitizer sees any load past the end of a plane.
                let planes: Vec<Vec<u8>> = (0..width).map(|_| rng.bytes(n)).collect();
                let views: Vec<&[u8]> = planes.iter().map(Vec::as_slice).collect();
                let expected = reference_merge(&planes);
                for isa in available_isas() {
                    let misalign = n % 33;
                    let mut buffer = guarded(n * width, misalign);
                    merge_planes(isa, &views, guarded_dst(&mut buffer, n * width, misalign));
                    let context = format!("{isa:?} width {width} n {n}");
                    assert_guards(&buffer, n * width, misalign, &context);
                    let start = GUARD_BYTES + misalign;
                    assert_eq!(&buffer[start..start + n * width], &expected[..], "{context}");
                }
            }
        }
    }

    /// Every width, every compressed plane, every length: the R1 matrix. A load that saw an already
    /// overwritten byte would show here as a mismatch, since all data is random.
    #[test]
    fn in_place_matrix_scalar_equals_avx2_equals_reference() {
        let mut rng = Rng(2);
        for width in [2_usize, 4, 8] {
            for compressed in 0..width {
                for n in 0..MATRIX_ELEMENTS {
                    let planes: Vec<Vec<u8>> = (0..width).map(|_| rng.bytes(n)).collect();
                    let expected = reference_merge(&planes);
                    let stored: Vec<&[u8]> = planes
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != compressed)
                        .map(|(_, plane)| plane.as_slice())
                        .collect();
                    for isa in available_isas() {
                        let len = n * width;
                        let misalign = n % 33;
                        let mut buffer = guarded(len, misalign);
                        let dst = guarded_dst(&mut buffer, len, misalign);
                        dst[len - n..].copy_from_slice(&planes[compressed]);
                        merge_in_place(isa, compressed, &stored, dst);
                        let context = format!("{isa:?} width {width} plane {compressed} n {n}");
                        assert_eq!(dst, &expected[..], "{context}");
                        assert_guards(&buffer, len, misalign, &context);
                    }
                }
            }
        }
    }

    /// The same at the exact end of an allocation, where a sanitizer catches a wide load.
    #[test]
    fn in_place_stays_inside_an_exact_allocation() {
        let mut rng = Rng(3);
        for width in [2_usize, 4, 8] {
            for n in [31, 32, 33, 63, 64, 65, 95, 96, 97] {
                let planes: Vec<Vec<u8>> = (0..width).map(|_| rng.bytes(n)).collect();
                let stored: Vec<&[u8]> = planes[..width - 1].iter().map(Vec::as_slice).collect();
                for isa in available_isas() {
                    let mut dst = vec![0; n * width];
                    dst.shrink_to_fit();
                    let tail = dst.len() - n;
                    dst[tail..].copy_from_slice(&planes[width - 1]);
                    merge_in_place(isa, width - 1, &stored, &mut dst);
                    assert_eq!(dst, reference_merge(&planes), "{isa:?} {width} {n}");
                }
            }
        }
    }

    #[test]
    fn gather_and_merge_fields_are_inverse_for_every_table() {
        let mut rng = Rng(4);
        for table in [FieldTable::Q80, FieldTable::Q4K, FieldTable::Q5K, FieldTable::Q6K] {
            let block = table.block_bytes() as usize;
            for blocks in [0, 1, 2, 37] {
                let source = rng.bytes(block * blocks);
                let streams: Vec<Vec<u8>> = table
                    .fields()
                    .iter()
                    .map(|field| {
                        let (offset, len) = (field.offset as usize, field.len as usize);
                        let mut stream = vec![0; blocks * len];
                        gather_field(&source, block, offset, len, &mut stream);
                        let expected: Vec<u8> = source
                            .chunks_exact(block)
                            .flat_map(|b| b[offset..offset + len].iter().copied())
                            .collect();
                        assert_eq!(stream, expected, "{table:?}");
                        stream
                    })
                    .collect();
                let views: Vec<&[u8]> = streams.iter().map(Vec::as_slice).collect();
                let mut merged = vec![0; source.len()];
                merge_fields(table, &views, &mut merged);
                assert_eq!(merged, source, "{table:?} {blocks}");
            }
        }
    }

    #[test]
    fn rotation_matches_integer_rotation_and_inverts() {
        let mut rng = Rng(5);
        for width in WIDTHS {
            let bytes = width.bytes() as usize;
            for elements in 0..160 {
                let source = rng.bytes(elements * bytes);
                let expected: Vec<u8> = source
                    .chunks_exact(bytes)
                    .flat_map(|e| {
                        let mut wide = [0; 8];
                        wide[..bytes].copy_from_slice(e);
                        let bits = width.bytes() * 8;
                        let value = u64::from_le_bytes(wide);
                        let high = value >> (bits - 1);
                        let rotated = ((value << 1) | high) & (u64::MAX >> (64 - bits));
                        rotated.to_le_bytes()[..bytes].to_vec()
                    })
                    .collect();
                for isa in available_isas() {
                    let mut data = source.clone();
                    rotate(isa, Rotation::Left, width, &mut data);
                    assert_eq!(data, expected, "{isa:?} {width:?} {elements}");
                    rotate(isa, Rotation::Right, width, &mut data);
                    assert_eq!(data, source, "{isa:?} {width:?} {elements}");
                }
            }
        }
    }

    #[test]
    fn rotation_moves_the_sign_bit_to_bit_zero() {
        for isa in available_isas() {
            let mut data = [0x01, 0x80];
            rotate(isa, Rotation::Left, ElementWidth::Two, &mut data);
            assert_eq!(data, [0x03, 0x00]);
        }
    }
}
