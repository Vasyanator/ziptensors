//! File: `codec/test_support.rs`
//! Purpose: seeded data generators and the kernel list shared by the codec tests.
//!
//! Key functions: [`Rng`], [`available_isas`], [`guarded`].
//!
//! Implementation notes: compiled for tests only. Fixtures are generated, never read from disk.

use super::kernels::Isa;

/// Bytes of canary on each side of a guarded destination.
pub(super) const GUARD_BYTES: usize = 64;
/// Value of the canary bytes.
const GUARD_VALUE: u8 = 0xA5;

/// `SplitMix64`: small, seeded and good enough for test data.
#[derive(Debug)]
pub(super) struct Rng(pub(super) u64);

impl Rng {
    /// Next 64 random bits.
    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A random value below `limit`, which is not zero.
    pub(super) fn below(&mut self, limit: usize) -> usize {
        usize::try_from(self.next_u64() % limit as u64).expect("below a usize")
    }

    /// One random byte.
    pub(super) fn byte(&mut self) -> u8 {
        self.next_u64().to_le_bytes()[0]
    }

    /// `len` random bytes.
    pub(super) fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len + 8);
        while out.len() < len {
            out.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        out.truncate(len);
        out.shrink_to_fit();
        out
    }
}

/// Every kernel set this machine can run.
pub(super) fn available_isas() -> Vec<Isa> {
    let detected = Isa::detect();
    if detected == Isa::Scalar { vec![Isa::Scalar] } else { vec![Isa::Scalar, detected] }
}

/// A `len` byte zeroed destination between two canary bands, offset by `misalign` bytes.
pub(super) fn guarded(len: usize, misalign: usize) -> Vec<u8> {
    vec![GUARD_VALUE; GUARD_BYTES + misalign + len + GUARD_BYTES]
}

/// The destination inside a buffer from [`guarded`].
pub(super) fn guarded_dst(buffer: &mut [u8], len: usize, misalign: usize) -> &mut [u8] {
    &mut buffer[GUARD_BYTES + misalign..GUARD_BYTES + misalign + len]
}

/// Asserts that both canary bands of a buffer from [`guarded`] are intact.
pub(super) fn assert_guards(buffer: &[u8], len: usize, misalign: usize, context: &str) {
    let (head, rest) = buffer.split_at(GUARD_BYTES + misalign);
    let tail = &rest[len..];
    assert!(head.iter().all(|&b| b == GUARD_VALUE), "front canary changed: {context}");
    assert!(tail.iter().all(|&b| b == GUARD_VALUE), "rear canary changed: {context}");
}
