//! File: codec/workspace.rs
//! Purpose: the reusable per-worker state of chunk encoding and decoding.
//!
//! Main responsibilities: owns the scratch buffers, the lazily created zstd contexts and the
//! selected instruction set, so that a worker allocates once and not per chunk.
//!
//! Key structures: [`Workspace`].
//!
//! Implementation notes: buffers grow exactly, to the largest chunk seen, and never shrink. Their sizes derive
//! from the caller's raw length, never from a value in a file, and that length is limited by
//! `MAX_CHUNK_BYTES`, so a hostile chunk cannot inflate them.

use super::kernels::Isa;
use super::stream::ZstdContexts;

/// Scratch state for one worker. Create one per thread, or keep them in a pool; it is `Send`, and a
/// call needs exclusive access. Reusing one across chains and policies is fine, and it never changes
/// the bytes produced.
#[derive(Debug)]
pub struct Workspace {
    /// Kernels to run. Detected once at construction.
    pub(super) isa: Isa,
    /// Whether a chunk with one compressed plane may merge inside `dst`. Always true outside tests,
    /// which clear it to prove that the scratch path gives identical bytes.
    pub(super) in_place: bool,
    /// Decoded compressed streams while they wait to be merged.
    pub(super) scratch: Vec<u8>,
    /// Rotated copy of the chunk being encoded.
    pub(super) rotated: Vec<u8>,
    /// Residual of the delta chunk being encoded against its base.
    pub(super) residual: Vec<u8>,
    /// The stream being encoded.
    pub(super) stream: Vec<u8>,
    /// Output of the compressor, kept only if the stream does not end up stored.
    pub(super) compressed: Vec<u8>,
    /// zstd contexts.
    pub(super) zstd: ZstdContexts,
}

impl Workspace {
    /// Creates an empty workspace, detecting the instruction set. Allocates nothing yet.
    #[must_use]
    pub fn new() -> Self {
        let isa = Isa::detect();
        tracing::debug!(?isa, "codec workspace created");
        Self {
            isa,
            in_place: true,
            scratch: Vec::new(),
            rotated: Vec::new(),
            residual: Vec::new(),
            stream: Vec::new(),
            compressed: Vec::new(),
            zstd: ZstdContexts::default(),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `buffer` resized to exactly `len` initialized bytes, with unspecified contents. Zeroes only
/// when it grows, so a worker pays for this once per size.
///
/// The old allocation is freed first and the new one is exactly `len` long. The contents need no
/// keeping, and growing in place would briefly hold both, or leave up to twice `len` by `Vec` doubling,
/// either of which breaks the reader's memory bound of spec section 9.2.
pub(super) fn sized(buffer: &mut Vec<u8>, len: usize) -> &mut [u8] {
    if buffer.len() < len {
        *buffer = Vec::new();
        buffer.resize(len, 0);
    }
    &mut buffer[..len]
}

#[cfg(test)]
mod tests {
    use super::{Workspace, sized};

    #[test]
    fn workspace_can_move_between_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<Workspace>();
    }

    #[test]
    fn sized_returns_the_exact_length_and_keeps_capacity() {
        let mut buffer = Vec::new();
        assert_eq!(sized(&mut buffer, 16).len(), 16);
        assert_eq!(sized(&mut buffer, 4).len(), 4);
        assert_eq!(buffer.len(), 16);
    }

    #[test]
    fn sized_grows_to_the_exact_length() {
        let mut buffer = Vec::new();
        sized(&mut buffer, 700);
        sized(&mut buffer, 1000);
        assert_eq!(buffer.capacity(), 1000);
    }
}
