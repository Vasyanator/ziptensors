//! File: codec/mod.rs
//! Purpose: chunk encoding and decoding: byte plane and field layouts, bit rotation, delta
//! residuals, widening views, stream framing and scratch workspaces. Works on memory only.
//!
//! Main responsibilities: declares the submodules and re-exports the public entry points.
//!
//! Key structures: [`ChunkMeta`], [`StreamPolicy`], [`Workspace`].
//!
//! Key functions: [`encode_chunk`], [`decode_chunk_into`], [`encode_delta_chunk`],
//! [`decode_delta_chunk_into`], [`residual`], [`combine`], [`locate_error`], [`widen_into`],
//! [`narrow_exact`], [`narrow_into`].
//!
//! Implementation notes: spec sections 6 and 7 define everything here. `Chain`, `View`, field tables
//! and stream codec ids belong to `crate::chain`. `unsafe` lives in `kernels_x86.rs` alone.

mod chunk;
mod delta;
mod kernels;
#[cfg(target_arch = "x86_64")]
mod kernels_x86;
mod stream;
#[cfg(test)]
mod test_support;
mod widen;
mod workspace;

pub use chunk::{
    ChunkMeta, MAX_CHUNK_BYTES, decode_chunk_into, decode_delta_chunk_into, encode_chunk,
    encode_delta_chunk, locate_error,
};
pub use delta::{combine, residual};
pub use stream::StreamPolicy;
pub use widen::{narrow_exact, narrow_into, widen_into};
pub use workspace::Workspace;
