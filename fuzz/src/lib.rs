//! File: fuzz/src/lib.rs
//! Purpose: crate root of the fuzz crate, which exposes the harness to the targets.
//!
//! Implementation notes: `harness.rs` is a plain file so that `tests/malicious.rs` in
//! `ziptensors-core` can compile the same source on stable.

pub mod harness;
