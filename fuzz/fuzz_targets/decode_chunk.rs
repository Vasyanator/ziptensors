//! File: fuzz/fuzz_targets/decode_chunk.rs
//! Purpose: fuzz target. Stored chunks, hostile and near-valid, through `decode_chunk_into`.
//!
//! Implementation notes: the work and the oracles are in `src/harness.rs`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| ziptensors_fuzz::harness::decode_chunk(input));
