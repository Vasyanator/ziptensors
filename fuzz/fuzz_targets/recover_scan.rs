//! File: fuzz/fuzz_targets/recover_scan.rs
//! Purpose: fuzz target. Any bytes as a file to recover, scanned for its latest valid generation.
//!
//! Implementation notes: the work and the oracles are in `src/harness.rs`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| ziptensors_fuzz::harness::recover_scan(input));
