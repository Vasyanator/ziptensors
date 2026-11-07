//! File: fuzz/fuzz_targets/profile_scan.rs
//! Purpose: fuzz target. Any bytes as a source file, through the format parsers.
//!
//! Implementation notes: the work and the oracles are in `src/harness.rs`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| ziptensors_fuzz::harness::profile_scan(input));
