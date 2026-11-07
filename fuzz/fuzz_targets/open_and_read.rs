//! File: fuzz/fuzz_targets/open_and_read.rs
//! Purpose: fuzz target. Any bytes as a container, opened and read every way a caller can.
//!
//! Implementation notes: the work and the oracles are in `src/harness.rs`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| ziptensors_fuzz::harness::open_and_read(input));
