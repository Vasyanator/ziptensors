//! File: fuzz/src/bin/make_seeds.rs
//! Purpose: writes the seed corpus of the three targets, so that no corpus lives in the tree.
//!
//! Implementation notes: takes the output directory as its one argument and creates
//! `<dir>/<target>/`. Seeds are generated, the same on every run. Profile seeds come from the
//! fixtures of `ziptensors-core`'s tests.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[path = "../../../crates/ziptensors-core/tests/common/fixtures.rs"]
mod fixtures;

use fixtures::{Elem, GgufOptions, OnnxModel, OnnxTensor, PtZipOptions, TensorSpec};
use ziptensors_fuzz::harness::{
    PLAIN_CHAINS, chain_table, delta_sample, one_plane_body, sample, to_fixed_up_input,
};

/// Bytes after the four control bytes of a `decode_chunk` seed.
const CHUNK_BODY: usize = 2048;
/// Bit of a `decode_chunk` form byte that makes the base of a delta chain zero.
const ZERO_BASE: u8 = 0x10;

/// Writes `seeds` as files `0`, `1`, ... under `dir`.
fn write_all(dir: &Path, seeds: &[Vec<u8>]) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    for (id, seed) in seeds.iter().enumerate() {
        fs::write(dir.join(id.to_string()), seed)?;
    }
    Ok(())
}

/// Seeds of `open_and_read`: the sample and the delta sample in both input forms, for a few seeds.
fn container_seeds() -> Vec<Vec<u8>> {
    let mut seeds = Vec::new();
    for seed in 0..6 {
        let file = if seed < 4 { sample(seed).file } else { delta_sample(seed).file };
        seeds.push(to_fixed_up_input(&file));
        seeds.push([&[0], file.as_slice()].concat());
    }
    seeds
}

/// Seeds of `decode_chunk`: every chain, form and policy over one patterned body, delta chains with
/// either base, then the planes chains with exactly one compressed stream, under the policy at
/// index 2, for the in-place route; with a delta step, over a zero base, so the residual is that body.
fn chunk_seeds() -> Vec<Vec<u8>> {
    let body: Vec<u8> = (0..CHUNK_BODY)
        .map(|at| if at % 4 == 3 { 0 } else { u8::try_from(at % 251).unwrap_or(0) })
        .collect();
    let chains = chain_table();
    let mut seeds = Vec::new();
    for chain in 0..u8::try_from(chains.len()).unwrap_or(u8::MAX) {
        let bases: &[u8] = if usize::from(chain) < PLAIN_CHAINS { &[0] } else { &[0, ZERO_BASE] };
        for form in [1, 2, 5, 6, 9, 10, 13, 14] {
            for base in bases {
                seeds.push([&[chain, form | base, 0xFF, 0x03], body.as_slice()].concat());
            }
        }
    }
    let delta = u8::try_from(PLAIN_CHAINS).unwrap_or(u8::MAX);
    for (chain, width) in [(4, 2), (8, 4), (12, 8)] {
        // `chain` is `planes(width)` in the harness table, `delta + chain` the same after a delta step.
        // Half a body of blocks stays inside the body.
        let body = one_plane_body(u64::from(chain), CHUNK_BODY, width);
        for (chain, base) in [(chain, 0), (delta + chain, ZERO_BASE)] {
            let block = chains.get(usize::from(chain)).map(|chain| chain.block_bytes());
            let block = block.and_then(|block| usize::try_from(block).ok()).unwrap_or(width);
            let blocks = u8::try_from(CHUNK_BODY / 2 / block).unwrap_or(u8::MAX);
            seeds.push([&[chain, 0b1001 | base, blocks, 0], body.as_slice()].concat());
            seeds.push([&[chain, 0b1010 | base, blocks, 0], body.as_slice()].concat());
        }
    }
    seeds
}

/// Seeds of `profile_scan`: one small file of each source format.
fn profile_seeds() -> Vec<Vec<u8>> {
    let specs = [TensorSpec::new("a", &[64], Elem::F32), TensorSpec::new("b", &[32], Elem::F16)];
    let onnx = OnnxModel {
        initializers: vec![OnnxTensor::raw("w", &[32], Elem::F32)],
        ..OnnxModel::default()
    };
    vec![
        fixtures::safetensors(1, &specs).bytes,
        fixtures::gguf(2, &specs, &GgufOptions::default()).bytes,
        fixtures::onnx(3, &onnx).bytes,
        fixtures::pt_zip(4, &specs, &PtZipOptions::default()).bytes,
    ]
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::env::args_os().nth(1).map(PathBuf::from).ok_or("usage: make_seeds <dir>")?;
    write_all(&root.join("open_and_read"), &container_seeds())?;
    write_all(&root.join("decode_chunk"), &chunk_seeds())?;
    write_all(&root.join("profile_scan"), &profile_seeds())?;
    Ok(())
}
