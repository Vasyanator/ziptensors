//! File: tests/write_memory.rs
//! Purpose: asserts the writer's memory bound with a counting allocator: at most 6 × threads × chunk
//! plus the index, and 9 × threads × chunk for a pack that streams delta blobs, independent of tensor
//! size.
//!
//! Main responsibilities: encode files that take the costliest paths, and compare the peak of live
//! Rust allocations with the bound, at several thread counts and two tensor sizes.
//!
//! Key structures: `Counting`, the global allocator of this test binary.
//!
//! Implementation notes: a binary of its own, because a global allocator counts the whole process and
//! other tests would disturb it; it holds one test for the same reason. Allocations inside zstd's C
//! code bypass this allocator: a compression context is a few MiB per thread, constant. The mix goes
//! through every buffer of a workspace: a rotated chunk (`max` floats), a large stream (`Q8_0` fields) and
//! a large compressed stream (`whole` on noise), then a duplicate and a late widening mismatch. The
//! delta case streams a finetune against its base, so every chunk in the window carries its base
//! chunk and each workspace grows a residual buffer, with `rotl1` in both trials.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "common/fixtures.rs"]
mod fixtures;

use fixtures::{Elem, GgufOptions, Rng, TensorSpec};
use ziptensors_core::codec::widen_into;
use ziptensors_core::{EncodeOptions, Mode, PackBuilder, View};

/// Counts live bytes and their peak, passing everything to the system allocator.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let live = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

fn shrank(by: usize) {
    LIVE.fetch_sub(by, Ordering::Relaxed);
}

// SAFETY: every call forwards to `System` with its arguments unchanged, and the counters neither
// allocate nor touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract is forwarded.
        unsafe { System.dealloc(pointer, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counted as a move, the worst an allocator may do: both blocks live at the moment of the
        // copy. A bound proven under this also holds for a realloc that works in place.
        grew(new_size);
        // SAFETY: the caller's contract is forwarded.
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if pointer.is_null() {
            shrank(new_size);
        } else {
            shrank(layout.size());
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const CHUNK: usize = 1 << 20;
/// What the bound leaves to fixed-size bookkeeping and the small index of these files. It does not grow
/// with the chunk size, the thread count or the tensors.
const FIXED_SLACK: usize = 256 << 10;

/// Unique scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let name = format!("ziptensors-write-memory-{}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// A GGUF file of tensors of `tensor_bytes` each: noise in a `Q8_0` tensor, the costliest field split,
/// and a float tensor.
fn quantized_file(tensor_bytes: u64) -> Vec<u8> {
    let specs = [
        TensorSpec::new("q8", &[tensor_bytes / 34 * 32], Elem::Q80),
        TensorSpec::new("f32", &[tensor_bytes / 4], Elem::F32),
    ];
    fixtures::gguf(1, &specs, &GgufOptions::default()).bytes
}

/// A safetensors file of noise under a `whole` chain, the largest compressed stream.
fn noise_file(tensor_bytes: u64) -> Vec<u8> {
    let specs = [TensorSpec::new("fp8", &[tensor_bytes], Elem::F8E4M3)];
    fixtures::safetensors(2, &specs).bytes
}

/// An `F32` file exact as `F16` except for its last element, the worst of the widening speculation.
fn late_mismatch_file(tensor_bytes: u64) -> Vec<u8> {
    let elements = tensor_bytes / 4;
    let specs = [TensorSpec::new("wide", &[elements], Elem::F32)];
    let mut fixture = fixtures::safetensors(3, &specs);
    let mut rng = Rng::new(4);
    let narrow = fixtures::payload(&mut rng, Elem::F16, usize::try_from(elements).expect("fits") * 2);
    let tensor = &fixture.tensors[0];
    let start = usize::try_from(tensor.offset).expect("fits");
    let end = start + usize::try_from(tensor.len).expect("fits");
    widen_into(View::F16ToF32, &narrow, &mut fixture.bytes[start..end]).expect("sizes match");
    fixture.bytes[end - 4..end].copy_from_slice(&0x3F80_0001_u32.to_le_bytes());
    fixture.bytes
}

/// Encodes all of `inputs` with `threads` threads, and returns the peak of live bytes above the start.
fn peak_of(inputs: &[PathBuf], output: &Path, threads: usize) -> usize {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let options = EncodeOptions { mode: Mode::Max, threads };
    let mut builder = PackBuilder::new(output, &options).expect("starts");
    for (at, input) in inputs.iter().enumerate() {
        builder.add_file(input, &format!("input-{at}")).expect("adds");
    }
    let report = builder.finish().expect("commits");
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    assert!(report.members.iter().any(|member| member.widen_restreams == 1));
    assert!(report.total.dedup_saved > 0);
    peak
}

/// Writes the generated files for tensors of `tensor_bytes`, one of them twice for a duplicate.
fn inputs(scratch: &Scratch, tensor_bytes: u64) -> Vec<PathBuf> {
    let files = [
        quantized_file(tensor_bytes),
        noise_file(tensor_bytes),
        late_mismatch_file(tensor_bytes),
    ];
    let mut paths = Vec::new();
    for (at, bytes) in files.iter().enumerate() {
        let path = scratch.0.join(format!("{tensor_bytes}-{at}"));
        fs::write(&path, bytes).expect("input");
        paths.push(path);
    }
    paths.push(paths[0].clone());
    paths
}

/// A base of two `BF16` tensors of `tensor_bytes` each and its finetune, with sparse edits that make
/// both tensors delta blobs.
fn delta_inputs(scratch: &Scratch, tensor_bytes: u64) -> [PathBuf; 2] {
    let specs = [
        TensorSpec::new("a", &[tensor_bytes / 2], Elem::Bf16),
        TensorSpec::new("b", &[tensor_bytes / 2], Elem::Bf16),
    ];
    let base = fixtures::safetensors(5, &specs);
    let mut tuned = base.clone();
    let data = usize::try_from(tuned.tensors[0].offset).expect("fits");
    tuned.bytes[data..].iter_mut().step_by(14).for_each(|byte| *byte ^= 1);
    let base_path = scratch.0.join(format!("{tensor_bytes}-base"));
    let tuned_path = scratch.0.join(format!("{tensor_bytes}-tuned"));
    fs::write(&base_path, &base.bytes).expect("input");
    fs::write(&tuned_path, &tuned.bytes).expect("input");
    [base_path, tuned_path]
}

/// Encodes the base and its finetune against it with `threads` threads, and returns the peak of live
/// bytes above the start.
fn delta_peak_of(inputs: &[PathBuf; 2], output: &Path, threads: usize) -> usize {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let options = EncodeOptions { mode: Mode::Max, threads };
    let mut builder = PackBuilder::new(output, &options).expect("starts");
    builder.add_file(&inputs[0], "base").expect("adds");
    let tuned = builder.add_file_delta(&inputs[1], "tuned", "base").expect("adds");
    builder.finish().expect("commits");
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    assert_eq!(tuned.delta_blobs, 2);
    peak
}

#[test]
fn encoding_stays_within_its_chunks_per_thread_whatever_the_tensor_size() {
    const SMALL: u64 = 16 << 20;
    const LARGE: u64 = 64 << 20;
    let scratch = Scratch::new();
    let small = inputs(&scratch, SMALL);
    let large = inputs(&scratch, LARGE);
    let output = scratch.0.join("output");
    for threads in [1, 2, 4] {
        let bound = 6 * threads * CHUNK + FIXED_SLACK;
        let at_small = peak_of(&small, &output, threads);
        let at_large = peak_of(&large, &output, threads);
        eprintln!("threads {threads}: {at_small} at 16 MiB, {at_large} at 64 MiB, bound {bound}");
        assert!(at_small <= bound, "{at_small} bytes at 16 MiB tensors, bound {bound}");
        assert!(at_large <= bound, "{at_large} bytes at 64 MiB tensors, bound {bound}");
        // Timing moves the peak by at most what the window holds, 2 × threads chunks. Four times the
        // bytes must add nothing beyond that.
        assert!(at_large <= at_small + 2 * threads * CHUNK, "{at_large} over {at_small}");
    }

    let small = delta_inputs(&scratch, SMALL);
    let large = delta_inputs(&scratch, LARGE);
    for threads in [1, 2, 4] {
        let bound = 9 * threads * CHUNK + FIXED_SLACK;
        let at_small = delta_peak_of(&small, &output, threads);
        let at_large = delta_peak_of(&large, &output, threads);
        eprintln!("delta, threads {threads}: {at_small} at 16 MiB, {at_large} at 64 MiB, bound {bound}");
        assert!(at_small <= bound, "{at_small} bytes at 16 MiB tensors, bound {bound}");
        assert!(at_large <= bound, "{at_large} bytes at 64 MiB tensors, bound {bound}");
        // The window holds two buffers per chunk here, so timing moves the peak by twice as much.
        assert!(at_large <= at_small + 4 * threads * CHUNK, "{at_large} over {at_small}");
    }
}
