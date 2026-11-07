//! File: testkit.rs
//! Purpose: test only: real containers built in a scratch directory from generated files.
//!
//! Main responsibilities: [`Scratch`], a unique directory under the system temp directory removed on
//! drop; [`Fixture`], a container written by the public `PackBuilder` from seeded, safetensors-like
//! files, with the bytes of every member for comparison.
//!
//! Key structures: [`Scratch`], [`Fixture`].
//!
//! Implementation notes: also included by `tests/fuse.rs` through `#[path]`, so it names nothing of
//! this crate. Data comes from splitmix64 with a fixed seed: no test reads a real model.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::{env, fs, process};

use ziptensors_core::{EncodeOptions, Mode, PackBuilder};

static NEXT_SCRATCH: AtomicU32 = AtomicU32::new(0);

/// A directory of one test, removed on drop.
#[derive(Debug)]
pub struct Scratch(PathBuf);

impl Scratch {
    /// A new empty directory whose name holds `label`, the process id and a counter.
    pub fn new(label: &str) -> Self {
        let id = NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("ziptensors-mount-{label}-{}-{id}", process::id()));
        fs::create_dir_all(&path).expect("create scratch directory");
        Self(path)
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // A test must not fail in drop; a leftover directory under the temp directory is reported.
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove scratch {}: {error}", self.0.display());
        }
    }
}

/// A container in a scratch directory and the content of each of its members.
#[derive(Debug)]
pub struct Fixture {
    /// The container file.
    pub container: PathBuf,
    /// Member name and bytes, in the order they were added.
    pub files: Vec<(String, Vec<u8>)>,
}

impl Fixture {
    /// A model-like tree: a 3 MiB safetensors file two levels down (several chunks), a smaller one, a
    /// JSON file and an empty file.
    pub fn standard(scratch: &Scratch) -> Self {
        Self::build(
            scratch,
            &[
                ("models/unet/diffusion.safetensors", 3 << 20),
                ("models/vae.safetensors", 300 << 10),
                ("config.json", 0),
                ("empty.bin", 0),
            ],
        )
    }

    /// A container of members called `names` with about `len` bytes each: a safetensors file of one
    /// `F16` tensor for a name ending in `.safetensors`, a small JSON text for `.json`, seeded bytes
    /// otherwise, exactly `len` of them.
    pub fn build(scratch: &Scratch, members: &[(&str, usize)]) -> Self {
        let options = EncodeOptions { mode: Mode::Fast, threads: 2 };
        let container = scratch.path().join("fixture.ziptensors");
        let mut builder = PackBuilder::new(&container, &options).expect("start container");
        let mut files = Vec::new();
        for (seed, (name, len)) in (1_u64..).zip(members) {
            let extension = Path::new(name).extension().and_then(|extension| extension.to_str());
            let content = if extension == Some("safetensors") {
                safetensors(seed, len / 2)
            } else if extension == Some("json") {
                format!("{{\"model\":\"fixture\",\"seed\":{seed}}}\n").into_bytes()
            } else {
                bytes(seed, *len)
            };
            // Sources are named by number: member names may clash as paths (`a` and `a/b`).
            let source = scratch.path().join(format!("source-{seed}"));
            fs::write(&source, &content).expect("write source");
            builder.add_file(&source, name).expect("add member");
            files.push(((*name).to_owned(), content));
        }
        builder.finish().expect("finish container");
        Self { container, files }
    }

    /// Sum of the member lengths.
    pub fn total_len(&self) -> u64 {
        self.files.iter().map(|(_, content)| u64::try_from(content.len()).expect("small")).sum()
    }
}

/// `len` bytes from `seed` by splitmix64, with low-entropy runs so that the codecs have work.
fn bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|at| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let noise = (z ^ (z >> 31)).to_le_bytes()[0];
            if at % 2 == 1 { noise & 0x3c } else { noise }
        })
        .collect()
}

/// A safetensors file of one `F16` tensor `t` of `elements` elements: plausible half floats, whose
/// high byte (sign, exponent) varies little.
fn safetensors(seed: u64, elements: usize) -> Vec<u8> {
    let data = bytes(seed, elements * 2);
    let header = format!(
        "{{\"t\":{{\"dtype\":\"F16\",\"shape\":[{elements}],\"data_offsets\":[0,{}]}}}}",
        data.len()
    );
    let header_len = u64::try_from(header.len()).expect("a short header");
    let mut file = header_len.to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    file
}
