//! File: testkit.rs
//! Purpose: test only: real containers built in a scratch directory from generated files.
//!
//! Main responsibilities: [`Scratch`], a unique directory under the system temp directory removed on
//! drop; [`Fixture`], a container written by the public `PackBuilder` from seeded, safetensors-like
//! files, with the bytes of every member for comparison; [`OverlayFixture`], a model directory where
//! some files are per-file containers (written by the public `encode_file`), for the overlay mount.
//!
//! Key structures: [`Scratch`], [`Fixture`], [`OverlayFixture`].
//!
//! Implementation notes: also included by `tests/fuse.rs` and `tests/winfsp.rs` through `#[path]`, so
//! it names nothing of this crate. Data comes from splitmix64 with a fixed seed: no test reads a real
//! model.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::{env, fs, process};

use ziptensors_core::{EncodeOptions, Mode, PackBuilder, encode_file};

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

/// A model directory under a scratch directory, with what an overlay of it must show.
#[derive(Debug)]
pub struct OverlayFixture {
    /// The directory to mount.
    pub root: PathBuf,
    /// Every file the overlay shows, `/`-separated, with its bytes, sorted by name.
    pub shown: Vec<(String, Vec<u8>)>,
    /// Every directory the overlay shows, `/`-separated, sorted.
    pub dirs: Vec<String>,
    /// Files on disk the overlay hides, `/`-separated: the container of a kept pair, and links.
    pub hidden: Vec<String>,
    /// The per-file containers, `/`-separated, sorted.
    pub containers: Vec<String>,
}

impl OverlayFixture {
    /// `scratch/tree` holding: `config.json` and `notes/readme.txt` as they are; three per-file
    /// containers, `unet/diffusion_pytorch_model.ziptensors` (3 MiB, several chunks),
    /// `text_encoder/deep/nested/model.ziptensors` and `weights.ziptensors` (raw bytes); a kept pair
    /// `vae/model.safetensors` beside its container `vae/model.ziptensors`; a pack of two members
    /// `pack.ziptensors`, shown as it is; an empty directory `empty`; and on Unix a link
    /// `link.safetensors` to `config.json`, left out.
    pub fn standard(scratch: &Scratch) -> Self {
        let root = scratch.path().join("tree");
        let config = b"{\"model\":\"overlay\"}\n".to_vec();
        let readme = bytes(11, 5000);
        let unet = safetensors(12, 3 << 19);
        let vae = safetensors(13, 100 << 10);
        let text = safetensors(14, 50 << 10);
        let weights = bytes(15, 200 << 10);
        let mut shown = vec![
            ("config.json".to_owned(), config),
            ("notes/readme.txt".to_owned(), readme),
            ("unet/diffusion_pytorch_model.safetensors".to_owned(), unet),
            ("vae/model.safetensors".to_owned(), vae),
            ("text_encoder/deep/nested/model.safetensors".to_owned(), text),
            ("weights.bin".to_owned(), weights),
        ];
        for (name, content) in &shown {
            write_file(&root.join(name), content);
        }
        let containers = [
            per_file(&root.join("unet/diffusion_pytorch_model.safetensors"), false),
            per_file(&root.join("vae/model.safetensors"), true),
            per_file(&root.join("text_encoder/deep/nested/model.safetensors"), false),
            per_file(&root.join("weights.bin"), false),
        ];
        let pack = Fixture::build(scratch, &[("a.bin", 1000), ("b/c.bin", 2000)]);
        let pack_bytes = fs::read(&pack.container).expect("read pack");
        write_file(&root.join("pack.ziptensors"), &pack_bytes);
        shown.push(("pack.ziptensors".to_owned(), pack_bytes));
        fs::create_dir(root.join("empty")).expect("create empty directory");
        #[cfg(unix)]
        std::os::unix::fs::symlink("config.json", root.join("link.safetensors")).expect("create link");
        let link = cfg!(unix).then_some("link.safetensors");
        let hidden = ["vae/model.ziptensors"].into_iter().chain(link).map(str::to_owned).collect();
        shown.sort();
        let mut dirs: Vec<String> = ["empty", "notes", "text_encoder", "text_encoder/deep", "text_encoder/deep/nested", "unet", "vae"]
            .map(str::to_owned)
            .to_vec();
        dirs.sort();
        let mut containers: Vec<String> = containers
            .iter()
            .map(|path| {
                let rel = path.strip_prefix(&root).expect("under the root");
                rel.to_str().expect("UTF-8").replace('\\', "/")
            })
            .collect();
        containers.sort();
        Self { root, shown, dirs, hidden, containers }
    }

    /// The bytes the overlay shows under `name`.
    pub fn content(&self, name: &str) -> &[u8] {
        let found = self.shown.iter().find(|(shown, _)| shown == name);
        &found.unwrap_or_else(|| panic!("{name} is not shown")).1
    }
}

/// Writes `content` at `path`, creating its directories.
fn write_file(path: &Path, content: &[u8]) {
    fs::create_dir_all(path.parent().expect("a file in a directory")).expect("create directories");
    fs::write(path, content).expect("write file");
}

/// Encodes `original` into a per-file container beside it (`<stem>.ziptensors`, member named after
/// the file) and removes the original unless `keep`. Returns the container.
pub fn per_file(original: &Path, keep: bool) -> PathBuf {
    let container = original.with_extension("ziptensors");
    let options = EncodeOptions { mode: Mode::Fast, threads: 2 };
    encode_file(original, &container, &options).expect("encode a per-file container");
    if !keep {
        fs::remove_file(original).expect("remove the original");
    }
    container
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
