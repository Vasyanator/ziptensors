//! File: tests/c_smoke.rs
//! Purpose: compiles `tests/c/smoke.c` against the checked-in header, links the real shared library,
//! and runs it over a generated file.
//!
//! Main responsibilities: prove the header, the exported symbols and the struct layouts agree for an
//! actual C compiler.
//!
//! Implementation notes: Unix only, since the link and the library lookup are spelled for a Unix `cc`.
//! Skips with one line if `cc` is absent. The library is found beside this test binary, where cargo
//! puts it because the crate is also an rlib; a missing library fails, as it would hide the test.
//! Everything is written under a unique directory in the system temp directory, removed at the end.

#![cfg(unix)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

/// The name of the tensor `generated_input` holds at a fixed place.
const TENSOR: &str = "weights";

/// Removes a directory when dropped, on success and on a failed assert.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let dir = env::temp_dir().join(format!("ziptensors-capi-smoke-{}", process::id()));
        fs::create_dir(&dir).expect("create the scratch directory");
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// A small safetensors file with one F32 tensor of seeded noise.
fn generated_input() -> Vec<u8> {
    let mut state = 0x5EED_u32;
    let payload: Vec<u8> = (0..4096_u32 * 4)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_le_bytes()[3]
        })
        .collect();
    let header = format!(
        r#"{{"{TENSOR}":{{"dtype":"F32","shape":[4096],"data_offsets":[0,{}]}}}}"#,
        payload.len()
    );
    let header_len = u64::try_from(header.len()).expect("a short header");
    let mut file = header_len.to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    file
}

/// Whether a C compiler answers.
fn have_cc() -> bool {
    Command::new("cc").arg("--version").output().is_ok_and(|output| output.status.success())
}

/// The shared library of this build.
fn shared_library() -> PathBuf {
    let exe = env::current_exe().expect("path of the test binary");
    let deps = exe.parent().expect("the test binary has a directory");
    let name = format!("{}ziptensors_capi{}", env::consts::DLL_PREFIX, env::consts::DLL_SUFFIX);
    let library = deps.join(name);
    assert!(library.is_file(), "no shared library at {}", library.display());
    library
}

#[test]
fn c_program_roundtrips_through_the_shared_library() {
    if !have_cc() {
        eprintln!("skipped: no working `cc` on PATH");
        return;
    }
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let scratch = Scratch::new();
    let library = shared_library();
    let library_dir = library.parent().expect("the library has a directory");
    let exe = scratch.0.join("smoke");
    let compiled = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-I"])
        .arg(crate_dir.join("include"))
        .arg(crate_dir.join("tests/c/smoke.c"))
        .arg(&library)
        .arg(format!("-Wl,-rpath,{}", library_dir.display()))
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run cc");
    assert!(compiled.status.success(), "cc: {}", String::from_utf8_lossy(&compiled.stderr));

    let input = scratch.0.join("input.safetensors");
    fs::write(&input, generated_input()).expect("write the input");
    let ran = Command::new(&exe)
        .arg(&input)
        .arg(TENSOR)
        .arg(&scratch.0)
        .output()
        .expect("run the C program");
    assert!(ran.status.success(), "smoke: {}", String::from_utf8_lossy(&ran.stderr));
}
