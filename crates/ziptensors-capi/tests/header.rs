//! File: tests/header.rs
//! Purpose: proves `include/ziptensors.h` is what cbindgen generates from the current sources.
//!
//! Main responsibilities: regenerate the header in memory and compare it with the checked-in copy.
//!
//! Implementation notes: this is the only way the header is produced, so the generator version is the
//! one `Cargo.lock` pins. With `ZIPTENSORS_UPDATE_HEADER=1` the test rewrites the file instead of
//! comparing; off by default, since a test must not write into the tree.

use std::env;
use std::fs;
use std::path::Path;

/// The header cbindgen makes from this crate's sources, as bytes.
fn generate(crate_dir: &Path) -> Vec<u8> {
    let config = crate_dir.join("cbindgen.toml");
    let config = cbindgen::Config::from_file(config).expect("cbindgen.toml parses");
    let bindings = cbindgen::Builder::new()
        .with_config(config)
        .with_src(crate_dir.join("src/lib.rs"))
        .generate()
        .expect("cbindgen generates the header");
    let mut text = Vec::new();
    bindings.write(&mut text);
    text
}

#[test]
fn header_is_up_to_date() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = crate_dir.join("include/ziptensors.h");
    let generated = generate(crate_dir);
    if env::var_os("ZIPTENSORS_UPDATE_HEADER").is_some() {
        fs::write(&path, &generated).expect("write the header");
        return;
    }
    let checked_in = fs::read(&path).expect("read the checked-in header");
    assert!(
        checked_in == generated,
        "{} is stale: rerun this test with ZIPTENSORS_UPDATE_HEADER=1 and review the diff",
        path.display()
    );
}
