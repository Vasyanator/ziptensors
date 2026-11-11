//! File: pack.rs
//! Purpose: `pack`: the core's walk of the inputs, the delta base, and feeding the builder.
//!
//! Main responsibilities:
//! - refuse an output the walk would pick up, and walk the inputs, through
//!   `write::inputs::{check_output, collect_inputs}` (no link followed, `*.incomplete` skipped, names
//!   checked and sorted per input, all before the output is started, since a pack can run for hours);
//! - with `--delta-base`, add that file first and store every other file against it.
//!
//! Key functions: [`run`].
//!
//! Implementation notes: `Error::SymlinkInput` from `add_file` is still a skip: a walk and the read of
//! the file can be hours apart. Everything else stops the pack, in particular every I/O failure, so a
//! container never silently misses a file. The delta base is never skipped. Which tensors pair with
//! the base is the core's decision, so every other file goes through `add_file_delta`, whatever its
//! format: one that pairs nothing is stored as `add_file` would store it, at the cost of one read of
//! the base's header.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use ziptensors_core::write::inputs::check_output;
use ziptensors_core::{EncodeOptions, Error, Inputs, PackBuilder, collect_inputs};

use crate::cli::PackArgs;
use crate::commands::STDOUT_ERROR;
use crate::report::{self, Progress, Skipped};

/// Logs and counts a file that became a symbolic link after the walk. Links are neither followed
/// nor stored, spec section 2.
fn skip_symlink(path: &Path, skipped: &mut Skipped) {
    tracing::info!(path = %path.display(), "skipping symbolic link");
    skipped.symlinks += 1;
}

/// Moves the input that is the file at `base` to the front of `inputs` and returns its member name. The
/// file is compared by what it is, through canonical paths, so `base` may name a file given directly
/// or one under an input directory, by any path. Called before the output is started.
///
/// # Errors
/// Returns an error if `base` cannot be read, is not a regular file, or is no file the walk found.
fn base_first(inputs: &mut [(PathBuf, String)], base: &Path) -> anyhow::Result<String> {
    let canonical = fs::canonicalize(base)
        .with_context(|| format!("cannot read the delta base {}", base.display()))?;
    if !canonical.is_file() {
        bail!("the delta base {} is not a regular file", base.display());
    }
    let mut found = None;
    for (at, (input, _)) in inputs.iter().enumerate() {
        let path = fs::canonicalize(input)
            .with_context(|| format!("cannot read input {}", input.display()))?;
        if path == canonical {
            found = Some(at);
            break;
        }
    }
    let Some(at) = found else {
        bail!(
            "the delta base {} is not among the files to pack: give it as an input, or put it under an input directory",
            base.display()
        );
    };
    // The base is added before every file stored against it; the others keep their order.
    inputs[..=at].rotate_right(1);
    Ok(inputs[0].1.clone())
}

/// `pack`: walks the inputs, adds every file, finishes the container. With `--delta-base` the base is
/// added first and every other file through `PackBuilder::add_file_delta` against it.
///
/// # Errors
/// Returns an error if the output is among the inputs, if there is nothing to pack, if the delta base is
/// not among the files to pack, on any failure of the walk, and on any failure of the builder other than
/// a file that became a link (the delta base becoming one is a failure too). The output is then not
/// created.
pub(crate) fn run(args: &PackArgs, out: &mut impl Write) -> anyhow::Result<()> {
    if let Err(error) = check_output(&args.output, &args.inputs) {
        let what = if matches!(error, Error::OutputIsInput { .. }) {
            format!(
                "the output {} is inside an input, which would pack it into itself",
                args.output.display()
            )
        } else {
            "cannot read the inputs".to_owned()
        };
        return Err(error).context(what);
    }
    let Inputs { files: mut inputs, skipped_symlinks, skipped_incomplete } =
        collect_inputs(&args.inputs).context("cannot read the inputs")?;
    let mut skipped = Skipped { symlinks: skipped_symlinks, incomplete: skipped_incomplete };
    if inputs.is_empty() {
        bail!(
            "nothing to pack: no regular file among the inputs ({} symbolic links, {} incomplete files skipped)",
            skipped.symlinks,
            skipped.incomplete
        );
    }
    let base_member =
        args.delta_base.as_deref().map(|base| base_first(&mut inputs, base)).transpose()?;
    tracing::info!(files = inputs.len(), delta_base = base_member.as_deref(), "walk finished");
    let options = args.write.options(EncodeOptions::for_pack());
    let mut builder = PackBuilder::new(&args.output, &options)
        .with_context(|| format!("cannot write {}", args.output.display()))?;
    let mut progress = Progress::new(inputs.len());
    for (at, (path, member)) in inputs.iter().enumerate() {
        // With a delta base, `base_first` put it at 0, and every other file is stored against it.
        let base = base_member.as_deref().filter(|_| at > 0);
        let added = match base {
            Some(base) => builder.add_file_delta(path, member, base),
            None => builder.add_file(path, member),
        };
        match added {
            Ok(added) => progress.record(&added),
            // A base that became a link is not skipped: every later file names it.
            Err(Error::SymlinkInput { path }) if base_member.is_none() || at > 0 => {
                skip_symlink(&path, &mut skipped);
            }
            Err(error) => {
                return Err(error).with_context(|| format!("cannot pack {}", path.display()));
            }
        }
    }
    tracing::info!(output = %args.output.display(), "all files added, verifying the container");
    let packed = builder
        .finish()
        .with_context(|| format!("cannot finish {}", args.output.display()))?;
    report::write_encode(out, &args.output, &packed).context(STDOUT_ERROR)?;
    report::write_skipped(out, skipped).context(STDOUT_ERROR)
}

#[cfg(test)]
mod tests {
    use super::base_first;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::{env, fs, process};
    use ziptensors_core::collect_inputs;

    static NEXT: AtomicU32 = AtomicU32::new(0);

    /// A directory under the system temp directory, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!("ziptensors-pack-{}-{id}", process::id()));
            fs::create_dir_all(&path).expect("create scratch");
            Self(path)
        }

        fn file(&self, relative: &str) {
            let path = self.0.join(relative);
            let parent = path.parent().expect("a file under the scratch directory");
            fs::create_dir_all(parent).expect("create parent");
            fs::write(path, b"x").expect("write file");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                eprintln!("cannot remove scratch: {error}");
            }
        }
    }

    #[test]
    fn the_delta_base_moves_to_the_front_and_the_rest_keep_their_order() {
        let scratch = Scratch::new();
        for name in ["tree/a.bin", "tree/base.bin", "tree/c.bin", "other.bin"] {
            scratch.file(name);
        }
        let roots = [scratch.0.join("tree"), scratch.0.join("other.bin")];
        let mut inputs = collect_inputs(&roots).expect("walk").files;
        // Another spelling of the same file still finds it.
        let base = scratch.0.join("tree/../tree/base.bin");
        let member = base_first(&mut inputs, &base).expect("among the inputs");
        assert_eq!(member, "base.bin");
        let members: Vec<_> = inputs.iter().map(|(_, member)| member.as_str()).collect();
        assert_eq!(members, ["base.bin", "a.bin", "c.bin", "other.bin"]);
    }

    #[test]
    fn a_delta_base_outside_the_inputs_or_not_a_file_is_refused() {
        let scratch = Scratch::new();
        scratch.file("tree/a.bin");
        scratch.file("elsewhere.bin");
        let mut inputs = collect_inputs(&[scratch.0.join("tree")]).expect("walk").files;
        let outside = base_first(&mut inputs, &scratch.0.join("elsewhere.bin"));
        let error = outside.expect_err("not walked");
        assert!(error.to_string().contains("not among the files to pack"), "{error}");
        let directory = base_first(&mut inputs, &scratch.0.join("tree"));
        let error = directory.expect_err("a directory");
        assert!(error.to_string().contains("not a regular file"), "{error}");
        let missing = base_first(&mut inputs, &scratch.0.join("absent.bin"));
        let error = missing.expect_err("missing");
        assert!(error.to_string().contains("cannot read the delta base"), "{error}");
    }
}
