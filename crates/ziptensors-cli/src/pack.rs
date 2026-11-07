//! File: pack.rs
//! Purpose: `pack`: turning files and directory trees into the member list, and feeding the builder.
//!
//! Main responsibilities:
//! - walk each input without following links, in sorted order of member names;
//! - skip symbolic links and `*.incomplete` files with a log line, and count them;
//! - refuse bad names before the output is started, since a pack can run for hours;
//! - with `--delta-base`, add that file first and store every other file against it.
//!
//! Key structures: [`Walk`], [`Input`].
//!
//! Key functions: [`run`].
//!
//! Implementation notes: a symbolic link is skipped here and not left to `PackBuilder::add_file`, which
//! checks the name first, so a link with an unsafe name would stop the pack. `Error::SymlinkInput` from
//! `add_file` is still a skip: a walk and the read of the file can be hours apart. Everything other than
//! these two skips stops the pack, in particular every I/O failure, so a container never silently misses
//! a file. The delta base is never skipped. Which tensors pair with the base is the core's decision, so
//! every other file goes through `add_file_delta`, whatever its format: one that pairs nothing is stored
//! as `add_file` would store it, at the cost of one read of the base's header.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs::{self, FileType};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail};
use ziptensors_core::format::check_restore_name;
use ziptensors_core::{EncodeOptions, Error, PackBuilder};

use crate::cli::PackArgs;
use crate::commands::STDOUT_ERROR;
use crate::report::{self, Progress, Skipped};

/// Name suffix of partial downloads, which are not model files yet.
const INCOMPLETE_SUFFIX: &str = ".incomplete";

/// One file to add.
#[derive(Debug, PartialEq, Eq)]
struct Input {
    /// Where it is.
    path: PathBuf,
    /// Its name in the pack: relative to the input it was found under, `/` separated.
    member: String,
}

/// What a walk found.
#[derive(Debug, Default)]
struct Walk {
    /// Files to add: per input in the order given, then sorted by member name.
    inputs: Vec<Input>,
    /// What was left out.
    skipped: Skipped,
}

/// Logs and counts a symbolic link. Links are neither followed nor stored, spec section 2.
fn skip_symlink(path: &Path, skipped: &mut Skipped) {
    tracing::info!(path = %path.display(), "skipping symbolic link");
    skipped.symlinks += 1;
}

impl Walk {
    /// Adds the file or directory tree at `root`.
    ///
    /// # Errors
    /// Returns an error if `root` or anything under it cannot be read, a name is not UTF-8, or an entry
    /// is neither a file, a directory nor a link.
    fn add_root(&mut self, root: &Path) -> anyhow::Result<()> {
        let meta = fs::symlink_metadata(root)
            .with_context(|| format!("cannot read input {}", root.display()))?;
        let first = self.inputs.len();
        let mut pending = Vec::new();
        if meta.is_dir() {
            pending.push((root.to_path_buf(), String::new()));
        } else {
            self.add_entry(root, meta.file_type(), "", &mut pending)?;
        }
        while let Some((dir, prefix)) = pending.pop() {
            let entries = fs::read_dir(&dir)
                .with_context(|| format!("cannot list directory {}", dir.display()))?;
            for entry in entries {
                let entry =
                    entry.with_context(|| format!("cannot list directory {}", dir.display()))?;
                let file_type = entry
                    .file_type()
                    .with_context(|| format!("cannot read {}", entry.path().display()))?;
                self.add_entry(&entry.path(), file_type, &prefix, &mut pending)?;
            }
        }
        // Member names and not walk order: `read_dir` has no order of its own, and the order of members
        // is part of what makes the output reproducible.
        self.inputs[first..].sort_unstable_by(|a, b| a.member.cmp(&b.member));
        Ok(())
    }

    /// Sorts one directory entry into files, skips or `pending` directories. `prefix` is the member name
    /// of the parent, empty at the root.
    fn add_entry(
        &mut self,
        path: &Path,
        file_type: FileType,
        prefix: &str,
        pending: &mut Vec<(PathBuf, String)>,
    ) -> anyhow::Result<()> {
        if file_type.is_symlink() {
            skip_symlink(path, &mut self.skipped);
            return Ok(());
        }
        let name = path.file_name().and_then(OsStr::to_str).ok_or_else(|| {
            anyhow!("{}: a member name must be valid UTF-8", path.display())
        })?;
        let member =
            if prefix.is_empty() { name.to_owned() } else { format!("{prefix}/{name}") };
        if file_type.is_dir() {
            pending.push((path.to_path_buf(), member));
        } else if name.ends_with(INCOMPLETE_SUFFIX) {
            tracing::info!(path = %path.display(), "skipping incomplete file");
            self.skipped.incomplete += 1;
        } else if file_type.is_file() {
            self.inputs.push(Input { path: path.to_path_buf(), member });
        } else {
            bail!("{}: not a regular file, a directory or a link", path.display());
        }
        Ok(())
    }
}

/// Checks every member name the way the builder will, so that a bad one fails now and not hours in.
///
/// # Errors
/// Returns [`Error::UnsafeMemberName`] with the file as context, or an error naming a duplicate.
fn check_names(inputs: &[Input]) -> anyhow::Result<()> {
    let mut seen = HashSet::with_capacity(inputs.len());
    for input in inputs {
        check_restore_name(&input.member)
            .with_context(|| format!("cannot pack {}", input.path.display()))?;
        if !seen.insert(input.member.as_str()) {
            bail!(
                "two inputs give the member name {:?}, the second is {}",
                input.member,
                input.path.display()
            );
        }
    }
    Ok(())
}

/// Moves the input that is the file at `base` to the front of `inputs` and returns its member name. The
/// file is compared by what it is, through canonical paths, so `base` may name a file given directly
/// or one under an input directory, by any path. Called before the output is started.
///
/// # Errors
/// Returns an error if `base` cannot be read, is not a regular file, or is no file the walk found.
fn base_first(inputs: &mut [Input], base: &Path) -> anyhow::Result<String> {
    let canonical = fs::canonicalize(base)
        .with_context(|| format!("cannot read the delta base {}", base.display()))?;
    if !canonical.is_file() {
        bail!("the delta base {} is not a regular file", base.display());
    }
    let mut found = None;
    for (at, input) in inputs.iter().enumerate() {
        let path = fs::canonicalize(&input.path)
            .with_context(|| format!("cannot read input {}", input.path.display()))?;
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
    Ok(inputs[0].member.clone())
}

/// Refuses an output a walk of `roots` would pick up: as a file of its own, or at a path under a directory
/// that is walked, whether it exists yet or not, since the next run over the same tree would pack the
/// last result into the new one. A name the walk skips and a link, which it skips too, are fine. So is
/// an output whose directory does not exist, which the builder reports.
///
/// # Errors
/// Returns an error naming the root, or if a root cannot be read.
fn check_output(output: &Path, roots: &[PathBuf]) -> anyhow::Result<()> {
    let Some(name) = output.file_name() else { return Ok(()) };
    let skipped = name.to_str().is_some_and(|name| name.ends_with(INCOMPLETE_SUFFIX));
    let is_link = fs::symlink_metadata(output).is_ok_and(|meta| meta.is_symlink());
    let parent = output.parent().filter(|parent| !parent.as_os_str().is_empty());
    let Ok(parent) = fs::canonicalize(parent.unwrap_or(Path::new("."))) else { return Ok(()) };
    if skipped || is_link {
        return Ok(());
    }
    let target = parent.join(name);
    for root in roots {
        let meta = fs::symlink_metadata(root)
            .with_context(|| format!("cannot read input {}", root.display()))?;
        // The walk does not enter a root that is a link.
        if meta.is_symlink() {
            continue;
        }
        let canonical = fs::canonicalize(root)
            .with_context(|| format!("cannot read input {}", root.display()))?;
        let walked = if meta.is_dir() { target.starts_with(&canonical) } else { target == canonical };
        if walked {
            bail!(
                "the output {} is inside the input {}, which would pack it into itself",
                output.display(),
                root.display()
            );
        }
    }
    Ok(())
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
    check_output(&args.output, &args.inputs)?;
    let mut walk = Walk::default();
    for root in &args.inputs {
        walk.add_root(root)?;
    }
    let Walk { mut inputs, mut skipped } = walk;
    if inputs.is_empty() {
        bail!(
            "nothing to pack: no regular file among the inputs ({} symbolic links, {} incomplete files skipped)",
            skipped.symlinks,
            skipped.incomplete
        );
    }
    check_names(&inputs)?;
    let base_member =
        args.delta_base.as_deref().map(|base| base_first(&mut inputs, base)).transpose()?;
    tracing::info!(
        files = inputs.len(),
        symlinks_skipped = skipped.symlinks,
        incomplete_skipped = skipped.incomplete,
        delta_base = base_member.as_deref(),
        "walk finished"
    );
    let options = args.write.options(EncodeOptions::for_pack());
    let mut builder = PackBuilder::new(&args.output, &options)
        .with_context(|| format!("cannot write {}", args.output.display()))?;
    let mut progress = Progress::new(inputs.len());
    for (at, input) in inputs.iter().enumerate() {
        // With a delta base, `base_first` put it at 0, and every other file is stored against it.
        let base = base_member.as_deref().filter(|_| at > 0);
        let added = match base {
            Some(base) => builder.add_file_delta(&input.path, &input.member, base),
            None => builder.add_file(&input.path, &input.member),
        };
        match added {
            Ok(added) => progress.record(&added),
            // A base that became a link is not skipped: every later file names it.
            Err(Error::SymlinkInput { path }) if base_member.is_none() || at > 0 => {
                skip_symlink(&path, &mut skipped);
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("cannot pack {}", input.path.display()));
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
    use super::{Input, Walk, base_first, check_names, check_output};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::{env, fs, process};

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

    fn input(member: &str) -> Input {
        Input { path: PathBuf::from(member), member: member.to_owned() }
    }

    #[test]
    fn members_are_sorted_relative_to_the_root_with_forward_slashes() {
        let scratch = Scratch::new();
        for name in ["z.bin", "a/c.bin", "a/b/d.bin", "a.bin"] {
            scratch.file(name);
        }
        let mut walk = Walk::default();
        walk.add_root(&scratch.0).expect("walk");
        let members: Vec<_> = walk.inputs.iter().map(|input| input.member.as_str()).collect();
        assert_eq!(members, ["a.bin", "a/b/d.bin", "a/c.bin", "z.bin"]);
    }

    #[test]
    fn a_file_root_is_named_after_itself_and_incomplete_files_are_counted() {
        let scratch = Scratch::new();
        scratch.file("dir/x.bin");
        scratch.file("dir/x.bin.incomplete");
        let mut walk = Walk::default();
        walk.add_root(&scratch.0.join("dir/x.bin")).expect("walk file");
        walk.add_root(&scratch.0.join("dir")).expect("walk directory");
        assert_eq!(walk.inputs.len(), 2);
        assert!(walk.inputs.iter().all(|input| input.member == "x.bin"));
        assert_eq!(walk.skipped.incomplete, 1);
    }

    #[test]
    fn an_output_the_walk_would_pick_up_is_refused() {
        let scratch = Scratch::new();
        scratch.file("tree/deep/x.bin");
        scratch.file("other/y.bin");
        let tree = scratch.0.join("tree");
        let inside = |path: &str, roots: &[PathBuf]| check_output(&scratch.0.join(path), roots);
        let roots = [tree.clone()];
        // Whether the output exists yet does not matter: the next run would pick it up.
        assert!(inside("tree/out.pack", &roots).is_err());
        assert!(inside("tree/deep/x.bin", &roots).is_err());
        assert!(inside("tree/deep/../out.pack", &roots).is_err());
        assert!(inside("tree/out.pack", &[scratch.0.join("other"), tree.clone()]).is_err());
        assert!(inside("other/y.bin", &[scratch.0.join("other/y.bin")]).is_err());

        inside("out.pack", &roots).expect("beside the tree");
        inside("other/out.pack", &roots).expect("in another tree");
        inside("tree/out.incomplete", &roots).expect("a name the walk skips");
        inside("missing/out.pack", &roots).expect("for the builder to report");
    }

    #[cfg(unix)]
    #[test]
    fn an_output_through_a_link_is_compared_where_it_really_is() {
        let scratch = Scratch::new();
        scratch.file("tree/x.bin");
        let link = scratch.0.join("link");
        std::os::unix::fs::symlink(scratch.0.join("tree"), &link).expect("links");
        let roots = [scratch.0.join("tree")];
        assert!(check_output(&link.join("out.pack"), &roots).is_err());
        check_output(&scratch.0.join("out.pack"), &[link]).expect("a linked root is not walked");
    }

    #[test]
    fn a_missing_root_is_an_error() {
        let scratch = Scratch::new();
        let mut walk = Walk::default();
        let error = walk.add_root(&scratch.0.join("absent")).expect_err("missing root");
        assert!(format!("{error:#}").contains("cannot read input"), "{error:#}");
    }

    #[test]
    fn the_delta_base_moves_to_the_front_and_the_rest_keep_their_order() {
        let scratch = Scratch::new();
        for name in ["tree/a.bin", "tree/base.bin", "tree/c.bin", "other.bin"] {
            scratch.file(name);
        }
        let mut walk = Walk::default();
        walk.add_root(&scratch.0.join("tree")).expect("walk directory");
        walk.add_root(&scratch.0.join("other.bin")).expect("walk file");
        // Another spelling of the same file still finds it.
        let base = scratch.0.join("tree/../tree/base.bin");
        let member = base_first(&mut walk.inputs, &base).expect("among the inputs");
        assert_eq!(member, "base.bin");
        let members: Vec<_> = walk.inputs.iter().map(|input| input.member.as_str()).collect();
        assert_eq!(members, ["base.bin", "a.bin", "c.bin", "other.bin"]);
    }

    #[test]
    fn a_delta_base_outside_the_inputs_or_not_a_file_is_refused() {
        let scratch = Scratch::new();
        scratch.file("tree/a.bin");
        scratch.file("elsewhere.bin");
        let mut walk = Walk::default();
        walk.add_root(&scratch.0.join("tree")).expect("walk");
        let outside = base_first(&mut walk.inputs, &scratch.0.join("elsewhere.bin"));
        let error = outside.expect_err("not walked");
        assert!(error.to_string().contains("not among the files to pack"), "{error}");
        let directory = base_first(&mut walk.inputs, &scratch.0.join("tree"));
        let error = directory.expect_err("a directory");
        assert!(error.to_string().contains("not a regular file"), "{error}");
        let missing = base_first(&mut walk.inputs, &scratch.0.join("absent.bin"));
        let error = missing.expect_err("missing");
        assert!(error.to_string().contains("cannot read the delta base"), "{error}");
    }

    #[test]
    fn names_are_checked_before_anything_starts() {
        assert!(check_names(&[input("a"), input("b/c")]).is_ok());
        let error = check_names(&[input("a"), input("a")]).expect_err("duplicate");
        assert!(error.to_string().contains("two inputs"), "{error}");
        let error = check_names(&[input("a:b")]).expect_err("unsafe");
        assert!(format!("{error:#}").contains("unsafe member name"), "{error:#}");
    }
}
