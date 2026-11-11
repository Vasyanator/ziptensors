//! File: edit.rs
//! Purpose: the commands that change a container in its own file: `edit`, `compact` and `recover`.
//!
//! Key functions: [`edit`], [`compact`], [`recover`].
//!
//! Implementation notes: argument translation only. Every rule of an edit (which operation sets are
//! legal, the result order, the refusals of each strategy) is `ziptensors_core::edit`'s, and every
//! rule of a recovery `ziptensors_core::recover`'s. `-F` maps to `Strategy::Append`, `-R` to
//! `Strategy::InPlace`, neither to `Strategy::Rewrite`, `-C` to `repack`; clap already refused the
//! combinations of two of them. `--delta-base` applies to every added or replacing file except the
//! one that is the base itself. `-R` prints a warning line on stderr before anything starts, since
//! an interruption from then on damages the container.

use std::ffi::OsStr;
use std::io::{self, ErrorKind, Write};
use std::path::Path;

use anyhow::{Context, anyhow};
use ziptensors_core::{
    EditOp, EditOptions, Error, Mode, RecoverOptions, ScanBudget, Strategy,
};

use crate::cli::{CompactArgs, EditArgs, NewFile, RecoverArgs};
use crate::commands::{STDOUT_ERROR, open_options};
use crate::errors::{ContainerContext, Hint};
use crate::report;

/// What a `recover` that cannot open its file for writing should suggest instead.
const READ_ONLY_HINT: &str = "the file cannot be opened for writing: \
`ziptensors recover --dry-run <CONTAINER>` only reads it and says what a recovery would cut";

/// The strategy of `-F`/`-R`, and how the report names it.
fn strategy(fast: bool, risky: bool) -> (Strategy, &'static str) {
    match (fast, risky) {
        (true, _) => (Strategy::Append, "appended, -F"),
        (false, true) => (Strategy::InPlace, "rewritten in place, -R"),
        (false, false) => (Strategy::Rewrite, "rewritten"),
    }
}

/// Tells the user, before an in-place rewrite of `path` starts, what an interruption costs.
fn warn_risky(path: &Path) {
    let mut stderr = io::stderr();
    // A warning that cannot be printed changes nothing about the run, which the user asked for; the
    // help of -R carries the same text.
    let _ = writeln!(
        stderr,
        "warning: -R rewrites {} in place: an interruption from now on DAMAGES the container, and \
         `ziptensors recover` may not repair it; close every reader of the file first",
        path.display()
    );
}

/// The member name of `file`: the name of `--add-as`, or the file name for `--add`.
///
/// # Errors
/// Returns an error if the name is not valid UTF-8, or the path of `--add` has no file name.
fn new_member_name(file: &NewFile) -> anyhow::Result<String> {
    let name: &OsStr = match &file.name {
        Some(name) => name,
        None => file
            .path
            .file_name()
            .ok_or_else(|| anyhow!("--add {}: the path names no file", file.path.display()))?,
    };
    name.to_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("{}: a member name must be valid UTF-8", name.to_string_lossy()))
}

/// The operations of `args`: removes, replaces, then adds in command line order. The core treats the
/// operations as a set, except that adds keep this order at the end of the result.
///
/// # Errors
/// Returns an error for a member name that is not valid UTF-8.
fn operations(args: &EditArgs) -> anyhow::Result<Vec<EditOp>> {
    let base_for = |name: &str| args.delta_base.clone().filter(|base| base != name);
    let mut ops: Vec<EditOp> =
        args.remove.iter().map(|name| EditOp::Remove { name: name.clone() }).collect();
    // Clap's `num_args = 2` leaves no remainder.
    let (pairs, _) = args.replace.as_chunks::<2>();
    for [name, path] in pairs {
        let name = name
            .to_str()
            .ok_or_else(|| anyhow!("{}: a member name must be valid UTF-8", name.to_string_lossy()))?
            .to_owned();
        let delta_base = base_for(&name);
        ops.push(EditOp::Replace { name, path: path.into(), delta_base });
    }
    for file in &args.new_files {
        let name = new_member_name(file)?;
        let delta_base = base_for(&name);
        ops.push(EditOp::Add { path: file.path.clone(), name, delta_base });
    }
    Ok(ops)
}

/// `edit`: the operations of `args` on its container.
///
/// # Errors
/// Returns an error for a member name that is not UTF-8, and every error of `core::edit`, naming
/// the container.
pub(crate) fn edit(args: &EditArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let ops = operations(args)?;
    let (strategy, label) = strategy(args.fast, args.risky);
    let options = EditOptions {
        strategy,
        mode: args.write.mode.map_or(Mode::Max, Mode::from),
        repack: args.compact,
        threads: args.write.threads,
        open: open_options(args.write.threads, &args.budget),
    };
    if args.risky {
        warn_risky(&args.container);
    }
    let edited = ziptensors_core::edit(&args.container, &ops, &options)
        .with_context(|| ContainerContext { what: "cannot edit", path: args.container.clone() })?;
    report::write_edit(out, &args.container, label, &edited).context(STDOUT_ERROR)
}

/// `compact`: an edit with no operation, which drops dead data, and with `-C` re-encodes every member.
///
/// # Errors
/// Every error of `core::edit`, naming the container.
pub(crate) fn compact(args: &CompactArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let (strategy, label) = strategy(false, args.risky);
    let options = EditOptions {
        strategy,
        mode: args.mode.map_or(Mode::Max, Mode::from),
        repack: args.compact,
        threads: args.threads,
        open: open_options(args.threads, &args.budget),
    };
    if args.risky {
        warn_risky(&args.container);
    }
    let compacted = ziptensors_core::edit(&args.container, &[], &options)
        .with_context(|| ContainerContext { what: "cannot compact", path: args.container.clone() })?;
    report::write_edit(out, &args.container, label, &compacted).context(STDOUT_ERROR)
}

/// Whether `error` is the core's refusal to open a file for writing for lack of permission.
fn is_write_refusal(error: &Error) -> bool {
    matches!(
        error,
        Error::Io { op: "open", source, .. }
            if matches!(source.kind(), ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem)
    )
}

/// `recover`: shortens the container to its latest valid generation, or with `--dry-run` says what
/// would be cut. A cut, done or planned, also prints a warning line on stderr naming the generation
/// kept, since everything after it is lost.
///
/// # Errors
/// Every error of `core::recover`, naming the container; a file that cannot be opened for writing
/// carries a [`Hint`] to `--dry-run`.
pub(crate) fn recover(args: &RecoverArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let options = RecoverOptions {
        open: open_options(args.threads, &args.budget),
        budget: ScanBudget {
            max_candidates: args.max_candidates,
            max_scan_bytes: args.max_scan_bytes.unwrap_or(u64::MAX),
        },
        verify: !args.no_verify,
        dry_run: args.dry_run,
    };
    let recovered = ziptensors_core::recover(&args.container, &options)
        .map_err(|error| {
            let read_only = !args.dry_run && is_write_refusal(&error);
            let error = anyhow::Error::new(error);
            if read_only { error.context(Hint(READ_ONLY_HINT)) } else { error }
        })
        .with_context(|| ContainerContext { what: "cannot recover", path: args.container.clone() })?;
    report::write_recovered(out, &args.container, args.dry_run, &recovered).context(STDOUT_ERROR)?;
    if let Some(warning) = report::recover_warning(&recovered) {
        // The result is on stdout already; a warning that cannot reach stderr changes nothing of it.
        let _ = writeln!(io::stderr(), "{warning}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{is_write_refusal, new_member_name, strategy};
    use crate::cli::NewFile;
    use std::ffi::OsString;
    use std::io;
    use std::path::PathBuf;
    use ziptensors_core::{Error, Strategy};

    #[test]
    fn flags_choose_one_strategy() {
        assert_eq!(strategy(false, false).0, Strategy::Rewrite);
        assert_eq!(strategy(true, false).0, Strategy::Append);
        assert_eq!(strategy(false, true).0, Strategy::InPlace);
    }

    #[test]
    fn an_added_file_is_named_after_itself_unless_named() {
        let plain = NewFile { name: None, path: PathBuf::from("dir/model.safetensors") };
        assert_eq!(new_member_name(&plain).expect("named"), "model.safetensors");
        let named = NewFile { name: Some(OsString::from("a/b.bin")), path: PathBuf::from("x") };
        assert_eq!(new_member_name(&named).expect("named"), "a/b.bin");
        let nameless = NewFile { name: None, path: PathBuf::from("/") };
        assert!(new_member_name(&nameless).is_err());
    }

    #[test]
    fn only_a_refused_open_is_a_write_refusal() {
        let io_error = |op, kind| Error::Io { op, path: PathBuf::new(), source: io::Error::from(kind) };
        assert!(is_write_refusal(&io_error("open", io::ErrorKind::PermissionDenied)));
        assert!(!is_write_refusal(&io_error("open", io::ErrorKind::NotFound)));
        assert!(!is_write_refusal(&io_error("set_len", io::ErrorKind::PermissionDenied)));
    }
}
