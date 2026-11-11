//! File: tree.rs
//! Purpose: `compress` and `decompress`: per-file containers over files and directory trees, through
//! `ziptensors_core::tree::{compress_tree, decompress_tree}`.
//!
//! Key functions: [`compress`], [`decompress`].
//!
//! Implementation notes: argument translation and reporting only. Which files are weights, how
//! containers are named, collisions, pairs and when a copy may be removed are the core's rules. Each
//! path given is one run of the core, in order; a run that fails as a whole (a collision, an
//! unreadable root) stops the command, while a file that fails inside a run is one `FAILED` line,
//! the run goes on, and the command exits with 1 at the end. A directory without `-r` was refused as
//! a usage error by `Cli::parse_checked`. Warnings of a run (skipped links) go to stderr.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, bail};
use ziptensors_core::tree::{
    CompressTreeOptions, DecompressTreeOptions, TreeReport, compress_tree, decompress_tree,
};
use ziptensors_core::{EncodeOptions, Error};

use crate::cli::{CompressArgs, DecompressArgs};
use crate::commands::STDOUT_ERROR;
use crate::report::{self, TreeDirection};

/// Prints `report` of a run in `direction` and its warnings, and returns how many files failed in it.
///
/// # Errors
/// Returns the failure of a write to `out`.
fn print_run(
    report: &TreeReport,
    direction: TreeDirection,
    out: &mut impl Write,
) -> anyhow::Result<usize> {
    report::write_tree(out, report, direction).context(STDOUT_ERROR)?;
    let mut stderr = io::stderr();
    for warning in &report.warnings {
        // The results are on stdout; a warning that cannot reach stderr changes none of them.
        let _ = writeln!(stderr, "warning: {warning}");
    }
    Ok(report.totals().failed)
}

/// Runs `run` (a run in `direction`) over every path, printing each report, and fails at the end
/// if a file failed.
///
/// # Errors
/// Returns the first error of a whole run, naming its path, or an error counting the failed files.
fn each_path(
    paths: &[PathBuf],
    direction: TreeDirection,
    out: &mut impl Write,
    run: impl Fn(&std::path::Path) -> Result<TreeReport, Error>,
) -> anyhow::Result<()> {
    let verb = match direction {
        TreeDirection::Compress => "compress",
        TreeDirection::Decompress => "decompress",
    };
    let mut failed = 0_usize;
    for path in paths {
        let report = run(path).with_context(|| format!("cannot {verb} {}", path.display()))?;
        failed = failed.saturating_add(print_run(&report, direction, out)?);
    }
    if failed > 0 {
        // The lines above name each file and why; the error line only counts them.
        out.flush().context(STDOUT_ERROR)?;
        bail!("{failed} file(s) failed to {verb}; the lines marked FAILED say why");
    }
    Ok(())
}

/// `compress`: every weight file under each path becomes a container of one member beside it.
///
/// # Errors
/// See [`each_path`].
pub(crate) fn compress(args: &CompressArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let mut options = CompressTreeOptions::default();
    options.keep = args.tree.keep;
    options.follow_symlinks = args.follow_symlinks;
    options.dry_run = args.tree.dry_run;
    options.encode = args.write.options(EncodeOptions::for_pack());
    options.max_total_decoded = args.budget.max_total_decoded.unwrap_or(u64::MAX);
    each_path(&args.tree.paths, TreeDirection::Compress, out, |path| compress_tree(path, &options))
}

/// `decompress`: every container of one member under each path is restored beside it.
///
/// # Errors
/// See [`each_path`].
pub(crate) fn decompress(args: &DecompressArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let mut options = DecompressTreeOptions::default();
    options.keep = args.tree.keep;
    options.force = args.force;
    options.dry_run = args.tree.dry_run;
    options.threads = args.threads;
    options.max_total_decoded = args.budget.max_total_decoded.unwrap_or(u64::MAX);
    each_path(&args.tree.paths, TreeDirection::Decompress, out, |path| {
        decompress_tree(path, &options)
    })
}
