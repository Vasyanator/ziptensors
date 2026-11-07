//! File: commands.rs
//! Purpose: one function per subcommand: open or build through the core, hand the result to `report`.
//!
//! Key functions: [`run`], [`open`], [`open_budgeted`], and the private `encode`, `restore`, `extract`,
//! `merge`, `inspect`, `verify`. `pack` has its own file, for the directory walk, and `mount` its own,
//! for the signal handling of a foreground mount.
//!
//! Implementation notes: nothing here decides anything about bytes. The one rule enforced on this
//! side of the API is the core's own: [`restore_all`] runs every member name through
//! `format::check_restore_name` before it touches the disk, as `Member::restore_file` consults no name.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use anyhow::{Context, bail};
use ziptensors_core::format::check_restore_name;
use ziptensors_core::{
    Container, EncodeOptions, OpenOptions, PackBuilder, encode_file, layer_report,
};

use crate::cli::{
    BudgetArgs, Command, EncodeArgs, ExtractArgs, InspectArgs, MergeArgs, RestoreArgs, VerifyArgs,
};
use crate::report::{self, Progress};

/// What stdout failures say.
pub(crate) const STDOUT_ERROR: &str = "cannot write to standard output";

/// Runs `command`, with results on stdout.
///
/// # Errors
/// Returns whatever failed, with context naming the file. `errors::report` prints it.
pub(crate) fn run(command: &Command) -> anyhow::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match command {
        Command::Encode(args) => encode(args, &mut out),
        Command::Pack(args) => crate::pack::run(args, &mut out),
        Command::Restore(args) => restore(args, &mut out),
        Command::Extract(args) => extract(args, &mut out),
        Command::Merge(args) => merge(args, &mut out),
        Command::Inspect(args) => inspect(args, &mut out),
        Command::Verify(args) => verify(args, &mut out),
        #[cfg(any(target_os = "linux", windows))]
        Command::Mount(args) => crate::mount::run(args, &mut out),
    }?;
    out.flush().context(STDOUT_ERROR)
}

/// Opens the container at `path`, with default limits.
///
/// # Errors
/// Returns what [`Container::open`] returns, naming `path`.
fn open(path: &Path, threads: usize) -> anyhow::Result<Container> {
    open_with(path, &OpenOptions { threads, ..OpenOptions::default() })
}

/// Opens the container at `path` under the budget of `budget`, for what writes or shows members.
///
/// # Errors
/// Returns what [`Container::open`] returns, naming `path`.
pub(crate) fn open_budgeted(path: &Path, threads: usize, budget: &BudgetArgs) -> anyhow::Result<Container> {
    let defaults = OpenOptions::default();
    let max_total_decoded = budget.max_total_decoded.unwrap_or(defaults.max_total_decoded);
    open_with(path, &OpenOptions { threads, max_total_decoded, ..defaults })
}

/// Opens the container at `path`.
fn open_with(path: &Path, options: &OpenOptions) -> anyhow::Result<Container> {
    Container::open(path, options)
        .with_context(|| format!("cannot open container {}", path.display()))
}

/// `encode`: one file to a container of one member.
fn encode(args: &EncodeArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let options = args.write.options(EncodeOptions::for_encode());
    let encoded = encode_file(&args.input, &args.output, &options)
        .with_context(|| format!("cannot encode {}", args.input.display()))?;
    report::write_encode(out, &args.output, &encoded).context(STDOUT_ERROR)
}

/// `restore`: dispatches on `--all`.
fn restore(args: &RestoreArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let container = open_budgeted(&args.container, args.threads, &args.budget)?;
    if let Some(dir) = &args.all {
        return restore_all(&container, dir, out);
    }
    let Some(output) = &args.output else {
        // Clap requires one of the two; this keeps the outcome an error and not a panic if that changes.
        bail!("restore needs an output path or --all");
    };
    let member = container.single().with_context(|| {
        format!(
            "cannot restore {}: use --all or extract for a pack",
            args.container.display()
        )
    })?;
    let restored = member
        .restore_file(output)
        .with_context(|| format!("cannot restore to {}", output.display()))?;
    report::write_restored(out, member.name()?, output, &restored).context(STDOUT_ERROR)
}

/// `restore --all`: every member under `dir`. All names are checked before anything is created, so a
/// hostile pack leaves no trace at all.
fn restore_all(container: &Container, dir: &Path, out: &mut impl Write) -> anyhow::Result<()> {
    let members = container.members();
    for info in members {
        check_restore_name(&info.name).context("refusing to restore this container")?;
    }
    fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let mut bytes = 0_u64;
    for (done, info) in members.iter().enumerate() {
        let path = dir.join(&info.name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }
        let restored = container
            .member(&info.name)?
            .restore_file(&path)
            .with_context(|| format!("cannot restore to {}", path.display()))?;
        bytes = bytes.saturating_add(restored.bytes);
        tracing::info!(
            done = done + 1,
            total = members.len(),
            member = %info.name,
            bytes = restored.bytes,
            cumulative_bytes = bytes,
            "restored"
        );
    }
    report::write_restored_all(out, members.len(), bytes, dir).context(STDOUT_ERROR)
}

/// `extract`: one named member.
fn extract(args: &ExtractArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let container = open_budgeted(&args.container, args.threads, &args.budget)?;
    let member = container
        .member(&args.member)
        .with_context(|| format!("cannot extract from {}", args.container.display()))?;
    let restored = member
        .restore_file(&args.output)
        .with_context(|| format!("cannot extract to {}", args.output.display()))?;
    report::write_restored(out, &args.member, &args.output, &restored).context(STDOUT_ERROR)
}

/// `merge`: every member of every input, copied. All inputs open before the output is started.
fn merge(args: &MergeArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let sources = args
        .inputs
        .iter()
        .map(|path| open(path, args.threads))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let total = sources.iter().map(|source| source.members().len()).sum();
    let options = EncodeOptions { threads: args.threads, ..EncodeOptions::for_pack() };
    let mut builder = PackBuilder::new(&args.output, &options)
        .with_context(|| format!("cannot write {}", args.output.display()))?;
    let mut progress = Progress::new(total);
    for (source, path) in sources.iter().zip(&args.inputs) {
        for info in source.members() {
            let copied = builder.add_member_from(source, &info.name).with_context(|| {
                format!("cannot copy member {:?} of {}", info.name, path.display())
            })?;
            progress.record(&copied);
        }
    }
    let merged = builder
        .finish()
        .with_context(|| format!("cannot finish {}", args.output.display()))?;
    report::write_encode(out, &args.output, &merged).context(STDOUT_ERROR)
}

/// `inspect`: what the container says about itself. Reads no member data, and its index only.
fn inspect(args: &InspectArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let container = open(&args.container, 1)?;
    let meta = fs::metadata(&args.container)
        .with_context(|| format!("cannot read {}", args.container.display()))?;
    let written = if args.json {
        let layers = layer_report(&container)
            .with_context(|| format!("cannot read {}", args.container.display()))?;
        report::write_inspect_json(out, &args.container, meta.len(), container.members(), &layers)
    } else {
        report::write_inspect(out, &args.container, meta.len(), container.members())
    };
    written.context(STDOUT_ERROR)
}

/// `verify`: decodes every member against its hash. Stops at the first failure.
fn verify(args: &VerifyArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let container = open(&args.container, args.threads)?;
    let members = container.members();
    let mut bytes = 0_u64;
    for (done, info) in members.iter().enumerate() {
        container
            .member(&info.name)?
            .verify()
            .with_context(|| format!("verification of {} failed", args.container.display()))?;
        bytes = bytes.saturating_add(info.len);
        tracing::info!(
            done = done + 1,
            total = members.len(),
            member = %info.name,
            cumulative_bytes = bytes,
            "verified"
        );
    }
    report::write_verified(out, members.len(), bytes).context(STDOUT_ERROR)
}
