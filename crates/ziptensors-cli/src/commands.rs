//! File: commands.rs
//! Purpose: one function per reading subcommand, plus the dispatcher and the shared open helpers:
//! open or build through the core, hand the result to `report`.
//!
//! Key functions: [`run`], [`open_budgeted`], and the private `encode`, `restore`, `extract`,
//! `merge`, `inspect`, `verify`. `pack` has its own file, for the delta base; `edit`, `compact` and
//! `recover` theirs (`edit.rs`); `compress` and `decompress` theirs (`tree.rs`); `mount` its own, for
//! the signal handling of a foreground mount.
//!
//! Implementation notes: nothing here decides anything about bytes. `extract ... -` streams the
//! member to standard output through `Member::decode_to`, so its bytes leave before the member's hash
//! is known; the result line then goes to standard error, which keeps standard output the member alone.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use anyhow::{Context, bail};
use ziptensors_core::{
    Container, EncodeOptions, OpenOptions, PackBuilder, check_member_names, encode_file, layer_report,
};

use crate::cli::{
    BudgetArgs, Command, EncodeArgs, ExtractArgs, InspectArgs, MergeArgs, RestoreArgs, VerifyArgs,
};
use crate::errors::ContainerContext;
use crate::report::{self, Progress};

/// What stdout failures say.
pub(crate) const STDOUT_ERROR: &str = "cannot write to standard output";

/// What stderr failures say, for the result line of `extract ... -`.
const STDERR_ERROR: &str = "cannot write to standard error";

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
        Command::Edit(args) => crate::edit::edit(args, &mut out),
        Command::Compact(args) => crate::edit::compact(args, &mut out),
        Command::Recover(args) => crate::edit::recover(args, &mut out),
        Command::Restore(args) => restore(args, &mut out),
        Command::Extract(args) => extract(args, &mut out),
        Command::Merge(args) => merge(args, &mut out),
        Command::Inspect(args) => inspect(args, &mut out),
        Command::Verify(args) => verify(args, &mut out),
        Command::Compress(args) => crate::tree::compress(args, &mut out),
        Command::Decompress(args) => crate::tree::decompress(args, &mut out),
        #[cfg(any(target_os = "linux", windows))]
        Command::Mount(args) => crate::mount::run(args, &mut out),
    }?;
    out.flush().context(STDOUT_ERROR)
}

/// The open options of a command with `threads` and the budget of `budget`; everything else is the
/// core's default. No budget given means none, as in the core.
#[must_use]
pub(crate) fn open_options(threads: usize, budget: &BudgetArgs) -> OpenOptions {
    let defaults = OpenOptions::default();
    let max_total_decoded = budget.max_total_decoded.unwrap_or(defaults.max_total_decoded);
    OpenOptions { threads, max_total_decoded, ..defaults }
}

/// Opens the container at `path` under the budget of `budget`.
///
/// # Errors
/// Returns what [`Container::open`] returns, naming `path`.
pub(crate) fn open_budgeted(path: &Path, threads: usize, budget: &BudgetArgs) -> anyhow::Result<Container> {
    Container::open(path, &open_options(threads, budget)).with_context(|| ContainerContext {
        what: "cannot open container",
        path: path.to_path_buf(),
    })
}

/// `encode`: one file to a container of one member.
fn encode(args: &EncodeArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let options = args.write.options(EncodeOptions::for_encode());
    let encoded = encode_file(&args.input, &args.output, &options)
        .with_context(|| format!("cannot encode {}", args.input.display()))?;
    report::write_encode(out, &args.output, &encoded).context(STDOUT_ERROR)
}

/// `restore`: the only member, or with `--all` every member through `Container::restore_all`, which
/// checks every name before it creates anything.
fn restore(args: &RestoreArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let container = open_budgeted(&args.container, args.threads, &args.budget)?;
    if let Some(dir) = &args.all {
        let restored = container.restore_all(dir).with_context(|| {
            format!("cannot restore {} to {}", args.container.display(), dir.display())
        })?;
        return report::write_restored_all(out, restored.members, restored.bytes, dir)
            .context(STDOUT_ERROR);
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

/// `extract`: one named member, to a file or, for `-`, to stdout with the result line on stderr.
fn extract(args: &ExtractArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let container = open_budgeted(&args.container, args.threads, &args.budget)?;
    let member = container
        .member(&args.member)
        .with_context(|| format!("cannot extract from {}", args.container.display()))?;
    if args.to_stdout() {
        // Bytes leave before the member hash is known: a mismatch fails the command after the fact,
        // and the caller of the process must discard what it read (documented in the help of OUTPUT).
        let restored = member.decode_to(out).with_context(|| {
            format!("cannot extract {} to standard output", args.member)
        })?;
        out.flush().context(STDOUT_ERROR)?;
        return report::write_restored(&mut io::stderr().lock(), &args.member, &args.output, &restored)
            .context(STDERR_ERROR);
    }
    let restored = member
        .restore_file(&args.output)
        .with_context(|| format!("cannot extract to {}", args.output.display()))?;
    report::write_restored(out, &args.member, &args.output, &restored).context(STDOUT_ERROR)
}

/// `merge`: every member of every input, copied. All inputs open, and every member name is checked
/// across them, before the output is started, so a name two inputs share copies nothing.
fn merge(args: &MergeArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let sources = args
        .inputs
        .iter()
        .map(|path| open_budgeted(path, args.threads, &args.budget))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let names = sources.iter().flat_map(|source| source.members().iter().map(|info| info.name.as_str()));
    check_member_names(names).context("cannot merge the inputs")?;
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
    let container = open_budgeted(&args.container, 1, &args.budget)?;
    let meta = fs::metadata(&args.container)
        .with_context(|| format!("cannot read {}", args.container.display()))?;
    let tensors = if args.tensors {
        let members = container.members().iter().map(|info| container.member(&info.name));
        let members = members
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("cannot read {}", args.container.display()))?;
        Some(members)
    } else {
        None
    };
    let tensors = tensors.as_deref();
    let written = if args.json {
        let layers = layer_report(&container)
            .with_context(|| format!("cannot read {}", args.container.display()))?;
        let document = report::InspectDocument {
            path: &args.container,
            file_bytes: meta.len(),
            members: container.members(),
            layers: &layers,
            tensors,
        };
        report::write_inspect_json(out, &document)
    } else {
        report::write_inspect(out, &args.container, meta.len(), container.members(), tensors)
    };
    written.context(STDOUT_ERROR)
}

/// `verify`: decodes every member against its hash. Stops at the first failure.
fn verify(args: &VerifyArgs, out: &mut impl Write) -> anyhow::Result<()> {
    let container = open_budgeted(&args.container, args.threads, &args.budget)?;
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
