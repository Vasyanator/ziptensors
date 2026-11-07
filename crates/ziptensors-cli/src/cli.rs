//! File: cli.rs
//! Purpose: the command line grammar: subcommands, their arguments, and what they translate to.
//!
//! Key structures: [`Cli`], [`Command`], [`WriteArgs`], [`ModeArg`], one `*Args` per subcommand.
//!
//! Implementation notes: this file parses and nothing else. `Cli::parse_checked` adds the one check
//! across arguments clap cannot state: `pack --delta-base` with a mode other than `max` is a usage
//! error. `--threads 0`, the default, means one thread per logical CPU, as in the core. A missing
//! `--mode` keeps the default of the entry point: `balanced` for `encode`, `max` for `pack`. `mount`
//! exists on Linux (FUSE) and Windows (`WinFsp`), the systems with a backend; its thread flag is per
//! backend (`--fuse-threads`, `--winfsp-threads`), and on Windows its help carries the notice the
//! `WinFsp` license requires.

use std::path::PathBuf;

use clap::error::ErrorKind;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
#[cfg(any(target_os = "linux", windows))]
use ziptensors_core::CacheOptions;
use ziptensors_core::{EncodeOptions, Mode};

/// Lossless tensor container: encode, pack, restore, inspect, verify.
#[derive(Debug, Parser)]
#[command(name = "ziptensors", version)]
pub(crate) struct Cli {
    /// What to do.
    #[command(subcommand)]
    pub(crate) command: Command,
}

impl Cli {
    /// Parses the command line and checks what clap cannot express, exiting with clap's usage error
    /// (code 2) on a conflict: `pack --delta-base` with a `--mode` other than `max`, the only mode in
    /// which `PackBuilder::add_file_delta` stores delta blobs. A flag that would be ignored for the
    /// hours of a pack is refused rather than warned about.
    #[must_use]
    pub(crate) fn parse_checked() -> Self {
        let cli = Self::parse();
        if let Command::Pack(pack) = &cli.command
            && pack.delta_base.is_some()
            && let Some(mode) = pack.write.mode
        {
            match mode {
                ModeArg::Max => {}
                ModeArg::Fast | ModeArg::Balanced => Self::command()
                    .error(
                        ErrorKind::ArgumentConflict,
                        "--delta-base needs --mode max: no other mode stores delta blobs",
                    )
                    .exit(),
            }
        }
        cli
    }
}

/// The subcommands.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Encode one file as a container of one member.
    Encode(EncodeArgs),
    /// Pack files and directory trees into one container.
    Pack(PackArgs),
    /// Restore the only member of a container, or with --all every member.
    Restore(RestoreArgs),
    /// Restore one named member of a container.
    Extract(ExtractArgs),
    /// Copy every member of several containers into a new one, with no recompression.
    Merge(MergeArgs),
    /// Print what a container holds.
    Inspect(InspectArgs),
    /// Decode every member and compare it with its stated BLAKE3.
    Verify(VerifyArgs),
    /// Show every member as a read-only file under a directory, until Ctrl-C.
    #[cfg(target_os = "linux")]
    Mount(MountArgs),
    /// Show every member as a read-only file on a drive letter or a new directory, until Ctrl-C.
    #[cfg(windows)]
    Mount(MountArgs),
}

/// Writer effort, `Mode` of the core as a command line value.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum ModeArg {
    /// Plane split and entropy coding only.
    Fast,
    /// Adds widening views and field split of quantized data.
    Balanced,
    /// Adds the rotation trial and blob deduplication.
    Max,
}

impl From<ModeArg> for Mode {
    fn from(mode: ModeArg) -> Self {
        match mode {
            ModeArg::Fast => Self::Fast,
            ModeArg::Balanced => Self::Balanced,
            ModeArg::Max => Self::Max,
        }
    }
}

/// Options of the commands that compress.
#[derive(Debug, Args)]
pub(crate) struct WriteArgs {
    /// Which transforms to try. Never changes what a restore produces.
    #[arg(long, value_enum)]
    pub(crate) mode: Option<ModeArg>,
    /// Compression and verification threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
}

impl WriteArgs {
    /// `defaults` of an entry point with these arguments applied.
    #[must_use]
    pub(crate) fn options(&self, defaults: EncodeOptions) -> EncodeOptions {
        EncodeOptions {
            mode: self.mode.map_or(defaults.mode, Mode::from),
            threads: self.threads,
        }
    }
}

/// Arguments of `encode`.
#[derive(Debug, Args)]
pub(crate) struct EncodeArgs {
    /// The file to encode.
    pub(crate) input: PathBuf,
    /// The container to write. An existing file is replaced once the new one is proven.
    pub(crate) output: PathBuf,
    #[command(flatten)]
    pub(crate) write: WriteArgs,
}

/// Arguments of `pack`.
#[derive(Debug, Args)]
pub(crate) struct PackArgs {
    /// The container to write. An existing file is replaced once the new one is proven.
    pub(crate) output: PathBuf,
    /// Files and directories. Members of a directory are named relative to it.
    #[arg(required = true)]
    pub(crate) inputs: Vec<PathBuf>,
    /// A file among the inputs, or under an input directory, to add first and to store every other
    /// file against: tensors that pair with its tensors become delta blobs where that is smaller.
    /// Needs --mode max, the default of pack.
    #[arg(long, value_name = "FILE")]
    pub(crate) delta_base: Option<PathBuf>,
    #[command(flatten)]
    pub(crate) write: WriteArgs,
}

/// Arguments of `restore`.
#[derive(Debug, Args)]
pub(crate) struct RestoreArgs {
    /// The container.
    pub(crate) container: PathBuf,
    /// Where to write the only member.
    #[arg(required_unless_present = "all", conflicts_with = "all")]
    pub(crate) output: Option<PathBuf>,
    /// Restore every member under this directory, which is created if needed.
    #[arg(long, value_name = "DIR")]
    pub(crate) all: Option<PathBuf>,
    /// Decode threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// The budget of commands that write what a container holds to disk.
#[derive(Debug, Args)]
pub(crate) struct BudgetArgs {
    /// Refuse the container, before reading any data, if its members add up to more than this many
    /// bytes. There is no limit by default: set one for a file you do not trust.
    #[arg(long, value_name = "BYTES")]
    pub(crate) max_total_decoded: Option<u64>,
}

/// Arguments of `extract`.
#[derive(Debug, Args)]
pub(crate) struct ExtractArgs {
    /// The container.
    pub(crate) container: PathBuf,
    /// Name of the member.
    pub(crate) member: String,
    /// Where to write it.
    pub(crate) output: PathBuf,
    /// Decode threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// Arguments of `merge`.
#[derive(Debug, Args)]
pub(crate) struct MergeArgs {
    /// The container to write. An existing file is replaced once the new one is proven.
    pub(crate) output: PathBuf,
    /// The containers to copy from, in order. A member name may appear once.
    #[arg(required = true)]
    pub(crate) inputs: Vec<PathBuf>,
    /// Verification threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
}

/// Arguments of `inspect`.
#[derive(Debug, Args)]
pub(crate) struct InspectArgs {
    /// The container.
    pub(crate) container: PathBuf,
    /// Print one JSON document instead of a table; the schema is in `src/MODULE_README.md`.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Arguments of `verify`.
#[derive(Debug, Args)]
pub(crate) struct VerifyArgs {
    /// The container.
    pub(crate) container: PathBuf,
    /// Decode threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
}

/// The notice and link the `WinFsp` license (FLOSS exception) requires in the user interface of a
/// program that uses its DLL.
#[cfg(windows)]
const WINFSP_NOTICE: &str = "Windows mounts use WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos, \
https://github.com/winfsp/winfsp. Install it from https://winfsp.dev/rel/.";

/// Most dispatcher threads `--winfsp-threads` accepts. `WinFsp` starts them as a chain after reporting
/// success, so a count the system cannot create would fail only later, as a stopped volume.
#[cfg(windows)]
const MAX_WINFSP_THREADS: i64 = 64;

/// Arguments of `mount`.
#[cfg(any(target_os = "linux", windows))]
#[derive(Debug, Args)]
#[cfg_attr(windows, command(after_help = WINFSP_NOTICE))]
pub(crate) struct MountArgs {
    /// The container.
    pub(crate) container: PathBuf,
    /// An existing directory to show the members under.
    #[cfg(target_os = "linux")]
    pub(crate) mountpoint: PathBuf,
    /// A free drive letter (X:), or a directory that does not exist yet (its parent must); it is
    /// created for the mount and removed after.
    #[cfg(windows)]
    pub(crate) mountpoint: PathBuf,
    /// Decode threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    /// Threads that serve requests of the kernel; 0 counts as 1.
    #[cfg(target_os = "linux")]
    #[arg(long, default_value_t = 4)]
    pub(crate) fuse_threads: usize,
    /// Threads that serve requests of the file system driver, at most 64; 0 means one per logical CPU,
    /// from 4 to 16, and 1 counts as 2.
    #[cfg(windows)]
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(0..=MAX_WINFSP_THREADS))]
    pub(crate) winfsp_threads: u32,
    /// Decoded bytes kept in memory for reads smaller than a chunk. At least twice the largest chunk.
    #[arg(long, value_name = "BYTES", default_value_t = CacheOptions::default().max_bytes)]
    pub(crate) cache_size: u64,
    /// Windows decoded by one cache miss, the missed one included; 0 means the decode thread count.
    #[arg(long, value_name = "WINDOWS", default_value_t = 0)]
    pub(crate) readahead: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}
