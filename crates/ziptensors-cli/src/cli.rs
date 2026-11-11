//! File: cli.rs
//! Purpose: the command line grammar: subcommands, their arguments, and what they translate to.
//!
//! Key structures: [`Cli`], [`Command`], [`WriteArgs`], [`ModeArg`], [`BudgetArgs`], [`TreeArgs`],
//! [`NewFile`], one `*Args` per subcommand.
//!
//! Implementation notes: this file parses and nothing else. `Cli::parse_checked` adds the checks
//! across arguments clap cannot state: `--delta-base` with a mode other than `max` (`pack`, `edit`),
//! `compress`/`decompress` of a directory without `-r`, and `mount` of a directory without `--overlay`
//! or of a file with it, all usage errors (exit 2); it also
//! restores the command line order of `edit --add` and `--add-as`, which clap keeps per flag. The
//! conflicts of `-F`, `-R` and `-C` and the operation `edit` needs are stated to clap itself.
//! `--threads 0`, the default, means one thread per logical CPU, as in the core. A missing `--mode`
//! keeps the default of the entry point: `balanced` for `encode`, `max` for everything else. `mount`
//! exists on Linux (FUSE) and Windows (`WinFsp`), the systems with a backend; its thread flag is per
//! backend (`--fuse-threads`, `--winfsp-threads`), its pool limits need `--overlay`, and on Windows its
//! help carries the notice the `WinFsp` license requires.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use clap::error::ErrorKind;
use clap::parser::ArgMatches;
use clap::{ArgGroup, Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
#[cfg(any(target_os = "linux", windows))]
use clap::builder::RangedU64ValueParser;
#[cfg(any(target_os = "linux", windows))]
use ziptensors_core::CacheOptions;
use ziptensors_core::{EncodeOptions, Mode};
#[cfg(any(target_os = "linux", windows))]
use ziptensors_mount::OverlayOptions;

/// Lossless tensor container: encode, pack, edit, restore, inspect, verify.
#[derive(Debug, Parser)]
#[command(name = "ziptensors", version)]
pub(crate) struct Cli {
    /// What to do.
    #[command(subcommand)]
    pub(crate) command: Command,
}

impl Cli {
    /// Parses the command line and checks what clap cannot express, exiting with clap's usage error
    /// (code 2) on a conflict:
    /// - `pack` or `edit` with `--delta-base` and a `--mode` other than `max`, the only mode in which
    ///   the writer stores delta blobs: a flag that would be ignored for the hours of a run is refused
    ///   rather than warned about;
    /// - `compress` or `decompress` of a directory without `-r`. A path that cannot be looked at is
    ///   left to the command, which reports why.
    ///
    /// Also fills [`EditArgs::new_files`] in command line order.
    #[must_use]
    pub(crate) fn parse_checked() -> Self {
        let matches = Self::command().get_matches();
        let mut cli = Self::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
        let delta_mode = if let Command::Pack(pack) = &cli.command {
            pack.delta_base.as_ref().and(pack.write.mode)
        } else if let Command::Edit(edit) = &cli.command {
            edit.delta_base.as_ref().and(edit.write.mode)
        } else {
            None
        };
        match delta_mode {
            None | Some(ModeArg::Max) => {}
            Some(ModeArg::Fast | ModeArg::Balanced) => usage_error(
                ErrorKind::ArgumentConflict,
                "--delta-base needs --mode max: no other mode stores delta blobs",
            ),
        }
        let tree = if let Command::Compress(args) = &cli.command {
            Some(&args.tree)
        } else if let Command::Decompress(args) = &cli.command {
            Some(&args.tree)
        } else {
            None
        };
        if let Some(tree) = tree.filter(|tree| !tree.recursive)
            && let Some(dir) = tree.paths.iter().find(|path| is_dir(path))
        {
            usage_error(
                ErrorKind::InvalidValue,
                &format!("{} is a directory: pass -r to work through a directory tree", dir.display()),
            );
        }
        #[cfg(any(target_os = "linux", windows))]
        if let Command::Mount(mount) = &cli.command {
            check_mount_source(mount);
        }
        if let Command::Edit(edit) = &mut cli.command
            && let Some(("edit", sub)) = matches.subcommand()
        {
            edit.new_files = new_files_in_order(edit, sub);
        }
        cli
    }
}

/// Exits with clap's usage error of `kind` (code 2) and `message`.
fn usage_error(kind: ErrorKind, message: &str) -> ! {
    Cli::command().error(kind, message).exit()
}

/// Whether `path` is a directory, not following a link: the tree walk does not enter a root that is
/// a link to a directory either.
fn is_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| meta.is_dir())
}

/// Exits with a usage error if the source of `mount` is a directory without `--overlay`, or an
/// existing path that is not a directory with it. Links are followed, as the overlay resolves its root;
/// a path that cannot be read is left to the command, which names the OS error.
#[cfg(any(target_os = "linux", windows))]
fn check_mount_source(mount: &MountArgs) {
    let Ok(meta) = fs::metadata(&mount.source) else { return };
    let shown = mount.source.display();
    if mount.overlay && !meta.is_dir() {
        usage_error(
            ErrorKind::InvalidValue,
            &format!("{shown} is not a directory: --overlay shows a directory tree; mount a container without it"),
        );
    }
    if !mount.overlay && meta.is_dir() {
        usage_error(
            ErrorKind::InvalidValue,
            &format!("{shown} is a directory: pass --overlay to show a tree of compressed files"),
        );
    }
}

/// The files `edit` adds, `--add` and `--add-as` merged in the order they were given, which is the
/// order of the new members at the end of the result. Clap keeps the values per flag, with the
/// position of each value on the command line.
fn new_files_in_order(edit: &EditArgs, matches: &ArgMatches) -> Vec<NewFile> {
    let positions = |id: &str| matches.indices_of(id).map(Iterator::collect::<Vec<_>>).unwrap_or_default();
    let plain = positions("add").into_iter().zip(&edit.add).map(|(at, path)| {
        (at, NewFile { name: None, path: path.clone() })
    });
    // `--add-as` takes two values: each pair is placed at the position of its NAME.
    let named_positions = positions("add_as").into_iter().step_by(2);
    // Clap's `num_args = 2` leaves no remainder.
    let (pairs, _) = edit.add_as.as_chunks::<2>();
    let named = named_positions.zip(pairs).map(|(at, [name, path])| {
        (at, NewFile { name: Some(name.clone()), path: PathBuf::from(path) })
    });
    let mut files: Vec<_> = plain.chain(named).collect();
    files.sort_by_key(|(at, _)| *at);
    files.into_iter().map(|(_, file)| file).collect()
}

/// The subcommands.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Encode one file as a container of one member.
    Encode(EncodeArgs),
    /// Pack files and directory trees into one container.
    Pack(PackArgs),
    /// Add, remove or replace members of a container.
    Edit(EditArgs),
    /// Drop the dead data of earlier edits from a container, or with -C re-encode every member.
    Compact(CompactArgs),
    /// Shorten a container whose end was torn (an interrupted edit) to its latest valid generation.
    Recover(RecoverArgs),
    /// Restore the only member of a container, or with --all every member.
    Restore(RestoreArgs),
    /// Restore one named member of a container, to a file or to standard output.
    Extract(ExtractArgs),
    /// Copy every member of several containers into a new one, with no recompression.
    Merge(MergeArgs),
    /// Print what a container holds.
    Inspect(InspectArgs),
    /// Decode every member and compare it with its stated BLAKE3.
    Verify(VerifyArgs),
    /// Replace every weight file under a path by a container of one member beside it.
    Compress(CompressArgs),
    /// Restore every container of one member under a path beside it, and remove the container.
    Decompress(DecompressArgs),
    /// Show every member, or with --overlay a compressed directory tree, as read-only files under a
    /// directory, until Ctrl-C.
    #[cfg(target_os = "linux")]
    Mount(MountArgs),
    /// Show every member, or with --overlay a compressed directory tree, as read-only files on a drive
    /// letter or a new directory, until Ctrl-C.
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

/// The decode budget of every command that opens a container.
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
    /// Where to write it, or `-` for standard output (the result line then goes to standard error).
    /// A file is written only once its BLAKE3 is proven. Standard output receives the bytes as they
    /// are decoded, before the hash of the whole member is known: a mismatch is reported after the
    /// fact, with exit code 1, and whatever read the output must discard it.
    pub(crate) output: PathBuf,
    /// Decode threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

impl ExtractArgs {
    /// Whether the output is standard output, spelled `-`.
    #[must_use]
    pub(crate) fn to_stdout(&self) -> bool {
        self.output.as_os_str() == "-"
    }
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
    /// The budget of each input.
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// Arguments of `inspect`.
#[derive(Debug, Args)]
pub(crate) struct InspectArgs {
    /// The container.
    pub(crate) container: PathBuf,
    /// Print one JSON document instead of a table; the schema is in `docs/cli.md` of the source
    /// repository.
    #[arg(long)]
    pub(crate) json: bool,
    /// Also list the tensors of every member: name, offset, length, dtype and shape.
    #[arg(long)]
    pub(crate) tensors: bool,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// Arguments of `verify`.
#[derive(Debug, Args)]
pub(crate) struct VerifyArgs {
    /// The container.
    pub(crate) container: PathBuf,
    /// Decode threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// Help of `-R`/`--risky`, shared by `edit` and `compact`. It must say loudly what the strategy
/// risks: the core cannot undo an interrupted rewrite in place.
const RISKY_HELP: &str = "RISKY: rewrite the container inside its own file, with no temporary copy \
and no extra disk space. AN INTERRUPTION (crash, power loss, full disk, kill) DAMAGES THE CONTAINER, \
and `ziptensors recover` MAY NOT BE ABLE TO REPAIR IT: keep a backup or the original files. Close \
every program that reads the file (a mount included) first: it would read wrong bytes.";

/// Help of `-C`/`--compact`, shared by `edit` and `compact`.
const REPACK_HELP: &str = "Re-encode every kept member with --mode, deduplication and delta pairing \
redone over the whole result. Slow; not with -F or -R.";

/// Arguments of `edit`.
#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("operation").required(true).multiple(true).args(["add", "add_as", "remove", "replace"])
))]
pub(crate) struct EditArgs {
    /// The container to change.
    pub(crate) container: PathBuf,
    /// Add a file as a new member named after the file, at the end. May be repeated.
    #[arg(long, value_name = "PATH")]
    pub(crate) add: Vec<PathBuf>,
    /// Add a file as a new member NAME, at the end. May be repeated; new members keep the order of
    /// --add and --add-as on the command line.
    #[arg(long, num_args = 2, value_names = ["NAME", "PATH"])]
    pub(crate) add_as: Vec<OsString>,
    /// Remove a member. May be repeated.
    #[arg(long, value_name = "NAME")]
    pub(crate) remove: Vec<String>,
    /// Replace the content of member NAME with a file, in the member's place. May be repeated.
    #[arg(long, num_args = 2, value_names = ["NAME", "PATH"])]
    pub(crate) replace: Vec<OsString>,
    /// A member to store every added or replacing file against, as delta blobs where that is
    /// smaller: a kept member, or a file of this edit given before the others. Needs --mode max.
    #[arg(long, value_name = "MEMBER")]
    pub(crate) delta_base: Option<String>,
    /// Append only the new data, a new index and a new footer to the end of the file. Fast and needs
    /// space only for what is new; the old data stays in the file as dead bytes (`compact` drops
    /// them). An interruption leaves a file that `ziptensors recover` restores to its old content.
    #[arg(short = 'F', long, conflicts_with_all = ["risky", "compact"])]
    pub(crate) fast: bool,
    #[arg(short = 'R', long, help = RISKY_HELP, conflicts_with = "compact")]
    pub(crate) risky: bool,
    #[arg(short = 'C', long, help = REPACK_HELP)]
    pub(crate) compact: bool,
    #[command(flatten)]
    pub(crate) write: WriteArgs,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
    /// `--add` and `--add-as` in command line order, filled by [`Cli::parse_checked`].
    #[arg(skip)]
    pub(crate) new_files: Vec<NewFile>,
}

/// A file `edit` adds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NewFile {
    /// The member name of `--add-as`; `None` for `--add`, which names the member after the file.
    pub(crate) name: Option<OsString>,
    /// The file.
    pub(crate) path: PathBuf,
}

/// Arguments of `compact`.
#[derive(Debug, Args)]
pub(crate) struct CompactArgs {
    /// The container to compact.
    pub(crate) container: PathBuf,
    #[arg(short = 'R', long, help = RISKY_HELP, conflicts_with = "compact")]
    pub(crate) risky: bool,
    #[arg(short = 'C', long, help = REPACK_HELP)]
    pub(crate) compact: bool,
    /// Which transforms -C tries; without -C nothing is re-encoded and the mode does nothing.
    #[arg(long, value_enum)]
    pub(crate) mode: Option<ModeArg>,
    /// Compression and verification threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// Arguments of `recover`.
#[derive(Debug, Args)]
pub(crate) struct RecoverArgs {
    /// The container to shorten. A file that opens as it is is left alone.
    pub(crate) container: PathBuf,
    /// Report what would be cut and write nothing. Works on a file you cannot write.
    #[arg(long)]
    pub(crate) dry_run: bool,
    /// Skip decoding the generation found before cutting. Only for a file never edited with -R:
    /// after an interrupted rewrite in place an older generation may look valid while its data moved.
    #[arg(long)]
    pub(crate) no_verify: bool,
    /// Most candidate generations checked in full before giving up.
    #[arg(long, value_name = "N", default_value_t = ziptensors_core::recover::DEFAULT_MAX_CANDIDATES)]
    pub(crate) max_candidates: u32,
    /// Most distinct bytes read while searching backwards for a footer; no limit by default.
    #[arg(long, value_name = "BYTES")]
    pub(crate) max_scan_bytes: Option<u64>,
    /// Verification threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// What `compress` and `decompress` share: where to work, and how far.
#[derive(Debug, Args)]
pub(crate) struct TreeArgs {
    /// Files, or with -r directory trees. Files of the other kind are left untouched.
    #[arg(required = true)]
    pub(crate) paths: Vec<PathBuf>,
    /// Work through directories. Needed for a directory.
    #[arg(short = 'r', long)]
    pub(crate) recursive: bool,
    /// Keep the redundant copy: every original beside its container (compress), every container
    /// beside its restored file (decompress).
    #[arg(short = 'k', long)]
    pub(crate) keep: bool,
    /// Report what would be done and write nothing.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

/// Arguments of `compress`.
#[derive(Debug, Args)]
pub(crate) struct CompressArgs {
    #[command(flatten)]
    pub(crate) tree: TreeArgs,
    /// Compress the target of each link to a weight file into a container in place of the link.
    /// Breaks a Hugging Face cache, whose snapshots/ link to blobs/: compress a plain copy instead.
    #[arg(long)]
    pub(crate) follow_symlinks: bool,
    #[command(flatten)]
    pub(crate) write: WriteArgs,
    /// The budget of each container already there and opened for a check.
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
}

/// Arguments of `decompress`.
#[derive(Debug, Args)]
pub(crate) struct DecompressArgs {
    #[command(flatten)]
    pub(crate) tree: TreeArgs,
    /// Restore over a different file already at the restore path.
    #[arg(short = 'f', long)]
    pub(crate) force: bool,
    /// Decode threads; 0 means one per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,
    #[command(flatten)]
    pub(crate) budget: BudgetArgs,
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
    /// The container, or with --overlay a directory.
    #[arg(value_name = "CONTAINER|DIR")]
    pub(crate) source: PathBuf,
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
    /// Show a directory instead of a container: each file `compress -r` turned into a container
    /// appears as the original it holds, every other file as it is. Containers open on first read;
    /// the cache size is shared by the open ones, and the other flags apply to each.
    #[arg(long)]
    pub(crate) overlay: bool,
    /// With --overlay: containers kept open at once; the least recently read one is closed first.
    #[arg(
        long,
        value_name = "N",
        requires = "overlay",
        value_parser = RangedU64ValueParser::<usize>::new().range(1..),
        default_value_t = OverlayOptions::default().max_open_containers
    )]
    pub(crate) max_open_containers: usize,
    /// With --overlay: files that are not containers kept open at once.
    #[arg(
        long,
        value_name = "N",
        requires = "overlay",
        value_parser = RangedU64ValueParser::<usize>::new().range(1..),
        default_value_t = OverlayOptions::default().max_open_files
    )]
    pub(crate) max_open_files: usize,
}
