//! File: tree/compress.rs
//! Purpose: `compress_tree`: every weight file of a tree becomes a per-file container beside it,
//! named after it with the extension replaced, and the original goes once the container is
//! committed and verified.
//!
//! Main responsibilities:
//! - plan: classify every plain file with `profile::classify_weight`, map each weight file to its
//!   container name, and refuse the whole run on any name collision before anything is written;
//! - run each file in the crash-safe order: verified commit of the container, re-check of the
//!   original, removal, directory sync;
//! - on a rerun, resolve a pair (container and original both present) by hashing the original
//!   against the member's BLAKE3 every time, then verifying the container, or replace a stale one;
//! - follow symbolic links only when asked, compressing the target's bytes into a container at the
//!   link's place and removing the link, never the target.
//!
//! Key structures: [`CompressTreeOptions`], the private `Source` (one file to compress) and `Plan`.
//!
//! Key functions: [`compress_tree`], and the crate-private `compress_tree_with` that takes the
//! fault seam.
//!
//! Implementation notes: at every point a crash leaves the original, the container, or both, never
//! neither: the original is removed only after the container's commit returned, which includes its
//! full decode against the source, and only if the original's length and modification time (and a
//! followed link's target) are unchanged since before that commit. A failed file does not stop the
//! run. Followed links run before plain files, so a link to a weight file of the same tree reads
//! its target before that target is compressed and removed.

use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::io::FileSource;
use crate::profile::classify_weight;
use crate::read::{Container, OpenOptions};
use crate::write::{EncodeOptions, EncodeReport, PackBuilder, encode_file};

use super::view::{FileStamp, SkipReason, ViewEntry, ViewOptions, inspect_file, scan_view};
use super::{
    Action, CONTAINER_EXTENSION, Fault, FileOutcome, FileStatus, Stage, TreeReport, TreeWarning,
    finish_report, hash_file, log_outcome, no_fault, remove_redundant,
};

/// How [`compress_tree`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompressTreeOptions {
    /// Keep every original beside its container. Default `false`: an original is removed once its
    /// container is committed and verified.
    pub keep: bool,
    /// Follow symbolic links to regular files: the target's bytes become a container at the link's
    /// place, named after the link, and the link (never its target) is removed. Default `false`:
    /// links are skipped and the report warns about them. Links to directories are never entered.
    pub follow_symlinks: bool,
    /// Plan only: report what would be done ([`FileStatus::Planned`]) and write nothing. Conflicts
    /// are still an error.
    pub dry_run: bool,
    /// How each container is written. Default [`EncodeOptions::for_pack`] ([`crate::Mode::Max`],
    /// every CPU): a per-file container shares nothing with its neighbours, so the strongest mode
    /// is the default; quantized GGUF needs `Balanced` or `Max`, as `Fast` stores it raw. The
    /// threads also verify a pair's container.
    pub encode: EncodeOptions,
    /// Budget on the decoded size of each container that is opened, in bytes, as
    /// [`OpenOptions::max_total_decoded`]. Default `u64::MAX`, no budget.
    pub max_total_decoded: u64,
}

impl Default for CompressTreeOptions {
    fn default() -> Self {
        Self {
            keep: false,
            follow_symlinks: false,
            dry_run: false,
            encode: EncodeOptions::for_pack(),
            max_total_decoded: u64::MAX,
        }
    }
}

/// One weight file to compress.
#[derive(Debug)]
struct Source {
    /// The file, or the followed link, that the container replaces.
    path: PathBuf,
    /// The file whose bytes are compressed: `path` itself, or a followed link's resolved target.
    read_from: PathBuf,
    /// For a followed link, where it pointed when it was planned.
    link: Option<PathBuf>,
    /// The member name: the file name of `path`.
    name: String,
    /// The container: `path` with its extension replaced.
    target: PathBuf,
    /// Whether a container is already at `target`.
    action: SourceAction,
}

/// What a source needs, decided by what is at its container path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceAction {
    /// Nothing is there: compress.
    Compress,
    /// A container of this original is there: resolve the pair.
    CheckPair,
}

impl SourceAction {
    /// The public name of the action, for a dry run.
    fn public(self) -> Action {
        match self {
            Self::Compress => Action::Compress,
            Self::CheckPair => Action::CheckPair,
        }
    }
}

/// What the plan found: the files to compress and the outcomes already decided.
#[derive(Debug, Default)]
struct Plan {
    sources: Vec<Source>,
    outcomes: Vec<FileOutcome>,
    symlinks_skipped: usize,
}

/// Compresses every weight file under `root` (a directory, or a single file) into a container of
/// one member beside it: `dir/name.ext` becomes `dir/name.ziptensors` holding the member
/// `name.ext`, a name without extension gets `.ziptensors` appended. Which files are weight files
/// is decided by content alone (`profile::classify_weight`); every other file is left untouched.
///
/// The whole run is planned first. A collision refuses it before anything is written: two weight
/// files mapping to one container name (compared ignoring case, as on Windows and macOS), a
/// container name taken by anything that is not a container of exactly that original, or an
/// original whose name another container already restores to. Then each file runs on its own: a
/// failure is reported as [`FileStatus::Failed`] and the run goes on. A rerun is idempotent:
/// where container and original both exist, the original is hashed every time and compared with
/// the member's stated BLAKE3 and length; equal, the container is verified and the original
/// removed (kept under `keep`); different, the container is stale and replaced; a container that
/// does not verify fails the file and nothing is removed.
///
/// # Errors
/// [`Error::TreeConflict`] listing every collision, nothing written. [`Error::Io`] if the root or
/// a directory cannot be walked. Everything that goes wrong with one file is in its outcome.
pub fn compress_tree(root: &Path, options: &CompressTreeOptions) -> Result<TreeReport, Error> {
    compress_tree_with(root, options, no_fault)
}

/// [`compress_tree`] with the fault seam of the crash-order tests; production passes
/// [`no_fault`] only.
pub(crate) fn compress_tree_with(
    root: &Path,
    options: &CompressTreeOptions,
    fault: Fault,
) -> Result<TreeReport, Error> {
    tracing::info!(
        root = %root.display(),
        keep = options.keep,
        follow_symlinks = options.follow_symlinks,
        dry_run = options.dry_run,
        mode = ?options.encode.mode,
        "compressing tree"
    );
    let Plan { mut sources, mut outcomes, symlinks_skipped } = plan(root, options)?;
    let mut warnings = Vec::new();
    if symlinks_skipped > 0 {
        let warning = TreeWarning::SymlinksSkipped { count: symlinks_skipped };
        tracing::warn!(%warning, "symbolic links skipped");
        warnings.push(warning);
    }
    // Links first, see the file header; then in path order.
    sources.sort_by(|a, b| (a.link.is_none(), &a.path).cmp(&(b.link.is_none(), &b.path)));
    for source in sources {
        let outcome = if options.dry_run {
            let mut outcome = FileOutcome::new(source.path, FileStatus::Planned(source.action.public()));
            outcome.counterpart = Some(source.target);
            outcome
        } else {
            run_source(&source, options, fault)
        };
        log_outcome(&outcome);
        outcomes.push(outcome);
    }
    let report = TreeReport { root: root.to_path_buf(), dry_run: options.dry_run, files: outcomes, warnings };
    Ok(finish_report(report, "compress"))
}

/// Scans the tree, classifies every file and decides the action of every weight file.
///
/// # Errors
/// [`Error::TreeConflict`] for any collision, and the errors of [`scan_view`].
fn plan(root: &Path, options: &CompressTreeOptions) -> Result<Plan, Error> {
    let view_options = ViewOptions { max_total_decoded: options.max_total_decoded };
    let view = scan_view(root, &view_options)?;
    // Which container restores to which path, to find an original another container claims.
    let mut claims: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    for entry in &view.entries {
        if let ViewEntry::PerFile(per_file) | ViewEntry::Shadowed(per_file) = entry {
            claims.entry(per_file.presented.clone()).or_default().push(per_file.container.clone());
        }
    }
    let mut plan = Plan::default();
    let mut containers = Vec::new();
    for entry in view.entries {
        match entry {
            ViewEntry::Directory { .. } => {}
            ViewEntry::Plain { path, .. } => match classify(&path) {
                Ok(true) => plan_source(&mut plan, path.clone(), path, None),
                Ok(false) => plan.outcomes.push(FileOutcome::new(path, FileStatus::NotWeight)),
                Err(error) => plan.outcomes.push(FileOutcome::new(path, FileStatus::Failed(error))),
            },
            ViewEntry::Skipped { path, reason: SkipReason::Symlink, .. } if options.follow_symlinks => {
                follow(&mut plan, path);
            }
            ViewEntry::Skipped { path, reason, .. } => {
                if reason == SkipReason::Symlink {
                    plan.symlinks_skipped += 1;
                }
                plan.outcomes.push(FileOutcome::new(path, FileStatus::Skipped(reason)));
            }
            ViewEntry::PerFile(_)
            | ViewEntry::Shadowed(_)
            | ViewEntry::Foreign { .. }
            | ViewEntry::Broken { .. } => containers.push(entry),
        }
    }
    let conflicts = decide_actions(&mut plan, &claims, view_options);
    if !conflicts.is_empty() {
        tracing::error!(root = %root.display(), count = conflicts.len(), ?conflicts, "tree refused: name conflicts");
        return Err(Error::TreeConflict { root: root.to_path_buf(), conflicts });
    }
    report_containers(&mut plan, containers);
    tracing::info!(
        root = %root.display(),
        weight_files = plan.sources.len(),
        other = plan.outcomes.len(),
        "tree compression planned"
    );
    Ok(plan)
}

/// Decides the action of every source from what is at its container path, and returns the
/// conflicts, described. A source whose container path cannot be looked at moves to the outcomes
/// as failed.
fn decide_actions(
    plan: &mut Plan,
    claims: &HashMap<PathBuf, Vec<PathBuf>>,
    view_options: ViewOptions,
) -> Vec<String> {
    let mut conflicts = Vec::new();
    let mut by_folded_target: HashMap<String, PathBuf> = HashMap::new();
    let mut failed = Vec::new();
    for (index, source) in plan.sources.iter_mut().enumerate() {
        // Two targets equal ignoring case are one file on Windows and macOS: the second commit
        // would replace the first container while its original is already gone.
        let folded = source.target.to_string_lossy().to_lowercase();
        if let Some(first) = by_folded_target.insert(folded, source.path.clone()) {
            conflicts.push(format!(
                "{} and {} both map to the container {}",
                first.display(),
                source.path.display(),
                source.target.display()
            ));
        }
        if let Some(claimants) = claims.get(&source.path) {
            for container in claimants.iter().filter(|container| **container != source.target) {
                conflicts.push(format!(
                    "{} is already what the container {} restores to",
                    source.path.display(),
                    container.display()
                ));
            }
        }
        match occupant(&source.target, &source.name, view_options) {
            Ok(Occupant::Free) => source.action = SourceAction::Compress,
            Ok(Occupant::Ours) => source.action = SourceAction::CheckPair,
            Ok(Occupant::Other(what)) => conflicts.push(format!(
                "the container name {} of {} is taken by {what}",
                source.target.display(),
                source.path.display()
            )),
            Err(error) => failed.push((index, error)),
        }
    }
    // Removed back to front, so the indices still hold.
    for (index, error) in failed.into_iter().rev() {
        let source = plan.sources.remove(index);
        let mut outcome = FileOutcome::new(source.path, FileStatus::Failed(error));
        outcome.counterpart = Some(source.target);
        plan.outcomes.push(outcome);
    }
    conflicts
}

/// Adds the outcome of every container entry that is not some source's container: a per-file
/// container whose original is gone, a foreign container, a broken one.
fn report_containers(plan: &mut Plan, containers: Vec<ViewEntry>) {
    for entry in containers {
        // A container at some original's target is that original's business.
        if plan.sources.iter().any(|source| source.target == entry.path()) {
            continue;
        }
        let outcome = match entry {
            ViewEntry::PerFile(per_file) | ViewEntry::Shadowed(per_file) => {
                let mut outcome =
                    FileOutcome::new(per_file.container, FileStatus::AlreadyCompressed);
                outcome.counterpart = Some(per_file.presented);
                outcome.container_len = Some(per_file.stamp.len);
                outcome
            }
            ViewEntry::Foreign { path, reason, .. } => {
                FileOutcome::new(path, FileStatus::Foreign(reason))
            }
            ViewEntry::Broken { path, error, .. } => {
                FileOutcome::new(path, FileStatus::Failed(error))
            }
            ViewEntry::Directory { path, .. }
            | ViewEntry::Plain { path, .. }
            | ViewEntry::Skipped { path, .. } => {
                // `plan` sets aside container entries only; nothing to report for these.
                tracing::debug!(path = %path.display(), "not a container entry");
                continue;
            }
        };
        plan.outcomes.push(outcome);
    }
}

/// Whether the regular file at `path` is a weight file.
///
/// # Errors
/// [`Error::Io`] if it cannot be opened or read.
fn classify(path: &Path) -> Result<bool, Error> {
    let source = FileSource::open(path)?;
    let kind = classify_weight(&source)?;
    tracing::debug!(path = %path.display(), ?kind, "classified");
    Ok(kind.is_some())
}

/// Adds the weight file at `path`, read from `read_from`, as a source; its action is decided once
/// every source is known. A name that is not UTF-8 cannot be a member name and is skipped.
fn plan_source(plan: &mut Plan, path: PathBuf, read_from: PathBuf, link: Option<PathBuf>) {
    let Some(name) = path.file_name().and_then(|name| name.to_str()).map(str::to_owned) else {
        plan.outcomes.push(FileOutcome::new(path, FileStatus::Skipped(SkipReason::NonUtf8)));
        return;
    };
    let target = path.with_extension(CONTAINER_EXTENSION);
    plan.sources.push(Source { path, read_from, link, name, target, action: SourceAction::Compress });
}

/// Plans a followed link: a link to a regular weight file becomes a source read from its resolved
/// target; a link to a directory, to nothing, or to a special file is skipped; a link to another
/// file is not a weight file. A target outside the tree is fine, since it is only read.
fn follow(plan: &mut Plan, link: PathBuf) {
    let skipped = |link, reason| FileOutcome::new(link, FileStatus::Skipped(reason));
    let meta = match fs::metadata(&link) {
        Ok(meta) => meta,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            plan.outcomes.push(skipped(link, SkipReason::DanglingSymlink));
            return;
        }
        Err(source) => {
            let error = Error::Io { op: "metadata", path: link.clone(), source };
            plan.outcomes.push(FileOutcome::new(link, FileStatus::Failed(error)));
            return;
        }
    };
    if meta.is_dir() {
        plan.outcomes.push(skipped(link, SkipReason::SymlinkToDirectory));
        return;
    }
    if !meta.is_file() {
        plan.outcomes.push(skipped(link, SkipReason::Special));
        return;
    }
    let resolved = fs::canonicalize(&link)
        .and_then(|target| fs::read_link(&link).map(|pointed| (target, pointed)))
        .map_err(|source| Error::Io { op: "resolve path", path: link.clone(), source });
    let (target, pointed) = match resolved {
        Ok(resolved) => resolved,
        Err(error) => {
            plan.outcomes.push(FileOutcome::new(link, FileStatus::Failed(error)));
            return;
        }
    };
    match classify(&target) {
        Ok(true) => plan_source(plan, link, target, Some(pointed)),
        Ok(false) => plan.outcomes.push(FileOutcome::new(link, FileStatus::NotWeight)),
        Err(error) => plan.outcomes.push(FileOutcome::new(link, FileStatus::Failed(error))),
    }
}

/// What is at a source's container path.
#[derive(Debug)]
enum Occupant {
    /// Nothing.
    Free,
    /// A container whose one member is named after the source, or a file that starts with the
    /// container magic and does not open: a pair to resolve, where a broken container fails the
    /// file and removes nothing.
    Ours,
    /// Anything else, described for the conflict message.
    Other(String),
}

/// Looks at the container path `target` of the original named `name`. Asked of the file system,
/// so a file whose name differs only in case is found where the file system treats it as the same.
///
/// # Errors
/// [`Error::Io`] with op `"metadata"` if the path cannot be looked at.
fn occupant(target: &Path, name: &str, options: ViewOptions) -> Result<Occupant, Error> {
    let meta = match fs::symlink_metadata(target) {
        Ok(meta) => meta,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Occupant::Free),
        Err(source) => return Err(Error::Io { op: "metadata", path: target.to_path_buf(), source }),
    };
    if !meta.is_file() {
        return Ok(Occupant::Other("something that is not a regular file".to_owned()));
    }
    let rel = target.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(match inspect_file(target.to_path_buf(), rel, options) {
        ViewEntry::PerFile(per_file) | ViewEntry::Shadowed(per_file) if per_file.member_name == name => {
            Occupant::Ours
        }
        ViewEntry::PerFile(per_file) | ViewEntry::Shadowed(per_file) => {
            Occupant::Other(format!("a container of {:?}", per_file.member_name))
        }
        // An unreadable container is resolved as a pair, which fails without removing anything;
        // an unreadable file that is no container fails the same way when it is opened.
        ViewEntry::Broken { .. } => Occupant::Ours,
        ViewEntry::Foreign { .. } => Occupant::Other("a container that is not a per-file container".to_owned()),
        ViewEntry::Plain { .. } => Occupant::Other("a file that is not a container".to_owned()),
        ViewEntry::Skipped { .. } | ViewEntry::Directory { .. } => {
            Occupant::Other("a temporary or partial file".to_owned())
        }
    })
}

/// What a source looked like before its bytes were read, compared again before it is removed.
#[derive(Debug, PartialEq, Eq)]
struct SourceState {
    /// Stamp of the file whose bytes are read.
    stamp: FileStamp,
    /// For a followed link, where it points.
    link: Option<PathBuf>,
}

impl SourceState {
    /// The current state of `source`.
    ///
    /// # Errors
    /// [`Error::Io`] if the file or link cannot be looked at.
    fn of(source: &Source) -> Result<Self, Error> {
        let link = match &source.link {
            Some(_) => Some(fs::read_link(&source.path).map_err(|error| Error::Io {
                op: "read link",
                path: source.path.clone(),
                source: error,
            })?),
            None => None,
        };
        Ok(Self { stamp: FileStamp::of(&source.read_from)?, link })
    }
}

/// Runs one source and turns every failure into its outcome.
fn run_source(source: &Source, options: &CompressTreeOptions, fault: Fault) -> FileOutcome {
    let mut outcome =
        FileOutcome::new(source.path.clone(), FileStatus::Planned(source.action.public()));
    outcome.counterpart = Some(source.target.clone());
    let result = match source.action {
        SourceAction::Compress => {
            compress_new(source, options, fault, &mut outcome, FileStatus::Compressed)
        }
        SourceAction::CheckPair => resolve_pair(source, options, fault, &mut outcome),
    };
    outcome.status = match result {
        Ok(status) => status,
        Err(error) => FileStatus::Failed(error),
    };
    outcome
}

/// Writes the container of `source`, then removes the original unless `keep`: the order of the
/// file header. `done` is the status on success.
///
/// # Errors
/// Any error of the encode (the container is then not committed, or for
/// [`Error::CommittedNotDurable`] committed but the original is kept), [`Error::InputChanged`] if
/// the original changed before its removal, and the errors of the removal.
fn compress_new(
    source: &Source,
    options: &CompressTreeOptions,
    fault: Fault,
    outcome: &mut FileOutcome,
    done: FileStatus,
) -> Result<FileStatus, Error> {
    let before = SourceState::of(source)?;
    outcome.original_len = Some(before.stamp.len);
    let report = encode(source, &options.encode)?;
    outcome.container_len = Some(report.output_len);
    fault(Stage::Committed, &source.target)?;
    if options.keep {
        return Ok(done);
    }
    remove_original(source, &before, fault, outcome)?;
    Ok(done)
}

/// Encodes the bytes of `source` into its container.
///
/// # Errors
/// Every error of [`PackBuilder`] and [`encode_file`].
fn encode(source: &Source, encode: &EncodeOptions) -> Result<EncodeReport, Error> {
    if source.link.is_some() {
        // The member is named after the link, which is what the tree calls the file; the bytes
        // come from the resolved target, which `add_file` accepts since it is no link.
        let mut builder = PackBuilder::new(&source.target, encode)?;
        builder.add_file(&source.read_from, &source.name)?;
        builder.finish()
    } else {
        encode_file(&source.path, &source.target, encode)
    }
}

/// Removes the original of `source` if it is still as `before` described it.
///
/// # Errors
/// [`Error::InputChanged`] naming the original if it changed, which keeps it, and the errors of
/// [`remove_redundant`].
fn remove_original(
    source: &Source,
    before: &SourceState,
    fault: Fault,
    outcome: &mut FileOutcome,
) -> Result<(), Error> {
    if SourceState::of(source)? != *before {
        tracing::error!(path = %source.path.display(), "the original changed after it was compressed; it is kept");
        return Err(Error::InputChanged { path: source.path.clone() });
    }
    remove_redundant(&source.path, outcome, fault)
}

/// Resolves a source whose container already exists: hash the original against the member, then
/// verify the container and remove the original (unless `keep`), or replace a stale container.
///
/// # Errors
/// Any error of opening or verifying the container (nothing is removed then),
/// [`Error::MemberNotFound`] if the container no longer holds a member of the source's name,
/// [`Error::InputChanged`] if the original or the container changed meanwhile, and the errors of
/// [`compress_new`] for a stale container.
fn resolve_pair(
    source: &Source,
    options: &CompressTreeOptions,
    fault: Fault,
    outcome: &mut FileOutcome,
) -> Result<FileStatus, Error> {
    let before = SourceState::of(source)?;
    outcome.original_len = Some(before.stamp.len);
    let container_stamp = FileStamp::of(&source.target)?;
    outcome.container_len = Some(container_stamp.len);
    let open = OpenOptions {
        threads: options.encode.threads,
        max_total_decoded: options.max_total_decoded,
        ..OpenOptions::default()
    };
    let container = Container::open(&source.target, &open)?;
    let member = container.single()?;
    if member.name()? != source.name {
        return Err(Error::MemberNotFound { name: source.name.clone() });
    }
    let (len, hash) = hash_file(&source.read_from)?;
    if len != member.len() || hash != member.blake3() {
        tracing::warn!(
            path = %source.path.display(),
            container = %source.target.display(),
            "the container states other content than the original: replacing it"
        );
        drop(member);
        drop(container);
        return compress_new(source, options, fault, outcome, FileStatus::Recompressed);
    }
    member.verify()?;
    drop(member);
    drop(container);
    if options.keep {
        return Ok(FileStatus::PairVerified);
    }
    if FileStamp::of(&source.target)? != container_stamp {
        tracing::error!(container = %source.target.display(), "the container changed while it was verified; the original is kept");
        return Err(Error::InputChanged { path: source.target.clone() });
    }
    remove_original(source, &before, fault, outcome)?;
    Ok(FileStatus::PairResolved)
}
