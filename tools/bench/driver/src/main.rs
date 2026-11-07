//! File: main.rs
//! Purpose: the measuring half of the acceptance benchmark: one process per measurement, one JSON
//! document on stdout, so that `tools/bench/` can evict caches before it and read peak memory after it.
//!
//! Main responsibilities:
//! - `load-raw`, `load-decode`, `load-many`: read every tensor of a member from the raw file, through
//!   `decode_tensor_into`, or through `read_many`;
//! - `random`: seeded random reads of one size, raw or through `read_at`, with per-read latencies;
//! - `tensors`: the tensor ranges of a member, which is the plan of `load-raw`;
//! - `seek-map`: disk contiguous runs per member, from the index alone;
//! - `encode-memory`: writes a one-member container through `PackBuilder` and reports the peak of Rust
//!   allocations of the write, the measurement of risk R3.
//!
//! Key structures: [`Cli`], [`Target`], [`Latency`], [`Layout`], [`Memory`].
//!
//! Key functions: [`load_decode`], [`load_many`], [`random`], [`member_runs`], [`encode_memory`].
//!
//! Implementation notes: this exists because the Python module has no `read_many` and no view of the
//! chunks. It uses only public API of `ziptensors-core`. Time is measured from before the container is
//! opened, as a loader pays that too. Memory comes in two forms. `rust_*` figures come from the counting
//! allocator of `counting.rs` and are what the gates compare with the bounds of `ARCHITECTURE.md`, which
//! are stated in Rust allocations: the peak of the measured phase above a baseline taken after the open
//! and after the destination is allocated, so it excludes the index and the destination as the read bound
//! does. `peak_rss_bytes` is `VmHWM` of this process, reported and never gated, as it also holds code,
//! zstd's C contexts and allocator retention. `load-raw` needs no container at run time, so its memory is
//! not inflated by the index.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]

mod counting;

use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write as _;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use rustix::fs::{Advice, fadvise};
use ziptensors_core::format::{FOOTER_LEN, Index, Limits, MemberRecord, parse_footer};
use ziptensors_core::{Container, EncodeOptions, Member, Mode, OpenOptions, PackBuilder, ReadRequest};

use counting::Counting;

/// Every Rust allocation of the process goes through this counter.
#[global_allocator]
static ALLOCATOR: Counting = Counting::new();

/// Arguments of the driver.
#[derive(Debug, Parser)]
#[command(about = "Timing driver of the ziptensors acceptance benchmark")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Which member of a container: a name, or none for the only member of a single-file container.
#[derive(Debug, clap::Args)]
struct MemberArgs {
    /// The container.
    #[arg(long)]
    container: PathBuf,
    /// The member. Without it the container must hold exactly one.
    #[arg(long)]
    member: Option<String>,
    /// Decode threads, 0 for one per logical CPU.
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

/// One measurement.
#[derive(Debug, Subcommand)]
enum Command {
    /// Writes `<offset> <len>` of every tensor to `--out`, in tensor order.
    Tensors {
        #[command(flatten)]
        member: MemberArgs,
        /// The plan file for `load-raw`.
        #[arg(long)]
        out: PathBuf,
    },
    /// Prints the disk runs of every member of a container.
    SeekMap {
        /// The container.
        #[arg(long)]
        container: PathBuf,
    },
    /// Reads the ranges of a plan from a raw file into one buffer sized for the largest of them.
    LoadRaw {
        /// The raw file.
        #[arg(long)]
        file: PathBuf,
        /// A plan written by `tensors`.
        #[arg(long)]
        plan: PathBuf,
    },
    /// Calls `decode_tensor_into` for each tensor in tensor order, into one buffer sized for the largest.
    LoadDecode(MemberArgs),
    /// Calls `read_many` once for all tensors, into one buffer of their total size.
    LoadMany(MemberArgs),
    /// Times seeded random reads.
    Random {
        /// Read the raw file instead of the container's member.
        #[arg(long)]
        raw: Option<PathBuf>,
        #[command(flatten)]
        member: MemberArgs,
        /// Bytes per read.
        #[arg(long)]
        size: usize,
        /// Reads per pass.
        #[arg(long)]
        count: usize,
        /// Seed of the offsets. The same seed gives the same offsets for the same length.
        #[arg(long)]
        seed: u64,
        /// Passes over the same offsets.
        #[arg(long, default_value_t = 1)]
        passes: usize,
        /// When the page cache is dropped.
        #[arg(long, value_enum)]
        cold: Cold,
    },
    /// Writes `--input` as a one-member container at `--output` and reports the Rust peak of the write.
    /// The output stays; the caller removes it.
    EncodeMemory {
        /// The file to encode. Its file name is the member name.
        #[arg(long)]
        input: PathBuf,
        /// The container to write. Its directory takes the temporary file of the commit.
        #[arg(long)]
        output: PathBuf,
        /// The writer mode.
        #[arg(long, value_enum)]
        mode: ModeArg,
        /// Compression threads, 0 for one per logical CPU.
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
}

/// The writer modes, as arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ModeArg {
    Fast,
    Balanced,
    Max,
}

impl ModeArg {
    /// The library's mode.
    fn mode(self) -> Mode {
        match self {
            Self::Fast => Mode::Fast,
            Self::Balanced => Mode::Balanced,
            Self::Max => Mode::Max,
        }
    }

    /// The name of the mode in the JSON record.
    fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Max => "max",
        }
    }
}

/// When a `random` run drops the page cache of the file it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Cold {
    /// Never: the caller evicted before the run, so only the first pass is cold.
    First,
    /// Before every read, outside the timed part, so that no read profits from an earlier one.
    Each,
}

/// `value` as a JSON string.
fn json_string(value: &str) -> String {
    let mut text = String::with_capacity(value.len() + 2);
    text.push('"');
    for character in value.chars() {
        match character {
            '"' => text.push_str("\\\""),
            '\\' => text.push_str("\\\\"),
            control if u32::from(control) < 0x20 => {
                // A write to a `String` cannot fail, and the trait forces a `Result` on it.
                write!(text, "\\u{:04x}", u32::from(control)).unwrap_or(());
            }
            other => text.push(other),
        }
    }
    text.push('"');
    text
}

/// Peak resident memory of this process in bytes, `VmHWM` of `/proc/self/status`.
///
/// # Errors
/// Returns an error if the file cannot be read or has no such line.
fn peak_rss() -> anyhow::Result<u64> {
    let status = fs::read_to_string("/proc/self/status").context("cannot read /proc/self/status")?;
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .ok_or_else(|| anyhow!("/proc/self/status has no VmHWM line"))?;
    let kib: u64 = line
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .with_context(|| format!("cannot parse VmHWM {line:?}"))?;
    kib.checked_mul(1024).ok_or_else(|| anyhow!("VmHWM overflows"))
}

/// The thread count a container or a writer will really use for `threads`, for the record and the
/// memory bounds. The library keeps its own count private; this is the same rule: `threads`, or one per
/// logical CPU for 0.
fn effective_threads(threads: usize) -> usize {
    if threads != 0 {
        return threads;
    }
    thread::available_parallelism().map_or(1, usize::from)
}

/// Opens a container for reading with the library defaults, which is what a loader gets.
fn open(args: &MemberArgs) -> anyhow::Result<Member> {
    let options = OpenOptions { threads: args.threads, ..OpenOptions::default() };
    let container = Container::open(&args.container, &options)
        .with_context(|| format!("cannot open {}", args.container.display()))?;
    match &args.member {
        Some(name) => container.member(name),
        None => container.single(),
    }
    .with_context(|| format!("no such member in {}", args.container.display()))
}

/// Rust allocation figures of one read measurement, from the counting allocator.
#[derive(Debug, Default)]
struct Memory {
    /// Live bytes the open of the container left: its in-memory index state. 0 without a container.
    index: usize,
    /// Peak of the measured phase above its baseline, which is taken after the open and after the
    /// destination and the request list are allocated. This is what the read bound limits.
    peak: usize,
    /// The largest `chunk_size` of the container, the "chunk" of the bound. 0 without a container.
    chunk: u32,
    /// `VmHWM` when the measured phase ended, before reading the chunk size could raise it.
    rss: u64,
}

impl Memory {
    /// The figures as JSON fields, `peak_rss_bytes` included.
    fn fields(&self) -> String {
        format!(
            "\"peak_rss_bytes\":{},\"rust_index_bytes\":{},\"rust_peak_bytes\":{},\"chunk_bytes\":{}",
            self.rss, self.index, self.peak, self.chunk
        )
    }
}

/// The JSON fields common to all load measurements.
fn load_fields(
    tensors: usize,
    bytes: u64,
    seconds: f64,
    destination: u64,
    memory: &Memory,
) -> String {
    format!(
        "\"release\":{},\"tensors\":{tensors},\"bytes\":{bytes},\"seconds\":{seconds:.6},\
         \"destination_bytes\":{destination},{}",
        !cfg!(debug_assertions),
        memory.fields()
    )
}

/// Opens a container like [`open`] and returns the live Rust bytes the open left with the member.
fn open_counted(args: &MemberArgs) -> anyhow::Result<(Member, usize)> {
    let before = ALLOCATOR.live();
    let member = open(args)?;
    Ok((member, ALLOCATOR.live().saturating_sub(before)))
}

/// `tensors`: writes the plan of `load-raw`.
fn tensors(args: &MemberArgs, out: &Path) -> anyhow::Result<()> {
    let member = open(args)?;
    let mut text = String::new();
    for tensor in member.tensors() {
        writeln!(text, "{} {}", tensor.offset, tensor.len).context("cannot format the plan")?;
    }
    fs::write(out, text).with_context(|| format!("cannot write {}", out.display()))?;
    println!("{{\"tensors\":{},\"len\":{}}}", member.tensors().len(), member.len());
    Ok(())
}

/// Parses a plan file into `(offset, len)` pairs.
fn parse_plan(text: &str) -> anyhow::Result<Vec<(u64, usize)>> {
    text.lines()
        .map(|line| {
            let (offset, len) =
                line.split_once(' ').ok_or_else(|| anyhow!("bad plan line {line:?}"))?;
            Ok((
                offset.parse().with_context(|| format!("bad plan line {line:?}"))?,
                len.parse().with_context(|| format!("bad plan line {line:?}"))?,
            ))
        })
        .collect()
}

/// `load-raw`: the baseline. `pread` is what `readinto` after a seek does.
fn load_raw(file: &Path, plan: &Path) -> anyhow::Result<()> {
    let text = fs::read_to_string(plan).with_context(|| format!("cannot read {}", plan.display()))?;
    let ranges = parse_plan(&text)?;
    let largest = ranges.iter().map(|(_, len)| *len).max().unwrap_or(0);
    let mut buffer = vec![0_u8; largest];
    let phase = ALLOCATOR.phase();
    let start = Instant::now();
    let source = File::open(file).with_context(|| format!("cannot open {}", file.display()))?;
    let mut bytes = 0_u64;
    for (offset, len) in &ranges {
        let dst = buffer.get_mut(..*len).ok_or_else(|| anyhow!("a plan range outgrew the buffer"))?;
        source
            .read_exact_at(dst, *offset)
            .with_context(|| format!("cannot read {} at {offset}", file.display()))?;
        bytes += u64::try_from(*len).context("a length overflows u64")?;
    }
    let seconds = start.elapsed().as_secs_f64();
    let memory = Memory { peak: phase.peak_above(), rss: peak_rss()?, ..Memory::default() };
    let destination = u64::try_from(largest).context("a length overflows u64")?;
    println!("{{{}}}", load_fields(ranges.len(), bytes, seconds, destination, &memory));
    Ok(())
}

/// `load-decode`: the candidate in tensor order.
fn load_decode(args: &MemberArgs) -> anyhow::Result<()> {
    let start = Instant::now();
    let (member, index) = open_counted(args)?;
    let open_seconds = start.elapsed().as_secs_f64();
    let tensors = member.tensors();
    let largest = tensors.iter().map(|tensor| tensor.len).max().unwrap_or(0);
    let mut buffer = vec![0_u8; usize::try_from(largest).context("a tensor exceeds memory")?];
    let phase = ALLOCATOR.phase();
    let mut bytes = 0_u64;
    for tensor in tensors {
        let len = usize::try_from(tensor.len).context("a tensor exceeds memory")?;
        let dst = buffer.get_mut(..len).ok_or_else(|| anyhow!("a tensor outgrew the buffer"))?;
        member
            .decode_tensor_into(&tensor.name, dst)
            .with_context(|| format!("cannot decode {}", tensor.name))?;
        bytes += tensor.len;
    }
    let seconds = start.elapsed().as_secs_f64();
    let peak = phase.peak_above();
    // After the timed part, so that reading the index again cannot warm the cache of the run.
    let rss = peak_rss()?;
    let memory = Memory { index, peak, chunk: largest_chunk(&args.container)?, rss };
    println!(
        "{{{},\"open_seconds\":{open_seconds:.6},\"threads\":{}}}",
        load_fields(tensors.len(), bytes, seconds, largest, &memory),
        effective_threads(args.threads)
    );
    Ok(())
}

/// `load-many`: the candidate with the disk moving one way.
fn load_many(args: &MemberArgs) -> anyhow::Result<()> {
    let start = Instant::now();
    let (member, index) = open_counted(args)?;
    let open_seconds = start.elapsed().as_secs_f64();
    let tensors = member.tensors();
    let total: u64 = tensors.iter().map(|tensor| tensor.len).sum();
    let mut buffer = vec![0_u8; usize::try_from(total).context("the member exceeds memory")?];
    let mut requests = Vec::with_capacity(tensors.len());
    let mut rest = buffer.as_mut_slice();
    for tensor in tensors {
        let len = usize::try_from(tensor.len).context("a tensor exceeds memory")?;
        let (dst, tail) = rest
            .split_at_mut_checked(len)
            .ok_or_else(|| anyhow!("a tensor outgrew the buffer"))?;
        requests.push(ReadRequest { offset: tensor.offset, dst });
        rest = tail;
    }
    // The request list describes the destination, so it lies below the baseline like the buffer.
    let phase = ALLOCATOR.phase();
    member.read_many(&mut requests).context("read_many failed")?;
    let seconds = start.elapsed().as_secs_f64();
    let peak = phase.peak_above();
    let rss = peak_rss()?;
    let memory = Memory { index, peak, chunk: largest_chunk(&args.container)?, rss };
    println!(
        "{{{},\"open_seconds\":{open_seconds:.6},\"threads\":{}}}",
        load_fields(tensors.len(), total, seconds, total, &memory),
        effective_threads(args.threads)
    );
    Ok(())
}

/// The next value of splitmix64 at `state`. The offsets only have to be spread and reproducible.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// `count` offsets of reads of `size` bytes, spread over a file of `len` bytes.
fn offsets(seed: u64, count: usize, size: usize, len: u64) -> anyhow::Result<Vec<u64>> {
    let size = u64::try_from(size).context("a size overflows u64")?;
    ensure!(size <= len, "a read of {size} bytes does not fit a file of {len} bytes");
    let span = len - size + 1;
    let mut state = seed;
    Ok((0..count).map(|_| splitmix64(&mut state) % span).collect())
}

/// What a `random` run reads.
enum Target {
    /// A raw file.
    Raw(File),
    /// A member of a container. `Container` is held by the `Member`.
    Packed(Member),
}

impl Target {
    /// Fills `dst` from `offset` of the file or member.
    fn read(&self, offset: u64, dst: &mut [u8]) -> anyhow::Result<()> {
        match self {
            Self::Raw(file) => file.read_exact_at(dst, offset).context("raw read failed"),
            Self::Packed(member) => member.read_at(offset, dst).context("read_at failed"),
        }
    }
}

/// Latency figures of one pass.
struct Latency {
    p50: Duration,
    p95: Duration,
    p99: Duration,
    mean: Duration,
    total: Duration,
}

impl Latency {
    /// Summarizes `samples`, which it sorts. An empty pass gives zeros.
    fn new(samples: &mut [Duration]) -> anyhow::Result<Self> {
        samples.sort_unstable();
        let at = |percent: usize| {
            let last = samples.len().saturating_sub(1);
            samples.get(last * percent / 100).copied().unwrap_or_default()
        };
        let total: Duration = samples.iter().sum();
        let count = u32::try_from(samples.len()).context("too many samples")?.max(1);
        Ok(Self { p50: at(50), p95: at(95), p99: at(99), mean: total / count, total })
    }

    /// The figures as a JSON object, in milliseconds and seconds.
    fn json(&self) -> String {
        let ms = |duration: Duration| duration.as_secs_f64() * 1000.0;
        format!(
            "{{\"p50_ms\":{:.4},\"p95_ms\":{:.4},\"p99_ms\":{:.4},\"mean_ms\":{:.4},\"seconds\":{:.4}}}",
            ms(self.p50),
            ms(self.p95),
            ms(self.p99),
            ms(self.mean),
            self.total.as_secs_f64()
        )
    }
}

/// Runs of `random`, as parsed.
struct RandomArgs<'a> {
    raw: Option<&'a Path>,
    member: &'a MemberArgs,
    size: usize,
    count: usize,
    seed: u64,
    passes: usize,
    cold: Cold,
}

/// `random`: times each read alone. With [`Cold::Each`] the cache is dropped between reads through a
/// descriptor of its own, because a `Container` does not hand out its file.
fn random(args: &RandomArgs<'_>) -> anyhow::Result<()> {
    let (target, len, evict_path, index) = if let Some(path) = args.raw {
        let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let len = file.metadata().context("cannot read the file length")?.len();
        (Target::Raw(file), len, path.to_path_buf(), 0)
    } else {
        let (member, index) = open_counted(args.member)?;
        let len = member.len();
        (Target::Packed(member), len, args.member.container.clone(), index)
    };
    let evict = File::open(&evict_path)
        .with_context(|| format!("cannot open {}", evict_path.display()))?;
    let spots = offsets(args.seed, args.count, args.size, len)?;
    let mut buffer = vec![0_u8; args.size];
    let mut passes = Vec::with_capacity(args.passes);
    // Allocated once, before the baseline, so that the measured phase holds only what the reads
    // allocate and the short JSON text of each pass.
    let mut samples = Vec::with_capacity(spots.len());
    let phase = ALLOCATOR.phase();
    for _ in 0..args.passes {
        samples.clear();
        for offset in &spots {
            if args.cold == Cold::Each {
                fadvise(&evict, 0, None, Advice::DontNeed).context("posix_fadvise failed")?;
            }
            let start = Instant::now();
            target
                .read(*offset, &mut buffer)
                .with_context(|| format!("cannot read {} bytes at {offset}", args.size))?;
            samples.push(start.elapsed());
        }
        passes.push(Latency::new(&mut samples)?.json());
    }
    let peak = phase.peak_above();
    let rss = peak_rss()?;
    let chunk = match &target {
        Target::Raw(_) => 0,
        Target::Packed(_) => largest_chunk(&args.member.container)?,
    };
    let memory = Memory { index, peak, chunk, rss };
    println!(
        "{{\"release\":{},\"size\":{},\"count\":{},\"threads\":{},\"passes\":[{}],\
         \"destination_bytes\":{},{}}}",
        !cfg!(debug_assertions),
        args.size,
        args.count,
        effective_threads(args.member.threads),
        passes.join(","),
        args.size,
        memory.fields()
    );
    Ok(())
}

/// Reads and parses the index of the container at `path`. Trusts the file: this is a benchmark of a
/// container it wrote, and `Container::open` proved it before.
fn read_index(path: &Path) -> anyhow::Result<(Index, u64)> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let file_len = file.metadata().context("cannot read the file length")?.len();
    let limits = Limits::default();
    let footer_at = file_len.checked_sub(FOOTER_LEN).ok_or_else(|| anyhow!("file too short"))?;
    let mut bytes = [0_u8; 64];
    file.read_exact_at(&mut bytes, footer_at).context("cannot read the footer")?;
    let footer = parse_footer(&bytes).context("bad footer")?;
    footer.check_layout(file_len, &limits).context("bad footer")?;
    let stored_len = usize::try_from(footer.index_stored_len).context("index too long")?;
    let raw_len = usize::try_from(footer.index_raw_len).context("index too long")?;
    let mut stored = vec![0_u8; stored_len];
    file.read_exact_at(&mut stored, footer.index_offset).context("cannot read the index")?;
    let raw = zstd::bulk::decompress(&stored, raw_len).context("cannot decode the index")?;
    footer.check_index(&raw).context("bad index")?;
    let index = Index::parse(&raw, &limits).context("bad index")?;
    Ok((index, footer.index_raw_len))
}

/// Disk layout of one member.
#[derive(Debug, Default, PartialEq, Eq)]
struct Layout {
    /// Maximal stretches of chunks read without moving the head, in member order.
    runs: u64,
    /// Sum of the distances between runs, in file bytes.
    jump_bytes: u64,
    /// Stored bytes of the chunks visited.
    stored_bytes: u64,
}

/// Walks the chunks of `member` in member order and counts where a sequential read must seek: the seek
/// proxy of risk R6. A chunk shared by two neighbouring extents counts once, as a read decodes it once.
fn member_runs(index: &Index, member: &MemberRecord) -> anyhow::Result<Layout> {
    let mut layout = Layout::default();
    let mut previous: Option<(u64, u64)> = None;
    for extent in index.member_extents(member).context("bad member")? {
        let blob = index
            .blobs()
            .get(usize::try_from(extent.blob_id).context("bad blob id")?)
            .ok_or_else(|| anyhow!("extent names a missing blob"))?;
        let chunks = index.blob_chunks(blob).context("bad blob")?;
        let factor = extent.view().context("bad view")?.length_factor();
        let canonical = extent.virt_len / factor;
        if canonical == 0 {
            continue;
        }
        let chunk_size = u64::from(blob.chunk_size);
        let first = extent.blob_offset / chunk_size;
        let last = (extent.blob_offset + canonical - 1) / chunk_size;
        let first = usize::try_from(first).context("chunk number overflows")?;
        let last = usize::try_from(last).context("chunk number overflows")?;
        let visited = chunks.get(first..=last).ok_or_else(|| anyhow!("extent outside its blob"))?;
        for chunk in visited {
            let end = chunk.file_offset + u64::from(chunk.stored_len);
            match previous {
                Some((start, _)) if start == chunk.file_offset => continue,
                Some((_, head)) if head == chunk.file_offset => {}
                Some((_, head)) => {
                    layout.runs += 1;
                    layout.jump_bytes += head.abs_diff(chunk.file_offset);
                }
                None => layout.runs += 1,
            }
            layout.stored_bytes += u64::from(chunk.stored_len);
            previous = Some((chunk.file_offset, end));
        }
    }
    Ok(layout)
}

/// `seek-map`: one record per member, and the raw index size the memory bound needs.
fn seek_map(path: &Path) -> anyhow::Result<()> {
    let (index, index_raw_len) = read_index(path)?;
    let mut members = Vec::with_capacity(index.members().len());
    for member in index.members() {
        let name = index.string(member.name).context("bad member name")?;
        let layout = member_runs(&index, member).with_context(|| format!("member {name}"))?;
        members.push(format!(
            "{{\"name\":{},\"len\":{},\"extents\":{},\"runs\":{},\"jump_bytes\":{},\"stored_bytes\":{}}}",
            json_string(name),
            member.file_len,
            member.extent_count,
            layout.runs,
            layout.jump_bytes,
            layout.stored_bytes
        ));
    }
    let largest_chunk = chunk_of(&index);
    println!(
        "{{\"index_raw_bytes\":{index_raw_len},\"largest_chunk\":{largest_chunk},\"members\":[{}]}}",
        members.join(",")
    );
    Ok(())
}

/// The largest `chunk_size` of the blobs of `index`, 0 for a container without blobs. The memory bounds
/// are stated in it.
fn chunk_of(index: &Index) -> u32 {
    index.blobs().iter().map(|blob| blob.chunk_size).max().unwrap_or(0)
}

/// The largest `chunk_size` of the container at `path`, read from its index.
fn largest_chunk(path: &Path) -> anyhow::Result<u32> {
    let (index, _) = read_index(path)?;
    Ok(chunk_of(&index))
}

/// Arguments of `encode-memory`, as parsed.
struct EncodeArgs<'a> {
    input: &'a Path,
    output: &'a Path,
    mode: ModeArg,
    threads: usize,
}

/// `encode-memory`: writes `input` as a one-member container through `PackBuilder::add_file` and
/// `finish`, and prints the Rust peaks of the write above the live bytes before it.
///
/// `ingest_peak_bytes` covers `new` and `add_file`, `finish_peak_bytes` the commit with its
/// verification, `rust_peak_bytes` the larger. `index_bytes` is the peak of opening the written file
/// with one thread, no pool: the in-memory index of this write as the commit's verification holds it,
/// the "plus the index" of the write bound. `chunk_bytes` is the largest `chunk_size` of the output.
fn encode_memory(args: &EncodeArgs<'_>) -> anyhow::Result<()> {
    let name = args
        .input
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| anyhow!("{} has no UTF-8 file name", args.input.display()))?;
    let input_bytes = fs::metadata(args.input)
        .with_context(|| format!("cannot read the length of {}", args.input.display()))?
        .len();
    let options = EncodeOptions { mode: args.mode.mode(), threads: args.threads };
    let start = Instant::now();
    let write = ALLOCATOR.phase();
    let mut builder = PackBuilder::new(args.output, &options)
        .with_context(|| format!("cannot start {}", args.output.display()))?;
    builder
        .add_file(args.input, name)
        .with_context(|| format!("cannot encode {}", args.input.display()))?;
    let ingest = write.peak_above();
    let held = ALLOCATOR.live().saturating_sub(write.base());
    // A new phase restarts the peak, so the commit is measured from what the builder holds now.
    let commit = ALLOCATOR.phase();
    let report = builder.finish().with_context(|| format!("cannot commit {}", args.output.display()))?;
    let finish = held.saturating_add(commit.peak_above());
    let seconds = start.elapsed().as_secs_f64();
    // Before the open and the index read below, which this record must not count.
    let rss = peak_rss()?;
    let open_options = OpenOptions { threads: 1, ..OpenOptions::default() };
    let opening = ALLOCATOR.phase();
    let container = Container::open(args.output, &open_options)
        .with_context(|| format!("cannot open {}", args.output.display()))?;
    let index = opening.peak_above();
    drop(container);
    println!(
        "{{\"release\":{},\"mode\":{},\"threads\":{},\"seconds\":{seconds:.6},\"input_bytes\":{input_bytes},\
         \"output_bytes\":{},\"ingest_peak_bytes\":{ingest},\"finish_peak_bytes\":{finish},\
         \"rust_peak_bytes\":{},\"index_bytes\":{index},\"chunk_bytes\":{},\"peak_rss_bytes\":{}}}",
        !cfg!(debug_assertions),
        json_string(args.mode.name()),
        effective_threads(args.threads),
        report.output_len,
        ingest.max(finish),
        largest_chunk(args.output)?,
        rss
    );
    Ok(())
}

/// Runs the chosen measurement.
fn run(cli: &Cli) -> anyhow::Result<()> {
    match &cli.command {
        Command::Tensors { member, out } => tensors(member, out),
        Command::SeekMap { container } => seek_map(container),
        Command::LoadRaw { file, plan } => load_raw(file, plan),
        Command::LoadDecode(member) => load_decode(member),
        Command::LoadMany(member) => load_many(member),
        Command::Random { raw, member, size, count, seed, passes, cold } => {
            if *size == 0 || *passes == 0 {
                bail!("--size and --passes must not be 0");
            }
            random(&RandomArgs {
                raw: raw.as_deref(),
                member,
                size: *size,
                count: *count,
                seed: *seed,
                passes: *passes,
                cold: *cold,
            })
        }
        Command::EncodeMemory { input, output, mode, threads } => encode_memory(&EncodeArgs {
            input,
            output,
            mode: *mode,
            threads: *threads,
        }),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    run(&cli)?;
    std::io::stdout().flush().context("cannot flush stdout")
}

#[cfg(test)]
mod tests {
    use super::{Latency, json_string, offsets, parse_plan};
    use std::time::Duration;

    #[test]
    fn strings_are_escaped_for_json() {
        assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\u000a\"");
        assert_eq!(json_string("торт.jpg"), "\"торт.jpg\"");
    }

    #[test]
    fn offsets_are_reproducible_and_keep_the_read_inside_the_file() {
        let first = offsets(7, 1000, 4096, 10_000).expect("offsets");
        assert_eq!(first, offsets(7, 1000, 4096, 10_000).expect("offsets"));
        assert!(first.iter().all(|offset| offset + 4096 <= 10_000));
        assert_ne!(first, offsets(8, 1000, 4096, 10_000).expect("offsets"));
        assert_eq!(offsets(1, 3, 10, 10).expect("offsets"), [0, 0, 0]);
        assert!(offsets(1, 3, 11, 10).is_err());
    }

    #[test]
    fn plans_parse_and_bad_lines_fail() {
        assert_eq!(parse_plan("0 8\n8 16\n").expect("plan"), [(0, 8), (8, 16)]);
        assert!(parse_plan("0\n").is_err());
        assert!(parse_plan("x 1\n").is_err());
    }

    #[test]
    fn percentiles_come_from_sorted_samples() {
        let mut samples: Vec<_> = (1..=100).rev().map(Duration::from_millis).collect();
        let latency = Latency::new(&mut samples).expect("latency");
        assert_eq!(latency.p50, Duration::from_millis(50));
        assert_eq!(latency.p99, Duration::from_millis(99));
        assert_eq!(latency.total, Duration::from_millis(5050));
        let empty = Latency::new(&mut []).expect("latency");
        assert_eq!(empty.p50, Duration::ZERO);
    }
}
