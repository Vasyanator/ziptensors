# Editing containers

A container can be changed after it was written: members added, removed or replaced, dead data dropped,
or every member re-encoded. This page explains the three ways an edit can be written to disk, what each
guarantees when it is interrupted, and how to repair a container whose end was torn.

Flags are listed in [cli.md](cli.md#edit); Python details are in [python.md](python.md), Rust details in
[rust.md](rust.md), C details in [c-api.md](c-api.md).

## Quick guide

```sh
# Add files, one under a chosen member name
ziptensors edit models.ziptensors --add lora.safetensors --add-as vae/vae.safetensors new-vae.safetensors

# Remove one member and replace the content of another
ziptensors edit models.ziptensors --remove model.safetensors --replace config.json config-v2.json

# Append only: fast, needs space only for the new data
ziptensors edit models.ziptensors --add extra.safetensors -F

# Drop the dead data that appends leave behind
ziptensors compact models.ziptensors

# After an interrupted -F: see what recover would cut, then cut it
ziptensors recover models.ziptensors --dry-run
ziptensors recover models.ziptensors
```

Which strategy to use:

| Situation | Use |
|---|---|
| The normal case; there is free space for one more copy of the container | the default (no flag) |
| Adding to a large container often, little free space for the result, speed matters | `-F` (append), then `compact` now and then |
| No free space for a second copy, and you have a backup or the original files | `-R` (in place) |
| You want the best size after many edits, or a different mode | `-C` (repack) or `compact -C` |

## Operations

| Flag | Python | Rust | Effect |
|---|---|---|---|
| `--add PATH` | `AddOp(path)` | `EditOp::Add { path, name, delta_base: None }` | New member named after the file, at the end. |
| `--add-as NAME PATH` | `AddOp(path, name)` | `EditOp::Add { .. }` | New member `NAME`, at the end. |
| `--remove NAME` | `RemoveOp(name)` | `EditOp::Remove { name }` | Member removed. |
| `--replace NAME PATH` | `ReplaceOp(name, path)` | `EditOp::Replace { name, path, delta_base: None }` | Member's content replaced, in its place. |

Rules, checked before anything is written:

- The operations are a set: each member name may appear in one operation only. An add of an existing
  name is refused (`EditRefused`, use `--replace`); a remove or replace of a missing member fails with
  `MemberNotFound`.
- Order of the result: kept members in their old order, a replaced member in its old slot, added
  members at the end in the order given (`--add` and `--add-as` keep their command line order).
- An added name must be one a restore accepts (no `..`, no leading `/` or `\`, no `:`); otherwise
  `UnsafeMemberName`.
- Input files must be regular files: a symbolic link is refused (`SymlinkInput`), and so is the
  container itself (`OutputIsInput`).
- Kept members stay byte-identical. New files are encoded in `--mode` (default `max`).
- Every strategy decodes every member of the result and checks it against its BLAKE3 before the result
  becomes visible.

### Delta against a member

`--delta-base MEMBER` stores every added or replacing file against `MEMBER` as delta blobs where that is
smaller, as `pack --delta-base` does. The base is a kept member, or a file of the same edit given before
the others; the base file itself is stored plain. It needs `--mode max` (the default; another mode is a
usage error, exit 2). Two strategies restrict it:

- `-F` can add delta blobs only to a container that already holds delta blobs; on any other it is
  refused (`EditRefused`) and the default rewrite works.
- `-R` cannot use a kept member as base, because kept data moves during the rewrite; a file added by
  the same edit can be the base.

```sh
ziptensors edit pair.ziptensors --add-as finetune-v2.safetensors finetune-v2.safetensors --delta-base base.safetensors
```

## The three strategies

### Default: rewrite through a temporary file

The edited container is written to a temporary file in the same directory
(`<container>.<pid>-<n>.ziptensors-tmp`), decoded in full and checked against every member's BLAKE3,
synced, and renamed over the container. Dead data of earlier appends is dropped on the way.

- **On any failure or interruption** the container is untouched; the temporary file is removed on a
  failure. A killed process can leave the temporary file behind: delete it once no ziptensors process
  is running.
- **One exception:** `CommittedNotDurable` (code 20, Unix) means the rename happened and the new
  container is complete, but syncing the directory failed, so a power loss before the system syncs on
  its own may bring back the old file. Run `sync`.
- **Disk space:** a full second copy of the result, in the container's directory.
- **Open readers:** on Linux and macOS, a program that already has the container open keeps reading the
  old version. On Windows the rename can be refused while another program has the file open; close it
  first.

### `-F`, `--fast`: append

Only the new data is written, after the old end of the file, followed by a new index and footer. Kept
members are referenced where they are. The old index and footer stay inside the file as dead bytes,
and so does the data of removed or replaced members; `compact` drops them.

The append writes the new chunks, decodes every member of the result through the file and checks its
BLAKE3 before writing any byte of the new index, writes and syncs the index, then writes and syncs the
footer. No byte of the old container is changed.

- **On a failure** the file is cut back to its old length: it is the old container, byte for byte.
- **If that cut fails** the command fails with `TrailingBytes` (code 23): the old container is intact
  at the start of the file, followed by bytes that could not be removed. Close every program that has
  the file open, then run `ziptensors recover`.
- **If the process is killed or the machine loses power** the old container is intact below a partial
  tail, and the file does not open (`MalformedIndex`, usually "end magic is wrong") until
  `ziptensors recover` cuts the tail. No reader falls back to the old version on its own.
- **Disk space:** only the new data, index and footer.
- **Open readers:** a program that opened the container before the append keeps reading the old version;
  one that opens it during the append fails to open it.
- Not with `-C`; `compact` has no append form.

### `-R`, `--risky`: in place

The container is rewritten inside its own file: the chunks of kept members move towards the start, new
files follow, then a new index and footer, and the file is cut to its new length. There is no temporary
file. Before its first write the command prints a warning on standard error.

- **Before the first write** (all checks, the plan, refusals) a failure leaves the file untouched.
- **After the first write, any failure or interruption damages the container.** A failure is reported as
  `DamagedInPlace` (code 21). A crash or power loss leaves a file that does not open, and
  `ziptensors recover` may not be able to repair it: it refuses to hand back an older generation whose
  data has moved. Restore the container from a backup, or pack it again from the original files.
- **If only the final cut fails** (on Windows, for example, when another program maps the file), the
  command fails with `TrailingBytes` (code 23): the new container is complete at the start of the file,
  and `ziptensors recover` cuts the rest.
- **Disk space:** none beyond the growth of the file, if the edit adds more than it removes. Memory: one
  8 MiB move buffer above the usual writer memory.
- **Open readers:** close every program that reads the file first, a mount included. There is no lock;
  a reader that stays open reads moved bytes and fails (`ChunkCorrupt`) or, with chunk verification off,
  gets wrong bytes.
- Not with `-C`, and not with a kept member as delta base.

Use `-R` only when there is a backup or the original files.

### Comparison

| | Default | `-F` | `-R` |
|---|---|---|---|
| Extra disk space | whole result | new data only | none (growth only) |
| Speed | rewrites everything | writes only what is new | moves the kept data |
| Interrupted | container untouched | old version intact, needs `recover` | container damaged |
| Dead data | dropped | kept until `compact` | dropped |
| Readers during the edit | keep the old file (Unix) | old readers keep working | must be closed |
| `-C` (repack) | yes | no | no |

## Compaction and repack

`ziptensors compact CONTAINER` is an edit with no operation: it writes the container again without dead
bytes. `edit` and `compact` print `dead bytes before`, the bytes of the old file no member needed.
`compact -R` compacts in place, with the risks above.

`-C` (`edit -C`, or `compact -C`) re-encodes every kept member with `--mode`, and deduplication and
delta pairing are redone over the whole result. It is slow, works only with the default strategy, and is
the way to move a container to another mode or to regain the size of a fresh `pack` after many edits.
A kept member that was stored as delta is paired again with the first earlier kept member that holds
its base; with none, it is stored plain. Outside `max` no delta blob is stored. `compact` without `-C`
ignores `--mode`.

```sh
ziptensors compact models.ziptensors -C --mode max
```

## Generations

Each container's footer carries a generation number: 1 in a fresh file, and every edit writes the
previous value plus one (a value already at 2^64 − 1 stays there). `edit`, `compact` and `recover`
print it. It is informational: readers open whatever the last 64 bytes of the file describe, and never
choose a version by its number.

An append leaves every earlier generation inside the file, together with the data it names. That is
what lets `recover` go back to the last complete one after an interrupted append.

## recover

`ziptensors recover CONTAINER` shortens a container whose end does not close a valid container to the
latest generation that is valid. It runs only when asked; no reader recovers on its own.

What it does:

1. If the file opens as it is, nothing is done (`intact, nothing to do`).
2. Otherwise it searches backwards from the end for footers. Each candidate is checked as if the file
   ended there, with every rule a reader applies when it opens a file.
3. By default every member of the candidate is decoded and checked against its BLAKE3. A candidate that
   fails is passed over, and the search goes on to the next older one.
4. The file is cut to the end of the generation found and synced. Nothing before that point is written.

```text
$ ziptensors recover models.ziptensors --dry-run
would recover (dry run, nothing written): models.ziptensors
generation: 5
file bytes: 41306
valid bytes: 29369
verified: 31182 bytes
scanned bytes: 41242
passed over: 0
warning: recovered generation 5; the 11937 bytes after it (41306 -> 29369 bytes) would be discarded, with any newer data they held
```

`passed over` counts candidates later in the file than the one kept that looked like a footer but failed
the checks. After a torn append it is 0. Above 0, newer data may have been in the bytes that are cut: check
the members of the result before relying on it.

| Flag | Default | Use |
|---|---|---|
| `--dry-run` | off | Say what would be cut; write nothing. Works on a read-only file. Run this first. |
| `--no-verify` | off | Skip the full decode of the candidate. Faster; only for a file never edited with `-R`. |
| `--max-candidates N` | `64` | Bound on candidates checked in full. |
| `--max-scan-bytes BYTES` | no limit | Bound on distinct bytes read during the backward search. |
| `--max-total-decoded BYTES` | none | Decode budget each candidate is judged under. |

When to use it:

- The file fails to open after an interrupted `edit -F` (killed process, power loss, full disk).
- A command failed with `TrailingBytes` (code 23); close every program that has the file open first.

What it can and cannot fix:

- It can only remove bytes from the end. Damage in the middle of the file is not repaired.
- After an interrupted `-R` it usually cannot help: the older generations it finds no longer verify, and
  it fails rather than hand back wrong data.
- If no valid generation is found, it fails with `Unrecoverable` (code 24), or, when a generation was
  structurally valid but did not verify, with that verification error. The file is unchanged.
- If a candidate is over the limits it was given (`BudgetExceeded`), or uses something this build does
  not know, the search stops with that error instead of cutting past a possibly genuine generation.
- Without `--dry-run` it opens the file for writing even when nothing needs cutting; a read-only file
  fails with a hint to use `--dry-run`.

## Errors of editing and recovery

| Code | Name | Meaning | What to do |
|---:|---|---|---|
| 20 | `CommittedNotDurable` | The rename happened and the new file is complete, but the directory sync failed. | Run `sync`. A power loss before it may bring back the old file. |
| 21 | `DamagedInPlace` | An `-R` edit failed after it began moving data. The container is damaged. | Restore it from a backup, or pack it again from the original files. |
| 22 | `EditRefused` | The request cannot be applied: a name in two operations, an add of an existing name, a delta base that is removed or comes later, `-F` with a delta base on a container without delta blobs, `-R` with a kept member as base. Nothing was written. | Change the request, often the strategy. |
| 23 | `TrailingBytes` | A valid container is followed by bytes that could not be cut off (failed cut-back of `-F`, failed final cut of `-R`). | Close every program that has the file open, then `ziptensors recover`. |
| 24 | `Unrecoverable` | `recover` found no valid generation within its bounds. Nothing was changed. | Restore the file from a backup. |
| 25 | `TreeConflict` | `compress`/`decompress` names collide (see [directories.md](directories.md#collisions)). Nothing was written under that path; paths named earlier on the command line are already done. | Rename or remove the files named. |

The command line exits with 1 for each of these and names the code in its `ERROR` log line. Python
raises `NotDurableError` (20), `CorruptionError` (21), `InputError` (22, 25), `ZiptensorsIoError` (23)
and `FormatError` (24), each with the number in its `code` attribute. Rust returns the matching
`ziptensors_core::Error` variant; C returns `ZT_STATUS_COMMITTED_NOT_DURABLE` to
`ZT_STATUS_TREE_CONFLICT`.

## The same operations from Python and Rust

Python:

```python
import ziptensors
from ziptensors import AddOp, RemoveOp, ReplaceOp

report = ziptensors.edit("models.ziptensors", [
    AddOp("lora.safetensors"),
    AddOp("new-vae.safetensors", "vae/vae.safetensors"),
    RemoveOp("model.safetensors"),
    ReplaceOp("config.json", "config-v2.json"),
])
print(report.kept, report.added, report.removed, report.replaced, report.generation)

ziptensors.edit("models.ziptensors", [AddOp("extra.safetensors")], strategy="append")   # -F
ziptensors.compact("models.ziptensors")                                                  # compact
report = ziptensors.recover("models.ziptensors", dry_run=True)
print(report.valid_len, report.file_len, report.passed_over)
```

`strategy` is `"rewrite"` (default), `"append"` or `"in_place"`; `repack=True` is `-C`. Close any
`Container` open on the file before a rewrite or an in-place edit.

Rust:

```rust
use std::path::{Path, PathBuf};
use ziptensors_core::{EditOp, EditOptions, RecoverOptions, Strategy, edit, recover};

fn update() -> Result<(), ziptensors_core::Error> {
    let container = Path::new("models.ziptensors");
    let ops = [
        EditOp::Add { path: PathBuf::from("lora.safetensors"), name: "lora.safetensors".into(), delta_base: None },
        EditOp::Remove { name: "model.safetensors".into() },
    ];
    let report = edit(container, &ops, &EditOptions::default())?;
    println!("generation {}", report.generation);

    let append = EditOptions { strategy: Strategy::Append, ..EditOptions::default() };
    edit(container, &[EditOp::Add { path: "extra.safetensors".into(), name: "extra.safetensors".into(), delta_base: None }], &append)?;

    edit(container, &[], &EditOptions::default())?; // compact
    let check = recover(container, &RecoverOptions { dry_run: true, ..RecoverOptions::default() })?;
    println!("valid {} of {} bytes", check.valid_len, check.file_len);
    Ok(())
}
```
