# Module: crates/ziptensors-core/src/edit

## Purpose

Editing an existing container: `Add`, `Remove`, `Replace` of members, compaction (an edit with no
operation) and repack (every kept member re-encoded), by one of the commit strategies. The one owner of
the edit rules; frontends translate their arguments into `EditOp`s and `EditOptions` and call `edit`.
Spec invariant 12 (a)/(b) and section 9.3 define the commit protocols and generations it follows; the
in-place strategy (`-R`) is outside invariant 12 by design: it trades crash safety for space, and an
interruption damages the container.

## Architecture

```text
edit(path, ops, options)
  check_strategy      combinations refused without a look at the file (Append + repack, Append + no op,
                        InPlace + repack)
  Opened::open        one handle (read + write under Append and InPlace, with a clone kept for the
                        sink): FileSource, Container::from_source + header + footer, the same bytes for
                        every decision
  plan::resolve       op set -> result order (Steps), delta bases (Kept | New), counts
  bit-0 rule          Append + any delta base while the header's required bit 0 is clear -> EditRefused
  kept-base rule      InPlace + a delta base that is a kept member -> EditRefused
  check_inputs        every input file: regular, no symlink, not the container
  dead_bytes          what the source holds that no member needs (report only)
  rewrite::run        PackBuilder::new beside the container -> Keep: keep_member_from (with a source
                        when named as a base; add_source over MemberSource under repack) -> File:
                        add_from_file -> drop the source -> finish_with(Replace, gen + 1)
  append::run         Seed::of(source) -> Sink::append_to(the clone) -> PackBuilder::with_sink -> Keep: carry_member
                        (records only) -> File: add_from_file (a kept base through BaseCatalog::from_member
                        + DeltaSpec::Member) -> finish_with(Append, gen + 1)
  inplace::run        dry phase: inplace::plan (kept blobs, moves) -> Seed::in_place (kept blobs renumbered,
                        chunks rebased) -> length check; then Sink::in_place(the clone) -> PackBuilder::with_sink
                        -> move_data per move -> Keep: carry_member -> File: add_from_file (a base only
                        DeltaSpec::Added) -> finish_with(InPlace, gen + 1)
```

Every check above the strategy runs before any byte is written. A failure after the builder exists ends
in `PackBuilder::fail` (or the commit's own failure ending), never in a plain drop, so that an append
that cannot be cut back surfaces as `Error::TrailingBytes`, and an in-place rewrite that began writing
as `Error::DamagedInPlace`.

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Public API (`edit`, `EditOp`, `Strategy`, `EditOptions`, `EditReport`), the pre-write checks, `Opened`, what members keep alive (`reached_blobs`, `live_ranges`) and dead-byte accounting, dispatch, the test seams (`Hooks`). |
| `plan.rs` | `resolve`: the operation-set rules and the member order of the result. Pure, over names. |
| `source.rs` | `MemberSource`: a member of an open container as a `RandomAccess` input (repack, delta bases). |
| `rewrite.rs` | `Rewrite` and `repack`, including the re-pairing rule of kept delta members. |
| `append.rs` | `Append`: a seeded builder over the append sink. |
| `inplace.rs` | `InPlace`: the pure planner (`plan`, `plan_moves`, `MOVE_BUFFER`) and `run` over the in-place sink. The proof of P1-P4 is in its header. |
| `tests.rs` | `cfg(test)`: `resolve` rules, the append protocol under crash injection, verification before visibility, `torn_append_fixture` for the tests of `recover`, the in-place planner simulated on byte arrays, and the in-place strategy under faults. Public-API cases are in `tests/edit.rs`. |

## Contracts and invariants

- `ops` is a set: a name in two operations, an `Add` of an existing member, a delta base that is the
  member itself, is removed, or comes after its dependant in the result is `EditRefused`; a `Remove` or
  `Replace` of a missing member, or a base naming nothing, is `MemberNotFound`.
- Names: only an `Add` introduces a name, and only it must be one a restore accepts
  (`UnsafeMemberName`, checked in `plan::resolve` and again by the builder under `NameRule::New`). Kept,
  replaced and repacked members keep the names of the valid source (`Step::name_rule`, `NameRule::Kept`:
  unique only), so a legal member such as `../x` never blocks an edit (spec invariant 11 binds a restore).
- Append and in-place identity: the source is opened once, read and write; the container, the seed, the
  sink and the commit's verification all go through that one handle (`Opened::writer` is its clone),
  and the path is never resolved again. A file renamed over the path meanwhile is never written; a
  length that differs from the one at open (written through another handle) is `InputChanged`.
- Result order, identical for every strategy: kept members in source order, a `Replace` in its member's
  slot, `Add`s at the end in operation order.
- Generation: the source footer's `generation + 1` (saturating), for every strategy.
- Strategy refusals (`EditRefused`, before any write): `Append` with `repack`; `Append` with no operation;
  `Append` with a delta base while required bit 0 is clear; `InPlace` with `repack`; `InPlace` with a
  delta base that is a kept member (ingest would read it through the source container, whose chunk
  offsets are stale once data moved; a base added by the same edit is read from its file). Choosing two
  strategies at once (`-F` with `-R`) is a usage error of the frontends: `Strategy` holds one.
- A delta base is a kept member or an `Add`/`Replace` file earlier in the result. Under `Rewrite` a kept
  base is copied as usual and registered with the builder (`keep_member_from` with a source), so every base is
  `DeltaSpec::Added` in this pack's own ids; under `Append` the builder is seeded, so a kept base is
  `DeltaSpec::Member` with the source's catalog and ids.
- Repack (only `Rewrite`): every kept member goes through `add_source(MemberSource, SourceStamp::Member)`
  in `options.mode`, so dedup and pairing are redone over the whole result. A kept member whose extents
  reach a delta blob pairs again with the first kept member BEFORE it in the result that holds that
  delta's base blob under an identity extent; with none it is stored plain. Outside `max` no delta blob
  is stored (policy).
- Kept members are byte-identical to the source: copied stored chunks (`Rewrite`), records over the same
  bytes (`Append`), the same stored chunks moved (`InPlace`), or re-encoded and decoded against their
  stated hash (repack). The commit decodes
  every member of the result against its source hash under every strategy.
- `EditReport::dead_bytes_before` counts source bytes no member needs: unreached blobs, older indexes
  and footers; chunk records that share bytes count once.

## Failure semantics

| Strategy | Failure | File after |
|---|---|---|
| any | a check before the first write | untouched, byte for byte |
| `Rewrite` | anything | untouched; the temporary file is removed (`CommittedNotDurable` aside, see `write`) |
| `Append` | a write, the verification, inputs, the index, the footer | cut back to its old length `L`: the source byte for byte |
| `Append` | the cut back itself fails | `TrailingBytes { valid_len: L }`; `recover` truncates it |
| `Append` | the process dies | source in `[0, L)`, then a torn tail; the file does not open until `recover` |
| `InPlace` | a check, the plan, the seed, before the first write | untouched, byte for byte |
| `InPlace` | anything after the first write (a write, an input, the verification) | `DamagedInPlace`: left as it is, damaged; restore from a backup or the inputs |
| `InPlace` | the final cut fails (Windows: another process maps the file) | `TrailingBytes { valid_len }`: the new container is complete and durable below it; `recover` truncates (footers of earlier generations left in the dead tail fail its verification and are passed over) |
| `InPlace` | the process dies | damaged; it does not open, and `recover` (which verifies by default) refuses rather than hand back a generation whose chunks moved |

### In place: why no kept byte is lost before it is copied

Kept ranges (the chunks of every blob a kept member reaches, with the bases of its delta blobs) are
sorted by offset; identical chunk records collapse to one range, since validation proves ranges
identical or disjoint (invariant 4). Range `i` goes to `dst_i = 64 + sum of the lengths before it`.
P1 `dst_i <= src_i`, since disjoint ranges inside `[64, I)` leave no more than `src_i - 64` bytes before.
P2 move `i` writes `[dst_i, dst_i + len_i)`, below the start of every later range. P3 one move that
overlaps itself is copied forward in pieces read before written (`Sink::copy_within`, at most
`MOVE_BUFFER` = 8 MiB at once, the strategy's one buffer). P4 new data starts at the end of the moved
data, a rollback never goes below it and never truncates, and the file is cut once, at the end. The
planner checks P1 and the ranges itself and refuses (`EditRefused`, nothing written) rather than trust
validation. Blob ids keep their old order, so a base stays below its delta blob. The first write of the
sink zeroes the source footer and syncs: a torn rewrite never ends in a generation whose chunks moved,
and `recover` after a failed final cut finds the new footer rather than the old one. Verification runs
through a splice view before the new index is written, so a result that does not verify is never
indexed.

## Live readers

- `Append`: a `Container` opened before the edit keeps reading below `L`, which never changes. One
  opened during the append fails at open (`MalformedIndex`, invariant 1); readers never fall back.
- `Rewrite`: the edit drops its own handles before the rename. A caller's `Container` on the same file
  keeps the old inode on Unix; on Windows the rename may be refused while it is open (`Io`, op
  `rename`), so callers drop theirs first.
- `InPlace`: a `Container`, `ReadCache` or mount open on the file reads moved bytes: `ChunkCorrupt`
  under the default chunk verification, wrong bytes with verification off. No lock exists; callers
  close every reader first. The frontends say so in the help of `-R`.

## Editing map

- Operation rules and result order: `plan.rs`, with its tests in `tests.rs`.
- A new strategy: a `Strategy` variant, its refusals in `check_strategy`, its `run` beside `rewrite.rs`,
  `append.rs` and `inplace.rs`, its ending in `write::commit` (never here).
- The in-place planner and its proof: `inplace.rs`; the moves and the seal: `write/commit.rs`
  (`Sink::copy_within`, `Sink::seal`); the renumbered seed: `Seed::in_place` and
  `format::IndexBuilder::seed_subset`.
- Re-pairing under repack: `rewrite::repack_base` / `register_owner`.
- What a builder over an existing container may do: `write/builder.rs` (`Seed`, `carry_member`,
  `finish_with`).
