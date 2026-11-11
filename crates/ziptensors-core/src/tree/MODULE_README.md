# Module: crates/ziptensors-core/src/tree

## Purpose

Per-file containers over a directory tree. `compress_tree` turns every weight file into a container of
one member beside it (`vae/diffusion_pytorch_model.safetensors` becomes
`vae/diffusion_pytorch_model.ziptensors` holding the member `diffusion_pytorch_model.safetensors`) and
removes the original once the container is proven; `decompress_tree` does the reverse; `scan_view` is
the read-only view both use, and the overlay mount builds on.

## Architecture

`tree -> write, read, profile, format, io`. Nothing here owns format logic: containers are written by
`write::encode_file` (or a `PackBuilder` for a followed link) and restored by `Member::restore_file`,
both of which verify before their rename. This module decides which file goes where and when the
redundant copy may go.

```text
scan_view (walk via write::inputs::walk, magic, index, restore path lookup)
  -> compress: classify plain files (profile::classify_weight), map to container names,
               refuse collisions, then per file: commit -> re-check original -> remove -> sync dir
  -> decompress: refuse view conflicts, then per container:
               restore -> re-check container -> remove -> sync dir
```

## Files and submodules

| Path | Role |
|---|---|
| `mod.rs` | Public surface; report types (`TreeReport`, `FileOutcome`, `FileStatus`, `Action`, `TreeTotals`, `TreeWarning`); shared steps (`hash_file`, `remove_redundant`, directory sync); the crate-private fault seam (`Stage`, `Fault`, `no_fault`). |
| `view.rs` | `scan_view`, `TreeView`, `ViewEntry`, `PerFileEntry`, `Conflict`, `FileStamp`. Owner of container recognition (magic, never extension), of "per-file" (exactly one member whose name is one safe file name), and of the restore path lookup. `inspect_file` classifies one path for `compress.rs`. |
| `compress.rs` | `compress_tree`, `CompressTreeOptions`: plan (classification, target names, collisions), followed links, pair resolution, stale containers. |
| `decompress.rs` | `decompress_tree`, `DecompressTreeOptions`: restore, pair resolution, `force`. |
| `tests.rs` | Crash order through the fault seam, totals. The public behaviour is tested in `crates/ziptensors-core/tests/tree.rs`. |

## Contracts and invariants

- **Never neither.** At any crash point the original, its twin, or both exist. A copy is removed only
  after its twin's commit returned (full decode and BLAKE3 against the source), and only if the copy's
  length and modification time (and a followed link's target) are unchanged since before that commit.
  `CommittedNotDurable` from a commit keeps the copy. A failed directory sync after a removal is a
  `Failed(Io "sync directory")` outcome with `removed` set: the removed copy may come back after a power
  loss, which a rerun resolves.
- **Collisions refuse the whole run before any write** (`Error::TreeConflict`), dry runs included:
  compress refuses two weight files whose container names are equal ignoring case, a container name
  taken by anything but a per-file container of exactly that original, and an original that another
  container already restores to; decompress refuses the view's conflicts (several containers restoring
  to one name, or to a directory or a container file).
- **Pairs are hashed every time.** Where both copies exist, the uncompressed one is hashed (BLAKE3 and
  length) against the member record. Compress: equal, then the container is fully verified, then the
  original goes (unless `keep`); different, the container is stale and replaced; verification failure
  removes nothing. Decompress: equal, the container goes; different, the container fails unless `force`.
- Per-file failures do not stop a run; reruns are idempotent (a settled tree reports no-ops only).
- Which files are weights is `profile::classify_weight` alone; non-weight files are never touched.
- Restore paths are looked up on the file system (`symlink_metadata`), not among the walked names, so a
  case-insensitive file system finds the file a rename would replace.
- Links: skipped with `TreeWarning::SymlinksSkipped` by default. `follow_symlinks` compresses a link to
  a regular weight file into a container at the link's place, named after the link, and removes the link,
  never the target; a link to a directory is never entered, a dangling one is skipped, a target outside
  the tree is only read, two links to one blob give two full containers (no cross-file dedup). Followed
  links run before plain files, so a link to an in-tree weight reads it before it is compressed. In a
  Hugging Face cache this replaces `snapshots/` links by containers and keeps `blobs/`, so the cache
  grows and stops being a cache; the warning says so.
- `*.ziptensors-tmp` files are skipped and never removed: they cannot be proven unused.
- Content only, as `restore_file`: modification time and permissions are not restored.
- The member name is authoritative; the `.ziptensors` file name is a convention.

## Known gaps

- Windows: no directory sync after a removal (same as the commit's rename, `io::TempFile`), so the
  durability of a removal rests on the file system's journal.
- The directory sync policy repeats the private one of `io::temp`; making that one `pub(crate)` would
  leave one owner.

## Editing map

- What counts as a per-file container, how a restore path is found, view conflicts: `view.rs`.
- Container naming, collision rules, link following, pair rules of compress: `compress.rs`.
- Restore rules, `force`: `decompress.rs`.
- Report shape, removal and directory sync: `mod.rs`.
- Which files are weights: `profile::classify_weight`, not here. How a tree is walked: `write::inputs::walk`.
