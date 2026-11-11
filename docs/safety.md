# Security and robustness

This page is for anyone who opens containers they did not make, or relies on ziptensors to keep data
intact. It says what the library checks, what it does not protect against, and what you must set
yourself.

In short:

1. **Set a decode budget** (`max_total_decoded`) for every container you did not make.
2. **The hashes detect damage; they do not authenticate.** Trust a container only as far as you
   trust its source, as you would the original model file.
3. **Member names come from the file.** ziptensors checks them before joining them to a directory;
   your own code must do the same.
4. ziptensors **never executes or unpickles** anything it stores.

## Opening untrusted containers

A container is untrusted input. The reader treats it that way: every length and offset is validated
with checked arithmetic before use, each declared size is checked against the caller's limits before
the read or allocation it sizes (the index against `max_index_bytes`, the members against the decode
budget before any member data is read), and every failure is reported as a typed error. Format
features, transforms and codecs that the library does not know are rejected (fail closed), never
guessed at.

What the reader validates at open, before any tensor data is read:

- the header and footer: magic bytes, version, required feature bits, their own hashes, and that the
  index lies exactly where the footer says;
- the index: its size against `max_index_bytes`, that it decompresses to exactly the declared length,
  its XXH3 hash, and every record (names valid UTF-8 and unique, every offset and count in range,
  chunk ranges inside the data region and not partially overlapping, extents that cover each member
  exactly with no gap or overlap, at most 16 dimensions per tensor);
- the decode budget: the sum of member lengths against `max_total_decoded`, and the total decode work
  against `max_decode_amplification` (on top of a fixed 256 MiB floor).

While reading, each chunk's decoded length is enforced exactly, and with chunk verification on (the
default) its XXH3-64 is checked. Reads stay within a fixed memory bound proportional to
`threads × chunk size` whatever the file says; see the memory contract in [rust.md](rust.md#threads-and-memory).

### The decode budget: `max_total_decoded`

The library default is **no budget** (`u64::MAX`). Without one, a small hostile file can declare
members of enormous size, and a restore or verify will try to decode all of it. Set the largest total
size you are prepared to decode:

| Interface | How to set it | Note |
|---|---|---|
| Command line | `--max-total-decoded <BYTES>` on `restore`, `extract`, `verify`, `inspect`, `merge`, `edit`, `compact`, `recover`, `compress`, `decompress`, `mount` | See [cli.md](cli.md). |
| Python | `max_total_decoded=` on `ziptensors.Container(...)` and on the functions that open containers | 0 is a budget of zero bytes. See [python.md](python.md). |
| Rust | `OpenOptions::max_total_decoded`, and the field of the same name in the tree, overlay and edit options | 0 is a budget of zero bytes. See [rust.md](rust.md). |
| C | `zt_open_options.max_total_decoded` (and in `zt_edit_options`, `zt_recover_options`) | **0 means no budget in C**, since C has no "unset". See [c-api.md](c-api.md). |

A container over the budget is refused at open with `BudgetExceeded` (C: `ZT_STATUS_BUDGET_EXCEEDED`),
after its index is read but before any member data is read. Leave `max_index_bytes` (default 256 MiB, capped at 1 GiB) and
`max_decode_amplification` (default 16) at their defaults unless a file you trust needs more.

## What the hashes protect, and what they do not

Each chunk carries an XXH3-64 hash of its decoded bytes, and each member a BLAKE3 hash of the whole
original file. They detect **accidental damage**: a disk error, a truncated download, a bit flip.

They do **not authenticate** a file. There is no signature and no key, and the hashes are stored in
the same file, so anyone who can change a container can recompute every hash and the forged file
passes every check. In detail:

| Check | What it proves | What it does not prove |
|---|---|---|
| Chunk verification on (`Verify::Chunk`, `ZT_VERIFY_CHUNK`, the default) | Each decoded chunk matches the hash in the index. A damaged chunk fails its read with `ChecksumMismatch` or `ChunkCorrupt`. | That the index itself is genuine. XXH3 is no defence against a forger. A plain read (`read_at`) never checks the member's BLAKE3. |
| Chunk verification off (`Verify::Off`, `ZT_VERIFY_OFF`, Python `verify="off"`) | Only that decoded lengths are exact. | Anything about the content: a read returns whatever decodes. Use it only for files you trust, when you need the speed. |
| Whole-member BLAKE3 (`verify`, `restore`, `restore_file`, `restore --all`, `extract` to a file, and the final decode of every write) | The member's bytes are exactly the bytes whose hash the file states. A restored file appears only after this check passed. | That the stated hash is the hash of the file you expected: it comes from the same container. Compare it with a hash from a trusted source if that matters. |

Also not covered by any member hash: member names (a renamed member stays valid), the tensor table
(names, shapes, dtypes, offsets) and the recognized profile. These are hints that never decide
restored bytes, but code that trusts them (for example to size a buffer) must check them.

Streaming a member to standard output (`extract ... -`, Rust `Member::decode_to`) writes bytes as
they are decoded, before the whole-member hash is known. A mismatch is reported only at the end, and
whatever consumed the output must then discard it.

## Member names and restore paths

Member names are `/`-separated paths stored in the file, so a hostile container could hold names
such as `../../.bashrc` or `/etc/passwd`. Whenever ziptensors joins a member name to a directory
(`restore --all`, `Container::restore_all`, `zt_restore_all`, Python `restore_all`, `decompress`), it
first checks the name and refuses it unless:

- it is not empty and has no NUL byte;
- it does not start with `/` or `\`;
- no part of it, split on both `/` and `\`, is `..`;
- it has no `:` (which rules out drive letters and Windows alternate data streams).

`restore --all` checks **every** name before it creates anything: a container with a single unsafe
name leaves no trace, not even the output directory. The error is `UnsafeMemberName`
(C: `ZT_STATUS_UNSAFE_MEMBER_NAME`).

The writers apply the same rule to names they introduce. An edit keeps the names of kept and replaced
members as they are, so a container that ziptensors edited but did not create can still hold a name
that a restore refuses. `restore` of a single member and `extract` write only to the path you give.

If your own code uses member names as paths, check them the same way first: in Rust with
`ziptensors_core::format::check_restore_name`; in C and Python, with an equivalent check of your own.
Restore writes file content only: modification times, permissions and symbolic links are not
restored, and a container never stores links.

## Source files you compress

When encoding, ziptensors recognizes safetensors, GGUF, ONNX and PyTorch zip files to compress them
better. These parsers only choose how bytes are compressed: a wrong or failed parse costs compression
ratio, never correctness, and a file the parsers do not understand is stored as opaque bytes. Every
write decodes the whole result and compares it with the BLAKE3 of the source before the output
appears, so a parser problem cannot produce a container that restores different bytes.

## Pickle: nothing is unpickled

PyTorch checkpoints (`.pt`, `.bin`, `.pth`) contain pickle data, which can run arbitrary code when
loaded. ziptensors never unpickles anything:

- the PyTorch zip parser reads only ZIP structures, and a legacy (pre-1.6) `torch.save` file is
  recognized by comparing its first bytes, nothing more;
- `ziptensors.torch.load_state_dict` loads only safetensors and GGUF members, whose tensor tables
  state real dtypes and shapes, and refuses any other member.

To load a PyTorch checkpoint stored in a container, open the member as a file and use PyTorch's
restricted loader: `torch.load(member.open(), weights_only=True)`. What you do with restored bytes
afterwards (for example `torch.load` without `weights_only`) is outside ziptensors.

## Writes, edits and crashes

Every write (encode, pack, edit, compress, restore) decodes and verifies its whole result before it
becomes visible; there is no option to skip that. How an interruption (crash, power loss, full disk)
leaves the file depends on how it is written:

| Operation | After an interruption |
|---|---|
| encode, pack, restore, edit with the default rewrite strategy, compact | The output appears by one rename, after verification and sync. Before that, the old file (if any) is untouched; a leftover temporary file next to the output is all that may remain. |
| edit with the append strategy (`edit -F`, `--fast`) | The old container is intact below the new data. After a failure the library cuts the partial tail off. After a crash or power loss, or if that cut fails, the file does not open until `ziptensors recover` restores the previous generation. |
| edit or compact in place (`-R`, `--risky`) | **Risky.** An interruption after the first write leaves a damaged container that may not be recoverable. Keep a backup or the original inputs. |
| `compress` / `decompress` of a tree | Each file is committed and verified before its twin is removed; a rerun resolves a half-finished pair. |

One status means the output **was** written despite the error: `CommittedNotDurable`
(`ZT_STATUS_COMMITTED_NOT_DURABLE`, Python `NotDurableError`): the verified file is in place, but its
directory could not be synced, so a power loss may bring back the previous file.

A reader never falls back to an older generation by itself; only an explicit `recover` shortens a
file, and by default it verifies the generation it keeps. See [editing.md](editing.md) for the edit
strategies in full.

## The mount is read-only

`ziptensors mount` (and the Rust and Python mount APIs) shows containers as read-only files: every
open for writing is refused, files show read-only permissions, and nothing under the mount point can
be changed. On Linux the mount is private to the user who made it. Member names a platform cannot
show safely (unsafe names, `.` and `..` parts, names that collide, and on Windows reserved device names
and forbidden characters) refuse the whole mount. Set `--max-total-decoded` when mounting files you do
not trust. See [mount.md](mount.md).

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**
(<https://github.com/Vasyanator/ziptensors/security/advisories/new>). Do not open a public issue.
Include the version or commit, the platform, and a way to reproduce, ideally a small container or a
script that builds one. The full policy, including what is in and out of scope, is in
[SECURITY.md](../SECURITY.md).

In scope are, for example: a crash, panic, hang or out-of-bounds access caused by a container; memory
or decode work beyond the budgets above; wrong bytes returned without an error, with chunk
verification on, from a container that was not deliberately forged; a member name that makes a
restore or the mount write or show anything outside the target. Out of scope: resource use of a
container opened without `max_total_decoded`, and forged containers whose hashes were recomputed.
