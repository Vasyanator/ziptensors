# ziptensors container format, version 0.3

Status: normative. This document owns every field width and every id. The implementation cites it
(`crates/ziptensors-core/src/chain.rs` for all ids except dtypes,
`crates/ziptensors-core/src/types.rs` for dtypes); a change to one requires the other.

Version 0.2 differs from version 0.1 only in the dtype registry of section 5.3, which gains the ggml
block types that had no id. A 0.1 reader reads every 0.2 file, since an unknown dtype is legal.
Version 0.3 adds delta blobs (section 7a): required feature bit 0, blob kind 1, step `delta_xor`
and the contents of `PARM`, all reserved before. A 0.3 file without delta blobs uses none of them
and is read by a 0.1 or 0.2 reader; a file with delta blobs sets bit 0, which such a reader rejects
at the header. Statements below about version 0.1 hold for later versions too, unless they say
otherwise. The file name of this document carries no version; the version it describes is the one
in its title, written in the header as `major` 0, `minor` 3 (section 3.1). Section 3.4 states what
stays compatible across versions.

The words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119. Sections 1 to 10 define the format
and what a conforming reader and writer do. Section 11 is writer policy: it decides what a writer
chooses, and nothing in it is visible to a reader.

## 1. Conventions

- All integers are little-endian. There are no varints and no JSON.
- Tensor bytes are never byte-swapped. All transforms are defined on bytes or on little-endian
  elements by integer operations, so big-endian hosts decode the same bytes.
- `XXH3` means XXH3-64 with seed 0. `BLAKE3` means BLAKE3 with a 256-bit output, unkeyed.
- `a..b` is the half-open range from `a` included to `b` excluded.
- `1 MiB` is 2^20 bytes, `1 GiB` is 2^30 bytes.
- All arithmetic on values read from a file MUST be checked: an overflow is a rejection.

The conventional file extension is `.ziptensors`. A file is identified by its magic, never by its
extension, and nothing in a reader depends on the file name.

## 2. Model

A container has four layers.

| Layer | What it is |
|---|---|
| member | A virtual file: a name, a length, a hash, a list of extents. |
| extent | A view over a byte range of one blob, mapped into one range of the member. |
| blob | Content-addressed bytes, cut into chunks and sharing one transform chain. |
| chunk | A self-contained unit of stored bytes: they decode with nothing outside themselves. A chunk of a delta blob is then combined with one chunk of its base blob, section 7a. |

A single-file container and a pack are the same thing: a pack has N member records, a single file
has one. There is one layout, and no path outside the file is ever referenced.

Below the members sits an optional tensor table. It is metadata over member offsets. Restoring a
member never consults it, so a damaged or lying tensor record cannot affect restored bytes.

**Byte-exact means content.** A restored member has the same bytes as the source file. Modification
time, permissions and symbolic links are not stored and not restored. A writer skips symbolic links.

## 3. File layout

```text
[Header, 64 bytes][data region: chunks, no padding required][index: one zstd frame][Footer, 64 bytes]
```

The index sits at the end so that a writer can stream. A reader reads the header and the last
64 bytes, then the index, then data.

### 3.1 Header

| Offset | Size | Field | Value |
|---:|---:|---|---|
| 0 | 8 | magic | `89 5A 54 53 0D 0A 1A 0A` |
| 8 | 2 | major | 0 |
| 10 | 2 | minor | 3 |
| 12 | 4 | header_len | 64 |
| 16 | 8 | required_features | bit set, section 3.3 |
| 24 | 8 | optional_features | bit set, section 3.3 |
| 32 | 24 | reserved | zero |
| 56 | 8 | header_xxh3 | XXH3 of bytes `0..56` |

### 3.2 Footer

The footer is the last 64 bytes of the file. Offsets are relative to the start of the footer.

| Offset | Size | Field | Value |
|---:|---:|---|---|
| 0 | 8 | index_offset | file offset of the stored index |
| 8 | 8 | index_stored_len | bytes of the stored index |
| 16 | 8 | index_raw_len | bytes of the decompressed index |
| 24 | 8 | index_xxh3 | XXH3 of the decompressed index |
| 32 | 8 | generation | 1 in a fresh file; an edit writes the previous value + 1; informational, no value is rejected (section 9.3) |
| 40 | 8 | reserved | zero |
| 48 | 8 | footer_xxh3 | XXH3 of footer bytes `0..48` |
| 56 | 8 | end_magic | ASCII `ZTS-END` followed by one zero byte |

`index_raw_len` has a reader cap, 256 MiB by default and configurable. It MUST NOT exceed 1 GiB in
any file: a reader rejects such a file whatever its configuration.

### 3.3 Versioning and features

- A file whose `major` the reader does not implement MUST be rejected. A reader of major 0
  implements major 0 only.
- A minor version only adds ids, sections and feature bits. It never changes the meaning of anything
  that exists.
  A reader accepts any `minor` of its major and fails on the first id it does not know, under
  invariant 9.
- `header_len` is 64 in every minor of major 0. Any other value is a rejection.
- A set bit in `required_features` that the reader does not implement MUST be rejected.
- Bits in `optional_features` that the reader does not know MUST be ignored.
- Versions 0.1 and 0.2 set no feature bits. Version 0.3 defines required bit 0, delta blobs
  (section 7a). Reserved required bits are listed in section 10.

Reserved fields, padding fields and fields marked zero MUST be written as zero. A reader MUST reject
a non-zero value in them, except in sections it skips.

### 3.4 Compatibility and stability

This section is a promise about bytes on disk.

- Version 0.3 is the first public version of the format. Versions 0.1 and 0.2 preceded it only in
  unpublished builds; their files stay readable, since a reader accepts every minor of its major.
- A file written by any published version stays readable by every later reader. A later reader never
  drops an id, section, feature bit or major that a published writer could produce.
- Changes within a major are additive and come under a new minor, as section 3.3 allows. An older
  reader reads a newer file unless the file uses something it does not know; then section 3.3 and
  invariant 9 make it fail with an error, never with wrong bytes.
- A change that alters the width, position or meaning of anything that exists needs a new major. A
  reader of the new major keeps reading every earlier major; a reader of an earlier major rejects the
  new one, under section 3.3.

Covered: the byte layout, the ids and the reader rules of sections 1 to 10. Not covered: the writer
policy of section 11, so a later writer may store the same input as different bytes, which restore to
the same content; and the programming interfaces. The Rust API of the crates, the C ABI and the
Python API are versioned separately from the format, by the semantic version of the crates
(Cargo rules: below 1.0, a minor bump may break the API). The Python package `ziptensors` carries
the version of the `ziptensors-py` crate. The C ABI also carries `ZT_ABI_VERSION`, returned at run
time by `zt_abi_version`, which is raised whenever a declaration of the header changes in a way that
breaks callers built against an older one. A library version says nothing about the format version
it writes, and the reverse.

## 4. Index

The stored index is exactly one zstd frame with the content size flag set. Decompressed, it has this
layout, with all offsets relative to the start of the decompressed index:

```text
u32 section_count
u32 0
section_count entries of 32 bytes
section bodies
```

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | tag, four ASCII bytes |
| 4 | 4 | flags: bit 0 required, other bits zero |
| 8 | 8 | offset of the body |
| 16 | 8 | len of the body, in bytes |
| 24 | 8 | count of records |

Rules for the section table:

- Tags are unique.
- Each body lies inside the decompressed index, after the table, starting at an offset that is a
  multiple of 8. Bodies are pairwise disjoint. Bytes between bodies are ignored; a writer fills them
  with zero.
- A fixed-record section has `len == count × record size`. A byte heap section (`STRS`, `PARM`) has
  `len == count`.
- A section the reader does not know is rejected if its required bit is set and skipped otherwise.
- A reader MAY bound the summed `name_len` over `MEMB` and `TENS`, since many records over one heap
  cost records × heap to check. Exceeding the bound is a budget failure, not a malformed file.
- A reader MAY bound the decode work of a container: the sum, over the extents of all its members, of
  the raw lengths of all chunks the extent touches. Reading part of a chunk decodes all of it, so many
  small extents over one chunk cost extents × chunk. A chunk of a delta blob costs its raw length
  plus the raw length of the base chunk it is combined with (section 7a). Exceeding the bound is a
  budget failure, not a malformed file. The reference reader's default is `16 × Σ file_len + 256 MiB`, with one floor for the
  whole container, so that many small members do not multiply it. An extent over a whole blob costs
  that blob, so a container of such extents has work of at most `Σ file_len`. A writer that aims small
  extents into a shared blob SHOULD pick a `chunk_size` that keeps a container within that default.
- Sections `STRS`, `CHAN`, `CHST`, `BLOB`, `CHNK`, `MEMB`, `EXTN`, `TENS`, `SHAP` MUST all be present,
  with the required bit set. Any of them may have zero records. `PARM` is optional: a file with delta
  blobs MUST have it with the required bit set, a file without them MUST NOT set that bit, and a
  writer omits `PARM` when no blob has parameters. Bytes of `PARM` no blob refers to are ignored.

| Tag | Record | Bytes | Meaning |
|---|---|---:|---|
| `STRS` | UTF-8 heap | 1 | strings, referenced by `(u32 off, u32 len)` in bytes |
| `CHAN` | chain | 8 | a transform chain |
| `CHST` | step | 8 | one transform step |
| `BLOB` | blob | 80 | a blob |
| `CHNK` | chunk | 24 | a chunk |
| `MEMB` | member | 72 | a member |
| `EXTN` | extent | 32 | an extent |
| `TENS` | tensor | 32 | a tensor table row |
| `SHAP` | dims | 8 | one `u64` dimension |
| `PARM` | byte heap | 1 | blob-kind parameters, referenced by `BLOB.params_off` and `params_len`; section 7a |

### 4.1 Records

Offsets are in bytes within the record. A reference such as `chain_id` or `first_chunk` is a
zero-based record index into the named section.

**CHAN, 8 bytes**

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 4 | first_step | index into `CHST` |
| 4 | 2 | step_count | 1 to 8 |
| 6 | 2 | zero | |

**CHST, 8 bytes**

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 2 | codec_id | section 5.1 |
| 2 | 2 | param0 | meaning depends on `codec_id` |
| 4 | 4 | param1 | meaning depends on `codec_id` |

**BLOB, 80 bytes**

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 32 | blake3 | identity, section 8; all zero when the writer did not compute it |
| 32 | 8 | raw_len | canonical bytes in the blob |
| 40 | 8 | first_chunk | index into `CHNK` |
| 48 | 4 | chunk_count | |
| 52 | 4 | chunk_size | raw bytes per chunk, except the last |
| 56 | 4 | chain_id | index into `CHAN` |
| 60 | 1 | kind | 0 chunked; 1 delta, section 7a; every other value is rejected |
| 61 | 1 | flags | zero |
| 62 | 2 | dtype_hint | section 5.3 |
| 64 | 8 | params_off | into `PARM`; zero for kind 0, a multiple of 8 for kind 1 |
| 72 | 4 | params_len | zero for kind 0, 16 for kind 1 |
| 76 | 4 | zero | |

**CHNK, 24 bytes**

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 8 | file_offset | file offset of the stored chunk |
| 8 | 8 | xxh3_decoded | XXH3 of the chunk's canonical decoded bytes, before any view |
| 16 | 4 | stored_len | stored bytes, including the prologue |
| 20 | 4 | zero | |

**MEMB, 72 bytes**

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 4 | name_off | into `STRS` |
| 4 | 4 | name_len | |
| 8 | 8 | file_len | member length in bytes |
| 16 | 32 | blake3 | BLAKE3 of the whole member content |
| 48 | 8 | first_extent | index into `EXTN` |
| 56 | 4 | extent_count | |
| 60 | 4 | first_tensor | index into `TENS` |
| 64 | 4 | tensor_count | |
| 68 | 1 | profile | section 5.4 |
| 69 | 3 | zero | |

A member name is a path string used only by a restore, with `/` as the separator.

**EXTN, 32 bytes**

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 8 | virt_offset | start in the member |
| 8 | 8 | virt_len | length in the member, at least 1 |
| 16 | 4 | blob_id | index into `BLOB` |
| 20 | 2 | view | 0 for identity, otherwise section 7 |
| 22 | 2 | zero | |
| 24 | 8 | blob_offset | start in the blob's canonical bytes |

**TENS, 32 bytes**

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 4 | name_off | into `STRS` |
| 4 | 4 | name_len | |
| 8 | 8 | virt_offset | start in the member |
| 16 | 8 | virt_len | length in the member, may be 0 |
| 24 | 4 | shape_off | index into `SHAP` |
| 28 | 2 | ndim | 0 to 16 |
| 30 | 2 | dtype | section 5.3 |

## 5. Id registry

### 5.1 Step and view ids (`u16`)

| Id | Name | Kind | Parameters |
|---|---|---|---|
| 0x0000 | invalid | | never a valid step; in `EXTN.view` it means identity |
| 0x0100 | whole | layout | `param0 = 0`, `param1 = 0` |
| 0x0101 | planes | layout | `param0` = width, one of 2, 4, 8; `param1 = 0` |
| 0x0102 | fields | layout | `param0` = table id, section 6.3; `param1 = 0` |
| 0x0200 | rotl1 | element | `param0` = width, one of 2, 4, 8; `param1 = 0` |
| 0x0301 | f16 to f32 | view | valid only in `EXTN.view` |
| 0x0302 | bf16 to f32 | view | valid only in `EXTN.view` |
| 0x0400 | delta_xor | delta | `param0` = width, one of 2, 4, 8; `param1 = 0`; version 0.3, section 7a |
| 0x0401 to 0x04FF | reserved, delta | | rejected |
| 0x0500 to 0x05FF | reserved, derived | | rejected |
| 0x8000 and above | private | | rejected by default readers |

Every other id is unknown and rejected, as is a step or view with parameters outside this table.

### 5.2 Chains

A chain is, in encoding order, an optional delta step, an optional `rotl1` step, then exactly one
layout step. So `step_count` is 1 to 3 (1 or 2 before version 0.3). The limit of 8 in the record is a
structural bound. The chain of a delta blob has a delta step and the chain of a kind 0 blob has none
(section 7a); a chain no blob uses is valid with or without one.

The **chain block** is the unit a blob's lengths must be multiples of:

| Chain | Chain block |
|---|---|
| `whole` | 1 |
| `planes(w)` | `w` |
| `fields(t)` | block bytes of table `t` |
| `rotl1(W)` then a layout | least common multiple of `W` and the layout's block above |
| a delta step of width `D` first | least common multiple of `D` and the block of the rest of the chain |

### 5.3 dtype ids (`u16`)

A dtype is a hint. An unknown dtype is legal; an unknown step is not. Id 0 means no dtype was
determined.

| Id | Name | Block bytes | | Id | Name | Block bytes |
|---|---|---:|---|---|---|---:|
| 0x0001 | BOOL | 1 | | 0x0010 | F16 | 2 |
| 0x0002 | U8 | 1 | | 0x0011 | BF16 | 2 |
| 0x0003 | I8 | 1 | | 0x0012 | F32 | 4 |
| 0x0004 | U16 | 2 | | 0x0013 | F64 | 8 |
| 0x0005 | I16 | 2 | | 0x0020 | F8_E4M3 | 1 |
| 0x0006 | U32 | 4 | | 0x0021 | F8_E5M2 | 1 |
| 0x0007 | I32 | 4 | | 0x0022 | NIBBLE (packed 4-bit, stored in bytes) | 1 |
| 0x0008 | U64 | 8 | | | | |
| 0x0009 | I64 | 8 | | | | |

The ggml block types follow. Each is one ggml block struct of `ggml-common.h`: a tensor row of `n`
elements is `n / elements` whole blocks, and `n` is a multiple of `elements`. The name is the member
of `GGMLQuantizationType` in the reference `gguf` Python package, and the ggml type is its value
there, which is how a GGUF file names the type. Ids 0x0030 to 0x0033 are from version 0.1; the rest
were added in version 0.2. Only 0x0030 to 0x0033 have a field table, section 6.3.

| Id | Name | ggml type | Block elements | Block bytes |
|---|---|---:|---:|---:|
| 0x0030 | Q8_0 | 8 | 32 | 34 |
| 0x0031 | Q4_K | 12 | 256 | 144 |
| 0x0032 | Q5_K | 13 | 256 | 176 |
| 0x0033 | Q6_K | 14 | 256 | 210 |
| 0x0034 | Q4_0 | 2 | 32 | 18 |
| 0x0035 | Q4_1 | 3 | 32 | 20 |
| 0x0036 | Q5_0 | 6 | 32 | 22 |
| 0x0037 | Q5_1 | 7 | 32 | 24 |
| 0x0038 | Q8_1 | 9 | 32 | 40 |
| 0x0039 | Q2_K | 10 | 256 | 84 |
| 0x003A | Q3_K | 11 | 256 | 110 |
| 0x003B | Q8_K | 15 | 256 | 292 |
| 0x003C | IQ2_XXS | 16 | 256 | 66 |
| 0x003D | IQ2_XS | 17 | 256 | 74 |
| 0x003E | IQ3_XXS | 18 | 256 | 98 |
| 0x003F | IQ1_S | 19 | 256 | 50 |
| 0x0040 | IQ4_NL | 20 | 32 | 18 |
| 0x0041 | IQ3_S | 21 | 256 | 110 |
| 0x0042 | IQ2_S | 22 | 256 | 82 |
| 0x0043 | IQ4_XS | 23 | 256 | 136 |
| 0x0044 | IQ1_M | 29 | 256 | 56 |
| 0x0045 | TQ1_0 | 34 | 256 | 54 |
| 0x0046 | TQ2_0 | 35 | 256 | 66 |
| 0x0047 | MXFP4 | 39 | 32 | 17 |

### 5.4 Profile ids (`u8`)

| Id | Profile |
|---|---|
| 0 | opaque |
| 1 | safetensors |
| 2 | gguf |
| 3 | onnx |
| 4 | pt_zip |

A profile is informational. An unknown id is legal.

## 6. Chunks

### 6.1 Stored form

A chunk is self-contained:

```text
[u32 stream word × S][stream 0 bytes][stream 1 bytes]…[stream S-1 bytes]
```

The first `4 × S` bytes are the prologue. Each stream word is a little-endian `u32`:

```text
word = (codec << 24) | stored_len        codec: high 8 bits, stored_len: low 24 bits
```

`S` and the raw length of every stream follow from the chain and the chunk's raw length (section
6.3), so they are not stored.

| Stream codec | Meaning |
|---:|---|
| 0 | store: the stream bytes are the raw stream |
| 1 | zstd |
| others | rejected |

A zstd stream is exactly one zstd frame with the content size flag set, the unused and reserved
descriptor bits clear (so a frame has no malleable byte), no frame checksum, no
dictionary, and a window of at most 2^24 bytes. A decoder MUST limit the window to 2^24 and MUST
require the output length to equal the stream's raw length. It need not count frames: the guarantee
it enforces is the exact output length and the chunk hash.

A stream's `stored_len` MUST NOT exceed its raw length. A store stream's `stored_len` equals it. So
`stored_len` of a chunk never exceeds `4 × S` plus the chunk's raw length.

A chunk's bytes depend only on its raw content and the chain, and for a chunk of a delta blob on its
base chunk. A chunk's raw length is `chunk_size` except for the last chunk of a blob, which has
`raw_len − (chunk_count − 1) × chunk_size` bytes.

`chunk_size` is at least 1, at most 8 MiB (8 388 608), and a multiple of the chain block.

### 6.2 Decoding

Given a chunk's stored bytes, chain and raw length `R`:

1. Compute `S` and the stream raw lengths.
2. Read `S` words. Require that `4 × S` plus the sum of their `stored_len` equals the chunk's
   `stored_len`.
3. Decode each stream to exactly its raw length.
4. Reverse the layout: place each stream's bytes back at their positions in an `R`-byte buffer.
5. If the chain has `rotl1(W)`, rotate every `W`-byte little-endian element right by 1 bit.
6. If the chain has a delta step, the result so far is the residual: combine it with the base bytes of
   section 7a.
7. The result is the chunk's canonical bytes. Unless verification was disabled by the caller, require
   that their XXH3 equals `xxh3_decoded`. The default is to verify.

### 6.3 Transforms

Encoding is the reverse of section 6.2. Start from the canonical bytes.

**delta_xor(W)**: given the base bytes `b` of section 7a, every byte `t` of the canonical bytes
becomes `t XOR b` at the same position. The result is the residual, the input of the steps below.
`W` does not change the operation; it only enters the chain block.

**rotl1(W)**: every consecutive `W`-byte group is read as a little-endian unsigned integer, rotated
left by one bit, and written back in place. For floats this moves the sign bit below the mantissa and
aligns the exponent with a byte boundary.

**whole**: `S = 1`. The stream is the input.

**planes(w)**: `n = R / w`, `S = w`, every stream has `n` bytes. Byte `i` of stream `j` is byte
`i × w + j` of the input. Stream 0 is the least significant byte of little-endian elements.

**fields(t)**: the input is a sequence of `m = R / B` blocks of `B` bytes. A table lists fields as
`(offset, length)` within a block. `S` is the number of fields. Stream `k` has `m × length(k)` bytes
and is the concatenation, over all blocks in order, of bytes `offset(k)..offset(k) + length(k)`.

| t | Layout | B | S | Fields `(offset, length)`, in stream order |
|---:|---|---:|---:|---|
| 1 | Q8_0 | 34 | 3 | (0,1) (1,1) (2,32) |
| 2 | Q4_K | 144 | 6 | (0,1) (1,1) (2,1) (3,1) (4,12) (16,128) |
| 3 | Q5_K | 176 | 7 | (0,1) (1,1) (2,1) (3,1) (4,12) (16,32) (48,128) |
| 4 | Q6_K | 210 | 5 | (0,128) (128,64) (192,16) (208,1) (209,1) |

The tables follow the block structs of ggml's `ggml-common.h`, with each `ggml_half` split into its
two bytes. An unknown table id is rejected.

## 7. Widening views

A view is a property of an extent, not a blob step. The blob holds narrow canonical bytes, and the
extent presents them widened. Identity (`view = 0`) has `virt_len` equal to the blob range length.
Both views double the length: the blob range is `blob_offset..blob_offset + virt_len / 2`.

For both views `blob_offset` and `virt_len / 2` are multiples of 2 and `virt_len` is a multiple of 4.
There is no alignment requirement on `virt_offset`. Elements are little-endian. Each input element
`x` (`u16`) becomes one output element (`u32`), and the definition is by integer operations, not by
hardware behaviour.

**0x0302, bf16 to f32:** `x << 16`.

**0x0301, f16 to f32:** let `s = x >> 15`, `e = (x >> 10) & 31`, `f = x & 1023`.

| Case | Result |
|---|---|
| `e = 0`, `f = 0` | `s << 31` |
| `e = 0`, `f ≠ 0`, `p` = index of the highest set bit of `f` | `(s << 31) \| ((p + 103) << 23) \| ((f − (1 << p)) << (23 − p))` |
| `1 ≤ e ≤ 30` | `(s << 31) \| ((e + 112) << 23) \| (f << 13)` |
| `e = 31` | `(s << 31) \| (255 << 23) \| (f << 13)` |

This is the exact-value conversion. NaN payloads are shifted left by 13 and the signalling bit is
preserved, so the view is injective. A writer MAY use a view on an extent only if every element of
that extent round-trips bitwise, that is if the source bytes equal the view applied to the narrowed
bytes.

## 7a. Delta blobs

Version 0.3. A delta blob, of kind 1, stores its canonical bytes, the target, as a residual against a
range of the canonical bytes of another blob, its base. Its chunks are chunks of section 6 whose chain
starts with a delta step. A reader that does not implement delta blobs fails closed three times: at
the header (required feature bit 0), at the section table (`PARM` with the required bit), and at the
blob (kind 1, step 0x0400).

**Parameters.** `params_off` and `params_len` of a kind 1 blob locate 16 bytes in `PARM`:

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 4 | base_blob | index into `BLOB` |
| 4 | 4 | zero | |
| 8 | 8 | base_offset | start in the base's canonical bytes |

**Rules.** These are invariant 13.

- `params_len` is 16, `params_off` is a multiple of 8, and the 16 bytes lie inside `PARM`.
- `base_blob` is lower than the delta blob's own index, and the base has kind 0. So there are no
  cycles and no deltas of deltas, and one pass over `BLOB` checks everything.
- The base's `chunk_size` equals the delta blob's, `base_offset` is a multiple of it, and
  `base_offset + raw_len` is at most the base's `raw_len`.
- The chain of a kind 1 blob has a delta step; the chain of a kind 0 blob has none.
- Required feature bit 0 is set exactly when at least one blob has kind 1, and `PARM` has its required
  bit exactly then (section 4).

**Pairing.** Chunk `i` of a delta blob is combined with chunk `base_offset / chunk_size + i` of its base.
By the rules above, that base chunk exists and its raw length is at least the raw length `R` of chunk
`i`. The **base bytes** of chunk `i` are the first `R` canonical bytes of that base chunk, decoded and,
unless the caller disabled verification, checked against the base chunk's own `xxh3_decoded`.

**Combining.** `delta_xor(W)`: every byte of the target is the residual byte XOR the base byte at the
same position. Further operations would take the ids 0x0401 to 0x04FF.

**Meaning.** Everything section 8 and the extents say about a blob's bytes refers to the target: the
identity `blake3`, `raw_len`, `xxh3_decoded` of each chunk, extent ranges and views. A delta blob is
used by extents like any other blob; its base needs no extent. Decoding a chunk of a delta blob decodes
two chunks, which the decode work of section 4 counts.

## 8. Blob identity and reuse

A blob's identity is BLAKE3 of its canonical bytes: after narrowing, before `rotl1` and layout. So a
widened F32 tensor and its F16 sibling share one blob, and the view on the extent is what tells them
apart. All zero in `BLOB.blake3` means the writer did not compute the identity; such a blob is not
matched against others, and a reader does not use the field to restore. Chunk hashes and the member
hash carry integrity.

Because a chunk's stored bytes depend on nothing else, a blob or a chunk within it may be copied into
another container without recompression. Stored chunk bytes, `xxh3_decoded`, `stored_len`, `raw_len`,
`chunk_size`, `blake3` and the chain's steps are unchanged; only the positions (`file_offset`,
`first_chunk`, `chain_id`) are rebased. A delta blob is copied together with its base, or onto a blob
of the target container with the same canonical bytes, kind and `chunk_size`, and its `PARM` entry is
rewritten with the base's new index.

Extents reference any blob range, so tensors of different source formats share one blob. Skeleton
fragments (the bytes of a source file that are not tensor data) and opaque files are blobs too.

Member content is the concatenation, in extent order, of each extent's bytes: the blob range with the
extent's view applied. Blobs may be shared between members, and so may extents.

## 9. Invariants

A reader MUST reject a file that breaks invariants 1 to 11 or 13, without panicking and before using
the offending value. Each invariant has a reject test in the reference implementation.

1. Header, footer and index hashes are valid. `index_offset + index_stored_len + 64 == file_len`,
   `index_offset ≥ 64`. Magic, `end_magic`, `header_len` and reserved fields hold their values.
2. The decompressed index length equals `index_raw_len`, which is within the cap. Every record array
   satisfies `len == count × size` by checked arithmetic. The section table satisfies section 4.
3. Every string, shape and section reference is in range. Member and tensor names are valid UTF-8.
   Member names are unique in the file, and tensor names are unique within a member. `ndim ≤ 16` and
   `first_tensor + tensor_count` is within `TENS`.
4. Chunk byte ranges lie in `64..index_offset` and are pairwise identical or disjoint. Identical means
   equal `file_offset` and `stored_len`. Bytes of the data region that no chunk references are allowed
   and ignored.
5. `first_chunk + chunk_count` is in range. `chunk_count == ceil(raw_len / chunk_size)`. `chunk_size`
   satisfies section 6.1. `raw_len` is a multiple of the chain block. `chain_id` is in range and the
   chain satisfies section 5.2. Each chunk's `stored_len` is at least `4 × S` and at most `4 × S` plus
   its raw length.
6. Extents of a member are sorted and partition `0..file_len` exactly, with no gaps or overlaps. Each
   has `virt_len ≥ 1`, so a member of length 0 has no extents.
7. An extent's blob range lies in the blob. The view's length ratio is exact and the range is element
   aligned, per section 7. A kind 0 blob has `params_off` and `params_len` zero; kind 1 is invariant
   13.
8. Each tensor range lies in `0..file_len` of its member. `shape_off + ndim` is in range. A dtype's
   element size is not checked against the range. Tensor ranges are never trusted for restore.
9. An unknown required feature, required section, step, view, stream codec or blob kind is rejected.
10. The sum of `file_len` over the members to be decoded is within the caller's budget, checked before
    any data read. In each chunk, the stream word sum plus the prologue equals `stored_len`, every
    stream's `stored_len` is at most its raw length, the decoded length is exact, and the XXH3 of the
    canonical decoded bytes, before any view and for a delta blob after combining, matches.
11. When restoring members into a directory, a member name is rejected unless all of these hold. It is
    not empty and has no NUL byte. It does not start with `/` or `\`. No part of it, split on `/` and
    on `\`, is `..`. It has no `:`, which excludes drive prefixes and alternate streams. A reader never
    opens a path taken from a file except the caller's root joined with a name checked here, and
    nothing is written for a rejected name.
12. Commit protocols. A writer makes a file visible by one of the two protocols below, and neither
    has a way to skip the decode. A file produced in any other way, such as by rewriting its bytes in
    place, is outside this invariant, and an interruption can leave it damaged. A reader never
    recovers on its own: a file whose last 64 bytes do not end a valid layout is rejected under
    invariant 1, whatever older generation it contains (section 9.3).
    - (a) Replace. The writer creates a temporary file in the output directory, decodes the full
      result and compares BLAKE3 against the source hash, syncs the file, and only then renames it
      into place. On failure no final file exists and the temporary file is removed.
    - (b) Append, to a valid file of length `L`. No byte below `L` is written and the header stays as
      it is, so the new index keeps every `BLOB`, `CHNK`, `CHAN` and `PARM` record of the old one (a
      record no member reaches is legal, invariant 4), and an append adds a delta blob only to a file
      whose required bit 0 is already set (invariant 13). The writer, in this order: writes the new
      chunks from `L`; decodes every member of the result, read through the file with the new index
      and footer after it, and compares BLAKE3 against the source hash, before any byte of the new
      index is written; writes the stored index and syncs the file; writes the footer, with
      `generation` one above the old footer's, and syncs the file again. A footer is therefore never
      durable before the index and data it names. On failure before the second sync completes, the
      writer truncates the file to `L`, which restores the old file byte for byte; if that truncation
      fails, it reports that the file holds a valid generation of length `L` followed by trailing
      bytes.
13. Delta blobs satisfy the rules of section 7a: their `PARM` entry, their base, their chain, and the
    agreement of required feature bit 0 and of `PARM`'s required bit with the blobs.

### 9.1 Reading procedure

1. Read the header and check it. Read the footer and check it.
2. Check `index_raw_len` against the cap, read the stored index, decompress it into a buffer of exactly
   `index_raw_len` bytes, and check length and hash.
3. Parse and validate the section table and records: invariants 2 to 9 and 13.
4. Check the decode budget: the first half of invariant 10.
5. To read a member range, find the covering extents by `virt_offset`, map each to a blob range,
   widening as needed, map that to chunks by `chunk_size`, and decode them. Reading part of a chunk
   still decodes the whole chunk, and a chunk of a delta blob also decodes its base chunk.
6. To restore a member, read all of it, and require BLAKE3 of the output to equal `MEMB.blake3`, as
   the final step before the output becomes visible.

### 9.2 Reader memory bound

A reader that reads in batches of at most `threads × chunk` stored bytes, and uses scratch of at most
`threads × chunk × view factor`, with a view factor of 2 for widening, needs at most
`3 × threads × chunk` bytes above the destination buffer and the index. At 12 threads and 1 MiB chunks
this is 36 MiB. The rule on `stored_len` in section 6.1 keeps the bound for hostile files.

A chunk of a delta blob needs two stored chunks and one more canonical chunk, its base's. A batch
that always admits its first decode whole holds at most `threads × chunk + chunk` stored bytes, and
scratch is at most three chunks per thread, so a read that touches delta blobs needs at most
`(4 × threads + 1) × chunk`: 49 MiB at 12 threads and 1 MiB chunks. A read that touches none keeps
the bound above.

Decoding a chunk's single compressed stream directly into the destination is an optimization. It
applies only to a whole chunk with exactly one compressed stream, so a reader MUST also implement the
scratch path, which every other chunk needs. The two produce identical bytes.

### 9.3 Generations, appending and recovery

`generation` in the footer is 1 in a fresh file, and an edit writes the previous value + 1. It is
informational: a reader rejects no value and orders nothing by it.

Appending to a file needs no version change: new data follows the old index, then a new index and a
footer, and the old index and footer become unreferenced bytes of the data region, which invariant 4
allows. Under commit protocol (b) of invariant 12 every earlier footer therefore stays in the file
together with the index and chunks it names, so an interrupted append leaves the last committed
generation intact below the partial tail.

A reader MUST NOT fall back to an older generation. It reads the footer in the last 64 bytes only
(section 9.1) and rejects the file when that footer or its layout is invalid. A tool MAY, on an
explicit request, shorten a file to its latest valid generation by this procedure:

1. If the file is valid as it is, nothing is done.
2. Otherwise, for each offset `p` with `64 ≤ p ≤ file_len − 64` at which `end_magic` ends at `p + 64`,
   from the end of the file backwards: take the prefix `0..p + 64` as the whole file and apply steps 1
   to 4 of section 9.1 to it, with `file_len = p + 64`. The first prefix that passes is the latest
   valid generation.
3. Before shortening, the tool SHOULD decode every member of that prefix and check its BLAKE3 (step 6
   of section 9.1). A prefix can pass step 2 while some of its chunks were overwritten by a write
   outside invariant 12, such as an interrupted rewrite in place. A prefix that fails this check is
   passed over, and the search of step 2 goes on to the next older offset.
4. Truncate the file to `p + 64` and sync it. Nothing below `p + 64` is written.

The tool MUST bound the number of candidates it checks and the bytes it scans, since a hostile file can
hold many self-consistent footers, and MUST report exceeding a bound, or finding no candidate, as a
failure.

## 10. Reserved for later versions

These ids and bits are reserved and rejected. Their semantics are not defined here. Version 0.1
reserved also required bit 0, blob kind 1, step 0x0400 and the contents of `PARM`, which version
0.3 defines for delta blobs (section 7a); a 0.1 or 0.2 reader rejects a `PARM` with the required
bit set and skips it otherwise.

| Reservation | Value |
|---|---|
| Required feature, derived blobs | bit 1 |
| Required feature, multi-volume | bit 2 |
| Blob kind, derived | 2 |
| Step ids, further delta operations | 0x0401 to 0x04FF |
| Step ids, derived | 0x0500 to 0x05FF |

## 11. Writer policy

Nothing in this section is visible to a reader.

| Class | `fast` | `balanced` | `max` |
|---|---|---|---|
| F32, F16, BF16 | planes, per-stream store or h | adds widen views | adds rotl1 trial per blob |
| FP8 | whole, h | same | same |
| Q8_0, Q4_K, Q5_K, Q6_K | store | fields, h | same |
| other ggml block types | skeleton | same | same |
| U8 nibble, unknown integer | store | store | store |
| skeleton, opaque | whole, zstd 3 | same | same |
| identity | file hash only | same | adds BLAKE3 blob dedup |
| delta | none | none | `delta_xor` of the float width on F64, F32, F16, BF16 ranges against a base the caller names, per blob by a trial |

`pack` defaults to `max`, `encode` to `balanced`.

- **Store** in the table means `whole` with stream codec 0.
- **Skeleton** means the bytes get a tensor record but no typed blob: they stay in the skeleton.
- **h** is zstd with parameters that minimize matching: level 1, `windowLog 17`, `hashLog 6`,
  `chainLog 6`, `searchLog 1`, `minMatch 7`, `targetLength 0`, strategy `fast`. zstd has no switch that
  disables its match finder, so this is a choice of parameters and not a format property.
- **Per-stream store or h:** a stream is stored when compression gains less than 0.5%. Under every mode
  a stream whose compressed form is not smaller than the raw stream is stored, as section 6.1 requires.
- **Chunk size** is 1 MiB rounded down to a multiple of the chain block, so Q8_0 uses
  `floor(1 MiB / 34) × 34` = 1 048 560 bytes.
- **Minimum range:** a source range under 4 KiB stays in the skeleton to limit per-blob overhead. It
  remains reachable by tensor name.
- **Identity** in the `file hash only` modes leaves `BLOB.blake3` all zero.
- **Delta** pairs tensors of a file with tensors of a base file the caller names, by name, length and
  dtype. Names are compared without a leading `model.diffusion_model.`, `model.` or `module.`, and
  with a last component `scale` after one ending in `norm` read as `weight`. A paired range under the
  identity view whose base blob is a plain blob of the same chunk size gets a trial on its sample:
  plain, rotated, delta and rotated delta, the smallest wins, in that order on ties, so a tensor is a
  delta blob only where that stores smaller. A tensor equal to its base is a dedup hit, not a delta
  blob. FP8, quantized and integer ranges get no delta: valid, but unmeasured.

## 12. Implementation notes

Informative. The format has no platform dependency. The reference implementation compiles zstd's C
sources, so it needs a C compiler per target, though it exposes no C API of zstd and links no system
library. Plain cross-compilation to macOS fails in that build step; `cargo-zigbuild` is the supported
route, and linking there needs `SDKROOT`.
