# ziptensors roadmap: layers beyond v0.1

Candidate work for later versions. Not commitments; each item starts with the experiment that can kill
it. Evidence is from `dev-docs/feasibility/inventory-result.md` (sections 4 and "Beyond scope:
derivation"), sampled unless stated.

Why these two: for one standalone BF16, F16 or F32 model v0.1 is within about 1-3 pp of practical
limits (mantissa bytes measure 7.95-8.00 bits of entropy per 8). The remaining large gains are between
models of one collection, which is what both items exploit.

## 1. Derived blobs: quantized files rebuilt from their source

**Idea.** A Q8_0 GGUF or an FP8 safetensors made from a BF16 model held in the same container is
stored as a reference to the source tensors plus the quantization rule, with a residual for any block
the rule does not reproduce. Restore runs the rule and must still prove the whole-file BLAKE3.

**Evidence.**

| target | bytes | rule | result |
|---|---:|---|---|
| klein-9b Q8_0 | 9.27 GB | ggml reference: `d = amax/127`, round half away, `d` as F16 | 0 bad blocks of 6.9M |
| flux3 Q8_0 | 5.67 GB | same | 0 bad blocks of 16.0M |
| fp8r flux3 | 4.83 GB | `x / row_scale` to e4m3fn | 100% of 42M elements |
| fp8 klein-9b | 8.72 GB | `x / scale` to e4m3fn | 99.99998%, 6 elements differ (likely ties) |

Potential on the measured collection: about 25 GB, unconfirmed on whole files.

**Format.** Reserved in v0.1 without semantics: required-feature bit 1, blob kind 2, step ids
0x0500-0x05FF (`spec/ziptensors-format.md`). Older readers fail closed on them.

**Open questions and risks.**
- Bit-exact reproduction depends on the producer's arithmetic: rounding mode, F16 conversion of scales,
  FMA, thread order. Each rule is a versioned step id, never "whatever the current code does".
- Q4_K/Q5_K/Q6_K use an iterative quantizer, and imatrix files make them depend on data not on disk:
  expect residual-only gains there, or none.
- Matching a target tensor to its source needs name maps (GGUF vs diffusers names, permuted Q/K).
- Decode cost: the source tensor must be decoded to rebuild the target, so random access to a derived
  member reads more than its own bytes; the decode budget must count it.

**First experiment.** Whole-file reproduction of every Q8_0 and FP8 file that has a source on disk,
with the residual size per file.

## 2. Delta blobs: finetunes against their base

**Implemented as format 0.3** (spec section 7a, writer policy in section 11): a tensor of a finetune
is stored as `delta_xor` of the float width against the same tensor of a base the caller names, then
byte-plane coded; depth 1, base and delta chunks of equal size. Entry points: `PackBuilder::add_file_delta`,
`ziptensors pack --delta-base`, `ziptensors.pack(..., delta_base=)`. On FLUX.2 klein 9B and a community
full finetune of it (BF16) the finetune stores in 7.09 GB instead of 9.27 GB after dedup (2.18 GB more saved, 92 delta blobs), as
predicted; full-member decode on HDD is 2-3 times slower and disk-bound
(`dev-docs/benchmarks/delta-v0.3.md`).

**Evidence that led to it** (compressed/raw, zstd-3 with byte grouping, 24 sampled tensors per pair):

| pair (base -> target) | identical GB | changed GB | alone | XOR | sub | gain GB |
|---|---:|---:|---:|---:|---:|---:|
| klein-9b -> community finetune (BF16) | 4.199 | 13.959 | 0.781 | 0.578 | 0.563 | 3.05 |
| Qwen3VL Q8 -> community finetune Q8 | 2.481 | 1.767 | 0.961 | 0.392 | 0.345 | 1.09 |
| sdxl vae fp16 -> fp32 | 0 | 0.334 | 0.928 | 0.576 | 0.550 | 0.13 |
| klein-4b TE -> flux3 TE, shard 1 | 0 | 4.967 | 0.783 | 0.721 | 0.718 | 0.32 |
| PaddleOCR official -> separately trained variant | 0 | 1.917 | 0.779 | 0.753 | 0.752 | 0.05 |

Independently trained models gain nothing (PaddleOCR pair), so delta stays opt-in per pair.

**Open.**
- Automatic base selection: matching by tensor names, shapes and a sampled similarity test, so a
  directory of finetunes needs no `--delta-base`. Today the caller names one base per pack.
- Other residual operations (step ids 0x0401-0x04FF): wrapping or order-preserving subtraction measured
  worse than XOR on the full klein pair (7.37 and 7.39 GB against 7.09), but the sampled Qwen3VL Q8
  pair favours subtraction; quantized, FP8 and integer data get no delta yet (unmeasured).
- Pairs across dtypes (the sdxl VAE fp16 -> fp32 row) need a delta under a widening view; today delta
  applies under the identity view only.
- Read speed on HDD: a delta member also reads its base, interleaved; a random 1 MiB read moved about
  three times the bytes of a plain one. Placement or read planning, not decode parallelism, is the lever
  (benchmark above).
- Merge brings a delta blob's base along; removing or rewriting a base inside a container must still
  be refused or materialize its dependants once containers can be edited.

## 3. Rank coding of quantized tensors against a reference (generalizes section 1)

**Idea.** Instead of reproducing a quantizer, use a reference tensor of the same weights (BF16, or a
larger quant of the same model) as side information. For each block of the target quant, store its
block parameters (scales, mins, sign bits) verbatim; for each code, enumerate the type's candidate codes,
order them by the distance of their dequantized value to the reference value, and store the rank of the
true code. Ranks are mostly 0 and entropy-code to a fraction of a bit per weight. Decode repeats the
ordering and picks the code by rank. Section 1's exact rules are the special case where every rank is 0.

**Why it is universal.** It needs no knowledge of how the file was made (imatrix, iterative scale search,
llama.cpp version): per GGUF type only a dequantizer and a candidate enumerator.

| Family | Candidate | Enumeration |
|---|---|---|
| Q4_0..Q8_1, K-quants | integer on the block's uniform grid | rounding, then +-1, +-2 |
| IQ4_NL, IQ4_XS | entry of the 16-value non-linear table | nearest table entries |
| IQ1_*, IQ2_*, IQ3_* | lattice codebook vector (256-2048 entries, groups of 8) plus signs | nearest vectors |
| TQ1_0, TQ2_0 | ternary value | trivial |

An unknown type or an unmatched tensor is stored as today.

**Paper estimate, unmeasured.** Q6_K's grid is 4x finer than Q4_K's, so only weights near a Q4 rounding
boundary are ambiguous: Q4_K_M at about 0.5-1 bit per weight instead of 4.5. A set of BF16 + Q8_0 + Q6_K +
Q5_K_M + Q4_K_M of an 8B model (~42 GB) might shrink from ~35 GB today to ~13-16 GB. IQ1/IQ2 gain less, since
an aggressive search picks their codes. A second stage may predict block scales from the reference too.

**Format.** Fits the reserved derived-blob kind; one versioned step id per (type, enumerator), so a
llama.cpp change never alters old files. Older readers fail closed.

**Open questions and risks.**
- Determinism: encoder and decoder must order candidates identically on every platform and SIMD level,
  so distances use integer arithmetic or f32 with a fixed operation order and no FMA. The main risk.
- Decode cost: the reference tensor must be decoded first; lattice IQ types also search their codebook.
  The decode budget must count both.
- Order: the most precise variant is the reference; coding a larger quant against a smaller one gains
  little.
- Tensor matching by name and shape across files, including GGUF vs diffusers names and permuted Q/K.

**First experiment.** One small model (3-7B) with BF16 and as many GGUF types as possible from one
producer: the rank distribution and entropy per type against BF16, and Q4_K_M against Q6_K.

## 4. Grouped reads in the CPU load path of `ziptensors.torch`

Measured 2026-11-07 on a synthetic 3.2 GB BF16 file, one fresh process per sample: `load_state_dict` to
cpu runs about 17% below decoding into a reused buffer. About half is the kernel zeroing fresh pages
(any loader returning new memory pays it); the other half is one `read_at` per tensor, which ends every
call on a partial batch and decodes a chunk shared by two neighbouring tensors twice. A Python
`Member.read_many` binding over core `Member::read_many` (rejecting overlapping buffers before it builds
mutable slices) and one call per group of tensors of at least 64 MiB would recover an estimated 5-9%.
The memory promise of the loader then holds per group, not per tensor. Optional; not a defect.

## 5. Overlapping reads with decode

The calling thread reads a batch, then waits for its decode, so 12 threads decode a member only
1.5-1.9x faster than one (Linux and Windows, `dev-docs/benchmarks/windows-v0.3.md`). Reading batch n+1
while batch n decodes would lift that, at the cost of a second batch buffer in the read bound. On
Windows the pool dispatch also costs 0.1-0.2 ms per random 1 MiB read (1 thread is faster than 12
there); decoding a read of one or two chunks inline would remove it.
