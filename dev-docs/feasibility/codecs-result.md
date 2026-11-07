# Codec study: which lossless pipeline per storage class, how much it saves, and how fast it decodes into a caller buffer
## Verdict
Feasible, with a simpler design than the brief: split byte planes, STORE noise planes, and code only the top plane with an entropy-only pass. LZ adds nothing to float weights: zstd-19 equals no-LZ zstd (BF16 32.78 vs 32.80%) at 80x the compress cost.
Decode reaches the memory-copy ceiling (17 GB/s) at 12 threads with 1.7 MiB RSS above the destination. HDD load drops by the full saved share; a CPU bottleneck is not reachable on this HDD.
No separate fast/balanced/max is justified for floats: one pipeline wins all three, plus an optional rotl1 for +0.9 pp on BF16 and +0.3 pp on F32.
Confidence: high for sizes (deterministic, bit-exact roundtrip), medium-high for quiet-run speeds; NVMe is a calculation, not a measurement.

### Final table (Rust prototype, bit-exact by xxh3; quiet-run rows marked q, others ran at load ~8-20, best of 3)
Pipeline notation: `L` store the low planes, `h` = zstd frames with match finder disabled (Huffman literals only), standard zstd decoder.
| Class (data set) | Mode | Pipeline | Saved % | Comp GB/s 12T | Dec GB/s 1T | Dec GB/s 12T |
|---|---|---|---|---:|---:|---:|
| BF16 (klein 9B, 2 GiB, 1 MiB chunk) q | fast + balanced | planes, L, h on hi | 32.79 | 3.38 | 3.57 | 17.2 |
| BF16 q | max | rotl1, planes, L, h | 33.69 | 2.96 | 3.53 | 16.6 |
| F16 (SDXL unet, 2 GiB) q | all three | planes, L, h on hi | 15.60 | 3.45 | 3.27 | 16.5 |
| F32 true (LaMa, AE, MangaOCR, 963 MiB) q | fast + balanced | planes, L×3, h on hi | 16.26 | 3.54 | 4.08 | 16.7 |
| F32 true q | max | rotl1, planes, L×3, h | 16.59 | 3.04 | 4.54 | 15.5 |
| FP8 E4M3 (klein 9B fp8, 512 MiB) | all three | raw bytes, h | 17.96 | 2.94 | 1.85 | 12.3 |
| Q8_0 (1 GiB) | fast | raw, h | 3.69 | 5.81 | 1.78 | 12.2 |
| Q8_0 | max | d_lo,d_hi,qs streams, h | 5.76 | 6.00 | 1.64 | 5.1 |
| Q4_K (512 MiB) | fast | raw, h | 2.27 | 7.32 | 1.82 | 13.1 |
| Q4_K | max | 6 field streams, h | 4.57 | 6.29 | 1.68 | 5.2 |
| Q5_K / Q6_K (512 MiB) | max | field streams, h | 1.59 / 3.03 | 4.08 / 5.24 | 4.18 / 2.94 | 8.3 / 8.0 |
| NVFP4 U8 nibbles (512 MiB) | all | stored | 0.0 | 2.48 | 7.06 | 17.3 |
Reference: plain stored memcpy with first-touch faults 6.89 / 17.0 GB/s q (`raw:2 store`). Old-style BG+zstd-3 on BF16: 28.7%, 1.73 GB/s compress, 2.22 / 8.37 GB/s decode. zstd-19 on the hi plane: BF16 32.78% at 0.038 GB/s, F16 15.56%, F32 16.24%.
Python-only classes (no Rust row): NF4 U8 raw z1 2.6%, nvfp4 F8 scales 50.3%, I8 ONNX 26%, sparse F32 68.6%, F16-upcast F32 57.0%.

### Brief's claims
| Claim | Verdict | Numbers |
|---|---|---|
| FP32 ceiling 20-25% | REFUTED for big models | BG+z19 16.1-16.3%; order-0 16.5-16.6; conditional bound 16.8-17.3. Paddle det 19.5 (bound 20.8), tiny rec 15.0. Exceptions are not codec gains: F16-upcast 57%, sparse 68.6% |
| BF16 30-33% | CONFIRMED | 32.5-33.7% entropy-only across 10 BF16 groups; bound 34.1-34.9; plain BG+z3 is only 28.6-29.9 |
| FP16 lower than FP32 | PARTLY | 15.1-15.6%, about 1 pp below F32. Outlier te9b 26.0% (low plane 6.23 bits) |
| xz not worth it | CONFIRMED, stronger | BG+xz is WORSE than BG+z19: BF16 31.3-32.0 vs 32.5-33.0, F32 15.7-15.9 vs 16.0-16.2; +0.3 pp only on two small models (Paddle det, tiny rec) |
| Bit transposition useless | MOSTLY | Full bitshuffle clearly worse (BF16 26.9 vs 32.8, F32 13.8-14.7 vs 16.0-16.2). rotl1 is not useless: +0.90 pp BF16, +0.33 pp F32, ~0 F16 |
| Two low FP32 bytes are noise | CONFIRMED | 8.00, 8.00, 7.97 bits; top plane 2.76, not ~3 |
| Quantized barely compresses | CONFIRMED for GGUF/NVFP4/NF4; REFUTED for FP8 | Q 0-4.4% raw, 1.5-6.5% with fields; FP8 17.2-18% |
| Delta to a neighbour (section 6 asks; within-tensor form here) | REFUTED | All four variants lose: BF16 25.7-27.5 vs 28.7, F16 11-12 vs 15.5 |

## Evidence
Sample: 25 files, 41 (file, dtype) groups, up to 256 MiB per group as 4 MiB stratified element-aligned chunks, seed 1 (`codecs/out/sample.txt`). All file and output paths below are relative to `codecs/`, the study's scratch directory, which is not kept; the Rust prototype is `prototype-ztp/`.
- Headers: `out/survey.txt`; parsers `src/parse.py` (pickle opcodes via pickletools only, nothing unpickled).
- a, b, c: `out/report-light.txt`, `out/report-heavy.txt`, `out/report-planes.txt`. BF16 hi plane bits/byte: H0 2.717, no-LZ 2.755, z1 2.967, z3 3.402, z9 3.209, z19 2.756, lz4 5.495. z3 is worse than no-LZ because spurious matches on noise cost more than they save. Low plane stays 8.0 everywhere.
- d: fields vs planes in `out/report-light.txt`, `out/report-field2.txt` (BF16 32.78->33.52, Paddle-VL 33.70->34.12, F16 15.53->15.57, F32 16.27->16.52).
- e: `out/report-light.txt` third table. H(E)=2.585 bits, H(E|prev)=2.578, H(E|prev,above)=2.575; H(mantissa top|E)=6.924 vs 6.972; sign exactly 1.000. Scope: chunk 2.585, tensor 2.585, global 2.617 bits; global tables usually cost 0.1-0.2 bits (LaMa best, Paddle det up to 0.8). Bound minus BG+z19: BF16 1.8 pp, F32 0.7 pp, F16 0.1 pp, Paddle det 1.3 pp, all under the 3 pp gate. So no rANS prototype: not worth it. rotl1 already takes half of that gap.
- f: `out/report-light.txt` second table.
- g: layouts from `ggml-common.h` structs (`block_q8_0`, `block_q4_K`, `block_q5_K`, `block_q6_K`) and `gguf` 0.19 `GGML_QUANT_SIZES`; `parse.gguf` asserts sizes against tensor offsets. Field numbers: `out/report-quant.txt`, `out/report-field2.txt`. Bounds (order-1, per field): Q8_0 6.5-7.0, Q4_K 5.7, Q5_K 2.5-2.6, Q6_K 3.5-4.6. Scale stream delta loses (Q8_0 9.84->10.76 MB). fp8r: `F8_E4M3` weights, per-row F32 `weight_scale` (metadata `per_row_fp32`). nvfp4: U8 packed nibbles row-wise, `[3072,1536]` for 3072 columns, `weight_scale` F8 one per 16 elements. Nibble H 3.95 bits, FP8 S 1.00 + E 2.58 + M 2.98 bits; field split of FP8 offers nothing over raw h (17.96% vs field order-0 18.0%).
- h: sampled pages all-zero or constant 0.00% in every float group except sparse dat2 27.15% (44% zero elements) and Paddle det 1.09% zero elements (`out/report-light.txt`).
- Decode methods (BF16 store,z1, 4 MiB, 1T / 12T GB/s, RSS above destination at 12T, `out/report-bench.txt`): inplace 2.66 / 12.7, 1.5 MiB; joint 2.75 / 12.5, 25.5 MiB; scratch+AVX2 2.72 / 9.7, 49.6 MiB; scratch+scalar 2.68 / 9.4, 49.6 MiB; stride passes 1.88 / 8.3, 25.5 MiB. Scalar matters on F32: 2.57 / 8.2 vs joint 3.60 / 13.9.
- Whole-tensor scratch is avoidable. Scratch variants peak at threads × chunk (12 × 4 MiB). In-place: decode the one compressed plane into the tail of the destination chunk, then merge forward with stored planes read from the compressed buffer; the write head stays behind the read head, zero scratch. Needs exactly one compressed plane, which is the normal case. Plane streams cannot be stride-decoded by zstd itself, it needs contiguous output.
- Chunk sweep BF16 store,h (saved %, dec 1T / 12T): 256 KiB 32.785, 3.29 / 16.4; 1 MiB 32.794, 3.42 / 16.6; 4 MiB 32.796, 3.57 / 13.1; 16 MiB 32.797, 3.40 / 12.0. store,z1 loses at 256 KiB: 30.73 vs 31.47. Ratio is flat, so pick 1 MiB: larger chunks run slower at 12T, likely L2 locality (not isolated).
- Disk (`out/disk2.txt`, `src/dread.py`, O_DIRECT, idle check clean): 134 and 175 MB/s; earlier 74-164 MB/s runs overlapped foreign reads. RAID1 serves one sequential stream from one disk (only sdc sectors moved). 17.3 GB BF16 file (18.16 GB) at 175 MB/s: raw 104 s, compressed 70 s, rotl1 69 s. F16/F32: 88 s / 87 s per 18.16 GB.
- Bottleneck: needed raw rate = disk / (1 - saved). HDD: 260 MB/s, 1 thread is 13x over. NVMe ESTIMATE, not measured: 7 GB/s needs 10.4 GB/s, about 4 threads; 3.5 GB/s needs 5.2 GB/s, 2 threads. 18.16 GB: 7 GB/s raw 2.6 s, compressed 1.74 s; 3.5 GB/s raw 5.2 s, compressed 3.5 s. CPU becomes the limit only when decode speed falls below the compressed disk speed scaled up; compressed loads slower than raw only if decode < disk read speed, i.e. 1 thread on NVMe over ~3.5 GB/s.

## Not verified
- NVMe: this machine has no measured figure here. End-to-end with compressed files actually on HDD: nothing was written there by rule, load times are computed from measured reads.
- Prototype decodes from a RAM buffer, so read-side copies, and chunk read queueing, are excluded. It chunks byte ranges of one dtype, not per tensor; the F32 set is concatenated samples (963 MiB, under the 2-4 GiB asked, since real F32 in the roots is scarce); Q5/Q6/FP8 sets 512 MiB.
- Non-q rows ran beside other sessions; 12T figures there are conservative (same 1 MiB pipeline: 16.6 under load vs 17.2 quiet). Field-layout 12T rows use an unoptimized scalar stride path, so 5 GB/s is a floor.
- xz speed not timed. Per-thread compress speed not measured. AVX-512 and non-x86 untested. Whole-file zero/constant tensors not scanned here (inventory study); this is sampled chunks only.
- Order-1 bounds use a 12-bit-per-cell static cost proxy, not a real coder. Q8_0 structure (block max near ±127) not exploited; NVFP4 nibble conditioning on block scale not tested.

## Recommended next step
Spec v0.1 around: per-dtype plane split, per-chunk per-stream store-or-entropy decision (gain under 0.5% stores), 1 MiB chunks, single-compressed-plane in-place merge, no LZ for floats. Keep stored for quantized by default, FP8 raw + h. Add `rotl1` as an optional transform step. Before committing, evaluate a dedicated Huffman/rANS plane codec only if a further ~0.4 pp (H128k 33.2 vs h 32.8) matters; otherwise zstd frames keep the decoder standard. Combine with `inventory` downcast and dedup results first, they exceed anything here (57%).

## Side findings
- Names mislead: `4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.safetensors` is F32; PaddleOCR-VL is BF16, not F32; a community finetune of the klein-9b text encoder is F16; SDXL `unet/diffusion_pytorch_model.safetensors` is F32 holding F16 values (low plane 0 bits).
- `dd iflag=direct` fails here with `IO error: Invalid input` (uutils dd), so `src/dread.py` replaced it. `Krea-2-Turbo/model.safetensors` has 289 extents, its 74 MB/s was partly fragmentation.
- A shared cargo `build-dir` set in the user-level cargo config overrides `CARGO_TARGET_DIR` for intermediates; only the binary moves.
