# Inventory: bytes by class, downcast exactness, dedup and delta potential across two model roots
GB = 10^9 bytes. The two roots are a private model library. Output and script names below refer to
the study's scratch files, which are not kept. "Other" torch class = .pt/.ckpt/.pth/.ts/.pth.tar/.pkl.

## Verdict
- 361.14 GB (336.3 GiB) unique bytes in 546 files: root A 139.01 GB, root B 222.13 GB. 99.55% is tensor payload. Already-narrow data dominates: BF16 43.2% (155.9 GB incl. GGUF), GGUF quantized 30.6% (110.4 GB), packed U8 9.8%, F8 3.8%, F16 6.5%, F32 only 5.7% (20.5 GB).
- **Reclaimable by dedup + exact downcast, no entropy coding: 47.28 GB = 13.09%.** Exact tensor dedup 32.82 GB; SDXL F32 files fully redundant with their fp16 siblings 13.53 GB; F32 narrowing 0.92 GB. Result is exhaustive, not an estimate (see "Scan coverage").
- File-level dedup finds only 0.79 GB. Tensor-level finds 41x more: the largest case (16.62 GB, NF4 single file vs two shards) has identical tensors in different containers.
- Headline beyond the brief: 28.5 GB more is likely recomputable from BF16 siblings (Q8_0 GGUF, fp8, fp8r reproduce bit-exactly on samples). Not included in the 47.28 GB.
- Delta works only for real partial edits (community finetunes); it does nothing for independently trained models (PaddleOCR-VL pair: 3.4%).
- Confidence: high for sections 1-3, medium for delta ratios (sampled), medium for derivation (sampled, but 0 bad blocks).

## Evidence
### 1. Bytes by class (`inventory-classes.csv`, `.txt`; script `inv_parse.py`, `inv_class.py`)
| format | dtype | A GB | B GB | ALL GB | % | files | layout |
|---|---|---:|---:|---:|---:|---:|---|
| safetensors | BF16 | 54.144 | 97.049 | 151.193 | 41.87 | 31 | |
| gguf | Q8_0 | 48.675 | 4.274 | 52.949 | 14.66 | 8 | |
| safetensors | U8 | 2.024 | 33.230 | 35.255 | 9.76 | 4 | **all 4-bit nibble packed**, see below |
| gguf | Q4_K | 14.716 | 10.935 | 25.651 | 7.10 | 7 | |
| safetensors | F16 | 0 | 23.319 | 23.319 | 6.46 | 5 | |
| gguf | Q5_K | 2.406 | 17.284 | 19.691 | 5.45 | 3 | |
| safetensors | F32 | 0.462 | 17.036 | 17.499 | 4.85 | 15 | 13.54 GB is widened F16 |
| safetensors | F8_E4M3 | 5.085 | 8.724 | 13.809 | 3.82 | 3 | scaled, see below |
| gguf | Q6_K | 3.541 | 8.529 | 12.070 | 3.34 | 6 | |
| gguf | BF16 | 3.039 | 1.633 | 4.672 | 1.29 | 7 | |
| torch class | F32 | 1.715 | 0 | 1.715 | 0.47 | 15 | |
| onnx | F32 | 1.254 | 0 | 1.254 | 0.35 | 25 | |
| onnx | F16 | 0.173 | 0 | 0.173 | 0.05 | 1 | |
| gguf | F16 / F32 | 0.159 | 0.003 | 0.162 | 0.04 | 16 | |
| rest | I8,U8,I64,F16,F64 | 0.113 | 0 | 0.113 | 0.03 | | |

Non-tensor overhead: safetensors 2.44 MB (0.001%), GGUF 50.3 MB (0.044%), ONNX 2.60 MB (0.174%), torch class 7.96 MB (0.455%), other files 1,554 MB (of which `.mxr` 1,399 MB, ONNX Runtime MIGraphX compile cache under root A `.cache/`; 445 files). No F64 beyond 3 tiny tensors.

Byte accounting: 19 symlinks skipped (all A -> B, including `flux-2-klein-9b.safetensors` and a community text-encoder finetune), 0 hardlinks, 0 HF `blobs/` dirs, 5 `*.incomplete` excluded (2.158 GB). Extents verified non-overlapping and in range; 0 parse failures. Sum matches `du` (339G incl. incompletes).

Declared dtype does not describe the data:
- **bnb NF4** (`flux.2-dev/transformer/*`, `transformer-nf4-single`, 33.23 GB U8): `.weight` is U8 `[bytes,1]`, flat, losing the shape; 2 nibbles per byte. `.absmax` U8 is nested-quantized scale (1 per 64 elements), plus F32 `.nested_absmax`, `.quant_map`, `.nested_quant_map`, and `.quant_state.bitsandbytes__nf4`, a U8 JSON blob with the true shape. Example: shape [6144,15360] gives 47,185,920 bytes (`inventory-extra.txt`).
- **nvfp4** (`FLUX.3-image-klein/variants/nvfp4`, 2.02 GB U8): U8 `[N,K/2]` packed e2m1, even element in low nibble; `weight_scale` F8_E4M3 `[N,K/16]` (0.253 GB), `weight_scale_2` and `input_scale` F32 scalars. Decode: `e2m1(w) * weight_scale[:, k//16] * weight_scale_2` (`variants/nvfp4/manifest.json`).
- **fp8r** (`variants/fp8r`, 4.83 GB): F8_E4M3 with per-row F32 `weight_scale` `[N]`: `weight.float() * weight_scale[:,None]`.
- **fp8** (`flux-2-klein-9b-fp8.safetensors`, 8.72 GB): F8_E4M3 with scalar F32 `weight_scale`, `input_scale`; `_quantization_metadata` JSON in the header. 0.709 GB of excluded layers stay BF16.
- Widened F32: SDXL unet/TE/TE2 are F16 in F32 tags; Krea model (1.28 of 1.29 GB) and VAE are BF16 in F32 tags. The community text-encoder finetune is BF16 in F16 tags except 2 tensor families (below). Reverse case: `4x_IllustrationJaNai...bf16.safetensors` is named bf16 but is F32, 37% not exact.

### 2. Downcast exactness (`inventory-dedup.txt`, `inventory-downcast.csv`; `inv_sample.py`, `inv_full.py`)
Bit comparison; -0, NaN, denormal safe. 0 NaN in samples. Exact = 100% of elements on a full scan.
| file | tag | GB | exact -> F16 | exact -> BF16 | note |
|---|---|---:|---:|---:|---|
| sdxl unet | F32 | 10.270 | 10.270 (1680/1680) | 0 | hash = fp16 sibling |
| sdxl text_encoder_2 | F32 | 2.779 | 2.779 (517/517) | 0 | hash = fp16 sibling |
| sdxl text_encoder | F32 | 0.492 | 0.492 (196/196) | 0 | hash = fp16 sibling |
| Krea-2-Turbo/model | F32 | 1.286 | 0 | 1.280 (8/174) | |
| Krea-2-Turbo/vae | F32 | 0.508 | 0 | 0.508 (194/194) | |
| Reline 4x...bf16 | F32 | 0.121 | 0.076 | 0.076 | |
| flux1-fill Q8_0 gguf | F32 | 0.034 | 0.001 | 0.034 | |
| community TE finetune | F16 | 16.381 | n/a | 11.550 (327/399) | same width, no saving |
| flux1-fill Q8_0 gguf | F16 | 0.120 | n/a | 0.120 (5/5) | same width, no saving |
All other F32 (LaMa, OCR, VAEs incl. sdxl vae at 0.012% and Qwen 2.1 VAE) are genuine: F16 hit rate 0.00x%, ruled out. True F16 files show the expected 12.5% (1/8) BF16 hit rate; sdxl vae fp16 is a lossy rounding of F32, not a widening. Sample-then-full-scan disproved only 19.4 MB of 27.1 GB survivors.

### 3. Dedup (`inventory-dedup.txt`, `inventory-dedup-groups.csv`, `inventory-wholefile.csv`; `inv_dedup.py`)
Saved: within one file **0.247 GB**, within one model dir **28.414 GB**, global **32.822 GB (9.09%)**; root A alone 7.832, root B alone 21.632, so cross-root only 3.36 GB. By dtype: U8 16.62, BF16 12.61, Q8_0 2.48, F32 1.05. 5,980 groups. Whole-file duplicates 34 groups, 0.790 GB (subset of the above, except 0.039 GB tokenizers/json).
| saved GB | files |
|---:|---|
| 9.999 + 6.624 | `flux.2-dev/transformer-nf4-single` vs `transformer/` shards 1, 2 (100% of the NF4 file) |
| 3.490 | klein-9b vs a community full finetune of it (BF16) |
| 2.835 | klein-9b BF16 layers vs its Q4_K_M and Q8_0 GGUF: excluded layers cross safetensors and GGUF byte-identically |
| 2.705 + 1.227 | flux3 model vs fp8r and nvfp4 excluded layers |
| 2.480 | Qwen3VL-4B Q8_0 vs a community finetune of it, Q8_0 (Krea) |
| 1.019 + 0.324 | flux3 model vs flux3 Q4_K_M, Q8_0 |
| 0.705 | klein-4b transformer vs its GGUFs |
| 0.408 | LaMa three copies |
| 0.176 | PaddleOCR det: ONNX vs Legacy ONNX vs TorchScript (cross-format) |
| 0.168 | klein-4B vae = standalone vae (whole file, A vs B) |
The suspected "same TE/VAE under several dirs" is real but small: VAE 0.17 GB; TE copies are distinct models or differ in quant level.

### 4. Near-duplicates (`inventory-delta.json`, `inventory-delta-te.json`, `inventory-extra.txt`; `inv_delta.py`, `inv_delta_te.py`)
Non-identical tensors only; sample up to 24 tensors (size weighted), 8 MiB mid slice each. Ratios are compressed/raw, zstd-3 with byte grouping. "gain" = changed GB x (alone - best delta).
| pair (base -> target) | identical GB | changed GB | same elements | alone | XOR | sub | gain GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PaddleOCR official -> separately trained variant | 0.00 | 1.917 | 0.4% | 0.779 | 0.753 | 0.752 | 0.05 |
| klein-9b -> community finetune | 4.199 | 13.959 | 5.1% | 0.781 | 0.578 | 0.563 | 3.05 |
| sdxl unet fp16 -> fp32 | 0 | 10.269 | 100% after widening | 0.591 | 0.00003 | 0.00003 | exact, see 2 |
| sdxl vae fp16 -> fp32 | 0 | 0.334 | 0.01% | 0.928 | 0.576 | 0.550 | 0.13 |
| Qwen3VL Q8 -> community finetune Q8 | 2.481 | 1.767 | 54.5% (bytes) | 0.961 | 0.392 | 0.345 | 1.09 |
| klein-4b TE -> flux3 TE, shard 1 | 0 | 4.967 | 0.7% | 0.783 | 0.721 | 0.718 | 0.32 |
Prompt pairs that do not exist as posed:
- **Normal klein-9b TE is not on disk.** The base model's directory holds only configs and `.incomplete` partial shards (BF16). Used read-only as base: of 9 tensors fully in the prefixes, 7 are bit-exact F16 casts of normal BF16, and 2 (`mlp.down_proj`) differ, 6% same. Globally the non-exact tensors are only `mlp.down_proj` (3.62 GB) and `self_attn.o_proj` (1.21 GB). So the finetune touched 29% of bytes. down_proj slice 16 MiB: alone 0.846, XOR 0.545. Gain requires the normal TE, so 0 today.
- **FLUX.2-klein-9B vs -standalone**: no tensor file in the former; a second directory symlinks to the standalone copy. No pair.
- klein-4b vs klein-9b transformers: different naming and sizes, dropped.

### 5. Reclaimable (`inventory-dedup.txt`, last lines)
Global dedup 32.822 + F32 dropped because F16 sibling exists 13.532 + F32 narrowed on unique content 0.922 (Krea 0.894) = **47.276 GB (13.09%)**, no double count. Not included: delta gains (about 4.6 GB before entropy), derivation below.

### Beyond scope: derivation (`inventory-derive.json`, `inventory-derive3.json`, `inv_derive*.py`)
Quantized siblings reproduced from BF16 on disk, bit-exact:
| target | bytes | rule | test | result |
|---|---:|---|---|---|
| klein-9b Q8_0 | 9.27 GB | ggml ref: `d=amax/127`, `id=1/d`, round half away, d as F16 | every Q8_0 tensor, first+last 1 MiB, 6.9M blocks | 0 bad blocks |
| flux3 Q8_0 | 5.67 GB | same | 16.0M blocks | 0 bad blocks |
| fp8r flux3 | 4.83 GB | `x/row_scale` -> e4m3fn | 10 tensors, 42M elements | 100% |
| fp8 klein-9b | 8.72 GB | `x/scale` -> e4m3fn | 12 tensors, 50M elements | 99.99998% (6 elements, likely ties) |

### Scan coverage
Full: all headers; all extents <=64 KiB; 5,987 extents (94.5 GB) fully hashed; every tensor surviving a sample fully rescanned. Sampled: one 64 KiB mid window for each larger tensor (91.3M of 5.13G F32 elements, 53.7M of 11.8G F16 elements). That cannot miss a result: equal tensors share their mid window and one mismatch rules out exactness. Unique-length non-F32/F16 tensors were not read, since they cannot duplicate. Overall read about 95 GB plus 1.1 GB, in about 35 min with another job reading the same disk.

## Not verified
- Tensors duplicated across different tensor boundaries or sub-tensor chunks: not searched; only whole-extent identity.
- Cross-dtype join covers F32 vs hashed F16/BF16 only. Krea F32 vs its BF16 (0.89 GB counted as narrowing) may reclaim up to 0.89 GB more.
- Derivation of Q4_K/Q5_K/Q6_K (iterative quantizer), klein-4b Q8_0 (names differ), TE Q8_0 (llama names, possible permutes), nvfp4, and full-tensor middles. Klein-9b Q4_K_M mixes quant levels. Source for Fill Q8_0, Krea TE and mistral is not on disk.
- Delta ratios are sample-based; 24 tensors, no CI computed. Pair list is manual, other related pairs may exist.
- `.mxr` internals not parsed; presumably compiled forms of the ONNX models, so regenerable cache, not checked.

## Recommended next step
Design the store around three lossless layers, in this order, and measure them on the codec study's numbers: (1) content-addressed tensors independent of container; (2) canonical narrowing with a typed "widened from F16/BF16" transform; (3) deterministic "derived from source + quantizer version + exception list" for Q8_0 and fp8. Then run a full-tensor derivation pass before committing to (3), and extend it to Q4_K with ggml sources.

## Side findings
- 2.158 GB of abandoned `.incomplete` downloads in a model directory's `.cache`.
- `flux.2-dev/transformer-nf4-single` is a 16.6 GB exact tensor copy of the shards; 1.4 GB `.mxr` cache; two duplicate ONNX/Torch directories are byte copies.
- NF4 files lose shape in the tensor table; any dtype-aware codec must read the JSON quant_state blob.
