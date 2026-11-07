# Delta blobs (format 0.3) on a real finetune pair

Measured 2026-11-07 with the release CLI of the delta work (`ziptensors pack --delta-base`) and the
Python module of the same tree for decode timing. GB is 10^9 bytes. 12 logical CPUs, 60 GiB RAM, the
containers on a two-disk HDD RAID 1. Pair: base FLUX.2 klein 9B (BF16 safetensors), finetune a
community full finetune of FLUX.2 klein 9B (BF16) in the same directory, 18.157 GB each, all BF16,
201 tensors.

## Method

- (a) `pack a <base> <finetune>`, (b) `pack b <base> <finetune> --delta-base <base>`, both `max`
  (the default), default threads, `nice -n 19 ionice -c3`, `/usr/bin/time -v` for wall time and peak RSS.
- `restore b --all`, then `cmp` of both restored members against the sources.
- Decode: `Member.read_at` of the finetune member into a 64 MiB buffer over the whole member, and 200
  reads of 1 MiB at seeded random offsets (the same offsets for a and b), `threads=12`. `fincore` before
  each run; `/proc/self/io` `read_bytes` for what came from the disk.
- In-RAM control: 121 paired tensors (4.0 GB of each file) copied to `/dev/shm`, packed both ways
  there, decoded the same way. This removes the disk and leaves the CPU cost.
- The machine was shared: another benchmark held about one core throughout, and a foreign
  writer kept the RAID busy (I/O pressure 30-45 %). Disk timings carry that noise; the spread of
  repeated runs is given.

## Size

| | (a) no base | (b) `--delta-base` | planner prediction |
|---|---:|---:|---:|
| container | 21.344 GB (41.22 % saved) | 19.160 GB (47.23 % saved) | |
| base member stored | 12.069 GB | 12.069 GB | 12.068 GB (alone) |
| finetune member stored | 9.275 GB | 7.090 GB | 9.275 GB / 7.090 GB |
| finetune: dedup saved (109 identical tensors) | 4.198 GB raw | 4.198 GB raw | 4.199 GB |
| delta blobs | 0 | 92 | 92 changed tensors |
| gain of delta beyond dedup | | 2.184 GB | 2.185 GB |

The writer's 64 KiB sample trial chose delta for all 92 changed tensors and lost nothing against the
planner's whole-tensor probe (best rotation per tensor). Restore of (b): both members `cmp`-identical.

## Encode

| | wall | base added | finetune added | verify | peak RSS | CPU user |
|---|---:|---:|---:|---:|---:|---:|
| (a) | 15:20 | 5:40 | 4:13 | 5:27 | 52.9 MB | 58 s |
| (b) | 17:03 | 5:17 | 8:10 | 3:36 | 67.2 MB | 73 s |

Adding the finetune against its base takes about twice as long: the reader also reads the 14 GB of
paired base tensors from the base file, interleaved with the finetune, on the same disk. CPU averages
under 0.1 of a core over the wall time; the write is disk-bound. RSS stays below the 9 × threads × chunk
bound (108 MiB at 12 threads) plus the index.

`restore b --all` (36.3 GB written): 10:16, peak RSS 59 MB.

## Decode of the finetune member

Page cache held at most about 4 GB of either container between runs (`fincore`), so every disk run
below is effectively cold; runs meant as "warm" repeats were not warm and are listed as repeats.

| full member, 18.16 GB | runs (s) | MB/s | CPU user | stored bytes behind it |
|---|---|---:|---:|---:|
| (a) threads 12 | 110, 107, 148 | 123-170 | 10 s | 12.07 GB |
| (b) threads 12 | 330, 253, 333 | 55-72 | 23 s | 19.16 GB (7.09 delta + 12.07 base) |
| (b) threads 24 | 273 | 66 | 22 s | |

| 200 random 1 MiB reads | runs (s) | read from disk (last three runs) |
|---|---|---:|
| (a) | 4.4, 3.9, 3.7, 4.2, 5.7 | 278-281 MB |
| (b) | 58.9, 16.6, 17.7, 21.0, 29.5 | 813-832 MB |

The first two runs of each row used `ionice -c3`, the last three `ionice -c2 -n7` with new seeds and
the I/O counter.

In RAM (`/dev/shm`, 4.0 GB finetune subset, two runs each, both within 3 %):

| | threads 12 | threads 24 |
|---|---:|---:|
| (a) | 10.2 GB/s, 1.8 s CPU | 7.9 GB/s |
| (b) | 3.6 GB/s, 5.5 s CPU (about 5 cores busy) | 3.4 GB/s |

## Reading of the numbers: the threads/2 question

A delta read batch of threads × chunk holds about threads / 2 delta jobs, so delta decode uses about
half the pool. Here that is not what limits a read:

- On this HDD a full delta member decodes at 55-72 MB/s with 7-10 % CPU: disk-bound, by seeks between
  the delta blobs and their base, and by reading 19.2 GB instead of 12.1 GB. 24 threads (twice the jobs
  per batch and twice the run length per read) gave 273 s, inside the 253-333 s spread of 12 threads.
- In RAM the delta decode runs at 3.6 GB/s, about 50 times the disk rate, and 24 threads do not raise
  it, so even there the job count per batch is not the limit.

Recommendation: keep the agreed (4t+1) × chunk bound and the batch shape. Work on delta read speed
belongs to I/O: fewer seeks between a delta blob and its base, and the random-read amplification below.

Follow-up (2026-11-07): a random 1 MiB read of (b) took about 4.1 MB from the disk where its two delta
chunks and two base chunks store about 2.3 MB. The reader requested exactly those bytes, but as four
preads (base 0, base 1, delta 0, delta 1); Linux took each second read of an adjacent pair for a
sequential stream and read ahead 1-2 MiB (`read_ahead_kb` 8192 on this RAID). Replaying the pattern on
the same disk: 4.1-5.4 MB with four preads, 2.31-2.47 MB with the same bytes as two. The reader now joins
chunks of a batch that touch in the file into one read (two preads per such read on a synthetic pair);
the real pair was not re-measured, and the 4-5x time ratio was not explained beyond the extra traffic.
