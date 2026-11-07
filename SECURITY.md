# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**
(<https://github.com/Vasyanator/ziptensors/security/advisories/new>). Do not open a public issue for
a vulnerability.

Include the version or commit, the platform, and a way to reproduce, ideally a small container file
or a script that builds one. You will get an answer in the advisory thread; a fix is published
together with a GitHub security advisory.

## Supported versions

Fixes go into the latest release. The project is below 1.0, so there are no maintained older branches.

## Scope

ziptensors reads files that may come from anywhere, so the main attack surface is the **reader of
untrusted containers**: the header, footer and index parsers, the chunk decoder, delta blobs, the
read cache, `restore --all` and `extract` (member names joined to a directory), the filesystem mount,
and the C and Python bindings that hand buffers to it. In scope, for example:

- a crash, panic, hang or out-of-bounds access caused by a container file;
- memory or decode work beyond the budgets set in `OpenOptions` (`max_index_bytes`,
  `max_total_decoded`, `max_decode_amplification`) or the documented read memory bound;
- a read that returns wrong bytes without an error while chunk verification is on, for a container
  that was not deliberately forged (see below);
- a member name that makes `restore --all` or the mount write or show anything outside the target;
- the profile parsers (safetensors, GGUF, ONNX, PyTorch zip) misbehaving on a hostile source file
  during `encode` or `pack`, beyond a lower compression ratio.

Not in scope:

- resource use of a container opened **without** `max_total_decoded`: the library default is no
  budget, and an untrusted file needs one (`--max-total-decoded` on the command line,
  `max_total_decoded=` in Python);
- a forged container whose hashes were recomputed (see below);
- what another program does with restored bytes, for example unpickling a restored PyTorch
  checkpoint. ziptensors never executes or unpickles content.

## What the hashes guarantee

Each chunk carries an XXH3-64 of its decoded bytes, and each member a BLAKE3 hash of the whole
original file. They **detect accidental damage**: a corrupted chunk fails its read, and `verify`,
`restore` and `extract` compare the whole member against its BLAKE3.

They **do not authenticate** a file. There is no signature or key, and the hashes are stored in the
same file, so anyone who can change a container can recompute every hash and the forged file passes
every check. Member names and tensor tables are not covered by the member hashes at all. Trust a
container only as far as you trust its source, as you would the original model file.
