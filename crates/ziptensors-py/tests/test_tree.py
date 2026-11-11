"""File: tests/test_tree.py
Purpose: contract tests of `compress_tree` and `decompress_tree` and their report classes.

Main responsibilities: a round trip of a small tree in a temporary directory (weight files become
containers and come back byte for byte, other files are untouched), the dry run, `keep`, the
symbolic-link warning, a refused collision, and a per-file failure as an exception object.

Key functions: `make_tree`.

Implementation notes: data is generated from seeds under pytest's `tmp_path` with the helpers of
`test_ziptensors.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_ziptensors import compressible_bytes, make_safetensors, seeded_bytes
import ziptensors
from ziptensors import Container, InputError, ZiptensorsError

CODE_TREE_CONFLICT = 25


def make_tree(root: Path) -> dict[Path, bytes]:
    """A directory with two weight files (one nested), a config and a text file. Returns every
    file's path and content."""
    (root / "unet").mkdir(parents=True)
    make_safetensors(root / "model.safetensors", {"w": ("F32", [4096], compressible_bytes(1, 16384))})
    make_safetensors(root / "unet" / "unet.safetensors", {"b": ("F32", [256], seeded_bytes(2, 1024))})
    (root / "config.json").write_text('{"hidden": 8}', encoding="utf-8")
    (root / "unet" / "README.txt").write_text("not a weight file", encoding="utf-8")
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def by_name(report: ziptensors.TreeReport) -> dict[str, ziptensors.FileOutcome]:
    """The outcomes of `report` by file name."""
    return {outcome.path.name: outcome for outcome in report.files}


def test_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    originals = make_tree(root)

    report = ziptensors.compress_tree(root, threads=1)
    assert not report.dry_run and not report.has_failures
    assert report.root == root
    outcomes = by_name(report)
    assert outcomes["model.safetensors"].status == "compressed"
    assert outcomes["model.safetensors"].counterpart == root / "model.ziptensors"
    assert outcomes["model.safetensors"].removed == root / "model.safetensors"
    assert outcomes["model.safetensors"].error is None
    assert outcomes["config.json"].status == "not_weight"
    assert report.totals.compressed == 2
    assert report.totals.original_bytes == (
        len(originals[root / "model.safetensors"]) + len(originals[root / "unet" / "unet.safetensors"])
    )
    assert not (root / "model.safetensors").exists()
    with Container(root / "unet" / "unet.ziptensors") as container:
        assert [info.name for info in container.members] == ["unet.safetensors"]
    assert (root / "config.json").read_bytes() == originals[root / "config.json"]

    back = ziptensors.decompress_tree(root, threads=1)
    assert not back.has_failures
    assert back.totals.restored == 2
    # Decompress sums the bytes it brought back and the containers it read.
    assert back.totals.original_bytes == report.totals.original_bytes
    assert back.totals.container_bytes == report.totals.container_bytes
    assert by_name(back)["model.ziptensors"].status == "restored"
    restored = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert restored == originals


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    originals = make_tree(root)
    report = ziptensors.compress_tree(root, dry_run=True)
    assert report.dry_run
    planned = by_name(report)["model.safetensors"]
    assert (planned.status, planned.reason) == ("planned", "compress")
    assert report.totals.planned == 2
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == originals


def test_keep_and_rerun(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    make_tree(root)
    ziptensors.compress_tree(root, keep=True, threads=1)
    assert (root / "model.safetensors").exists() and (root / "model.ziptensors").exists()
    again = ziptensors.compress_tree(root, keep=True, threads=1)
    assert by_name(again)["model.safetensors"].status == "pair_verified"
    back = ziptensors.decompress_tree(root, keep=True, threads=1)
    assert by_name(back)["model.ziptensors"].status == "pair_verified"
    assert (root / "model.ziptensors").exists()


def test_symlinks_are_skipped_with_a_warning(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    make_tree(root)
    try:
        (root / "alias.safetensors").symlink_to(root / "model.safetensors")
    except OSError as error:
        pytest.skip(f"symbolic links are not available here: {error}")
    report = ziptensors.compress_tree(root, threads=1)
    link = by_name(report)["alias.safetensors"]
    assert (link.status, link.reason) == ("skipped", "symlink")
    assert report.totals.skipped == 1
    assert len(report.warnings) == 1 and "symbolic links" in report.warnings[0]


def test_collision_refuses_the_run(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    make_tree(root)
    # `model.gguf` would be compressed to `model.ziptensors` too.
    make_safetensors(root / "model.gguf", {"w": ("F32", [4], bytes(16))})
    with pytest.raises(InputError) as caught:
        ziptensors.compress_tree(root, threads=1)
    assert caught.value.code == CODE_TREE_CONFLICT
    assert not list(root.rglob("*.ziptensors"))


def test_a_failed_file_carries_its_exception(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    make_tree(root)
    ziptensors.compress_tree(root, threads=1)
    # A different file where the container restores to fails that container without `force`.
    (root / "model.safetensors").write_bytes(b"something else")
    report = ziptensors.decompress_tree(root, threads=1)
    assert report.has_failures and report.totals.failed == 1
    failed = by_name(report)["model.ziptensors"]
    assert failed.status == "failed"
    assert isinstance(failed.error, ZiptensorsError)
    assert failed.reason == str(failed.error)
    assert (root / "model.safetensors").read_bytes() == b"something else"
    forced = ziptensors.decompress_tree(root, force=True, threads=1)
    assert not forced.has_failures
