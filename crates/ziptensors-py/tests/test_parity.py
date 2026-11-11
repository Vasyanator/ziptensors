"""File: tests/test_parity.py
Purpose: contract tests of the module's surface beyond reading and packing: open options, closing,
`layer_report`, `restore_all`, `read_many`, `PackBuilder`, directory packing, `edit`, `compact`,
`recover`, `enable_stderr_logging` and `__version__`.

Main responsibilities: call every one of these once against generated data, check what each
promises on failure (nothing written, nothing left behind), and check that the long ones release
the GIL.

Key functions: `make_pack`, `members_of`.

Implementation notes: data is generated from seeds under pytest's `tmp_path`, with the helpers of
`test_ziptensors.py`. Every `Container` on a file is closed before an edit of it, the rule on
Windows. numpy is optional and its test skips without it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import sys

import pytest

from test_ziptensors import (  # noqa: F401 - `fixture_slow` is a fixture, used by its name
    compressible_bytes,
    fixture_slow,
    make_safetensors,
    seeded_bytes,
    stall_during,
    SLOW_SECONDS,
)
import ziptensors
from ziptensors import (
    AddOp,
    Container,
    CorruptionError,
    FormatError,
    InputError,
    NotFoundError,
    PackBuilder,
    RemoveOp,
    ReplaceOp,
    ZiptensorsIoError,
)

CODE_EDIT_REFUSED = 22
CODE_UNRECOVERABLE = 24


def make_files(tmp_path: Path, count: int, seed: int = 10) -> list[Path]:
    """`count` small safetensors files `m0.safetensors`, ... with distinct content."""
    files = []
    for at in range(count):
        data = compressible_bytes(seed + at, 32 * 1024)
        path = make_safetensors(tmp_path / f"m{at}.safetensors", {"w": ("F32", [8192], data)})
        files.append(path)
    return files


def make_pack(tmp_path: Path, files: list[Path], name: str = "set.ziptensors") -> Path:
    """A container of `files` under their file names."""
    output = tmp_path / name
    ziptensors.pack(files, output, threads=1)
    return output


def members_of(path: Path) -> dict[str, bytes]:
    """Every member of the container at `path`, name to content, closing it afterwards."""
    with Container(path, threads=1) as container:
        return {
            info.name: container.member(info.name).read(0, info.len)
            for info in container.members
        }


def test_version_is_the_crate_version() -> None:
    assert isinstance(ziptensors.__version__, str)
    assert ziptensors.__version__.count(".") == 2


def test_open_options(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    output = make_pack(tmp_path, files)
    for verify in ("chunk", "off"):
        with Container(
            output,
            threads=1,
            verify=verify,
            max_index_bytes=1 << 20,
            max_decode_amplification=64,
            max_total_decoded=1 << 30,
        ) as container:
            member = container.member()
            assert member.read(0, member.len) == files[0].read_bytes()
    with pytest.raises(ValueError, match="unknown verify"):
        Container(output, verify="full")
    with pytest.raises(TypeError):
        Container(output, 1)  # the options are keyword-only


def test_zero_budget_still_means_zero(tmp_path: Path) -> None:
    output = make_pack(tmp_path, make_files(tmp_path, 1))
    with pytest.raises(ziptensors.BudgetError):
        Container(output, max_total_decoded=0)


def test_close_and_context_manager(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    output = make_pack(tmp_path, files)
    with Container(output) as container:
        member = container.member()
    with pytest.raises(ValueError, match="closed"):
        container.member()
    with pytest.raises(ValueError, match="closed"):
        _ = container.members
    container.close()  # idempotent
    # A member taken before the close keeps reading.
    assert member.read(0, member.len) == files[0].read_bytes()


def test_layer_report_matches_the_encode(tmp_path: Path) -> None:
    files = make_files(tmp_path, 2)
    output = tmp_path / "set.ziptensors"
    encoded = ziptensors.pack(files, output, threads=1)
    with Container(output) as container:
        total, classes = container.layer_report()
    assert total.raw == encoded.total.raw
    assert total.stored == encoded.total.stored
    assert [entry.dtype for entry in classes] == [entry.dtype for entry in encoded.classes]


def test_restore_all(tmp_path: Path) -> None:
    files = make_files(tmp_path, 2)
    output = tmp_path / "set.ziptensors"
    ziptensors.pack([files[0], (files[1], "sub/deep/b.safetensors")], output, threads=1)
    target = tmp_path / "out"
    with Container(output) as container:
        report = container.restore_all(target)
    assert report.members == 2
    assert report.bytes == sum(path.stat().st_size for path in files)
    assert (target / files[0].name).read_bytes() == files[0].read_bytes()
    assert (target / "sub/deep/b.safetensors").read_bytes() == files[1].read_bytes()


def test_restore_all_into_a_file_fails_cleanly(tmp_path: Path) -> None:
    """A container with an unsafe name cannot be made through this API (every writer refuses
    one), so the hostile-name case is the core's test; here a directory that cannot be made."""
    output = make_pack(tmp_path, make_files(tmp_path, 1))
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")
    with Container(output) as container, pytest.raises(ZiptensorsIoError):
        container.restore_all(blocker)
    assert blocker.read_bytes() == b"x"


def test_read_many_into_bytearrays_and_views(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    content = files[0].read_bytes()
    with Container(make_pack(tmp_path, files)) as container:
        member = container.member()
        whole = bytearray(200)
        view = memoryview(whole)
        tail = bytearray(17)
        member.read_many([(1000, view[100:200]), (0, view[0:100]), (len(content) - 17, tail)])
        member.read_many([])
    assert bytes(whole) == content[0:100] + content[1000:1100]
    assert bytes(tail) == content[-17:]


def test_read_many_into_numpy(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    files = make_files(tmp_path, 1)
    content = files[0].read_bytes()
    with Container(make_pack(tmp_path, files)) as container:
        member = container.member()
        floats = np.zeros(64, dtype=np.float32)
        raw = np.zeros(33, dtype=np.uint8)
        member.read_many([(4096, floats), (7, raw)])
    assert floats.tobytes() == content[4096 : 4096 + 256]
    assert raw.tobytes() == content[7:40]


def test_read_many_refuses_overlapping_buffers(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    with Container(make_pack(tmp_path, files)) as container:
        member = container.member()
        shared = bytearray(64)
        with pytest.raises(BufferError, match="overlap"):
            member.read_many([(0, shared), (100, shared)])
        view = memoryview(shared)
        with pytest.raises(BufferError, match="overlap"):
            member.read_many([(0, view[0:40]), (100, view[30:64])])
        assert bytes(shared) == bytes(64), "nothing is written before the refusal"
        # Empty buffers alias nothing, and adjacent views do not overlap.
        member.read_many([(0, view[0:32]), (5, view[32:32]), (32, view[32:64])])
        assert bytes(shared) == files[0].read_bytes()[:64]


def test_read_many_checks_every_range_first(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    with Container(make_pack(tmp_path, files)) as container:
        member = container.member()
        first = bytearray(8)
        with pytest.raises(ziptensors.RangeError):
            member.read_many([(0, first), (member.len, bytearray(1))])


def test_pack_builder(tmp_path: Path) -> None:
    files = make_files(tmp_path, 3)
    source = make_pack(tmp_path, [files[2]], "source.ziptensors")
    output = tmp_path / "built.ziptensors"
    with PackBuilder(output, threads=1) as builder, Container(source) as container:
        assert builder.add_file(files[0]).name == files[0].name
        assert builder.add_file_delta(files[1], "tuned", files[0].name).name == "tuned"
        copied = builder.add_member_from(container, files[2].name)
        assert copied.copied_bytes > 0
        report = builder.finish()
    assert [member.name for member in report.members] == [files[0].name, "tuned", files[2].name]
    assert members_of(output) == {
        files[0].name: files[0].read_bytes(),
        "tuned": files[1].read_bytes(),
        files[2].name: files[2].read_bytes(),
    }
    for call in (lambda: builder.add_file(files[0], "again"), builder.finish):
        with pytest.raises(ValueError, match="finished or closed"):
            call()
    builder.close()  # nothing to drop, still fine


def test_pack_builder_exit_without_finish_drops_the_temporary_file(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with PackBuilder(out_dir / "never.ziptensors", threads=1) as builder:
        builder.add_file(files[0])
        assert list(out_dir.iterdir()), "the temporary file exists while the pack is open"
    assert not list(out_dir.iterdir())
    with pytest.raises(ValueError, match="finished or closed"):
        builder.finish()


def test_pack_builder_exit_on_an_exception(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(RuntimeError, match="caller"):
        with PackBuilder(out_dir / "never.ziptensors", threads=1) as builder:
            builder.add_file(files[0])
            raise RuntimeError("caller")
    assert not list(out_dir.iterdir())


def test_pack_builder_refuses_like_the_core(tmp_path: Path) -> None:
    files = make_files(tmp_path, 1)
    with PackBuilder(tmp_path / "x.ziptensors", threads=1) as builder:
        with pytest.raises(InputError):
            builder.add_file(files[0], "../escape")
        with pytest.raises(NotFoundError):
            builder.add_file_delta(files[0], "d", "absent")
    with pytest.raises(ValueError, match="unknown mode"):
        PackBuilder(tmp_path / "y.ziptensors", mode="ultra")


def test_pack_walks_directories(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    files = make_files(tree, 1)
    nested = make_safetensors(tree / "sub" / "n.safetensors", {"w": ("F32", [4], bytes(16))})
    (tree / "part.bin.incomplete").write_bytes(b"partial")
    try:
        (tree / "link.safetensors").symlink_to(files[0])
    except OSError as error:
        pytest.skip(f"symbolic links are not available here: {error}")
    loose = make_files(tmp_path, 2, seed=40)[1]

    found = ziptensors.collect_inputs([tree])
    assert [name for _, name in found.files] == [files[0].name, "sub/n.safetensors"]
    assert (found.skipped_symlinks, found.skipped_incomplete) == (1, 1)

    output = tmp_path / "set.ziptensors"
    report = ziptensors.pack([tree, loose], output, threads=1)
    assert [member.name for member in report.members] == [
        files[0].name,
        "sub/n.safetensors",
        loose.name,
    ]
    assert members_of(output)["sub/n.safetensors"] == nested.read_bytes()
    with pytest.raises(InputError):
        ziptensors.pack([tree], tree / "inside.ziptensors", threads=1)
    assert not (tree / "inside.ziptensors").exists()


def test_edit_strategies(tmp_path: Path) -> None:
    files = make_files(tmp_path, 4)
    output = make_pack(tmp_path, files[:2])
    first, second, third, _ = (path.name for path in files)

    appended = ziptensors.edit(output, [AddOp(files[2])], strategy="append", threads=1)
    assert (appended.kept, appended.added, appended.generation) == (2, 1, 2)
    assert list(members_of(output)) == [first, second, third]

    rewritten = ziptensors.edit(
        output, [RemoveOp(second), ReplaceOp(first, files[3])], threads=1
    )
    assert (rewritten.kept, rewritten.removed, rewritten.replaced) == (1, 1, 1)
    assert rewritten.generation == 3
    assert rewritten.dead_bytes_before > 0, "the append left the old index behind"
    assert members_of(output) == {first: files[3].read_bytes(), third: files[2].read_bytes()}

    in_place = ziptensors.edit(
        output, [AddOp(files[1], "again")], strategy="in_place", threads=1
    )
    assert in_place.added == 1
    assert members_of(output) == {
        first: files[3].read_bytes(),
        third: files[2].read_bytes(),
        "again": files[1].read_bytes(),
    }


def test_edit_refusals_leave_the_file_alone(tmp_path: Path) -> None:
    files = make_files(tmp_path, 2)
    output = make_pack(tmp_path, files[:1])
    before = output.read_bytes()
    with pytest.raises(InputError) as caught:
        ziptensors.edit(output, [AddOp(files[1], files[0].name)], threads=1)
    assert caught.value.code == CODE_EDIT_REFUSED
    with pytest.raises(InputError):
        ziptensors.edit(output, [AddOp(files[1])], strategy="append", repack=True)
    with pytest.raises(NotFoundError):
        ziptensors.edit(output, [RemoveOp("absent")])
    with pytest.raises(ValueError, match="unknown strategy"):
        ziptensors.edit(output, [], strategy="sideways")
    with pytest.raises(TypeError, match="AddOp, RemoveOp or ReplaceOp"):
        ziptensors.edit(output, [("add", files[1])])
    assert output.read_bytes() == before


def test_compact_drops_dead_bytes(tmp_path: Path) -> None:
    files = make_files(tmp_path, 2)
    output = make_pack(tmp_path, files[:1])
    ziptensors.edit(output, [AddOp(files[1])], strategy="append", threads=1)
    grown = output.stat().st_size
    with pytest.raises(InputError):
        ziptensors.compact(output, strategy="append")
    report = ziptensors.compact(output, threads=1)
    assert report.dead_bytes_before > 0
    assert output.stat().st_size < grown
    assert report.encode.output_len == output.stat().st_size
    ziptensors.compact(output, strategy="in_place", threads=1)
    assert list(members_of(output)) == [files[0].name, files[1].name]


def test_recover_a_torn_append(tmp_path: Path) -> None:
    files = make_files(tmp_path, 2)
    output = make_pack(tmp_path, files[:1])
    first_len = output.stat().st_size
    ziptensors.edit(output, [AddOp(files[1])], strategy="append", threads=1)
    # An append cut short: the new footer is gone, the old generation is intact.
    with output.open("r+b") as file:
        file.truncate(output.stat().st_size - 10)
    with pytest.raises(FormatError):
        Container(output)

    planned = ziptensors.recover(output, dry_run=True)
    assert (planned.valid_len, planned.truncated) == (first_len, False)
    # The torn footer is not a candidate: nothing was passed over.
    assert planned.passed_over == 0
    assert output.stat().st_size > first_len

    report = ziptensors.recover(output, threads=1)
    assert (report.valid_len, report.generation, report.truncated) == (first_len, 1, True)
    assert report.verified_bytes == files[0].stat().st_size
    assert output.stat().st_size == first_len
    assert members_of(output) == {files[0].name: files[0].read_bytes()}

    untouched = ziptensors.recover(output)
    assert (untouched.truncated, untouched.verified_bytes) == (False, None)
    assert untouched.passed_over == 0


def test_recover_reports_garbage(tmp_path: Path) -> None:
    junk = tmp_path / "junk.ziptensors"
    junk.write_bytes(seeded_bytes(5, 4096))
    with pytest.raises(FormatError):
        ziptensors.recover(junk)
    assert junk.read_bytes() == seeded_bytes(5, 4096)


def test_enable_stderr_logging_in_a_child(tmp_path: Path) -> None:
    """In a child process, so the global subscriber does not outlive this test. A second call
    succeeds and changes nothing; the walk of `collect_inputs` logs one info line."""
    script = (
        "import sys, ziptensors\n"
        "ziptensors.enable_stderr_logging('info')\n"
        "ziptensors.enable_stderr_logging('debug')\n"
        "try:\n"
        "    ziptensors.enable_stderr_logging('loud')\n"
        "except ValueError as error:\n"
        "    assert 'unknown level' in str(error), error\n"
        "else:\n"
        "    sys.exit('an unknown level was accepted')\n"
        "ziptensors.collect_inputs([sys.argv[1]])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert "inputs collected" in done.stderr
    assert "DEBUG" not in done.stderr, "the second call must not change the level"


def long_actions(slow: tuple[Path, Path], scratch: Path) -> dict[str, Callable[[], object]]:
    """The slow calls added beside the reader, by name, each on the slow fixture."""
    source, output = slow
    member = Container(output, threads=1).member()
    half = member.len // 2
    first, second = bytearray(half), bytearray(member.len - half)
    builder = PackBuilder(scratch / "built", mode="balanced", threads=1)
    copy = scratch / "copy.ziptensors"
    copy.write_bytes(output.read_bytes())
    tree = scratch / "tree"
    tree.mkdir()
    (tree / source.name).write_bytes(source.read_bytes())
    return {
        "read_many": lambda: member.read_many([(0, first), (half, second)]),
        "restore_all": lambda: Container(output, threads=1).restore_all(scratch / "all"),
        "pack_builder_add_file": lambda: builder.add_file(source),
        "edit": lambda: ziptensors.edit(
            copy, [AddOp(source, "again")], mode="balanced", threads=1
        ),
        "compress_tree": lambda: ziptensors.compress_tree(tree, mode="balanced", threads=1),
    }


@pytest.mark.parametrize(
    "name",
    ["read_many", "restore_all", "pack_builder_add_file", "edit", "compress_tree"],
)
def test_long_calls_release_the_gil(name: str, slow: tuple[Path, Path], tmp_path: Path) -> None:
    action = long_actions(slow, tmp_path)[name]
    stall, duration = stall_during(action)
    assert duration >= SLOW_SECONDS / 2
    assert stall < duration / 2


def test_recover_releases_the_gil(slow: tuple[Path, Path], tmp_path: Path) -> None:
    """A file that opens as it is is not verified, so the copy gets a torn tail first; the scan
    then verifies every member, which is the slow part."""
    copy = tmp_path / "copy.ziptensors"
    copy.write_bytes(slow[1].read_bytes() + bytes(100))
    stall, duration = stall_during(lambda: ziptensors.recover(copy, threads=1))
    assert duration >= SLOW_SECONDS / 2
    assert stall < duration / 2
