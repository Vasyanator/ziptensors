//! File: tests/cli.rs
//! Purpose: process tests of the `ziptensors` binary: exit codes, streams, and each command end to end.
//!
//! Implementation notes: every test works in its own `Scratch` under the system temp directory, on data
//! generated here. Nothing is written in the repository.

mod common;

use std::fs;
use std::path::Path;

use common::{Scratch, bytes, hostile_pack, parse_json, safetensors, summary_line, ziptensors};

const MIB: usize = 1 << 20;

#[test]
fn usage_errors_exit_with_2() {
    let cases: [&[&str]; 9] = [
        &[],
        &["frobnicate"],
        &["encode", "only-input"],
        &["encode", "a", "b", "--mode", "bogus"],
        &["restore", "a"],
        &["restore", "a", "b", "--all", "c"],
        &["pack", "only-output"],
        // Only `max` stores delta blobs; the check runs before any file is looked at.
        &["pack", "out", "in", "--delta-base", "in", "--mode", "fast"],
        &["pack", "out", "in", "--mode", "balanced", "--delta-base", "in"],
    ];
    for args in cases {
        let output = ziptensors(args);
        assert_eq!(output.code, 2, "{args:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{args:?}: {output:?}");
        if args.contains(&"--delta-base") {
            assert!(output.stderr.contains("--delta-base needs --mode max"), "{output:?}");
        }
    }
}

#[test]
fn help_exits_with_0_on_stdout() {
    let output = ziptensors(["--help"]);
    assert_eq!(output.code, 0, "{output:?}");
    for command in ["encode", "pack", "restore", "extract", "merge", "inspect", "verify"] {
        assert!(output.stdout.contains(command), "{output:?}");
    }
}

#[test]
fn a_missing_container_exits_with_1_and_says_why() {
    let scratch = Scratch::new();
    let absent = scratch.path("absent.ziptensors");
    let output = ziptensors(["verify".as_ref(), absent.as_os_str()]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(output.stderr.contains("error: cannot open container"), "{output:?}");
    assert!(output.stderr.contains("hint:"), "{output:?}");
    // The technical side: a structured log with the stable code.
    assert!(output.stderr.contains("ERROR"), "{output:?}");
    assert!(output.stderr.contains("code_value=1"), "{output:?}");
}

#[test]
fn a_file_that_is_not_a_container_exits_with_1() {
    let scratch = Scratch::new();
    let file = scratch.write("plain.bin", &bytes(1, 4096));
    let output = ziptensors(["inspect".as_ref(), file.as_os_str()]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stderr.contains("not a ziptensors file"), "{output:?}");
}

#[test]
fn encode_and_restore_are_byte_exact_in_every_mode() {
    let scratch = Scratch::new();
    let source = safetensors(2, MIB);
    let input = scratch.write("model.safetensors", &source);
    for mode in ["fast", "balanced", "max"] {
        let container = scratch.path(&format!("{mode}.ziptensors"));
        let restored = scratch.path(&format!("{mode}.restored"));
        let encoded = ziptensors([
            "encode".as_ref(),
            input.as_os_str(),
            container.as_os_str(),
            "--mode".as_ref(),
            mode.as_ref(),
            "--threads".as_ref(),
            "2".as_ref(),
        ]);
        assert_eq!(encoded.code, 0, "{encoded:?}");
        assert_eq!(summary_line(&encoded.stdout, "members"), "1");
        let done = ziptensors([
            "restore".as_ref(),
            container.as_os_str(),
            restored.as_os_str(),
        ]);
        assert_eq!(done.code, 0, "{done:?}");
        assert!(fs::read(&restored).expect("read restored") == source, "{mode}");
        let verified = ziptensors(["verify".as_ref(), container.as_os_str()]);
        assert_eq!(verified.code, 0, "{verified:?}");
        assert!(verified.stdout.contains("verified 1 members"), "{verified:?}");
    }
}

#[test]
fn pack_skips_links_and_incomplete_files_and_restores_the_rest() {
    let scratch = Scratch::new();
    let first = bytes(3, 2 * MIB);
    let second = safetensors(4, 1024);
    scratch.write("tree/b/second.safetensors", &second);
    scratch.write("tree/a.bin", &first);
    scratch.write("tree/c.bin.incomplete", b"partial");
    #[cfg(unix)]
    {
        let target = scratch.path("tree/a.bin");
        std::os::unix::fs::symlink(&target, scratch.path("tree/link.bin")).expect("file link");
        std::os::unix::fs::symlink(scratch.path("tree/b"), scratch.path("tree/link-dir"))
            .expect("directory link");
        // A link is skipped whatever it is called: the name would be refused as a member.
        std::os::unix::fs::symlink(&target, scratch.path("tree/bad:name")).expect("odd link");
    }
    let container = scratch.path("out/pack.ziptensors");
    fs::create_dir_all(scratch.path("out")).expect("create output directory");
    let packed = ziptensors([
        "pack".as_ref(),
        container.as_os_str(),
        scratch.path("tree").as_os_str(),
        "--threads".as_ref(),
        "2".as_ref(),
    ]);
    assert_eq!(packed.code, 0, "{packed:?}");
    assert_eq!(summary_line(&packed.stdout, "members"), "2");
    let links = if cfg!(unix) { 3 } else { 0 };
    let skipped = format!("{links} symbolic links, 1 incomplete files");
    assert_eq!(summary_line(&packed.stdout, "skipped"), skipped);
    assert!(packed.stderr.contains("skipping incomplete file"), "{packed:?}");
    if cfg!(unix) {
        assert!(packed.stderr.contains("skipping symbolic link"), "{packed:?}");
    }
    // One progress line per member, with the running total.
    let progress: Vec<_> =
        packed.stderr.lines().filter(|line| line.contains("cumulative_saved=")).collect();
    assert_eq!(progress.len(), 2, "{packed:?}");
    assert!(progress[0].contains("done=1") && progress[0].contains("member=a.bin"), "{progress:?}");
    assert!(progress[1].contains("member=b/second.safetensors"), "{progress:?}");

    let dir = scratch.path("restored");
    let restored = ziptensors([
        "restore".as_ref(),
        container.as_os_str(),
        "--all".as_ref(),
        dir.as_os_str(),
    ]);
    assert_eq!(restored.code, 0, "{restored:?}");
    assert!(fs::read(dir.join("a.bin")).expect("read a.bin") == first);
    assert!(fs::read(dir.join("b/second.safetensors")).expect("read second") == second);
    assert_eq!(scratch.list("restored"), ["a.bin", "b"]);
    assert_eq!(scratch.list("out"), ["pack.ziptensors"]);
}

#[test]
fn pack_stops_on_failures_with_no_output() {
    let scratch = Scratch::new();
    let good = scratch.write("in/one/same.bin", &bytes(5, 1024));
    scratch.write("in/two/same.bin", &bytes(6, 1024));
    scratch.write("only/x.bin.incomplete", b"partial");
    let container = scratch.path("out/pack.ziptensors");
    fs::create_dir_all(scratch.path("out")).expect("create output directory");
    let absent = scratch.path("absent");
    let cases = [
        ("cannot read input", vec![good.clone(), absent]),
        ("two inputs", vec![scratch.path("in/one"), scratch.path("in/two")]),
        ("nothing to pack", vec![scratch.path("only")]),
    ];
    for (message, inputs) in cases {
        let mut args = vec!["pack".into(), container.clone().into_os_string()];
        args.extend(inputs.into_iter().map(std::path::PathBuf::into_os_string));
        let output = ziptensors(args);
        assert_eq!(output.code, 1, "{output:?}");
        assert!(output.stderr.contains(message), "{output:?}");
        assert!(scratch.list("out").is_empty(), "{:?}", scratch.list("out"));
    }
}

/// A safetensors file of one `F32` tensor and the same file with the lowest bit of every 14th data
/// byte flipped: a finetune whose residual against the first is sparse.
fn base_and_finetune(seed: u64, elements: usize) -> (Vec<u8>, Vec<u8>) {
    let base = safetensors(seed, elements);
    let data_start = base.len() - elements * 4;
    let mut tuned = base.clone();
    tuned[data_start..].iter_mut().step_by(14).for_each(|byte| *byte ^= 1);
    (base, tuned)
}

#[test]
fn pack_with_a_delta_base_stores_the_finetune_against_it_and_restores_both() {
    let scratch = Scratch::new();
    let (base, tuned) = base_and_finetune(21, MIB / 2);
    let unrelated = bytes(22, 64 * 1024);
    // The base sorts last by name, so the summary proves it was moved to the front.
    scratch.write("tree/a-tuned.safetensors", &tuned);
    scratch.write("tree/config.bin", &unrelated);
    let base_path = scratch.write("tree/z-base.safetensors", &base);
    let container = scratch.path("pack.ziptensors");
    let packed = ziptensors([
        "pack".as_ref(),
        container.as_os_str(),
        scratch.path("tree").as_os_str(),
        "--delta-base".as_ref(),
        base_path.as_os_str(),
        "--threads".as_ref(),
        "2".as_ref(),
    ]);
    assert_eq!(packed.code, 0, "{packed:?}");
    assert_eq!(summary_line(&packed.stdout, "members"), "3");
    assert_eq!(summary_line(&packed.stdout, "delta blobs"), "1");
    let progress: Vec<_> =
        packed.stderr.lines().filter(|line| line.contains("cumulative_saved=")).collect();
    assert_eq!(progress.len(), 3, "{packed:?}");
    assert!(progress[0].contains("member=z-base.safetensors"), "{progress:?}");
    assert!(progress[0].contains("delta_blobs=0"), "{progress:?}");
    assert!(progress[1].contains("member=a-tuned.safetensors"), "{progress:?}");
    assert!(progress[1].contains("delta_blobs=1"), "{progress:?}");
    assert!(progress[2].contains("member=config.bin"), "{progress:?}");
    assert!(progress[2].contains("delta_blobs=0"), "{progress:?}");

    let inspected = ziptensors(["inspect".as_ref(), container.as_os_str(), "--json".as_ref()]);
    assert_eq!(inspected.code, 0, "{inspected:?}");
    let dir = scratch.path("restored");
    let restored = ziptensors([
        "restore".as_ref(),
        container.as_os_str(),
        "--all".as_ref(),
        dir.as_os_str(),
    ]);
    assert_eq!(restored.code, 0, "{restored:?}");
    assert!(fs::read(dir.join("z-base.safetensors")).expect("read base") == base);
    assert!(fs::read(dir.join("a-tuned.safetensors")).expect("read finetune") == tuned);
    assert!(fs::read(dir.join("config.bin")).expect("read config") == unrelated);

    // The same pack without a base stores the finetune alone, in more bytes.
    let plain = scratch.path("plain.ziptensors");
    let alone = ziptensors([
        "pack".as_ref(),
        plain.as_os_str(),
        scratch.path("tree").as_os_str(),
        "--threads".as_ref(),
        "2".as_ref(),
    ]);
    assert_eq!(alone.code, 0, "{alone:?}");
    assert_eq!(summary_line(&alone.stdout, "delta blobs"), "0");
    let len = |path: &Path| fs::metadata(path).expect("container").len();
    assert!(len(&container) < len(&plain), "{} vs {}", len(&container), len(&plain));
}

#[test]
fn pack_refuses_a_delta_base_that_is_not_packed_before_writing() {
    let scratch = Scratch::new();
    let (base, tuned) = base_and_finetune(23, 1024);
    let tuned_path = scratch.write("in/tuned.safetensors", &tuned);
    let base_path = scratch.write("elsewhere/base.safetensors", &base);
    fs::create_dir_all(scratch.path("out")).expect("create output directory");
    let container = scratch.path("out/pack.ziptensors");
    let cases = [
        ("is not among the files to pack", base_path),
        ("is not a regular file", scratch.path("in")),
        ("cannot read the delta base", scratch.path("absent.safetensors")),
    ];
    for (message, base) in cases {
        let output = ziptensors([
            "pack".as_ref(),
            container.as_os_str(),
            tuned_path.as_os_str(),
            "--delta-base".as_ref(),
            base.as_os_str(),
        ]);
        assert_eq!(output.code, 1, "{output:?}");
        assert!(output.stderr.contains(message), "{output:?}");
        assert!(scratch.list("out").is_empty(), "{:?}", scratch.list("out"));
    }
}

#[cfg(unix)]
#[test]
fn pack_refuses_a_file_name_a_restore_would_refuse_before_writing() {
    let scratch = Scratch::new();
    scratch.write("tree/bad:name.bin", &bytes(7, 1024));
    let container = scratch.path("pack.ziptensors");
    let output = ziptensors([
        "pack".as_ref(),
        container.as_os_str(),
        scratch.path("tree").as_os_str(),
    ]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stderr.contains("unsafe member name"), "{output:?}");
    assert!(!container.exists());
}

#[test]
fn restore_all_refuses_a_hostile_name_before_writing_anything() {
    for name in ["../x", "a/../../x", "/x", "a:b"] {
        let scratch = Scratch::new();
        let container = scratch.path("hostile.ziptensors");
        hostile_pack(&container, &["fine.bin", name]);
        // The file itself opens: the name is only refused where a path is built from it.
        let inspected = ziptensors(["inspect".as_ref(), container.as_os_str()]);
        assert_eq!(inspected.code, 0, "{name}: {inspected:?}");

        let dir = scratch.path("deep/out");
        let output = ziptensors([
            "restore".as_ref(),
            container.as_os_str(),
            "--all".as_ref(),
            dir.as_os_str(),
        ]);
        assert_eq!(output.code, 1, "{name}: {output:?}");
        assert!(output.stderr.contains("unsafe member name"), "{name}: {output:?}");
        assert!(output.stderr.contains("code_value=14"), "{name}: {output:?}");
        // Neither the safe member nor the output directory exists, let alone the target of the name.
        assert!(!scratch.path("deep").exists(), "{name}");
        assert_eq!(scratch.list(""), ["hostile.ziptensors"], "{name}");
    }
}

#[test]
fn extract_restores_one_member_of_a_pack() {
    let scratch = Scratch::new();
    let wanted = bytes(8, MIB);
    scratch.write("tree/keep/wanted.bin", &wanted);
    scratch.write("tree/other.bin", &bytes(9, 4096));
    let container = scratch.path("pack.ziptensors");
    let packed = ziptensors([
        "pack".as_ref(),
        container.as_os_str(),
        scratch.path("tree").as_os_str(),
    ]);
    assert_eq!(packed.code, 0, "{packed:?}");

    let out = scratch.path("wanted.out");
    let extracted = ziptensors([
        "extract".as_ref(),
        container.as_os_str(),
        "keep/wanted.bin".as_ref(),
        out.as_os_str(),
    ]);
    assert_eq!(extracted.code, 0, "{extracted:?}");
    assert!(fs::read(&out).expect("read extracted") == wanted);

    let missing = ziptensors([
        "extract".as_ref(),
        container.as_os_str(),
        "no/such.bin".as_ref(),
        scratch.path("missing.out").as_os_str(),
    ]);
    assert_eq!(missing.code, 1, "{missing:?}");
    assert!(missing.stderr.contains("member not found"), "{missing:?}");
    assert!(missing.stderr.contains("inspect"), "{missing:?}");
    assert!(!scratch.path("missing.out").exists());

    // `restore` without --all needs a single member, and says what to use instead.
    let plain = ziptensors([
        "restore".as_ref(),
        container.as_os_str(),
        scratch.path("plain.out").as_os_str(),
    ]);
    assert_eq!(plain.code, 1, "{plain:?}");
    assert!(plain.stderr.contains("use --all or extract"), "{plain:?}");
}

#[test]
fn merge_copies_members_of_several_containers() {
    let scratch = Scratch::new();
    let first = bytes(10, MIB);
    let second = bytes(11, MIB);
    let first_input = scratch.write("first.bin", &first);
    let second_input = scratch.write("second.bin", &second);
    let first_container = scratch.path("first.ziptensors");
    let second_container = scratch.path("second.ziptensors");
    for (input, container) in
        [(&first_input, &first_container), (&second_input, &second_container)]
    {
        let encoded =
            ziptensors(["encode".as_ref(), input.as_os_str(), container.as_os_str()]);
        assert_eq!(encoded.code, 0, "{encoded:?}");
    }
    let merged_path = scratch.path("merged.ziptensors");
    let merged = ziptensors([
        "merge".as_ref(),
        merged_path.as_os_str(),
        first_container.as_os_str(),
        second_container.as_os_str(),
    ]);
    assert_eq!(merged.code, 0, "{merged:?}");
    assert_eq!(summary_line(&merged.stdout, "members"), "2");
    assert!(merged.stderr.contains("cumulative_saved="), "{merged:?}");

    let dir = scratch.path("restored");
    let restored = ziptensors([
        "restore".as_ref(),
        merged_path.as_os_str(),
        "--all".as_ref(),
        dir.as_os_str(),
    ]);
    assert_eq!(restored.code, 0, "{restored:?}");
    assert!(fs::read(dir.join("first.bin")).expect("read first") == first);
    assert!(fs::read(dir.join("second.bin")).expect("read second") == second);

    // The same member twice is the core's refusal, and leaves nothing behind.
    let twice_path = scratch.path("twice.ziptensors");
    let twice = ziptensors([
        "merge".as_ref(),
        twice_path.as_os_str(),
        first_container.as_os_str(),
        first_container.as_os_str(),
    ]);
    assert_eq!(twice.code, 1, "{twice:?}");
    assert!(!twice_path.exists());
}

#[test]
fn inspect_json_is_one_document_with_the_documented_schema() {
    let scratch = Scratch::new();
    let model = safetensors(12, 4096);
    scratch.write("tree/model.safetensors", &model);
    scratch.write("tree/a.bin", &bytes(13, 4096));
    #[cfg(unix)]
    scratch.write("tree/quo\"te.bin", b"name to escape");
    let container = scratch.path("pack.ziptensors");
    let packed = ziptensors([
        "pack".as_ref(),
        container.as_os_str(),
        scratch.path("tree").as_os_str(),
    ]);
    assert_eq!(packed.code, 0, "{packed:?}");

    let output = ziptensors(["inspect".as_ref(), container.as_os_str(), "--json".as_ref()]);
    assert_eq!(output.code, 0, "{output:?}");
    let document = parse_json(&output.stdout);
    assert_eq!(document.get("schema_version").number(), 1);
    assert_eq!(document.get("path").string(), container.display().to_string());
    let file_bytes = fs::metadata(&container).expect("stat container").len();
    assert_eq!(document.get("file_bytes").number(), file_bytes);

    let members = document.get("members").array();
    let lens: u64 = members.iter().map(|member| member.get("len").number()).sum();
    assert_eq!(document.get("raw_bytes").number(), lens);
    assert_eq!(
        document.get("saved_bytes").number(),
        lens.saturating_sub(file_bytes),
    );
    let layers = document.get("layers");
    assert_eq!(layers.get("raw").number(), lens);
    assert!(layers.get("stored").number() < file_bytes);
    let classes = document.get("classes").array();
    assert_eq!(classes[0].get("class").string(), "skeleton");
    let class_raw: u64 = classes.iter().map(|class| class.get("raw").number()).sum();
    assert_eq!(class_raw, lens);

    let names: Vec<_> = members.iter().map(|member| member.get("name").string()).collect();
    let mut expected = vec!["a.bin", "model.safetensors"];
    if cfg!(unix) {
        expected.push("quo\"te.bin");
    }
    assert_eq!(names, expected);

    let model_row = &members[1];
    let model_len = u64::try_from(model.len()).expect("a short file");
    assert_eq!(model_row.get("len").number(), model_len);
    assert_eq!(model_row.get("profile").string(), "safetensors");
    assert_eq!(model_row.get("tensor_count").number(), 1);
    let digest = model_row.get("blake3").string();
    assert_eq!(digest.len(), 64);
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()), "{digest}");

    let text = ziptensors(["inspect".as_ref(), container.as_os_str()]);
    assert_eq!(text.code, 0, "{text:?}");
    assert!(text.stdout.contains("model.safetensors"), "{text:?}");
    assert!(text.stdout.contains("safetensors"), "{text:?}");
}

#[test]
fn verify_finds_a_damaged_container() {
    let scratch = Scratch::new();
    let input = scratch.write("data.bin", &bytes(14, MIB));
    let container = scratch.path("data.ziptensors");
    let encoded = ziptensors(["encode".as_ref(), input.as_os_str(), container.as_os_str()]);
    assert_eq!(encoded.code, 0, "{encoded:?}");
    let mut file = fs::read(&container).expect("read container");
    // In the data region: past the header, well before the index at the end.
    file[1024] ^= 0x40;
    fs::write(&container, file).expect("write damaged container");
    let output = ziptensors(["verify".as_ref(), container.as_os_str()]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(output.stderr.contains("verification of"), "{output:?}");
    assert!(output.stderr.contains("damaged"), "{output:?}");
}

#[test]
fn pack_refuses_an_output_inside_a_walked_directory() {
    let scratch = Scratch::new();
    scratch.write("tree/a.bin", b"payload");
    let output = scratch.path("tree/out.ziptensors");
    let packed = ziptensors([
        "pack".as_ref(),
        output.as_os_str(),
        scratch.path("tree").as_os_str(),
    ]);
    assert_eq!(packed.code, 1, "{packed:?}");
    assert!(packed.stderr.contains("into itself"), "{packed:?}");
    assert_eq!(scratch.list("tree"), ["a.bin"]);
}

#[test]
fn restore_takes_a_budget_and_refuses_the_container_as_its_output() {
    let scratch = Scratch::new();
    scratch.write("a.bin", &[7; 1000]);
    let container = scratch.path("a.ziptensors");
    let encoded = ziptensors([
        "encode".as_ref(),
        scratch.path("a.bin").as_os_str(),
        container.as_os_str(),
    ]);
    assert_eq!(encoded.code, 0, "{encoded:?}");

    let restored = scratch.path("restored");
    let restore = |budget: &str, out: &Path| {
        ziptensors([
            "restore".as_ref(),
            container.as_os_str(),
            out.as_os_str(),
            "--max-total-decoded".as_ref(),
            budget.as_ref(),
        ])
    };
    let refused = restore("999", &restored);
    assert_eq!(refused.code, 1, "{refused:?}");
    assert!(refused.stderr.contains("more than this reader accepts"), "{refused:?}");
    assert!(!restored.exists());
    assert_eq!(restore("1000", &restored).code, 0);
    assert_eq!(fs::read(&restored).expect("restored"), [7; 1000]);

    let before = fs::read(&container).expect("container");
    let same = restore("1000", &container);
    assert_eq!(same.code, 1, "{same:?}");
    assert!(same.stderr.contains("output is the input"), "{same:?}");
    assert_eq!(fs::read(&container).expect("container"), before);
}

#[cfg(target_os = "linux")]
#[test]
fn mount_refuses_a_tree_it_cannot_show_and_names_every_bad_member() {
    let scratch = Scratch::new();
    let container = scratch.path("clash.ziptensors");
    hostile_pack(&container, &["a", "a/b", "fine.bin", "x//y"]);
    let mountpoint = scratch.path("mnt");
    fs::create_dir(&mountpoint).expect("create mountpoint");
    // Names are checked before anything is mounted, so this needs no FUSE.
    let output = ziptensors(["mount".as_ref(), container.as_os_str(), mountpoint.as_os_str()]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stderr.contains("2 member name(s) cannot be shown under posix rules"), "{output:?}");
    assert!(output.stderr.contains("\"a\" (it is a file and also the directory of member \"a/b\")"), "{output:?}");
    assert!(output.stderr.contains("\"x//y\" (it has an empty path part)"), "{output:?}");
    assert!(!output.stderr.contains("\"fine.bin\""), "{output:?}");
    assert_eq!(output.stdout, "", "{output:?}");
}

#[cfg(windows)]
#[test]
fn mount_refuses_names_windows_cannot_show_before_loading_winfsp() {
    let scratch = Scratch::new();
    let container = scratch.path("clash.ziptensors");
    hostile_pack(&container, &["CON.bin", "a.bin", "A.bin", "fine.bin"]);
    // Names are checked before WinFsp is loaded, so this passes on a machine without WinFsp.
    let output = ziptensors(["mount".as_ref(), container.as_os_str(), "X:".as_ref()]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stderr.contains("2 member name(s) cannot be shown under windows rules"), "{output:?}");
    assert!(output.stderr.contains("\"A.bin\" (it is the same path as member \"a.bin\")"), "{output:?}");
    assert!(output.stderr.contains("\"CON.bin\" (a path part is a reserved device name)"), "{output:?}");
    assert!(!output.stderr.contains("\"fine.bin\""), "{output:?}");
    assert_eq!(output.stdout, "", "{output:?}");
}

#[cfg(windows)]
#[test]
fn mount_help_carries_the_winfsp_notice() {
    let output = ziptensors(["mount", "--help"]);
    assert_eq!(output.code, 0, "{output:?}");
    assert!(output.stdout.contains("WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos"), "{output:?}");
    assert!(output.stdout.contains("https://github.com/winfsp/winfsp"), "{output:?}");
    assert!(output.stdout.contains("--winfsp-threads"), "{output:?}");
    assert!(!output.stdout.contains("--fuse-threads"), "{output:?}");
}

#[cfg(windows)]
#[test]
fn mount_refuses_a_mount_point_in_use_once_and_bounds_its_threads() {
    let scratch = Scratch::new();
    let container = scratch.path("ok.ziptensors");
    hostile_pack(&container, &["fine.bin"]);
    let existing = scratch.path("exists");
    fs::create_dir(&existing).expect("create");
    // Refused before WinFsp is loaded, so this passes on a machine without WinFsp.
    for point in [existing.as_os_str(), "C:".as_ref()] {
        let output = ziptensors(["mount".as_ref(), container.as_os_str(), point]);
        assert_eq!(output.code, 1, "{output:?}");
        assert_eq!(output.stderr.matches("cannot mount on").count(), 1, "{output:?}");
        assert!(output.stderr.contains("in use or the directory already exists"), "{output:?}");
    }
    let output = ziptensors(["mount".as_ref(), container.as_os_str(), "C:".as_ref(), "--winfsp-threads".as_ref(), "65".as_ref()]);
    assert_eq!(output.code, 2, "{output:?}");
}
