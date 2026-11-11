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
    let cases: [&[&str]; 18] = [
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
        &["edit", "c.ziptensors", "--add", "x", "--delta-base", "m", "--mode", "fast"],
        // An edit needs an operation; two strategies, or repack with another strategy, conflict.
        &["edit", "c.ziptensors"],
        &["edit", "c.ziptensors", "--add", "x", "-F", "-R"],
        &["edit", "c.ziptensors", "--add", "x", "-F", "-C"],
        &["edit", "c.ziptensors", "--add", "x", "-R", "-C"],
        &["edit", "c.ziptensors", "--add-as", "only-name"],
        &["compact", "c.ziptensors", "-R", "-C"],
        &["compact", "c.ziptensors", "-F"],
        &["recover"],
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
    let commands = [
        "encode", "pack", "edit", "compact", "recover", "restore", "extract", "merge", "inspect", "verify",
        "compress", "decompress",
    ];
    for command in commands {
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
        ("cannot read the inputs", vec![good.clone(), absent]),
        ("is used twice", vec![scratch.path("in/one"), scratch.path("in/two")]),
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

    // A member name in two inputs is refused before the output is started: no member is copied,
    // and the error names the member, not damage.
    let twice_path = scratch.path("twice.ziptensors");
    let twice = ziptensors([
        "merge".as_ref(),
        twice_path.as_os_str(),
        first_container.as_os_str(),
        second_container.as_os_str(),
        first_container.as_os_str(),
    ]);
    assert_eq!(twice.code, 1, "{twice:?}");
    assert!(!twice_path.exists());
    assert!(twice.stderr.contains("code_value=26"), "{twice:?}");
    assert!(twice.stderr.contains(r#"member name "first.bin" is used twice"#), "{twice:?}");
    assert!(twice.stderr.contains("hint: two inputs would be stored under one member name"), "{twice:?}");
    assert!(!twice.stderr.contains("copying member"), "{twice:?}");
    assert!(!twice.stderr.contains("damaged"), "{twice:?}");
}

#[test]
fn pack_refuses_two_inputs_with_one_member_name() {
    let scratch = Scratch::new();
    let first = scratch.write("a/w.bin", &bytes(14, 4096));
    let second = scratch.write("b/w.bin", &bytes(15, 4096));
    let output = scratch.path("out.ziptensors");
    let packed = ziptensors(["pack".as_ref(), output.as_os_str(), first.as_os_str(), second.as_os_str()]);
    assert_eq!(packed.code, 1, "{packed:?}");
    assert!(!output.exists());
    assert!(packed.stderr.contains("code_value=26"), "{packed:?}");
    assert!(packed.stderr.contains(r#"member name "w.bin" is used twice"#), "{packed:?}");
    assert!(packed.stderr.contains("hint: two inputs would be stored under one member name"), "{packed:?}");
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

/// Packs `files` (name, content) under `tree/` of `scratch` into `pack.ziptensors` and returns its path.
fn pack_of(scratch: &Scratch, files: &[(&str, &[u8])]) -> std::path::PathBuf {
    for (name, content) in files {
        scratch.write(&format!("tree/{name}"), content);
    }
    let container = scratch.path("pack.ziptensors");
    let packed = ziptensors([
        "pack".as_ref(),
        container.as_os_str(),
        scratch.path("tree").as_os_str(),
        "--threads".as_ref(),
        "2".as_ref(),
    ]);
    assert_eq!(packed.code, 0, "{packed:?}");
    container
}

/// The member names of `container`, in file order, through `inspect --json`.
fn member_names(container: &Path) -> Vec<String> {
    let output = ziptensors(["inspect".as_ref(), container.as_os_str(), "--json".as_ref()]);
    assert_eq!(output.code, 0, "{output:?}");
    let document = parse_json(&output.stdout);
    let members = document.get("members").array();
    members.iter().map(|member| member.get("name").string().to_owned()).collect()
}

/// Restores every member of `container` under `dir` and returns the content of `name` there.
fn restored(container: &Path, dir: &Path, name: &str) -> Vec<u8> {
    let output = ziptensors(["restore".as_ref(), container.as_os_str(), "--all".as_ref(), dir.as_os_str()]);
    assert_eq!(output.code, 0, "{output:?}");
    fs::read(dir.join(name)).expect("read restored member")
}

#[test]
fn edit_adds_removes_and_replaces_in_every_strategy() {
    let a = bytes(30, 64 * 1024);
    let b = safetensors(31, 4096);
    let c = bytes(32, 32 * 1024);
    let replacement = bytes(33, 48 * 1024);
    let added = safetensors(34, 2048);
    for strategy in [None, Some("-F"), Some("-R")] {
        let scratch = Scratch::new();
        let container = pack_of(&scratch, &[("a.bin", &a), ("b.safetensors", &b), ("c.bin", &c)]);
        let new_file = scratch.write("new.safetensors", &added);
        let replacing = scratch.write("replacement.bin", &replacement);
        let mut args = vec![
            "edit".into(),
            container.clone().into_os_string(),
            "--add".into(),
            new_file.into_os_string(),
            "--remove".into(),
            "c.bin".into(),
            "--replace".into(),
            "a.bin".into(),
            replacing.into_os_string(),
            "--threads".into(),
            "2".into(),
        ];
        args.extend(strategy.map(std::ffi::OsString::from));
        let output = ziptensors(args);
        assert_eq!(output.code, 0, "{strategy:?}: {output:?}");
        assert_eq!(summary_line(&output.stdout, "members"), "3", "{output:?}");
        assert!(output.stdout.contains("1 kept, 1 added, 1 replaced, 1 removed"), "{output:?}");
        assert_eq!(summary_line(&output.stdout, "generation"), "2", "{output:?}");
        let risky_warning = output.stderr.contains("warning: -R rewrites");
        assert_eq!(risky_warning, strategy == Some("-R"), "{output:?}");
        // The kept member keeps its place, the replacement takes its member's, the new file goes last.
        assert_eq!(member_names(&container), ["a.bin", "b.safetensors", "new.safetensors"]);
        let verified = ziptensors(["verify".as_ref(), container.as_os_str()]);
        assert_eq!(verified.code, 0, "{verified:?}");
        let dir = scratch.path("restored");
        assert!(restored(&container, &dir, "a.bin") == replacement);
        assert!(fs::read(dir.join("b.safetensors")).expect("kept") == b);
        assert!(fs::read(dir.join("new.safetensors")).expect("added") == added);
    }
}

#[test]
fn edit_keeps_the_command_line_order_of_added_files_and_names_them() {
    let scratch = Scratch::new();
    let container = pack_of(&scratch, &[("a.bin", &bytes(35, 4096))]);
    let first = scratch.write("first.bin", &bytes(36, 4096));
    let second = scratch.write("second.bin", &bytes(37, 4096));
    let third = scratch.write("third.bin", &bytes(38, 4096));
    let output = ziptensors([
        "edit".as_ref(),
        container.as_os_str(),
        "--add".as_ref(),
        first.as_os_str(),
        "--add-as".as_ref(),
        "dir/named.bin".as_ref(),
        second.as_os_str(),
        "--add".as_ref(),
        third.as_os_str(),
    ]);
    assert_eq!(output.code, 0, "{output:?}");
    assert_eq!(member_names(&container), ["a.bin", "first.bin", "dir/named.bin", "third.bin"]);
}

#[test]
fn edit_refusals_change_nothing_and_say_so() {
    let scratch = Scratch::new();
    let content = bytes(39, 4096);
    let container = pack_of(&scratch, &[("a.bin", &content)]);
    let before = fs::read(&container).expect("container");
    let same_name = scratch.write("other/a.bin", &bytes(40, 4096));
    let output = ziptensors(["edit".as_ref(), container.as_os_str(), "--add".as_ref(), same_name.as_os_str()]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stderr.contains("edit refused"), "{output:?}");
    assert!(output.stderr.contains("code_value=22"), "{output:?}");
    assert!(output.stderr.contains("hint: nothing was written"), "{output:?}");
    assert_eq!(fs::read(&container).expect("container"), before);

    let missing = ziptensors(["edit".as_ref(), container.as_os_str(), "--remove".as_ref(), "nope".as_ref()]);
    assert_eq!(missing.code, 1, "{missing:?}");
    assert!(missing.stderr.contains("member not found"), "{missing:?}");
    assert_eq!(fs::read(&container).expect("container"), before);
}

#[test]
fn compact_drops_what_appends_left_behind() {
    let scratch = Scratch::new();
    let content = bytes(41, 256 * 1024);
    let container = pack_of(&scratch, &[("a.bin", &content), ("b.bin", &bytes(42, 128 * 1024))]);
    let removed = ziptensors(["edit".as_ref(), container.as_os_str(), "--remove".as_ref(), "b.bin".as_ref(), "-F".as_ref()]);
    assert_eq!(removed.code, 0, "{removed:?}");
    let len = |path: &Path| fs::metadata(path).expect("container").len();
    let appended = len(&container);
    for flags in [&[][..], &["-C"][..], &["-R"][..]] {
        let mut args = vec!["compact".into(), container.clone().into_os_string()];
        args.extend(flags.iter().map(std::ffi::OsString::from));
        let output = ziptensors(args);
        assert_eq!(output.code, 0, "{flags:?}: {output:?}");
        assert!(output.stdout.contains("0 added, 0 replaced, 0 removed"), "{output:?}");
        assert!(len(&container) < appended, "{flags:?}");
        assert_eq!(member_names(&container), ["a.bin"]);
    }
    assert!(restored(&container, &scratch.path("restored"), "a.bin") == content);
}

#[test]
fn extract_to_standard_output_writes_the_member_alone() {
    let scratch = Scratch::new();
    let wanted = bytes(43, MIB + 12_345);
    let container = pack_of(&scratch, &[("keep/wanted.bin", &wanted), ("other.bin", &bytes(44, 4096))]);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ziptensors"))
        .args(["extract".as_ref(), container.as_os_str(), "keep/wanted.bin".as_ref(), "-".as_ref()])
        .output()
        .expect("run the binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(output.stdout == wanted, "{} bytes on stdout", output.stdout.len());
    assert!(stderr.contains("restored keep/wanted.bin to -"), "{stderr}");
    assert!(!scratch.path("-").exists());
}

#[test]
fn inspect_lists_tensors_on_request() {
    let scratch = Scratch::new();
    let container = pack_of(&scratch, &[("a.bin", &bytes(45, 4096)), ("model.safetensors", &safetensors(46, 1000))]);
    let output = ziptensors([
        "inspect".as_ref(),
        container.as_os_str(),
        "--json".as_ref(),
        "--tensors".as_ref(),
        "--max-total-decoded".as_ref(),
        "1000000".as_ref(),
    ]);
    assert_eq!(output.code, 0, "{output:?}");
    let document = parse_json(&output.stdout);
    assert_eq!(document.get("schema_version").number(), 1);
    let members = document.get("members").array();
    assert!(members[0].get("tensors").array().is_empty());
    let tensors = members[1].get("tensors").array();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].get("name").string(), "t");
    assert_eq!(tensors[0].get("dtype").string(), "F32");
    assert_eq!(tensors[0].get("len").number(), 4000);
    assert!(tensors[0].get("offset").number() > 0);
    let shape: Vec<_> = tensors[0].get("shape").array().iter().map(common::Value::number).collect();
    assert_eq!(shape, [1000]);

    // Without --tensors the key is absent; the budget refuses before anything is read.
    let plain = ziptensors(["inspect".as_ref(), container.as_os_str(), "--json".as_ref()]);
    assert!(!plain.stdout.contains("\"tensors\""), "{plain:?}");
    let text = ziptensors(["inspect".as_ref(), container.as_os_str(), "--tensors".as_ref()]);
    assert_eq!(text.code, 0, "{text:?}");
    assert!(text.stdout.contains("F32") && text.stdout.contains("[1000]"), "{text:?}");
    let refused = ziptensors(["inspect".as_ref(), container.as_os_str(), "--max-total-decoded".as_ref(), "10".as_ref()]);
    assert_eq!(refused.code, 1, "{refused:?}");
    assert!(refused.stderr.contains("more than this reader accepts"), "{refused:?}");
}

#[test]
fn verify_and_merge_take_a_budget() {
    let scratch = Scratch::new();
    let container = pack_of(&scratch, &[("a.bin", &[1; 1000])]);
    let verify = ziptensors(["verify".as_ref(), container.as_os_str(), "--max-total-decoded".as_ref(), "999".as_ref()]);
    assert_eq!(verify.code, 1, "{verify:?}");
    let merged = scratch.path("merged.ziptensors");
    let merge = ziptensors([
        "merge".as_ref(),
        merged.as_os_str(),
        container.as_os_str(),
        "--max-total-decoded".as_ref(),
        "999".as_ref(),
    ]);
    assert_eq!(merge.code, 1, "{merge:?}");
    assert!(!merged.exists());
}

/// A container edited once with `-F`, whose last `cut` bytes are then lost: the newest footer is
/// torn, and the first generation is intact before `first_len`. Returns `(container, first_len)`.
fn torn_append(scratch: &Scratch, cut: u64) -> (std::path::PathBuf, u64) {
    let container = pack_of(scratch, &[("a.bin", &bytes(47, 64 * 1024))]);
    let first_len = fs::metadata(&container).expect("container").len();
    let added = scratch.write("added.bin", &bytes(48, 32 * 1024));
    let output = ziptensors(["edit".as_ref(), container.as_os_str(), "--add".as_ref(), added.as_os_str(), "-F".as_ref()]);
    assert_eq!(output.code, 0, "{output:?}");
    let file = fs::OpenOptions::new().write(true).open(&container).expect("open for cutting");
    let len = file.metadata().expect("length").len();
    file.set_len(len - cut).expect("tear the tail");
    (container, first_len)
}

#[test]
fn recover_shortens_a_torn_append_to_the_older_generation() {
    let scratch = Scratch::new();
    let (container, first_len) = torn_append(&scratch, 10);
    let torn_len = fs::metadata(&container).expect("container").len();
    let broken = ziptensors(["verify".as_ref(), container.as_os_str()]);
    assert_eq!(broken.code, 1, "{broken:?}");
    // The torn footer is a malformed index, whose hint points to the recovery dry run.
    assert!(broken.stderr.contains("code_value=6"), "{broken:?}");
    // The hint names the container the user gave, not a placeholder.
    let hint = broken.stderr.lines().find(|line| line.starts_with("hint: ")).expect("a hint line");
    assert!(hint.contains("`ziptensors recover --dry-run "), "{hint}");
    assert!(!hint.contains("<CONTAINER>"), "{hint}");
    let name = container.file_name().and_then(|name| name.to_str()).expect("a UTF-8 file name");
    // Unquoted, or quoted for the shell: single quotes on Unix, double quotes on Windows.
    let ends = [format!("{name}`"), format!("{name}'`"), format!("{name}\"`")];
    assert!(ends.iter().any(|end| hint.contains(end.as_str())), "{hint}");

    let dry = ziptensors(["recover".as_ref(), container.as_os_str(), "--dry-run".as_ref()]);
    assert_eq!(dry.code, 0, "{dry:?}");
    assert!(dry.stdout.contains("would recover"), "{dry:?}");
    assert_eq!(summary_line(&dry.stdout, "valid bytes"), first_len.to_string());
    assert_eq!(summary_line(&dry.stdout, "generation"), "1");
    assert!(dry.stderr.contains("warning: recovered generation 1"), "{dry:?}");
    assert!(dry.stderr.contains("would be discarded"), "{dry:?}");
    // A torn footer is not a candidate: nothing was passed over, and the warning does not say so.
    assert_eq!(summary_line(&dry.stdout, "passed over"), "0");
    assert!(!dry.stderr.contains("passed over"), "{dry:?}");
    assert_eq!(fs::metadata(&container).expect("container").len(), torn_len);

    let done = ziptensors(["recover".as_ref(), container.as_os_str(), "--max-candidates".as_ref(), "8".as_ref()]);
    assert_eq!(done.code, 0, "{done:?}");
    assert!(done.stdout.starts_with("recovered:"), "{done:?}");
    assert!(done.stderr.contains(&format!("({torn_len} -> {first_len} bytes) were discarded")), "{done:?}");
    assert_eq!(fs::metadata(&container).expect("container").len(), first_len);
    assert_eq!(member_names(&container), ["a.bin"]);

    let again = ziptensors(["recover".as_ref(), container.as_os_str()]);
    assert_eq!(again.code, 0, "{again:?}");
    assert!(again.stdout.contains("intact, nothing to do"), "{again:?}");
    assert!(!again.stderr.contains("warning:"), "{again:?}");
}

#[test]
fn recover_says_when_nothing_is_left_to_recover() {
    let scratch = Scratch::new();
    let container = pack_of(&scratch, &[("a.bin", &bytes(49, 4096))]);
    let mut file = fs::read(&container).expect("container");
    // The header alone, then garbage: no footer anywhere.
    file.truncate(64);
    file.extend_from_slice(&bytes(50, 4096));
    fs::write(&container, &file).expect("write");
    let output = ziptensors(["recover".as_ref(), container.as_os_str()]);
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stderr.contains("code_value=24"), "{output:?}");
    assert!(output.stderr.contains("hint: no valid container was found"), "{output:?}");
    assert_eq!(fs::read(&container).expect("container"), file);
}

#[cfg(unix)]
#[test]
fn recover_of_a_file_it_cannot_write_points_to_the_dry_run() {
    use std::os::unix::fs::PermissionsExt;
    let scratch = Scratch::new();
    let (container, _) = torn_append(&scratch, 10);
    fs::set_permissions(&container, fs::Permissions::from_mode(0o444)).expect("read only");
    let output = ziptensors(["recover".as_ref(), container.as_os_str()]);
    if output.code == 0 {
        // A privileged user opens the file anyway: nothing to observe here.
        eprintln!("skipping: the file opened for writing despite its mode");
        return;
    }
    assert_eq!(output.code, 1, "{output:?}");
    assert!(output.stderr.contains("hint: the file cannot be opened for writing"), "{output:?}");
    assert!(output.stderr.contains("--dry-run"), "{output:?}");
    let dry = ziptensors(["recover".as_ref(), container.as_os_str(), "--dry-run".as_ref()]);
    assert_eq!(dry.code, 0, "{dry:?}");
}

#[test]
fn compress_and_decompress_round_trip_a_tree() {
    let scratch = Scratch::new();
    let model = safetensors(51, 64 * 1024);
    let nested = safetensors(52, 16 * 1024);
    let config = b"{\"not\": \"weights\"}".to_vec();
    scratch.write("tree/model.safetensors", &model);
    scratch.write("tree/sub/nested.safetensors", &nested);
    scratch.write("tree/config.json", &config);
    #[cfg(unix)]
    std::os::unix::fs::symlink(scratch.path("tree/model.safetensors"), scratch.path("tree/link.safetensors"))
        .expect("link");
    let tree = scratch.path("tree");

    let no_r = ziptensors(["compress".as_ref(), tree.as_os_str()]);
    assert_eq!(no_r.code, 2, "{no_r:?}");
    assert!(no_r.stderr.contains("pass -r"), "{no_r:?}");

    let dry = ziptensors(["compress".as_ref(), "-r".as_ref(), tree.as_os_str(), "--dry-run".as_ref()]);
    assert_eq!(dry.code, 0, "{dry:?}");
    assert!(dry.stdout.contains("would: "), "{dry:?}");
    assert!(dry.stdout.contains("dry run, nothing written"), "{dry:?}");
    assert!(!scratch.path("tree/model.ziptensors").exists());

    let compressed = ziptensors(["compress".as_ref(), "-r".as_ref(), tree.as_os_str(), "--threads".as_ref(), "2".as_ref()]);
    assert_eq!(compressed.code, 0, "{compressed:?}");
    assert!(compressed.stdout.contains("compressed: "), "{compressed:?}");
    assert!(compressed.stdout.contains("2 compressed"), "{compressed:?}");
    assert!(compressed.stdout.contains("not a weight file: "), "{compressed:?}");
    assert!(!scratch.path("tree/model.safetensors").exists());
    assert!(scratch.path("tree/model.ziptensors").exists());
    assert!(scratch.path("tree/sub/nested.ziptensors").exists());
    assert!(fs::read(scratch.path("tree/config.json")).expect("untouched") == config);
    if cfg!(unix) {
        assert!(compressed.stderr.contains("warning: 1 symbolic links were skipped"), "{compressed:?}");
        assert!(compressed.stderr.contains("--follow-symlinks"), "{compressed:?}");
        assert!(compressed.stderr.contains("Hugging Face"), "{compressed:?}");
    }

    // A single container needs no -r.
    let single = scratch.path("tree/sub/nested.ziptensors");
    let one = ziptensors(["decompress".as_ref(), single.as_os_str(), "-k".as_ref()]);
    assert_eq!(one.code, 0, "{one:?}");
    assert!(fs::read(scratch.path("tree/sub/nested.safetensors")).expect("restored") == nested);
    assert!(single.exists());

    let decompressed = ziptensors(["decompress".as_ref(), "-r".as_ref(), tree.as_os_str()]);
    assert_eq!(decompressed.code, 0, "{decompressed:?}");
    assert!(decompressed.stdout.contains("restored: "), "{decompressed:?}");
    // Both arrows of a line point from the container to the restored file.
    let restored_line = decompressed.stdout.lines().find(|line| line.starts_with("restored: ")).expect("a restored line");
    assert!(restored_line.contains("model.ziptensors -> "), "{restored_line}");
    assert!(restored_line.contains(&format!(" -> {} bytes)", model.len())), "{restored_line}");
    // The restored model and the nested pair (restored above with -k) are the bytes brought back.
    let bytes_line = decompressed.stdout.lines().find(|line| line.starts_with("bytes: ")).expect("a bytes line");
    assert!(
        bytes_line.ends_with(&format!(" in containers -> {} uncompressed", model.len() + nested.len())),
        "{bytes_line}"
    );
    assert!(fs::read(scratch.path("tree/model.safetensors")).expect("restored") == model);
    assert!(fs::read(scratch.path("tree/sub/nested.safetensors")).expect("restored") == nested);
    assert!(!scratch.path("tree/model.ziptensors").exists());
    assert!(!single.exists());
}

#[test]
fn decompress_fails_a_container_whose_restore_path_holds_other_bytes() {
    let scratch = Scratch::new();
    let model = safetensors(53, 4096);
    let file = scratch.write("dir/model.safetensors", &model);
    let compressed = ziptensors(["compress".as_ref(), file.as_os_str(), "-k".as_ref()]);
    assert_eq!(compressed.code, 0, "{compressed:?}");
    assert!(scratch.path("dir/model.ziptensors").exists());
    let other = safetensors(54, 4096);
    fs::write(&file, &other).expect("overwrite");
    let container = scratch.path("dir/model.ziptensors");
    let failed = ziptensors(["decompress".as_ref(), container.as_os_str()]);
    assert_eq!(failed.code, 1, "{failed:?}");
    assert!(failed.stdout.contains("FAILED: "), "{failed:?}");
    assert!(failed.stderr.contains("1 file(s) failed to decompress"), "{failed:?}");
    assert!(fs::read(&file).expect("kept") == other);
    let forced = ziptensors(["decompress".as_ref(), container.as_os_str(), "--force".as_ref()]);
    assert_eq!(forced.code, 0, "{forced:?}");
    assert!(fs::read(&file).expect("restored") == model);
}

#[cfg(any(target_os = "linux", windows))]
#[test]
fn mount_overlay_flags_and_sources_are_checked_before_anything_runs() {
    let scratch = Scratch::new();
    let file = scratch.write("plain.bin", b"not a directory");
    let dir = scratch.path("tree");
    fs::create_dir(&dir).expect("create tree");
    let point = scratch.path("mnt");
    let cases: [(&[&std::ffi::OsStr], &str); 5] = [
        (&[file.as_os_str(), point.as_os_str(), "--overlay".as_ref()], "is not a directory: --overlay"),
        (&[dir.as_os_str(), point.as_os_str()], "pass --overlay"),
        (&[dir.as_os_str(), point.as_os_str(), "--max-open-files".as_ref(), "2".as_ref()], "--overlay"),
        (&[dir.as_os_str(), point.as_os_str(), "--max-open-containers".as_ref(), "2".as_ref()], "--overlay"),
        (&[dir.as_os_str(), point.as_os_str(), "--overlay".as_ref(), "--max-open-containers".as_ref(), "0".as_ref()], "0"),
    ];
    for (args, says) in cases {
        let output = ziptensors(std::iter::once("mount".as_ref()).chain(args.iter().copied()));
        assert_eq!(output.code, 2, "{args:?}: {output:?}");
        assert!(output.stderr.contains(says), "{args:?}: {output:?}");
        assert_eq!(output.stdout, "", "{args:?}: {output:?}");
    }
}

/// A running `ziptensors mount` child on Linux. Drop unmounts lazily and kills the child, so a failed
/// assertion leaves no mount behind; [`MountChild::finish`] ends it the normal way.
#[cfg(target_os = "linux")]
struct MountChild {
    child: std::process::Child,
    mountpoint: std::path::PathBuf,
    stdout: std::sync::mpsc::Receiver<String>,
    stderr: Option<std::thread::JoinHandle<String>>,
}

#[cfg(target_os = "linux")]
impl MountChild {
    /// How long a step of the child may take before the test fails.
    const PATIENCE: std::time::Duration = std::time::Duration::from_secs(120);

    /// Starts the binary with `args`, mounting on `mountpoint`, and waits for its `mounted` line.
    fn start(args: &[&std::ffi::OsStr], mountpoint: &Path) -> Self {
        use std::io::{BufRead, BufReader, Read};
        use std::process::{Command, Stdio};

        let mut child = Command::new(env!("CARGO_BIN_EXE_ziptensors"))
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the mount");
        let out = child.stdout.take().expect("stdout is piped");
        let (send, stdout) = std::sync::mpsc::channel();
        // Lines go through a channel so that a child that never mounts fails the test, not hangs it.
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { return };
                if send.send(line).is_err() {
                    return;
                }
            }
        });
        let mut err = child.stderr.take().expect("stderr is piped");
        // Drained all the time: the INFO log of a mount would fill the pipe and stop the child.
        let stderr = Some(std::thread::spawn(move || {
            let mut text = String::new();
            if let Err(error) = err.read_to_string(&mut text) {
                text = format!("{text}\n(stderr unreadable: {error})");
            }
            text
        }));
        let mut mounted = Self { child, mountpoint: mountpoint.to_path_buf(), stdout, stderr };
        let line = mounted.stdout.recv_timeout(Self::PATIENCE);
        let Ok(line) = line else {
            let log = mounted.stop_log();
            panic!("no mounted line ({line:?}); stderr:\n{log}");
        };
        assert!(line.starts_with("mounted "), "{line}");
        mounted
    }

    /// Unmounts with `fusermount3 -u` as a user would, and returns the exit code and the next stdout
    /// line of the child.
    fn finish(mut self) -> (i32, String) {
        // Retried until PATIENCE: other tests of this binary fork children concurrently, and a fork
        // briefly duplicates every descriptor of this process, so a file just read under the mount
        // can still be open in a child that has not reached `exec` yet, and the kernel says EBUSY.
        let deadline = std::time::Instant::now() + Self::PATIENCE;
        loop {
            let status = std::process::Command::new("fusermount3")
                .arg("-u")
                .arg(&self.mountpoint)
                .status()
                .expect("run fusermount3");
            if status.success() {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "fusermount3 -u: {status}");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let deadline = std::time::Instant::now() + Self::PATIENCE;
        let code = loop {
            if let Some(status) = self.child.try_wait().expect("poll the mount") {
                break status.code().expect("the mount exits, it is not killed");
            }
            assert!(std::time::Instant::now() < deadline, "the mount did not end: {}", self.stop_log());
            std::thread::sleep(std::time::Duration::from_millis(50));
        };
        let line = self.stdout.recv_timeout(Self::PATIENCE).unwrap_or_default();
        (code, line)
    }

    /// Ends the child at once and returns what it logged.
    fn stop_log(&mut self) -> String {
        if let Err(error) = self.child.kill() {
            eprintln!("cannot kill the mount: {error}");
        }
        if let Err(error) = self.child.wait() {
            eprintln!("cannot wait for the mount: {error}");
        }
        let reader = self.stderr.take();
        reader.map_or_else(String::new, |reader| reader.join().unwrap_or_else(|_| "(stderr reader panicked)".to_owned()))
    }
}

#[cfg(target_os = "linux")]
impl Drop for MountChild {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        let unmounted = std::process::Command::new("fusermount3").arg("-u").arg("-z").arg(&self.mountpoint).status();
        if let Err(error) = unmounted {
            eprintln!("cannot run fusermount3 -u -z {}: {error}", self.mountpoint.display());
        }
        let log = self.stop_log();
        eprintln!("mount stopped by the test; its log:\n{log}");
    }
}

/// Why this machine cannot mount without privileges, or `None` if it can: the skip rule of the
/// mount crate's FUSE tests.
#[cfg(target_os = "linux")]
fn fuse_unavailable() -> Option<String> {
    if let Err(error) = fs::OpenOptions::new().read(true).write(true).open("/dev/fuse") {
        return Some(format!("/dev/fuse cannot be opened for reading and writing: {error}"));
    }
    let on_path = std::env::var_os("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join("fusermount3").is_file()));
    if on_path { None } else { Some("fusermount3 is not on PATH".to_owned()) }
}

#[cfg(target_os = "linux")]
#[test]
fn mount_overlay_shows_a_compressed_tree_as_its_originals() {
    if let Some(reason) = fuse_unavailable() {
        eprintln!("skipped: {reason}");
        return;
    }
    let scratch = Scratch::new();
    let model = safetensors(61, 64 * 1024);
    let nested = safetensors(62, 16 * 1024);
    let config = b"{\"not\": \"weights\"}".to_vec();
    scratch.write("tree/model.safetensors", &model);
    scratch.write("tree/sub/nested.safetensors", &nested);
    scratch.write("tree/config.json", &config);
    let tree = scratch.path("tree");
    let compressed = ziptensors(["compress".as_ref(), "-r".as_ref(), tree.as_os_str()]);
    assert_eq!(compressed.code, 0, "{compressed:?}");
    assert!(scratch.path("tree/model.ziptensors").exists());
    assert!(!scratch.path("tree/model.safetensors").exists());
    let mountpoint = scratch.path("mnt");
    fs::create_dir(&mountpoint).expect("create mountpoint");

    // One container open at a time, so reading the second closes the first.
    let mounted = MountChild::start(
        &[
            "mount".as_ref(),
            "--overlay".as_ref(),
            tree.as_os_str(),
            mountpoint.as_os_str(),
            "--max-open-containers".as_ref(),
            "1".as_ref(),
            "--cache-size".as_ref(),
            "33554432".as_ref(),
            "--threads".as_ref(),
            "2".as_ref(),
        ],
        &mountpoint,
    );
    assert_eq!(scratch.list("mnt"), ["config.json", "model.safetensors", "sub"]);
    assert_eq!(scratch.list("mnt/sub"), ["nested.safetensors"]);
    assert!(fs::read(mountpoint.join("model.safetensors")).expect("read through the mount") == model);
    assert!(fs::read(mountpoint.join("sub/nested.safetensors")).expect("read through the mount") == nested);
    assert!(fs::read(mountpoint.join("model.safetensors")).expect("read again") == model);
    assert_eq!(fs::read(mountpoint.join("config.json")).expect("read through the mount"), config);
    let (code, line) = mounted.finish();
    assert_eq!(code, 0, "{line}");
    assert!(line.starts_with("unmounted "), "{line}");
    let mounts = fs::read_to_string("/proc/mounts").expect("read /proc/mounts");
    let shown = mountpoint.display().to_string();
    assert!(!mounts.lines().any(|line| line.contains(&shown)), "{mounts}");
}
