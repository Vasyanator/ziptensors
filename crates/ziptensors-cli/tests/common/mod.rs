//! File: tests/common/mod.rs
//! Purpose: what the process tests of the `ziptensors` binary share.
//!
//! Main responsibilities:
//! - [`Scratch`], a unique directory under the system temp directory, removed on drop;
//! - [`ziptensors`], which runs the real binary and captures its streams;
//! - [`bytes`], seeded compressible data, and [`safetensors`], a minimal real source file;
//! - [`hostile_pack`], a container with chosen member names that no writer would produce;
//! - [`parse_json`], a strict parser, because the crate has no JSON dependency.
//!
//! Key structures: [`Scratch`], [`Output`], [`Value`].

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::{env, fs, process};

use ziptensors_core::format::{FOOTER_LEN, Footer, HEADER_LEN, Header, IndexBuilder, MemberRecord};
use ziptensors_core::Profile;

static NEXT_SCRATCH: AtomicU32 = AtomicU32::new(0);

/// BLAKE3 of no bytes.
const EMPTY_BLAKE3: [u8; 32] = [
    0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6, 0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49,
    0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7, 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62,
];

/// A directory of one test, removed on drop.
pub struct Scratch(PathBuf);

impl Scratch {
    /// A new empty directory.
    pub fn new() -> Self {
        let id = NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("ziptensors-cli-{}-{id}", process::id()));
        fs::create_dir_all(&path).expect("create scratch directory");
        Self(path)
    }

    /// The path of `relative` in this directory.
    pub fn path(&self, relative: &str) -> PathBuf {
        self.0.join(relative)
    }

    /// Writes `content` at `relative`, with its parents.
    pub fn write(&self, relative: &str, content: &[u8]) -> PathBuf {
        let path = self.path(relative);
        let parent = path.parent().expect("a file in the scratch directory has a parent");
        fs::create_dir_all(parent).expect("create parent");
        fs::write(&path, content).expect("write file");
        path
    }

    /// The names directly in `relative`, sorted.
    pub fn list(&self, relative: &str) -> Vec<String> {
        let entries = fs::read_dir(self.path(relative)).expect("list directory");
        let mut names: Vec<_> = entries
            .map(|entry| entry.expect("read entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort_unstable();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

/// What a run of the binary did.
#[derive(Debug)]
pub struct Output {
    /// Exit code.
    pub code: i32,
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
}

/// Runs the binary with `args`. Paths go through `arg`, so they need no UTF-8.
pub fn ziptensors<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let output = Command::new(env!("CARGO_BIN_EXE_ziptensors"))
        .args(args)
        .output()
        .expect("run the binary");
    Output {
        code: output.status.code().expect("the binary exits, it is not killed"),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// `len` bytes with low-entropy runs from `seed`, by splitmix64, so that every codec path has work.
pub fn bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|at| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let noise = (z ^ (z >> 31)).to_le_bytes()[0];
            if at % 4 == 3 { noise & 1 } else { noise }
        })
        .collect()
}

/// A safetensors file of one `F32` tensor `t` of `elements` elements.
pub fn safetensors(seed: u64, elements: usize) -> Vec<u8> {
    let data = bytes(seed, elements * 4);
    let header = format!(
        "{{\"t\":{{\"dtype\":\"F32\",\"shape\":[{elements}],\"data_offsets\":[0,{}]}}}}",
        data.len()
    );
    let header_len = u64::try_from(header.len()).expect("a short header");
    let mut file = header_len.to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    file
}

/// Writes a container of empty members called `names` to `path`. A real writer refuses a name like
/// `../x`, and opening a file does not look at names, so this is what reaches a restore.
pub fn hostile_pack(path: &Path, names: &[&str]) {
    let mut index = IndexBuilder::new();
    for name in names {
        let name = index.add_string(name).expect("add name");
        let member = MemberRecord {
            name,
            file_len: 0,
            blake3: EMPTY_BLAKE3,
            first_extent: 0,
            extent_count: 0,
            first_tensor: 0,
            tensor_count: 0,
            profile: Profile::Opaque,
        };
        index.add_member(member).expect("add member");
    }
    let raw = index.serialize();
    let stored = zstd::bulk::compress(&raw, 3).expect("compress index");
    let stored_len = u64::try_from(stored.len()).expect("a short index");
    let mut file = Header::current().to_bytes().to_vec();
    file.extend_from_slice(&stored);
    let generation = ziptensors_core::format::GENERATION_INITIAL;
    file.extend_from_slice(&Footer::for_index(HEADER_LEN, stored_len, &raw, generation).to_bytes());
    assert_eq!(u64::try_from(file.len()).ok(), Some(HEADER_LEN + stored_len + FOOTER_LEN));
    fs::write(path, file).expect("write hostile pack");
}

/// A JSON value. `true`, `false`, `null` and non-integers are rejected: the schema has none.
#[derive(Debug, PartialEq, Eq)]
pub enum Value {
    /// A string.
    String(String),
    /// A number.
    Number(u64),
    /// An array.
    Array(Vec<Value>),
    /// An object.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The value of `key`, which must exist.
    pub fn get(&self, key: &str) -> &Self {
        let Self::Object(fields) = self else { panic!("not an object: {self:?}") };
        let found = fields.iter().find(|(name, _)| name == key);
        &found.unwrap_or_else(|| panic!("no key {key} in {self:?}")).1
    }

    /// This number.
    pub fn number(&self) -> u64 {
        let Self::Number(value) = self else { panic!("not a number: {self:?}") };
        *value
    }

    /// This string.
    pub fn string(&self) -> &str {
        let Self::String(value) = self else { panic!("not a string: {self:?}") };
        value
    }

    /// This array.
    pub fn array(&self) -> &[Self] {
        let Self::Array(items) = self else { panic!("not an array: {self:?}") };
        items
    }
}

/// A strict recursive descent over `text`.
struct Parser<'a> {
    rest: &'a str,
}

impl Parser<'_> {
    /// Takes `expected` from the front, or fails.
    fn eat(&mut self, expected: char) -> Result<(), String> {
        self.rest = self.rest.strip_prefix(expected).ok_or_else(|| {
            format!("expected {expected:?} at {:?}", self.rest.get(..20).unwrap_or(self.rest))
        })?;
        Ok(())
    }

    /// Whether the input continues with `next`, taking it if so.
    fn accept(&mut self, next: char) -> bool {
        self.rest.strip_prefix(next).is_some_and(|rest| {
            self.rest = rest;
            true
        })
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat('"')?;
        let mut text = String::new();
        let mut chars = self.rest.char_indices();
        while let Some((at, character)) = chars.next() {
            match character {
                '"' => {
                    self.rest = &self.rest[at + 1..];
                    return Ok(text);
                }
                '\\' => match chars.next().map(|(_, escape)| escape) {
                    Some('"') => text.push('"'),
                    Some('\\') => text.push('\\'),
                    Some('u') => {
                        let mut value = 0;
                        for _ in 0..4 {
                            let digit = chars.next().and_then(|(_, digit)| digit.to_digit(16));
                            value = value * 16 + digit.ok_or("bad \\u escape")?;
                        }
                        text.push(char::from_u32(value).ok_or("bad \\u value")?);
                    }
                    other => return Err(format!("bad escape {other:?}")),
                },
                control if control < ' ' => return Err("raw control in a string".to_owned()),
                other => text.push(other),
            }
        }
        Err("unterminated string".to_owned())
    }

    fn number(&mut self) -> Result<u64, String> {
        let end = self.rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(self.rest.len());
        let (digits, rest) = self.rest.split_at(end);
        self.rest = rest;
        digits.parse().map_err(|error| format!("bad number {digits:?}: {error}"))
    }

    fn value(&mut self) -> Result<Value, String> {
        if self.rest.starts_with('"') {
            return self.string().map(Value::String);
        }
        if self.accept('[') {
            let mut items = Vec::new();
            if self.accept(']') {
                return Ok(Value::Array(items));
            }
            loop {
                items.push(self.value()?);
                if self.accept(']') {
                    return Ok(Value::Array(items));
                }
                self.eat(',')?;
            }
        }
        if self.accept('{') {
            let mut fields = Vec::new();
            if self.accept('}') {
                return Ok(Value::Object(fields));
            }
            loop {
                let key = self.string()?;
                self.eat(':')?;
                fields.push((key, self.value()?));
                if self.accept('}') {
                    return Ok(Value::Object(fields));
                }
                self.eat(',')?;
            }
        }
        self.number().map(Value::Number)
    }
}

/// Parses one JSON document followed by a line end, and nothing else.
pub fn parse_json(text: &str) -> Value {
    let mut parser = Parser { rest: text };
    let value = parser.value().unwrap_or_else(|error| panic!("{error} in {text:?}"));
    assert_eq!(parser.rest, "\n", "text after the document");
    value
}

/// `value` of a `name: value` line of a summary.
pub fn summary_line<'a>(text: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name}: ");
    let line = text.lines().find_map(|line| line.strip_prefix(prefix.as_str()));
    line.unwrap_or_else(|| panic!("no line {name} in {text}"))
}
