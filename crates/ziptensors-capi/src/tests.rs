//! File: tests.rs
//! Purpose: tests of every entry point, called through its `extern "C"` declaration.
//!
//! Main responsibilities: roundtrips, invalid arguments, error text, the panic barrier and the status
//! mapping.
//!
//! Key structures: [`Scratch`], a temporary directory; [`Fixture`], an encoded container.
//!
//! Implementation notes: these live inside the crate since the panic injection of `barrier.rs` exists
//! only in its test build. Input is generated from a seed; scratch goes under the system temp directory
//! and is removed by `Drop`.

use std::env;
use std::ffi::{CStr, CString, c_char};
use std::fs;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::process;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use ziptensors_core::{Dtype, ErrorCode};

use crate::barrier::panic_on_next_entry;
use crate::{
    ZT_ABI_VERSION, ZtContainer, ZtEncodeOptions, ZtMember, ZtMode, ZtOpenOptions, ZtPack, ZtStatus,
    ZtTensorDesc, ZtVerify, zt_abi_version, zt_close, zt_decode_tensor_into, zt_encode_file,
    zt_last_error, zt_member_close, zt_member_count, zt_member_len, zt_member_name, zt_member_open,
    zt_open, zt_pack_add_file,
    zt_pack_finish, zt_pack_free, zt_pack_new, zt_read_at, zt_restore_file, zt_tensor_count,
    zt_tensor_info,
};

/// A unique directory under the system temp directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("ziptensors-capi-{}-{id}", process::id()));
        fs::create_dir(&dir).expect("create the scratch directory");
        Self(dir)
    }

    /// A path in the directory, as a C string.
    fn path(&self, name: &str) -> CString {
        c_path(&self.0.join(name))
    }

    /// The names now in the directory, sorted.
    fn listing(&self) -> Vec<String> {
        let entries = fs::read_dir(&self.0).expect("list the scratch directory");
        let mut names: Vec<String> = entries
            .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
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

fn c_path(path: &Path) -> CString {
    CString::new(path.to_str().expect("a UTF-8 temp path")).expect("no NUL in a temp path")
}

/// `len` bytes of seeded noise.
fn noise(len: usize) -> Vec<u8> {
    let mut state = 0x5EED_u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_le_bytes()[3]
        })
        .collect()
}

/// Length of the tensor `alpha`, which has shape `[4, 256]` of F32.
const ALPHA_LEN: usize = 4 * 256 * 4;
/// Length of the tensor `beta`, which has shape `[32]` of U8.
const BETA_LEN: usize = 32;

/// A safetensors file with the tensors `alpha` and `beta`, and where the payload starts.
fn safetensors_file() -> (Vec<u8>, usize) {
    let end = ALPHA_LEN + BETA_LEN;
    let header = format!(
        concat!(
            r#"{{"alpha":{{"dtype":"F32","shape":[4,256],"data_offsets":[0,{}]}},"#,
            r#""beta":{{"dtype":"U8","shape":[32],"data_offsets":[{},{}]}}}}"#,
        ),
        ALPHA_LEN, ALPHA_LEN, end,
    );
    let mut file = u64::try_from(header.len()).expect("short").to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    let payload_at = file.len();
    file.extend_from_slice(&noise(end));
    (file, payload_at)
}

/// A container of one member, encoded from [`safetensors_file`].
struct Fixture {
    scratch: Scratch,
    source: Vec<u8>,
    payload_at: usize,
    input: CString,
    output: CString,
}

impl Fixture {
    fn new() -> Self {
        let scratch = Scratch::new();
        let (source, payload_at) = safetensors_file();
        let input = scratch.path("input.safetensors");
        fs::write(scratch.0.join("input.safetensors"), &source).expect("write the input");
        let output = scratch.path("single.ziptensors");
        let options = encode_options(ZtMode::Fast);
        // SAFETY: both strings are live, and the options are a valid struct of the right size.
        let status = unsafe { zt_encode_file(input.as_ptr(), output.as_ptr(), &raw const options) };
        assert_eq!(status, ZtStatus::Ok, "{}", last_error());
        Self { scratch, source, payload_at, input, output }
    }

    /// Opens the container and its single member.
    fn open(&self) -> (*mut ZtContainer, *mut ZtMember) {
        let container = open(&self.output);
        let mut member = ptr::null_mut();
        // SAFETY: the handle is live and `member` is writable.
        let status = unsafe { zt_member_open(container, ptr::null(), &raw mut member) };
        assert_eq!(status, ZtStatus::Ok, "{}", last_error());
        (container, member)
    }
}

fn open(path: &CStr) -> *mut ZtContainer {
    let options = open_options();
    let mut container = ptr::null_mut();
    // SAFETY: the string and the options are live, and `container` is writable.
    let status = unsafe { zt_open(path.as_ptr(), &raw const options, &raw mut container) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    container
}

fn open_options() -> ZtOpenOptions {
    ZtOpenOptions {
        struct_size: size_of::<ZtOpenOptions>(),
        threads: 1,
        max_index_bytes: 0,
        max_total_decoded: 0,
        max_decode_amplification: 0,
        verify: ZtVerify::Chunk as i32,
    }
}

fn encode_options(mode: ZtMode) -> ZtEncodeOptions {
    ZtEncodeOptions { struct_size: size_of::<ZtEncodeOptions>(), threads: 1, mode: mode as i32 }
}

/// Closes both handles of [`Fixture::open`].
fn close(container: *mut ZtContainer, member: *mut ZtMember) {
    // SAFETY: both are live handles and are not used again.
    unsafe {
        zt_member_close(member);
        zt_close(container);
    }
}

/// The last error of this thread as a string.
fn last_error() -> String {
    let mut text = vec![0; 4096];
    // SAFETY: `text` holds `text.len()` bytes.
    let len = unsafe { zt_last_error(text.as_mut_ptr().cast(), text.len()) };
    assert!(len < text.len());
    text.truncate(len);
    String::from_utf8(text).expect("the error text is UTF-8")
}

/// Asserts a status and that the error text holds `needle`.
#[track_caller]
fn assert_fails(status: ZtStatus, expected: ZtStatus, needle: &str) {
    assert_eq!(status, expected, "{}", last_error());
    let text = last_error();
    assert!(text.contains(needle), "{needle:?} is not in {text:?}");
}

/// A struct of the tensor table, with its name as a Rust string.
fn tensor(member: *const ZtMember, index: usize) -> (ZtTensorDesc, String) {
    let mut desc = MaybeUninit::<ZtTensorDesc>::uninit();
    // SAFETY: the handle is live and `desc` is writable.
    let status = unsafe { zt_tensor_info(member, index, desc.as_mut_ptr()) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: the call succeeded, so it wrote the whole struct.
    let desc = unsafe { desc.assume_init() };
    // SAFETY: the name is NUL-terminated and lives as long as the member.
    let name = unsafe { CStr::from_ptr(desc.name) }.to_str().expect("UTF-8").to_owned();
    (desc, name)
}

#[test]
fn every_error_code_has_the_status_of_the_same_value() {
    let codes = [
        ErrorCode::Io,
        ErrorCode::NotZiptensors,
        ErrorCode::UnsupportedVersion,
        ErrorCode::UnsupportedFeature,
        ErrorCode::UnknownCodec,
        ErrorCode::MalformedIndex,
        ErrorCode::BudgetExceeded,
        ErrorCode::ChunkCorrupt,
        ErrorCode::ChecksumMismatch,
        ErrorCode::OutOfBounds,
        ErrorCode::DestinationSize,
        ErrorCode::TensorNotFound,
        ErrorCode::MemberNotFound,
        ErrorCode::UnsafeMemberName,
        ErrorCode::VerifyFailed,
        ErrorCode::InputChanged,
        ErrorCode::Zstd,
        ErrorCode::SymlinkInput,
        ErrorCode::OutputIsInput,
        ErrorCode::CommittedNotDurable,
    ];
    for code in codes {
        assert_eq!(ZtStatus::from_code(code) as i32, code.as_i32(), "{code:?}");
        assert!(code.as_i32() > 0);
    }
    assert_eq!(ZtStatus::Ok as i32, 0);
}

#[test]
fn dtype_constants_are_the_ids_of_core() {
    let pairs = [
        (crate::ZT_DTYPE_NONE, Dtype::Unknown(0)),
        (crate::ZT_DTYPE_BOOL, Dtype::Bool),
        (crate::ZT_DTYPE_U8, Dtype::U8),
        (crate::ZT_DTYPE_I8, Dtype::I8),
        (crate::ZT_DTYPE_U16, Dtype::U16),
        (crate::ZT_DTYPE_I16, Dtype::I16),
        (crate::ZT_DTYPE_U32, Dtype::U32),
        (crate::ZT_DTYPE_I32, Dtype::I32),
        (crate::ZT_DTYPE_U64, Dtype::U64),
        (crate::ZT_DTYPE_I64, Dtype::I64),
        (crate::ZT_DTYPE_F16, Dtype::F16),
        (crate::ZT_DTYPE_BF16, Dtype::Bf16),
        (crate::ZT_DTYPE_F32, Dtype::F32),
        (crate::ZT_DTYPE_F64, Dtype::F64),
        (crate::ZT_DTYPE_F8_E4M3, Dtype::F8E4M3),
        (crate::ZT_DTYPE_F8_E5M2, Dtype::F8E5M2),
        (crate::ZT_DTYPE_NIBBLE, Dtype::Nibble),
        (crate::ZT_DTYPE_Q8_0, Dtype::Q80),
        (crate::ZT_DTYPE_Q4_K, Dtype::Q4K),
        (crate::ZT_DTYPE_Q5_K, Dtype::Q5K),
        (crate::ZT_DTYPE_Q6_K, Dtype::Q6K),
        (crate::ZT_DTYPE_Q4_0, Dtype::Q40),
        (crate::ZT_DTYPE_Q4_1, Dtype::Q41),
        (crate::ZT_DTYPE_Q5_0, Dtype::Q50),
        (crate::ZT_DTYPE_Q5_1, Dtype::Q51),
        (crate::ZT_DTYPE_Q8_1, Dtype::Q81),
        (crate::ZT_DTYPE_Q2_K, Dtype::Q2K),
        (crate::ZT_DTYPE_Q3_K, Dtype::Q3K),
        (crate::ZT_DTYPE_Q8_K, Dtype::Q8K),
        (crate::ZT_DTYPE_IQ2_XXS, Dtype::Iq2Xxs),
        (crate::ZT_DTYPE_IQ2_XS, Dtype::Iq2Xs),
        (crate::ZT_DTYPE_IQ3_XXS, Dtype::Iq3Xxs),
        (crate::ZT_DTYPE_IQ1_S, Dtype::Iq1S),
        (crate::ZT_DTYPE_IQ4_NL, Dtype::Iq4Nl),
        (crate::ZT_DTYPE_IQ3_S, Dtype::Iq3S),
        (crate::ZT_DTYPE_IQ2_S, Dtype::Iq2S),
        (crate::ZT_DTYPE_IQ4_XS, Dtype::Iq4Xs),
        (crate::ZT_DTYPE_IQ1_M, Dtype::Iq1M),
        (crate::ZT_DTYPE_TQ1_0, Dtype::Tq10),
        (crate::ZT_DTYPE_TQ2_0, Dtype::Tq20),
        (crate::ZT_DTYPE_MXFP4, Dtype::Mxfp4),
    ];
    for (id, dtype) in pairs {
        assert_eq!(Dtype::from_wire(id), dtype);
    }
    // One constant per registered id, plus `ZT_DTYPE_NONE`.
    let registered =
        (0..=u16::MAX).filter(|&id| !matches!(Dtype::from_wire(id), Dtype::Unknown(_))).count();
    assert_eq!(pairs.len(), registered + 1);
}

#[test]
fn members_are_listed_with_terminated_names() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let mut count = 0;
    // SAFETY: the handle is live and the slot is writable.
    assert_eq!(unsafe { zt_member_count(container, &raw mut count) }, ZtStatus::Ok);
    assert_eq!(count, 1);
    let (mut name, mut name_len) = (ptr::null(), 0);
    // SAFETY: the handle is live and the slots are writable.
    let status = unsafe { zt_member_name(container, 0, &raw mut name, &raw mut name_len) };
    assert_eq!(status, ZtStatus::Ok);
    // SAFETY: the name is NUL-terminated and lives as long as `container`.
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    assert_eq!(bytes.len(), name_len);
    assert_ne!(bytes, b"");

    let (mut untouched, mut untouched_len) = (ptr::null(), 7);
    // SAFETY: as above.
    let status = unsafe { zt_member_name(container, 1, &raw mut untouched, &raw mut untouched_len) };
    assert_eq!(status, ZtStatus::InvalidArgument);
    assert!(untouched.is_null());
    assert_eq!(untouched_len, 7);
    close(container, member);
}

#[test]
fn member_listing_refuses_null() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let (mut name, mut name_len, mut count) = (ptr::null(), 0, 0);
    // SAFETY: every pointer is NULL or valid, and NULL is what is under test.
    unsafe {
        assert_eq!(zt_member_count(ptr::null(), &raw mut count), ZtStatus::NullArgument);
        assert_eq!(zt_member_count(container, ptr::null_mut()), ZtStatus::NullArgument);
        let null_handle = zt_member_name(ptr::null(), 0, &raw mut name, &raw mut name_len);
        assert_eq!(null_handle, ZtStatus::NullArgument);
        let null_name = zt_member_name(container, 0, ptr::null_mut(), &raw mut name_len);
        assert_eq!(null_name, ZtStatus::NullArgument);
        let null_len = zt_member_name(container, 0, &raw mut name, ptr::null_mut());
        assert_eq!(null_len, ZtStatus::NullArgument);
    }
    close(container, member);
}

#[test]
fn abi_version_matches_the_constant() {
    assert_eq!(zt_abi_version(), ZT_ABI_VERSION);
}

#[test]
fn handles_may_cross_threads() {
    fn shared<T: Send + Sync>() {}
    fn movable<T: Send>() {}
    shared::<ZtContainer>();
    shared::<ZtMember>();
    movable::<ZtPack>();
}

#[test]
fn reads_return_the_source_bytes() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let mut len = 0_u64;
    // SAFETY: the handle is live and `len` is writable.
    assert_eq!(unsafe { zt_member_len(member, &raw mut len) }, ZtStatus::Ok);
    assert_eq!(len, u64::try_from(fixture.source.len()).expect("small"));

    let end = fixture.source.len();
    for (start, end) in [(0, end), (7, 4099), (end - 1, end)] {
        let mut dst = vec![0; end - start];
        let offset = u64::try_from(start).expect("small");
        // SAFETY: the handle is live and `dst` holds `dst.len()` bytes.
        let status = unsafe { zt_read_at(member, offset, dst.as_mut_ptr(), dst.len()) };
        assert_eq!(status, ZtStatus::Ok, "{}", last_error());
        assert_eq!(dst, fixture.source[start..end]);
    }
    close(container, member);
}

#[test]
fn a_read_fills_uninitialized_memory() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let len = fixture.source.len();
    let mut dst: Vec<u8> = Vec::with_capacity(len);
    // SAFETY: the handle is live and the capacity holds `len` bytes. Nothing has initialized them.
    let status = unsafe { zt_read_at(member, 0, dst.as_mut_ptr(), len) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: the call succeeded, so every byte up to `len` was written.
    unsafe { dst.set_len(len) };
    assert_eq!(dst, fixture.source);
    close(container, member);
}

#[test]
fn tensors_are_described_and_decoded() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let mut count = 0;
    // SAFETY: the handle is live and `count` is writable.
    assert_eq!(unsafe { zt_tensor_count(member, &raw mut count) }, ZtStatus::Ok);
    assert_eq!(count, 2);

    let (alpha, name) = tensor(member, 0);
    assert_eq!(name, "alpha");
    assert_eq!(alpha.name_len, 5);
    assert_eq!(alpha.offset, u64::try_from(fixture.payload_at).expect("small"));
    assert_eq!(alpha.len, u64::try_from(ALPHA_LEN).expect("small"));
    assert_eq!(alpha.ndim, 2);
    // SAFETY: `ndim` dimensions live as long as the member.
    let shape = unsafe { std::slice::from_raw_parts(alpha.shape, alpha.ndim) };
    assert_eq!(shape, [4, 256]);
    assert_eq!(alpha.dtype, ziptensors_core::Dtype::F32.to_wire());

    let mut dst = vec![0; ALPHA_LEN];
    // SAFETY: the handle and the name are live, and `dst` holds `dst.len()` bytes.
    let status =
        unsafe { zt_decode_tensor_into(member, c"alpha".as_ptr(), dst.as_mut_ptr(), dst.len()) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    assert_eq!(dst, fixture.source[fixture.payload_at..fixture.payload_at + ALPHA_LEN]);

    let (beta, name) = tensor(member, 1);
    assert_eq!(name, "beta");
    assert_eq!(beta.dtype, ziptensors_core::Dtype::U8.to_wire());
    close(container, member);
}

#[test]
fn restore_writes_the_source_bytes() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let out = fixture.scratch.path("restored.bin");
    // SAFETY: the handle and the string are live.
    let status = unsafe { zt_restore_file(member, out.as_ptr()) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    let restored = fs::read(fixture.scratch.0.join("restored.bin")).expect("read the restored file");
    assert_eq!(restored, fixture.source);
    close(container, member);
}

#[test]
fn a_member_outlives_its_container() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    // SAFETY: a live handle, not used again.
    unsafe { zt_close(container) };
    let mut dst = vec![0; 16];
    // SAFETY: the member is live and `dst` holds 16 bytes.
    let status = unsafe { zt_read_at(member, 0, dst.as_mut_ptr(), dst.len()) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    assert_eq!(dst, fixture.source[..16]);
    // SAFETY: a live handle, not used again.
    unsafe { zt_member_close(member) };
}

#[test]
fn a_member_serves_threads_at_once() {
    // A wrapper, since the raw pointer alone is not `Send`. The test handle lives past the scope.
    #[derive(Clone, Copy)]
    struct Shared(*const ZtMember);
    // SAFETY: `ZtMember` is `Sync`, and the C contract allows calls from any number of threads.
    unsafe impl Send for Shared {}

    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let shared = Shared(member);
    let source = &fixture.source;
    thread::scope(|scope| {
        for offset in 0..4 {
            scope.spawn(move || {
                let handle = shared;
                let (desc, name) = tensor(handle.0, offset % 2);
                assert_eq!(desc.name_len, name.len());
                let mut dst = vec![0; 1024];
                // SAFETY: the handle outlives the scope and `dst` holds `dst.len()` bytes.
                let status = unsafe { zt_read_at(handle.0, 64, dst.as_mut_ptr(), dst.len()) };
                assert_eq!(status, ZtStatus::Ok, "{}", last_error());
                assert_eq!(dst, source[64..1088]);
            });
        }
    });
    close(container, member);
}

#[test]
fn null_arguments_are_named() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let path = c"x".as_ptr();
    let mut container_out = ptr::null_mut();
    let mut member_out = ptr::null_mut();
    let mut pack_out = ptr::null_mut();
    let mut count = 0;
    let mut len = 0;
    let mut byte = 0;
    let dst = &raw mut byte;
    let null: *const c_char = ptr::null();
    let mut desc = MaybeUninit::<ZtTensorDesc>::uninit();
    // SAFETY: every pointer is NULL or valid, and NULL is what these calls must survive.
    let calls = unsafe {
        [
            (zt_open(null, null.cast(), &raw mut container_out), "path"),
            (zt_open(path, null.cast(), ptr::null_mut()), "out"),
            (zt_member_open(ptr::null(), null, &raw mut member_out), "container"),
            (zt_member_open(container, null, ptr::null_mut()), "out"),
            (zt_member_len(ptr::null(), &raw mut len), "member"),
            (zt_member_len(member, ptr::null_mut()), "out"),
            (zt_read_at(ptr::null(), 0, dst, 1), "member"),
            (zt_read_at(member, 0, ptr::null_mut(), 1), "dst"),
            (zt_decode_tensor_into(ptr::null(), path, dst, 1), "member"),
            (zt_decode_tensor_into(member, null, dst, 1), "name"),
            (zt_decode_tensor_into(member, c"alpha".as_ptr(), ptr::null_mut(), 1), "dst"),
            (zt_tensor_count(ptr::null(), &raw mut count), "member"),
            (zt_tensor_count(member, ptr::null_mut()), "out"),
            (zt_tensor_info(ptr::null(), 0, desc.as_mut_ptr()), "member"),
            (zt_tensor_info(member, 0, ptr::null_mut()), "out"),
            (zt_restore_file(ptr::null(), path), "member"),
            (zt_restore_file(member, null), "path"),
            (zt_encode_file(null, path, ptr::null()), "input"),
            (zt_encode_file(path, null, ptr::null()), "output"),
            (zt_pack_new(null, ptr::null(), &raw mut pack_out), "output"),
            (zt_pack_new(path, ptr::null(), ptr::null_mut()), "out"),
            (zt_pack_add_file(ptr::null_mut(), path, path), "pack"),
            (zt_pack_finish(ptr::null_mut()), "pack"),
        ]
    };
    for (index, (status, argument)) in calls.into_iter().enumerate() {
        // Only the status survives the array: the text is of the last call, checked below.
        assert_eq!(status, ZtStatus::NullArgument, "call {index}, `{argument}`");
    }
    // SAFETY: NULL is what the call must survive, and the handle is live.
    let status = unsafe { zt_read_at(member, 0, ptr::null_mut(), 1) };
    assert_fails(status, ZtStatus::NullArgument, "zt_read_at: argument `dst` is NULL");
    assert!(container_out.is_null() && member_out.is_null() && pack_out.is_null());
    close(container, member);
}

#[test]
fn a_failure_sets_the_out_slot_to_null() {
    let fixture = Fixture::new();
    let junk = ptr::dangling_mut::<ZtContainer>();
    let mut container = junk;
    let missing = fixture.scratch.path("missing");
    // SAFETY: the string is live, and the slot is writable.
    let status = unsafe { zt_open(missing.as_ptr(), ptr::null(), &raw mut container) };
    assert_fails(status, ZtStatus::Io, "missing");
    assert!(container.is_null());

    let (open_container, member) = fixture.open();
    let mut slot = ptr::dangling_mut::<ZtMember>();
    // SAFETY: the handle and the string are live, and the slot is writable.
    let status = unsafe { zt_member_open(open_container, c"nope".as_ptr(), &raw mut slot) };
    assert_fails(status, ZtStatus::MemberNotFound, "nope");
    assert!(slot.is_null());
    close(open_container, member);

    let mut pack = ptr::dangling_mut::<ZtPack>();
    let unusable = fixture.scratch.path("no-such-directory/pack");
    // SAFETY: the string is live, and the slot is writable.
    let status = unsafe { zt_pack_new(unusable.as_ptr(), ptr::null(), &raw mut pack) };
    assert_eq!(status, ZtStatus::Io);
    assert!(pack.is_null());
}

#[test]
fn invalid_utf8_is_refused() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let bad = c"bad\xFF".as_ptr();
    let mut slot = ptr::null_mut();
    // SAFETY: the string is live and NUL-terminated, and the slot is writable.
    let status = unsafe { zt_open(bad, ptr::null(), &raw mut slot) };
    assert_fails(status, ZtStatus::InvalidUtf8, "`path`");
    let mut member_slot = ptr::null_mut();
    // SAFETY: as above, and the handle is live.
    let status = unsafe { zt_member_open(container, bad, &raw mut member_slot) };
    assert_fails(status, ZtStatus::InvalidUtf8, "`name`");
    let mut byte = 0;
    // SAFETY: as above, and `byte` is writable.
    let status = unsafe { zt_decode_tensor_into(member, bad, &raw mut byte, 1) };
    assert_fails(status, ZtStatus::InvalidUtf8, "`name`");
    close(container, member);
}

#[test]
fn a_foreign_struct_size_is_refused() {
    let fixture = Fixture::new();
    let output = fixture.output.as_ptr();
    let mut options = open_options();
    let input = fixture.input.as_ptr();
    let mut encode = encode_options(ZtMode::Default);
    let mut container = ptr::null_mut();
    for wrong in [0, 8, size_of::<ZtOpenOptions>() + 8] {
        options.struct_size = wrong;
        // SAFETY: the string and the struct are live, and the slot is writable.
        let status = unsafe { zt_open(output, &raw const options, &raw mut container) };
        assert_fails(status, ZtStatus::StructSize, "struct_size");
        assert!(container.is_null());
        encode.struct_size = wrong;
        let out = fixture.scratch.path("wrong");
        // SAFETY: the strings and the struct are live.
        let status = unsafe { zt_encode_file(input, out.as_ptr(), &raw const encode) };
        assert_fails(status, ZtStatus::StructSize, "struct_size");
    }
}

#[test]
fn only_the_first_field_is_read_before_the_size_is_known() {
    let fixture = Fixture::new();
    // A struct as an older header might have declared it: nothing but the size is backed by memory.
    let short = size_of::<usize>();
    let mut container = ptr::null_mut();
    let raw = (&raw const short).cast::<ZtOpenOptions>();
    // SAFETY: `zt_open` reads only the first word of a struct that states another size.
    let status = unsafe { zt_open(fixture.output.as_ptr(), raw, &raw mut container) };
    assert_fails(status, ZtStatus::StructSize, "struct_size");
}

#[test]
fn unknown_enum_values_are_refused() {
    let fixture = Fixture::new();
    let output = fixture.output.as_ptr();
    let mut options = open_options();
    options.verify = 77;
    let mut container = ptr::null_mut();
    // SAFETY: the string and the struct are live, and the slot is writable.
    let status = unsafe { zt_open(output, &raw const options, &raw mut container) };
    assert_fails(status, ZtStatus::InvalidArgument, "zt_verify");

    let input = fixture.input.as_ptr();
    let encode = ZtEncodeOptions { mode: -1, ..encode_options(ZtMode::Default) };
    let out = fixture.scratch.path("wrong");
    // SAFETY: the strings and the struct are live.
    let status = unsafe { zt_encode_file(input, out.as_ptr(), &raw const encode) };
    assert_fails(status, ZtStatus::InvalidArgument, "zt_mode");
    assert!(!fixture.scratch.0.join("wrong").exists());
}

#[test]
fn verify_off_and_null_options_open() {
    let fixture = Fixture::new();
    let output = fixture.output.as_ptr();
    let options = ZtOpenOptions { verify: ZtVerify::Off as i32, ..open_options() };
    let mut container = ptr::null_mut();
    // SAFETY: the string and the struct are live, and the slot is writable.
    let status = unsafe { zt_open(output, &raw const options, &raw mut container) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: a live handle, not used again.
    unsafe { zt_close(container) };
    // SAFETY: the string is live, the slot is writable, and NULL options are allowed.
    let status = unsafe { zt_open(output, ptr::null(), &raw mut container) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: a live handle, not used again.
    unsafe { zt_close(container) };
}

#[test]
fn a_budget_in_the_options_reaches_core() {
    let fixture = Fixture::new();
    let output = fixture.output.as_ptr();
    let options = ZtOpenOptions { max_total_decoded: 1, ..open_options() };
    let mut container = ptr::null_mut();
    // SAFETY: the string and the struct are live, and the slot is writable.
    let status = unsafe { zt_open(output, &raw const options, &raw mut container) };
    assert_eq!(status, ZtStatus::BudgetExceeded, "{}", last_error());
}

#[test]
fn a_refused_read_does_not_touch_the_destination() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let source_len = u64::try_from(fixture.source.len()).expect("small");
    let mut dst = vec![0xA5; 64];
    let cases = [(source_len, 1), (source_len - 1, 2), (u64::MAX, 2)];
    for (offset, len) in cases {
        // SAFETY: the handle is live and `dst` holds more than `len` bytes.
        let status = unsafe { zt_read_at(member, offset, dst.as_mut_ptr(), len) };
        assert_fails(status, ZtStatus::OutOfBounds, "zt_read_at");
    }
    // SAFETY: as above, with the name live.
    let status =
        unsafe { zt_decode_tensor_into(member, c"beta".as_ptr(), dst.as_mut_ptr(), BETA_LEN - 1) };
    assert_fails(status, ZtStatus::DestinationSize, "32");
    // SAFETY: as above.
    let status =
        unsafe { zt_decode_tensor_into(member, c"nope".as_ptr(), dst.as_mut_ptr(), BETA_LEN) };
    assert_fails(status, ZtStatus::TensorNotFound, "nope");
    assert!(dst.iter().all(|byte| *byte == 0xA5));
    close(container, member);
}

#[test]
fn a_length_a_slice_cannot_have_is_refused() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let mut byte = 0;
    // SAFETY: the check fails before anything is written through the pointer.
    let status = unsafe { zt_read_at(member, 0, &raw mut byte, usize::MAX) };
    assert_fails(status, ZtStatus::InvalidArgument, "`len`");
    close(container, member);
}

#[test]
fn an_empty_read_may_pass_null() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    // SAFETY: the handle is live, and NULL is allowed with a length of 0.
    let status = unsafe { zt_read_at(member, 0, ptr::null_mut(), 0) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    close(container, member);
}

#[test]
fn a_tensor_index_past_the_end_is_refused() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let mut desc = MaybeUninit::<ZtTensorDesc>::uninit();
    // SAFETY: the handle is live and `desc` is writable.
    let status = unsafe { zt_tensor_info(member, 2, desc.as_mut_ptr()) };
    assert_fails(status, ZtStatus::InvalidArgument, "tensor index 2 of 2");
    close(container, member);
}

#[test]
fn a_file_that_is_no_container_is_named_so() {
    let scratch = Scratch::new();
    fs::write(scratch.0.join("plain"), noise(4096)).expect("write");
    let path = scratch.path("plain");
    let mut container = ptr::null_mut();
    // SAFETY: the string is live and the slot is writable.
    let status = unsafe { zt_open(path.as_ptr(), ptr::null(), &raw mut container) };
    assert_fails(status, ZtStatus::NotZiptensors, "not a ziptensors file");
}

#[test]
fn the_last_error_cuts_safely_and_reports_its_length() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let name = c"\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9";
    let mut slot = ptr::null_mut();
    // SAFETY: the handle and the string are live, and the slot is writable.
    let status = unsafe { zt_member_open(container, name.as_ptr(), &raw mut slot) };
    assert_eq!(status, ZtStatus::MemberNotFound);
    let full = last_error();
    assert!(full.ends_with("éééé"), "{full}");

    // SAFETY: NULL and a zero capacity write nothing.
    assert_eq!(unsafe { zt_last_error(ptr::null_mut(), 100) }, full.len());
    let mut canary: c_char = 0x55;
    // SAFETY: a capacity of 0 writes nothing, so the single byte is plenty.
    assert_eq!(unsafe { zt_last_error(&raw mut canary, 0) }, full.len());
    assert_eq!(canary, 0x55);

    // Every capacity: a terminated, valid, prefix of the text, and nothing past `cap`.
    for cap in 1..=full.len() + 2 {
        let mut buf = vec![0x55_u8; cap + 8];
        // SAFETY: `buf` holds more than `cap` bytes.
        let len = unsafe { zt_last_error(buf.as_mut_ptr().cast(), cap) };
        assert_eq!(len, full.len());
        assert!(buf[cap..].iter().all(|byte| *byte == 0x55), "wrote past cap {cap}");
        let text = CStr::from_bytes_until_nul(&buf).expect("terminated").to_str().expect("UTF-8");
        assert!(full.starts_with(text));
        assert!(text.len() < cap);
    }
    close(container, member);
}

#[test]
fn a_nul_in_the_error_text_is_spelled_out() {
    crate::barrier::guarded("test", || {
        Err(crate::status::Failure::Invalid { detail: "a\0b".to_owned() })
    });
    assert!(last_error().ends_with("a\\0b"), "{}", last_error());
}

#[test]
fn the_last_error_belongs_to_the_thread_and_a_success_clears_it() {
    let fixture = Fixture::new();
    let mut slot = ptr::null_mut();
    let missing = fixture.scratch.path("missing");
    // SAFETY: the string is live and the slot is writable.
    let status = unsafe { zt_open(missing.as_ptr(), ptr::null(), &raw mut slot) };
    assert_eq!(status, ZtStatus::Io);
    assert_ne!(last_error(), "");
    thread::spawn(|| assert_eq!(last_error(), "")).join().expect("the other thread");
    assert_ne!(last_error(), "");
    let (container, member) = fixture.open();
    assert_eq!(last_error(), "");
    close(container, member);
}

#[test]
fn closing_null_does_nothing() {
    // SAFETY: NULL is a no-op by contract.
    unsafe {
        zt_close(ptr::null_mut());
        zt_member_close(ptr::null_mut());
        zt_pack_free(ptr::null_mut());
    }
}

/// A pack of the fixture's input under the names `one` and `dir/two`.
fn pack_two(fixture: &Fixture, output: &CStr) {
    let options = encode_options(ZtMode::Max);
    let mut pack = ptr::null_mut();
    // SAFETY: the strings and the struct are live, and the slot is writable.
    let status = unsafe { zt_pack_new(output.as_ptr(), &raw const options, &raw mut pack) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    for name in [c"one", c"dir/two"] {
        // SAFETY: the handle and the strings are live.
        let status = unsafe { zt_pack_add_file(pack, fixture.input.as_ptr(), name.as_ptr()) };
        assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    }
    // SAFETY: the handle is live.
    assert_eq!(unsafe { zt_pack_finish(pack) }, ZtStatus::Ok, "{}", last_error());
    // SAFETY: a live handle, not used again.
    unsafe { zt_pack_free(pack) };
}

#[test]
fn a_pack_roundtrips_its_members() {
    let fixture = Fixture::new();
    let output = fixture.scratch.path("pack.ziptensors");
    pack_two(&fixture, &output);
    let container = open(&output);
    let mut slot = ptr::null_mut();
    // SAFETY: the handle is live and the slot is writable; NULL asks for a single member.
    let status = unsafe { zt_member_open(container, ptr::null(), &raw mut slot) };
    assert_fails(status, ZtStatus::MemberNotFound, "single");
    for name in [c"one", c"dir/two"] {
        let mut member = ptr::null_mut();
        // SAFETY: the handle and the string are live, and the slot is writable.
        let status = unsafe { zt_member_open(container, name.as_ptr(), &raw mut member) };
        assert_eq!(status, ZtStatus::Ok, "{}", last_error());
        let mut dst = vec![0; fixture.source.len()];
        // SAFETY: the handle is live and `dst` holds `dst.len()` bytes.
        let status = unsafe { zt_read_at(member, 0, dst.as_mut_ptr(), dst.len()) };
        assert_eq!(status, ZtStatus::Ok, "{}", last_error());
        assert_eq!(dst, fixture.source);
        // SAFETY: a live handle, not used again.
        unsafe { zt_member_close(member) };
    }
    // SAFETY: a live handle, not used again.
    unsafe { zt_close(container) };
}

#[test]
fn every_mode_encodes() {
    let fixture = Fixture::new();
    let input = fixture.input.as_ptr();
    let modes = [ZtMode::Default, ZtMode::Fast, ZtMode::Balanced, ZtMode::Max];
    for (index, mode) in modes.into_iter().enumerate() {
        let out = fixture.scratch.path(&format!("mode-{index}"));
        let options = encode_options(mode);
        // SAFETY: the strings and the struct are live.
        let status = unsafe { zt_encode_file(input, out.as_ptr(), &raw const options) };
        assert_eq!(status, ZtStatus::Ok, "{}", last_error());
        let container = open(&out);
        // SAFETY: a live handle, not used again.
        unsafe { zt_close(container) };
    }
    // NULL options take the defaults of the entry point.
    let out = fixture.scratch.path("defaults");
    // SAFETY: the strings are live, and NULL options are allowed.
    let status = unsafe { zt_encode_file(fixture.input.as_ptr(), out.as_ptr(), ptr::null()) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
}

#[test]
fn a_pack_ends_at_finish() {
    let fixture = Fixture::new();
    let output = fixture.scratch.path("pack.ziptensors");
    let mut pack = ptr::null_mut();
    // SAFETY: the strings are live, the slot is writable, and NULL options are allowed.
    let status = unsafe { zt_pack_new(output.as_ptr(), ptr::null(), &raw mut pack) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: the handle is live.
    assert_eq!(unsafe { zt_pack_finish(pack) }, ZtStatus::Ok, "{}", last_error());
    // SAFETY: the handle stays allocated after a finish.
    let status = unsafe { zt_pack_finish(pack) };
    assert_fails(status, ZtStatus::InvalidState, "already finished");
    // SAFETY: as above, and the strings are live.
    let status = unsafe { zt_pack_add_file(pack, fixture.input.as_ptr(), c"late".as_ptr()) };
    assert_fails(status, ZtStatus::InvalidState, "already finished");
    // SAFETY: a live handle, not used again.
    unsafe { zt_pack_free(pack) };
}

#[test]
fn a_pack_refuses_what_core_refuses_and_goes_on() {
    let fixture = Fixture::new();
    let output = fixture.scratch.path("pack.ziptensors");
    let mut pack = ptr::null_mut();
    // SAFETY: the strings are live, the slot is writable, and NULL options are allowed.
    let status = unsafe { zt_pack_new(output.as_ptr(), ptr::null(), &raw mut pack) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    let input = fixture.input.as_ptr();
    // SAFETY: the handle and the strings are live.
    let status = unsafe { zt_pack_add_file(pack, input, c"../evil".as_ptr()) };
    assert_fails(status, ZtStatus::UnsafeMemberName, "evil");
    // SAFETY: as above.
    let status = unsafe { zt_pack_add_file(pack, input, c"good".as_ptr()) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: as above.
    let status = unsafe { zt_pack_add_file(pack, input, c"good".as_ptr()) };
    assert_fails(status, ZtStatus::MalformedIndex, "used twice");
    let missing = fixture.scratch.path("missing");
    // SAFETY: as above.
    let status = unsafe { zt_pack_add_file(pack, missing.as_ptr(), c"other".as_ptr()) };
    assert_eq!(status, ZtStatus::Io, "{}", last_error());
    // SAFETY: the handle is live.
    assert_eq!(unsafe { zt_pack_finish(pack) }, ZtStatus::Ok, "{}", last_error());
    // SAFETY: a live handle, not used again.
    unsafe { zt_pack_free(pack) };
}

#[test]
fn freeing_an_unfinished_pack_leaves_nothing() {
    let fixture = Fixture::new();
    let before = fixture.scratch.listing();
    let output = fixture.scratch.path("pack.ziptensors");
    let mut pack = ptr::null_mut();
    // SAFETY: the strings are live, the slot is writable, and NULL options are allowed.
    let status = unsafe { zt_pack_new(output.as_ptr(), ptr::null(), &raw mut pack) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: the handle and the strings are live.
    let status = unsafe { zt_pack_add_file(pack, fixture.input.as_ptr(), c"one".as_ptr()) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // SAFETY: a live handle, not used again.
    unsafe { zt_pack_free(pack) };
    assert_eq!(fixture.scratch.listing(), before);
}

/// Runs `call` with a panic injected into its barrier, and asserts the barrier held.
#[track_caller]
fn assert_barrier_holds(call: impl FnOnce() -> ZtStatus) {
    panic_on_next_entry();
    assert_fails(call(), ZtStatus::Panic, "internal panic: injected panic");
}

#[test]
fn every_status_entry_stops_a_panic() {
    let fixture = Fixture::new();
    let (container, member) = fixture.open();
    let path = fixture.output.as_ptr();
    let mut byte = 0;
    let dst = &raw mut byte;
    let mut count = 0;
    let mut len = 0;
    let mut desc = MaybeUninit::<ZtTensorDesc>::uninit();
    let mut container_out = ptr::null_mut();
    let mut member_out = ptr::null_mut();
    let mut pack_out = ptr::null_mut();
    let null: *const c_char = ptr::null();
    // SAFETY: arguments of every call below are valid, and the injected panic comes before any of
    // them is used.
    unsafe {
        assert_barrier_holds(|| zt_open(path, null.cast(), &raw mut container_out));
        assert_barrier_holds(|| zt_member_open(container, null, &raw mut member_out));
        assert_barrier_holds(|| zt_member_len(member, &raw mut len));
        assert_barrier_holds(|| zt_read_at(member, 0, dst, 1));
        assert_barrier_holds(|| zt_decode_tensor_into(member, c"beta".as_ptr(), dst, 1));
        assert_barrier_holds(|| zt_tensor_count(member, &raw mut count));
        assert_barrier_holds(|| zt_member_count(container, &raw mut count));
        assert_barrier_holds(|| zt_tensor_info(member, 0, desc.as_mut_ptr()));
        assert_barrier_holds(|| zt_restore_file(member, path));
        assert_barrier_holds(|| zt_encode_file(path, path, null.cast()));
        assert_barrier_holds(|| zt_pack_new(path, null.cast(), &raw mut pack_out));
        assert_barrier_holds(|| zt_pack_add_file(ptr::null_mut(), path, path));
        assert_barrier_holds(|| zt_pack_finish(ptr::null_mut()));
    }
    close(container, member);
}

#[test]
fn every_entry_without_a_status_stops_a_panic() {
    panic_on_next_entry();
    assert_eq!(zt_abi_version(), 0);
    assert!(last_error().contains("zt_abi_version: internal panic"));

    panic_on_next_entry();
    // SAFETY: the injected panic comes first, and the buffer would be valid anyway.
    assert_eq!(unsafe { zt_last_error(ptr::null_mut(), 0) }, 0);
    assert!(last_error().contains("zt_last_error: internal panic"));

    let closers: [(&str, fn()); 3] = [
        // SAFETY: NULL is a no-op by contract.
        ("zt_close", || unsafe { zt_close(ptr::null_mut()) }),
        // SAFETY: NULL is a no-op by contract.
        ("zt_member_close", || unsafe { zt_member_close(ptr::null_mut()) }),
        // SAFETY: NULL is a no-op by contract.
        ("zt_pack_free", || unsafe { zt_pack_free(ptr::null_mut()) }),
    ];
    for (name, close) in closers {
        panic_on_next_entry();
        close();
        assert!(last_error().contains(&format!("{name}: internal panic")));
    }
}

#[test]
fn a_panic_in_a_pack_call_ends_the_pack() {
    let fixture = Fixture::new();
    let output = fixture.scratch.path("pack.ziptensors");
    let mut pack = ptr::null_mut();
    // SAFETY: the strings are live, the slot is writable, and NULL options are allowed.
    let status = unsafe { zt_pack_new(output.as_ptr(), ptr::null(), &raw mut pack) };
    assert_eq!(status, ZtStatus::Ok, "{}", last_error());
    // A panic with the builder out of the handle, as a panic inside core would have it.
    // SAFETY: the handle is live and nothing else uses it.
    let builder = unsafe { (*pack).builder.take() };
    drop(builder);
    // SAFETY: as above, and the strings are live.
    let status = unsafe { zt_pack_add_file(pack, fixture.input.as_ptr(), c"one".as_ptr()) };
    assert_fails(status, ZtStatus::InvalidState, "panicked");
    // SAFETY: a live handle, not used again.
    unsafe { zt_pack_free(pack) };
}
