//! File: `profile/legacy_torch.rs`
//! Purpose: recognizes a legacy (pre-1.6, pickle stream) `torch.save` file by its first bytes.
//!
//! Main responsibilities: compares the file start with the pickle of torch's magic number, the
//! first object every legacy save writes, for pickle protocols 2 to 5.
//!
//! Key functions: [`matches`].
//!
//! Implementation notes: nothing is unpickled; the check is a byte comparison over at most
//! [`PREFIX_BYTES`] bytes. The constants come from `torch/serialization.py` of the `PyTorch`
//! source: `MAGIC_NUMBER = 0x1950A86A20F9469CFC6C`, `DEFAULT_PROTOCOL = 2`, and `_legacy_save`
//! dumps `MAGIC_NUMBER` first with `pickle_module.dump(..., protocol=pickle_protocol)`. The byte
//! sequences below are `CPython`'s `pickle.dumps(MAGIC_NUMBER, protocol=N)`. Such a file has no
//! profile id: it is compressed as opaque bytes, so this module only answers "is it a weight file".

use crate::error::Error;
use crate::io::RandomAccess;

/// Pickle `PROTO` opcode, followed by the protocol number.
const PROTO: u8 = 0x80;

/// Pickle `FRAME` opcode (protocol 4 and later), followed by an 8-byte little-endian length.
const FRAME: u8 = 0x95;

/// `LONG1` with a 10-byte payload (torch's 80-bit magic number, little endian), then `STOP`.
/// This is the whole pickle body of `MAGIC_NUMBER` in every protocol from 2 on.
const MAGIC_BODY: [u8; 13] =
    [0x8A, 0x0A, 0x6C, 0xFC, 0x9C, 0x46, 0xF9, 0x20, 0x6A, 0xA8, 0x50, 0x19, 0x2E];

/// Bytes of a `FRAME` opcode with its length field.
const FRAME_HEADER_BYTES: usize = 9;

/// Longest prefix the check compares: `PROTO N`, a frame header, and the body (protocols 4, 5).
const PREFIX_BYTES: usize = 2 + FRAME_HEADER_BYTES + MAGIC_BODY.len();

/// Whether `src` starts with the pickle of torch's legacy magic number, in pickle protocol 2, 3,
/// 4 or 5.
///
/// Reads at most [`PREFIX_BYTES`] bytes from offset 0 and never interprets anything past them.
/// For protocols 4 and 5 the frame must hold exactly the magic number's body: a longer frame
/// would hold the next pickle too, which `torch.load` cannot read. Not recognized: a save made
/// with a custom `pickle_module` that writes other bytes, protocol 6 or later, protocols 0 and 1,
/// and the older tar-based format.
///
/// # Errors
/// Returns the error of a failed read of `src`.
pub(super) fn matches(src: &dyn RandomAccess) -> Result<bool, Error> {
    let mut head = [0u8; PREFIX_BYTES];
    // A length that does not fit usize is far longer than the prefix.
    let take = usize::try_from(src.len()).map_or(PREFIX_BYTES, |len| len.min(PREFIX_BYTES));
    let head = &mut head[..take];
    src.read_exact_at(0, head)?;
    Ok(is_magic_pickle(head))
}

/// Whether `head` begins with one of the accepted pickles of the magic number.
fn is_magic_pickle(head: &[u8]) -> bool {
    let Some((&[PROTO, version], rest)) = head.split_first_chunk::<2>() else {
        return false;
    };
    match version {
        2 | 3 => rest.starts_with(&MAGIC_BODY),
        4 | 5 => {
            let Some((frame, body)) = rest.split_first_chunk::<FRAME_HEADER_BYTES>() else {
                return false;
            };
            let [opcode, length @ ..] = *frame;
            let body_len = u64::try_from(MAGIC_BODY.len()).ok();
            opcode == FRAME
                && Some(u64::from_le_bytes(length)) == body_len
                && body.starts_with(&MAGIC_BODY)
        }
        _ => false,
    }
}
