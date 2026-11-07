//! File: json.rs
//! Purpose: a value that prints as JSON, for `inspect --json`.
//!
//! Key structures: [`Json`].
//!
//! Implementation notes: output only, and only the kinds the document uses, which keeps the crate free
//! of a serialization dependency for one document. Numbers are `u64` printed exactly; a consumer that
//! reads them as doubles loses precision above 2^53.

use std::fmt::{self, Display, Formatter, Write};

/// A JSON value. Object keys are fixed by this crate, need no escaping and keep their order.
#[derive(Debug)]
pub(crate) enum Json {
    /// A string, escaped on printing. Member names come from the file and may hold anything.
    String(String),
    /// A number.
    Number(u64),
    /// An array.
    Array(Vec<Json>),
    /// An object.
    Object(Vec<(&'static str, Json)>),
}

/// Prints `text` as a JSON string, per RFC 8259 section 7: `"`, `\` and controls are escaped.
fn write_string(f: &mut Formatter<'_>, text: &str) -> fmt::Result {
    f.write_char('"')?;
    for character in text.chars() {
        match character {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            control if u32::from(control) < 0x20 => write!(f, "\\u{:04x}", u32::from(control))?,
            other => f.write_char(other)?,
        }
    }
    f.write_char('"')
}

impl Display for Json {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(text) => write_string(f, text),
            Self::Number(value) => write!(f, "{value}"),
            Self::Array(items) => {
                f.write_char('[')?;
                for (at, item) in items.iter().enumerate() {
                    if at != 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_char(']')
            }
            Self::Object(fields) => {
                f.write_char('{')?;
                for (at, (key, value)) in fields.iter().enumerate() {
                    if at != 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "\"{key}\":{value}")?;
                }
                f.write_char('}')
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn strings_escape_quotes_backslashes_and_controls() {
        let text = Json::String("a\"b\\c\nd\u{1}é".to_owned()).to_string();
        assert_eq!(text, "\"a\\\"b\\\\c\\u000ad\\u0001é\"");
    }

    #[test]
    fn containers_keep_order_and_print_compact() {
        let value = Json::Object(vec![
            ("z", Json::Number(u64::MAX)),
            ("a", Json::Array(vec![Json::Number(1), Json::String("x".to_owned())])),
            ("empty", Json::Array(Vec::new())),
        ]);
        assert_eq!(value.to_string(), "{\"z\":18446744073709551615,\"a\":[1,\"x\"],\"empty\":[]}");
    }
}
