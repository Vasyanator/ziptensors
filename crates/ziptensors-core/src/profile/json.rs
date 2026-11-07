//! File: profile/json.rs
//! Purpose: the minimal pull JSON reader behind the safetensors header.
//!
//! Main responsibilities: walks one JSON text held in memory, decodes strings and reads unsigned
//! integers, and skips any value without building it.
//!
//! Key structures: [`Json`].
//!
//! Implementation notes: this crate has no JSON dependency, and the header needs so little of
//! JSON that a pull reader is smaller than a tree. Input is `&str`, so slices taken at the ASCII
//! delimiters are valid UTF-8 and need no second check. Nesting is capped, because skipping
//! recurses.

use super::{ParseResult, Stop};

/// Deepest nesting [`Json::skip_value`] follows.
const MAX_DEPTH: u32 = 64;

/// A cursor over a JSON text. Every method stops as malformed on invalid syntax.
pub(super) struct Json<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Json<'a> {
    /// Starts at the first byte of `text`.
    pub(super) fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    /// Consumes whitespace, then `byte`, which must be next.
    pub(super) fn expect(&mut self, byte: u8) -> ParseResult<()> {
        self.skip_space();
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(Stop::Malformed("unexpected JSON syntax"))
        }
    }

    /// Consumes whitespace, then `byte` if it is next. Returns whether it was.
    pub(super) fn eat(&mut self, byte: u8) -> bool {
        self.skip_space();
        let found = self.peek() == Some(byte);
        if found {
            self.pos += 1;
        }
        found
    }

    /// Stops as malformed unless only whitespace is left.
    pub(super) fn finish(&mut self) -> ParseResult<()> {
        self.skip_space();
        if self.pos == self.text.len() {
            Ok(())
        } else {
            Err(Stop::Malformed("data after the JSON text"))
        }
    }

    /// Reads a string value and decodes its escapes.
    pub(super) fn string(&mut self) -> ParseResult<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let run = self.pos;
            while self.peek().is_some_and(|b| b != b'"' && b != b'\\' && b >= 0x20) {
                self.pos += 1;
            }
            out.push_str(self.text.get(run..self.pos).ok_or(TRUNCATED)?);
            match self.next_byte()? {
                b'"' => return Ok(out),
                b'\\' => out.push(self.escape()?),
                _ => return Err(Stop::Malformed("control character in a JSON string")),
            }
        }
    }

    /// Reads a non-negative integer of at most 64 bits.
    pub(super) fn uint(&mut self) -> ParseResult<u64> {
        self.skip_space();
        let start = self.pos;
        let mut value = 0_u64;
        while let Some(digit) = self.peek().filter(u8::is_ascii_digit) {
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add(u64::from(digit - b'0')))
                .ok_or(Stop::Malformed("JSON integer does not fit 64 bits"))?;
            self.pos += 1;
        }
        if self.pos == start {
            return Err(Stop::Malformed("expected a JSON integer"));
        }
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(Stop::Malformed("JSON number is not an integer"));
        }
        Ok(value)
    }

    /// Skips one value of any type, checking its syntax.
    pub(super) fn skip_value(&mut self) -> ParseResult<()> {
        self.skip_nested(0)
    }

    /// Skips one value that sits `depth` containers deep.
    fn skip_nested(&mut self, depth: u32) -> ParseResult<()> {
        if depth > MAX_DEPTH {
            return Err(Stop::Malformed("JSON nesting is too deep"));
        }
        self.skip_space();
        match self.peek() {
            Some(b'"') => self.string().map(|_| ()),
            Some(b'{') => self.skip_object(depth),
            Some(b'[') => self.skip_array(depth),
            Some(b't') => self.literal("true"),
            Some(b'f') => self.literal("false"),
            Some(b'n') => self.literal("null"),
            Some(b'-' | b'0'..=b'9') => {
                self.skip_number();
                Ok(())
            }
            _ => Err(Stop::Malformed("expected a JSON value")),
        }
    }

    fn skip_object(&mut self, depth: u32) -> ParseResult<()> {
        self.expect(b'{')?;
        if self.eat(b'}') {
            return Ok(());
        }
        loop {
            self.string()?;
            self.expect(b':')?;
            self.skip_nested(depth + 1)?;
            if self.eat(b'}') {
                return Ok(());
            }
            self.expect(b',')?;
        }
    }

    fn skip_array(&mut self, depth: u32) -> ParseResult<()> {
        self.expect(b'[')?;
        if self.eat(b']') {
            return Ok(());
        }
        loop {
            self.skip_nested(depth + 1)?;
            if self.eat(b']') {
                return Ok(());
            }
            self.expect(b',')?;
        }
    }

    /// Skips a number. Its value is never used, so only the character set is checked.
    fn skip_number(&mut self) {
        let is_number_byte =
            |byte| matches!(byte, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E');
        while self.peek().is_some_and(is_number_byte) {
            self.pos += 1;
        }
    }

    fn literal(&mut self, word: &str) -> ParseResult<()> {
        if self.text.get(self.pos..).is_some_and(|rest| rest.starts_with(word)) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(Stop::Malformed("expected a JSON literal"))
        }
    }

    /// Decodes the escape after a backslash, joining a surrogate pair into one character.
    fn escape(&mut self) -> ParseResult<char> {
        let simple = match self.next_byte()? {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => return self.unicode_escape(),
            _ => return Err(Stop::Malformed("unknown JSON escape")),
        };
        Ok(simple)
    }

    fn unicode_escape(&mut self) -> ParseResult<char> {
        let first = self.hex4()?;
        let code = if (0xD800..0xDC00).contains(&first) {
            self.expect_raw(b"\\u")?;
            let second = self.hex4()?;
            if !(0xDC00..0xE000).contains(&second) {
                return Err(Stop::Malformed("unpaired JSON surrogate"));
            }
            0x1_0000 + ((first - 0xD800) << 10) + (second - 0xDC00)
        } else {
            first
        };
        char::from_u32(code).ok_or(Stop::Malformed("JSON escape is not a character"))
    }

    fn expect_raw(&mut self, bytes: &[u8]) -> ParseResult<()> {
        for &wanted in bytes {
            if self.next_byte()? != wanted {
                return Err(Stop::Malformed("unpaired JSON surrogate"));
            }
        }
        Ok(())
    }

    fn hex4(&mut self) -> ParseResult<u32> {
        let mut value = 0;
        for _ in 0..4 {
            let digit = char::from(self.next_byte()?)
                .to_digit(16)
                .ok_or(Stop::Malformed("bad hex digit in a JSON escape"))?;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn next_byte(&mut self) -> ParseResult<u8> {
        let byte = self.peek().ok_or(TRUNCATED)?;
        self.pos += 1;
        Ok(byte)
    }
}

/// The stop for a text that ends inside a value.
const TRUNCATED: Stop = Stop::Malformed("JSON text ends inside a value");

#[cfg(test)]
mod tests {
    use super::Json;
    use crate::profile::Stop;

    fn string_of(text: &str) -> Option<String> {
        Json::new(text).string().ok()
    }

    fn skip_then_finish(text: &str) -> bool {
        let mut json = Json::new(text);
        json.skip_value().is_ok() && json.finish().is_ok()
    }

    #[test]
    fn strings_decode_every_escape() {
        let all = string_of(r#""a\"\\\/\b\f\n\r\tz""#);
        assert_eq!(all.as_deref(), Some("a\"\\/\u{8}\u{c}\n\r\tz"));
        assert_eq!(string_of(r#""\u0041\u00e9""#).as_deref(), Some("Aé"));
        assert_eq!(string_of(r#""\ud83d\ude00""#).as_deref(), Some("\u{1F600}"));
        assert_eq!(string_of("\"слой.0\"").as_deref(), Some("слой.0"));
    }

    #[test]
    fn bad_strings_stop() {
        for text in [
            "\"open",
            "\"bad\\q\"",
            "\"\\u12\"",
            "\"\\ud83d\"",
            "\"\\ud83d\\u0041\"",
            "\"\\ude00\"",
            "\"a\nb\"",
            "\"\\",
        ] {
            assert!(string_of(text).is_none(), "{text:?}");
        }
    }

    #[test]
    fn integers_refuse_what_is_not_a_u64() {
        assert_eq!(Json::new(" 18446744073709551615").uint().ok(), Some(u64::MAX));
        for text in ["18446744073709551616", "-1", "1.5", "2e3", "", "x"] {
            assert!(Json::new(text).uint().is_err(), "{text:?}");
        }
    }

    #[test]
    fn any_value_can_be_skipped() {
        for text in [
            r#"{"a": [1, -2.5e3, true, false, null, "s", {}], "b": {"c": []}}"#,
            " [ ] ",
            "0",
            "\"x\"",
        ] {
            assert!(skip_then_finish(text), "{text:?}");
        }
    }

    #[test]
    fn broken_values_stop() {
        for text in ["{", "[1,", "[1 2]", "{\"a\"}", "{\"a\":}", "{1:2}", "tru", "", "}", "[,]"] {
            assert!(!skip_then_finish(text), "{text:?}");
        }
        assert!(!skip_then_finish("{} x"));
    }

    #[test]
    fn deep_nesting_stops_without_recursing_far() {
        let open = "[".repeat(1_000_000);
        let mut json = Json::new(&open);
        assert!(matches!(json.skip_value(), Err(Stop::Malformed("JSON nesting is too deep"))));
        let fits = format!("{}{}", "[".repeat(64), "]".repeat(64));
        assert!(skip_then_finish(&fits));
    }
}
