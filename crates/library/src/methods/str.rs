use heck::ToSnakeCase;
use macros::native;
use vm::{RtErr, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    let nid = api.add_method(len);
    api.mark_intrinsic(nid, api::Intrinsic::Len);
    let nid = api.add_method(contains);
    api.mark_intrinsic(nid, api::Intrinsic::In);
    api.add_method(starts_with);
    api.add_method(ends_with);
    api.add_method(to_upper);
    api.add_method(to_lower);
    api.add_method(trim);
    api.add_method(to_snake);
    api.add_method(split);
    api.add_method(find);
    api.add_method(find_all);
    api.add_method(lines);
    api.add_method(capitalize);
    api.add_method(to_int);
    api.add_method(ord);
    api.add_method(is_empty);
    api.add_method(replace);
    api.add_method(repeat);
}

#[native]
/// Returns the number of characters. This counts Unicode characters, not bytes: `"héllo".len()` is
/// `5`.
///
/// ```mimas
/// let n = "hello".len(); // 5
/// ```
fn len(s: &str) -> usize {
    s.chars().count()
}

#[native]
/// Returns whether `needle` appears anywhere in the string. This is the same check as the
/// [`in` operator](../reference/collections/in-expressions.md).
///
/// ```mimas
/// let a = "hello".contains("ell"); // true
/// let b = "ell" in "hello";        // true
/// ```
fn contains(s: &str, needle: &str) -> bool {
    s.contains(needle)
}

#[native]
/// Returns whether the string begins with `prefix`.
///
/// ```mimas
/// let a = "mimas".starts_with("mi"); // true
/// ```
fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

#[native]
/// Returns whether the string ends with `suffix`.
///
/// ```mimas
/// let a = "readme.md".ends_with(".md"); // true
/// ```
fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

#[native]
/// Returns the string with every letter in uppercase. This follows Unicode's rules, which can turn
/// one character into several.
///
/// ```mimas
/// let a = "Hello".to_upper();  // "HELLO"
/// let b = "straße".to_upper(); // "STRASSE"
/// ```
fn to_upper(s: &str) -> String {
    s.to_uppercase()
}

#[native]
/// Returns the string with every letter in lowercase, following Unicode's rules.
///
/// ```mimas
/// let a = "Hello".to_lower(); // "hello"
/// ```
fn to_lower(s: &str) -> String {
    s.to_lowercase()
}

#[native]
/// Returns the string without the whitespace at its start and end. Whitespace inside the string
/// stays.
///
/// ```mimas
/// let a = "  two words \n".trim(); // "two words"
/// ```
fn trim(s: &str) -> String {
    s.trim().to_string()
}

#[native]
/// Returns the string in snake case: lowercase words joined by underscores. Spaces, punctuation,
/// and changes from lowercase to uppercase all mark where one word ends and the next begins.
///
/// ```mimas
/// let a = "PlayerHealth".to_snake(); // "player_health"
/// let b = "max hp".to_snake();       // "max_hp"
/// let c = "HTTPServer".to_snake();   // "http_server"
/// ```
fn to_snake(s: &str) -> String {
    s.to_snake_case()
}

#[native]
/// Returns the pieces of the string between each occurrence of `separator`. Two separators in a
/// row leave an empty string between them.
///
/// ```mimas
/// let a = "a,b,c".split(",");     // ["a", "b", "c"]
/// let b = "a,,b".split(",");      // ["a", "", "b"]
/// let c = "no commas".split(","); // ["no commas"]
/// ```
fn split(s: &str, delim: &str) -> Vec<String> {
    s.split(&delim).map(|v| v.to_string()).collect()
}

#[native]
/// Returns the string's lines, without their line endings. A line can end in `\n` or `\r\n`, and
/// a line ending at the very end of the string doesn't add an empty line.
///
/// ```mimas
/// let a = "one\ntwo\r\nthree\n".lines(); // ["one", "two", "three"]
/// ```
fn lines(s: &str) -> Vec<String> {
    s.lines().map(From::from).collect()
}

// An invalid pattern is an expected failure (raise with the regex error);
// no match is an honest absence (null). So `[str]?!`: unwrap to `T?`.
#[native]
/// Searches the string for the first match of the regular expression `pattern`. Returns the whole
/// match followed by the text of each capture group, or `null` if nothing matched or `pattern`
/// isn't a valid regular expression.
///
/// A capture group that didn't take part in the match is left out of the array, which moves the
/// groups after it down a position.
///
/// The pattern syntax is that of the Rust [`regex`](https://docs.rs/regex/latest/regex/#syntax)
/// crate. Inside a mimas string, each backslash in the pattern is written twice (`"\\d+"` matches
/// a run of digits).
///
/// ```mimas
/// let date = "2024-06-01".find("(\\d+)-(\\d+)-(\\d+)");
/// // date is ["2024-06-01", "2024", "06", "01"]
/// let none = "abc".find("\\d"); // null
/// ```
fn find(s: &str, pattern: &str) -> Raisable<Option<Vec<String>>> {
    match regex::Regex::new(pattern) {
        Err(e) => Raisable::Raised(e.to_string()),
        Ok(re) => Raisable::Ok(re.captures(s).map(|caps| {
            caps.iter()
                .flatten()
                .map(|m| m.as_str().to_string())
                .collect()
        })),
    }
}

#[native]
/// Returns every match of the regular expression `pattern`, each in the form [`find`](#find)
/// returns. Matches don't overlap. The array is empty when nothing matched, and the result is
/// `null` only when `pattern` isn't a valid regular expression.
///
/// ```mimas
/// let pairs = "x=1, y=22".find_all("(\\w)=(\\d+)");
/// // pairs is [["x=1", "x", "1"], ["y=22", "y", "22"]]
/// ```
fn find_all(s: &str, pattern: &str) -> Raisable<Option<Vec<Vec<String>>>> {
    match regex::Regex::new(pattern) {
        Err(e) => Raisable::Raised(e.to_string()),
        Ok(re) => Raisable::Ok(Some(
            re.captures_iter(s)
                .map(|caps| {
                    caps.iter()
                        .flatten()
                        .map(|m| m.as_str().to_string())
                        .collect::<Vec<String>>()
                })
                .collect(),
        )),
    }
}

/// Literal validator shared by the regex methods: a literal pattern is checked at solve
/// time -- one that compiles proves the raise branch unreachable (`T?!` narrows to `T?`),
/// and one that doesn't is a compile error instead of a raised value.
fn valid_regex(args: &[shared::Literal]) -> Result<(), String> {
    match args.first() {
        Some(shared::Literal::Str(p)) => {
            regex::Regex::new(p).map(|_| ()).map_err(|e| e.to_string())
        }
        _ => Ok(()),
    }
}

vm::inventory::submit! {
    vm::api::NativeValidator {
        path: concat!(module_path!(), "::find"),
        validate: valid_regex,
    }
}

vm::inventory::submit! {
    vm::api::NativeValidator {
        path: concat!(module_path!(), "::find_all"),
        validate: valid_regex,
    }
}

#[native]
/// Returns the string with its first character in uppercase. The rest of the string is
/// unchanged.
///
/// ```mimas
/// let a = "élan".capitalize();  // "Élan"
/// let b = "hELLO".capitalize(); // "HELLO"
/// ```
fn capitalize(s: &str) -> String {
    s.chars()
        .next()
        .map(|c| c.to_uppercase().collect::<String>() + &s[c.len_utf8()..])
        .unwrap_or_default()
}

#[native]
/// Parses the string as a base-10 `int`, or returns `null` if it isn't one. Whitespace at either
/// end is ignored, and a leading `+` or `-` is allowed.
///
/// ```mimas
/// let a = " -12 ".to_int(); // -12
/// let b = "12px".to_int();  // null
/// let c = "4.2".to_int();   // null
/// ```
fn to_int(s: &str) -> Option<i64> {
    s.trim().parse::<i64>().ok()
}

#[native]
/// Returns the Unicode code point of the first character, or `null` if the string is empty.
///
/// ```mimas
/// let a = "A".ord(); // 65
/// let b = "".ord();  // null
/// ```
fn ord(s: &str) -> Option<i64> {
    s.chars().next().map(|c| c as i64)
}

#[native]
/// Returns whether the string has no characters. A string of spaces isn't empty.
///
/// ```mimas
/// let a = "".is_empty();  // true
/// let b = " ".is_empty(); // false
/// ```
fn is_empty(s: &str) -> bool {
    s.is_empty()
}

#[native]
/// Returns the string with every occurrence of `needle` replaced by `replacement`.
///
/// ```mimas
/// let a = "a-b-c".replace("-", "+"); // "a+b+c"
/// ```
fn replace(s: &str, needle: &str, new: &str) -> String {
    s.replace(needle, new)
}

#[native]
/// Returns the string repeated `count` times. A `count` of `0` gives an empty string.
///
/// A negative `count` is a runtime error.
///
/// ```mimas
/// let a = "ab".repeat(3);   // "ababab"
/// let line = "-".repeat(20);
/// ```
fn repeat(s: &str, count: i64) -> Result<String, RtErr> {
    let count = usize::try_from(count)
        .map_err(|_| RtErr::InvalidArgument("repeat count cannot be negative".into()))?;
    if s.len()
        .checked_mul(count)
        .is_none_or(|total| total > isize::MAX as usize)
    {
        return Err(RtErr::InvalidArgument(
            "repeat result would be too large".into(),
        ));
    }
    Ok(s.repeat(count))
}
