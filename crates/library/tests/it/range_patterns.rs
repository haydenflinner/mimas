//! Range patterns in a `match` arm and three messages (combo item C18).

use crate::test_runner;

fn refused(src: &str) -> String {
    match vm::Vm::compile(src, library::std) {
        Ok(_) => panic!("compiled: {src}"),
        Err(e) => format!("{e:?}"),
    }
}

// e10: `lo..hi` excludes `hi`, `lo..=hi` includes it; negative and float
// bounds work, a range may sit inside a tuple pattern, and the arm's own
// guard still applies.
// r[verify mimas.match.range]
test_run!(
    a_range_arm_matches_its_bounds,
    "fn band(n: int) -> str { match n { 0..3 => \"low\", 3..10 => \"mid\", _ => \"hi\" } }",
    "band(2)" => r#""low""#,
    "band(3)" => r#""mid""#,
    "band(10)" => r#""hi""#,
    r#"match 10 { 3..=10 => "in", _ => "out" }"# => r#""in""#,
    r#"match -2 { -5..0 => "neg", _ => "other" }"# => r#""neg""#,
    r#"match 0.5 { 0.0..1.0 => "unit", _ => "out" }"# => r#""unit""#,
    r#"match (1, 7) { (0..2, y) if y > 5 => "a", _ => "b" }"# => r#""a""#,
    r#"match (1, 3) { (0..2, y) if y > 5 => "a", _ => "b" }"# => r#""b""#,
);

// r[verify mimas.match.range]
#[test]
fn a_range_pattern_outside_an_arm_or_in_an_or_is_refused() {
    assert!(refused("let 0..3 = 2\n").contains("only stands in a `match` arm"));
    let err = refused("let s = match 2 { 0..3 | 7..9 => \"x\", _ => \"y\" }\n");
    assert!(err.contains("one side of `|`"), "{err}");
}

// g11: the help says what to write.
// r[verify mimas.impl.bare-name]
#[test]
fn impl_with_type_params_says_to_drop_them() {
    let err = refused("struct Box<T> { v: T }\nimpl Box<T> { fn get(self) -> T { self.v } }\n");
    assert!(err.contains("write `impl Box`"), "{err}");
}

// d04: a field read on an option that a bare arm bound points at `x? =>`.
#[test]
fn a_field_on_a_bare_option_arm_shows_the_bind_help() {
    let err = refused(
        "fn f(ok: bool) -> (int, int)? { if ok { (1, 2) } else { null } }\n\
         let p = f(true)\nlet v = match p { null => 0, q => q.0 }\n",
    );
    assert!(err.contains("`x? =>`"), "{err}");
}

// x03 (C18c): a closure that counts with a captured `int` was a
// generator that never ended; L1 made the write an error.
// r[verify mimas.closure.capture-write]
#[test]
fn a_counting_closure_is_refused() {
    let err = refused(
        "fn steps() -> () -> px? {\n  let i = 0\n  return || { i += 1; if i > 3 { null } else { i * 1.0px } }\n}\n",
    );
    assert!(err.contains("captured a copy"), "{err}");
}
