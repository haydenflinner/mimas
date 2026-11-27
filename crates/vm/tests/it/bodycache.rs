//! Compiled bodies kept across compiles (`compile::bodycache`): a second
//! compile of the same program takes every body from the cache, an edit
//! recompiles only what it touched, and what comes out always runs as a
//! fresh compile would. Stats are per thread, so these hold with the other
//! tests compiling in parallel.

use vm::{Captured, Vm};

fn value(source: &str) -> Captured {
    let mut vm = Vm::execute(source, |_| {}).expect("compiled and ran");
    vm.resolve_name("TEST_VALUE").expect("TEST_VALUE bound")
}

const PROGRAM: &str = "
struct P { x: int, y: int }
fn sum(p: P) -> int { p.x + p.y }
fn scale(p: P, k: int) -> P { P { x: p.x * k, y: p.y * k } }
fn tri(n: int) -> int { if n == 0 { 0 } else { n + tri(n - 1) } }
fn pick(n: int) -> int { if n > 5 { sum(P { x: n, y: 1 }) } else { tri(n) } }
let TEST_VALUE = sum(scale(P { x: 1, y: 2 }, 3)) + tri(10) + pick(7);
";

#[test]
fn a_recompile_takes_every_body_from_the_cache() {
    // the store is process-wide, and the other tests here compile PROGRAM
    // in parallel: a `tri` of its own keeps this test's first compile cold
    // whatever ran before it
    let program = PROGRAM.replace("if n == 0 { 0 }", "if n <= 0 { 0 }");
    assert_eq!(value(&program), Captured::Int(9 + 55 + 8));
    let (_, cold_misses) = compile::body_cache_stats();
    assert!(cold_misses > 0 || std::env::var("MIMAS_BODYCACHE").is_ok_and(|v| v == "0"));
    assert_eq!(value(&program), Captured::Int(9 + 55 + 8));
    let (hits, misses) = compile::body_cache_stats();
    if std::env::var("MIMAS_BODYCACHE").is_ok_and(|v| v == "0") {
        return;
    }
    assert_eq!(misses, 0, "an unchanged program missed");
    assert!(hits >= 4, "only {hits} hits");
}

#[test]
fn an_edit_recompiles_what_it_touched_and_runs_the_new_code() {
    assert_eq!(value(PROGRAM), Captured::Int(9 + 55 + 8));
    // `tri`'s base case moves: tri(10) and nothing else changes
    let edited = PROGRAM.replace("if n == 0 { 0 }", "if n == 0 { 100 }");
    assert_eq!(value(&edited), Captured::Int(9 + 155 + 8));
    let (hits, misses) = compile::body_cache_stats();
    if std::env::var("MIMAS_BODYCACHE").is_ok_and(|v| v == "0") {
        return;
    }
    assert!(misses >= 1, "the edited body hit");
    assert!(hits >= 2, "the untouched bodies missed ({hits} hits, {misses} misses)");
    // and back: the first form's bodies are still there
    assert_eq!(value(PROGRAM), Captured::Int(9 + 55 + 8));
    assert_eq!(compile::body_cache_stats().1, 0);
}

#[test]
fn a_changed_struct_recompiles_the_bodies_that_read_it() {
    // `get` is the same text in both programs; only the type of `P.x` moves
    let a = "
struct P { x: int }
fn get(p: P) -> P { let q = p; q }
let TEST_VALUE = get(P { x: 3 }).x;
";
    let b = "
struct P { x: str }
fn get(p: P) -> P { let q = p; q }
let TEST_VALUE = get(P { x: \"three\" }).x;
";
    assert_eq!(value(a), Captured::Int(3));
    assert_eq!(value(b), Captured::Str("three".to_string()));
    assert_eq!(value(a), Captured::Int(3));
}
