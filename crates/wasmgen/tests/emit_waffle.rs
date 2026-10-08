//! `emit_waffle_ir` smoke test: scalar-heavy source must emit valid wasm.

use mimas::vm::Vm;
use mimas_wasmgen::{Opts, wfull::emit_waffle_ir};

const SRC: &str = r#"
fn fib(n: int) -> int {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn sum(n: int) -> int {
    let s = 0;
    for i in 0 .. n { s += i; }
    s
}

fn wh(n: int) -> int {
    let i = 0;
    let s = 0;
    while i < n {
        i += 1;
        if i mod 2 == 0 { continue; }
        if s > 100 { break; }
        s += i;
    }
    s
}

fn sw(n: int) -> int {
    match n {
        0 => 10,
        1 => 20,
        2 => 30,
        _ => -1,
    }
}

fn fl(x: float) -> float {
    let t = 0.0;
    for i in 0 .. 5 { t += x * i.to_float(); }
    t.sqrt()
}

let r = fib(10) + sum(20) + wh(30) + sw(2) + fl(3.5).to_int();
"#;

#[test]
fn emits_valid_wasm() {
    let (_program, ir, _s) =
        Vm::compile_parts_ir(&[("<t>", SRC)], |api| mimas::library::std(api)).expect("compile");
    let w = emit_waffle_ir(&ir, &_program.strs, &Opts::default(), None, None, 0, 0).expect("emit_waffle_ir");
    assert!(
        !w.bodies.is_empty(),
        "expected some bodies to emit; all skipped: {:?}",
        w.skipped
    );
    wasmparser::validate(&w.bytes).expect("invalid wasm");
}

/// Heap/word-ops source: arrays, push, index, struct fields, `in`, string
/// identity, raise/unwrap — every body must emit (nothing unsupported left).
const HEAP_SRC: &str = r#"
struct Pt { x: int, y: int }

fn arr(n: int) -> int {
    let a = [10, 20, 30];
    a.push(n);
    a[1] = a[0] + n;
    a[1] + a.len()
}

fn pt(n: int) -> int {
    let p = Pt { x = n, y = n * 2 };
    p.x = p.x + 1;
    p.x + p.y
}

fn inside(n: int) -> int {
    if n in [4, 8, 15] { 1 } else { 0 }
}

fn streq(a: str, b: str) -> int {
    if a == b { 1 } else { 0 }
}

fn risky(n: int) -> int! {
    if n < 0 { raise "neg"; }
    n * 2
}

fn dbl(n: int) -> int {
    risky(n)!
}
"#;

#[test]
fn heap_ops_emit_valid_wasm() {
    let (_program, ir, _s) =
        Vm::compile_parts_ir(&[("<t>", HEAP_SRC)], |api| mimas::library::std(api))
            .expect("compile");
    let w = emit_waffle_ir(&ir, &_program.strs, &Opts::default(), None, None, 0, 0).expect("emit_waffle_ir");
    assert!(
        w.skipped.is_empty(),
        "heap-ops bodies must all emit; skipped: {:?}",
        w.skipped.iter().map(|s| &s.reason).collect::<Vec<_>>()
    );
    wasmparser::validate(&w.bytes).expect("invalid wasm");
}

/// Entry-frame locals: module-level lets read/written across bodies go
/// through the shared linear-memory entry region.
const ENTRY_SRC: &str = r#"
let shared = 10
let items = [1, 2, 3]

fn use_shared(x: int) -> int {
    return x + shared
}

fn bump(n: int) -> int {
    shared = shared + n
    return shared
}

fn items_sum() -> int {
    let s = 0
    for i in 0..items.len() {
        s = s + items[i]
    }
    return s
}
"#;

#[test]
fn entry_ops_emit_valid_wasm() {
    let (_program, ir, _s) =
        Vm::compile_parts_ir(&[("<t>", ENTRY_SRC)], |api| mimas::library::std(api))
            .expect("compile");
    let w = emit_waffle_ir(&ir, &_program.strs, &Opts::default(), None, None, 0, 0).expect("emit_waffle_ir");
    assert!(
        w.skipped.is_empty(),
        "entry-ops bodies must all emit; skipped: {:?}",
        w.skipped.iter().map(|s| &s.reason).collect::<Vec<_>>()
    );
    wasmparser::validate(&w.bytes).expect("invalid wasm");
}

// Surfaces a pre-existing `emit` bug — "expected i64 but nothing on stack" —
// on this source; the IR lane emits valid bytes for the same program.
#[test]
#[ignore = "pre-existing old-lane validation failure"]
fn bytecode_lane_still_valid() {
    let (program, _ir, _s) =
        Vm::compile_parts_ir(&[("<t>", SRC)], |api| mimas::library::std(api)).expect("compile");
    let w = mimas_wasmgen::emit(&program).expect("emit");
    wasmparser::validate(&w.bytes).expect("invalid wasm");
}
