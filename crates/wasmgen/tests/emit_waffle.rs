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
    let w = emit_waffle_ir(&ir, &_program.strs, &Opts::default(), None, None, 0, 0)
        .expect("emit_waffle_ir");
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
    let w = emit_waffle_ir(&ir, &_program.strs, &Opts::default(), None, None, 0, 0)
        .expect("emit_waffle_ir");
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
    let w = emit_waffle_ir(&ir, &_program.strs, &Opts::default(), None, None, 0, 0)
        .expect("emit_waffle_ir");
    assert!(
        w.skipped.is_empty(),
        "entry-ops bodies must all emit; skipped: {:?}",
        w.skipped.iter().map(|s| &s.reason).collect::<Vec<_>>()
    );
    wasmparser::validate(&w.bytes).expect("invalid wasm");
}

/// Tuples lower to arrays (`NewArray` + pushes) but `t.0` / `t.1` lower to
/// slotted `GetField`, which read the struct-instance layout: `.0` came back
/// as the array's len|cap word and `.1` as element 0 (found baking the
/// hear-the-zeros page, whose prime-power weights all read wrong). Run the
/// module under wasmtime and read the echoed values back.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_tuple_reads_its_own_fields() {
    use std::sync::{Arc, Mutex};
    use wasmtime::{Engine, Global, Linker, Module, Store, Val, ValType};
    let src = "fn pick(t: (float, float), i: int) -> float { if i == 0 { t.0 } else { t.1 } }\n\
               let p = (2.5, 7.25)\n\
               pick(p, 0)\n\
               pick(p, 1)\n\
               p.1\n";
    let lib = Vm::new().install_library(mimas::library::std);
    let echo_id = lib
        .natives()
        .find_map(|(id, e)| match e {
            api::ApiEntry::Function(f) if f.module.is_empty() && f.name == "echo" => Some(id.index() as u32),
            _ => None,
        })
        .expect("echo native");
    let (program, ir, _s) = Vm::compile_parts_ir(&[("<t>", src)], mimas::library::std).expect("compile");
    let w = emit_waffle_ir(&ir, &program.strs, &Opts::default(), None, None, 0, 0).expect("emit");
    assert!(w.skipped.is_empty(), "{:?}", w.skipped);
    let engine = Engine::default();
    let module = Module::new(&engine, &w.bytes[..]).expect("wasm");
    let mut linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let seen: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    for imp in module.imports() {
        match imp.ty() {
            wasmtime::ExternType::Func(ft) => {
                let is_echo = imp.name().starts_with(&format!("n{echo_id}_"));
                let cap = seen.clone();
                let rets: Vec<ValType> = ft.results().collect();
                linker
                    .func_new(imp.module(), imp.name(), ft, move |_c, p, r| {
                        if is_echo {
                            let v = match &p[0] {
                                Val::F64(b) => f64::from_bits(*b),
                                Val::I64(i) => f64::from_bits(*i as u64),
                                _ => f64::NAN,
                            };
                            cap.lock().unwrap().push(v);
                        }
                        for (i, out) in r.iter_mut().enumerate() {
                            *out = match rets[i] {
                                ValType::I32 => Val::I32(0),
                                ValType::I64 => Val::I64(0),
                                ValType::F64 => Val::F64(0),
                                _ => Val::I32(0),
                            };
                        }
                        Ok(())
                    })
                    .unwrap();
            }
            wasmtime::ExternType::Global(gt) => {
                let g = Global::new(&mut store, gt, Val::I32(0)).unwrap();
                linker.define(&mut store, imp.module(), imp.name(), g).unwrap();
            }
            _ => panic!("unexpected import {}", imp.name()),
        }
    }
    let inst = linker.instantiate(&mut store, &module).expect("instantiate");
    let b0 = inst.get_func(&mut store, "b0").expect("b0");
    let mut res: Vec<Val> = b0.ty(&store).results().map(|_| Val::I32(0)).collect();
    b0.call(&mut store, &[], &mut res).expect("run");
    let got = seen.lock().unwrap().clone();
    // the echoes end with pick(p, 0), pick(p, 1), p.1
    assert_eq!(&got[got.len() - 3..], &[2.5, 7.25, 7.25], "{got:?}");
}
