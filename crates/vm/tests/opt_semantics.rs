// Results the IR optimizer must not change: aliasing of struct locals under
// inlining and scalar replacement (compile/src/tests/opt_tests.rs pins what
// the passes do; these pin that it is invisible).
#[macro_use]
mod vm_test_utils;

use vm::Captured::*;

const P: &str = "struct P { x: float }
impl P { fn inc(self) { self.x = self.x + 1.0; } }
fn bump(p: P) { p.x = p.x + 1.0; }
fn mk(x: float) -> P { P { x = x } }";

test_vm!(
    mutation_through_an_inlined_param_is_seen,
    P,
    "{ let p = P { x = 1.0 }; bump(p); p.x }" => Float(2.0),
    "{ let p = P { x = 0.0 }; p.inc(); p.inc(); p.x }" => Float(2.0),
);

test_vm!(
    a_copy_keeps_the_first_object_when_the_source_is_rebound,
    P,
    "{ let a = P { x = 1.0 }; let b = a; a = P { x = 5.0 }; b.x }" => Float(1.0),
    "{ let a = P { x = 1.0 }; let b = a; a = P { x = 5.0 }; a.x + b.x }" => Float(6.0),
);

test_vm!(
    a_struct_in_an_array_keeps_identity,
    P,
    "{ let v = P { x = 3.0 }; let xs = [v]; v.x = 4.0; xs[0].x }" => Float(4.0),
);

test_vm!(
    returned_structs_compose,
    P,
    "{ let p = mk(2.0); let q = mk(3.0); p.x * q.x }" => Float(6.0),
    "mk(mk(4.0).x).x" => Float(4.0),
);

test_vm!(
    structural_equality_survives,
    P,
    "{ let a = P { x = 1.0 }; let b = P { x = 1.0 }; a == b }" => Bool(true),
    "{ let a = P { x = 1.0 }; let b = P { x = 2.0 }; a == b }" => Bool(false),
);

test_vm!(
    a_struct_per_loop_iteration,
    P,
    // (the VM tests run without std, so no `to_float`: count in floats)
    "{ let t = 0.0; let f = 0.0; for i in 4 { let p = P { x = f }; f = f + 1.0; t = t + p.x; } t }" => Float(6.0),
    "{ let t = 0.0; let f = 0.0; for i in 4 { let p = mk(f); p.inc(); f = f + 1.0; t = t + p.x; } t }" => Float(10.0),
);

test_vm!(
    a_struct_chosen_by_a_branch,
    P,
    "{ let c = true; let p = if c { P { x = 1.0 } } else { P { x = 2.0 } }; p.x }" => Float(1.0),
);

test_vm!(
    nested_struct_fields,
    "struct V { x: float }\nstruct W { v: V, k: float }\nfn f() -> float { let w = W { v = V { x = 2.0 }, k = 3.0 }; w.v.x = w.v.x + w.k; w.v.x }",
    "f()" => Float(5.0),
);

// ---- locals to SSA: loop-carried values, swaps, structs chosen by control flow

test_vm!(
    loop_carried_scalars_and_swaps,
    "",
    "{ let a = 1; let b = 2; for i in 3 { let t = a; a = b; b = t; } a * 10 + b }" => Int(21),
    "{ let a = 1; let b = 2; let c = 3; for i in 4 { let t = a; a = b; b = c; c = t; } a * 100 + b * 10 + c }" => Int(231),
    "{ let s = 0; let k = 1; while k < 6 { s = s + k; k = k + 1; } s }" => Int(15),
    "{ let x = 0; let c = true; if c { x = 5; } else { x = 7; } x * 2 }" => Int(10),
);

test_vm!(
    loop_carried_structs,
    P,
    "{ let p = P { x = 1.0 }; for i in 4 { p = P { x = p.x * 2.0 }; } p.x }" => Float(16.0),
    "{ let p = P { x = 1.0 }; for i in 3 { p.x = p.x + 1.0; } p.x }" => Float(4.0),
    // three swaps: p ends on the 9
    "{ let p = P { x = 1.0 }; let q = P { x = 9.0 }; for i in 3 { let t = p; p = q; q = t; } p.x * 10.0 + q.x }" => Float(91.0),
    // +1, ×2, +1, ×2, +1 from 1: 2, 4, 5, 10, 11
    "{ let p = mk(1.0); for i in 5 { let c = i mod 2 == 0; p = if c { mk(p.x + 1.0) } else { P { x = p.x * 2.0 } }; } p.x }" => Float(11.0),
);

test_vm!(
    a_struct_rebuilt_in_a_loop_then_mutated,
    P,
    "{ let p = P { x = 0.0 }; for i in 3 { p = P { x = p.x + 1.0 }; } p.inc(); p.x }" => Float(4.0),
);

// ---- value types: an adt nothing mutates has no identity to observe

const V: &str = "struct V { x: float, y: float }
fn v(x: float, y: float) -> V { V { x = x, y = y } }
fn keep(p: V) -> float { p.x + p.y }
fn pick(c: bool, a: V, b: V) -> V { if c { a } else { b } }";

test_vm!(
    value_structs_escape_as_copies,
    V,
    "{ let d = v(1.0, 2.0); let xs = [d]; let ys = [d]; d.x + xs[0].y + ys[0].x }" => Float(4.0),
    "{ let d = v(1.0, 2.0); keep(d) + d.x }" => Float(4.0),
    "{ let p = v(1.0, 2.0); for i in 3 { p = v(p.x + 1.0, p.y); } keep(p) }" => Float(6.0),
    "{ let a = v(1.0, 0.0); let b = v(2.0, 0.0); let c = pick(false, a, b); c.x + a.x }" => Float(3.0),
    "{ let a = v(1.0, 0.0); let b = v(1.0, 0.0); a == b }" => Bool(true),
);

test_vm!(
    a_mutated_adt_stays_a_reference,
    P,
    "{ let p = P { x = 1.0 }; let q = p; q.x = 5.0; p.x }" => Float(5.0),
    "{ let p = P { x = 1.0 }; let xs = [p]; xs[0].x = 7.0; p.x }" => Float(7.0),
);

// ---- self tail calls: recursion as deep as you like, params rebound in parallel

test_vm!(
    self_tail_calls_do_not_grow_the_stack,
    "fn count(n: int, acc: int) -> int { if n == 0 { acc } else { count(n - 1, acc + n) } }
     fn swap(a: int, b: int, n: int) -> int { if n == 0 { a * 10 + b } else { swap(b, a, n - 1) } }
     fn rot(a: int, b: int, c: int, n: int) -> int { if n == 0 { a * 100 + b * 10 + c } else { rot(b, c, a, n - 1) } }
     fn thru(a: int, n: int) -> int { if n == 0 { a } else { thru(a, n - 1) } }",
    "count(100000, 0)" => Int(5000050000),
    "swap(1, 2, 3)" => Int(21),
    "rot(1, 2, 3, 4)" => Int(231),
    "thru(7, 50000)" => Int(7),
);

test_vm!(
    a_read_of_a_slot_written_later_is_a_snapshot,
    V,
    "{ let n = 1.0; let d = v(n, 0.0); n = 5.0; d.x }" => Float(1.0),
    "{ let n = 1.0; let d = v(n, 0.0); let e = d; n = 5.0; e.x + n }" => Float(6.0),
);

// ---- closures across a self tail call: captured by value, before the rebinding

test_vm!(
    closures_cross_a_tail_call,
    "fn acc(f: (int) -> int, n: int) -> int { if n == 0 { f(1) } else { acc(|x| f(x) * 2, n - 1) } }
     fn go(k: int, f: (int) -> int, n: int) -> int { if n == 0 { f(k) } else { go(k + 1, f, n - 1) } }
     fn h(a: int, f: (int) -> int, n: int) -> int { if n == 0 { f(0) + a } else { h(a + 1, |x| a + x, n - 1) } }",
    // a fresh closure per step, built from the current `f`
    "acc(|x| x, 5)" => Int(32),
    // a closure passed through a deep tail recursion untouched
    "go(0, |x| x + 100, 20000)" => Int(20100),
    // the closure captures `a` as it was, not as the call rebinds it
    "h(0, |x| x, 2)" => Int(3),
);
