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
