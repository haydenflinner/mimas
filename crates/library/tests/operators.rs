#[macro_use]
mod test_runner;

// `a <op> b` desugars to `a.<name>(b)` on the left operand's type, else
// `b.<rname>(a)` on the right's — the methods are ordinary `impl` members.

const VEC: &str = "
struct V { x: float, y: float }
impl V {
    fn add(self, o: V) -> V { V { x = self.x + o.x, y = self.y + o.y } }
    fn sub(self, o: V) -> V { V { x = self.x - o.x, y = self.y - o.y } }
    fn mul(self, s: float) -> V { V { x = self.x * s, y = self.y * s } }
    fn rmul(self, s: float) -> V { V { x = self.x * s, y = self.y * s } }
    fn rsub(self, s: float) -> V { V { x = s - self.x, y = s - self.y } }
    fn neg(self) -> V { V { x = -self.x, y = -self.y } }
    fn lt(self, o: V) -> bool { self.x < o.x }
    fn band(self, o: V) -> V { self.add(o) }
    fn bor(self, o: V) -> V { self.sub(o) }
}
let a = V { x = 1.0, y = 2.0 };
let b = V { x = 3.0, y = 4.0 };
";

test_run!(
    binary_ops_dispatch_to_impl_methods,
    VEC,
    "(a + b).x" => "4",
    "(a + b).y" => "6",
    "(a - b).y" => "-2",
    "(a * 2.0).x" => "2",
    "(a & b).x" => "4",
    "(a | b).y" => "-2",
    "(-a).y" => "-2",
);

test_run!(
    r_named_methods_cover_left_owned_ops,
    VEC,
    // `s - v` finds `v.rsub(s)` — `self` is the right operand
    "(10.0 - a).x" => "9",
    "(10.0 - a).y" => "8",
    "(2.0 * b).y" => "8",
    "(b * 2.0).y" => "8",
);

test_run!(
    comparisons_dispatch_and_may_return_non_bool,
    VEC,
    "(a < b)" => "true",
    "(b < a)" => "false",
    // `lt` is a real method — its signature is the contract, bool isn't required
    "a.lt(b)" => "true",
);

test_run!(
    compound_assign_runs_the_method,
    VEC,
    "{ let c = a; c += b; c.x }" => "4",
    "{ let c = a; c -= b; c.x }" => "-2",
);

test_run!(
    equality_stays_structural,
    VEC,
    "a == V { x = 1.0, y = 2.0 }" => "true",
    "a != b" => "true",
);

// natives get the same surface: Vec2/Vec3 ship `mul`/`rmul`/`div`/`neg` in
// std::math (component scale, symmetric `s * v`, scalar `/`, unary `-`).
test_run!(
    vec2_native_ops,
    "use std::math::Vec2; let v = Vec2::new(3.0, 4.0);",
    "(v * 2.0).x" => "6",
    "(2.0 * v).y" => "8",
    "(v / 2.0).y" => "2",
    "(-v).x" => "-3",
    "(v + Vec2::new(1.0, 1.0)).x" => "4",
);

// not an operator impl: a method named `add` with extra required params is an
// ordinary method — `a + b` still reports the plain invalid-op error
test_fail!(
    wrong_arity_method_isnt_an_op_impl,
    "struct S { n: int }
     impl S { fn add(self, o: S, k: int) -> S { S { n = self.n + o.n + k } } }
     S { n = 1 } + S { n = 2 }",
);

test_fail!(
    operand_mismatch_errors,
    "struct V { x: float }
     impl V { fn add(self, o: V) -> V { V { x = self.x + o.x } } }
     V { x = 1.0 } + \"nope\"",
    // `s - v` where `rsub` takes a float — a str operand still fails the sig
    "struct V { x: float }
     impl V { fn rsub(self, s: float) -> V { V { x = s - self.x } } }
     \"x\" - V { x = 1.0 }",
    // no impl anywhere: `int - V` finds neither `int.sub` nor `V.rsub`
    "struct V { x: float }
     impl V { fn add(self, o: V) -> V { V { x = self.x + o.x } } }
     1 - V { x = 1.0 }",
);
