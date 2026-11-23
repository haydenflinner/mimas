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
    fn inv(self) -> V { V { x = self.y, y = self.x } }
    fn lt(self, o: V) -> bool { self.x < o.x }
    fn band(self, o: V) -> V { self.add(o) }
    fn bor(self, o: V) -> V { self.sub(o) }
    // `nand`/`nor` name methods directly — the words are operators yet
    // still ordinary identifiers, unlike `mod`/`xor`/`and`/`or`/`not`.
    fn nand(self, o: V) -> V { V { x = self.y, y = o.y } }
    fn nor(self, o: V) -> V { V { x = o.x, y = self.x } }
    fn rnor(self, s: float) -> V { V { x = -1.0, y = -1.0 } }
}
let a = V { x = 1.0, y = 2.0 };
let b = V { x = 3.0, y = 4.0 };
";

test_run!(
    // r[verify mimas.op.method]
    binary_ops_dispatch_to_impl_methods,
    VEC,
    "(a + b).x" => "4.0",
    "(a + b).y" => "6.0",
    "(a - b).y" => "-2.0",
    "(a * 2.0).x" => "2.0",
    "(a & b).x" => "4.0",
    "(a | b).y" => "-2.0",
    "(-a).y" => "-2.0",
    // `!a` calls `a.inv()` — and `not a` spells the same call
    "(!a).x" => "2.0",
    "(not a).x" => "2.0",
    // bool has no `inv`: `!` stays the builtin rule
    "!false" => "true",
);

test_run!(
    r_named_methods_cover_left_owned_ops,
    VEC,
    // `s - v` finds `v.rsub(s)` — `self` is the right operand
    "(10.0 - a).x" => "9.0",
    "(10.0 - a).y" => "8.0",
    "(2.0 * b).y" => "8.0",
    "(b * 2.0).y" => "8.0",
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
    "{ let c = a; c += b; c.x }" => "4.0",
    "{ let c = a; c -= b; c.x }" => "-2.0",
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
    "(v * 2.0).x" => "6.0",
    "(2.0 * v).y" => "8.0",
    "(v / 2.0).y" => "2.0",
    "(-v).x" => "-3.0",
    "(v + Vec2::new(1.0, 1.0)).x" => "4.0",
);

// `nand`/`nor` run as builtin gates on bools (eager — no short-circuit) and
// per bit-lane on ints; the words `nand`/`nor` and glyphs `⍲`/`⍱` spell the
// same ops, and an impl method of the same name wins dispatch on an ADT.
test_run!(
    nand_nor_builtin_gates,
    "",
    "true nand true" => "false",
    "true nand false" => "true",
    "false nor false" => "true",
    "false nor true" => "false",
    "(true ⍲ false) == false ⍱ false" => "true",
    // the existential gate: `a nand b` is `!(a && b)` — `nor` its dual, `!(a || b)`
    "(true nand false) == !(true && false)" => "true",
    "(false nor true) == !(false || true)" => "true",
    // bit-lane on ints: `12 nand 10` is `~(12 & 10)` = `~8`
    "12 nand 10" => "-9",
    "12 nor 10" => "-15",
);

test_run!(
    // `==`/`!=` answer numerically across int/float like `<`/`>` already did —
    // bound operands force the runtime `Val` compare (literal pairs fold)
    mixed_int_float_equality_is_numeric,
    "let i = 5; let f = 5.0;",
    "i == f" => "true",
    "f == i" => "true",
    "i != f" => "false",
    "i == 5.0" => "true",
    "5.0 == i" => "true",
    "i - 1 == f" => "false",
    "i != 4.5" => "true",
);

test_run!(
    nand_nor_dispatch_to_impl_methods,
    VEC,
    "(a nand b).x" => "2.0",
    "(a ⍲ b).y" => "4.0",
    // the word and the glyph spell the same call
    "(a nand b).x == (a ⍲ b).x" => "true",
    "(a nor b).x" => "3.0",
    // `2.0 nor v` finds `v.rnor(2.0)` — self owns the impl on the right
    "(2.0 nor b).x" => "-1.0",
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

// ---- the named operators ---------------------------------------------
//
// Julia's table, Mimas's rule: `∧ ⋅ × ∘ ⊠ ∩` bind like `*`, `∨ ∪ ⊖ ⊎`
// like `+`, and none has a meaning until a type gives it one — `a ∧ b` is
// `a.wedge(b)` (or `b.rwedge(a)`) and nothing else.

const GA: &str = "
struct V { x: float, y: float }
struct B { xy: float }
struct S { s: str }
impl V {
    fn dot(self, o: V) -> float { self.x * o.x + self.y * o.y }
    fn wedge(self, o: V) -> B { B { xy = self.x * o.y - self.y * o.x } }
    fn rcross(self, k: float) -> V { V { x = -k * self.y, y = k * self.x } }
    fn compose(self, o: V) -> V { V { x = self.x + o.x, y = self.y + o.y } }
}
impl S {
    fn union(self, o: S) -> S { S { s = f\"({self.s}∪{o.s})\" } }
    fn intersect(self, o: S) -> S { S { s = f\"({self.s}∩{o.s})\" } }
    fn vee(self, o: S) -> S { S { s = f\"({self.s}∨{o.s})\" } }
    fn wedge(self, o: S) -> S { S { s = f\"({self.s}∧{o.s})\" } }
}
let a = V { x = 1.0, y = 0.0 };
let b = V { x = 0.0, y = 1.0 };
let p = S { s = \"p\" };
let q = S { s = \"q\" };
let r = S { s = \"r\" };
";

test_run!(
    // r[verify mimas.op.glyphs]
    // r[verify mimas.op.named]
    named_operators_dispatch_to_their_methods,
    GA,
    "a ⋅ b" => "0.0",
    "a · a" => "1.0",
    "(a ∧ b).xy" => "1.0",
    "(b ∧ a).xy" => "-1.0",
    // `k × v` finds `v.rcross(k)` — the right operand owns the impl
    "(2.0 × a).y" => "2.0",
    "(a ∘ b).x" => "1.0",
    // `⋅` binds like `*`: `a ⋅ b + 1.0` is `(a ⋅ b) + 1.0`
    "a ⋅ b + 1.0" => "1.0",
    // `∩`/`∧` bind tighter than `∪`/`∨`, like `*` over `+`
    "(p ∪ q ∩ r).s" => "\"(p∪(q∩r))\"",
    "(p ∨ q ∧ r).s" => "\"(p∨(q∧r))\"",
    "(p ∪ q ∪ r).s" => "\"((p∪q)∪r)\"",
    // `∪` on lists is the `union` native `+` already dispatches to
    "[1, 2] ∪ [3]" => "[1, 2, 3]",
);

test_run!(
    // r[verify mimas.op.cmp-glyphs]
    comparison_and_membership_glyphs,
    "1 ≠ 2" => "true",
    "2 ≤ 2" => "true",
    "3 ≥ 4" => "false",
    "2 ∈ [1, 2]" => "true",
    "3 ∉ [1, 2]" => "true",
);

// r[verify mimas.op.named]
// a named operator has no built-in meaning: on numbers, strings or a type
// without the method it is an error that names the method
test_fail!(
    named_operator_without_a_method_is_an_error,
    "1 ∧ 2",
    "\"a\" ⋅ \"b\"",
    "1.0 × 2.0",
    "struct P { x: float }
     P { x = 1.0 } ∘ P { x = 2.0 }",
);

// ---- operand-typed overloads ----------------------------------------
//
// One name, several operands: `fn mul(self, o: V)` beside `fn mul(self,
// k: float)`. `v * w` and `v * 2.0` each pick the method whose operand
// fits, and so does the call `v.mul(..)`.

const OVERLOADS: &str = "
struct V { x: float, y: float }
impl V {
    fn mul(self, o: V) -> float { self.x * o.x + self.y * o.y }
    fn mul(self, k: float) -> V { V { x = self.x * k, y = self.y * k } }
    fn rmul(self, k: float) -> V { V { x = self.x * k, y = self.y * k } }
    fn add(self, o: V) -> V { V { x = self.x + o.x, y = self.y + o.y } }
    fn add(self, k: float) -> V { V { x = self.x + k, y = self.y + k } }
}
let a = V { x = 1.0, y = 2.0 };
let b = V { x = 3.0, y = 4.0 };
";

test_run!(
    // r[verify mimas.op.overload]
    operators_pick_the_overload_the_operand_fits,
    OVERLOADS,
    "a * b" => "11.0",
    "(a * 2.0).y" => "4.0",
    "(2.0 * a).y" => "4.0",
    "(a + b).x" => "4.0",
    "(a + 1.0).x" => "2.0",
    "a.mul(b)" => "11.0",
    "a.mul(2.0).x" => "2.0",
    "a.add(1.0).y" => "3.0",
    // the overloads compose with the named operators and with `+=`
    "{ let v = a; v *= 2.0; v.x }" => "2.0",
);

// r[verify mimas.op.overload]
// the same operand type twice is the duplicate it always was, and an
// unannotated operand can't tell overloads apart
test_fail!(
    same_operand_twice_is_a_duplicate,
    "struct V { x: float }
     impl V {
         fn mul(self, o: V) -> float { self.x * o.x }
         fn mul(self, o: V) -> V { V { x = self.x * o.x } }
     }",
    "struct V { x: float }
     impl V {
         fn mul(self, o: V) -> float { self.x * o.x }
         fn mul(self, k) -> V { V { x = self.x * k } }
     }",
    "struct V { x: float }
     impl V {
         fn scale(self, k: float, n: int) -> V { V { x = self.x * k } }
         fn scale(self, k: int, n: int) -> V { V { x = self.x } }
     }",
);
