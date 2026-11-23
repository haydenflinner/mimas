// Operators through a pact bound, and pact functions overloaded by operand.
//
// `x * y` with `x: T` under `<T: Num>` (or `x: Num` outright) is the pact's
// `mul`, dispatched at runtime over the bound's implementers; `x * 3.0`
// picks the `mul` whose operand is a float. The textbook payoff: one `f`
// written over `Num` runs on a plain wrapper and on a dual number, and the
// dual's `d` is f's exact derivative.

const NUM: &str = "
pact Num {
    fn add(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn mul(self, k: float) -> Self;
    fn rmul(self, k: float) -> Self;
    fn sin(self) -> Self;
    fn val(self) -> float;
}
struct R { v: float }
impl Num for R {
    fn add(self, o: R) -> R { R { v = self.v + o.v } }
    fn mul(self, o: R) -> R { R { v = self.v * o.v } }
    fn mul(self, k: float) -> R { R { v = self.v * k } }
    fn rmul(self, k: float) -> R { R { v = self.v * k } }
    fn sin(self) -> R { R { v = self.v.sin() } }
    fn val(self) -> float { self.v }
}
struct Dual { v: float, d: float }
impl Num for Dual {
    fn add(self, o: Dual) -> Dual { Dual { v = self.v + o.v, d = self.d + o.d } }
    fn mul(self, o: Dual) -> Dual { Dual { v = self.v * o.v, d = self.v * o.d + self.d * o.v } }
    fn mul(self, k: float) -> Dual { Dual { v = self.v * k, d = self.d * k } }
    fn rmul(self, k: float) -> Dual { Dual { v = self.v * k, d = self.d * k } }
    fn sin(self) -> Dual { Dual { v = self.v.sin(), d = self.v.cos() * self.d } }
    fn val(self) -> float { self.v }
}
fn seed(x: float) -> Dual { Dual { v = x, d = 1.0 } }
fn f<T: Num>(x: T) -> T { x * x * 3.0 + x.sin() }
fn g<T: Num>(x: T) -> T { 2.0 * x }
fn h<T: Num>(x: T) -> T { x.mul(3.0) + x.mul(x) }
fn dbl(x: Num) -> Num { x + x }
let two = R { v = 2.0 };
";

test_run!(
    // r[verify mimas.pact.binary]
    binary_operators_dispatch_through_a_bound,
    NUM,
    // `x + x` on a pact-typed value, and on a bounded param
    "dbl(two).val()" => "4.0",
    "dbl(seed(1.5)).val()" => "3.0",
    // `2.0 * x` finds `rmul` — the right operand owns the impl
    "g(two).val()" => "4.0",
    "g(seed(1.0)).val()" => "2.0",
    // `x * x` stays `T`: the result feeds `+` and `.sin()` through the same bound
    "((f(two).val() - (12.0 + 2.0.sin())).abs() < 1.0e-9)" => "true",
);

test_run!(
    // r[verify mimas.pact.overload]
    overloads_pick_by_operand_through_a_bound,
    NUM,
    // `x * x` is `mul(self, o: Self)`, `x * 3.0` is `mul(self, k: float)`
    "h(two).val()" => "10.0",
    "h(seed(2.0)).val()" => "10.0",
    // the dual carries f's derivative: d/dx (3x² + sin x) at 2 is 12 + cos 2
    "((f(seed(2.0)).d - (12.0 + 2.0.cos())).abs() < 1.0e-9)" => "true",
    // a concrete receiver picks its own overload the same way
    "two.mul(3.0).val()" => "6.0",
    "two.mul(two).val()" => "4.0",
    "(two * 3.0).val()" => "6.0",
);

// r[verify mimas.pact.overload]
// the same operand type twice, an overload with a default body, and an impl
// that supplies only one of the two overloads are each errors
test_fail!(
    pact_overloads_must_differ_and_be_complete,
    "pact P { fn mul(self, o: Self) -> Self; fn mul(self, o: Self) -> float; }",
    "pact P { fn mul(self, o: Self) -> Self; fn mul(self, k: float) -> Self { self } }",
    "pact P { fn mul(self, o: Self) -> Self; fn mul(self, k: float) -> Self; }
     struct R { v: float }
     impl P for R { fn mul(self, o: R) -> R { R { v = self.v * o.v } } }",
    // an operator the bound doesn't declare is still an error
    "pact P { fn add(self, o: Self) -> Self; }
     struct R { v: float }
     impl P for R { fn add(self, o: R) -> R { R { v = self.v + o.v } } }
     fn t<T: P>(x: T) -> T { x * x }
     t(R { v = 1.0 })",
);

// ---- impls for primitive types --------------------------------------
//
// `impl Num for float`: the pact's `Self` is the primitive, one
// `mul(self, o: float)` satisfies both overloads (both read `float` once
// `Self` is), and the runtime chain tests the value's kind through a
// sentinel id instead of an instance id. So `f(2.0)` runs the same `f`.

const PRIM: &str = "
pact Num {
    fn add(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn mul(self, k: float) -> Self;
    fn val(self) -> float;
}
struct Dual { v: float, d: float }
impl Num for Dual {
    fn add(self, o: Dual) -> Dual { Dual { v = self.v + o.v, d = self.d + o.d } }
    fn mul(self, o: Dual) -> Dual { Dual { v = self.v * o.v, d = self.v * o.d + self.d * o.v } }
    fn mul(self, k: float) -> Dual { Dual { v = self.v * k, d = self.d * k } }
    fn val(self) -> float { self.v }
}
impl Num for float {
    fn add(self, o: float) -> float { self + o }
    fn mul(self, o: float) -> float { self * o }
    fn val(self) -> float { self }
}
impl Num for int {
    fn add(self, o: int) -> int { self + o }
    fn mul(self, o: int) -> int { self * o }
    fn mul(self, k: float) -> int { (self.to_float() * k).to_int() }
    fn val(self) -> float { self.to_float() }
}
fn seed(x: float) -> Dual { Dual { v = x, d = 1.0 } }
fn f<T: Num>(x: T) -> T { x * x * 3.0 + x }
fn twice(x: Num) -> Num { x + x }
";

test_run!(
    // r[verify mimas.pact.primitive]
    primitives_implement_pacts,
    PRIM,
    // the same generic `f` on a bare float, an int and a dual
    "f(2.0)" => "14.0",
    "f(2)" => "14",
    "f(seed(2.0)).val()" => "14.0",
    "f(seed(2.0)).d" => "13.0",
    // the chain picks the arm by the value's kind at run time
    "twice(1.5).val()" => "3.0",
    "twice(2).val()" => "4.0",
    "twice(seed(1.0)).val()" => "2.0",
    // the pact's methods are plain methods on the primitive too
    "2.0.mul(3.0)" => "6.0",
    "3.add(4)" => "7",
    // inside the impl, `self * o` on two floats stays the built-in operator
    "2.5 * 2.0" => "5.0",
);

// r[verify mimas.pact.primitive]
// a primitive still can't re-declare a native it already has (`sin` on
// float), and a bound a primitive doesn't implement stays an error
test_fail!(
    primitive_impl_limits,
    "pact P { fn sin(self) -> Self; }
     impl P for float { fn sin(self) -> float { self } }",
    "pact P { fn add(self, o: Self) -> Self; }
     struct R { v: float }
     impl P for R { fn add(self, o: R) -> R { R { v = self.v + o.v } } }
     fn t<T: P>(x: T) -> T { x + x }
     t(2.0)",
);

// a pact name resolves for `P::member` paths but is not a value: a stray
// one (the second name of `impl P for S via Q Q`, or any bare use) is a
// compile error, not an ICE at codegen
test_fail!(
    pact_names_are_not_values,
    "pact P { fn m(self) -> int { 1 } }
     pact Q { fn m(self) -> int { 2 } }
     struct S { x: int }
     impl P for S via Q Q",
    "pact P { fn m(self) -> int { 1 } }
     fn f(x: int) -> int { x }
     f(P)",
    "pact P { fn m(self) -> int { 1 } }
     [P]",
);

const SIGNED: &str = "
pact Signed {
    fn neg(self) -> Self;
    fn add(self, o: Self) -> Self;
    fn zero(self) -> Self;
    fn twice_neg(self) -> Self { -(-self) + -self + self }
    check |a| a + -a == a.zero()
}
struct Money { c: int }
impl Money { fn samples() -> [Money] { [Money { c: 0 }, Money { c: 5 }, Money { c: -3 }] } }
impl Signed for Money {
    fn neg(self) -> Money { Money { c: -self.c } }
    fn add(self, o: Money) -> Money { Money { c: self.c + o.c } }
    fn zero(self) -> Money { Money { c: 0 } }
}
fn flip<T: Signed>(x: T) -> T { -x }
fn flip_pact(x: Signed) -> Signed { -x }";

test_run!(
    // r[verify mimas.pact.unary]
    unary_operators_dispatch_through_a_bound,
    SIGNED,
    // `-x` on a bounded param, on a pact-typed value, and on a pact's `Self`
    "flip(Money { c: 4 }).c" => "-4",
    "flip_pact(Money { c: 4 }) == Money { c: -4 }" => "true",
    "Money { c: 2 }.twice_neg().c" => "2",
);
