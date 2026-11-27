//! `impl S<T: Bound>` bounds a type's own parameter (L13, first half).

test_run!(
    // r[verify mimas.generic.impl-bound]
    an_impl_bounds_the_types_parameter,
    "pact Weigh { fn w(self) -> int; }
impl Weigh for int { fn w(self) -> int { self } }
struct Pair2<T> { a: T, b: T }
impl Pair2<T: Weigh> { fn total(self) -> int { self.a.w() + self.b.w() } }",
    "Pair2 { a = 2, b = 3 }.total()" => "5",
);

test_fail!(
    // r[verify mimas.generic.impl-bound]
    an_unbounded_impl_parameter_stays_refused,
    "struct Pair2<T> { a: T, b: T }
impl Pair2<T> { fn first(self) -> T { self.a } }
let TEST_VALUE = 1;",
    "pact Weigh { fn w(self) -> int; }
struct Pair2<T> { a: T, b: T }
impl Pair2<U: Weigh> { fn first(self) -> int { 1 } }
let TEST_VALUE = 1;",
);
