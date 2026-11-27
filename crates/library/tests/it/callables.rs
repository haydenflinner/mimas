//! Callables agree (L12): a `fn` binds with `let`, a function-typed field
//! is called as `s.f(x)`, a type with `fn call(self, …)` is called as
//! `v(…)`, and an optional function type needs no extra parentheses.

const SRC: &str = "fn double(x: int) -> int { x * 2 }
struct Btn { label: str, on: (int) -> int }
struct Adder { k: int }
impl Adder { fn call(self, x: int) -> int { self.k + x } }
let b = Btn { label = \"ok\", on = |x: int| x + 1 };
let add3 = Adder { k = 3 };";

test_run!(
    // r[verify mimas.call.let-fn]
    a_fn_binds_with_let,
    SRC,
    "{ let f = double; f(4) }" => "8",
);

test_run!(
    // r[verify mimas.call.field]
    a_function_field_is_called_like_a_method,
    SRC,
    "b.on(1)" => "2",
    "(b.on)(1)" => "2",
);

test_run!(
    // r[verify mimas.call.call-method]
    a_value_with_call_is_called,
    SRC,
    "add3(4)" => "7",
);

test_run!(
    // r[verify mimas.call.optional-fn]
    an_optional_function_type,
    "fn maybe(f: ((int) -> int)?, x: int) -> int { if f == null { x } else { f!(x) } }",
    "maybe(null, 5)" => "5",
    "maybe(|x: int| x * 10, 5)" => "50",
);

test_run!(
    // r[verify mimas.num.groups]
    // L2: a call's arguments are separated by every comma
    a_comma_in_a_call_always_separates,
    "fn clamp3(x: int, lo: int, hi: int) -> int { if x < lo { lo } else if x > hi { hi } else { x } }",
    "clamp3(150,100,200)" => "150",
    "clamp3(50,100,200)" => "100",
    "[clamp3(999,100,200), 1,000].len()" => "3",
);
