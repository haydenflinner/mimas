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
    // grouping applies in a call too: `1,234` is one number
    a_grouped_number_in_a_call_is_one_number,
    "fn one(a: int) -> int { a }",
    "one(1,234)" => "1234",
    "[1,000, 2].len()" => "2",
);

#[test]
// r[verify mimas.num.group-hint]
fn a_failed_call_names_a_grouped_number() {
    use crate::test_runner::try_execute;
    // arity: `100,200` is one number, so `clamp` gets two arguments
    let err = format!(
        "{:?}",
        try_execute(
            "fn clamp(x: int, lo: int, hi: int) -> int { if x < lo { lo } else if x > hi { hi } else { x } }\nlet x = 5;\nlet TEST_VALUE = clamp(x,100,200);"
        )
        .unwrap_err()
    );
    assert!(err.contains("`100,200` is one number here; write `100, 200` for two"), "{err}");
    // a type mismatch: one int where a pair is wanted
    let err = format!(
        "{:?}",
        try_execute("fn take(p: (int, int)) -> int { p.0 }\nlet TEST_VALUE = take(1,234);").unwrap_err()
    );
    assert!(err.contains("`1,234` is one number here; write `1, 234` for two"), "{err}");
    // no hint when no grouped number is in sight
    let err = format!(
        "{:?}",
        try_execute("fn take(p: (int, int)) -> int { p.0 }\nlet TEST_VALUE = take(1);").unwrap_err()
    );
    assert!(!err.contains("is one number here"), "{err}");
}

test_run!(
    // r[verify mimas.num.groups]
    // only well-formed thousands notation groups: a leading group of 1-3
    // digits with no leading `0`, then groups of exactly 3
    thousands_notation_is_well_formed,
    "fn n2(a: int, b: int) -> int { a * 1000 + b }\nfn n3(a: int, b: int, c: int) -> int { a * 1000000 + b * 1000 + c }\nfn one(a: int) -> int { a }",
    "n3(0,128,255)" => "128255",
    "n2(0,100)" => "100",
    "n2(0,500)" => "500",
    "n2(255128,0)" => "255128000",
    "n2(1234,567)" => "1234567",
    "one(1,234)" => "1234",
    "{ let n = 1,000,000; n }" => "1000000",
    "250,000.5" => "250000.5",
    "[1,000, 2].len()" => "2",
    "[100,200,300].len()" => "1",
);

#[test]
// r[verify mimas.num.group-hint]
fn rgb_with_unspaced_commas_groups_and_hints() {
    use crate::test_runner::try_execute;
    // `255,128` groups (a well-formed run), so `rgb` gets two arguments
    let err = format!(
        "{:?}",
        try_execute("fn rgb(r: int, g: int, b: int) -> int { r }\nlet TEST_VALUE = rgb(255,128,0);").unwrap_err()
    );
    assert!(err.contains("`255,128` is one number here; write `255, 128` for two"), "{err}");
    // a list literal argument: `[100,200,300]` is one number, and the
    // type check on the call then fails
    let err = format!(
        "{:?}",
        try_execute("fn three(xs: [str]) -> int { xs.len() }\nlet TEST_VALUE = three([100,200,300]);").unwrap_err()
    );
    assert!(err.contains("`100,200,300` is one number here; write `100, 200,300` for two"), "{err}");
}
