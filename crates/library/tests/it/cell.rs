//! `Cell<T>`: one shared, writable value a closure may write through.

test_run!(
    // r[verify mimas.std.cell]
    // SICP's make-withdraw: the returned closure keeps its balance
    make_withdraw_keeps_its_balance,
    "fn make_withdraw(start: int) -> (int) -> str {
    let b = Cell { value: start };
    |amount: int| {
        if amount > b.value { return \"Insufficient funds\"; }
        b.value -= amount;
        f\"{b.value}\"
    }
}
let w = make_withdraw(100);
let first = w(50);
let second = w(70);",
    "first" => "\"50\"",
    "second" => "\"Insufficient funds\"",
);

test_run!(
    // r[verify mimas.std.cell]
    // two closures share one cell
    two_closures_share_one_cell,
    "let c = Cell { value: 0 };
let inc = || { c.value += 1; };
let read = || c.value;
inc();
inc();",
    "read()" => "2",
    "c.value" => "2",
);

test_run!(
    // r[verify mimas.std.cell]
    // the field is a place: `inout` and `&` reach it
    a_cell_field_is_a_place,
    "fn bump(x: inout int) { x += 10; }
let c = Cell { value: 1 };
bump(&c.value);
// and from inside a closure that captured the cell
let again = || bump(&c.value);
again();",
    "c.value" => "21",
);

#[test]
// r[verify mimas.closure.capture-write]
fn the_capture_error_suggests_a_cell() {
    let err = format!(
        "{:?}",
        crate::test_runner::try_execute("fn f() -> int { let b = 1; let g = || { b = 2; }; g(); b }\nlet TEST_VALUE = f();").unwrap_err()
    );
    assert!(err.contains("wrap it in a `Cell`"), "{err}");
}
