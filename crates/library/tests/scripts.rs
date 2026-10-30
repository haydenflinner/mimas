#[macro_use]
mod test_runner;

// A script has no `main`: top-level items are visible throughout the file, wherever
// they are declared; top-level statements run in order.
// r[verify mimas.script.toplevel]
test_run!(
    items_are_visible_throughout_the_file,
    "let a = f(K, S { v = 2 }.v);
     let e = Dir::Up;
     fn f(x: int, y: int) -> int { x + y }
     const K = 3;
     struct S { v: int }
     enum Dir { Up, Down }",
    "a" => "5",
);

// r[verify mimas.script.toplevel]
test_run!(
    statements_run_top_to_bottom,
    "let log = [0];
     log.push(1);
     log.push(2);",
    "log.len()" => "3",
    "log[2]" => "2",
);

// A top-level `let` is only visible below its declaration.
// r[verify mimas.script.toplevel]
test_fail!(
    a_top_level_let_is_not_visible_above_itself,
    "let y = z; let z = 1;",
);
