use vm::Captured::*;

// a top-level `let` is a global: function bodies read the same entry-frame slot,
// so the value is live wherever the fn runs -- the wall the `use game` seeds hit.
test_vm!(
    fn_reads_top_level_let,
    "let v = 42;
     fn f() -> int { v }",
    "f()" => Int(42),
);

// reads see the binding's *current* value, not the one at fn-definition time
test_vm!(
    fn_sees_reassigned_value,
    "let v = 1;
     fn f() -> int { v }
     v = 2;",
    "f()" => Int(2),
);

// writes inside a fn update the shared slot -- not a fn-local copy
test_vm!(
    fn_writes_top_level_let,
    "let counter = 0;
     fn bump() { counter = counter + 1; }
     bump();
     bump();",
    "counter" => Int(2),
);

// globals hold real values, not just scalars -- a struct instance (the `Voice` case)
test_vm!(
    fn_reads_struct_global,
    "struct Voice { freq: int, gain: int }
     let bass = Voice { freq = 40, gain = 3 };
     fn fire() -> int { bass.freq * bass.gain }",
    "fire()" => Int(120),
);

// and field writes through a global reach the same instance
test_vm!(
    fn_mutates_struct_global,
    "struct Voice { freq: int }
     let bass = Voice { freq = 40 };
     fn tune() { bass.freq = bass.freq + 2; }
     tune();",
    "bass.freq" => Int(42),
);

// a fn-local `let` shadows the global inside that body only
test_vm!(
    fn_local_shadows_global,
    "let v = 1;
     fn f() -> int { let v = 9; v }",
    "f()" => Int(9),
    "v" => Int(1),
);

// ...and a param shadows it too
test_vm!(
    param_shadows_global,
    "let v = 1;
     fn f(v: int) -> int { v }",
    "f(7)" => Int(7),
    "v" => Int(1),
);

// closures read globals live like fns do -- no capture needed for these
test_vm!(
    closure_reads_global,
    "let v = 3;
     fn f() -> int { let g = || v; g() }",
    "f()" => Int(3),
);

test_vm!(
    closure_writes_global,
    "let v = 0;
     fn f() { let g = || { v = v + 10; }; g(); }
     f();",
    "v" => Int(10),
);

// ordinary closure captures still work beside globals
test_vm!(
    closure_captures_local_alongside_global,
    "let g_count = 100;
     fn f() -> int { let x = 5; let c = || x + g_count; c() }",
    "f()" => Int(105),
);

// scoping is sequential: a `let` is a statement, so a fn above it doesn't see it
test_fail!(
    global_not_visible_before_its_let,
    "fn f() -> int { v }
     let v = 1;",
);

// a `let` inside a block is no global -- it dies with the block
test_fail!(
    block_let_not_visible_in_fn,
    "{ let v = 1; }
     fn f() -> int { v }
     f();",
);

// loop vars stay loop-scoped even at top level
test_fail!(
    loop_var_not_visible_in_fn,
    "for i in 1..3 { }
     fn f() -> int { i }
     f();",
);

// two host calls into one fn see each other's write to a top-level `let`: the
// slot is the entry frame's, so a memo or counter there lasts between calls.
// r[verify mimas.global.host-call]
#[test]
fn host_calls_share_a_top_level_let() {
    let mut vm = vm::Vm::compile("let c = 0;\nfn bump() -> int { c += 1; c }", |_| {}).unwrap();
    vm.run().unwrap();
    assert!(matches!(vm.call_fn("bump"), Some(Int(1))));
    assert!(matches!(vm.call_fn("bump"), Some(Int(2))));
}

// C2 exports a page region's item under two keys (`game_loop` and
// `game-kit::game_loop`); the name a body shows (a pause point, a trace
// frame) must not follow a HashMap's iteration order. Twenty compiles, each
// with fresh hash seeds, must agree.
#[test]
fn a_region_item_has_one_name_across_compiles() {
    let src = "//# page: game-kit\nfn game_loop() -> int { 1 }\n//# page: \nlet x = game_loop();";
    for _ in 0..20 {
        let vm = vm::Vm::compile(src, |_| {}).unwrap();
        let body = vm.root().functions["game_loop"].body;
        assert_eq!(vm.chunk_name(body), Some("game_loop"));
    }
}
