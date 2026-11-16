#[macro_use]
mod test_runner;

test_run!(
    and_or_short_circuit,
    "let log: [int] = [];
     let effect = |v: bool| { log.push(1); v };
     let a = false && effect(true);
     let b = true && effect(true);
     let c = true || effect(true);
     let d = false || effect(true);",
    "log.len()" => "2",
);

test_run!(
    coalesce_short_circuits,
    "let log: [int] = [];
     let effect = || { log.push(1); 9 };
     let present: int? = 5;
     let absent: int? = null;
     let a = present ?? effect();
     let b = absent ?? effect();",
    "log.len()" => "1",
);

test_run!(
    option_call_skips_args_on_null,
    "struct S { x: int }
     impl S { fn m(self, v: int) -> int { v } }
     let log: [int] = [];
     let effect = || { log.push(1); 3 };
     let some: S? = S { x = 1 };
     let none: S? = null;
     let a = some?.m(effect());
     let b = none?.m(effect());",
    "log.len()" => "1",
);

// r[verify mimas.generator.decl]
// r[verify mimas.generator.pull]
// r[verify mimas.generator.end]
test_run!(
    generator_is_lazy_and_resumable,
    "subscript count(log: [int]) -> int {
        let i = 0;
        while i < 3 { log.push(i); yield i; i += 1; }
     }
     let log: [int] = [];
     let it = count(log);
     let before = log.len();
     let first = it();
     let second = it();
     let tail = for x in it { collect x };
     let exhausted = it();",
    "before" => "0",
    "first" => "0",
    "second" => "1",
    "tail" => "[2]",
    "log" => "[0, 1, 2]",
    "exhausted" => "null",
);

// r[verify mimas.generator.for]
test_run!(
    generator_break_keeps_cursor,
    "subscript naturals() -> int { let i = 0; loop { yield i; i += 1; } }
     let it = naturals();
     let found = for x in it { if x == 4 { break x; } };
     let next = it();",
    "found" => "4",
    "next" => "5",
);

test_run!(
    nested_generators_and_continue,
    "subscript leaf(n: int) -> int { for i in n { yield i; } }
     subscript tree() -> int { for n in 4 { for x in leaf(n) { yield x; } } }
     let out = for x in tree() { if x == 1 { continue; } collect x };
     let a = leaf(2);
     let b = leaf(3);
     let pair = [a(), b(), b(), a()];",
    "out" => "[0, 0, 0, 2]",
    "pair" => "[0, 0, 1, 1]",
);

test_fail!(
    generator_checks_yield_types,
    "subscript bad() -> int { yield false; }",
    "subscript bad() -> int? { yield null; }",
    "subscript bad() -> () { yield (); }",
    "subscript bad() -> int { let f = || { yield 1; }; }",
);

// r[verify mimas.generator.snapshot]
#[test]
fn generator_snapshot_replays_suspended_locals() {
    let mut vm = vm::Vm::execute(
        "subscript count() -> int { for i in 4 { yield i; } }
         let it = count(); let first = it();
         fn next() -> int? { it() }",
        library::std,
    )
    .unwrap();
    let snapshot = vm.snapshot().unwrap();
    assert_eq!(vm.call_fn_result("next").unwrap().to_string(), "1");
    assert_eq!(vm.call_fn_result("next").unwrap().to_string(), "2");
    vm.restore(&snapshot).unwrap();
    assert_eq!(vm.call_fn_result("next").unwrap().to_string(), "1");
}

// r[verify mimas.generator.budget]
#[test]
fn generator_shares_the_callers_fuel() {
    let mut vm = vm::Vm::compile(
        "subscript spin() -> int { loop {} yield 1; } let it = spin(); let x = it();",
        library::std,
    )
    .unwrap();
    vm.set_op_budget(1000);
    let err = vm.run().unwrap_err();
    assert!(
        format!("{err:?}").contains("fuel") || err.to_string().contains("budget"),
        "{err:?}"
    );
}

#[vm::native]
fn pause_generator_frame<'gc>(ctx: vm::Ctx<'gc>) {
    ctx.state().paused.set(true);
}

#[test]
fn generator_can_pause_a_game_frame_before_emitting() {
    let mut vm = vm::Vm::compile(
        "subscript frames() -> int { pause_generator_frame(); yield 7; }
         let it = frames(); let result = it();",
        |api| {
            library::std(api);
            api.add(pause_generator_frame);
        },
    )
    .unwrap();
    assert!(!vm.run_frame().unwrap());
    assert!(vm.run_frame().unwrap());
    assert_eq!(vm.resolve_name("result").unwrap().to_string(), "7");
}

unsafe extern "C" fn compiled_generator_must_fall_back(env: *const vm::bc::BodyEnv<'_>) {
    unsafe { *(*env).out = Err(vm::RtErr::Custom("compiled generator path was used".into())); }
}

#[test]
fn generator_program_uses_the_interpreter_with_compiled_bodies_installed() {
    let mut vm = vm::Vm::compile(
        "subscript one() -> int { yield 1; } let result = for x in one() { collect x };",
        library::std).unwrap();
    vm.install_bc(vec![Some(compiled_generator_must_fall_back); 512]);
    vm.run().unwrap();
    assert_eq!(vm.resolve_name("result").unwrap().to_string(), "[1]");
}

test_run!(
    generator_preserves_dimensions_and_contextual_names,
    "subscript lengths() -> px { let yield = 1px; yield = 2px; yield yield; }
     let out = for x in lengths() { collect x.to(1px) };",
    "out" => "[2.0]",
);

test_run!(
    generator_reads_and_writes_live_globals,
    "let global = 3;
     fn bump() { global += 1; }
     subscript values() -> int { yield global; bump(); yield global; }
     let it = values();
     let a = it(); global = 10; let b = it();",
    "a" => "3",
    "b" => "11",
    "global" => "11",
);

test_fail!(
    generator_cannot_suspend_an_effect_scope,
    "subscript bad() -> int { with rng = 3 { yield 1; } }",
);

test_run!(
    generator_can_pull_inside_an_effect_scope,
    "subscript dice() -> float { yield float::random(1.0); }
     let xs = with rng = [0.25] { for x in dice() { collect x } };",
    "xs" => "[0.25]",
);

// r[verify mimas.stmt.double-pipe]
test_run!(
    leading_double_pipe_evaluates_and_short_circuits,
    "let log: [int] = [];
     let effect = |v: bool| { log.push(1); v };
     let a = true
         || effect(false);
     let b = false
         || effect(true);
     let c = false
         || false && true
         || 2 + 3 == 5;
     let d = false
         or true;
     let mixed = true
         || false && false;
     let inline = true || false && false;
     let grouped = true
         || (false && false);",
    "a" => "true",
    "b" => "true",
    "c" => "true",
    "d" => "true",
    "mixed == inline" => "true",
    "grouped" => "true",
    "log.len()" => "1",
);

// r[verify mimas.stmt.double-pipe]
test_run!(
    leading_double_pipe_in_conditions_and_checks,
    "let chosen = if false
         || true { 7 } else { 9 };
     let n = 0;
     while n == 0
         || n == 1 { n += 1; }
     check false
         || true",
    "chosen" => "7",
    "n" => "2",
);

// r[verify mimas.stmt.double-pipe]
test_run!(
    zero_argument_closures_keep_their_spelling,
    "fn make() -> () -> int { || 7 }
     fn standalone() -> int { true;
         || 23;
         29 }
     fn call(f: () -> int) -> int { f() }
     let f =
         || 5;
     let g = make();
     let result = call(
         || 9);
     let block = { let x = 11; || x };
     let callbacks = [|| 13, || 17];",
    "f()" => "5",
    "g()" => "7",
    "result" => "9",
    "block()" => "11",
    "callbacks[0]()" => "13",
    "callbacks[1]()" => "17",
    "standalone()" => "29",
);
