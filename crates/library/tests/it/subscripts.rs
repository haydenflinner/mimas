// r[verify mimas.sub.call]
// r[verify mimas.sub.select]
test_run!(
    grid_get_set_and_compound,
    "struct Grid { width: int, cells: [int], writes: [int] }
     impl Grid {
        subscript(x: int, y: int) -> int {
            get { self.cells[y * self.width + x] }
            set { self.writes.push(new_value); self.cells[y * self.width + x] = new_value; }
        }
     }
     let g = Grid { width = 2, cells = [1, 2, 3, 4], writes = [] };
     g[1, 0] = 10;
     g[1, 0] += 7;
     let value = g[1, 0];",
    "value" => "17",
    "g.cells" => "[1, 17, 3, 4]",
    "g.writes" => "[10, 17]",
);

// r[verify mimas.yield.place]
test_run!(
    inout_freezes_the_storage_slot,
    "struct Grid { cells: [int], events: [str], indices: [int] }
     fn next(indices: [int]) -> int { indices.push(1); 0 }
     impl Grid {
        subscript(i: int) -> int {
            inout {
                self.events.push(\"before\");
                yield self.cells[next(self.indices)];
            }
        }
     }
     let g = Grid { cells = [3], events = [], indices = [] };
     g[0] += 4;",
    "g.cells" => "[7]",
    "g.indices.len()" => "1",
    "g.events" => "[\"before\"]",
);

// r[verify mimas.sub.remove]
test_run!(
    removal_uses_the_projecting_collection,
    "struct Element { touched: bool }
     impl Element { fn remove(self) { self.touched = true; } }
     struct Slots { cells: [Element], gone: [bool] }
     impl Slots {
        subscript(i: int) -> Element {
            get { self.cells[i] }
            set { self.cells[i] = new_value; }
            remove { self.gone[i] = true; }
        }
     }
     let g = Slots { cells = [Element { touched = false }], gone = [false] };
     g[0].remove();",
    "g.gone[0]" => "true",
    "g.cells[0].touched" => "false",
);

test_run!(
    named_accessors_and_properties,
    "struct Grid { cells: [int] }
     impl Grid {
        subscript at(i: int) -> int {
            get { self.cells[i] }
            set { self.cells[i] = new_value; }
        }
        subscript first() -> int { inout { yield self.cells[0]; } }
     }
     let g = Grid { cells = [1, 2] };
     g.at[1] = 8;
     g.first += 4;",
    "g.at[1]" => "8",
    "g.first" => "5",
);

test_fail!(
    missing_arms_are_errors,
    "struct G { a: [int] } impl G { subscript(i: int) -> int { get { self.a[i] } } } let g = G { a = [1] }; g[0] = 3;",
    "struct G { a: [int] } impl G { subscript(i: int) -> int { set { self.a[i] = new_value; } } } let g = G { a = [1] }; let v = g[0];",
    "struct G { a: [int] } impl G { subscript(i: int) -> int { get { self.a[i] } } } let g = G { a = [1] }; g[0].remove();",
);

// r[verify mimas.place.marker]
// r[verify mimas.place.let]
test_run!(
    borrowed_accessor_keeps_its_original_storage_and_policy,
    "struct Grid { cells: [int], gone: [bool], calls: [int] }
     fn index(calls: [int]) -> int { calls.push(1); 0 }
     impl Grid {
        subscript(i: int) -> int {
            inout { yield self.cells[index(self.calls)]; }
            remove { self.gone[i] = true; }
        }
     }
     let g = Grid { cells = [3], gone = [false], calls = [] };
     let original = g;
     let p = &g[0];
     let alias = &p;
     g = Grid { cells = [99], gone = [false], calls = [] };
     p += 4;
     alias.remove();",
    "p" => "7",
    "alias" => "7",
    "original.cells[0]" => "7",
    "original.calls.len()" => "1",
    "original.gone[0]" => "true",
    "g.cells[0]" => "99",
    "g.gone[0]" => "false",
);

test_run!(
    builtin_slots_keep_their_address_and_remove_from_the_list,
    "let xs = [1, 2, 3];
     let old = xs;
     let p = &xs[1];
     xs = [9];
     p = 8;
     p.remove();
     let record = Point { x = 4 };
     let field = &record.x;
     field += 3;
     struct Point { x: int }",
    "old" => "[1, 3]",
    "xs" => "[9]",
    "record.x" => "7",
);

// r[verify mimas.place.list-remove]
test_run!(
    builtin_removal_dispatches_on_the_list_place,
    "struct E { called: bool }
     impl E { fn remove(self) { self.called = true; } }
     let e = E { called = false };
     let xs = [e];
     xs[0].remove();",
    "xs.len()" => "0",
    "e.called" => "false",
);

// r[verify mimas.place.list-remove]
test_run!(
    nested_list_slot_removal_shifts_the_inner_list,
    "let xs = [[1, 2], [3, 4]];
     let v = xs[1][0].remove();",
    "v" => "3",
    "xs" => "[[1, 2], [4]]",
);

test_run!(
    nested_assignment_writes_back_a_computed_element,
    "struct E { value: int }
     struct G { cells: [int], writes: [int] }
     impl G {
        subscript(i: int) -> E {
            get { E { value = self.cells[i] } }
            set { self.writes.push(new_value.value); self.cells[i] = new_value.value; }
        }
     }
     let g = G { cells = [1], writes = [] };
     g[0].value = 8;
     g[0].value += 2;",
    "g.cells" => "[10]",
    "g.writes" => "[8, 10]",
);

// r[verify mimas.sub.end]
test_run!(
    finalizers_bracket_reads_writes_and_compound_updates,
    "struct G { cells: [int], log: [int] }
     impl G { subscript(i: int) -> int {
        inout { self.log.push(10); yield self.cells[i]; self.log.push(self.cells[i]); }
     } }
     let g = G { cells = [3], log = [] };
     let read = g[0]; g[0] = 7; g[0] += 2;",
    "read" => "3",
    "g.cells" => "[9]",
    "g.log" => "[10, 3, 10, 7, 10, 9]",
);

// r[verify mimas.sub.cleanup-order]
test_run!(
    borrowed_finalizers_follow_scope_and_run_in_reverse_order,
    "struct G { cells: [int], log: [int] }
     impl G { subscript(i: int) -> int {
        inout { self.log.push(i); yield self.cells[i]; self.log.push(10 + i); }
     } }
     fn touch(g: G) {
        let a = &g[0]; let b = &g[1];
        a += 1; b += 2; g.log.push(5);
     }
     let g = G { cells = [1, 2], log = [] }; touch(g);",
    "g.cells" => "[2, 4]",
    "g.log" => "[0, 1, 5, 11, 10]",
);

test_run!(
    finalizers_follow_return_break_and_continue,
    "struct G { cells: [int], log: [int] }
     impl G { subscript(i: int) -> int {
        inout { yield self.cells[i]; self.log.push(i); }
     } }
     fn touch(g: G) -> int {
        for i in 3 { let p = &g[i]; if i == 0 { continue; } break; }
        let p = &g[2]; return p;
     }
     let g = G { cells = [3, 4, 5], log = [] }; let read = touch(g);",
    "read" => "5",
    "g.log" => "[0, 1, 2]",
);

test_fail!(
    const_and_iteration_guards_survive_places,
    "const xs = [1, 2]; let p = &xs[0];",
    "const xs = [1, 2]; xs[0].remove();",
    "let xs = [1, 2]; for x in xs { xs[0].remove(); }",
    "let xs: [int; 2] = [1, 2]; xs[0].remove();",
);

test_run!(
    fixed_length_slots_are_writable_without_removal,
    "let xs: [int; 2] = [1, 2]; let p = &xs[0]; p = 8;",
    "xs" => "[8, 2]",
);

test_fail!(
    type_and_dimension_errors_are_checked_at_the_accessor,
    "struct G { a: [int] } impl G { subscript(i: int) -> int { inout { yield self.a[i]; } } } let g = G { a = [1] }; g[0] = false;",
    "struct G { a: [px] } impl G { subscript(i: int) -> px { inout { yield self.a[i]; } } } let g = G { a = [1px] }; g[0] = 2s;",
    "struct G { a: [px] } impl G { subscript(i: int) -> px { inout { yield self.a[i]; } } } let g = G { a = [1px] }; g[0] += 2s;",
);

test_run!(
    accessor_elements_can_be_optional,
    "struct G { a: [int?] } impl G { subscript(i: int) -> int? { inout { yield self.a[i]; } } }
     let g = G { a = [null, 3] };
     g[0] ??= 8;
     let p = &g[1]; p = null;",
    "g[0]" => "8",
    "g[1]" => "null",
);

test_fail!(
    place_alias_preserves_dimensions,
    "struct G { a: [px] } impl G { subscript(i: int) -> px { inout { yield self.a[i]; } } } let g = G { a = [1px] }; let p = &g[0]; p += 2s;",
    "let xs = [1px]; let p = &xs[0]; p = 2s;",
);

// r[verify mimas.sub.same-name]
test_fail!(
    accessor_names_cannot_hide_fields_or_methods,
    "struct G { first: int } impl G { subscript first() -> int { get { 3 } } }",
    "struct G { a: int } impl G { fn first(self) -> int { 1 } subscript first() -> int { get { 3 } } }",
    "struct G { a: [int] } impl G { subscript(i: int) -> int { get { self.a[i] } get { 3 } } }",
);

test_run!(
    free_accessor_can_share_a_function_name,
    "fn choose(i: int) -> int { i + 1 }
     pub subscript choose(i: int) -> int { get { i * 2 } }",
    "choose(3)" => "4",
    "choose[3]" => "6",
);

test_run!(
    receiver_keys_and_rhs_run_once_in_order,
    "struct G { cells: [int], log: [int] }
     impl G { subscript(x: int, y: int) -> int {
        get { self.log.push(30); self.cells[x + y] }
        set { self.log.push(40); self.cells[x + y] = new_value; }
     } }
     fn owner(g: G) -> G { g.log.push(10); g }
     fn key(g: G, n: int) -> int { g.log.push(n); n }
     fn rhs(g: G) -> int { g.log.push(20); 5 }
     let g = G { cells = [1, 2], log = [] };
     owner(g)[key(g, 0), key(g, 1)] += rhs(g);",
    "g.cells" => "[1, 7]",
    "g.log" => "[10, 0, 1, 30, 20, 40]",
);

test_fail!(
    removing_a_quantity_preserves_its_dimension,
    "let xs = [1px]; let removed = xs[0].remove(); let invalid = removed + 2s;",
);

test_run!(
    free_places_keep_their_removal_policy,
    "let cells = [1, 2];
     let gone = [false, false];
     subscript cell(i: int) -> int {
        inout { yield cells[i]; }
        remove { gone[i] = true; }
     }
     let p = &cell[1]; p = 7; p.remove();",
    "cells" => "[1, 7]",
    "gone" => "[false, true]",
);

#[test]
fn place_aliases_survive_vm_snapshot_restore() {
    let mut vm = vm::Vm::execute(
        "struct G { cells: [int] }
         impl G { subscript(i: int) -> int { inout { yield self.cells[i]; } } }
         let g = G { cells = [3] };
         let p = &g[0];
         fn bump() { p += 1; }
         fn read() -> int { p }",
        library::std,
    )
    .unwrap();
    let snapshot = vm.snapshot().unwrap();
    vm.call_fn_result("bump").unwrap();
    assert_eq!(vm.call_fn_result("read").unwrap().to_string(), "4");
    vm.restore(&snapshot).unwrap();
    assert_eq!(vm.call_fn_result("read").unwrap().to_string(), "3");
}

// r[verify mimas.sub.end]
// r[verify mimas.sub.cleanup-fault]
#[test]
fn faults_run_finalizers_and_preserve_the_original_error() {
    let mut vm = vm::Vm::execute(
        "struct G { cells: [int], log: [int] }
         impl G { subscript(i: int) -> int {
            inout { yield self.cells[i]; self.log.push(i); if i == 1 { panic(\"cleanup fault\"); } }
         } }
         let g = G { cells = [1, 2], log = [] };
         fn fail() { let a = &g[0]; let b = &g[1]; a += 1; panic(\"original fault\"); }
         fn log() -> [int] { g.log }",
        library::std,
    )
    .unwrap();
    let error = vm.call_fn_result("fail").unwrap_err();
    assert!(format!("{error:?}").contains("original fault"), "{error:?}");
    assert_eq!(vm.call_fn_result("log").unwrap().to_string(), "[1, 0]");
}

// r[verify mimas.sub.cleanup-budget]
#[test]
fn exhausted_fuel_still_runs_bounded_cleanup() {
    let mut vm = vm::Vm::execute(
        "struct G { cells: [int], log: [int] }
         impl G { subscript(i: int) -> int { inout { yield self.cells[i]; self.log.push(7); } } }
         let g = G { cells = [1], log = [] };
         fn fail() { let p = &g[0]; loop {} }
         fn log() -> [int] { g.log }",
        library::std,
    )
    .unwrap();
    vm.set_op_budget(300);
    assert!(vm.call_fn_result("fail").is_err());
    vm.set_op_budget(300);
    assert_eq!(vm.call_fn_result("log").unwrap().to_string(), "[7]");
}

#[test]
fn pending_finalizers_survive_generator_snapshot_restore() {
    let mut vm = vm::Vm::execute(
        "struct G { cells: [int], log: [int] }
         impl G { subscript(i: int) -> int { inout { yield self.cells[i]; self.log.push(7); } } }
         let g = G { cells = [1], log = [] };
         subscript work() -> int { { let p = &g[0]; yield 10; p += 1; } yield 20; }
         let it = work(); let first = it();
         fn next() -> int? { it() }
         fn log() -> [int] { g.log }",
        library::std,
    )
    .unwrap();
    let snap = vm.snapshot().unwrap();
    assert_eq!(vm.call_fn_result("log").unwrap().to_string(), "[]");
    assert_eq!(vm.call_fn_result("next").unwrap().to_string(), "20");
    assert_eq!(vm.call_fn_result("log").unwrap().to_string(), "[7]");
    vm.restore(&snap).unwrap();
    assert_eq!(vm.call_fn_result("next").unwrap().to_string(), "20");
    assert_eq!(vm.call_fn_result("log").unwrap().to_string(), "[7]");
}

test_run!(
    borrowed_setup_places_live_until_the_outer_finalizer,
    "struct G { cells: [int], log: [int] }
     impl G {
        subscript at(i: int) -> int { inout { yield self.cells[i]; self.log.push(10); } }
        subscript first() -> int {
            inout { let nested = &self.at[0]; yield self.cells[1]; self.log.push(nested); }
        }
     }
     fn touch(g: G) { let p = &g.first; p = 8; g.log.push(5); }
     let g = G { cells = [3, 4], log = [] }; touch(g);",
    "g.cells" => "[3, 8]",
    "g.log" => "[5, 3, 10]",
);

#[test]
fn escaped_place_closures_cannot_outlive_the_finalizer() {
    let mut vm = vm::Vm::execute(
        "struct G { cells: [int], log: [int] }
         impl G { subscript(i: int) -> int { inout { yield self.cells[i]; self.log.push(7); } } }
         fn escape(g: G) -> () -> int { let p = &g[0]; || p }
         let g = G { cells = [1], log = [] }; let escaped = escape(g);
         fn read() -> int { escaped() }",
        library::std,
    )
    .unwrap();
    let error = vm.call_fn_result("read").unwrap_err();
    assert!(format!("{error:?}").contains("left its scope"), "{error:?}");
}

#[test]
fn compound_update_fault_runs_cleanup_before_returning() {
    let mut vm = vm::Vm::execute(
        "struct G { cells: [int], log: [int] }
         impl G { subscript(i: int) -> int { inout { yield self.cells[i]; self.log.push(self.cells[i]); } } }
         let g = G { cells = [3], log = [] };
         fn bad() -> int { panic(\"bad rhs\"); }
         fn fail() { g[0] += bad(); }
         fn log() -> [int] { g.log }", library::std,
    ).unwrap();
    assert!(vm.call_fn_result("fail").is_err());
    assert_eq!(vm.call_fn_result("log").unwrap().to_string(), "[3]");
}

#[test]
fn looping_finalizers_cannot_hang_fault_unwinding() {
    let mut vm = vm::Vm::execute(
        "struct G { cells: [int] }
         impl G { subscript(i: int) -> int { inout { yield self.cells[i]; loop {} } } }
         let g = G { cells = [1] };
         fn fail() { let p = &g[0]; panic(\"original fault\"); }
         fn alive() -> int { 7 }",
        library::std,
    )
    .unwrap();
    let error = vm.call_fn_result("fail").unwrap_err();
    assert!(format!("{error:?}").contains("original fault"), "{error:?}");
    assert_eq!(vm.call_fn_result("alive").unwrap().to_string(), "7");
}

#[test]
fn recursive_cleanup_faults_respect_the_native_stack_limit() {
    let mut vm = vm::Vm::execute(
        "fn again() { __cleanup_register(again); panic(\"cleanup fault\"); }
         fn fail() { __cleanup_register(again); panic(\"original fault\"); }
         fn alive() -> int { 7 }",
        library::std,
    )
    .unwrap();
    let error = vm.call_fn_result("fail").unwrap_err();
    assert!(format!("{error:?}").contains("original fault"), "{error:?}");
    assert_eq!(vm.call_fn_result("alive").unwrap().to_string(), "7");
}

// r[verify mimas.list.get]
test_run!(
    indexed_get_is_safe_at_both_bounds,
    "let xs = [3, 4];",
    "xs.get(1)" => "4",
    "xs.get(-1)" => "null",
    "xs.get(2)" => "null",
);

test_fail!(
    indexed_get_keeps_element_dimensions,
    "let xs = [1px]; let x = xs.get(0)! + 2s;",
);

#[test]
fn indexed_get_and_remove_preserve_primitive_storage() {
    #[vm::native]
    fn primitive<'gc>(xs: vm::anon::SeqOf<'gc, 0>) -> bool {
        match xs.0 {
            vm::Val::IntArray(a) | vm::Val::FloatArray(a) => matches!(&*a.0.borrow(), vm::ArrayStore::Ints(_) | vm::ArrayStore::Floats(_)),
            _ => false,
        }
    }
    let mut vm = vm::Vm::execute(
        "let xs = [1, 2, 3]; let ys = [1.5, 2.5];
         let before = primitive(xs) && primitive(ys);
         let x = xs.get(1); let y = ys.get(1);
         xs[0].remove(); ys[0].remove();
         fn result() -> bool { before && primitive(xs) && primitive(ys) && xs[0] == 2 && ys[0] == 2.5 }",
        |api| { library::std(api); api.add(primitive); },
    ).unwrap();
    assert_eq!(vm.call_fn_result("result").unwrap().to_string(), "true");
}

// r[verify mimas.generator.member]
test_run!(
    member_generator_subscript_is_a_lazy_view,
    "struct S { xs: [int], floor: int }
     impl S {
        subscript live() -> int {
            for x in self.xs { if x > self.floor { yield x; } }
        }
     }
     let s = S { xs = [1, 2, 3, 4], floor = 1 };
     fn gather() -> [int] { let out: [int] = []; for v in s.live { out.push(v); } out }
     fn pull() -> int? { let it = s.live; it() }
     fn exhausted() -> int? { let it = s.live; for _v in s.live { it(); } it() }",
    "gather()" => "[2, 3, 4]",
    "pull()" => "2",
);

// r[verify mimas.generator.member]
test_run!(
    member_generator_runs_lazily_between_pulls,
    "struct S { xs: [int], log: [int] }
     impl S {
        subscript live() -> int {
            for x in self.xs { self.log.push(x); yield x; }
        }
     }
     let s = S { xs = [1, 2, 3], log = [] };
     fn after_two() -> [int] { let it = s.live; it(); it(); s.log }
     fn drained() -> int? { let it = s.live; it(); it(); it(); it() }",
    "after_two()" => "[1, 2]",
    "drained()" => "null",
);

// r[verify mimas.generator.member]
test_run!(
    member_generator_takes_bracket_keys,
    "struct S { xs: [int] }
     impl S {
        subscript window(k: int) -> int {
            let i = 0;
            while i < k && i < self.xs.len() {
                yield self.xs[i];
                i += 1;
            }
        }
     }
     let s = S { xs = [1, 2, 3] };
     fn gather() -> [int] { let out: [int] = []; for v in s.window[2] { out.push(v); } out }
     fn wide() -> [int] { let t: S = s; let out: [int] = []; for v in t.window[5] { out.push(v); } out }",
    "gather()" => "[1, 2]",
    "wide()" => "[1, 2, 3]",
);

// r[verify mimas.generator.member]
test_fail!(
    member_generator_rejects_writes_and_borrows,
    "struct S { xs: [int] }
     impl S { subscript live() -> int { for x in self.xs { yield x; } } }
     let s = S { xs = [1] };
     s.live = 5;",
    "struct S { xs: [int] }
     impl S { subscript live() -> int { for x in self.xs { yield x; } } }
     let s = S { xs = [1] };
     let p = &s.live;",
);

// r[verify mimas.generator.member]
test_fail!(
    member_generator_rejects_null_and_unit_elements,
    "struct S { xs: [int] }
     impl S { subscript live() -> int? { yield self.xs[0]; } }",
    "struct S { xs: [int] }
     impl S { subscript live() { } }",
);
