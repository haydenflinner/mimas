#[macro_use]
mod test_runner;

test_run!(
    len,
    "[1, 2, 3].len()" => "3",
    "[1].len()" => "1",
    "[].len()" => "0",
);

test_run!(
    len_after_push,
    "let a = [1];
     a.push(2);
     a.push(3);",
    "a.len()" => "3",
);

test_run!(
    push_grows_in_place,
    "let a: [int] = [];
     a.push(10);
     a.push(20);",
    "a.len()" => "2",
    "a[0]" => "10",
    "a[1]" => "20",
);

test_run!(
    push_returns_unit,
    "let a = [1];
     let r = a.push(2);",
    "r" => "null",
);

test_fail!(push_rejects_wrong_elem, "let a = [1, 2]; a.push(\"x\");");

test_run!(
    pop_returns_last,
    "let a = [1, 2, 3];
     let x = a.pop();",
    "x" => "3",
    "a.len()" => "2",
);

test_run!(
    pop_single_then_empty,
    "let a = [99];
     let first = a.pop();
     let second = a.pop();",
    "first" => "99",
    "second" => "null",
    "a.len()" => "0",
);

test_run!(
    pop_empty_is_null,
    "let a: [int] = [];",
    "a.pop()" => "null",
);

test_run!(
    pop_option_coalesce,
    "let a = [5]; let first = a.pop();",
    "first ?? -1" => "5",
    "a.pop() ?? -1" => "-1",
);

test_run!(
    contains,
    "let a = [1, 2, 3];",
    "a.contains(2)" => "true",
    "a.contains(9)" => "false",
);

test_run!(
    contains_empty_is_false,
    "let a: [int] = [];",
    "a.contains(1)" => "false",
);

test_run!(
    contains_str,
    r#"let a = ["x", "y"];"#,
    r#"a.contains("y")"# => "true",
    r#"a.contains("z")"# => "false",
);

test_run!(
    is_empty,
    "let e: [int] = [];
     let f = [1];",
    "e.is_empty()" => "true",
    "f.is_empty()" => "false",
);

test_run!(
    is_empty_after_pop,
    "let a = [1];
     a.pop();",
    "a.is_empty()" => "true",
);

// shuffle: in place, nondeterministic -- assert invariants
test_run!(
    shuffle_preserves_len,
    "let a = [1, 2, 3, 4, 5];
     a.shuffle();",
    "a.len()" => "5",
);

test_run!(
    shuffle_preserves_membership,
    "let a = [10, 20, 30];
     a.shuffle();",
    "a.contains(10)" => "true",
    "a.contains(20)" => "true",
    "a.contains(30)" => "true",
);

test_run!(
    shuffle_single_elem,
    "let a = [7];
     a.shuffle();",
    "a[0]" => "7",
    "a.len()" => "1",
);

test_run!(
    shuffle_empty,
    "let a: [int] = [];
     a.shuffle();",
    "a.len()" => "0",
);

test_run!(
    extend_appends,
    "let a = [1, 2];
     a.extend([3, 4]);",
    "a.len()" => "4",
    "a[2]" => "3",
    "a[3]" => "4",
);

test_run!(
    extend_with_empty,
    "let a = [1, 2];
     let e: [int] = [];
     a.extend(e);",
    "a.len()" => "2",
);

test_run!(
    extend_into_empty,
    "let a: [int] = [];
     a.extend([9, 8]);",
    "a.len()" => "2",
    "a[0]" => "9",
);

test_run!(
    extend_self,
    "let a = [1, 2];
     a.extend(a);",
    "a.len()" => "4",
);

test_fail!(
    extend_rejects_mismatched_elem,
    "let a = [1]; a.extend([\"x\"]);"
);

test_run!(
    enumerate_pairs,
    "let a = [10, 20, 30];
     let e = a.enumerate();",
    "e.len()" => "3",
    "e[0]" => "[0, 10]",
    "e[1]" => "[1, 20]",
    "e[2]" => "[2, 30]",
);

test_run!(
    enumerate_empty,
    "let a: [int] = [];",
    "a.enumerate().len()" => "0",
    "[1, 2, 3, 4].enumerate().len()" => "4",
);

test_run!(
    enumerate_index_access,
    r#"let a = ["a", "b"];
       let e = a.enumerate();"#,
    "e[1].0" => "1",
    "e[1].1" => r#""b""#,
);

test_run!(
    flatten_nested_ints,
    "let a = [[1, 2], [3], [4, 5]];
     let f = a.flatten();",
    "f.len()" => "5",
    "f[0]" => "1",
    "f[4]" => "5",
);

test_run!(
    flatten_nested_strs,
    r#"let a = [["x"], ["y", "z"]];"#,
    "a.flatten().len()" => "3",
    r#"a.flatten().contains("z")"# => "true",
);

test_run!(
    flatten_empty_outer,
    "let a: [[int]] = [];",
    "a.flatten().len()" => "0",
);

test_run!(
    flatten_empty_inners,
    "let a: [[int]] = [[], []];",
    "a.flatten().len()" => "0",
);

test_fail!(flatten_rejects_flat_array, "let _ = [1, 2, 3].flatten();");

// choose: nondeterministic -- assert membership/null
test_run!(
    choose_single,
    "let c: int = [42].choose()!;",
    "c" => "42",
);

test_run!(
    choose_empty_is_null,
    "let a: [int] = [];",
    "a.choose()" => "null",
);

test_run!(
    choose_member,
    "let a = [7, 7, 7];
     let c: int = a.choose()!;",
    "c" => "7",
);

test_run!(
    rand_choose_typed,
    "let n: int = [42].choose()!;
     let s: str = [\"hi\"].choose()!;",
    "n" => "42",
    "s" => r#""hi""#,
);

// join: [str].join
test_run!(
    join_basic,
    r#"let a = ["a", "b", "c"];"#,
    r#"a.join("-")"# => r#""a-b-c""#,
    r#"a.join(", ")"# => r#""a, b, c""#,
    r#"a.join("")"# => r#""abc""#,
);

test_run!(
    join_single,
    r#"["only"].join("-")"# => r#""only""#,
);

test_run!(
    join_empty,
    "let a: [str] = [];",
    r#"a.join("-")"# => r#""""#,
);

test_fail!(join_rejects_int_sep, r#"let a = ["x"]; let _ = a.join(5);"#);

// BUG: `[non-str].join(sep)` panics the VM (array.rs:105 `.expect`); recv erased to `[Val]`, solver
// never constrains to `[str]`. correct behavior: solve-time rejection (test_fail). this panic is
// NOT caught by try_execute -- it aborts the harness -- so guard it ignored until the type
// constraint lands.
test_fail!(
    join_int_array_should_reject,
    r#"let _ = [1, 2, 3].join("-");"#
);

// max / min on int and float arrays are two natives sharing each name (max_int + max_float);
// dispatch picks by unifying the recv pattern against the actual element type. cover both arms.
test_run!(
    max_min_int,
    "[3, 1, 4, 1, 5, 9, 2, 6].max()!" => "9",
    "[3, 1, 4, 1, 5, 9, 2, 6].min()!" => "1",
    "[42].max()!" => "42",
    "[42].min()!" => "42",
    "[-5, -1, -10].max()!" => "-1",
    "[-5, -1, -10].min()!" => "-10",
);

test_run!(
    max_min_int_empty_null,
    "let e: [int] = [];",
    "e.max()" => "null",
    "e.min()" => "null",
);

test_run!(
    max_min_float,
    "[1.5, 2.5, 0.5, 3.5].max()!" => "3.5",
    "[1.5, 2.5, 0.5, 3.5].min()!" => "0.5",
    "[2.25].max()!" => "2.25",
    "[2.25].min()!" => "2.25",
);

test_run!(
    max_min_float_empty_null,
    "let e: [float] = [];",
    "e.max()" => "null",
    "e.min()" => "null",
);

// non-orderable element types rejected at solve time, not a runtime crash (no-overload-matched
// arm of pick_method_overload)
test_fail!(max_rejects_bool, "let _ = [true, false].max();");
test_fail!(min_rejects_str, r#"let _ = ["a", "b"].min();"#);

test_run!(
    sum_int,
    "[1, 2, 3, 4].sum()" => "10",
    "[100].sum()" => "100",
    "[10, -3, -7].sum()" => "0",
);

test_run!(
    sum_int_empty_zero,
    "let e: [int] = [];",
    "e.sum()" => "0",
);

test_run!(
    sum_float,
    "[1.0, 2.0, 3.0].sum()" => "6",
    "[1.5, 2.5].sum()" => "4",
);

// empty float sum is 0.0 (renders `-0`, a whole-float display quirk); equals 0.0 numerically
test_run!(
    sum_float_empty_eq_zero,
    "let e: [float] = [];",
    "e.sum() == 0.0" => "true",
);

test_fail!(sum_rejects_bool, "let _ = [true, false].sum();");

// sort_by_int: in place, stable
test_run!(
    sort_by_int,
    "let a = [10, 20, 30];
     a.sort_by_int([3, 1, 2]);",
    "a[0]" => "20",
    "a[1]" => "30",
    "a[2]" => "10",
    "a.len()" => "3",
);

test_run!(
    sort_by_int_str_values,
    r#"let a = ["x", "y", "z"];
       a.sort_by_int([30, 10, 20]);"#,
    "a[0]" => r#""y""#,
    "a[1]" => r#""z""#,
    "a[2]" => r#""x""#,
);

test_run!(
    sort_by_int_already_sorted,
    "let a = [1, 2, 3];
     a.sort_by_int([1, 2, 3]);",
    "a[0]" => "1",
    "a[2]" => "3",
);

test_run!(
    sort_by_int_empty,
    "let a: [int] = [];
     a.sort_by_int([]);",
    "a.len()" => "0",
);

test_fail!(
    sort_by_int_rejects_float_keys,
    "let a = [1, 2, 3]; a.sort_by_int([1.0, 2.0, 3.0]);"
);

test_run!(
    sort_by_float,
    "let a = [10, 20, 30];
     a.sort_by_float([3.5, 1.5, 2.5]);",
    "a[0]" => "20",
    "a[1]" => "30",
    "a[2]" => "10",
);

test_run!(
    sort_by_float_str_values,
    r#"let a = ["a", "b", "c"];
       a.sort_by_float([2.5, 0.5, 1.5]);"#,
    "a[0]" => r#""b""#,
    "a[1]" => r#""c""#,
    "a[2]" => r#""a""#,
);

test_fail!(
    sort_by_float_rejects_int_keys,
    "let a = [1, 2, 3]; a.sort_by_float([1, 2, 3]);"
);

// argsort dispatches on the receiver key array
test_run!(
    argsort_int,
    "let k = [30, 10, 20];
     let order = k.argsort();",
    "order[0]" => "1",
    "order[1]" => "2",
    "order[2]" => "0",
    "order.len()" => "3",
);

test_run!(
    argsort_int_already_sorted,
    "let order = [1, 2, 3].argsort();",
    "order[0]" => "0",
    "order[1]" => "1",
    "order[2]" => "2",
);

test_run!(
    argsort_int_empty,
    "let k: [int] = [];",
    "k.argsort().len()" => "0",
);

test_run!(
    argsort_int_single,
    "[99].argsort()[0]" => "0",
);

test_run!(
    argsort_float,
    "let k = [3.0, 1.0, 2.0];
     let order = k.argsort();",
    "order[0]" => "1",
    "order[1]" => "2",
    "order[2]" => "0",
);

test_fail!(argsort_rejects_str, r#"let _ = ["a", "b"].argsort();"#);

// reorder: in place, -> ()!
test_run!(
    reorder_permutes,
    "let a = [10, 20, 30];
     a.reorder([2, 0, 1]);",
    "a[0]" => "30",
    "a[1]" => "10",
    "a[2]" => "20",
);

test_run!(
    reorder_identity,
    "let a = [1, 2, 3];
     a.reorder([0, 1, 2]);",
    "a[0]" => "1",
    "a[2]" => "3",
);

test_run!(
    reorder_can_duplicate,
    "let a = [10, 20];
     a.reorder([0, 0, 1]);",
    "a.len()" => "3",
    "a[0]" => "10",
    "a[1]" => "10",
    "a[2]" => "20",
);

test_run!(
    reorder_empty_perm_empties,
    "let a = [1, 2, 3];
     a.reorder([]);",
    "a.len()" => "0",
);

// inscribe_benchmarks.mim pattern: reorder data by a parallel key array's argsort
test_run!(
    reorder_by_argsort,
    r#"let langs = ["mimas", "rune", "luau"];
       let times = [0.126, 0.733, 0.098];
       langs.reorder(times.argsort());"#,
    "langs[0]" => r#""luau""#,
    "langs[1]" => r#""mimas""#,
    "langs[2]" => r#""rune""#,
);

test_fail!(reorder_rejects_oob, "let a = [1, 2]; a.reorder([0, 5]);");
test_fail!(
    reorder_rejects_negative,
    "let a = [1, 2]; a.reorder([-1, 0]);"
);
test_fail!(
    reorder_rejects_str_perm,
    r#"let a = ["x"]; a.reorder(["y"]);"#
);

// assoc fns: array::new / array::new_filled (lowercase form works)
test_run!(
    array_new_lowercase,
    "let a: [int] = array::new();
     a.push(1);
     a.push(2);",
    "a.len()" => "2",
    "a[0]" => "1",
);

test_run!(
    array_new_filled_lowercase,
    "let a = array::new_filled(0, 3);",
    "a.len()" => "3",
    "a[0]" => "0",
    "a[2]" => "0",
);

test_run!(
    array_new_filled_scalar,
    "let a = array::new_filled(true, 3);",
    "a.len()" => "3",
    "a.contains(true)" => "true",
    "a.contains(false)" => "false",
);

test_run!(
    array_new_filled_no_alias,
    "let a = array::new_filled([0], 2);
     a[0].push(9);",
    "a[0].len()" => "2",
    "a[1].len()" => "1",
);

test_run!(
    array_new_filled_clamps_negative,
    "let a = array::new_filled(0, -3);",
    "a.len()" => "0",
);

// chaining / option propagation
test_run!(
    chain_null_array_len,
    "let a: [int]? = null;",
    "a?.len()" => "null",
    "a?.len() ?? -1" => "-1",
);

test_run!(
    chain_alive_array_len,
    "let a: [int]? = [1, 2, 3];",
    "a?.len() ?? -1" => "3",
);

test_run!(
    chain_flatten_then_max,
    "let a = [[3, 1], [4, 1, 5]];",
    "a.flatten().max()!" => "5",
);

test_fail!(no_sort_by_method, "let a = [1, 2]; a.sort_by([1]);");

// -- higher-order methods ------------------------------------------------------
// intrinsics lowered to a generated loop that `Call`s the closure per element;
// signatures ride the anon-slot machinery: map is `[T] -> ((T) -> U) -> [U]`.

test_run!(
    map_ints,
    "[1, 2, 3].map(|x| x * 2)" => "[2, 4, 6]",
    "[10, -5].map(|x| x + 1)" => "[11, -4]",
);

test_run!(
    map_changes_element_type,
    "[1, 2, 3].map(|x| x > 1)" => "[false, true, true]",
    r#"[1, 2].map(|x| "n")"# => r#"["n", "n"]"#,
);

test_run!(
    map_floats_strs,
    "[1.5, 2.5].map(|x| x + 1.0)" => "[2.5, 3.5]",
    r#"["a", "b"].map(|s| s + "!")"# => r#"["a!", "b!"]"#,
);

test_run!(
    map_empty,
    "let e: [int] = [];",
    "e.map(|x| x * 2)" => "[]",
);

test_run!(
    map_does_not_mutate,
    "let a = [1, 2];
     let b = a.map(|x| x * 10);",
    "a" => "[1, 2]",
    "b" => "[10, 20]",
);

// fn items unify against the `(T) -> U` slot, not just closures
test_run!(
    map_named_fn,
    "fn double(x: int) -> int { x * 2 }",
    "[1, 2].map(double)" => "[2, 4]",
);

// captures ride through the closure value the generated loop calls each
// iteration (the same `Val::Closure` a plain `f(x)` would dispatch)
test_run!(
    map_closure_captures_env,
    "let base = 100;",
    "[1, 2].map(|x| x + base)" => "[101, 102]",
    "[1, 2].fold(base, |acc, x| acc + x)" => "103",
);

test_run!(
    map_filter_chain,
    "[1, 2, 3, 4].map(|x| x * 2).filter(|x| x > 4)" => "[6, 8]",
);

// the intrinsic lowers inside whatever body the call sits in -- a nested map
// emits its loop into the *closure's* body, not the caller's
test_run!(
    map_nested,
    "[[1, 2], [3]].map(|inner| inner.map(|x| x + 1))" => "[[2, 3], [4]]",
    "[[1], [2, 3]].map(|inner| inner.sum())" => "[1, 5]",
);

// each call instantiates the anon slots fresh -- one `map` can chain into another
// with a different element type and the checker tracks both
test_run!(
    map_inside_generic_fn,
    "fn run<T, U>(xs: [T], f: (T) -> U, g: (U) -> int) -> int {
        xs.map(|x| g(f(x))).sum()
    }",
    "run([1, 2], |x| x * 3, |y| y + 1)" => "11",
);

test_run!(
    map_on_option_receiver,
    "let a: [int]? = [1, 2];
     let n: [int]? = null;",
    "a?.map(|x| x * 2)" => "[2, 4]",
    "n?.map(|x| x * 2)" => "null",
);

test_run!(
    filter_basic,
    "[1, 2, 3, 4].filter(|x| x % 2 == 0)" => "[2, 4]",
    r#"["", "a", ""].filter(|s| s != "")"# => r#"["a"]"#,
);

test_run!(
    filter_nothing_matches,
    "let e: [int] = [];
     let none = [1, 3, 5].filter(|x| x % 2 == 0);",
    "none" => "[]",
    "e.filter(|x| true)" => "[]",
);

test_run!(
    fold_sum_and_type_change,
    "[1, 2, 3, 4].fold(0, |acc, x| acc + x)" => "10",
    "[[1], [2, 3]].fold(0, |acc, xs| acc + xs.len())" => "3",
    r#"[1, 2].fold("r", |acc, x| acc + "!")"# => r#""r!!""#,
);

test_run!(
    fold_empty_returns_init,
    "let e: [int] = [];",
    "e.fold(42, |acc, x| acc + x)" => "42",
);

test_run!(
    find_hit_and_miss,
    "[1, 2, 3].find(|x| x > 1)" => "2",
    "[1, 2, 3].find(|x| x > 9)" => "null",
    r#"["a", "b"].find(|s| s == "b")"# => r#""b""#,
);

test_run!(
    find_first_match_wins,
    "[5, 6, 7].find(|x| x > 5)" => "6",
    "[1].find(|x| x > 9) ?? -1" => "-1",
);

test_run!(
    any_basic,
    "[1, 2, 3].any(|x| x > 2)" => "true",
    "[1, 2, 3].any(|x| x > 9)" => "false",
);

test_run!(
    any_empty_is_false,
    "let e: [int] = [];",
    "e.any(|x| true)" => "false",
);

test_run!(
    all_basic,
    "[1, 2, 3].all(|x| x > 0)" => "true",
    "[1, 2, 3].all(|x| x > 1)" => "false",
);

test_run!(
    all_empty_is_true,
    "let e: [int] = [];",
    "e.all(|x| false)" => "true",
);

// the walk's bound is frozen at call time: a callback that grows the receiver
// can't extend the iteration (snapshot semantics, no runaway loop)...
test_run!(
    map_callback_push_does_not_extend,
    "let a = [1, 2];
     let m = a.map(|x| { a.push(0); x });",
    "m" => "[1, 2]",
    "a" => "[1, 2, 0, 0]",
);

// ...and one that shrinks it faults on the stale index instead of silently
// skipping elements
test_fail!(
    map_callback_pop_faults_oob,
    "let a = [1, 2, 3]; let _ = a.map(|x| { a.pop(); x });"
);

// a runtime fault inside the callback propagates out of the intrinsic -- mod by
// zero is a checked arithmetic error, it must not be swallowed
test_fail!(
    map_callback_fault_propagates,
    "let _ = [1, 2].map(|x| x % 0);"
);
test_fail!(
    fold_callback_fault_propagates,
    "let _ = [1].fold(0, |acc, x| x % 0);"
);

// solve-time rejections: wrong arity / non-fn arg / non-bool predicate all fail
// as ordinary type errors, never reach codegen
test_fail!(map_rejects_wrong_arity, "let _ = [1, 2].map(|a, b| a);");
test_fail!(map_rejects_non_fn_arg, "let _ = [1, 2].map(5);");
test_fail!(map_rejects_missing_arg, "let _ = [1, 2].map();");
test_fail!(
    map_rejects_mismatched_elem,
    r#"let _ = [1, 2].map(|s| s + "!");"#
);
test_fail!(filter_rejects_non_bool, "let _ = [1, 2].filter(|x| x + 1);");
test_fail!(fold_rejects_wrong_arity, "let _ = [1, 2].fold(0, |a| a);");
test_fail!(
    fold_rejects_mismatched_acc,
    r#"let _ = [1, 2].fold(0, |acc, x| acc + "!");"#
);
test_fail!(find_rejects_non_bool, "let _ = [1, 2].find(|x| x);");
test_fail!(any_rejects_non_bool, "let _ = [1, 2].any(|x| 1);");
test_fail!(all_rejects_non_bool, "let _ = [1, 2].all(|x| 1);");

// mutating the collection you're iterating is a compile error
test_fail!(
    mutate_iterated_array,
    "let a = [1, 2]; for x in a { a.push(0); }",
    "let a = [1, 2]; for x in a { a.pop(); }",
    "let a = [1, 2]; for x in a { a.extend([3]); }",
    "let a = [1, 2]; for x in a { a.shuffle(); }",
    "let a = [1, 2]; for x in a { a.reorder([0]); }",
    "let a = [1, 2]; for x in a { a[0] = 0; }",
);
// reads, element mutation, and aliases through call results are all fine
test_run!(
    reads_allowed_while_iterating,
    "let a = [1, 2];
     let n = 0;
     for x in a { n += a.len(); }",
    "n" => "4",
);
test_run!(
    mutate_element_in_loop_allowed,
    "let a = [[1], [2]];
     for inner in a { inner.push(0); }",
    "a[0].len()" => "2",
);
test_run!(
    push_after_loop_allowed,
    "let a = [1];
     for x in a {}
     a.push(2);",
    "a.len()" => "2",
);

// -- flat_map / mapi / foldi / zip ---------------------------------------------
// flat_map/mapi/foldi are intrinsics like map (the index variants ride the same
// generated loop -- `i` is a real int local), zip is a plain native: pairs are
// just 2-element arrays.

test_run!(
    flat_map_basic,
    "[1, 2].flat_map(|x| [x, x * 10])" => "[1, 10, 2, 20]",
    "[] .flat_map(|x: int| [x])" => "[]",
);

test_run!(
    flat_map_shrinks_and_grows,
    "[1, 2, 3].flat_map(|x| if x % 2 == 0 { [x] } else { [] })" => "[2]",
    r#"[1, 2].flat_map(|x| ["a", "b"])"# => r#"["a", "b", "a", "b"]"#,
);

test_run!(
    mapi_indexes,
    "[10, 20, 30].mapi(|i, x| x + i)" => "[10, 21, 32]",
    r#"["a", "b"].mapi(|i, s| s + i.to_str())"# => r#"["a0", "b1"]"#,
);

test_run!(
    foldi_index_and_acc,
    "[10, 20, 30].foldi(0, |i, acc, x| acc + i * x)" => "80",
    "[] .foldi(7, |i: int, acc: int, x: int| acc)" => "7",
);

test_run!(
    zip_pairs,
    "[1, 2, 3].zip([10, 20, 30])" => "[[1, 10], [2, 20], [3, 30]]",
    // truncates to the shorter side
    "[1, 2, 3].zip([10, 20])" => "[[1, 10], [2, 20]]",
    "[] .zip([1])" => "[]",
);

test_run!(
    zip_pairs_are_indexable,
    "let p = [1, 2].zip([3, 4]);
     let (a, b) = p[0];",
    "p[1].0" => "2",
    "p[0].1" => "3",
    // tuple patterns destructure a pair in one let
    "a * 10 + b" => "13",
);

test_run!(
    mapi_named_fn,
    "fn scale(i: int, x: int) -> int { i * x }",
    "[5, 5, 5].mapi(scale)" => "[0, 5, 10]",
);

test_fail!(flat_map_rejects_non_array, "let _ = [1, 2].flat_map(|x| x);");
test_fail!(mapi_rejects_one_arg, "let _ = [1, 2].mapi(|x| x);");
test_fail!(mapi_rejects_str_index, r#"let _ = [1, 2].mapi(|i: str, x| x);"#);
test_fail!(foldi_rejects_wrong_arity, "let _ = [1, 2].foldi(0, |i, acc| acc);");
test_fail!(zip_rejects_non_array, "let _ = [1, 2].zip(5);");
