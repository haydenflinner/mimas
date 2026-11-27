use crate::test_runner;

test_run!(
    len_basic,
    r#"let d = { a : 1, b : 2, c : 3 };"#,
    "d.len()" => "3",
);

test_run!(
    len_single,
    r#"let d = { only : 1 };"#,
    "d.len()" => "1",
);

test_run!(
    len_empty,
    "let d: {int} = dict::new();",
    "d.len()" => "0",
);

test_run!(
    len_str_values,
    r#"let d = { a : "x", b : "y" };"#,
    "d.len()" => "2",
);

test_run!(
    len_after_insert,
    r#"let d = { a : 1 };
       d.insert("b", 2);
       d.insert("c", 3);"#,
    "d.len()" => "3",
);

test_run!(
    len_after_remove,
    r#"let d = { a : 1, b : 2, c : 3 };
       d.remove("b");"#,
    "d.len()" => "2",
);

test_run!(
    len_insert_overwrite_no_growth,
    r#"let d = { a : 1 };
       d.insert("a", 99);"#,
    "d.len()" => "1",
);

test_run!(
    len_remove_absent_no_shrink,
    r#"let d = { a : 1, b : 2 };
       d.remove("zzz");"#,
    "d.len()" => "2",
);

test_run!(
    contains_key_present,
    r#"let d = { foo : 1, bar : 2 };"#,
    r#"d.contains_key("foo")"# => "true",
    r#"d.contains_key("bar")"# => "true",
);

test_run!(
    contains_key_absent,
    r#"let d = { foo : 1 };"#,
    r#"d.contains_key("baz")"# => "false",
    r#"d.contains_key("")"# => "false",
);

test_run!(
    contains_key_empty,
    "let d: {int} = dict::new();",
    r#"d.contains_key("anything")"# => "false",
);

test_run!(
    contains_key_after_insert,
    r#"let d = { a : 1 };
       d.insert("new", 2);"#,
    r#"d.contains_key("new")"# => "true",
);

test_run!(
    contains_key_after_remove,
    r#"let d = { a : 1, b : 2 };
       d.remove("a");"#,
    r#"d.contains_key("a")"# => "false",
    r#"d.contains_key("b")"# => "true",
);

// key must be a str; non-str key is a solve-time mismatch
test_fail!(
    contains_key_rejects_int_key,
    r#"let d = { a : 1 }; let _ = d.contains_key(5);"#
);

// dict index -> V?
test_run!(
    index_present_yields_value,
    r#"let d = { a : 10, b : 20 };"#,
    r#"d["a"]"# => "10",
    r#"d["b"]"# => "20",
);

test_run!(
    index_absent_yields_null,
    r#"let d = { a : 1 };"#,
    r#"d["nope"]"# => "null",
);

test_run!(
    index_empty_yields_null,
    "let d: {int} = dict::new();",
    r#"d["x"]"# => "null",
);

test_run!(
    index_coalesce,
    r#"let d = { a : 1 };"#,
    r#"d["a"] ?? -1"# => "1",
    r#"d["z"] ?? -1"# => "-1",
);

test_run!(
    index_unwrap_present,
    r#"let d = { a : 7 };"#,
    r#"d["a"]!"# => "7",
);

test_run!(
    index_into_typed_slot,
    r#"let d = { a : 1 };
       let v: int? = d["a"];
       let m: int? = d["z"];"#,
    "v" => "1",
    "m" => "null",
);

// nested option-chaining through dict indexing
test_run!(
    chain_propagation,
    r#"let nested = { a : { b : { c : 9 } } };
       let hole: {{{int}}} = { a : dict::new() };"#,
    r#"nested["a"]?["b"]?["c"]"# => "9",
    r#"hole["a"]?["b"]?["c"]"# => "null",
);

// non-str index key rejected (dicts are string-keyed)
test_fail!(
    index_rejects_int_key,
    r#"let d = { a : 1 }; let _ = d[5];"#
);

// dict.insert -> previous V?
test_run!(
    insert_overwrite_returns_old,
    r#"let d = { a : 1 };
       let prev: int? = d.insert("a", 99);"#,
    "prev" => "1",
    r#"d["a"]"# => "99",
);

test_run!(
    insert_new_key_returns_null,
    r#"let d = { a : 1 };
       let prev: int? = d.insert("b", 2);"#,
    "prev" => "null",
    r#"d["b"]"# => "2",
);

test_run!(
    insert_into_empty,
    r#"let d: {int} = dict::new();
       let prev: int? = d.insert("first", 5);"#,
    "prev" => "null",
    "d.len()" => "1",
    r#"d["first"]"# => "5",
);

test_run!(
    insert_str_values,
    r#"let d = { a : "x" };
       let prev: str? = d.insert("a", "y");"#,
    "prev" => r#""x""#,
    r#"d["a"]"# => r#""y""#,
);

test_run!(
    insert_repeated_overwrite,
    r#"let d = { k : 0 };
       d.insert("k", 1);
       let prev: int? = d.insert("k", 2);"#,
    "prev" => "1",
    r#"d["k"]"# => "2",
);

// dict.remove -> removed V?
test_run!(
    remove_present_returns_value,
    r#"let d = { a : 1, b : 2 };
       let r: int? = d.remove("b");"#,
    "r" => "2",
    r#"d.contains_key("b")"# => "false",
    "d.len()" => "1",
);

test_run!(
    remove_absent_returns_null,
    r#"let d = { a : 1 };
       let r: int? = d.remove("missing");"#,
    "r" => "null",
    "d.len()" => "1",
);

test_run!(
    remove_from_empty,
    r#"let d: {int} = dict::new();
       let r: int? = d.remove("x");"#,
    "r" => "null",
);

test_run!(
    remove_then_reinsert,
    r#"let d = { a : 1 };
       d.remove("a");
       let prev: int? = d.insert("a", 5);"#,
    "prev" => "null",
    r#"d["a"]"# => "5",
);

test_run!(
    remove_all_keys,
    r#"let d = { a : 1, b : 2 };
       d.remove("a");
       d.remove("b");"#,
    "d.len()" => "0",
    r#"d.contains_key("a")"# => "false",
);

// dict.pairs print like nested arrays (untyped/variadic render path)
test_run!(
    pairs_single,
    r#"let d = { a : 1 };"#,
    "d.pairs()" => r#"[["a", 1]]"#,
);

test_run!(
    pairs_empty,
    "let d: {int} = dict::new();",
    "d.pairs()" => "[]",
);

test_run!(
    pairs_iterate_sum_values,
    r#"let d = { a : 10, b : 20 };
       let total = 0;
       for p in d.pairs() { total = total + p.1; }"#,
    "total" => "30",
);

test_run!(
    pairs_len_and_index,
    r#"let d = { a : 1 };
       let p = d.pairs();"#,
    "p.len()" => "1",
    "p[0].1" => "1",
);

test_run!(
    insert_inline_coalesce,
    r#"let d = { a : 1 };"#,
    r#"d.insert("a", 5) ?? -99"# => "1",
);

test_run!(
    remove_inline_coalesce,
    r#"let d = { a : 1 };"#,
    r#"d.remove("a") ?? -99"# => "1",
);

// insert's value param is anon-linked to the receiver's value type
test_fail!(
    insert_rejects_wrong_value_type,
    r#"fn take(d: {int}) { d.insert("b", "oops"); } let d: {int} = { a : 1 }; take(d);"#
);

// mutating the dict you're iterating is a compile error
test_fail!(
    mutate_iterated_dict,
    r#"let d = { a : 1 }; for (k, v) in d { d.insert("b", 1); }"#,
    r#"let d = { a : 1 }; for (k, v) in d { d.remove("a"); }"#,
    r#"let d = { a : 1 }; for (k, v) in d { d["b"] = 1; }"#,
);
test_run!(
    reads_allowed_while_iterating,
    "let d = { a : 1, b : 2 };
     let n = 0;
     for p in d { n += p.1; }",
    "n" => "3",
);

// `{:}` is the empty dict; its value type comes from where it goes
// r[verify mimas.dict.empty-literal]
test_run!(
    empty_dict_literal,
    r#"let d: {int} = {:};
       let names: {[str]} = {:};
       names["forth"] = ["dup", "drop"];
       fn count(words: [str]) -> {int} {
           let seen: {int} = {:};
           for w in words { seen[w] = (seen[w] ?? 0) + 1; }
           seen
       }"#,
    "d.len()" => "0",
    r#"names["forth"]!.len()"# => "2",
    r#"count(["a", "b", "a"])["a"]!"# => "2",
    "{ let e: {str} = {:}; e.len() }" => "0",
);

// ---- value keys (`mimas.dict.key`) -----------------------------------------

// r[verify mimas.dict.key.type]
// r[verify mimas.dict.key.index]
test_run!(
    int_keys_look_up_by_key,
    r#"let d: {int: str} = {:};
       d[10] = "ten";
       d[3] = "three";"#,
    "d[3] ?? \"none\"" => "\"three\"",
    // a dict index by int is the key, never the position
    "d[0] ?? \"none\"" => "\"none\"",
    "d[1] ?? \"none\"" => "\"none\"",
    "d.keys()" => "[10, 3]",
);

// r[verify mimas.dict.key.hash]
test_run!(
    equal_tuples_are_one_key,
    r#"let d: {(int, int): int} = {:};
       d[(1, 2)] = 1;
       let p = (1, 2);
       d[p] = d[p]! + 5;
       d[(2, 1)] = 9;"#,
    "d.len()" => "2",
    "d[(1, 2)]!" => "6",
    "d[(2, 1)]!" => "9",
    "d[(3, 3)] ?? -1" => "-1",
);

// r[verify mimas.dict.key.in]
test_run!(
    in_takes_any_key_type,
    r#"let d: {(int, str): bool} = {:};
       d[(1, "a")] = true;
       let names = { a: 1 };"#,
    r#"(1, "a") in d"# => "true",
    r#"(1, "b") in d"# => "false",
    r#"!((2, "a") in d)"# => "true",
    r#""a" in names"# => "true",
);

// r[verify mimas.dict.key.for]
test_run!(
    for_walks_value_keys_in_order,
    r#"let d: {int: int} = {:};
       d[5] = 50; d[0] = 0; d[7] = 70;
       d.remove(0);
       d[0] = 1;
       let ks: [int] = [];
       let vs: [int] = [];
       for (k, v) in d { ks.push(k); vs.push(v); }"#,
    "ks" => "[5, 7, 0]",
    "vs" => "[50, 70, 1]",
);

test_run!(
    bool_dec_and_float_keys,
    r#"let b: {bool: str} = {:};
       b[true] = "yes"; b[false] = "no";
       let m: {dec: int} = {:};
       let wide: dec = 1.50;
       let narrow: dec = 1.5;
       m[wide] = 1;
       let f: {float: int} = {:};
       f[0.0] = 1; f[-0.0] = 2; f[0.5] = 3;"#,
    "b[false]!" => "\"no\"",
    // equal decs are one key, whatever their scale
    "m[narrow]!" => "1",
    "f.len()" => "2",
    "f[0.0]!" => "2",
);

// an atom is a value of the built-in `atom` enum, so it keys like one
test_run!(
    atom_keys,
    r#"let d = {:};
       d[#Red] = 1;
       d[#Blue] = 2;
       d[#Red] = d[#Red]! + 10;"#,
    "d[#Red]!" => "11",
    "#Blue in d" => "true",
    "d.len()" => "2",
    "tally([#A, #B, #A]).weight(#A)" => "2",
);

// a generic dict keys by whatever `T` is at the call
test_run!(
    generic_keys,
    r#"fn count<T>(xs: [T]) -> {T: int} {
           let out: {T: int} = {:};
           for x in xs { out[x] = (out[x] ?? 0) + 1; }
           out
       }"#,
    r#"count(["a", "b", "a"])["a"]!"# => "2",
    "count([(1, 1), (2, 2), (1, 1)])[(1, 1)]!" => "2",
    "count([3, 3, 3])[3]!" => "3",
);

// r[verify mimas.dict.key.copy]
test_run!(
    struct_key_is_its_value_at_insert,
    r#"struct P { x: int, y: int }
       let d: {P: str} = {:};
       let p = P { x: 1, y: 2 };
       d[p] = "a";
       p.x = 9;"#,
    "d[P { x: 1, y: 2 }] ?? \"missing\"" => "\"a\"",
    "d[p] ?? \"missing\"" => "\"missing\"",
    "d.len()" => "1",
);

test_run!(
    dict_methods_take_value_keys,
    r#"let e: {(int, str): int} = dict::new();
       e.insert((1, "x"), 2);
       e.insert((2, "y"), 3);
       let old = e.insert((1, "x"), 4);
       e.remove((2, "y"));"#,
    "old" => "2",
    r#"e.contains_key((1, "x"))"# => "true",
    r#"e.contains_key((2, "y"))"# => "false",
    "e.pairs()" => r#"[[[1, "x"], 4]]"#,
);

// `{str: V}` is `{V}`
test_run!(
    str_colon_is_the_plain_dict,
    r#"let a: {str: int} = { k: 1 };
       let b: {int} = a;"#,
    r#"b["k"]!"# => "1",
);

test_run_display!(
    value_keys_display_as_values,
    r#"{ let d: {int: str} = {:}; d[2] = "two"; d }"# => r#"{2: "two"}"#,
    r#"{ let d: {bool: int} = {:}; d[true] = 1; d }"# => "{true: 1}",
);

fn error_text(src: &str) -> String {
    match test_runner::try_execute(src) {
        Ok(()) => panic!("expected `{src}` to fail"),
        Err(e) => format!("{e:?}"),
    }
}

// r[verify mimas.dict.key.refused]
#[test]
fn checker_refuses_mutable_and_function_keys() {
    for (src, ty, why) in [
        ("let d: {[int]: int} = {:};", "`[int]` cannot be a dict key", "use a tuple"),
        ("let d: {{int}: int} = {:};", "cannot be a dict key", "a dict can change"),
        ("let d: {(int) -> int: int} = {:};", "cannot be a dict key", "a function has no equality"),
        // the first key settles an open `{:}`, and is checked too
        ("let d = {:}; d[[1, 2]] = 3;", "`[int]` cannot be a dict key", "use a tuple"),
        (
            "struct Bag { xs: [int] } let d: {Bag: int} = {:};",
            "cannot be a dict key",
            "a `Bag` holds a value that cannot be a key",
        ),
    ] {
        let e = error_text(src);
        assert!(e.contains(ty) && e.contains(why), "`{src}`: {e}");
    }
}

// r[verify mimas.dict.key.runtime]
#[test]
fn run_time_refuses_nan_keys() {
    let e = error_text("let f: {float: int} = {:}; let z = 0.0; f[z / z] = 1;");
    assert!(e.contains("NaN cannot be a dict key"), "{e}");
    let e = error_text("let f: {(int, float): int} = {:}; let z = 0.0; f[(1, z / z)] = 1;");
    assert!(e.contains("NaN cannot be a dict key"), "{e}");
}

// r[verify mimas.dict.key.copy]
#[test]
fn a_key_read_back_is_read_only() {
    let e = error_text(
        "struct P { x: int, y: int }
         let d: {P: int} = {:};
         d[P { x: 1, y: 2 }] = 5;
         let k = d.keys()[0];
         k.x = 7;",
    );
    assert!(e.contains("cannot change a constant") && e.contains("is a key a"), "{e}");
}

// r[verify mimas.dict.key.snapshot]
#[test]
fn value_keys_survive_snapshot_restore() {
    let mut vm = vm::Vm::execute(
        "let d: {(int, int): int} = {:};
         d[(1, 2)] = 1; d[(3, 4)] = 2; d[(5, 6)] = 3;
         fn bump() { d[(3, 4)] = d[(3, 4)]! + 10; d[(7, 8)] = 4; }
         fn read() -> str { f\"{d[(1, 2)]!} {d[(3, 4)]!} {d[(5, 6)]!} {d.len()}\" }",
        library::std,
    )
    .unwrap();
    let snapshot = vm.snapshot().unwrap();
    vm.call_fn_result("bump").unwrap();
    assert_eq!(vm.call_fn_result("read").unwrap().to_string(), "\"1 12 3 4\"");
    vm.restore(&snapshot).unwrap();
    assert_eq!(vm.call_fn_result("read").unwrap().to_string(), "\"1 2 3 3\"");
    vm.call_fn_result("bump").unwrap();
    assert_eq!(vm.call_fn_result("read").unwrap().to_string(), "\"1 12 3 4\"");
}

// A dict literal takes any key type `mimas.dict.key` allows (combo C15,
// e18): ints, negative numbers, floats, enum paths, tuples and structs.
// A bare name key stays a record field (`{x: 1}` keys the string "x").
// r[verify mimas.dict.key.literal]
test_run!(
    literal_value_keys,
    "enum C { R, G }\nstruct P { x: int }\nlet x = 5\nconst NAMES = {1: \"one\", 2: \"two\"}\n",
    r#"{1: "a", 2: "b"}[2]"# => r#""b""#,
    r#"{C::R: 1, C::G: 2}[C::G]"# => "2",
    r#"{(1, 2): "b", (3, 4): "c"}[(3, 4)]"# => r#""c""#,
    r#"{-1: "neg"}[-1]"# => r#""neg""#,
    r#"{1.5: "f"}[1.5]"# => r#""f""#,
    r#"{P { x: 1 }: "p"}[P { x: 1 }]"# => r#""p""#,
    r#"{x: 1}["x"]"# => "1",
    "{ x }" => "5",
    "NAMES[2]" => r#""two""#,
    "{\n  1: \"one\",\n  2: \"two\",\n}.len()" => "2",
);
