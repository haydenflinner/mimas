// assignment targets
test_ok!(
    valid_assignment_targets,
    "a = 1;",
    "a.b = 1;",
    "a[0] = 1;",
    "a.b.c = 1;"
);
test_fail!(
    invalid_assignment_targets,
    "a() = 1;",
    "fn() {} = 1;",
    "(a) = 1;",
    "true = 1;",
    "1 = 1;",
    "a + b = 1;"
);

// dot access
test_ok!(valid_dot_access, "a = b.c;", "a = b.0;");
test_fail!(invalid_dot_access_operator, "a = b.+;", "a = b.;");

// call arguments
test_ok!(
    positional_then_named_ok,
    "foo(1, a=2);",
    "foo(1, 2, a=3, b=4);"
);
test_fail!(named_before_positional, "foo(a=1, 2);", "foo(x=1, y=2, 3);");

// missing semicolons / unexpected tokens
test_fail!(missing_semicolon_let, "let a = 5", "let a = 5 let b = 6;");
test_fail!(missing_semicolon_assign, "a = 1");
// a `}` ends a statement the way a newline does -- `{ i += 1 }` needs no `;`
test_ok!(
    statement_before_closing_brace,
    "fn f() { a = 1 }",
    "fn f() { a += 1 }",
    "while true { i = i + 1 }",
    "fn f() { let x = 1 }",
    "if c { a.b = 1 } else { a[0] -= 2 }"
);
// `mut` is only an error right before a binding; alone it's an ordinary name
test_ok!(mut_is_still_a_name, "let mut = 1;", "let mut: int = 1;");
test_fail!(let_mut_is_rejected, "let mut i = 0;", "let mut (a, b) = (1, 2);");
test_fail!(unexpected_token_close_paren, "a = );", "a = ;");
test_fail!(unexpected_token_double_colon_leading, "a = ::;");

// expected identifier
test_fail!(
    expected_identifier_stray_symbol,
    "a = @;",
    "a = `;",
    "a = #;"
);
test_fail!(
    keyword_as_name,
    "let if = 1;",
    "let fn = 1;",
    "let self = 1;",
    "let match = 1;",
    "let return = 1;"
);

// unexpected end of input
test_fail!(unexpected_end_let, "let", "let a", "let a =", "let a = 1 +");
test_fail!(unexpected_end_block, "fn f() {", "if true", "match x {");

// unmatched brackets
test_fail!(unmatched_brackets, "{", "[", "(", "~{", "fn foo() {");

// impl items
test_ok!(
    valid_impl_items,
    "impl Foo { fn x() {} }",
    "impl Foo { const X: int = 1; }",
    "impl Drawable for Foo {}",
    "impl Drawable for Foo { fn draw() {} }"
);
test_fail!(
    invalid_impl_item,
    "impl Foo { let x = 1; }",
    "impl Foo { struct S; }",
    "impl Foo for Bar { x = 1; }"
);

// pact items
test_ok!(
    valid_pact_items,
    "pact Foo { fn bar() -> (); }",
    "pact Foo { const BAR: int; }",
    "pact Foo { fn bar() { 0 } }"
);
test_fail!(
    invalid_pact_item,
    "pact Foo { let x = 1; }",
    "pact Foo { struct S; }"
);
test_fail!(pact_const_annotation_required, "pact Foo { const BAR; }");
test_fail!(
    pact_const_with_default_rejected,
    "pact Foo { const BAR: int = 0; }"
);
test_fail!(
    pact_sig_default_param_rejected,
    "pact Foo { fn bar(x: int = 0); }"
);
test_fail!(
    pact_pub_marker_rejected,
    "pact Foo { pub fn bar() -> (); }",
    "pact Foo { pub const BAR: int; }"
);
// a sig's `;` may be left off before the pact's `}`, as before a newline
test_ok!(
    pact_sig_semi_optional_before_brace,
    "pact Foo { fn bar() -> () }",
    "pact Foo { const BAR: int }"
);
test_fail!(
    pact_sig_missing_semi,
    "pact Foo { fn bar() -> () fn baz() -> () }",
    "pact Foo { const BAR: int const BAZ: int; }"
);

// fn body required at top level
test_ok!(fn_with_body_ok, "fn foo() {}", "fn foo() -> int { 0 }");
test_fail!(missing_fn_body_toplevel, "fn foo();", "fn foo() -> int;");

// `#[test]` attributes
test_ok!(
    test_attribute_ok,
    "#[test] fn foo() {}",
    "#[test]\nfn foo() {}",
    "#[test] pub fn foo() {}"
);
test_fail!(
    test_attribute_rejected,
    "#[test] fn foo(x: int) {}",
    "#[test] fn foo(self) {}",
    "#[test] struct Foo {}",
    "#[test] const X = 1;",
    "#[unknown] fn foo() {}",
    "#[test] #[test] fn foo() {}",
    "impl Foo { #[test] fn foo() {} }"
);

// `assert!(expr)` -- parse-time rewrite; empty / extra args are rejected
test_ok!(
    assert_bang_ok,
    "assert!(true);",
    "assert!(1 == 2);",
    "assert!((1 == 2));",
    "assert!(1 in [1]);",
    "assert!(1 !in [2]);",
    "assert!(foo());",
    "assert!(1 == 2,);"
);
test_fail!(
    assert_bang_arity,
    "assert!();",
    "assert!(1, 2);",
    "assert!(1, 2, 3);"
);

// `#[tests] [ expr, ... ]`
test_ok!(
    tests_list_ok,
    "#[tests] [ true ]",
    "#[tests] [ true, false ]",
    "#[tests] [\n    1 == 1,\n    2 == 2,\n]",
    "#[tests] [ true, ]"
);
test_fail!(
    tests_list_rejected,
    "#[tests] []",
    "#[tests] fn foo() {}",
    "#[tests] struct Foo {}",
    "pub #[tests] [ true ]",
    "#[tests] pub [ true ]",
    "#[test] [ true ]",
    "#[tests] #[tests] [ true ]"
);

#[test]
fn tests_list_keeps_expressions() {
    let lexer = crate::lex::Lexer::new("#[tests] [ 1 == 2, true ]", 0, "test".into());
    let ast = crate::Parser::new(lexer).try_into_ast().unwrap();
    assert_eq!(ast.stmts().len(), 1);
    let text = ast.stmts()[0].to_string();
    assert!(text.contains("#[tests]"), "{text}");
    assert!(text.contains("1 == 2"), "{text}");
    assert!(text.contains("true"), "{text}");
    assert!(!text.contains("panic"), "{text}");
}

// `check <expr>` lines -- the literate spelling of a `#[tests]` case
test_ok!(
    checks_ok,
    "check x == 1",
    "check f(2) == 3",
    "check x",
    "check x == 1, y == 2",
    "check !done"
);
test_fail!(
    checks_rejected,
    // the retired list forms -- `:`/`{` error but still recover
    "check:",
    "check: x == 1",
    "check { }",
    "check { x == 1 }",
    "check: { x == 1 }",
    // `is` is gone -- plain `==` is the check spelling now
    "check x is 1",
    // the retired keywords are ordinary identifiers again
    "where: x == 1",
    "example { x }",
    // junk after a check used to spin the parser until its fuel guard panicked
    "check {\n    xs[1..] == [2]\n}",
    "check { a[1..] == b",
    // an unclosed `check {` followed by an item -- deleting the `}` used to
    // loop forever on `fn` (it can't open a check) until the fuel guard
    // panicked; panic-fuzzing found this as `guide.lit` minus one `}` token
    "check {\n    1 == 1\n\nfn f() -> int { 1 }",
    // a `check` alone on a line is a plain identifier, not a block opener --
    // the expr statement runs into a missing-semicolon error, never silently
    // swallowing the next line's expression as a check
    "check\n    f() == 1"
);

fn check_item(source: &str) -> crate::Tests {
    let lexer = crate::lex::Lexer::new(source, 0, "test".into());
    let ast = crate::Parser::new(lexer).try_into_ast().unwrap();
    let stmts = ast.stmts();
    assert_eq!(stmts.len(), 1, "{source} should be one item");
    let crate::StmtKind::Item(item) = stmts[0].kind() else {
        panic!("{source} should be an item")
    };
    let crate::ItemKind::Tests(tests) = item.kind() else {
        panic!("{source} should be a check list")
    };
    tests.clone()
}

#[test]
fn check_line_is_ordinary_equality() {
    // each `check` line is its own `#[tests]` item, named by its expr snippet
    let lexer = crate::lex::Lexer::new(
        "check square(0) == 0\ncheck square(4) == 16\n",
        0,
        "test".into(),
    );
    let ast = crate::Parser::new(lexer).try_into_ast().unwrap();
    assert_eq!(ast.stmts().len(), 2);
    for (i, want) in ["square(0) == 0", "square(4) == 16"].iter().enumerate() {
        let crate::StmtKind::Item(item) = ast.stmts()[i].kind() else {
            panic!("check line should be an item")
        };
        let crate::ItemKind::Tests(tests) = item.kind() else {
            panic!("check line should be a tests item")
        };
        assert_eq!(tests.cases.len(), 1);
        assert_eq!(&tests.cases[0].name.lexeme, want);
        assert!(matches!(
            tests.cases[0].expr.kind(),
            crate::ExprKind::Equality(_)
        ));
    }
    // a bare expr stays a bare check
    let tests = check_item("check alive()");
    assert_eq!(tests.cases[0].name.lexeme, "alive()");
    assert!(!matches!(
        tests.cases[0].expr.kind(),
        crate::ExprKind::Equality(_)
    ));
    // `,` chains a second check on the same line
    let tests = check_item("check f() == 0, g() == 1");
    assert_eq!(tests.cases.len(), 2);
}

#[test]
fn check_opens_before_a_spaced_paren_bracket_or_minus() {
    // juxtaposition would read `check (x)` as the call `check(x)`; a gap
    // after the word opens a check before `(`, `[` and `-` as before a name
    for (src, name) in [
        ("check (a * b).same(c)", "(a * b).same(c)"),
        ("check (f(0)) == 0", "(f(0)) == 0"),
        ("check [1 2] == xs", "[1 2] == xs"),
        ("check -x < 0", "-x < 0"),
    ] {
        let tests = check_item(src);
        assert_eq!(tests.cases.len(), 1, "{src}");
        assert_eq!(&tests.cases[0].name.lexeme, name, "{src}");
    }
}

#[test]
fn check_lines_stand_alone() {
    // no block, no paragraph -- consecutive `check` lines are independent items
    let lexer = crate::lex::Lexer::new(
        "fn f() -> int { 0 }\ncheck f() == 0\ncheck f() == 1\n\nfn g() {}\n",
        0,
        "test".into(),
    );
    let ast = crate::Parser::new(lexer).try_into_ast().unwrap();
    assert_eq!(ast.stmts().len(), 4);
    for i in [1, 2] {
        let crate::StmtKind::Item(item) = ast.stmts()[i].kind() else {
            panic!("check line should be an item")
        };
        assert!(matches!(item.kind(), crate::ItemKind::Tests(_)));
    }
}

#[test]
fn check_word_still_lexes_as_ident() {
    // `check` is only special before a same-line `:`/`{`/expr
    for src in [
        "check.push(x);",
        "let check = 5;",
        "check(x);",
        "check[0];",
        "check-1;",
        "check == 5;",
    ] {
        let lexer = crate::lex::Lexer::new(src, 0, "test".into());
        assert!(crate::Parser::new(lexer).try_into_ast().is_ok(), "{src}");
    }
}

// modules
test_ok!(single_module_ok, "module a;");
test_ok!(nested_module_path_ok, "module a::b;");

// a second `module` decl is caught by `module_name()`, not by `into_ast`
#[test]
fn already_inside_module() {
    let lexer = crate::lex::Lexer::new("module a; module b;", 0, "test".into());
    let ast = crate::Parser::new(lexer).try_into_ast().unwrap();
    assert!(
        ast.module_name().is_err(),
        "second `module` decl should be rejected"
    );
}

#[test]
fn single_module_name_ok() {
    let lexer = crate::lex::Lexer::new("module a;", 0, "test".into());
    let ast = crate::Parser::new(lexer).try_into_ast().unwrap();
    assert!(ast.module_name().is_ok());
}

// use
test_ok!(
    ordinary_use_ok,
    "use a::b;",
    "use foo::{ bar, baz };",
    "use foo::{ bar, baz, };",
    "use foo::*;"
);
test_fail!(glob_all_modules_rejected, "use a::*::*;");

// type annotations
test_ok!(tuple_type_one_element_ok, "let i: (int,) = z;");
test_fail!(
    single_type_parens_rejected,
    "fn f(x: (int)) -> int { 0 }",
    "let i: (int) = z;"
);
test_ok!(
    single_optional_annotation_ok,
    "let x: int? = a;",
    "fn f(x: int?) -> int { 0 }"
);
test_fail!(
    double_hook_annotation_rejected,
    "fn f(x: int??) -> int { 0 }",
    "let x: int?? = a;"
);

// pact bounds in annotations
test_ok!(
    pact_bound_parenthesized_ok,
    "fn f(d: (Foo + Bar)?) -> int { 0 }",
    "fn f(d: Foo + Bar) -> int { 0 }",
    "fn f(d: Foo + Bar + Baz) -> int { 0 }"
);
test_fail!(
    pact_bound_constraint_rejected,
    "fn f(d: Foo + Bar?) -> int { 0 }",
    "fn f(d: Foo? + Bar) -> int { 0 }",
    "fn f(d: Foo! + Bar?) -> int { 0 }"
);
test_fail!(
    non_pact_in_bound_rejected,
    "fn f(d: int + Foo) -> int { 0 }",
    "fn f(d: Foo + str) -> int { 0 }"
);

// reserved-but-unparsed syntax
test_fail!(let_mut_not_parsed, "let mut x = 5;");
test_fail!(
    as_cast_not_parsed,
    "a = 5 as float;",
    "let x: float = 5 as float;"
);
// the word forms of the logical operators are the operators themselves
test_ok!(
    word_operators_parse,
    "a = true and false;",
    "a = true or false;",
    "a = not true;",
    "a = not x and (y or not z);"
);
test_fail!(
    elif_not_parsed,
    "a = if true { 1 } elif false { 2 } else { 3 };"
);
test_fail!(single_quoted_string_rejected, "a = 'foo';");

// `1.2.3` lexes as Float(1.2) then `.3` tuple-index; parser accepts it
// (a solver error, NotATuple, catches it later) so it PARSES
test_ok!(malformed_number_parses_as_tuple_index, "a = 1.2.3;");

// access target permissiveness -- nonsensical receivers parse fine; the solver lints, not the
// parser
test_ok!(
    access_targets,
    "a = b.c.d;",
    "a = b().c;",
    "a = (b).c;",
    "a = 0.a;",
    "a = 0.0.a;",
    r#"a = "foo".a;"#,
    "a = [].a;",
    "a = ~{}.a;",
    "a = 0();",
    "a = 0[0];",
    r#"a = 0["foo"];"#,
    "a = 0.0[0];",
    r#"a = 0.0["foo"];"#,
    "a = [][0];",
    r#"a = []["foo"];"#,
    "a = ~{}[0];",
    r#"a = ~{}["foo"];"#,
    "a = true();",
    "a = true[0];",
    r#"a = true["foo"];"#,
    "a = \"foo\"[0];",
    r#"a = "foo"["foo"];"#,
    r#"a = "hello".len();"#
);
test_ok!(
    call_targets,
    "a.b();",
    "a()();",
    "(a)();",
    "true();",
    "0();",
    "0.0();",
    "\"foo\"();",
    "~{}();",
    "[]();"
);
test_ok!(
    equality_targets,
    "a = b.c == c;",
    "a = b() == c;",
    "a = (b) == c;",
    "a = true == b;"
);
test_ok!(
    evaluation_targets,
    "a = b.c + c;",
    "a = b() + c;",
    "a = (b) + c;",
    "a = true + b;",
    "a = b++ + b;"
);
test_ok!(
    logical_targets,
    "a = b.c && c;",
    "a = b() && c;",
    "a = (b) && c;",
    "a = true && b;"
);
test_ok!(
    unwrap_targets,
    "a = b!;",
    "a = b.c!;",
    "a = b()!;",
    "a = b!.c;",
    "a = b!()();"
);
test_ok!(
    coalescence_targets,
    "a = b ?? c;",
    "a = b.c ?? d;",
    "a = b() ?? c;",
    "a = (b) ?? c;"
);
test_ok!(
    in_targets,
    "a = 0 in b;",
    "a = b in c;",
    r#"a = "x" in "xyz";"#
);
test_fail!(in_does_not_chain, "a = b in c in d;");

// misc valid forms
test_ok!(while_statement_no_semicolon, "while true {}");
test_ok!(empty_tuple_struct, "struct Foo();");
test_ok!(struct_literal_in_grouping, "a = (Foo {});");
test_ok!(
    struct_literal_in_if_body,
    "a = if true { Foo {} } else { Foo {} };"
);
test_ok!(
    struct_literal_inside_a_condition_bracket,
    "if foo(Bar { a = 1 }) { c(); }",
    "if a == (if b { 1 } else { 2 }) { c(); }",
    "if a[Bar { x = 1 }] { c(); }",
    "while foo(Bar { a = 1 }) { c(); }",
    "for i in foo(Bar { a = 1 }) { c(); }"
);
test_ok!(pattern_match_or, "match x { 0 | 1 | 2 => 1, _ => 2 }");
test_ok!(match_arm_with_guard, "match x { 0 if true => 1, _ => 2 }");
test_ok!(struct_literal_trailing_comma, "a = Foo { a = 1, };");
test_ok!(
    fstring_valid,
    r#"a = f"hello";"#,
    r#"a = f"hello {name}";"#,
    r#"a = f"{1 + 2}";"#,
    r#"a = f"{foo("multi
     line")}";"#
);
test_ok!(
    valueless_break_before_statement,
    "loop {
         break
         let x = 1;
     }"
);
test_ok!(
    closure_with_return_type,
    "a = |a: int| -> int { a };",
    "a = || -> int { 1 };"
);
test_fail!(closure_return_type_after_body, "a = || { 1 } -> int;");
test_ok!(
    pact_default_body_no_semi,
    "pact Foo { fn bar() { 0 } }",
    "pact Foo { fn a() { 0 } fn b(); fn c() -> int { 1 } }"
);

test_ok!(
    long_else_if_ladder_parses,
    format!(
        "fn f(x) {{ if x == 0 {{ 0 }}{} else {{ -1 }} }}",
        (1..80)
            .map(|i| format!(" else if x == {i} {{ {i} }}"))
            .collect::<String>()
    )
);
test_ok!(
    deeply_nested_blocks_parse,
    format!("fn main() {{ {}{} }}", "if a { ".repeat(40), "}".repeat(40))
);
test_fail!(
    deep_nesting_does_not_overflow_the_stack,
    format!(
        "fn main() {{ let x = {}1{}; }}",
        "(".repeat(4000),
        ")".repeat(4000)
    )
);

// `T??` lexes as DoubleHook; the annotation loop turns it into a targeted diagnostic
test_fail!(doubled_option_annotation, "let a: int?? = 5;");

// misdirection: common habits from other languages get a teaching error
test_fail!(misdirect_elif, "if x == 1 {} elif x == 2 {}");
test_fail!(misdirect_as, "let y = x as float;");
test_fail!(misdirect_if_assign, "if x = 5 {}");

// generics: decls take params, annotations take applications, impls take neither
test_ok!(
    generic_declarations_parse,
    "fn id<T>(x: T) -> T { x }",
    "fn map<T, U>(xs: [T], f: (T) -> U) -> [U] { [] }",
    "struct Pair<A, B> { first: A, second: B }",
    "struct Point<T>(T, T)",
    "enum List<T> { E, C { first: T, rest: List<T> } }",
    "let p: Pair<int, str> = x;",
    "let l: List<int> = y;",
    "impl List { fn f(self) {} }"
);
test_fail!(
    impl_specialization_rejected,
    "impl List<T> {}",
    "impl Pact for List<T> {}"
);

#[test]
fn misdirection_renders_spelling() {
    let (_, errors) = crate::tests::utils::parse("let y = x as float;");
    let rendered = format!("{:?}", errors[0]);
    assert!(
        rendered.contains("to_float"),
        "the `as` hint should teach a conversion method: {rendered}"
    );
}

// `where` predicates on fns + fixed-size array annotations
test_ok!(
    where_clauses_parse,
    "fn f(xs: [int]) where xs.len() > 2 { xs }",
    "fn f(xs: [int], n: int) where xs.len() > n, n > 0 { xs[0] }",
    // `where` stays an ordinary identifier everywhere else
    "let where = 5;",
    "let b = where_fn(a);",
    // impl methods take them too
    "impl P { fn f(self, xs: [int]) where xs.len() > 0 {} }"
);
test_ok!(
    sized_array_annotations_parse,
    "let a: [int; 4] = x;",
    "let m: [f32; 4, 8] = x;",
    "fn f(m: [f32; 4, 8]) -> [f32; 4] { m[0] }",
    "let growable: [int] = y;"
);
test_fail!(
    malformed_where_clauses,
    "fn f(xs: [int]) where { xs }",
    "fn f(xs: [int]) where, { xs }"
);
test_fail!(
    malformed_sized_arrays,
    "let a: [int;] = x;",
    "let a: [int; x] = x;",
    "let a: [int; 4,] = x;"
);

#[test]
fn a_declared_fn_takes_back_an_operator_word() {
    // nand2tetris writes its own `not`/`and`/`or` gates: declaring the fn
    // makes the word a name on that source -- calls, passing it as a value
    let src = "fn nand(a: int, b: int) -> int { 1 - a * b }\n\
               fn not(a: int) -> int { nand(a, a) }\n\
               fn and(a: int, b: int) -> int { not(nand(a, b)) }\n\
               fn or(a: int, b: int) -> int { nand(not(a), not(b)) }\n\
               let f = not;\n\
               let ok = true && !false;\n";
    let lexer = crate::lex::Lexer::new(src, 0, "test".into());
    assert!(crate::Parser::new(lexer).try_into_ast().is_ok());
    // only the declared words: `or` stays the operator here
    assert_eq!(crate::lex::shadowed_words("fn not(a: int) -> int { a }"), [false, false, true]);
    assert_eq!(crate::lex::shadowed_words("pub fn and(a: int) -> int { a }"), [true, false, false]);
    // a mention in a comment, a longer name, or `fn` inside a word doesn't count
    assert_eq!(crate::lex::shadowed_words("// fn not(x)\nfn notify() {}\nlet defn not = 1"), [false; 3]);
}

#[test]
fn a_spliced_source_shadows_per_page() {
    // the host's splice marks each page; a page's `fn not` reaches the
    // pages that `use` it (downstream), never the pages it uses
    let src = "//# page: kit\nfn ok() -> bool { not false }\n\n\
               //# page: n2t-lib\nfn not(a: int) -> int { 1 - a }\n\n\
               //# page: gates\nuse \"n2t-lib\";\nuse \"kit\";\nlet x = not(1);\n";
    let lexer = crate::lex::Lexer::new(src, 0, "test".into());
    assert!(crate::Parser::new(lexer).try_into_ast().is_ok());
    let t = crate::lex::shadow_table(src);
    let at = |needle: &str| crate::lex::shadow_at(&t, src.find(needle).unwrap())[2];
    assert!(!at("not false"), "kit keeps the operator");
    assert!(at("fn not"), "the declaring page");
    assert!(at("not(1)"), "a page that uses it");
}

#[test]
fn opp_is_an_enum_and_its_flip() {
    // `opp Sky { Sun, Moon }` parses to the enum plus `impl Opp for
    // Sky`, both at the declaration
    let lexer = crate::lex::Lexer::new("opp Sky { Sun, Moon }\nlet x = 1;", 0, "test".into());
    let ast = crate::Parser::new(lexer).try_into_ast().unwrap();
    let kinds: Vec<String> = ast.stmts().iter().map(|s| s.to_string()).collect();
    assert_eq!(kinds.len(), 3, "{kinds:?}");
    assert!(kinds[0].starts_with("enum Sky"), "{kinds:?}");
    assert!(kinds[1].starts_with("impl Opp for Sky"), "{kinds:?}");
    // the word stays a name elsewhere
    for src in ["let opp = 1;", "opp(2);", "opp.x;", "opp == 3;"] {
        let lexer = crate::lex::Lexer::new(src, 0, "test".into());
        assert!(crate::Parser::new(lexer).try_into_ast().is_ok(), "{src}");
    }
    // exactly two plain ends
    for src in ["opp Tri { A, B, C }", "opp One { A }", "opp P { A(int), B }"] {
        let lexer = crate::lex::Lexer::new(src, 0, "test".into());
        assert!(crate::Parser::new(lexer).try_into_ast().is_err(), "{src}");
    }
}
