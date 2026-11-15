use crate::{
    IntoExpr, IntoStmt, block, components::*, expr::*, ident, ident_expr, int, item::*, lex::TyKw,
    stmt::*,
};
use pretty_assertions::assert_eq;
use shared::Location;

macro_rules! expr_test {
    ($name:ident, $source:expr, $expected:expr) => {
        #[cfg(test)]
        #[test]
        fn $name() {
            let expected: crate::expr::Expr = crate::expr::ExprKind::from($expected).into_expr();
            let lexer = crate::lex::Lexer::new($source, 0, "test".into());
            let mut parser = crate::Parser::new(lexer);
            let output = parser.expr();
            assert!(parser.errors().is_empty(), "{:?}", parser.errors());
            if output.kind() != expected.kind() {
                assert_eq!(output.to_string(), expected.to_string());
                assert_eq!(output.kind(), expected.kind());
                panic!();
            }
        }
    };
}

expr_test!(r#loop, "loop {}", Loop::new(block!()));

expr_test!(
    loop_no_block,
    "loop break",
    Loop::new(Break::new(None).into_expr())
);

expr_test!(return_expr_no_value, "return", Return::new(None));

expr_test!(
    return_expr_with_value,
    "return 0",
    Return::new(Some(int!(0)))
);

expr_test!(break_expr_no_value, "break", Break::new(None));

expr_test!(break_expr_with_value, "break 0", Break::new(Some(int!(0))));

expr_test!(collect_expr, "collect 0", Collect::new(int!(0)));

expr_test!(
    r#while,
    "while true {}",
    While::new(Literal::True.into_expr(), block!())
);

expr_test!(
    while_no_block,
    "while true break;",
    While::new(Literal::True.into_expr(), Break::new(None).into_expr())
);

expr_test!(
    while_let,
    "while let a = foo {}",
    While::new_with_pat(
        Pat::new(PatKind::Ident(ident!("a")), shared::Location::default(),),
        ident_expr!("foo"),
        block!()
    )
);

expr_test!(
    while_let_tuple,
    "while let (a, b) = foo {}",
    While::new_with_pat(
        Pat::new(
            PatKind::Tuple(vec![ident!("a").into(), ident!("b").into()]),
            shared::Location::default(),
        ),
        ident_expr!("foo"),
        block!()
    )
);

expr_test!(
    for_in_no_block,
    "for a in b if a a else null",
    For::new(
        ident!("a"),
        ident_expr!("b"),
        If::new_with_else(
            ident_expr!("a"),
            ident_expr!("a"),
            Literal::Null.into_expr()
        )
        .into_expr()
    )
);

expr_test!(
    for_in_with_block,
    "for a in b {}",
    For::new(ident!("a"), ident_expr!("b"), block!())
);

expr_test!(
    collection_no_block,
    "for a in b collect a",
    For::new(
        ident!("a"),
        ident_expr!("b"),
        Collect::new(ident_expr!("a")).into_expr()
    )
);

expr_test!(
    collection_with_block,
    "for a in b { collect a; }",
    For::new(
        ident!("a"),
        ident_expr!("b"),
        block!(Collect::new(ident_expr!("a")).into_expr().into_stmt())
    )
);

expr_test!(
    for_tuple_binding,
    "for (a, b) in z {}",
    For::new(
        Pat::new(
            PatKind::Tuple(vec![ident!("a").into(), ident!("b").into()]),
            shared::Location::default(),
        ),
        ident_expr!("z"),
        block!()
    )
);

expr_test!(
    for_range,
    "for a in 0..1 {}",
    For::new(
        ident!("a"),
        Range::new(
            Literal::Int(0).into_expr(),
            Literal::Int(1).into_expr(),
            false
        )
        .into_expr(),
        block!()
    )
);

expr_test!(
    for_range_inclusive,
    "for a in 0..=1 {}",
    For::new(
        ident!("a"),
        Range::new(
            Literal::Int(0).into_expr(),
            Literal::Int(1).into_expr(),
            true
        )
        .into_expr(),
        block!()
    )
);

// r[verify mimas.array.comprehension]
expr_test!(
    list_comprehension,
    "[a for x in xs]",
    For::new(
        ident!("x"),
        ident_expr!("xs"),
        Collect::new(ident_expr!("a")).into_expr()
    )
);

expr_test!(
    list_comprehension_filter,
    "[a for x in xs if p]",
    For::new(
        ident!("x"),
        ident_expr!("xs"),
        If::new(ident_expr!("p"), Collect::new(ident_expr!("a")).into_expr()).into_expr()
    )
);

expr_test!(r#if, "if a {}", If::new(ident_expr!("a"), block!()));

expr_test!(
    if_else,
    "if a {} else {}",
    If::new_with_else(ident_expr!("a"), block!(), block!())
);

expr_test!(
    if_no_block,
    "if a b else c",
    If::new_with_else(ident_expr!("a"), ident_expr!("b"), ident_expr!("c"))
);

expr_test!(
    if_let,
    "if let a = b {}",
    If {
        condition: ident_expr!("b"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(PatKind::Ident(ident!("a")), Location::SYNTHETIC)),
    }
);

expr_test!(
    if_let_tuple,
    "if let (a, b) = c {}",
    If {
        condition: ident_expr!("c"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Tuple(vec![ident!("a").into(), ident!("b").into()]),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_literal_int,
    "if let 5 = n {}",
    If {
        condition: ident_expr!("n"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Literal(Literal::Int(5)),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_literal_bool,
    "if let true = b {}",
    If {
        condition: ident_expr!("b"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Literal(Literal::True),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_literal_str,
    r#"if let "hi" = s {}"#,
    If {
        condition: ident_expr!("s"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Literal(Literal::String("hi".into())),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_null_bind,
    "if let a? = b {}",
    If {
        condition: ident_expr!("b"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::NullBind(Box::new(ident!("a").into())),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_null_bind_str,
    r#"if let "hi"? = s {}"#,
    If {
        condition: ident_expr!("s"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::NullBind(Box::new(Pat::new(
                PatKind::Literal(Literal::String("hi".into())),
                Location::default(),
            ))),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_null_bind_int,
    "if let 42? = n {}",
    If {
        condition: ident_expr!("n"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::NullBind(Box::new(Pat::new(
                PatKind::Literal(Literal::Int(42)),
                Location::default(),
            ))),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_null_bind_false,
    "if let false? = b {}",
    If {
        condition: ident_expr!("b"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::NullBind(Box::new(Pat::new(
                PatKind::Literal(Literal::False),
                Location::default(),
            ))),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_null_bind_struct,
    "if let Pair { a }? = p {}",
    If {
        condition: ident_expr!("p"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::NullBind(Box::new(Pat::new(
                PatKind::Struct(Box::new(ident_expr!("Pair")), {
                    let mut m = indexmap::IndexMap::new();
                    m.insert("a".into(), Pat::from(ident!("a")));
                    m
                }),
                Location::default(),
            ))),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_or,
    "if let 0 | 1 = n {}",
    If {
        condition: ident_expr!("n"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Or(vec![
                Pat::new(PatKind::Literal(Literal::Int(0)), Location::default()),
                Pat::new(PatKind::Literal(Literal::Int(1)), Location::default()),
            ]),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_struct,
    "if let Pair { a, b } = p {}",
    If {
        condition: ident_expr!("p"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Struct(Box::new(ident_expr!("Pair")), {
                let mut m = indexmap::IndexMap::new();
                m.insert("a".into(), Pat::from(ident!("a")));
                m.insert("b".into(), Pat::from(ident!("b")));
                m
            }),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_tuple_struct,
    "if let Wrap(x) = w {}",
    If {
        condition: ident_expr!("w"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::TupleVariant(Box::new(ident_expr!("Wrap")), vec![ident!("x").into()]),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_enum_unit_variant,
    "if let Color::Red = c {}",
    If {
        condition: ident_expr!("c"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Variant(Box::new(
                Access::DoubleColon {
                    left: ident_expr!("Color"),
                    right: ident!("Red"),
                }
                .into_expr(),
            )),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_enum_tuple_variant,
    "if let Shape::Circle(r) = s {}",
    If {
        condition: ident_expr!("s"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::TupleVariant(
                Box::new(
                    Access::DoubleColon {
                        left: ident_expr!("Shape"),
                        right: ident!("Circle"),
                    }
                    .into_expr(),
                ),
                vec![ident!("r").into()],
            ),
            Location::default(),
        )),
    }
);

expr_test!(
    if_let_enum_struct_variant,
    "if let Msg::Move { x, y } = m {}",
    If {
        condition: ident_expr!("m"),
        main_body: block!(),
        else_expr: None,
        binding: Some(Pat::new(
            PatKind::Struct(
                Box::new(
                    Access::DoubleColon {
                        left: ident_expr!("Msg"),
                        right: ident!("Move"),
                    }
                    .into_expr(),
                ),
                {
                    let mut m = indexmap::IndexMap::new();
                    m.insert("x".into(), Pat::from(ident!("x")));
                    m.insert("y".into(), Pat::from(ident!("y")));
                    m
                },
            ),
            Location::default(),
        )),
    }
);

expr_test!(
    r#match,
    "match foo {}",
    Match::new(ident_expr!("foo"), vec![], false)
);

expr_test!(
    match_with_one_case,
    "match foo {
        0 => {}
    }",
    Match::new(
        ident_expr!("foo"),
        vec![MatchCase::new(
            crate::components::Pat::new(
                crate::components::PatKind::Literal(crate::expr::Literal::Int(0)),
                shared::Location::default()
            ),
            None,
            block!()
        )],
        false
    )
);

expr_test!(
    match_with_multiple_cases,
    "match foo {
        0 => {},
        1 => {},
    }",
    Match::new(
        ident_expr!("foo"),
        vec![
            MatchCase::new(
                crate::components::Pat::new(
                    crate::components::PatKind::Literal(crate::expr::Literal::Int(0)),
                    shared::Location::default()
                ),
                None,
                block!()
            ),
            MatchCase::new(
                crate::components::Pat::new(
                    crate::components::PatKind::Literal(crate::expr::Literal::Int(1)),
                    shared::Location::default()
                ),
                None,
                block!()
            ),
        ],
        false,
    )
);

expr_test!(
    match_with_fallback,
    "match foo {
        _ => {},
    }",
    Match::new(
        ident_expr!("foo"),
        vec![MatchCase::new(
            crate::components::Pat::from(crate::expr::Ident::synthetic("_")),
            None,
            block!()
        )],
        false,
    )
);

expr_test!(block, "{}", Block::new(vec![]));

expr_test!(
    block_yield_value,
    "{ 0 }",
    Block::new_with_yield(vec![], int!(0))
);

expr_test!(
    multi_statement_block_yield,
    "{
        let a = 0;
        let b = 0;
        a + b   
    }",
    Block::new_with_yield(
        vec![
            Let::new(
                Pat::new(PatKind::Ident(ident!("a")), shared::Location::default()),
                int!(0),
                None
            )
            .into_stmt(),
            Let::new(
                Pat::new(PatKind::Ident(ident!("b")), shared::Location::default()),
                int!(0),
                None
            )
            .into_stmt(),
        ],
        Evaluation::new(ident_expr!("a"), EvaluationOp::Plus, ident_expr!("b")).into_expr()
    )
);

expr_test!(
    semicolon_consequences_block,
    "{
        c = a
        -b
    }",
    Block::new_with_yield(
        vec![
            Assignment::new(ident_expr!("c"), AssignmentOp::Identity, ident_expr!("a")).into_stmt()
        ],
        Unary::new(UnaryOp::Negative, ident_expr!("b")).into_expr(),
    )
);

expr_test!(
    infix_continues_inside_parens,
    "(1
        + 2
        + 3)",
    Grouping::new(
        Evaluation::new(
            Evaluation::new(int!(1), EvaluationOp::Plus, int!(2)).into_expr(),
            EvaluationOp::Plus,
            int!(3),
        )
        .into_expr(),
    )
);

expr_test!(
    infix_continues_when_operator_at_end_of_line,
    "1 +
        2",
    Evaluation::new(int!(1), EvaluationOp::Plus, int!(2))
);

expr_test!(
    block_in_block,
    "{{0}}",
    Block::new_with_yield(vec![], Block::new_with_yield(vec![], int!(0)).into_expr())
);

expr_test!(
    record_block_mix,
    "{{foo: {}}}",
    Block::new_with_yield(
        vec![],
        Literal::Record(vec![(ident!("foo"), block!())]).into_expr()
    )
);

expr_test!(
    and,
    "1 && 1",
    Logical::new(int!(1), LogicalOp::And, int!(1))
);

expr_test!(or, "1 || 1", Logical::new(int!(1), LogicalOp::Or, int!(1)));

expr_test!(
    addition,
    "1 + 1",
    Evaluation::new(int!(1), EvaluationOp::Plus, int!(1))
);

expr_test!(
    subtraction,
    "1 - 1",
    Evaluation::new(int!(1), EvaluationOp::Minus, int!(1))
);

expr_test!(
    multiplication,
    "1 * 1",
    Evaluation::new(int!(1), EvaluationOp::Multiply, int!(1))
);

expr_test!(
    division,
    "1 / 1",
    Evaluation::new(int!(1), EvaluationOp::Divide, int!(1))
);

expr_test!(
    modulo,
    "1 mod 1",
    Evaluation::new(int!(1), EvaluationOp::Modulo, int!(1))
);

expr_test!(
    div,
    "1 ~/ 1",
    Evaluation::new(int!(1), EvaluationOp::Div, int!(1))
);

expr_test!(
    bitwise_and,
    "1 & 1",
    Evaluation::new(int!(1), EvaluationOp::And, int!(1))
);

expr_test!(
    bitwise_or,
    "1 | 1",
    Evaluation::new(int!(1), EvaluationOp::Or, int!(1))
);

expr_test!(
    bitwise_chain,
    "1 | 1 | 1",
    Evaluation::new(
        Evaluation::new(int!(1), EvaluationOp::Or, int!(1)).into_expr(),
        EvaluationOp::Or,
        int!(1),
    )
);

expr_test!(
    bitwise_xor,
    "1 ⊕ 1",
    Evaluation::new(int!(1), EvaluationOp::Xor, int!(1))
);

// `⊻` is the same operator — lexed to the same token.
expr_test!(
    bitwise_xor_alt_glyph,
    "1 ⊻ 1",
    Evaluation::new(int!(1), EvaluationOp::Xor, int!(1))
);

// `xor` is the word spelling — contextual like `mod`, so `let xor = …`
// still binds an ordinary name.
expr_test!(
    xor_word,
    "a xor b",
    Evaluation::new(ident_expr!("a"), EvaluationOp::Xor, ident_expr!("b"))
);

// `^` joined identifier continuation: `x^2` is one name (the editor
// typesets it raised), not `x` xor `2`.
expr_test!(
    caret_joins_ident,
    "x^2 + 1",
    Evaluation::new(ident_expr!("x^2"), EvaluationOp::Plus, int!(1),)
);

// The retired spellings report a misdirection and parse on as xor.
#[test]
fn spaced_caret_is_old_xor() {
    let lexer = crate::lex::Lexer::new("a ^ b", 0, "test".into());
    let mut parser = crate::Parser::new(lexer);
    let expr = parser.expr();
    assert_eq!(expr.to_string(), "a ⊕ b");
    assert_eq!(parser.errors().len(), 1);
}

// `lit ^ lit` reads as a power — both sides are const, so the fold
// happens at desugar and no error fires.
expr_test!(caret_folds_lit_pow, "2^10", Literal::Int(1024));
expr_test!(
    caret_folds_float_pow,
    "2.0 ^ 0.5",
    Literal::Float(std::f64::consts::SQRT_2)
);
expr_test!(caret_folds_neg_exp, "2^-1", Literal::Float(0.5));
expr_test!(
    caret_folds_overflow_to_float,
    "2^63",
    Literal::Float(9.223372036854776e18)
);
// `-2^2` reads as `-(2^2)` per math convention; the parenthesized
// `(-2)^2` squares the negative.
expr_test!(caret_neg_base_hoists, "-2^2", Literal::Int(-4));
expr_test!(caret_paren_neg_base, "(-2)^2", Literal::Int(4));
expr_test!(caret_paren_neg_cube, "(-2)^3", Literal::Int(-8));

// With a non-literal side the `^` still reads as the retired xor — the
// diagnostic points at `.pow`, and recovery parses on as xor.
#[test]
fn caret_after_number_suggests_pow() {
    let lexer = crate::lex::Lexer::new("2 ^ x", 0, "test".into());
    let mut parser = crate::Parser::new(lexer);
    let expr = parser.expr();
    assert_eq!(expr.to_string(), "2 ⊕ x");
    assert_eq!(parser.errors().len(), 1);
    assert!(parser.errors()[0].to_string().contains("pow"));
}

expr_test!(
    dot_access_bitwise,
    "foo.bar | foo.bar",
    Evaluation::new(
        Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        EvaluationOp::Or,
        Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
    )
);

expr_test!(
    combo_math,
    "1 * 1 + 1 >> 1 & 1 == 1",
    Equality::new(
        Evaluation::new(
            Evaluation::new(
                Evaluation::new(
                    Evaluation::new(int!(1), EvaluationOp::Multiply, int!(1)).into_expr(),
                    EvaluationOp::Plus,
                    int!(1),
                )
                .into_expr(),
                EvaluationOp::BitShiftRight,
                int!(1),
            )
            .into_expr(),
            EvaluationOp::And,
            int!(1),
        )
        .into_expr(),
        EqualityOp::Equal,
        int!(1),
    )
);

expr_test!(
    math_on_block,
    "1 / { 1 }",
    Evaluation::new(
        int!(1),
        EvaluationOp::Divide,
        Block::new_with_yield(vec![], int!(1)).into_expr()
    )
);

expr_test!(
    bit_shift_left,
    "1 << 1",
    Evaluation::new(int!(1), EvaluationOp::BitShiftLeft, int!(1))
);

expr_test!(
    bit_shift_right,
    "1 >> 1",
    Evaluation::new(int!(1), EvaluationOp::BitShiftRight, int!(1))
);

expr_test!(
    less_than,
    "1 < 1",
    Equality::new(int!(1), EqualityOp::Less, int!(1))
);

expr_test!(
    less_than_or_equal,
    "1 <= 1",
    Equality::new(int!(1), EqualityOp::LessOrEqual, int!(1),)
);

expr_test!(
    greater_than,
    "1 > 1",
    Equality::new(int!(1), EqualityOp::Greater, int!(1),)
);

expr_test!(
    greater_than_or_equal,
    "1 >= 1",
    Equality::new(int!(1), EqualityOp::GreaterOrEqual, int!(1),)
);

expr_test!(
    equal,
    "1 == 1",
    Equality::new(int!(1), EqualityOp::Equal, int!(1))
);

expr_test!(
    bang_equal,
    "1 != 1",
    Equality::new(int!(1), EqualityOp::NotEqual, int!(1))
);

expr_test!(
    null_coalecence,
    "foo ?? 1",
    Coalescence::new(ident_expr!("foo"), int!(1))
);

expr_test!(not, "!foo", Unary::new(UnaryOp::Not, ident_expr!("foo")));

expr_test!(positive, "+1", Unary::new(UnaryOp::Positive, int!(1)));

expr_test!(negative, "-1", Unary::new(UnaryOp::Negative, int!(1)));

expr_test!(
    dot_unary,
    "!foo.bar",
    Unary::new(
        UnaryOp::Not,
        Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct
        }
        .into_expr(),
    )
);

expr_test!(
    ds_unary,
    "!foo[bar]",
    Unary::new(
        UnaryOp::Not,
        Access::Square {
            left: ident_expr!("foo"),
            key: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
    )
);

expr_test!(bitwise_not, "~1", Unary::new(UnaryOp::BitwiseNot, int!(1)));

expr_test!(
    fstring_literal_only,
    "f\"hello\"",
    FString::new(vec![FStringPart::Literal("hello".to_string())])
);

expr_test!(
    fstring_with_expr,
    "f\"{a}\"",
    FString::new(vec![FStringPart::Expr(ident_expr!("a"))])
);

expr_test!(
    fstring_mixed,
    "f\"foo {a} bar\"",
    FString::new(vec![
        FStringPart::Literal("foo ".to_string()),
        FStringPart::Expr(ident_expr!("a")),
        FStringPart::Literal(" bar".to_string()),
    ])
);

// r[verify mimas.str.tagged]
// `tag"…"` is sugar for `__lit_tag("…")` -- the host registers one prelude
// native per tag, and the literal is checked and typed through it.
expr_test!(
    tagged_literal_lowers_to_call,
    "pat\"bd*2 <cp hh>\"",
    Call::new(
        ident_expr!("__lit_pat"),
        vec![Argument::new(
            Literal::String("bd*2 <cp hh>".to_string()).into_expr()
        )]
    )
);

expr_test!(
    tagged_literal_unescapes_like_a_string,
    "keys\"a \\\"b\\\"\"",
    Call::new(
        ident_expr!("__lit_keys"),
        vec![Argument::new(
            Literal::String("a \"b\"".to_string()).into_expr()
        )]
    )
);

// a method chain hangs off the literal like off any call
expr_test!(
    tagged_literal_chains,
    "pat\"bd\".slow(2)",
    Call::new(
        Access::Dot {
            left: Call::new(
                ident_expr!("__lit_pat"),
                vec![Argument::new(Literal::String("bd".to_string()).into_expr())]
            )
            .into_expr(),
            right: ident_expr!("slow"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![Argument::new(int!(2))]
    )
);

// `{{` and `}}` are escape sequences for literal braces (python/rust convention).
expr_test!(
    fstring_escaped_braces,
    "f\"{{x}}\"",
    FString::new(vec![FStringPart::Literal("{x}".to_string())])
);

expr_test!(
    fstring_braces_around_interp,
    "f\"{{{a}}}\"",
    FString::new(vec![
        FStringPart::Literal("{".to_string()),
        FStringPart::Expr(ident_expr!("a")),
        FStringPart::Literal("}".to_string()),
    ])
);

// nested `"..."` inside an interp expression must not terminate the outer f-string
expr_test!(
    fstring_interp_with_string_index,
    "f\"{m[\"k\"]}\"",
    FString::new(vec![FStringPart::Expr(
        Access::Square {
            left: ident_expr!("m"),
            key: ExprKind::Literal(Literal::String("k".into())).into_expr(),
            kind: AccessKind::Direct,
        }
        .into_expr(),
    )])
);

// a `}` inside a nested string in the interp must not close the interp early
expr_test!(
    fstring_interp_with_brace_in_string,
    "f\"{[\"}\"]}\"",
    FString::new(vec![FStringPart::Expr(
        ExprKind::Literal(Literal::Array(vec![
            ExprKind::Literal(Literal::String("}".into())).into_expr(),
        ]))
        .into_expr(),
    )])
);

expr_test!(
    dot_access_call,
    "a.b()",
    Call::new(
        Access::Dot {
            left: ident_expr!("a"),
            right: ident_expr!("b"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![]
    )
);

expr_test!(
    struct_literal,
    "Foo {}",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![],
        base: None,
    })
);

expr_test!(
    struct_literal_with_fields,
    "Foo { a = 0 }",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![(FieldKey::Ident(ident!("a")), int!(0))],
        base: None,
    })
);

expr_test!(
    struct_literal_inferred,
    "Foo { a }",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![(FieldKey::Ident(ident!("a")), ident!("a").into_expr())],
        base: None,
    })
);

expr_test!(
    struct_literal_inferred_trail,
    "Foo { a, }",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![(FieldKey::Ident(ident!("a")), ident!("a").into_expr())],
        base: None,
    })
);

expr_test!(
    struct_literal_inferred_multi,
    "Foo { a, b }",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![
            (FieldKey::Ident(ident!("a")), ident!("a").into_expr()),
            (FieldKey::Ident(ident!("b")), ident!("b").into_expr()),
        ],
        base: None,
    })
);

expr_test!(
    struct_literal_inferred_mixed,
    "Foo { a, b = 0 }",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![
            (FieldKey::Ident(ident!("a")), ident!("a").into_expr()),
            (FieldKey::Ident(ident!("b")), int!(0)),
        ],
        base: None,
    })
);

expr_test!(
    enum_literal,
    "Foo::Bar {}",
    Literal::Struct(StructLiteral {
        name: Access::DoubleColon {
            left: ident_expr!("Foo"),
            right: ident!("Bar")
        }
        .into_expr(),
        fields: vec![],
        base: None,
    })
);

expr_test!(
    enum_literal_with_fields,
    "Foo::Bar { a = 0 }",
    Literal::Struct(StructLiteral {
        name: Access::DoubleColon {
            left: ident_expr!("Foo"),
            right: ident!("Bar")
        }
        .into_expr(),
        fields: vec![(FieldKey::Ident(ident!("a")), int!(0))],
        base: None,
    })
);

expr_test!(closure, "|| {}", Closure::new(vec![], block!()));

expr_test!(
    closure_without_parameters_with_return_type,
    "|| -> int { 1 }",
    Closure {
        parameters: vec![],
        body: Block::new_with_yield(vec![], int!(1)).into_expr(),
        return_type: Some(Annotation::Kw(TyKw::Int)),
    }
);

expr_test!(
    closure_with_parameters,
    "|a: int, b| {}",
    Closure::new(
        vec![
            Binding::new(ident!("a")).with_annotation(Annotation::Kw(TyKw::Int)),
            Binding::new(ident!("b")),
        ],
        block!()
    )
);

expr_test!(
    call_on_array,
    "a[0]()",
    Call::new(
        Access::Square {
            left: ident_expr!("a"),
            key: int!(0),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![]
    )
);

expr_test!(call, "foo()", Call::new(ident_expr!("foo"), vec![]));

expr_test!(
    call_with_args,
    "foo(0, 1, 2)",
    Call::new(
        ident_expr!("foo"),
        vec![
            Argument::new(int!(0)),
            Argument::new(int!(1)),
            Argument::new(int!(2)),
        ]
    )
);

expr_test!(
    call_with_named_arg,
    "foo(a=5)",
    Call::new(
        ident_expr!("foo"),
        vec![Argument::named(ident!("a"), int!(5))]
    )
);

expr_test!(
    call_trailing_commas,
    "foo(0, 1, 2)",
    Call::new(
        ident_expr!("foo"),
        vec![
            Argument::new(int!(0)),
            Argument::new(int!(1)),
            Argument::new(int!(2)),
        ]
    )
);

expr_test!(
    call_identity,
    "self.bar()",
    Call::new(
        Access::Dot {
            left: Ident::new("self", Default::default()).into_expr(),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(empty_array, "[]", ExprKind::Literal(Literal::Array(vec![])));

expr_test!(
    simple_array,
    "[0, 1, 2]",
    ExprKind::Literal(Literal::Array(vec![int!(0), int!(1), int!(2),]))
);

expr_test!(
    array_comma_trail,
    "[0, 1, 2,]",
    ExprKind::Literal(Literal::Array(vec![int!(0), int!(1), int!(2),]))
);

expr_test!(
    simple_tuple,
    "(0,)",
    ExprKind::Literal(Literal::Tuple(vec![int!(0)]))
);

expr_test!(
    multi_tuple,
    "(0, 1)",
    ExprKind::Literal(Literal::Tuple(vec![int!(0), int!(1)]))
);

expr_test!(
    tuple_comma_trail,
    "(0, 1,)",
    ExprKind::Literal(Literal::Tuple(vec![int!(0), int!(1)]))
);

expr_test!(
    dict_record,
    "{ foo: bar, fizz: buzz }",
    ExprKind::Literal(Literal::Record(vec![
        (ident!("foo"), ident_expr!("bar")),
        (ident!("fizz"), ident_expr!("buzz")),
    ]))
);

expr_test!(
    dict_record_comma_trail,
    "{ foo: bar, fizz: buzz, }",
    ExprKind::Literal(Literal::Record(vec![
        (ident!("foo"), ident_expr!("bar")),
        (ident!("fizz"), ident_expr!("buzz")),
    ]))
);

expr_test!(
    square_access,
    "foo[bar]",
    Access::Square {
        left: ident_expr!("foo"),
        key: ident_expr!("bar"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    chained_square_accesses,
    "foo[bar][buzz]",
    Access::Square {
        left: Access::Square {
            left: ident_expr!("foo"),
            key: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        key: ident_expr!("buzz"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    option_square_accesses,
    "foo?[buzz]",
    Access::Square {
        left: ident_expr!("foo"),
        key: ident_expr!("buzz"),
        kind: AccessKind::Option,
    }
);

expr_test!(
    square_access_call,
    "foo[bar]()",
    Call::new(
        Access::Square {
            left: ident_expr!("foo"),
            key: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(
    option_square_access_call,
    "foo?[bar]()",
    Call::new(
        Access::Square {
            left: ident_expr!("foo"),
            key: ident_expr!("bar"),
            kind: AccessKind::Option,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(
    identity_access,
    "self.bar",
    Access::Dot {
        left: Ident::new("self", Default::default()).into_expr(),
        right: ident_expr!("bar"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    direct_dot_access,
    "foo.bar",
    Access::Dot {
        left: ident_expr!("foo"),
        right: ident_expr!("bar"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    tuple_dot_access,
    "foo.1",
    Access::Dot {
        left: ident_expr!("foo"),
        right: int!(1),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    option_dot_access,
    "foo?.bar",
    Access::Dot {
        left: ident_expr!("foo"),
        right: ident_expr!("bar"),
        kind: AccessKind::Option,
    }
);

expr_test!(
    chained_dot_access,
    "foo.bar.buzz",
    Access::Dot {
        left: Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        right: ident_expr!("buzz"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    chained_option_dot_access,
    "foo.bar?.fizz?.buzz",
    Access::Dot {
        left: Access::Dot {
            left: Access::Dot {
                left: ident_expr!("foo"),
                right: ident_expr!("bar"),
                kind: AccessKind::Direct,
            }
            .into_expr(),
            right: ident_expr!("fizz"),
            kind: AccessKind::Option,
        }
        .into_expr(),
        right: ident_expr!("buzz"),
        kind: AccessKind::Option,
    }
);

expr_test!(
    direct_dot_access_to_call,
    "foo.bar()",
    Call::new(
        Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(
    tuple_index_then_method,
    "foo.0.bar()",
    Call::new(
        Access::Dot {
            left: Access::Dot {
                left: ident_expr!("foo"),
                right: int!(0),
                kind: AccessKind::Direct,
            }
            .into_expr(),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(
    tuple_index_then_field,
    "foo.0.bar",
    Access::Dot {
        left: Access::Dot {
            left: ident_expr!("foo"),
            right: int!(0),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        right: ident_expr!("bar"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    indirect_dot_access_to_call,
    "foo?.bar()",
    Call::new(
        Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Option,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(
    direct_dot_access_to_square_access,
    "foo.bar[0]",
    Access::Square {
        left: Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        key: int!(0),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    option_dot_access_to_square_access,
    "foo?.bar[0]",
    Access::Square {
        left: Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Option,
        }
        .into_expr(),
        key: int!(0),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    direct_dot_access_from_call,
    "foo().bar",
    Access::Dot {
        left: Call::new(ident_expr!("foo"), vec![]).into_expr(),
        right: ident_expr!("bar"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    indirect_dot_access_from_call,
    "foo()?.bar",
    Access::Dot {
        left: Call::new(ident_expr!("foo"), vec![]).into_expr(),
        right: ident_expr!("bar"),
        kind: AccessKind::Option,
    }
);

expr_test!(
    chained_calls,
    "foo().bar()",
    Call::new(
        Access::Dot {
            left: Call::new(ident_expr!("foo"), vec![]).into_expr(),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(
    chain_calls_with_call_parameter,
    "foo().bar(buzz())",
    Call::new(
        Access::Dot {
            left: Call::new(ident_expr!("foo"), vec![]).into_expr(),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![Argument::new(
            Call::new(ident_expr!("buzz"), vec![]).into_expr()
        )],
    )
);

expr_test!(
    square_dot_access,
    "foo[0].bar",
    Access::Dot {
        left: Access::Square {
            left: ident_expr!("foo"),
            key: int!(0),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        right: ident_expr!("bar"),
        kind: AccessKind::Direct,
    }
);

expr_test!(
    double_colon_access,
    "foo::bar",
    Access::DoubleColon {
        left: ident_expr!("foo"),
        right: ident!("bar"),
    }
);

expr_test!(
    double_colon_call,
    "foo::bar()",
    Call::new(
        Access::DoubleColon {
            left: ident_expr!("foo"),
            right: ident!("bar"),
        }
        .into_expr(),
        vec![]
    )
);

expr_test!(
    multiple_double_colon_access,
    "foo::bar::buzz",
    Access::DoubleColon {
        left: Access::DoubleColon {
            left: ident_expr!("foo"),
            right: ident!("bar"),
        }
        .into_expr(),
        right: ident!("buzz")
    }
);

expr_test!(
    unwrap,
    "foo!",
    Unwrap {
        expr: ident_expr!("foo")
    }
);

expr_test!(
    unwrap_access,
    "foo!.bar",
    Access::Dot {
        left: Unwrap {
            expr: ident_expr!("foo"),
        }
        .into_expr(),
        right: ident_expr!("bar"),
        kind: AccessKind::Direct
    }
);

expr_test!(
    unwrap_member,
    "foo.bar!",
    Unwrap {
        expr: Access::Dot {
            left: ident_expr!("foo"),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct
        }
        .into_expr()
    }
);

expr_test!(
    unwrap_call,
    "foo()!",
    Unwrap {
        expr: Call::new(ident_expr!("foo"), vec![]).into_expr()
    }
);

expr_test!(filled_grouping, "(0)", Grouping::lazy(int!(0)));
expr_test!(
    nested_grouping,
    "((0) * 0)",
    Grouping::lazy(
        Evaluation::new(
            Grouping::lazy(int!(0)).into_expr(),
            EvaluationOp::Multiply,
            int!(0),
        )
        .into_expr(),
    )
);

expr_test!(r#true, "true", ExprKind::Literal(Literal::True));
expr_test!(r#false, "false", ExprKind::Literal(Literal::False));
expr_test!(null, "null", ExprKind::Literal(Literal::Null));
expr_test!(unit, "()", ExprKind::Literal(Literal::Unit));
expr_test!(ident, "foo", ident!("foo"));
expr_test!(int, "0", Literal::Int(0));
expr_test!(float, "0.1", Literal::Float(0.1));

expr_test!(
    string,
    "\"foo\"",
    ExprKind::Literal(Literal::String("foo".into()))
);

expr_test!(
    multi_line_string,
    "\"foo\nfoo\"",
    ExprKind::Literal(Literal::String("foo\nfoo".into()))
);
expr_test!(
    oh_x_hex,
    "0xa0f9a0",
    ExprKind::Literal(Literal::Hex("a0f9a0".into()))
);

expr_test!(
    logically_joined_expressions,
    "foo == 1 && foo == 1 && foo == 1",
    Logical::new(
        Logical::new(
            Equality::new(ident_expr!("foo"), EqualityOp::Equal, int!(1)).into_expr(),
            LogicalOp::And,
            Equality::new(ident_expr!("foo"), EqualityOp::Equal, int!(1)).into_expr(),
        )
        .into_expr(),
        LogicalOp::And,
        Equality::new(ident_expr!("foo"), EqualityOp::Equal, int!(1)).into_expr(),
    )
);

expr_test!(
    comment_in_builder_chain,
    "
            foo()
            // nothing in here!
            .bar()
        ",
    Call::new(
        Access::Dot {
            left: Call::new(ident_expr!("foo"), vec![]).into_expr(),
            right: ident_expr!("bar"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![],
    )
);

expr_test!(int_with_underscores, "1_000_000", Literal::Int(1_000_000));

expr_test!(
    float_with_underscores,
    "1_000.000_1",
    Literal::Float(1000.0001)
);

expr_test!(
    addition_chain,
    "1 + 2 + 3",
    Evaluation::new(
        Evaluation::new(int!(1), EvaluationOp::Plus, int!(2)).into_expr(),
        EvaluationOp::Plus,
        int!(3),
    )
);

expr_test!(
    add_then_mul_precedence,
    "1 + 2 * 3",
    Evaluation::new(
        int!(1),
        EvaluationOp::Plus,
        Evaluation::new(int!(2), EvaluationOp::Multiply, int!(3)).into_expr(),
    )
);

expr_test!(
    mul_then_add_precedence,
    "1 * 2 + 3",
    Evaluation::new(
        Evaluation::new(int!(1), EvaluationOp::Multiply, int!(2)).into_expr(),
        EvaluationOp::Plus,
        int!(3),
    )
);

expr_test!(
    bit_shift_left_chained,
    "1 << 2 << 3",
    Evaluation::new(
        Evaluation::new(int!(1), EvaluationOp::BitShiftLeft, int!(2)).into_expr(),
        EvaluationOp::BitShiftLeft,
        int!(3),
    )
);

expr_test!(
    chained_eq_chain,
    "1 == 1 == 1",
    Equality::new(
        Equality::new(int!(1), EqualityOp::Equal, int!(1)).into_expr(),
        EqualityOp::Equal,
        int!(1),
    )
);

expr_test!(
    nested_array,
    "[[1, 2], [3, 4]]",
    ExprKind::Literal(Literal::Array(vec![
        Literal::Array(vec![int!(1), int!(2)]).into_expr(),
        Literal::Array(vec![int!(3), int!(4)]).into_expr(),
    ]))
);

expr_test!(
    coalescence_chained,
    "a ?? b ?? c",
    Coalescence::new(
        ident_expr!("a"),
        Coalescence::new(ident_expr!("b"), ident_expr!("c"),).into_expr()
    )
);

expr_test!(
    coalescence_with_or,
    "a || b ?? c",
    Coalescence::new(
        Logical::new(ident_expr!("a"), LogicalOp::Or, ident_expr!("b")).into_expr(),
        ident_expr!("c"),
    )
);

expr_test!(
    call_named_after_positional,
    "foo(0, b=1)",
    Call::new(
        ident_expr!("foo"),
        vec![
            Argument::new(int!(0)),
            Argument::named(ident!("b"), int!(1))
        ]
    )
);

expr_test!(
    raise_str_literal,
    "raise \"bad\"",
    Raise::new(Literal::String("bad".into()).into_expr())
);

expr_test!(
    raise_expr_value,
    "raise foo()",
    Raise::new(Call::new(ident_expr!("foo"), vec![]).into_expr())
);

expr_test!(
    absolve_with_closure,
    "foo() absolve |e| 0",
    Absolve::new(
        Call::new(ident_expr!("foo"), vec![]).into_expr(),
        Closure::new(vec![Binding::new(ident!("e"))], int!(0)).into_expr(),
    )
);

expr_test!(
    // a block-ended handler hands the chain back: `==` binds the absolved
    // value, not the handler (`f() absolve |e| { d } == v`)
    absolve_then_equality,
    "foo() absolve |e| {0} == 1",
    Equality::new(
        Absolve::new(
            Call::new(ident_expr!("foo"), vec![]).into_expr(),
            Closure::new(
                vec![Binding::new(ident!("e"))],
                Block::new_with_yield(vec![], int!(0)).into_expr(),
            )
            .into_expr(),
        )
        .into_expr(),
        EqualityOp::Equal,
        int!(1),
    )
);

expr_test!(
    // a *bare* handler body stays greedy: `|e| 0 == 1` reads `==` inside the
    // closure, same as `??`'s fallback (`a ?? b == c`) — only a `}` ends the
    // handler's own parse
    absolve_bare_handler_stays_greedy,
    "foo() absolve |e| 0 == 1",
    Absolve::new(
        Call::new(ident_expr!("foo"), vec![]).into_expr(),
        Closure::new(
            vec![Binding::new(ident!("e"))],
            Equality::new(int!(0), EqualityOp::Equal, int!(1)).into_expr(),
        )
        .into_expr(),
    )
);

expr_test!(
    // `absolve` sits looser than the binary ops: it guards the whole
    // expression to its left, `a + f() absolve h` is `(a + f()) absolve h`
    // (which may then fail solving, e.g. `int + int!` — parens recover
    // first: `a + (f() absolve h)`)
    absolve_guards_the_sum,
    "a + f() absolve |e| {0}",
    Absolve::new(
        Evaluation::new(
            ident_expr!("a"),
            EvaluationOp::Plus,
            Call::new(ident_expr!("f"), vec![]).into_expr(),
        )
        .into_expr(),
        Closure::new(
            vec![Binding::new(ident!("e"))],
            Block::new_with_yield(vec![], int!(0)).into_expr(),
        )
        .into_expr(),
    )
);

expr_test!(
    // same hand-back after a `??` whose fallback ended in a block
    coalescence_then_equality,
    "a ?? { b: 1 } == c",
    Equality::new(
        Coalescence::new(
            ident_expr!("a"),
            Literal::Record(vec![(ident!("b"), int!(1))]).into_expr(),
        )
        .into_expr(),
        EqualityOp::Equal,
        ident_expr!("c"),
    )
);

expr_test!(membership, "1 in 1", In::new(int!(1), int!(1), true));

expr_test!(not_membership, "1 !in 1", In::new(int!(1), int!(1), false));

expr_test!(
    membership_before_or,
    "1 in 1 || 1 !in 1",
    Logical::new(
        In::new(int!(1), int!(1), true).into_expr(),
        LogicalOp::Or,
        In::new(int!(1), int!(1), false).into_expr(),
    )
);

fn parse_expr(source: &str) -> String {
    let lexer = crate::lex::Lexer::new(source, 0, "test".into());
    crate::Parser::new(lexer).expr().to_string()
}

#[test]
fn assert_eq_rewrites_to_inverted_if() {
    let text = parse_expr("assert!(a == 2)");
    // `a` needs the evaluate-once temp; the literal `2` inlines
    assert!(text.contains("__assert_left_1"), "{text}");
    assert!(!text.contains("__assert_right_1"), "{text}");
    assert!(text.contains("!="), "{text}");
    assert!(text.contains("=="), "{text}");
    assert!(text.contains("panic"), "{text}");
    assert!(text.contains("assertion failed:"), "{text}");
}

#[test]
fn assert_eq_null_inlines_no_bare_null_let() {
    // `assert(v == null)` used to desugar `let __r = null`, tripping the
    // bare-null lint on generated code -- literals never get temps.
    let text = parse_expr("assert!(v == null)");
    assert!(!text.contains("let __assert_right"), "{text}");
    assert!(text.contains("!= null"), "{text}");
}

#[test]
fn assert_grouped_eq_still_rewrites() {
    let text = parse_expr("assert!((1 == 2))");
    assert!(text.contains("!="), "{text}");
    assert!(text.contains("panic"), "{text}");
}

#[test]
fn assert_in_rewrites_to_not_in() {
    let text = parse_expr("assert!(1 in xs)");
    assert!(text.contains("!in"), "{text}");
    assert!(text.contains("panic"), "{text}");
}

#[test]
fn assert_bool_rewrites_to_not() {
    let text = parse_expr("assert!(ready)");
    assert!(
        text.contains("!ready") || text.contains("! ready"),
        "{text}"
    );
    assert!(text.contains("panic"), "{text}");
    assert!(text.contains("assertion failed:"), "{text}");
}

#[test]
fn nested_asserts_get_distinct_temps() {
    let text = parse_expr("assert!(assert!(a == b) == ())");
    // inner `a`/`b` get pair 1's temps; the outer's left (the assert
    // block) gets pair 2's left temp; `()` is a literal and inlines
    assert!(text.contains("__assert_left_1"), "{text}");
    assert!(text.contains("__assert_left_2"), "{text}");
}

#[test]
fn other_ident_bang_call_is_still_unwrap() {
    let text = parse_expr("foo!(1)");
    assert!(
        text.contains("foo?!"),
        "expected unwrap-then-call, got {text}"
    );
}

// — `|>` pipe: the loosest infix, desugared to a leading call argument —

expr_test!(
    pipe_bare_callee,
    "x |> f",
    Call::new(ident_expr!("f"), vec![Argument::new(ident_expr!("x"))])
);

expr_test!(
    pipe_call,
    "x |> f(1, 2)",
    Call::new(
        ident_expr!("f"),
        vec![
            Argument::new(ident_expr!("x")),
            Argument::new(int!(1)),
            Argument::new(int!(2)),
        ]
    )
);

expr_test!(
    pipe_chains_left,
    "x |> f |> g(2)",
    Call::new(
        ident_expr!("g"),
        vec![
            Argument::new(
                Call::new(ident_expr!("f"), vec![Argument::new(ident_expr!("x"))]).into_expr(),
            ),
            Argument::new(int!(2)),
        ]
    )
);

expr_test!(
    pipe_looser_than_everything,
    "a + b |> f",
    Call::new(
        ident_expr!("f"),
        vec![Argument::new(
            Evaluation::new(ident_expr!("a"), EvaluationOp::Plus, ident_expr!("b")).into_expr(),
        )]
    )
);

expr_test!(
    pipe_onto_access_call,
    "x |> n.f(1)",
    Call::new(
        Access::Dot {
            left: ident_expr!("n"),
            right: ident_expr!("f"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![Argument::new(ident_expr!("x")), Argument::new(int!(1))]
    )
);

expr_test!(
    pipe_call_then_unwrap,
    "x |> f(1)!",
    Unwrap {
        expr: Call::new(
            ident_expr!("f"),
            vec![Argument::new(ident_expr!("x")), Argument::new(int!(1))],
        )
        .into_expr(),
    }
);

expr_test!(
    pipe_into_first_call_of_chain,
    "x |> f(1)!.g(2)",
    Call::new(
        Access::Dot {
            left: Unwrap {
                expr: Call::new(
                    ident_expr!("f"),
                    vec![Argument::new(ident_expr!("x")), Argument::new(int!(1))],
                )
                .into_expr(),
            }
            .into_expr(),
            right: ident_expr!("g"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![Argument::new(int!(2))]
    )
);

expr_test!(
    pipe_continues_on_next_line,
    "x\n  |> f\n  |> g(2)",
    Call::new(
        ident_expr!("g"),
        vec![
            Argument::new(
                Call::new(ident_expr!("f"), vec![Argument::new(ident_expr!("x"))]).into_expr(),
            ),
            Argument::new(int!(2)),
        ]
    )
);

// `x ± d` desugars to `Interval::within(x * 1.0, d * 1.0)`; `x ± d%` to
// `Interval::pm(x * 1.0, d * 0.01)`.
// r[verify mimas.interval.literal]
#[test]
fn plus_minus_absolute() {
    assert_eq!(parse_expr("x ± 0.5"), "Interval::within(x * 1, 0.5 * 1)");
}

#[test]
fn plus_minus_relative() {
    assert_eq!(
        parse_expr("30000 ± 10%"),
        "Interval::pm(30000 * 1, 10 * 0.01)"
    );
    assert_eq!(parse_expr("a ± b %"), "Interval::pm(a * 1, b * 0.01)");
}

#[test]
fn plus_minus_binds_like_additive() {
    // `±` sits with `+`/`-`: its left side takes the whole sum to the left,
    // its right side takes one multiplicative run.
    assert_eq!(
        parse_expr("a + b ± c * 2"),
        "Interval::within(a + b * 1, c * 2 * 1)"
    );
    assert_eq!(
        parse_expr("x ± d + e"),
        "Interval::within(x * 1, d * 1) + e"
    );
    assert_eq!(
        parse_expr("x ± d * e"),
        "Interval::within(x * 1, d * e * 1)"
    );
}

#[test]
fn plus_minus_tolerance_may_be_complex() {
    assert_eq!(parse_expr("x ± f(1)"), "Interval::within(x * 1, f(1) * 1)");
    assert_eq!(parse_expr("x ± -d"), "Interval::within(x * 1, -d * 1)");
    // a `%` in the tolerance is the literal's percent marker, so `mod` needs parens
    assert_eq!(
        parse_expr("x ± (a mod b)"),
        "Interval::within(x * 1, (a mod b) * 1)"
    );
    assert_eq!(
        parse_expr("x ± (a mod b)%"),
        "Interval::pm(x * 1, (a mod b) * 0.01)"
    );
}

// `x%` hugs its operand and spells percent — `50%` is `50 * 0.01`. A spaced `%`
// isn't postfix or modulo — `binary` reports it and parses on as `mod`.
#[test]
fn percent_postfix() {
    assert_eq!(parse_expr("50%"), "50 * 0.01");
    assert_eq!(parse_expr("rate%"), "rate * 0.01");
    assert_eq!(parse_expr("f(50%)"), "f(50 * 0.01)");
    assert_eq!(parse_expr("10%.abs()"), "10 * 0.01.abs()");
    // inside a `±` tolerance's parens, `%` is postfix again
    assert_eq!(
        parse_expr("x ± (50%)"),
        "Interval::within(x * 1, (50 * 0.01) * 1)"
    );
}

#[test]
fn spaced_percent_is_old_modulo() {
    let lexer = crate::lex::Lexer::new("a % b", 0, "test".into());
    let mut parser = crate::Parser::new(lexer);
    let expr = parser.expr();
    assert_eq!(expr.to_string(), "a mod b");
    assert_eq!(parser.errors().len(), 1);
}

// r[verify mimas.struct.update]
// `Foo { x = 1, ..base }` -- the listed fields, then the rest from `base`
expr_test!(
    struct_update_literal,
    "Foo { x = 1, ..base }",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![(FieldKey::Ident(ident!("x")), int!(1))],
        base: Some(ident_expr!("base")),
    })
);

expr_test!(
    struct_update_only_base,
    "Foo { ..base }",
    Literal::Struct(StructLiteral {
        name: ident_expr!("Foo"),
        fields: vec![],
        base: Some(ident_expr!("base")),
    })
);

// ---- `with` — a scoped effect region. Parses to ordinary calls on the
// internal `__fx_<fx>_push`/`__fx_<fx>_pop` pair around a `__with_N` temp,
// so nothing downstream learns a new node; `with` stays a plain name in
// every other shape.

// r[verify mimas.fx.with]
expr_test!(
    with_desugars_to_push_pop,
    "with rng = 7 { 42 }",
    Block {
        body: vec![
            Call::new(
                ident_expr!("__fx_rng_push"),
                vec![Argument::new(int!(7))]
            )
            .into_expr()
            .into_stmt(),
            Let::new(
                ident!("__with_1").into(),
                Block::new_with_yield(vec![], int!(42)).into_expr(),
                None,
            )
            .into_stmt(),
            Call::new(ident_expr!("__fx_rng_pop"), vec![])
                .into_expr()
                .into_stmt(),
        ],
        yielded_expr: Some(ident_expr!("__with_1")),
    }
);

// r[verify mimas.fx.with]
expr_test!(
    with_yields_its_body,
    "with rng = seed { 1 + 2 }",
    Block {
        body: vec![
            Call::new(
                ident_expr!("__fx_rng_push"),
                vec![Argument::new(ident_expr!("seed"))]
            )
            .into_expr()
            .into_stmt(),
            Let::new(
                ident!("__with_1").into(),
                Block::new_with_yield(
                    vec![],
                    Evaluation::new(int!(1), EvaluationOp::Plus, int!(2)).into_expr(),
                )
                .into_expr(),
                None,
            )
            .into_stmt(),
            Call::new(ident_expr!("__fx_rng_pop"), vec![])
                .into_expr()
                .into_stmt(),
        ],
        yielded_expr: Some(ident_expr!("__with_1")),
    }
);

/// `parse`'s errors as message strings — what a rejected `with` teaches.
fn with_errors(source: &str) -> Vec<String> {
    let (_, errors) = crate::tests::utils::parse(source);
    errors.iter().map(ToString::to_string).collect()
}

// r[verify mimas.fx.with]
#[test]
fn with_unknown_effect_named() {
    let errs = with_errors("let _x = with speed = 7 { 1 };");
    assert!(
        errs.iter().any(|e| e.contains("`speed` isn't an effect")),
        "{errs:?}"
    );
}

// r[verify mimas.fx.with]
#[test]
fn with_only_rng_scopes_yet() {
    let errs = with_errors("let _x = with net = m { fetch() };");
    assert!(
        errs.iter()
            .any(|e| e.contains("can't install `net` yet")),
        "{errs:?}"
    );
}

// r[verify mimas.fx.with]
#[test]
fn with_wants_a_block_body() {
    let errs = with_errors("let _x = with rng = 7;");
    assert!(
        errs.iter().any(|e| e.contains("wants a `{` body")),
        "{errs:?}"
    );
}

// r[verify mimas.fx.with.exits]
#[test]
fn with_refuses_boundary_exits() {
    // a structured exit skips the desugared restore — refused at parse
    for (src, exit) in [
        ("fn f() { with rng = 1 { return 0 } }", "`return`"),
        ("fn f() { with rng = 1 { raise 0 } }", "`raise`"),
        ("fn f() { with rng = 1 { g()? } }", "`?`"),
        ("fn f() { with rng = 1 { break } }", "`break`"),
        ("fn f() { with rng = 1 { continue } }", "`continue`"),
    ] {
        let errs = with_errors(src);
        assert!(
            errs.iter()
                .any(|e| e.contains("can't leave a `with` region") && e.contains(exit)),
            "`{src}`: {errs:?}"
        );
    }
}

// r[verify mimas.fx.with.exits]
#[test]
fn with_allows_bound_exits() {
    // a loop inside the region binds its own `break`/`continue`; a closure
    // binds its own `return`/`?` — none escape the region
    let (_, errors) = crate::tests::utils::parse(
        "fn f() { with rng = 1 {
            for x in [1] { break }
            for x in [1] { continue }
            let _g = || { return 0 };
            let _h = || { g()? };
        } }",
    );
    assert!(errors.is_empty(), "{errors:?}");
}

// r[verify mimas.fx.with]
#[test]
fn with_stays_a_name_elsewhere() {
    // `with` is contextual: a call, a binding, an `==` (not `=`) — all plain
    crate::tests::utils::assert_parses("with(1);");
    crate::tests::utils::assert_parses("let with = 3;");
    crate::tests::utils::assert_parses("let _x = with rng == 7;");
    // and a newline after `with` keeps it a name too
    crate::tests::utils::assert_parses("fn f() { let _x = with\nrng = 7; }");
}

// `∑ xs` is the big form of `+`: a prefix reduction, desugared to `xs.sum()`.
expr_test!(
    sum_reduction,
    "∑ xs",
    Call::new(
        Access::Dot {
            left: ident_expr!("xs"),
            right: ident_expr!("sum"),
            kind: AccessKind::Direct,
        }
        .into_expr(),
        vec![],
    )
);

// `a ⋅ b` is `a.dot(b)` through the named operators: `⋅` is DotOp, which
// parses to EvaluationOp::Dot and lowers to the `dot` method.
expr_test!(
    cdot_is_named_dot,
    "a ⋅ b",
    Evaluation::new(ident_expr!("a"), EvaluationOp::Dot, ident_expr!("b"))
);
