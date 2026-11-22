use crate::lex::{TokKind::*, TyKw};

tok_test!(whitepsace: " \n\t" =>);
tok_test!(int: "1" => Int(1));
tok_test!(int_with_underscores: "1_000_000" => Int(1_000_000));
tok_test!(int_with_irregular_underscores: "1_0_00___000" => Int(1_000_000));
// `_` is a separator anywhere inside a digit run, not only between two digits as in Rust:
// doubled (`1__2`), trailing (`1_`), and wrapped around the `.` (`1_.5`) all lex, the `_`s
// simply stripped -- `int_with_irregular_underscores` above already settled on the lax rule.
tok_test!(float_with_underscores: "1_0.5" => Float(10.5));
tok_test!(float_underscore_around_dot: "1_.5 1._5" => Float(1.5), Float(1.5));
tok_test!(int_trailing_underscore: "1_" => Int(1));
tok_test!(int_doubled_underscore: "1__2" => Int(12));
// a `_` where a number can't start is an identifier, same as Rust
tok_test!(leading_underscore_is_ident: "_1" => Ident("_1"));
// hex literals take no separators: the digits stop at `_`, which then leads an identifier
tok_test!(hex_underscore_tail: "0xFF_FF" => Hex("FF"), Ident("_FF"));
// float exponents: signed or not, separators allowed, and the number ends where the
// exponent does -- `1.5e2+1` is three tokens, not one (chompy's own exponent scan used
// to drop a char per digit and eat the tail)
tok_test!(float_exponent: "1.5e3" => Float(1500.0));
tok_test!(float_exponent_signed: "1.5e+3 1.5e-3" => Float(1500.0), Float(0.0015));
tok_test!(float_exponent_with_underscores: "1.5e1_0" => Float(15000000000.0));
// r[verify mimas.num.text]
// the whole literal parses in one go, so a float's own text reads back as
// the same float — a mantissa times `10^exp` rounded twice, and `float::MAX`
// as it prints came back as infinity
tok_test!(float_exponent_reads_back_at_the_top: "1.7976931348623157e308" => Float(f64::MAX));
tok_test!(float_exponent_reads_back_at_the_bottom: "5e-324 2.2250738585072014e-308" => Float(5e-324), Float(f64::MIN_POSITIVE));
tok_test!(float_exponent_is_correctly_rounded: "1e-7 1e21 1e23" => Float(1e-7), Float(1e21), Float(1e23));
tok_test!(float_exponent_then_code: "1.5e2+1" => Float(150.0), Plus, Int(1));
tok_test!(float_e_without_digits: "1.5else" => Float(1.5), Else);
// r[verify mimas.num.exponent]
// int mantissas take exponents too -- `1e-6` is one float, not `1` then `e-6` (the
// old lex made `(x).abs() < 1e-6` fail as "`int` cannot be called"). An `e` that
// isn't followed by digits still isn't an exponent: `1else` and `2e` stand.
tok_test!(int_exponent: "1e-6" => Float(1e-6));
tok_test!(int_exponent_signed: "1e+6 2e3" => Float(1e6), Float(2000.0));
tok_test!(int_exponent_with_underscores: "1e1_0" => Float(10_000_000_000.0));
tok_test!(int_exponent_then_code: "1e2+1" => Float(100.0), Plus, Int(1));
tok_test!(int_e_without_digits: "1else" => Int(1), Else);
tok_test!(int_e_alone: "2e" => Int(2), Ident("e"));
tok_test!(quantity_with_underscores: "1_0kW" => Quantity(10.0, "kW"));
tok_test!(float: "0.2" => Float(0.2));
tok_test!(float_zero: "0.0" => Float(0.0));
tok_test!(range: "0..1" => Int(0), DoubleDot, Int(1));
tok_test!(range_eq: "0..=1" => Int(0), DoubleDotEqual, Int(1));
tok_test!(access_int: "0.a" => Int(0), Dot, Ident("a"));
tok_test!(dot_vs_doubledot: ". .. ..=" => Dot, DoubleDot, DoubleDotEqual);
tok_test!(hex_0x: "0x1" => Hex("1"));
tok_test!(hex_upper: "0xFF" => Hex("FF"));
tok_test!(neg_int: "-1" => Minus, Int(1));
tok_test!(neg_float: "-1.5" => Minus, Float(1.5));

// a number glued to a unit is one token; anything else after a number stays as it was
tok_test!(quantity_int: "25kW" => Quantity(25.0, "kW"));
tok_test!(quantity_float: "0.14usd/kWh" => Quantity(0.14, "usd/kWh"));
tok_test!(quantity_exponent: "9.81m/s^2" => Quantity(9.81, "m/s^2"));
tok_test!(quantity_then_operator: "5kW+3kW" => Quantity(5.0, "kW"), Plus, Quantity(3.0, "kW"));
tok_test!(quantity_slash_variable_is_division: "5W/x" => Quantity(5.0, "W"), Slash, Ident("x"));
tok_test!(quantity_slash_number_is_division: "5W/2" => Quantity(5.0, "W"), Slash, Int(2));
tok_test!(quantity_method_call: "5kW.abs" => Quantity(5.0, "kW"), Dot, Ident("abs"));
tok_test!(spaced_unit_is_not_a_quantity: "5 kW" => Int(5), Ident("kW"));
tok_test!(number_then_non_unit_ident: "3x" => Int(3), Ident("x"));

tok_test!(empty_string: "\"\"" => String(""));
tok_test!(string: "\"foo\"" => String("foo"));
tok_test!(multiline_string: "\"foo\nfoo\"" => String("foo\nfoo"));
tok_test!(escape_string: "\"foo\\\"bar\"" => String("foo\\\"bar"));
tok_test!(fstring_plain: "f\"\"" => FString(""));
tok_test!(fstring: "f\"foo {a}\"" => FString("foo {a}"));
tok_test!(fstring_with_quoted_index: "f\"{m[\"k\"]}\"" => FString("{m[\"k\"]}"));
tok_test!(fstring_with_quoted_brace: "f\"{m[\"}\"]}\"" => FString("{m[\"}\"]}"));

// `"""…"""` — the docstring: one String token whose content may span
// lines and hold lone `"`s; an unassigned one in statement position is
// the Python-flavored doc comment the editor typesets.
tok_test!(docstring: "\"\"\"docs\"\"\"" => String("docs"));
tok_test!(docstring_multiline: "\"\"\"line one\nline two\"\"\"" => String("line one\nline two"));
tok_test!(docstring_holds_lone_quotes: "\"\"\"say \"hi\" now\"\"\"" => String("say \"hi\" now"));
tok_test!(docstring_empty: "\"\"\"\"\"\"" => String(""));
tok_test!(docstring_then_stmt: "\"\"\"d\"\"\" 1" => String("d"), Int(1));
// four quotes is a docstring opener + a lone `"` inside its body
tok_test!(docstring_four_quotes: "\"\"\"\"x\"\"\"" => String("\"x"));
tok_test!(ident: "foo" => Ident("foo"));

// kebab-case identifiers: `-` continues an identifier only when it sits directly between two
// identifier chars with no whitespace on either side. That's the whole disambiguation from the
// `Minus` token -- a `-` next to whitespace (or at the very start) is never absorbed here, so
// writing the operator with spaces around it is what keeps `foo - bar` meaning subtraction.
tok_test!(kebab_ident: "my-var" => Ident("my-var"));
tok_test!(kebab_ident_multi_segment: "my-long-var-name" => Ident("my-long-var-name"));
tok_test!(kebab_ident_trailing_digit: "x-1" => Ident("x-1"));
tok_test!(kebab_ident_with_underscore: "my_var-2" => Ident("my_var-2"));
tok_test!(kebab_ident_vs_spaced_minus: "foo - bar" => Ident("foo"), Minus, Ident("bar"));
tok_test!(kebab_ident_space_before_minus: "foo -bar" => Ident("foo"), Minus, Ident("bar"));
tok_test!(kebab_ident_space_after_minus: "foo- bar" => Ident("foo"), Minus, Ident("bar"));
tok_test!(kebab_ident_trailing_hyphen_not_absorbed: "foo-)" => Ident("foo"), Minus, RightParenthesis);
tok_test!(kebab_ident_double_hyphen_not_absorbed: "foo--bar" => Ident("foo"), Minus, Minus, Ident("bar"));
tok_test!(kebab_ident_leading_hyphen_is_minus: "-foo" => Minus, Ident("foo"));

tok_test!(tykw_int: "int" => TyKw(TyKw::Int));
tok_test!(tykw_float: "float" => TyKw(TyKw::Float));
tok_test!(tykw_str: "str" => TyKw(TyKw::Str));
tok_test!(tykw_bool: "bool" => TyKw(TyKw::Bool));

tok_test!(invalid_backtick: "`" => Invalid("`"));
tok_test!(hash: "#" => Comment("#"));
tok_test!(dollar: "$" => Dollar);
tok_test!(invalid_backslash: "\\" => Invalid("\\"));
tok_test!(invalid_multibyte: "a § b" => Ident("a"), Invalid("§"), Ident("b"));

tok_test!(line_comment: "1 // a comment\n2" => Int(1), Comment("// a comment"), Int(2));
tok_test!(hash_comment: "# section\n1" => Comment("# section"), Int(1));
tok_test!(hash_sub_comment: "## sub\n1" => Comment("## sub"), Int(1));
tok_test!(hash_attr_stays: "#[test] fn f() {}" => Hash, LeftSquare, Ident("test"), RightSquare, Fn, Ident("f"), LeftParenthesis, RightParenthesis, LeftBrace, RightBrace);
tok_test!(hash_no_space_stays: "#x" => Hash, Ident("x"));
tok_test!(comment_only: "// nothing here" => Comment("// nothing here"));
tok_test!(comment_trailing_eof: "1 // tail" => Int(1), Comment("// tail"));
tok_test!(doc_comment: "/// docs" => DocComment("/// docs"));

tok_test!(
    comment_lines_merge,
    "// a
     //
     // b
     1",
    Comment(
        "// a
     //
     // b"
    ),
    Int(1),
);

tok_test!(
    doc_comment_lines_merge,
    "/// a
     ///
     /// b",
    DocComment(
        "/// a
     ///
     /// b"
    ),
);

tok_test!(
    crlf_comment_lines_merge,
    "// a\r
     // b\r
     1",
    Comment(
        "// a\r
     // b"
    ),
    Int(1),
);

tok_test!(
    blank_line_splits_comments,
    "// a

     // b",
    Comment("// a"),
    Comment("// b"),
);

tok_test!(
    doc_comment_splits_from_comment,
    "// a
     /// b
     // c",
    Comment("// a"),
    DocComment("/// b"),
    Comment("// c"),
);

tok_test!(
    comment_before_slash,
    "// a
     / 2",
    Comment("// a"),
    Slash,
    Int(2),
);

tok_test!(
    line_comment_then_block_comment,
    "// a
     /* b */",
    Comment("// a"),
    Comment("/* b */"),
);

tok_test!(block_comment: "1 /* a comment */ 2" => Int(1), Comment("/* a comment */"), Int(2));
tok_test!(block_comment_only: "/* nothing here */" => Comment("/* nothing here */"));
tok_test!(block_comment_multiline: "1 /* a\nb\nc */ 2" =>
    Int(1), Comment("/* a\nb\nc */"), Int(2));
tok_test!(block_comment_empty_body: "1 /**/ 2" => Int(1), Comment("/**/"), Int(2));
tok_test!(block_comment_nested: "1 /* a /* b */ c */ 2" =>
    Int(1), Comment("/* a /* b */ c */"), Int(2));
tok_test!(block_comment_shared_slash: "1 /*/ 2 */ 3" => Int(1), Comment("/*/ 2 */"), Int(3));
tok_test!(block_comment_holds_line_comment: "1 /* // not a line comment */ 2" =>
    Int(1), Comment("/* // not a line comment */"), Int(2));
tok_test!(block_comment_between_tokens: "a/*x*/+/*y*/b" =>
    Ident("a"), Comment("/*x*/"), Plus, Comment("/*y*/"), Ident("b"));
tok_test!(block_comment_ignored_in_string: "\"/* not */\"" => String("/* not */"));
tok_test!(division_not_a_comment: "6 / 2" => Int(6), Slash, Int(2));

tok_test!(double_colon_path: "a::b" => Ident("a"), DoubleColon, Ident("b"));
tok_test!(tilde: "~{}" => Tilde, LeftBrace, RightBrace);
tok_test!(tilde_slash: "a ~/ b" => Ident("a"), TildeSlash, Ident("b"));
tok_test!(hook_dot: "a?.b" => Ident("a"), HookDot, Ident("b"));
tok_test!(hook_left_square: "a?[0]" => Ident("a"), HookLeftSquare, Int(0), RightSquare);
tok_test!(double_hook_eq: "a ??= b" => Ident("a"), DoubleHookEqual, Ident("b"));
tok_test!(pipe_greater_vs_closure: "a ||> b" => Ident("a"), DoublePipe, Greater, Ident("b"));
tok_test!(compound_assigns: "+= -= *= /= %= |= &= ^= ⊕= ~/=" =>
    PlusEqual, MinusEqual, StarEqual, SlashEqual, PercentEqual,
    PipeEqual, AmpersandEqual, CaretEqual, CircledPlusEqual, TildeSlashEqual);
tok_test!(at_path: "@9" => At, Int(9));
tok_test!(keyword_ident_boundary: "ifx" => Ident("ifx"));
tok_test!(keyword_self: "self" => SelfKeyword);

tok_test!(not_in_operator: "!in" => NotIn);
tok_test!(not_in_before_space: "!in xs" => NotIn, Ident("xs"));
tok_test!(bang_before_in_prefixed_ident: "!inside" => Bang, Ident("inside"));
tok_test!(bang_before_in_ident: "!in_range" => Bang, Ident("in_range"));

tok_test!(plus_minus: "x ± 0.5" => Ident("x"), PlusMinus, Float(0.5));
tok_test!(plus_minus_percent: "30000 ± 10%" => Int(30000), PlusMinus, Int(10), Percent);

// r[verify mimas.ident.symbols]
// symbol and emoji blocks are name characters; the operator symbols stay operators
tok_test!(symbol_names: "☀ 🌙 ↗ x⬤" => Ident("☀"), Ident("🌙"), Ident("↗"), Ident("x⬤"));
// the move glyphs, the supplemental arrow blocks included
tok_test!(move_glyph_names: "↻ ↺ ⇄ ⇅ ⤡ ⟳" => Ident("↻"), Ident("↺"), Ident("⇄"), Ident("⇅"), Ident("⤡"), Ident("⟳"));
tok_test!(symbol_operators_stay: "a ⊕ b" => Ident("a"), CircledPlus, Ident("b"));

// r[verify mimas.op.glyphs]
// the named operators are tokens of their own, with no built-in meaning
tok_test!(named_operators_multiplicative: "a ∧ b ⋅ c · d × e ∘ f ⊠ g ⊡ h ∩ i" => Ident("a"), Wedge, Ident("b"), DotOp, Ident("c"), DotOp, Ident("d"), Cross, Ident("e"), Compose, Ident("f"), Tensor, Ident("g"), SqDot, Ident("h"), Intersect, Ident("i"));
tok_test!(named_operators_additive: "a ∨ b ∪ c ⊖ d ⊎ e ⊞ f ⊟ g" => Ident("a"), Vee, Ident("b"), Union, Ident("c"), Ominus, Ident("d"), Uplus, Ident("e"), SqPlus, Ident("f"), SqMinus, Ident("g"));
tok_test!(named_operator_hugs_names: "a∧b" => Ident("a"), Wedge, Ident("b"));
// r[verify mimas.op.cmp-glyphs]
// the comparison and membership glyphs are the ASCII tokens
tok_test!(comparison_glyphs: "a ≠ b ≤ c ≥ d" => Ident("a"), BangEqual, Ident("b"), LessEqual, Ident("c"), GreaterEqual, Ident("d"));
tok_test!(membership_glyphs: "x ∈ xs ∉ ys" => Ident("x"), In, Ident("xs"), NotIn, Ident("ys"));
// `¬` spells `not`; `∧`/`∨` stay named operators, not `&&`/`||` spellings
tok_test!(not_glyph: "¬ x" => NotWord, Ident("x"));
tok_test!(reductions: "∑ ∏ ⨅ ⨆" => Sum, Prod, Inf, Sup);
tok_test!(direction_glyphs: "→ ← ↑ ↓ ↔ ↕" => RightArrow, LeftArrow, UpArrow, DownArrow, LeftRightArrow, UpDownArrow);

// r[verify mimas.comment.doc]
// the doc-comment family, Rust-aligned: `///` documents the item that
// follows, `//!` the enclosing scope; `/**`/`/*!` are the block forms.
// Three slashes or two stars exactly — `////` and `/***` are below.
tok_test!(doc_comment_block: "/** docs */" => DocComment("/** docs */"));
tok_test!(doc_comment_block_multiline: "/** a\nb */" => DocComment("/** a\nb */"));
tok_test!(inner_doc_comment: "//! the whole scope" => InnerDocComment("//! the whole scope"));
tok_test!(inner_doc_comment_block: "/*! the whole scope */" => InnerDocComment("/*! the whole scope */"));
tok_test!(doc_comment_empty_block: "/**/" => Comment("/**/"));
tok_test!(
    inner_doc_lines_merge,
    "//! a\n     //! b",
    InnerDocComment("//! a\n     //! b")
);
tok_test!(doc_then_inner_doc: "/// a\n//! b" => DocComment("/// a"), InnerDocComment("//! b"));
tok_test!(inner_doc_splits_from_comment: "// a\n//! b" => Comment("// a"), InnerDocComment("//! b"));

// r[verify mimas.comment.literal]
// `////` (four slashes up) and `/***` (three stars up) are literal
// comments — commented out harder, usually dead code — not doc comments
tok_test!(literal_line_comment: "//// dead code" => LiteralComment("//// dead code"));
tok_test!(literal_line_comment_long: "////// even more" => LiteralComment("////// even more"));
tok_test!(literal_block_comment: "/*** dead code */" => LiteralComment("/*** dead code */"));
tok_test!(literal_block_comment_solid: "/***/" => LiteralComment("/***/"));
tok_test!(literal_block_comment_longer: "/****/" => LiteralComment("/****/"));
tok_test!(
    literal_lines_merge,
    "//// a\n     //// b",
    LiteralComment("//// a\n     //// b")
);
tok_test!(literal_splits_from_doc: "/// a\n//// b" => DocComment("/// a"), LiteralComment("//// b"));
tok_test!(literal_splits_from_comment: "// a\n//// b" => Comment("// a"), LiteralComment("//// b"));

// r[verify mimas.prose.island]
// a line whose content opens with ``` starts a prose island: the lines
// up to a line whose whole content is ``` are swallowed verbatim — no
// per-line prefix, nothing inside is lexed
tok_test!(prose_island: "```\nplain words, 1 + 2, \"quotes\"\n```" =>
    ProseIsland("```\nplain words, 1 + 2, \"quotes\"\n```"));
tok_test!(prose_island_indented: "  ```\ntext\n   ```" =>
    ProseIsland("```\ntext\n   ```"));
tok_test!(prose_island_closer_trailing_blanks: "```\ntext\n```   " =>
    ProseIsland("```\ntext\n```   "));
tok_test!(prose_island_mid_module: "1\n```\ntext\n```\n2" =>
    Int(1), ProseIsland("```\ntext\n```"), Int(2));
tok_test!(prose_island_at_top_then_code: "```\nthe module doc\n```\nfn f() {}" =>
    ProseIsland("```\nthe module doc\n```"), Fn, Ident("f"), LeftParenthesis, RightParenthesis, LeftBrace, RightBrace);
tok_test!(prose_island_opener_trailing_text: "``` a note\ntext\n```" =>
    ProseIsland("``` a note\ntext\n```"));
// a longer fence inside is just island text — only a lone ``` line closes
tok_test!(prose_island_longer_fence_inside: "```\n````\n```" =>
    ProseIsland("```\n````\n```"));
// ``` with more text on the line is not a closer either
tok_test!(prose_island_fence_with_text_inside: "```\n``` nope\n```" =>
    ProseIsland("```\n``` nope\n```"));
// mid-line backticks are not a fence — one Invalid each, as before
tok_test!(midline_backticks: "a ``` b" => Ident("a"), Invalid("`"), Invalid("`"), Invalid("`"), Ident("b"));
tok_test!(backtick_after_code: "1 ```" => Int(1), Invalid("`"), Invalid("`"), Invalid("`"));
// a comment above an island doesn't merge into it — different trivia kinds
tok_test!(comment_then_island: "// a\n```\nb\n```" => Comment("// a"), ProseIsland("```\nb\n```"));

// an island that never closes eats the rest of the source and reports on
// the opener — the token test macro wants a clean error list, so this
// one walks the lexer itself
#[test]
fn prose_island_unterminated() {
    let mut lexer = crate::lex::Lexer::new("```\nnever closed\n1 + 2", 0, "test".into());
    let toks: Vec<_> = lexer.by_ref().map(|tok| tok.kind).collect();
    assert_eq!(toks, vec![ProseIsland("```\nnever closed\n1 + 2")]);
    let errors = lexer.take_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
}

#[test]
fn prose_island_opener_alone_is_unterminated() {
    let mut lexer = crate::lex::Lexer::new("```", 0, "test".into());
    let toks: Vec<_> = lexer.by_ref().map(|tok| tok.kind).collect();
    assert_eq!(toks, vec![ProseIsland("```")]);
    assert_eq!(lexer.take_errors().len(), 1);
}
