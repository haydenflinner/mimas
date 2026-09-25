use crate::lex::{TokKind::*, TyKw};

tok_test!(whitepsace: " \n\t" =>);
tok_test!(int: "1" => Int(1));
tok_test!(int_with_underscores: "1_000_000" => Int(1_000_000));
tok_test!(int_with_irregular_underscores: "1_0_00___000" => Int(1_000_000));
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
tok_test!(hash: "#" => Hash);
tok_test!(dollar: "$" => Dollar);
tok_test!(invalid_backslash: "\\" => Invalid("\\"));
tok_test!(invalid_multibyte: "a § b" => Ident("a"), Invalid("§"), Ident("b"));

tok_test!(line_comment: "1 // a comment\n2" => Int(1), Comment("// a comment"), Int(2));
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
tok_test!(tilde_dict: "~{}" => TildeLeftBrace, RightBrace);
tok_test!(tilde_slash: "a ~/ b" => Ident("a"), TildeSlash, Ident("b"));
tok_test!(hook_dot: "a?.b" => Ident("a"), HookDot, Ident("b"));
tok_test!(hook_left_square: "a?[0]" => Ident("a"), HookLeftSquare, Int(0), RightSquare);
tok_test!(double_hook_eq: "a ??= b" => Ident("a"), DoubleHookEqual, Ident("b"));
tok_test!(pipe_greater: "a |> b" => Ident("a"), PipeGreater, Ident("b"));
tok_test!(pipe_greater_vs_closure: "a ||> b" => Ident("a"), DoublePipe, Greater, Ident("b"));
tok_test!(compound_assigns: "+= -= *= /= %= |= &= ^= ~/=" =>
    PlusEqual, MinusEqual, StarEqual, SlashEqual, PercentEqual,
    PipeEqual, AmpersandEqual, CaretEqual, TildeSlashEqual);
tok_test!(at_path: "@9" => At, Int(9));
tok_test!(keyword_ident_boundary: "ifx" => Ident("ifx"));
tok_test!(keyword_self: "self" => SelfKeyword);

tok_test!(not_in_operator: "!in" => NotIn);
tok_test!(not_in_before_space: "!in xs" => NotIn, Ident("xs"));
tok_test!(bang_before_in_prefixed_ident: "!inside" => Bang, Ident("inside"));
tok_test!(bang_before_in_ident: "!in_range" => Bang, Ident("in_range"));
