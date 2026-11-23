//! The Z-set notation: each glyph lexes as its own token and parses to the
//! same tree as its ASCII spelling, at the precedence the spec gives it.
use crate::lex::TokKind::*;

/// Parses `src` as one expression and fails on any parse error.
fn parse(src: &str) -> crate::expr::Expr {
    let lexer = crate::lex::Lexer::new(src, 0, "test".into());
    let mut parser = crate::Parser::new(lexer);
    let out = parser.expr();
    assert!(parser.errors().is_empty(), "{src}: {:?}", parser.errors());
    out
}

/// `notation` and `ascii` parse to the same tree.
fn same(notation: &str, ascii: &str) {
    let (a, b) = (parse(notation), parse(ascii));
    assert_eq!(a.kind(), b.kind(), "`{notation}` should parse as `{ascii}`:\n{a}\n{b}");
}

// r[verify mimas.query.zset.stream-ops]
tok_test!(stream_glyphs: "∫ ∇ Δ z⁻¹ E⁻¹" => StreamOp("∫"), StreamOp("∇"), StreamOp("Δ"), StreamOp("z⁻¹"), StreamOp("E⁻¹"));
tok_test!(inverse_glyphs_are_one_token: "Δ⁻¹ c ∇⁻¹ c" => StreamOp("Δ⁻¹"), Ident("c"), StreamOp("∇⁻¹"), Ident("c"));
// a glyph that starts like a name is the operator only on its own
tok_test!(delta_in_a_name_stays_a_name: "Δt Δx2" => Ident("Δt"), Ident("Δx2"));
tok_test!(z_stays_a_name: "z zs z1" => Ident("z"), Ident("zs"), Ident("z1"));
tok_test!(e_stays_a_name: "E s" => Ident("E"), Ident("s"));
tok_test!(stream_glyph_glued: "∫zs" => StreamOp("∫"), Ident("zs"));
// r[verify mimas.query.zset.notation]
tok_test!(zset_postfix_glyphs: "z⁺ z⁻ ‖z‖ a ⋈ b" => Ident("z"), SupPlus, Ident("z"), SupMinus, Norm, Ident("z"), Norm, Ident("a"), Bowtie, Ident("b"));
// r[verify mimas.query.zset.incremental]
tok_test!(incremental_postfix: "q^Δ" => Ident("q"), Incremental);
tok_test!(incremental_after_paren: "(q)^Δ" => LeftParenthesis, Ident("q"), RightParenthesis, Incremental);
// `x^2` still superscripts a name, and so does a `^Δ` with more name after it
tok_test!(caret_names_stay: "x^2 x^Δt" => Ident("x^2"), Ident("x^Δt"));

// r[verify mimas.query.zset.stream-ops]
#[test]
fn stream_glyphs_are_their_methods() {
    same("∫ zs", "zs.integrate()");
    same("∇⁻¹ zs", "zs.integrate()");
    same("∇ zs", "zs.differentiate()");
    same("Δ zs", "zs.forward_difference()");
    same("Δ⁻¹ c", "c.sum_before()");
    same("z⁻¹ zs", "zs.delay()");
    same("E⁻¹ zs", "zs.delay()");
}

// r[verify mimas.query.zset.precedence]
#[test]
fn stream_glyphs_bind_like_unary_minus() {
    // postfix binds tighter: the glyph takes the whole call chain
    same("∫ zs.map(q)", "zs.map(q).integrate()");
    // and the infix operators are looser
    same("∇ zs == zs - z⁻¹ zs", "zs.differentiate() == zs - zs.delay()");
    same("∇ ∫ zs", "zs.integrate().differentiate()");
    same("-∫ zs", "-zs.integrate()");
    same("2 * ∫ zs", "2 * zs.integrate()");
}

// r[verify mimas.query.zset.notation]
// r[verify mimas.query.zset.precedence]
#[test]
fn postfix_parts_bind_tightest() {
    same("z⁺", "z.added()");
    same("z⁻", "z.removed()");
    same("z⁺ - z⁻", "z.added() - z.removed()");
    same("-z⁺", "-z.added()");
    same("z.f()⁺", "z.f().added()");
    same("‖z‖", "(z).size()");
    same("‖z⁺‖ + ‖z⁻‖ == ‖z‖", "(z.added()).size() + (z.removed()).size() == (z).size()");
    same("‖a - b‖", "(a - b).size()");
    same("2 · z", "2 ⋅ z");
}

// r[verify mimas.query.zset.join-op]
// r[verify mimas.query.zset.precedence]
#[test]
fn bowtie_binds_like_times() {
    same("a ⋈ b", "a.natural_join(b)");
    same("a ⋈ b + c", "a.natural_join(b) + c");
    same("a + b ⋈ c", "a + b.natural_join(c)");
    same("a ⋈ b ⋈ c", "a.natural_join(b).natural_join(c)");
    same("∇ a ⋈ b", "a.differentiate().natural_join(b)");
}

// r[verify mimas.query.zset.incremental]
#[test]
fn incremental_is_a_call() {
    same("q^Δ", "incremental(q)");
    same("q^Δ(s)", "incremental(q)(s)");
    same("(f ∘ g)^Δ", "incremental((f ∘ g))");
    same("f^Δ ∘ g^Δ", "incremental(f) ∘ incremental(g)");
}

// r[verify mimas.query.zset.stream-ops]
#[test]
fn a_glyph_alone_is_a_function() {
    // `Δ⁻¹ ∘ Δ`: each glyph with no operand is its method as a generic
    // function on streams of any group (`library/src/group.mim`)
    // r[verify mimas.group.stream]
    let e = parse("Δ⁻¹ ∘ Δ");
    let crate::expr::ExprKind::Evaluation(ev) = e.kind() else { panic!("{e}") };
    assert_eq!(ev.op, crate::expr::EvaluationOp::Compose);
    assert_eq!(ev.left.as_ident().map(|i| i.lexeme.as_str()), Some("__stream_sum_before"));
    assert_eq!(
        ev.right.as_ident().map(|i| i.lexeme.as_str()),
        Some("__stream_forward_difference")
    );
    same("f(∇)", "f(__stream_differentiate)");
    same("[∫, z⁻¹]", "[__stream_integrate, __stream_delay]");
}
