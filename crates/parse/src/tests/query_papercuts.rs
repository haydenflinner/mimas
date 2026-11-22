//! The query papercuts PageRank found (2026-11-22): `join t (a == x)`,
//! the bare `aggregate` verb, `derive … select …` on one line, and the
//! empty dict `{:}`. Each form parses to the tree its long spelling does.
use crate::expr::{ExprKind, Literal};

/// Parses `src` as one expression and fails on any parse error.
fn parse(src: &str) -> crate::expr::Expr {
    let lexer = crate::lex::Lexer::new(src, 0, "test".into());
    let mut parser = crate::Parser::new(lexer);
    let out = parser.expr();
    assert!(parser.errors().is_empty(), "{src}: {:?}", parser.errors());
    out
}

/// The parse errors of `src`.
fn errors(src: &str) -> usize {
    let lexer = crate::lex::Lexer::new(src, 0, "test".into());
    let mut parser = crate::Parser::new(lexer);
    let _ = parser.expr();
    parser.errors().len()
}

/// `short` and `long` parse to the same tree.
fn same(short: &str, long: &str) {
    let (a, b) = (parse(short), parse(long));
    assert_eq!(a.kind(), b.kind(), "`{short}` should parse as `{long}`:\n{a}\n{b}");
}

// r[verify mimas.query.join-on]
#[test]
fn join_on_pairs_keys() {
    same(
        "query { t\n join u (a == x) }",
        r#"__q_join_on(t, u, ["a"], ["x"], "inner")"#,
    );
    same(
        "query { t\n join g.links (node == src && k == j) \"left\" }",
        r#"__q_join_on(t, g.links, ["node", "k"], ["src", "j"], "left")"#,
    );
    same(
        "query z { t\n join u (a == x) }",
        r#"__z_cons(__z_join_on(__z_in(t), __z_in(u), ["a"], ["x"], "inner"))"#,
    );
    // the same-name form is unchanged
    same("query { t\n join u a }", r#"__q_join(t, u, ["a"], "inner")"#);
    // a glued `(` is still a call
    same("query { t\n join make(1) a }", r#"__q_join(t, make(1), ["a"], "inner")"#);
}

// r[verify mimas.query.join-on]
#[test]
fn join_on_refuses_other_conditions() {
    assert!(errors("query { t\n join u (a > x) }") > 0);
    assert!(errors("query { t\n join u (a == 1) }") > 0);
}

// r[verify mimas.query.aggregate-verb]
#[test]
fn aggregate_alone_is_a_group_with_no_key() {
    same(
        "query { t\n aggregate { n = count() } }",
        "query { t\n group { aggregate { n = count() } } }",
    );
    same(
        "query z { t\n aggregate { n = count() } }",
        "query z { t\n group { aggregate { n = count() } } }",
    );
}

// r[verify mimas.query.verb-ends-derive]
#[test]
fn derive_ends_at_a_verb() {
    same(
        "query { t\n derive n = 1 select n }",
        "query { t\n derive n = 1\n select n }",
    );
    same(
        "query { t\n derive n = a / 3.0 filter n > 1 }",
        "query { t\n derive n = a / 3.0\n filter n > 1 }",
    );
    // a verb word with `=` after it is a column
    same(
        "query { t\n derive n = 1, take = 2 }",
        "query { t\n derive { n = 1, take = 2 } }",
    );
}

// r[verify mimas.dict.empty-literal]
#[test]
fn empty_dict_literal() {
    let e = parse("{:}");
    assert!(matches!(e.kind(), ExprKind::Literal(Literal::Record(f)) if f.is_empty()), "{e}");
    // spaced the same
    let e = parse("{ : }");
    assert!(matches!(e.kind(), ExprKind::Literal(Literal::Record(f)) if f.is_empty()), "{e}");
    // `{}` stays the empty block
    assert!(!matches!(parse("{}").kind(), ExprKind::Literal(_)));
}
