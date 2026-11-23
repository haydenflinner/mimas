//! `#Name` atoms: the literal, its payloads, its arms, and the `#` forms
//! that stay what they were (`# ` section comments, `#[attr]`, `#1.10`).

use crate::{Ast, ExprKind, StmtKind, components::PatKind, expr::AtomPayload};

fn parse(src: &str) -> Ast {
    let (ast, errors) = super::utils::parse(src);
    assert!(errors.is_empty(), "{src}: {errors:?}");
    ast
}

fn errs(src: &str) -> usize {
    super::utils::parse(src).1.len()
}

/// The value of `let x = <expr>`, the only statement of `src`.
fn let_value(src: &str) -> crate::Expr {
    let ast = parse(src);
    let StmtKind::Let(l) = ast.own_stmts()[0].kind() else { panic!("{src}: not a let") };
    l.right.clone()
}

// r[verify mimas.atom.literal]
#[test]
fn bare_atom() {
    let e = let_value("let x = #EmptyList;");
    let ExprKind::Atom(a) = e.kind() else { panic!("{e}") };
    assert_eq!(a.name.lexeme, "EmptyList");
    assert_eq!(a.payload, AtomPayload::None);
    assert_eq!(e.to_string(), "#EmptyList");
}

// r[verify mimas.atom.payload]
#[test]
fn positional_and_named_payloads() {
    let e = let_value("let x = #EmptyList(\"the list was empty\");");
    let ExprKind::Atom(a) = e.kind() else { panic!("{e}") };
    assert!(matches!(&a.payload, AtomPayload::Positional(v) if v.len() == 1));

    let e = let_value("let x = #ParseError(line: 3, col: 4);");
    let ExprKind::Atom(a) = e.kind() else { panic!("{e}") };
    let AtomPayload::Named(fields) = &a.payload else { panic!("{e}") };
    let keys: Vec<_> = fields.iter().map(|(k, _)| k.lexeme.as_str()).collect();
    assert_eq!(keys, ["line", "col"]);
    assert_eq!(e.to_string(), "#ParseError(line: 3, col: 4)");

    // `=` separates a name like a record entry's
    let e = let_value("let x = #ParseError(line = 3);");
    let ExprKind::Atom(a) = e.kind() else { panic!("{e}") };
    assert!(matches!(&a.payload, AtomPayload::Named(f) if f.len() == 1));
}

// r[verify mimas.atom.payload]
#[test]
fn mixed_payload_is_an_error() {
    assert!(errs("let x = #P(1, line: 2);") > 0);
}

// r[verify mimas.atom.literal]
#[test]
fn atom_opens_a_statement_and_raises() {
    parse("#Done\n");
    parse("fn f() -> int! { raise #EmptyList }");
    parse("fn f() -> int! { raise #EmptyList(\"why\") }");
    parse("let xs = [#A, #B]; let s = f\"{#A}\";");
}

// r[verify mimas.atom.pattern]
#[test]
fn atom_arms() {
    let ast = parse("let x = match e { #A => 1, #B(n) => n, #C { line } => line, #D | #E => 0, _ => 2 };");
    let StmtKind::Let(l) = ast.own_stmts()[0].kind() else { panic!() };
    let ExprKind::Match(m) = l.right.kind() else { panic!() };
    let kinds: Vec<&'static str> = m
        .cases
        .iter()
        .map(|c| match c.pat().kind() {
            PatKind::Variant(p) if matches!(p.kind(), ExprKind::Atom(_)) => "atom",
            PatKind::TupleVariant(p, _) if matches!(p.kind(), ExprKind::Atom(_)) => "atom()",
            PatKind::Struct(p, _) if matches!(p.kind(), ExprKind::Atom(_)) => "atom{}",
            PatKind::Or(_) => "or",
            PatKind::Ident(_) => "_",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["atom", "atom()", "atom{}", "or", "_"]);
}

// r[verify mimas.atom.literal]
// the `#` forms an atom must not take over
#[test]
fn section_comment_attribute_and_exact_literal_unchanged() {
    let ast = parse("# a section\n## a sub-section\nlet a = 1;");
    assert_eq!(ast.own_stmts().len(), 1);
    let ast = parse("#[test]\nfn t() {}");
    let StmtKind::Item(item) = ast.own_stmts()[0].kind() else { panic!("not an item") };
    assert_eq!(item.attrs()[0].name.lexeme, "test");
    let e = let_value("let d = #1.10;");
    assert!(matches!(e.kind(), ExprKind::Literal(_)), "{e}");
    // a spaced `# Name` is a comment, not an atom
    let ast = parse("let a = 1;\n# Name\n");
    assert_eq!(ast.own_stmts().len(), 1);
}
