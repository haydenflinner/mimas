//! Rewrite `Foo { x = x }` to `Foo { x }` in Mimas sources — the field
//! shorthand of `mimas.struct.shorthand`. Usage: `pun [--dry] FILE...`;
//! each file is parsed, every struct-literal field whose value is the
//! field's own name is shortened, and the file is written back. A file
//! that does not parse is left alone and reported.
use parse::lex::Lexer;
use parse::{walk_stmts, Expr, ExprKind, FieldKey, Literal, Parser, Visitor};
use shared::Located;

/// `(key start, value end, name)` for every `name = name` field.
struct Puns<'s> {
    src: &'s str,
    found: Vec<(usize, usize, String)>,
}

impl Visitor for Puns<'_> {
    fn expr(&mut self, e: &Expr) {
        let ExprKind::Literal(Literal::Struct(s)) = e.kind() else { return };
        for (key, value) in &s.fields {
            let FieldKey::Ident(key) = key else { continue };
            let ExprKind::Ident(i) = value.kind() else { continue };
            if i.lexeme != key.lexeme {
                continue;
            }
            let (ks, ke) = (key.location.span().start(), key.location.span().end());
            let (vs, ve) = (value.location().span().start(), value.location().span().end());
            // already short: the value *is* the key token
            if vs < ke || self.src.get(ke..vs).map(str::trim) != Some("=") {
                continue;
            }
            self.found.push((ks, ve, key.lexeme.clone()));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dry = args.iter().any(|a| a == "--dry");
    let mut total = 0;
    for path in args.iter().filter(|a| *a != "--dry") {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let (ast, diags) = Parser::new(Lexer::new(&src, 0, path.clone())).into_ast();
        if !diags.is_empty() {
            eprintln!("{path}: skipped, {} parse diagnostic(s): {} {:?}", diags.len(), diags[0], diags[0]);
            continue;
        }
        let mut puns = Puns { src: &src, found: Vec::new() };
        walk_stmts(ast.stmts(), &mut puns);
        if puns.found.is_empty() {
            continue;
        }
        puns.found.sort();
        let mut out = src.clone();
        for (s, e, name) in puns.found.iter().rev() {
            out.replace_range(*s..*e, name);
        }
        // the rewrite must still parse to the same number of statements
        let (ast2, diags2) = Parser::new(Lexer::new(&out, 0, path.clone())).into_ast();
        assert!(diags2.is_empty() && ast2.stmts().len() == ast.stmts().len(), "{path}: rewrite broke the parse");
        println!("{path}: {}", puns.found.len());
        total += puns.found.len();
        if !dry {
            std::fs::write(path, out).unwrap();
        }
    }
    eprintln!("{total} field(s) shortened{}", if dry { " (dry run)" } else { "" });
}
