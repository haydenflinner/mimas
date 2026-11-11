//! Rewrite `~{…}` dicts to bare braces (`mimas.record.literal`,
//! `mimas.dict.type`): a filled literal `~{ a = 1, b = 2 }` becomes
//! `{ a: 1, b: 2 }`, a dict type `~{float}` becomes `{float}`, and the empty
//! literal `~{}` stays (`{}` is an empty block). Usage: `dicts [--dry] FILE...`.
//! A `.lit`/`.md` file is rewritten inside its ```` ```mimas ```` fences only.
//! A file or block that does not parse is left alone and reported, and every
//! rewrite must re-parse to the same number of statements.
use parse::lex::{Lexer, TokKind};
use parse::{Expr, ExprKind, Literal, Parser, Visitor, walk_stmts};
use shared::Located;

/// `(start, end, replacement)` edits on one source text.
struct Dicts<'s> {
    src: &'s str,
    /// starts of the filled literals' `~{`
    literal_starts: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
}

impl Visitor for Dicts<'_> {
    fn expr(&mut self, e: &Expr) {
        let ExprKind::Literal(Literal::Dictionary(fields)) = e.kind() else {
            return;
        };
        if fields.is_empty() {
            return;
        }
        let start = e.location().span().start();
        if self.src.get(start..start + 2) != Some("~{") {
            return;
        }
        self.literal_starts.push(start);
        self.edits.push((start, start + 2, "{".into()));
        for (key, value) in fields {
            let ke = key.location.span().end();
            let vs = value.location().span().start();
            if self.src.get(ke..vs).map(str::trim) == Some("=") {
                self.edits.push((ke, vs, ": ".into()));
            }
        }
    }
}

/// The rewritten text, or why it was left alone. `None` = nothing to do.
fn rewrite(src: &str, name: &str) -> Result<Option<(String, usize)>, String> {
    let (ast, diags) = Parser::new(Lexer::new(src, 0, name.to_string())).into_ast();
    if !diags.is_empty() {
        return Err(format!("{} parse diagnostic(s): {}", diags.len(), diags[0]));
    }
    let mut v = Dicts {
        src,
        literal_starts: Vec::new(),
        edits: Vec::new(),
    };
    walk_stmts(ast.stmts(), &mut v);
    // every other non-empty `~{` token is a dict type
    let toks: Vec<_> = Lexer::new(src, 0, name.to_string()).collect();
    for (i, t) in toks.iter().enumerate() {
        if t.kind != TokKind::TildeLeftBrace {
            continue;
        }
        let at = t.location.span().start();
        let next = toks.get(i + 1).map(|n| n.kind);
        if v.literal_starts.contains(&at) || next == Some(TokKind::RightBrace) {
            continue;
        }
        // a filled literal the walk didn't reach (it would be `~{ name =`)
        if matches!(next, Some(TokKind::Ident(_)))
            && matches!(toks.get(i + 2).map(|n| n.kind), Some(TokKind::Equal | TokKind::Colon))
        {
            return Err(format!("unvisited dict literal at byte {at}"));
        }
        v.edits.push((at, at + 2, "{".into()));
    }
    if v.edits.is_empty() {
        return Ok(None);
    }
    v.edits.sort();
    let n = v.edits.iter().filter(|e| e.2 == "{").count();
    let mut out = src.to_string();
    for (s, e, r) in v.edits.iter().rev() {
        out.replace_range(*s..*e, r);
    }
    let (ast2, diags2) = Parser::new(Lexer::new(&out, 0, name.to_string())).into_ast();
    if !diags2.is_empty() || ast2.stmts().len() != ast.stmts().len() {
        return Err(format!(
            "rewrite broke the parse: {}",
            diags2.first().map(|d| d.to_string()).unwrap_or_default()
        ));
    }
    Ok(Some((out, n)))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dry = args.iter().any(|a| a == "--dry");
    let mut total = 0;
    for path in args.iter().filter(|a| *a != "--dry") {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let fenced = path.ends_with(".lit") || path.ends_with(".md");
        let (out, n) = if fenced {
            // rewrite each ```mimas block on its own
            let mut out = String::new();
            let mut n = 0;
            let mut rest = src.as_str();
            while let Some(open) = rest.find("```mimas\n") {
                let body_at = open + "```mimas\n".len();
                let Some(close) = rest[body_at..].find("\n```") else { break };
                let body = &rest[body_at..body_at + close + 1];
                out.push_str(&rest[..body_at]);
                match rewrite(body, path) {
                    Ok(Some((b, k))) => {
                        out.push_str(&b);
                        n += k;
                    }
                    Ok(None) => out.push_str(body),
                    Err(e) => {
                        eprintln!("{path}: a block left alone: {e}");
                        out.push_str(body);
                    }
                }
                rest = &rest[body_at + close + 1..];
            }
            out.push_str(rest);
            (out, n)
        } else {
            match rewrite(&src, path) {
                Ok(Some(r)) => r,
                Ok(None) => continue,
                Err(e) => {
                    eprintln!("{path}: skipped, {e}");
                    continue;
                }
            }
        };
        if n == 0 || out == src {
            continue;
        }
        println!("{path}: {n}");
        total += n;
        if !dry {
            std::fs::write(path, out).unwrap();
        }
    }
    eprintln!("{total} dict(s) rewritten{}", if dry { " (dry run)" } else { "" });
}
