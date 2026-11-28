//! Rewrite `E::V` to `.V` in match arms whose enum the arms already name:
//! the variant shorthand of `mimas.enum.dot`. Usage: `dotvariant [--dry]
//! FILE...`. A `match` qualifies when every arm that names a variant names
//! one of the same enum `E` (a plain name, not a module path); its arms'
//! heads then become `.V`, which reads as `E::V` because the scrutinee is an
//! `E`. Arms of any other shape (`_`, a binding, a literal) are left as they
//! are, and payload patterns inside an arm are not touched. Other slots
//! (arguments, fields, returns) need the checker's types and are not
//! rewritten here. A file that does not parse is left alone and reported.
use parse::lex::Lexer;
use parse::{Access, Expr, ExprKind, Parser, Visitor, walk_stmts};
use parse::components::{Pat, PatKind};
use shared::Located;

/// `(head start, variant name start)` for every `E::` to drop.
struct Heads<'s> {
    src: &'s str,
    found: Vec<(usize, usize)>,
}

/// The enum a pattern's head names and the head's spans, or `None` when
/// the arm names no variant; `Err(())` when it names one we cannot shorten.
fn head(p: &Pat, out: &mut Vec<(String, usize, usize)>) -> Result<(), ()> {
    match p.kind() {
        PatKind::Or(alts) => alts.iter().try_for_each(|a| head(a, out)),
        PatKind::Variant(h) | PatKind::TupleVariant(h, _) | PatKind::Struct(h, _) => match h.kind() {
            ExprKind::Access(Access::DoubleColon { left, right }) => {
                let ExprKind::Ident(e) = left.kind() else { return Err(()) };
                out.push((
                    e.lexeme.clone(),
                    h.location().span().start(),
                    right.location.span().start(),
                ));
                Ok(())
            }
            // already `.V`/`#V`, or a bare struct name: nothing to shorten
            ExprKind::Atom(_) => Ok(()),
            _ => Err(()),
        },
        _ => Ok(()),
    }
}

impl Visitor for Heads<'_> {
    fn expr(&mut self, e: &Expr) {
        let ExprKind::Match(m) = e.kind() else { return };
        let mut heads = vec![];
        if m.cases.iter().try_for_each(|c| head(c.pat(), &mut heads)).is_err() {
            return;
        }
        let Some((first, ..)) = heads.first() else { return };
        if heads.iter().any(|(e, ..)| e != first) {
            return;
        }
        // only heads the source spells `E::` right there: a match the parser
        // desugared (an `opp`'s or a glyph enum's) points into its declaration
        let spelled = |s: usize, r: usize| {
            self.src.get(s..r).map(|t| t.split_whitespace().collect::<String>()) == Some(format!("{first}::"))
        };
        if heads.iter().any(|(_, s, r)| !spelled(*s, *r)) {
            return;
        }
        self.found.extend(heads.iter().map(|(_, s, r)| (*s, *r)));
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
            eprintln!("{path}: skipped, {} parse diagnostic(s): {}", diags.len(), diags[0]);
            continue;
        }
        let mut heads = Heads { src: &src, found: vec![] };
        walk_stmts(ast.stmts(), &mut heads);
        heads.found.sort();
        heads.found.dedup();
        if heads.found.is_empty() {
            continue;
        }
        total += heads.found.len();
        println!("{path}: {} arm(s)", heads.found.len());
        if dry {
            continue;
        }
        let mut out = String::with_capacity(src.len());
        let mut at = 0;
        for (start, name) in &heads.found {
            out.push_str(&src[at..*start]);
            out.push('.');
            at = *name;
        }
        out.push_str(&src[at..]);
        std::fs::write(path, out).unwrap_or_else(|e| panic!("{path}: {e}"));
    }
    println!("{total} arm(s) in all");
}
