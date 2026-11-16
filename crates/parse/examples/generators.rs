//! Convert trace producers that only append to a local array into generators.
//! Keep collection at the UI boundary so lesson history remains available.
use parse::{lex::Lexer, *};
use shared::Located;
use std::collections::HashSet;

fn span(e: &impl Located) -> std::ops::Range<usize> {
    e.span().start()..e.span().end()
}
fn whole_line(src: &str, mut r: std::ops::Range<usize>) -> std::ops::Range<usize> {
    if src.as_bytes().get(r.end) == Some(&b';') {
        r.end += 1;
    }
    let start = src[..r.start].rfind('\n').map_or(0, |i| i + 1);
    let end = src[r.end..].find('\n').map_or(src.len(), |i| r.end + i + 1);
    if src[start..r.start].trim().is_empty() && src[r.end..end].trim().is_empty() {
        start..end
    } else {
        r
    }
}
fn emit_collection(src: &str, e: &Expr) -> Option<String> {
    match e.kind() {
        ExprKind::Literal(Literal::Array(values)) => Some(
            values
                .iter()
                .map(|v| format!("yield {};", &src[span(v)]))
                .collect::<Vec<_>>()
                .join("\n    "),
        ),
        ExprKind::For(f) => {
            let ExprKind::Collect(c) = f.body.kind() else {
                return None;
            };
            Some(format!(
                "for {} in {} {{ yield {}; }}",
                f.binding,
                &src[span(&f.iterator)],
                &src[span(&c.value)]
            ))
        }
        _ => None,
    }
}
fn named(e: &Expr, name: &str) -> bool {
    matches!(e.kind(), ExprKind::Ident(i) if i.lexeme == name)
}
struct Producer<'s> {
    src: &'s str,
    edits: Vec<(std::ops::Range<usize>, String)>,
    allowed: HashSet<usize>,
    uses: HashSet<usize>,
    pushes: usize,
}
impl Visitor for Producer<'_> {
    fn expr(&mut self, e: &Expr) {
        if named(e, "steps") {
            self.uses.insert(e.span().start());
        }
        match e.kind() {
            ExprKind::Call(c) if c.arguments.len() == 1 => {
                if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind()
                    && named(left, "steps")
                    && named(right, "push")
                {
                    self.allowed.insert(left.span().start());
                    self.edits.push((
                        span(e),
                        format!("yield {}", &self.src[span(&c.arguments[0].value)]),
                    ));
                    self.pushes += 1;
                }
                if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind()
                    && named(left, "steps")
                    && named(right, "extend")
                {
                    if let Some(emit) = emit_collection(self.src, &c.arguments[0].value) {
                        self.allowed.insert(left.span().start());
                        self.edits.push((span(e), emit));
                        self.pushes += 1;
                    }
                }
            }
            ExprKind::Return(r) => {
                if let Some(value) = &r.value
                    && named(value, "steps")
                {
                    self.allowed.insert(value.span().start());
                    self.edits.push((span(e), "return".into()));
                }
            }
            _ => {}
        }
    }
}
fn main() {
    let mut count = 0;
    for path in std::env::args().skip(1) {
        let src = std::fs::read_to_string(&path).unwrap();
        let (ast, errors) = Parser::new(Lexer::new(&src, 0, path.clone())).into_ast();
        if !errors.is_empty() {
            eprintln!("skip {path}: parse errors");
            continue;
        }
        let mut edits = vec![];
        for stmt in ast.stmts() {
            let StmtKind::Item(item) = stmt.kind() else {
                continue;
            };
            let ItemKind::Function(f) = item.kind() else {
                continue;
            };
            if !f.name.lexeme.starts_with("trace") {
                continue;
            }
            let Some(components::Annotation::Array(element, dims)) = &f.return_type else {
                continue;
            };
            if !dims.is_empty() {
                continue;
            }
            let ExprKind::Block(block) = f.body.kind() else {
                continue;
            };
            let Some(tail) = &block.yielded_expr else {
                continue;
            };
            if !named(tail, "steps") {
                continue;
            }
            let Some(decl) = block.body.iter().find(|s| {
                matches!(s.kind(), StmtKind::Let(l)
                if matches!(l.left.kind(), components::PatKind::Ident(i) if i.lexeme == "steps"))
            }) else {
                continue;
            };
            let StmtKind::Let(l) = decl.kind() else {
                unreachable!()
            };
            let Some(initial) = emit_collection(&src, &l.right) else {
                continue;
            };
            let mut producer = Producer {
                src: &src,
                edits: vec![],
                allowed: HashSet::new(),
                uses: HashSet::new(),
                pushes: 0,
            };
            walk_expr(&f.body, &mut producer);
            producer.allowed.insert(tail.span().start());
            if producer.allowed != producer.uses {
                continue;
            }
            let signature = &src[item.span().start()..f.body.span().start()];
            if !signature.starts_with("fn ") {
                continue;
            }
            let args = f
                .parameters
                .iter()
                .map(|p| p.left.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let gen_name = format!("{}_steps", f.name.lexeme);
            let new_signature = signature
                .replacen("fn ", "subscript ", 1)
                .replacen(&f.name.lexeme, &gen_name, 1)
                .replacen(&format!("-> [{}]", element), &format!("-> {element}"), 1);
            if !new_signature.contains(&format!("-> {element}")) {
                continue;
            }
            edits.extend(producer.edits);
            if initial.is_empty() {
                edits.push((whole_line(&src, span(decl)), String::new()));
            } else {
                let mut decl_span = span(decl);
                if src.as_bytes().get(decl_span.end) == Some(&b';') {
                    decl_span.end += 1;
                }
                edits.push((decl_span, initial));
            }
            edits.push((whole_line(&src, span(tail)), String::new()));
            edits.push((item.span().start()..f.body.span().start(), new_signature));
            let at = f.body.span().end();
            edits.push((
                at..at,
                format!(
                    "\n\n{signature}{{\n    for step in {gen_name}({args}) {{ collect step }}\n}}"
                ),
            ));
            count += 1;
            println!("{path}: {} ({} yields)", f.name.lexeme, producer.pushes);
        }
        if edits.is_empty() {
            continue;
        }
        edits.sort_by_key(|(r, _)| r.start);
        assert!(edits.windows(2).all(|w| w[0].0.end <= w[1].0.start));
        let mut out = src;
        for (range, value) in edits.into_iter().rev() {
            out.replace_range(range, &value);
        }
        let (_, errors) = Parser::new(Lexer::new(&out, 0, path.clone())).into_ast();
        assert!(
            errors.is_empty(),
            "{path}: rewrite broke syntax: {errors:?}"
        );
        std::fs::write(&path, out).unwrap();
    }
    println!("{count} producers converted");
}
