//! Convert trace helpers with a `steps: [S]` output parameter to nested generators.
use parse::{lex::Lexer, *};
use shared::Located;
use std::collections::HashMap;
fn range(e: &impl Located) -> std::ops::Range<usize> {
    e.span().start()..e.span().end()
}
fn steps(e: &Expr) -> bool {
    matches!(e.kind(), ExprKind::Ident(i) if i.lexeme == "steps")
}
struct Calls<'s> {
    src: &'s str,
    helpers: HashMap<String, usize>,
    edits: Vec<(std::ops::Range<usize>, String)>,
}
impl Visitor for Calls<'_> {
    fn expr(&mut self, e: &Expr) {
        let ExprKind::Call(c) = e.kind() else { return };
        if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind()
            && steps(left)
            && matches!(right.kind(), ExprKind::Ident(i) if i.lexeme == "push")
            && c.arguments.len() == 1
        {
            self.edits.push((
                range(e),
                format!("yield {}", &self.src[range(&c.arguments[0].value)]),
            ));
        }
        if let ExprKind::Ident(i) = c.left.kind()
            && let Some(&output) = self.helpers.get(&i.lexeme)
        {
            assert!(steps(&c.arguments[output].value));
            let args = c
                .arguments
                .iter()
                .enumerate()
                .filter(|(n, _)| *n != output)
                .map(|(_, a)| self.src[range(&a.value)].to_string())
                .collect::<Vec<_>>()
                .join(", ");
            self.edits.push((
                range(e),
                format!("for emitted in {}({args}) {{ yield emitted; }}", i.lexeme),
            ));
        }
    }
}
fn main() {
    for path in std::env::args().skip(1) {
        let src = std::fs::read_to_string(&path).unwrap();
        let (ast, errors) = Parser::new(Lexer::new(&src, 0, path.clone())).into_ast();
        assert!(errors.is_empty(), "{errors:?}");
        let mut calls = Calls {
            src: &src,
            helpers: HashMap::new(),
            edits: vec![],
        };
        for stmt in ast.stmts() {
            let StmtKind::Item(item) = stmt.kind() else {
                continue;
            };
            let ItemKind::Function(f) = item.kind() else {
                continue;
            };
            let Some(output) = f
                .parameters
                .iter()
                .position(|p| p.left.to_string() == "steps")
            else {
                continue;
            };
            assert!(f.return_type.is_none());
            let Some(components::Annotation::Array(element, dims)) =
                &f.parameters[output].annotation
            else {
                panic!("expected an array output")
            };
            assert!(dims.is_empty());
            let params = f
                .parameters
                .iter()
                .enumerate()
                .filter(|(n, _)| *n != output)
                .map(|(_, p)| p.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            calls.edits.push((
                item.span().start()..f.body.span().start(),
                format!("subscript {}({params}) -> {element} ", f.name.lexeme),
            ));
            calls.helpers.insert(f.name.lexeme.clone(), output);
        }
        walk_stmts(ast.stmts(), &mut calls);
        calls.edits.sort_by_key(|(r, _)| r.start);
        assert!(calls.edits.windows(2).all(|w| w[0].0.end <= w[1].0.start));
        let mut out = src.clone();
        for (r, s) in calls.edits.into_iter().rev() {
            out.replace_range(r, &s);
        }
        let (_, errors) = Parser::new(Lexer::new(&out, 0, path.clone())).into_ast();
        assert!(errors.is_empty(), "{path}: {errors:?}");
        std::fs::write(&path, out).unwrap();
        println!("{path}: {} helpers", calls.helpers.len());
    }
}
