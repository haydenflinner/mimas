//! The protocol subset: what a page that will become Rust may use.
//!
//! A protocol page (the sync model's `sync-kit`, `sync-world`) is
//! emitted as a no-deps Rust crate. Mimas shares a list or struct on
//! assignment; Rust moves or clones it. The two agree only when no one
//! can see the sharing, and that is the subset's central rule:
//!
//! **A function mutates only what it made.** `push`, `extend`, an index
//! assignment or a field assignment may target only a local binding
//! that every assignment in the function gives a fresh value: a
//! literal, a comprehension, a struct literal (also with a `..base`
//! spread), an f-string, or an `if`/`match` whose every arm is fresh.
//! A parameter, a loop variable and a call's result are not fresh: a
//! call may hand back its argument. Reassigning a binding is always
//! allowed. Under this rule, sharing and cloning give the same results,
//! so the emitter may clone on every assignment.
//!
//! The other rules keep the emitter's input small:
//!
//! - no float literals (the model counts; dims and floats are out);
//! - no module paths (`game::text`): only `Type::Variant` paths;
//! - methods from a short pure list ([`METHODS`]);
//! - a closure only as the argument of `any` or `all`, and it mutates
//!   nothing;
//! - no `data()`, no records (`{ a = 1 }` without a struct name).
//!
//! Whatever this checker accepts, the emitter must handle: a page that
//! runs in the VM, passes here and cannot be emitted is a checker bug.

use parse::{
    Access, Ast, Expr, ExprKind, Item, ItemKind, Literal, Parser, Stmt, StmtKind, Visitor,
    components::PatKind, lex::Lexer, walk_expr, walk_stmts,
};
use shared::Located;
use std::collections::{BTreeMap, BTreeSet};

/// The methods a protocol page may call. Each has a plain Rust twin.
pub const METHODS: &[&str] = &[
    "len",
    "push",
    "extend",
    "any",
    "all",
    "starts_with",
    "ends_with",
    "contains",
    "split",
];

/// Methods that change their receiver in place.
const MUTATORS: &[&str] = &["push", "extend", "insert", "pop", "remove", "clear", "sort"];

/// One rule broken at one place.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Violation {
    /// 1-based line in the checked source.
    pub line: usize,
    /// The rule's short name: `alias`, `float`, `module`, `method`,
    /// `closure`, `record`, `parse`.
    pub rule: &'static str,
    pub msg: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}: {}", self.line, self.rule, self.msg)
    }
}

/// Check a whole page's source against the subset.
pub fn check_source(src: &str, name: &str) -> Vec<Violation> {
    let (ast, errs) = Parser::new(Lexer::new(src, 0, name.to_string())).into_ast();
    let lines = LineIndex::new(src);
    let mut out: Vec<Violation> = errs
        .iter()
        .map(|e| Violation { line: 0, rule: "parse", msg: format!("{e:?}") })
        .collect();
    out.extend(check_ast(&ast, &lines));
    out.sort();
    out.dedup();
    out
}

/// Byte offset to line.
pub struct LineIndex(Vec<usize>);

impl LineIndex {
    pub fn new(src: &str) -> Self {
        let mut starts = vec![0];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        LineIndex(starts)
    }
    pub fn line(&self, offset: usize) -> usize {
        match self.0.binary_search(&offset) {
            Ok(i) => i + 1,
            Err(i) => i,
        }
    }
}

/// Check the file's own statements (the prelude rides ahead of them
/// and is not the page's).
pub fn check_ast(ast: &Ast, lines: &LineIndex) -> Vec<Violation> {
    let own = ast.own_stmts();
    let mut g = Global { lines, out: Vec::new(), allowed_closures: BTreeSet::new() };
    walk_stmts(own, &mut g);
    let mut out = g.out;
    for s in own {
        if let StmtKind::Item(item) = s.kind() {
            check_item(item, lines, &mut out);
        }
    }
    out
}

fn check_item(item: &Item, lines: &LineIndex, out: &mut Vec<Violation>) {
    if let ItemKind::Function(f) = item.kind() {
        let mut scan = FnScan::default();
        for p in &f.parameters {
            bind_names(p.left.kind(), &mut scan.params);
        }
        walk_expr(&f.body, &mut scan);
        for (name, at) in &scan.mutated {
            let why = if scan.params.contains(name) {
                Some("a parameter")
            } else if scan.loop_vars.contains(name) {
                Some("a loop variable")
            } else if scan.stale.contains(name) {
                Some("a binding that is not always given a fresh value")
            } else if !scan.fresh.contains(name) {
                Some("not a local this function made")
            } else {
                None
            };
            if let Some(why) = why {
                out.push(Violation {
                    line: lines.line(*at),
                    rule: "alias",
                    msg: format!(
                        "`{name}` is mutated here but is {why}; mutate only a fresh local (copy first: `[x for x in {name}]`)"
                    ),
                });
            }
        }
        for at in &scan.through_path {
            out.push(Violation {
                line: lines.line(*at),
                rule: "alias",
                msg: "a mutation through a field or index path; rebuild the outer value instead".into(),
            });
        }
    }
}

fn bind_names(p: &PatKind, into: &mut BTreeSet<String>) {
    match p {
        PatKind::Ident(i) => {
            into.insert(i.lexeme.clone());
        }
        PatKind::Tuple(ps) | PatKind::TupleVariant(_, ps) | PatKind::Or(ps) => {
            for q in ps {
                bind_names(q.kind(), into);
            }
        }
        PatKind::Struct(_, m) => {
            for q in m.values() {
                bind_names(q.kind(), into);
            }
        }
        PatKind::NullBind(q) => bind_names(q.kind(), into),
        _ => {}
    }
}

/// Is `e` a value no one else holds?
fn fresh(e: &Expr) -> bool {
    match e.kind() {
        ExprKind::Literal(_) | ExprKind::FString(_) | ExprKind::Range(_) => true,
        // A comprehension: `[x for x in xs]` is a `For` collecting.
        ExprKind::For(f) => collects(&f.body),
        ExprKind::Grouping(g) => fresh(&g.inner),
        ExprKind::Block(b) => b.yielded_expr.as_ref().is_some_and(fresh),
        ExprKind::If(i) => fresh(&i.main_body) && i.else_expr.as_ref().is_some_and(fresh),
        ExprKind::Match(m) => m.cases.iter().all(|c| fresh(c.body())),
        _ => false,
    }
}

fn collects(e: &Expr) -> bool {
    match e.kind() {
        ExprKind::Collect(_) => true,
        ExprKind::Block(b) => {
            b.yielded_expr.as_ref().is_some_and(collects)
                || b.body.iter().any(|s| matches!(s.kind(), StmtKind::Expr(x) if collects(x)))
        }
        ExprKind::If(i) => collects(&i.main_body) || i.else_expr.as_ref().is_some_and(collects),
        ExprKind::For(f) => collects(&f.body),
        _ => false,
    }
}

/// The root binding a mutation target names: `x` for `x`, and `None`
/// (with `deep`) for `a.b` or `a[i].c`.
fn root(e: &Expr) -> Option<(String, bool)> {
    match e.kind() {
        ExprKind::Ident(i) => Some((i.lexeme.clone(), false)),
        ExprKind::Access(Access::Dot { left, .. }) | ExprKind::Access(Access::Square { left, .. }) => {
            root(left).map(|(n, _)| (n, true))
        }
        ExprKind::Grouping(g) => root(&g.inner),
        _ => None,
    }
}

/// Per-function facts for the aliasing rule, collected flow-insensitively.
#[derive(Default)]
struct FnScan {
    params: BTreeSet<String>,
    loop_vars: BTreeSet<String>,
    /// Bindings given a fresh value somewhere.
    fresh: BTreeSet<String>,
    /// Bindings given a value that is not fresh somewhere.
    stale: BTreeSet<String>,
    /// Binding → offset of its first mutation.
    mutated: BTreeMap<String, usize>,
    /// Mutations through `a.b` or `a[i].c`.
    through_path: Vec<usize>,
}

impl FnScan {
    fn assign(&mut self, name: &str, right: &Expr) {
        if fresh(right) {
            self.fresh.insert(name.to_string());
        } else {
            self.stale.insert(name.to_string());
        }
    }
    fn mutate(&mut self, target: &Expr, at: usize) {
        match root(target) {
            Some((n, false)) => {
                self.mutated.entry(n).or_insert(at);
            }
            Some((n, true)) => {
                // `x[i] = v` and `x.f = v` mutate `x` itself; deeper
                // paths reach a value `x` may share.
                if direct_path(target) {
                    self.mutated.entry(n).or_insert(at);
                } else {
                    self.through_path.push(at);
                }
            }
            None => self.through_path.push(at),
        }
    }
}

fn direct_path(e: &Expr) -> bool {
    match e.kind() {
        ExprKind::Access(Access::Dot { left, .. }) | ExprKind::Access(Access::Square { left, .. }) => {
            matches!(left.kind(), ExprKind::Ident(_))
        }
        _ => false,
    }
}

impl Visitor for FnScan {
    fn stmt(&mut self, s: &Stmt) {
        match s.kind() {
            StmtKind::Let(l) => {
                let mut names = BTreeSet::new();
                bind_names(l.left.kind(), &mut names);
                match l.left.kind() {
                    PatKind::Ident(i) => self.assign(&i.lexeme, &l.right),
                    // A destructured value is a part of another value.
                    _ => self.stale.extend(names),
                }
            }
            StmtKind::Assignment(a) => match a.left.kind() {
                ExprKind::Ident(i) => {
                    let n = i.lexeme.clone();
                    self.assign(&n, &a.right);
                }
                _ => self.mutate(&a.left, s.location().span.start),
            },
            _ => {}
        }
    }
    fn expr(&mut self, e: &Expr) {
        match e.kind() {
            ExprKind::For(f) => bind_names(f.binding.kind(), &mut self.loop_vars),
            ExprKind::Call(c) => {
                if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind() {
                    if let ExprKind::Ident(m) = right.kind() {
                        if MUTATORS.contains(&m.lexeme.as_str()) {
                            self.mutate(left, e.location().span.start);
                        }
                    }
                }
            }
            ExprKind::Closure(c) => {
                for p in &c.parameters {
                    bind_names(p.left.kind(), &mut self.params);
                }
            }
            _ => {}
        }
    }
}

/// Rules that need no scope: floats, module paths, methods, closures,
/// records, `data()`.
struct Global<'a> {
    lines: &'a LineIndex,
    out: Vec<Violation>,
    allowed_closures: BTreeSet<usize>,
}

impl Global<'_> {
    fn flag(&mut self, at: usize, rule: &'static str, msg: String) {
        self.out.push(Violation { line: self.lines.line(at), rule, msg });
    }
}

impl Visitor for Global<'_> {
    fn expr(&mut self, e: &Expr) {
        let at = e.location().span.start;
        match e.kind() {
            ExprKind::Literal(Literal::Float(_)) => {
                self.flag(at, "float", "a float literal; the protocol subset counts in ints".into())
            }
            ExprKind::Literal(Literal::Record(_)) => {
                self.flag(at, "record", "a record literal; name the struct".into())
            }
            ExprKind::Access(Access::DoubleColon { left, right }) => {
                if let ExprKind::Ident(m) = left.kind() {
                    if m.lexeme.chars().next().is_some_and(|c| c.is_lowercase()) {
                        self.flag(
                            at,
                            "module",
                            format!("`{}::{}` reaches a module; a protocol page uses no natives beyond its methods", m.lexeme, right.lexeme),
                        );
                    }
                }
            }
            ExprKind::Call(c) => {
                match c.left.kind() {
                    ExprKind::Access(Access::Dot { right, .. }) => {
                        if let ExprKind::Ident(m) = right.kind() {
                            let name = m.lexeme.as_str();
                            if !METHODS.contains(&name) {
                                self.flag(at, "method", format!("`.{name}()` is outside the subset ({})", METHODS.join(", ")));
                            }
                            if name == "any" || name == "all" {
                                for a in &c.arguments {
                                    if matches!(a.value.kind(), ExprKind::Closure(_)) {
                                        self.allowed_closures.insert(a.value.location().span.start);
                                    }
                                }
                            }
                        }
                    }
                    ExprKind::Ident(f) if f.lexeme == "data" => {
                        self.flag(at, "method", "`data()` reads outside the page".into())
                    }
                    _ => {}
                }
            }
            ExprKind::Closure(c) => {
                if !self.allowed_closures.contains(&at) {
                    self.flag(at, "closure", "a closure outside `any`/`all`".into());
                } else {
                    let mut scan = FnScan::default();
                    walk_expr(&c.body, &mut scan);
                    if !scan.mutated.is_empty() || !scan.through_path.is_empty() {
                        self.flag(at, "closure", "a closure that mutates".into());
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(src: &str) -> Vec<(&'static str, usize)> {
        check_source(src, "t").into_iter().map(|v| (v.rule, v.line)).collect()
    }

    #[test]
    fn a_fresh_local_may_be_mutated() {
        let src = "fn app(xs: [int], x: int) -> [int] {\n    let o: [int] = [y for y in xs];\n    o.push(x);\n    o\n}\n";
        assert_eq!(rules(src), vec![]);
    }

    #[test]
    fn a_parameter_or_call_result_may_not() {
        let src = "fn f(xs: [int]) -> [int] {\n    xs.push(1);\n    xs\n}\nfn g(xs: [int]) -> [int] {\n    let o = f(xs);\n    o.push(2);\n    o\n}\n";
        assert_eq!(rules(src), vec![("alias", 2), ("alias", 7)]);
    }

    #[test]
    fn reassigning_is_always_fine() {
        let src = "fn f(xs: [int]) -> [int] {\n    let o = xs;\n    o = [y for y in o];\n    o\n}\n";
        assert_eq!(rules(src), vec![]);
    }

    #[test]
    fn a_binding_once_stale_stays_stale() {
        let src = "fn f(xs: [int]) -> [int] {\n    let o: [int] = [];\n    o = xs;\n    o.push(1);\n    o\n}\n";
        assert_eq!(rules(src), vec![("alias", 4)]);
    }

    #[test]
    fn floats_modules_methods_and_closures() {
        let src = "fn f(xs: [int]) -> bool {\n    let a = 1.5;\n    game::text(1, 2);\n    let n = xs.sum();\n    let g = |x| x + 1;\n    xs.any(|x| x > 1)\n}\n";
        let got = rules(src);
        for want in [("float", 2), ("module", 3), ("method", 4), ("closure", 5)] {
            assert!(got.contains(&want), "{want:?} not in {got:?}");
        }
        assert!(!got.iter().any(|r| r.1 == 6), "{got:?}");
    }

    /// The sync model's own pages, when this crate sits in the dsa
    /// checkout: they keep the subset.
    #[test]
    fn the_sync_model_pages() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../host/web/lessons");
        for page in ["sync-kit.mimas", "sync-world.mimas"] {
            let Ok(src) = std::fs::read_to_string(dir.join(page)) else {
                eprintln!("{page}: not in a dsa checkout, skipped");
                continue;
            };
            let v = check_source(&src, page);
            for x in &v {
                eprintln!("{page}: {x}");
            }
            assert!(v.is_empty(), "{page} leaves the protocol subset: {v:?}");
        }
    }
}
