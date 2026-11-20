use std::sync::Arc;

use hashbrown::HashMap;
use miette::NamedSource;
use shared::{Located, Result, Span, units::Dim};

use crate::{
    Expr, Ident, Stmt, StmtKind, components::Pat, errors::AlreadyInsideModule, visit::Visitor,
    walk_stmts,
};

/// A collection of statements.
#[derive(Debug, Clone)]
pub struct Ast {
    name: String,
    /// `NamedSource` for the file this ast came from. Cloned into any diagnostics emitted
    /// from post-parse checks (e.g. [`Self::module_name`]) so they render with snippets.
    src: NamedSource<Arc<str>>,
    stmts: Vec<Stmt>,
    /// How many of `stmts` are the prelude's, which rides ahead of the
    /// file's own. Their spans index `prelude.mim`, not this file.
    prelude: usize,
    /// Doc comments without their slashes, keyed by where the token after each starts.
    docs: HashMap<usize, String>,
    /// Dimension of every unit-suffixed number literal (`25kW`), by node. The literal's value is
    /// already in coherent base units; this is what the dimension checker reads.
    quantities: HashMap<NodeId, Dim>,
    /// The unit text as written for each literal in `quantities` (`5ft` -> `"ft"`), kept so
    /// diagnostics can name the unit the user typed rather than its coherent form.
    quantity_units: HashMap<NodeId, String>,
    /// Number literals written with `#` (`#1.10`): exact decimals, a `dec`
    /// wherever they stand (r[impl mimas.dec.exact-literal]).
    exact_literals: std::collections::HashSet<NodeId>,
    /// The axis a direction word put on an expression (`up 50.0px`), by the node the
    /// parser wrapped the operand in. The sign is already in the wrapper.
    directions: HashMap<NodeId, Dim>,
}
impl Ast {
    /// Creates a new Ast with the given statements.
    pub(crate) fn new(
        name: String,
        src: NamedSource<Arc<str>>,
        stmts: Vec<Stmt>,
        prelude: usize,
        docs: HashMap<usize, String>,
        quantities: HashMap<NodeId, Dim>,
        quantity_units: HashMap<NodeId, String>,
        exact_literals: std::collections::HashSet<NodeId>,
        directions: HashMap<NodeId, Dim>,
    ) -> Self {
        Self {
            name,
            src,
            stmts,
            prelude,
            docs,
            quantities,
            quantity_units,
            exact_literals,
            directions,
        }
    }

    /// The number literals written with `#` (`#1.10`), which are `dec`.
    pub fn exact_literals(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.exact_literals.iter().copied()
    }

    /// The axis a direction word tagged the expression `id` with, if one did.
    pub fn direction(&self, id: NodeId) -> Option<Dim> {
        self.directions.get(&id).copied()
    }

    /// The dimension of the unit-suffixed literal `id`, when it is one.
    pub fn quantity(&self, id: NodeId) -> Option<Dim> {
        self.quantities.get(&id).copied()
    }

    /// Every `(node, dim)` pair in `quantities` -- the solver merges these per file so
    /// `table { kwh\n 5kWh }` columns can carry their unit into the schema.
    pub fn quantities(&self) -> impl Iterator<Item = (NodeId, Dim)> + '_ {
        self.quantities.iter().map(|(id, dim)| (*id, *dim))
    }

    /// The unit text as written for the quantity literal `id` (`5ft` -> `"ft"`), if it is one.
    pub fn quantity_unit(&self, id: NodeId) -> Option<&str> {
        self.quantity_units.get(&id).map(String::as_str)
    }

    /// Every `(node, unit-text)` pair in `quantity_units`, for the solver merge.
    pub fn quantity_units(&self) -> impl Iterator<Item = (NodeId, &str)> + '_ {
        self.quantity_units.iter().map(|(id, u)| (*id, u.as_str()))
    }

    /// Consumes the Ast into its inner collection of statements.
    pub fn unpack(self) -> Vec<Stmt> {
        self.stmts
    }

    /// Get a reference to the ast's statements, the prelude's first.
    pub fn stmts(&self) -> &[Stmt] {
        self.stmts.as_ref()
    }

    /// The file's own statements, past the prelude that rides ahead of
    /// them. Use this to slice the file's source by a statement's span:
    /// a prelude statement's span indexes `prelude.mim`, and slicing the
    /// file with it panics or reads the wrong text.
    pub fn own_stmts(&self) -> &[Stmt] {
        &self.stmts[self.prelude.min(self.stmts.len())..]
    }

    /// Get a mutable reference to the ast's statements.
    pub fn stmts_mut(&mut self) -> &mut Vec<Stmt> {
        &mut self.stmts
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn src(&self) -> NamedSource<Arc<str>> {
        self.src.clone()
    }

    /// Appends `other`'s statements, and the unit literals they carry, to
    /// this file: generated code (`expand_laws`) that should resolve names as
    /// this file does while its locations point at its own source.
    pub(crate) fn absorb(&mut self, other: Ast) {
        self.stmts.extend(other.stmts);
        self.quantities.extend(other.quantities);
        self.quantity_units.extend(other.quantity_units);
    }

    /// The innermost node whose span covers `offset` (a byte offset into the source), with that
    /// span. Idents count as nodes, so a name wins over the expr or pat it sits in.
    pub fn node_at(&self, offset: usize) -> Option<(NodeId, Span)> {
        struct Innermost {
            offset: usize,
            found: Option<(NodeId, Span)>,
        }
        impl Innermost {
            fn consider(&mut self, id: NodeId, span: Span) {
                let covers = span.start <= self.offset && self.offset < span.end;
                let tighter = self
                    .found
                    .is_none_or(|(_, found)| span.len() <= found.len());
                if covers && tighter {
                    self.found = Some((id, span));
                }
            }
        }
        impl Visitor for Innermost {
            fn expr(&mut self, expr: &Expr) {
                self.consider(expr.id(), expr.span());
            }

            fn pat(&mut self, pat: &Pat) {
                self.consider(pat.id(), pat.span());
            }

            fn ident(&mut self, ident: &Ident) {
                self.consider(ident.id, ident.span());
            }
        }

        let mut innermost = Innermost {
            offset,
            found: None,
        };
        walk_stmts(&self.stmts, &mut innermost);
        innermost.found
    }

    /// The doc comment above the token starting at `position` (a byte offset into the source),
    /// without its slashes. An item's doc is at the start of its span.
    pub fn docs_for(&self, position: usize) -> Option<&str> {
        self.docs.get(&position).map(String::as_str)
    }

    /// The `//# page: name` boundaries of a spliced source
    /// (`lex::PAGE_MARK`), as `(byte start, page name)` in source order.
    /// The own page's marks carry an empty name; a source without marks
    /// yields the one implicit `(0, "")` entry covering the whole file —
    /// every statement has a page either way. The same line scan
    /// `lex::shadow_table` makes, so the regions agree with per-page
    /// lexing.
    pub fn page_marks(&self) -> Vec<(usize, &str)> {
        let src: &str = self.src.inner();
        let mut marks = vec![(0, "")];
        let mut at = 0;
        let mut in_island = false;
        for line in src.split_inclusive('\n') {
            // a `//# page:` inside a prose island is island text, not a mark
            if !crate::lex::line_in_island(line, &mut in_island)
                && let Some(name) = line.strip_prefix(crate::lex::PAGE_MARK)
            {
                marks.push((at, name.trim_end()));
            }
            at += line.len();
        }
        marks
    }

    pub fn module_name(&self) -> Result<Option<String>> {
        // the prelude rides ahead of the file — a `module` decl is legal
        // first among the file's own statements, not first overall.
        let prelude = self.prelude;
        let module_name = self.stmts.get(prelude).and_then(|stmt| match stmt.kind() {
            StmtKind::Module(module) => Some(if module.name.lexeme == "@" {
                self.name.clone()
            } else {
                module.name.lexeme.clone()
            }),
            _ => None,
        });

        for stmt in self.stmts.iter().skip(prelude + usize::from(module_name.is_some())) {
            if let StmtKind::Module(module) = stmt.kind() {
                Err(AlreadyInsideModule {
                    src: self.src.clone(),
                    at: module.name.location.into(),
                })?
            }
        }

        Ok(module_name)
    }
}

/// A container for both expressions and statements.
///
/// NodeId now includes patterns, which aren't nodes in the sense that these are so...
/// yknow, that's bad
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// Contains a [Stmt].
    Stmt(Stmt),
    /// Contains an [Expr].
    Expr(Expr),
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Default)]
pub struct NodeId(u64);
impl NodeId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// The counter value this node was minted with. Stable across solver phases for the same
    /// ast node, so the checker derives ids from it (e.g. `ParamId` for `<T>` type params).
    pub fn index(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod own_stmts_tests {
    use crate::{lex::Lexer, parser::Parser};
    use shared::Located;

    // The prelude rides ahead of every file, and its spans index
    // prelude.mim. `own_stmts` is the file's statements alone, so each
    // span slices the file's own source.
    #[test]
    fn own_stmts_are_the_files_and_their_spans_slice_it() {
        let src = "let a = 1\nfn f(b: int) -> int { b }\n";
        let (ast, errors) = Parser::new(Lexer::new(src, 0, "t".into())).into_ast();
        assert!(errors.is_empty(), "{errors:?}");
        assert!(ast.stmts().len() > ast.own_stmts().len(), "the prelude rides ahead");
        let texts: Vec<&str> = ast
            .own_stmts()
            .iter()
            .map(|s| {
                let sp = s.location().span;
                src[sp.start()..sp.end()].trim()
            })
            .collect();
        assert_eq!(texts, ["let a = 1", "fn f(b: int) -> int { b }"]);
    }
}
