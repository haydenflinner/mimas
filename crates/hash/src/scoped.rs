//! Scoped canonicalization — roadmap step 2b/3 (post-solve `CanonicalItem`,
//! implemented without a `mimas-solve` dependency: the AST itself carries
//! enough structure to resolve every bare identifier).
//!
//! `canonical()` (step 2a) is a *token-level* pass: it erases the item's own
//! name but cannot tell a call to a same-source fn from a call to an external
//! one, and it keeps local variable names. `canonical_scoped()` walks the
//! parsed AST with a scope stack and produces a substitution map keyed on
//! token **Location** (never `NodeId` — a process-global counter), then
//! re-lexes the item source and rewrites identifier tokens in place:
//!
//! - `let`/pattern bindings, fn params, `for`/`match`/`if`-/`while`-bindings
//!   and closure params alpha-rename to `@v{n}` by traversal order, so
//!   `fn f(x) { let y = x; y }` and `fn f(q) { let w = q; w }` hash equal.
//! - A bare identifier resolving to a same-source top-level fn emits
//!   `@dep:<hash>` — the callee's *pre-substitution* `hash_item` — so
//!   renaming the callee is free (its own name is already erased) while
//!   editing the callee ripples into the caller's hash.
//! - A reference to the item's own name emits `@self`: a fixed marker, not
//!   the item's hash (a self-hash inside the canonical form is exactly the
//!   self-reference the roadmap warns about). Renaming a recursive fn is
//!   therefore free too.
//! - Mutual recursion terminates because dep hashes are pass-1 hashes
//!   (computed once, no fixed point). Cost: a rename of fn A still moves the
//!   scoped hash of a caller C when some *other* fn between them names A in
//!   its body, since pass-1 hashes bake body tokens. Full transitivity
//!   arrives with step 4's link-by-hash.
//!
//! Namespaces that are not user-fns stay literal (v1): member/method names
//! after `.`/`?`, `::` paths (`str::len`, `std::polars::x`), named args,
//! dict/struct-literal field keys, pattern paths, `use` paths, type names,
//! `self`. Type *parameters* (`fn id<T>`) alpha-rename like locals, since
//! they're binding sites, and `T` annotations resolve to them.
//!
//! `use "page";` (roadmap 4) crosses the page boundary: [`Globals::for_page`]
//! pulls each include's fns into the dep table the way the host splices
//! them — breadth-first, appended after the own source — so a `use`d name
//! emits the same `@dep:<hash>` a same-source callee does. Resolution is
//! positional, matching the interpreter's module rib: the latest decl at or
//! before the use site wins; when none precedes it, the last decl does —
//! an include's `row` shadows the page's own `fn row` only for calls
//! written *above* the own decl. Module-path `use`s (`use a::b::c`) stay
//! `Keep` slots: they shadow without a body we can hash.

use std::collections::{HashMap, HashSet, VecDeque};

use parse::components::{Annotation, Binding, Pat, PatKind};
use parse::lex::{Lexer, TokKind};
use parse::{
    Access, Expr, ExprKind, FStringPart, Function, Ident, ItemKind, Literal, Parser, Stmt,
    StmtKind, Use,
};
use shared::Located;

use crate::{Item, extract, hash_item};

/// `let x = x` resolves the right `x` against the *outer* scope: the binding
/// only exists after the statement. Same for fn-param defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Slot {
    /// A local binding — param, `let` pattern, `for`/`match`/`if`-`while`
    /// binding, closure param, nested fn or const name. Emits `@v{n}`.
    Local(usize),
    /// A `<T>` type parameter — resolves only in annotation position.
    Type(usize),
    /// A top-level `fn` — same-source or `use`d in — emits `@dep:<hash>`.
    Dep(String),
    /// A name that shadows globals without being one (a module-path `use`d
    /// import): resolves so that it *blocks* dep substitution but emits
    /// literally.
    Keep,
}

/// A top-level `fn` declaration the assembled program provides: its pass-1
/// content hash and byte position. Own items carry their real spans; each
/// `use "…"` page's items follow breadth-first, offset past the own source
/// the way the host splices them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Decl {
    hash: String,
    pos: usize,
}

/// The non-local names a fn's bare identifiers can resolve to: top-level
/// fns, own and `use "…"`d (by content hash), plus module-path `use`d
/// imports (kept literal — they shadow without a resolvable body). Built
/// once per page, shared by every item on it.
#[derive(Debug, Clone, Default)]
pub struct Globals {
    /// name → its declarations, in assembled-source order.
    decls: HashMap<String, Vec<Decl>>,
    keeps: HashSet<String>,
    /// `use "…"` names that resolved, in breadth-first (assembled) order.
    includes: Vec<String>,
}

impl Globals {
    /// Top-level fns of `source` with no includes — a page whose `use "…"`
    /// names all fail to resolve lands here too.
    pub fn for_source(source: &str) -> Self {
        Self::for_page(source, |_| None)
    }

    /// `source` plus its `use "…"` includes: `resolve` maps a used page
    /// name to that page's source (the host owns name → source). Includes
    /// splice breadth-first after the own source, each joined by "\n\n" —
    /// the order the assembler emits and the interpreter's module rib
    /// resolves against.
    pub fn for_page(source: &str, mut resolve: impl FnMut(&str) -> Option<String>) -> Self {
        let mut globals = Self::default();
        for item in extract(source) {
            globals.decl(&item.name, item.hash(), item.start);
        }
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        globals.use_items(source, &mut queue, &mut seen);
        let mut base = source.len();
        while let Some(name) = queue.pop_front() {
            let Some(inc) = resolve(&name) else {
                continue;
            };
            for item in extract(&inc) {
                globals.decl(&item.name, item.hash(), base + item.start);
            }
            globals.use_items(&inc, &mut queue, &mut seen);
            globals.includes.push(name);
            base += inc.len() + 2;
        }
        globals
    }

    /// Only module-path `use`d names, no fn deps — for callers testing a
    /// single item in isolation.
    pub fn empty() -> Self {
        Self::default()
    }

    /// `use "…"` names that resolved, in assembled order — the pages a
    /// link-by-hash loader fetches alongside this one.
    pub fn includes(&self) -> &[String] {
        &self.includes
    }

    /// The slot `name` resolves to for a use site at assembled-source byte
    /// `pos` — the interpreter's module-rib rule: the latest declaration
    /// at or before the use wins; when none precedes it, the last
    /// declaration does (a name only includes provide is still visible to
    /// the whole page, and a `use`d decl shadows an own `fn` written below
    /// the call). Module-path `use`d names block dep substitution.
    fn slot_at(&self, name: &str, pos: usize) -> Option<Slot> {
        if self.keeps.contains(name) {
            return Some(Slot::Keep);
        }
        let decls = self.decls.get(name)?;
        let decl = decls.iter().rfind(|d| d.pos <= pos).or_else(|| decls.last())?;
        Some(Slot::Dep(decl.hash.clone()))
    }

    fn decl(&mut self, name: &str, hash: String, pos: usize) {
        self.decls
            .entry(name.to_string())
            .or_default()
            .push(Decl { hash, pos });
    }

    /// `use` items in `source`: `use a::b::c`/`use a::{…}` names keep
    /// (they shadow but have no hashable body); `use a::*` binds unknown
    /// names — nothing to shadow with; `use "page"` enqueues for the
    /// `for_page` BFS, deduped case-insensitively like the host's
    /// includeNames. Unparseable source contributes nothing.
    fn use_items(
        &mut self,
        source: &str,
        queue: &mut VecDeque<String>,
        seen: &mut HashSet<String>,
    ) {
        let Ok(ast) = Parser::new(Lexer::new(source, 0, "globals".into())).try_into_ast() else {
            return;
        };
        for stmt in ast.stmts() {
            let StmtKind::Item(item) = stmt.kind() else {
                continue;
            };
            let ItemKind::Use(us) = item.kind() else {
                continue;
            };
            match us {
                Use::Singular(_, item) => self.keep(&item.lexeme),
                Use::Multi(_, items) => {
                    for item in items {
                        self.keep(&item.lexeme);
                    }
                }
                Use::All(_) => {}
                Use::Host(name) => {
                    if seen.insert(name.to_lowercase()) {
                        queue.push_back(name.clone());
                    }
                }
            }
        }
    }

    fn keep(&mut self, name: &str) {
        self.keeps.insert(name.to_string());
    }
}

/// `fn` name → its pass-1 content hash, for every extracted item. The dep
/// table scoped canonicalization substitutes with — pre-substitution, so
/// cycles need no fixed point (see the module docs for the cost).
pub fn dep_table(items: &[Item]) -> HashMap<String, String> {
    items
        .iter()
        .map(|item| (item.name.clone(), item.hash()))
        .collect()
}

/// A relocation site in the scoped canonical form — one `@dep:` or `@self`
/// marker a link-by-hash loader must patch. Sites collect in traversal
/// order, which is source order: the same order the markers appear in the
/// canonical text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reloc {
    /// A `@dep:<hash>` site — the identifier `name` (byte span `start..end`
    /// in the item source) was substituted by the callee's pass-1 content
    /// hash.
    Dep {
        name: String,
        hash: String,
        start: usize,
        end: usize,
    },
    /// A `@self` site — a recursive reference to the item being hashed.
    /// Links to the item's own address once it stores by hash.
    SelfRef { name: String, start: usize, end: usize },
}

impl Reloc {
    /// The dep-edge target hash — `Some` at `@dep:` sites, `None` at
    /// `@self`.
    pub fn hash(&self) -> Option<&str> {
        match self {
            Reloc::Dep { hash, .. } => Some(hash),
            Reloc::SelfRef { .. } => None,
        }
    }

    /// The name the programmer wrote at this site.
    pub fn name(&self) -> &str {
        match self {
            Reloc::Dep { name, .. } | Reloc::SelfRef { name, .. } => name,
        }
    }

    /// The byte span of the substituted identifier in the item source.
    pub fn span(&self) -> (usize, usize) {
        match self {
            Reloc::Dep { start, end, .. } | Reloc::SelfRef { start, end, .. } => (*start, *end),
        }
    }
}

/// One entry of a page's link-by-hash manifest: an item's address plus the
/// ordered reloc sites a loader patches when it instantiates the blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The item's written name (a label, not an address — renames don't
    /// move `hash`).
    pub name: String,
    /// The item's scoped content hash — the address it stores under.
    pub hash: String,
    /// The item's pass-1 (token-level) hash. Dep edges target *this*, not
    /// `hash` — `@dep:` markers substitute pass-1 hashes so cycles need no
    /// fixed point — so a loader resolves an edge by finding the entry
    /// whose `token_hash` matches, then loads `hash`.
    pub token_hash: String,
    /// Every `@dep:`/`@self` site, in canonical order. Empty when the
    /// item's source can't be scope-parsed (`hash` then carries the
    /// token-level fallback, same as [`hash_scoped`]).
    pub relocs: Vec<Reloc>,
}

impl Manifest {
    /// The dep-edge target hashes (`@dep:` sites only), in order,
    /// duplicates kept per site. Each target equals the *`token_hash`* of
    /// the entry it links to — see [`Manifest::token_hash`].
    pub fn dep_hashes(&self) -> impl Iterator<Item = &str> {
        self.relocs.iter().filter_map(Reloc::hash)
    }
}

/// Substitution map + scope stack for one item's AST walk.
struct Scoped<'s> {
    source: &'s str,
    /// The page's non-local environment — dep decls and `Keep` names.
    globals: &'s Globals,
    /// `source`'s byte offset in the assembled program: `use`d names
    /// resolve by use-site position, which is `base` + the ident's
    /// item-relative span.
    base: usize,
    /// The item's own name — resolves to `@self`, never a dep edge.
    own_name: String,
    scopes: Vec<HashMap<String, Slot>>,
    next: usize,
    /// `(start, end)` byte span → replacement token text. Keyed on Location:
    /// NodeIds are a process-global counter and can't key anything stable.
    subs: HashMap<(usize, usize), String>,
    /// The `@dep:`/`@self` sites, in the order the walk records them.
    relocs: Vec<Reloc>,
    /// Inside member names, `::` paths, struct-literal/pattern paths: every
    /// ident stays literal.
    keep_names: bool,
}

impl<'s> Scoped<'s> {
    fn new(source: &'s str, globals: &'s Globals, base: usize) -> Self {
        Self {
            source,
            globals,
            base,
            own_name: String::new(),
            scopes: vec![],
            next: 0,
            subs: HashMap::new(),
            relocs: Vec::new(),
            keep_names: false,
        }
    }

    /// Local scopes only — params, `let`/`for`/`match` bindings, nested
    /// items, block-level `use`s. Globals resolve separately (see
    /// [`Scoped::reference`]): the item's own name, then [`Globals`] deps.
    fn local(&self, name: &str) -> Option<&Slot> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    /// Record a replacement at `ident`'s token span — but only when the
    /// source there actually reads this name. The parser synthesizes idents
    /// at borrowed spans (`__q_*` calls, `±` lowering); trusting a span
    /// without checking its text would corrupt whatever token sits there.
    /// Returns whether a substitution was recorded — reloc sites list only
    /// spans that truly carry a marker in the canonical form.
    fn record(&mut self, ident: &Ident, text: String) -> bool {
        let span = ident.location.span;
        if span.is_synthetic() {
            return false;
        }
        if self.source.get(span.start..span.end) != Some(ident.lexeme.as_str()) {
            return false;
        }
        self.subs.insert((span.start, span.end), text);
        true
    }

    /// A binding site: assign the next alpha position, record `@v{n}` at the
    /// ident, and shadow any outer same-name binding. `self`/`Self` are
    /// keywords at use sites (`self.x` is `Access::Identity`) — renaming the
    /// binding would split name from uses, so they stay literal.
    fn bind(&mut self, ident: &Ident) {
        if ident.lexeme == "self" || ident.lexeme == "Self" {
            return;
        }
        let n = self.next;
        self.next += 1;
        self.record(ident, format!("@v{n}"));
        self.scope_mut()
            .insert(ident.lexeme.clone(), Slot::Local(n));
    }

    fn bind_ty(&mut self, ident: &Ident) {
        let n = self.next;
        self.next += 1;
        self.record(ident, format!("@v{n}"));
        self.scope_mut().insert(ident.lexeme.clone(), Slot::Type(n));
    }

    /// A bare-identifier *reference* in value position. Locals win over
    /// the item's own name, which wins over globals — `let f = ...; f()`
    /// shadows both a top-level `f` and a recursive self-call.
    fn reference(&mut self, ident: &Ident) {
        if self.keep_names {
            return;
        }
        if let Some(slot) = self.local(&ident.lexeme).cloned() {
            // Type params aren't values; Keep names stay literal.
            if let Slot::Local(n) = slot {
                self.record(ident, format!("@v{n}"));
            }
            return;
        }
        let span = ident.location.span;
        if ident.lexeme == self.own_name {
            if self.record(ident, "@self".to_string()) {
                self.relocs.push(Reloc::SelfRef {
                    name: ident.lexeme.clone(),
                    start: span.start,
                    end: span.end,
                });
            }
            return;
        }
        let Some(Slot::Dep(hash)) = self.globals.slot_at(&ident.lexeme, self.base + span.start)
        else {
            return;
        };
        if self.record(ident, format!("@dep:{hash}")) {
            self.relocs.push(Reloc::Dep {
                name: ident.lexeme.clone(),
                hash,
                start: span.start,
                end: span.end,
            });
        }
    }

    /// An identifier in annotation position: only a type param resolves.
    fn ty_reference(&mut self, ident: &Ident) {
        if self.keep_names {
            return;
        }
        if let Some(Slot::Type(n)) = self.local(&ident.lexeme) {
            let text = format!("@v{n}");
            self.record(ident, text);
        }
    }

    fn scope_mut(&mut self) -> &mut HashMap<String, Slot> {
        self.scopes.last_mut().expect("the fn's own frame is pushed first")
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    /// Walk an expr whose idents are all names, not variables: member-path
    /// spines (`a::b`), struct-literal and pattern paths.
    fn name_expr(&mut self, expr: &Expr) {
        let keep = std::mem::replace(&mut self.keep_names, true);
        self.expr(expr);
        self.keep_names = keep;
    }

    /// A fn item's contents: its own name is bound by the caller (globals
    /// for the top item, a local `v` for a nested one); this pushes the
    /// scope params, type params and the body share.
    fn function(&mut self, f: &Function) {
        self.push();
        for param in &f.type_params {
            self.bind_ty(param);
        }
        for parameter in &f.parameters {
            self.binding(parameter);
        }
        if let Some(return_type) = &f.return_type {
            self.annotation(return_type);
        }
        self.expr(&f.body);
        self.pop();
    }

    /// `pat [: ann] [= default]` — annotation and default resolve before the
    /// binding exists (a default sees earlier params, never itself).
    fn binding(&mut self, binding: &Binding) {
        if let Some(annotation) = &binding.annotation {
            self.annotation(annotation);
        }
        if let Some(right) = &binding.right {
            self.expr(right);
        }
        self.pat_bind(&binding.left);
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt.kind() {
            StmtKind::Let(l) => {
                if let Some(annotation) = &l.annotation {
                    self.annotation(annotation);
                }
                self.expr(&l.right);
                // `let x = v else { .. }` — the else branch runs when the
                // pattern fails, so the binding is not in scope there.
                if let Some(else_branch) = &l.else_branch {
                    self.expr(else_branch);
                }
                self.pat_bind(&l.left);
            }
            StmtKind::Assignment(a) => {
                self.expr(&a.left);
                self.expr(&a.right);
            }
            StmtKind::Expr(e) => self.expr(e),
            StmtKind::Item(item) => self.nested_item(item),
            // `module name;` is a file header, not a binding.
            StmtKind::Module(_) => {}
        }
    }

    /// Items inside blocks. Nested fns and consts bind their name in the
    /// enclosing scope — recursion falls out of ordinary scope resolution.
    /// Other item kinds (struct/enum/pact/impl/tests) keep every ident
    /// literal for v1; `use` shadows later lookups.
    fn nested_item(&mut self, item: &parse::Item) {
        match item.kind() {
            ItemKind::Function(f) => {
                self.bind(&f.name);
                self.function(f);
            }
            ItemKind::Const(con) => {
                if let Some(annotation) = &con.annotation {
                    self.annotation(annotation);
                }
                self.expr(&con.right);
                self.pat_bind(&con.left);
            }
            ItemKind::Use(us) => {
                let imported: Vec<String> = match us {
                    Use::Singular(_, item) => vec![item.lexeme.clone()],
                    Use::Multi(_, items) => items.iter().map(|i| i.lexeme.clone()).collect(),
                    Use::All(_) | Use::Host(_) => vec![],
                };
                for name in imported {
                    self.scope_mut().insert(name, Slot::Keep);
                }
            }
            _ => {}
        }
    }

    /// Binding occurrences: `PatKind::Ident` binds; struct/variant paths are
    /// type names — kept literal, never resolved against locals (a `let
    /// Foo = ..` must not rename `match x { Foo { a } => .. }`'s `Foo`).
    /// Literal-pattern elements are references, matching `references()`.
    fn pat_bind(&mut self, pat: &Pat) {
        match pat.kind() {
            PatKind::Ident(ident) => self.bind(ident),
            PatKind::Tuple(pats) | PatKind::Or(pats) => {
                for pat in pats {
                    self.pat_bind(pat);
                }
            }
            PatKind::Struct(path, fields) => {
                self.name_expr(path);
                // `fields` is a name→pat HashMap: sort by source position —
                // iteration order can't key anything stable.
                let mut fields: Vec<&Pat> = fields.values().collect();
                fields.sort_by_key(|pat| pat.span().start);
                for field in fields {
                    self.pat_bind(field);
                }
            }
            PatKind::TupleVariant(path, pats) => {
                self.name_expr(path);
                for pat in pats {
                    self.pat_bind(pat);
                }
            }
            PatKind::Variant(path) => self.name_expr(path),
            PatKind::NullBind(pat) => self.pat_bind(pat),
            PatKind::Literal(literal) => self.literal(literal),
            PatKind::Poison(_) => {}
        }
    }

    fn annotation(&mut self, annotation: &Annotation) {
        match annotation {
            Annotation::Ty(ident) => self.ty_reference(ident),
            Annotation::Option(inner)
            | Annotation::Result(inner)
            | Annotation::Array(inner)
            | Annotation::Dictionary(inner) => self.annotation(inner),
            Annotation::Tuple(members) => {
                for member in members {
                    self.annotation(member);
                }
            }
            Annotation::Function(params, ret) => {
                for param in params {
                    self.annotation(param);
                }
                self.annotation(ret);
            }
            // `Pair<A, B>`: the head is a path (literal); args recurse —
            // `Vec<T>` still finds the bound `T`.
            Annotation::Applied(_, args) => {
                for arg in args {
                    self.annotation(arg);
                }
            }
            // Path/Bounds segments are names; Quantity parts are units.
            Annotation::Path(_)
            | Annotation::Bounds(_)
            | Annotation::Quantity(_)
            | Annotation::Unit
            | Annotation::Kw(_)
            | Annotation::Poison(_) => {}
        }
    }

    fn literal(&mut self, literal: &Literal) {
        match literal {
            Literal::Array(exprs) | Literal::Tuple(exprs) => {
                for expr in exprs {
                    self.expr(expr);
                }
            }
            Literal::Dictionary(fields) => {
                // Keys are member names; only values resolve.
                for (_, value) in fields {
                    self.expr(value);
                }
            }
            Literal::Struct(struc) => {
                self.name_expr(&struc.name);
                for (_, value) in &struc.fields {
                    self.expr(value);
                }
            }
            _ => {}
        }
    }

    fn expr(&mut self, expr: &Expr) {
        match expr.kind() {
            ExprKind::Ident(ident) => self.reference(ident),
            ExprKind::Call(call) => {
                self.expr(&call.left);
                for argument in &call.arguments {
                    // `f(name = v)` — `name` is the callee's parameter name.
                    self.expr(&argument.value);
                }
            }
            ExprKind::Access(access) => match access {
                // `self.x` — the member is a name.
                Access::Identity { .. } => {}
                Access::Dot { left, .. } => {
                    // `.right` is an Ident or int member name — a method
                    // called `square` is not the fn `square`.
                    self.expr(left);
                }
                // `str::len`, `std::polars::x` — module paths stay literal.
                Access::DoubleColon { .. } => {}
                Access::Square { left, key, .. } => {
                    self.expr(left);
                    self.expr(key);
                }
            },
            ExprKind::Block(block) => {
                self.push();
                for stmt in &block.body {
                    self.stmt(stmt);
                }
                if let Some(yielded) = &block.yielded_expr {
                    self.expr(yielded);
                }
                self.pop();
            }
            ExprKind::Closure(closure) => {
                self.push();
                for parameter in &closure.parameters {
                    self.binding(parameter);
                }
                if let Some(return_type) = &closure.return_type {
                    self.annotation(return_type);
                }
                self.expr(&closure.body);
                self.pop();
            }
            ExprKind::For(fo) => {
                self.expr(&fo.iterator);
                self.push();
                self.pat_bind(&fo.binding);
                self.expr(&fo.body);
                self.pop();
            }
            ExprKind::If(i) => {
                self.expr(&i.condition);
                match &i.binding {
                    // `if let` — the pat scopes over the main body only.
                    Some(pat) => {
                        self.push();
                        self.pat_bind(pat);
                        self.expr(&i.main_body);
                        self.pop();
                    }
                    None => self.expr(&i.main_body),
                }
                if let Some(else_expr) = &i.else_expr {
                    self.expr(else_expr);
                }
            }
            ExprKind::While(w) => {
                self.expr(&w.header);
                match &w.binding {
                    Some(pat) => {
                        self.push();
                        self.pat_bind(pat);
                        self.expr(&w.body);
                        self.pop();
                    }
                    None => self.expr(&w.body),
                }
            }
            ExprKind::Match(m) => {
                self.expr(&m.identity);
                for case in &m.cases {
                    self.push();
                    self.pat_bind(case.pat());
                    if let Some(guard) = case.guard() {
                        self.expr(guard);
                    }
                    self.expr(case.body());
                    self.pop();
                }
            }
            ExprKind::Literal(literal) => self.literal(literal),
            ExprKind::FString(f) => {
                for part in &f.parts {
                    if let FStringPart::Expr(expr) = part {
                        self.expr(expr);
                    }
                }
            }
            ExprKind::Absolve(a) => {
                self.expr(&a.left);
                self.expr(&a.handler);
            }
            ExprKind::Equality(e) => {
                self.expr(&e.left);
                self.expr(&e.right);
            }
            ExprKind::Evaluation(e) => {
                self.expr(&e.left);
                self.expr(&e.right);
            }
            ExprKind::Logical(e) => {
                self.expr(&e.left);
                self.expr(&e.right);
            }
            ExprKind::Coalescence(c) => {
                self.expr(&c.left);
                self.expr(&c.right);
            }
            ExprKind::In(i) => {
                self.expr(&i.left);
                self.expr(&i.right);
            }
            ExprKind::Range(r) => {
                self.expr(&r.start);
                self.expr(&r.end);
            }
            ExprKind::Unary(u) => self.expr(&u.right),
            ExprKind::Unwrap(u) => self.expr(&u.expr),
            ExprKind::Demote(d) => self.expr(&d.expr),
            ExprKind::Grouping(g) => self.expr(&g.inner),
            ExprKind::Loop(l) => self.expr(&l.body),
            ExprKind::Collect(c) => self.expr(&c.value),
            ExprKind::Raise(r) => self.expr(&r.value),
            ExprKind::Break(b) => {
                if let Some(value) = &b.value {
                    self.expr(value);
                }
            }
            ExprKind::Return(r) => {
                if let Some(value) = &r.value {
                    self.expr(value);
                }
            }
            ExprKind::Continue(_) | ExprKind::Poison(_) => {}
        }
    }
}

/// The scope walk over one fn item's source: bind the item's own name to
/// `@self` and walk the fn body, recording substitutions and reloc sites.
/// `base` is the item's byte offset in the assembled program — dep
/// resolution is positional (see [`Globals::slot_at`]). `None` when the
/// source doesn't parse or holds no fn — the gate [`canonical_scoped`]
/// applies.
fn scoped_pass<'s>(item_source: &'s str, globals: &'s Globals, base: usize) -> Option<Scoped<'s>> {
    let ast = Parser::new(Lexer::new(item_source, 0, "scoped".into()))
        .try_into_ast()
        .ok()?;
    let function = ast.stmts().iter().find_map(|stmt| match stmt.kind() {
        StmtKind::Item(item) => item.as_function(),
        _ => None,
    })?;

    let mut st = Scoped::new(item_source, globals, base);
    st.own_name = function.name.lexeme.clone();
    st.function(function);
    Some(st)
}

/// Re-lex the item and swap every recorded ident for its canonical token —
/// the emitting half of [`canonical_scoped`]. `None` on any invalid token
/// or lexer error.
fn render_scoped(item_source: &str, st: &Scoped) -> Option<String> {
    let mut out = String::new();
    let mut lexer = Lexer::new(item_source, 0, "scoped".into());
    let mut name_erased = false;
    let mut after_fn = false;
    for tok in &mut lexer {
        if matches!(tok.kind, TokKind::Invalid(_)) {
            return None;
        }
        if tok.kind.is_comment() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        let span = tok.location.span();
        match tok.kind {
            TokKind::Float(v) => out.push_str(&crate::float_text(v)),
            TokKind::Ident(lexeme) => {
                if let Some(replacement) = st.subs.get(&(span.start(), span.end())) {
                    out.push_str(replacement);
                } else if !name_erased && after_fn {
                    out.push('_');
                    name_erased = true;
                } else {
                    out.push_str(lexeme);
                }
            }
            TokKind::FString(inner) => {
                out.push_str(&fstring_text(item_source, span.end(), inner, &st.subs));
            }
            kind => out.push_str(&kind.to_string()),
        }
        after_fn = matches!(tok.kind, TokKind::Fn);
    }
    if !lexer.take_errors().is_empty() {
        return None;
    }
    Some(out)
}

/// The scoped canonical form of one fn item's source: same layout as
/// [`crate::canonical`] (trivia dropped, tokens space-joined, own name
/// erased) plus `@v{n}` locals, `@dep:<hash>` same-source fn refs and
/// `@self` recursion. `None` when the source doesn't parse or holds no fn.
///
/// `globals` names what the body's free identifiers resolve to — see
/// [`Globals::for_source`] and [`Globals::for_page`]. The item is treated
/// as opening the assembled source (base 0); callers hashing one item of
/// a page want [`Item::scoped_canonical`], which positions it correctly.
pub fn canonical_scoped(item_source: &str, globals: &Globals) -> Option<String> {
    let st = scoped_pass(item_source, globals, 0)?;
    render_scoped(item_source, &st)
}

/// [`canonical_scoped`] with the item's real byte offset in the assembled
/// source — [`Item`] methods route through this so `use`d-name resolution
/// sees the true use-site position.
pub(crate) fn canonical_scoped_at(
    item_source: &str,
    globals: &Globals,
    base: usize,
) -> Option<String> {
    let st = scoped_pass(item_source, globals, base)?;
    render_scoped(item_source, &st)
}

/// [`deps`] with the item's real byte offset — see [`canonical_scoped_at`].
pub(crate) fn deps_at(item_source: &str, globals: &Globals, base: usize) -> Option<Vec<Reloc>> {
    scoped_pass(item_source, globals, base).map(|st| st.relocs)
}

/// The relocation sites of one fn item's source — every `@dep:`/`@self`
/// marker the scoped canonical form carries, in the order they appear in
/// it. This *is* the linking table: a link-by-hash loader reads this list
/// to patch callee references, and [`Manifest::dep_hashes`] derives the
/// edge set from it. `None` under the same gate as [`canonical_scoped`]
/// (source doesn't parse or holds no fn). Same base-0 caveat — page
/// callers want [`Item::relocs`].
pub fn deps(item_source: &str, globals: &Globals) -> Option<Vec<Reloc>> {
    scoped_pass(item_source, globals, 0).map(|st| st.relocs)
}

/// The link-by-hash manifest of a whole page against `globals`: for every
/// extracted fn, in extraction order, its scoped content hash, its token
/// hash and its ordered reloc sites — `{ name → (hash, [deps]) }` plus
/// the edge-target key. One [`Globals`] build covers all items, so
/// inter-fn and `use`-include edges resolve regardless of declaration
/// order. Items whose source can't be scope-parsed keep their token-level
/// fallback hash (the same one [`hash_scoped`] reports) and an empty
/// reloc list.
pub fn manifest_with(source: &str, globals: &Globals) -> Vec<Manifest> {
    extract(source)
        .iter()
        .map(|item| {
            let token_hash = item.hash();
            let (hash, relocs) = match scoped_pass(&item.source, globals, item.start) {
                Some(st) => {
                    // Same hash `hash_scoped` would report: blake3 of the
                    // rendered canonical form, token-hash fallback otherwise.
                    let hash = match render_scoped(&item.source, &st) {
                        Some(canonical) => {
                            blake3::hash(canonical.as_bytes()).to_hex().to_string()
                        }
                        None => token_hash.clone(),
                    };
                    (hash, st.relocs)
                }
                None => (token_hash.clone(), Vec::new()),
            };
            Manifest {
                name: item.name.clone(),
                hash,
                token_hash,
                relocs,
            }
        })
        .collect()
}

/// [`manifest_with`] over `source`'s own items alone — no `use "…"`
/// includes. Pages with includes want [`Globals::for_page`] +
/// `manifest_with`.
pub fn manifest(source: &str) -> Vec<Manifest> {
    manifest_with(source, &Globals::for_source(source))
}

/// blake3 of the scoped canonical form. Falls back to [`hash_item`] when
/// the source doesn't parse — the token-level form is still deterministic.
/// Same base-0 caveat as [`canonical_scoped`]; page callers want
/// [`Item::scoped_hash`].
pub fn hash_scoped(item_source: &str, globals: &Globals) -> String {
    match canonical_scoped(item_source, globals) {
        Some(canonical) => blake3::hash(canonical.as_bytes()).to_hex().to_string(),
        None => hash_item(item_source),
    }
}

/// `(name, scoped hash)` for every fn in `source`, in extraction order.
/// One [`Globals`] build covers all items, so inter-fn deps resolve in a
/// single pass regardless of declaration order.
pub fn scoped_hashes_with(source: &str, globals: &Globals) -> Vec<(String, String)> {
    extract(source)
        .iter()
        .map(|item| (item.name.clone(), item.scoped_hash(globals)))
        .collect()
}

/// [`scoped_hashes_with`] over `source`'s own items alone — no `use "…"`
/// includes.
pub fn scoped_hashes(source: &str) -> Vec<(String, String)> {
    scoped_hashes_with(source, &Globals::for_source(source))
}

/// Rewrite the interior of an `f"..."` token when substitutions land inside
/// it: interpolated idents have real (offset) locations, but the outer
/// token is one lexer unit, so byte-splice replacements into its text.
fn fstring_text(
    source: &str,
    end: usize,
    inner: &str,
    subs: &HashMap<(usize, usize), String>,
) -> String {
    // `inner` is the raw text between the quotes: it ends just before the
    // closing `"`. Trust nothing else about the token's shape.
    let close = end.saturating_sub(1);
    let open = close.saturating_sub(inner.len());
    if source.get(open..close) != Some(inner) {
        // Token doesn't look like `f"inner"` — keep it as lexed.
        return format!("f\"{inner}\"");
    }
    let mut edits: Vec<((usize, usize), &String)> = subs
        .iter()
        .filter(|((start, end), _)| *start >= open && *end <= close)
        .map(|(key, text)| (*key, text))
        .collect();
    if edits.is_empty() {
        return format!("f\"{inner}\"");
    }
    edits.sort_by_key(|((start, _), _)| *start);
    let mut rewritten = String::with_capacity(inner.len());
    let mut cursor = open;
    for ((start, end), text) in edits {
        rewritten.push_str(&source[cursor..start]);
        rewritten.push_str(text);
        cursor = end;
    }
    rewritten.push_str(&source[cursor..close]);
    format!("f\"{rewritten}\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_of<'a>(hashes: &'a [(String, String)], name: &str) -> &'a str {
        hashes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, h)| h.as_str())
            .unwrap_or_else(|| panic!("no scoped hash for {name}"))
    }

    fn canon_of(source: &str, name: &str) -> String {
        let items = extract(source);
        let globals = Globals::for_source(source);
        let item = items.iter().find(|i| i.name == name).unwrap();
        item.scoped_canonical(&globals).unwrap()
    }

    /// THE SPEC: renaming a callee (consistently, in its callers) leaves the
    /// caller's scoped hash untouched — while the token-level hash moves.
    #[test]
    fn renaming_a_callee_leaves_the_callers_scoped_hash() {
        let a = "fn square(n: int) -> int { n * n }\nfn twice(n: int) -> int { square(n) + square(n) }\n";
        let b = "fn sq(n: int) -> int { n * n }\nfn twice(n: int) -> int { sq(n) + sq(n) }\n";
        assert_eq!(hash_of(&scoped_hashes(a), "twice"), hash_of(&scoped_hashes(b), "twice"));
        // …and the callee itself is rename-free too.
        assert_eq!(
            hash_of(&scoped_hashes(a), "square"),
            hash_of(&scoped_hashes(b), "sq")
        );
        // Contrast: the token-level hash still moves on a rename.
        assert_ne!(hash_item("fn twice(n: int) { square(n) }"), hash_item("fn twice(n: int) { sq(n) }"));
        // The dep really was substituted — no `square`/`sq` text survives.
        let canon = canon_of(a, "twice");
        assert!(!canon.contains("square"), "{canon}");
        assert!(canon.contains("@dep:"), "{canon}");
    }

    /// `let` bindings and params alpha-rename by position: renamed locals
    /// and alpha-equal bodies hash identically.
    #[test]
    fn renaming_locals_and_params_is_free() {
        let a = "fn f(x: int) -> int { let y = x * 2; y }\n";
        let b = "fn f(q: int) -> int { let w = q * 2; w }\n";
        assert_eq!(hash_of(&scoped_hashes(a), "f"), hash_of(&scoped_hashes(b), "f"));
        let canon = canon_of(a, "f");
        assert!(canon.contains("@v0"), "{canon}");
        assert!(canon.contains("@v1"), "{canon}");
        assert!(!canon.contains('x') && !canon.contains('y'), "{canon}");
        // Alpha-equal multi-param bodies.
        let c = "fn f(a: int, b: int) -> int { a + b }\n";
        let d = "fn f(x: int, y: int) -> int { x + y }\n";
        assert_eq!(hash_of(&scoped_hashes(c), "f"), hash_of(&scoped_hashes(d), "f"));
        // NOT free: same names, different structure.
        let e = "fn f(x: int) -> int { let y = x * 3; y }\n";
        assert_ne!(hash_of(&scoped_hashes(a), "f"), hash_of(&scoped_hashes(e), "f"));
    }

    #[test]
    fn declaration_order_does_not_matter() {
        let a = "fn a() -> int { 1 }\nfn b() -> int { a() }\n";
        let b = "fn b() -> int { a() }\nfn a() -> int { 1 }\n";
        assert_eq!(hash_of(&scoped_hashes(a), "b"), hash_of(&scoped_hashes(b), "b"));
        assert_eq!(hash_of(&scoped_hashes(a), "a"), hash_of(&scoped_hashes(b), "a"));
    }

    /// Editing a callee's body ripples into the caller's hash: the dep
    /// marker carries the callee's content hash.
    #[test]
    fn editing_a_callee_ripples_the_caller() {
        let a = "fn square(n: int) -> int { n * n }\nfn twice(n: int) -> int { square(n) + square(n) }\n";
        let b = "fn square(n: int) -> int { n * n + 1 }\nfn twice(n: int) -> int { square(n) + square(n) }\n";
        assert_ne!(hash_of(&scoped_hashes(a), "twice"), hash_of(&scoped_hashes(b), "twice"));
        // But the callee's *callers'* own text didn't move: same body, new dep.
        let canon_a = canon_of(a, "twice");
        let canon_b = canon_of(b, "twice");
        assert_ne!(canon_a, canon_b);
        assert_eq!(canon_a.len(), canon_b.len());
    }

    /// A local binding shadows the same-name top-level fn: the call resolves
    /// to the local, not the dep hash.
    #[test]
    fn a_local_shadows_a_same_name_fn() {
        let src = "fn f() -> int { 1 }\nfn g() -> int { let f = 0; f() }\n";
        let canon = canon_of(src, "g");
        assert!(canon.contains("let @v0 = 0 ; @v0 ( )"), "{canon}");
        assert!(!canon.contains("@dep:"), "{canon}");
        // And the shadowed body alpha-renames like any other.
        let renamed = "fn f() -> int { 1 }\nfn g() -> int { let z = 0; z() }\n";
        assert_eq!(hash_of(&scoped_hashes(src), "g"), hash_of(&scoped_hashes(renamed), "g"));
        // …while a `let`-RHS still sees the outer fn: `let f = f` binds the
        // dep's value, then shadows the name.
        let rhs = "fn f() -> int { 1 }\nfn g() -> int { let f = f; f() }\n";
        let canon = canon_of(rhs, "g");
        assert!(canon.contains("let @v0 = @dep:"), "{canon}");
        assert!(canon.contains("; @v0 ( )"), "{canon}");
    }

    /// Recursion and mutual recursion: `@self` keeps renames free, and dep
    /// substitution is one pass — a↔b terminates.
    #[test]
    fn recursion_terminates_and_renames_free() {
        let f = "fn f(n: int) -> int { if n <= 0 { 0 } else { f(n - 1) } }\n";
        let g = "fn g(n: int) -> int { if n <= 0 { 0 } else { g(n - 1) } }\n";
        assert_eq!(hash_of(&scoped_hashes(f), "f"), hash_of(&scoped_hashes(g), "g"));
        assert!(canon_of(f, "f").contains("@self"), "{}", canon_of(f, "f"));
        // 2a couldn't do this: token-level hashing still differs on rename.
        assert_ne!(hash_item(f), hash_item(g));
        // Mutual recursion — must terminate and stay deterministic.
        let cyc = "fn a(n: int) -> int { b(n) }\nfn b(n: int) -> int { a(n) }\n";
        let first = scoped_hashes(cyc);
        let second = scoped_hashes(cyc);
        assert_eq!(first, second);
        assert!(canon_of(cyc, "a").contains("@dep:"));
        assert!(canon_of(cyc, "b").contains("@dep:"));
    }

    /// Stdlib/module/member names are not dep-substituted — only same-source
    /// top-level fns are.
    #[test]
    fn foreign_names_stay_literal() {
        // `square` defined on-page; a method named `square` is unrelated.
        let src = "fn square(n: int) -> int { n * n }\nfn f(o: int) -> int { o.square() }\n";
        let canon = canon_of(src, "f");
        assert!(canon.contains(". square ( )"), "{canon}");
        // `str::len` — a `::` path, kept literal.
        let src2 = "fn f(s: str) -> int { str::len(s) }\n";
        let canon2 = canon_of(src2, "f");
        assert!(canon2.contains("str :: len"), "{canon2}");
        // Unbound free idents stay literal too (a `len` with no same-source
        // `fn len` is external to us).
        let src3 = "fn f(s: str) -> int { len(s) }\n";
        assert!(canon_of(src3, "f").contains("len ( @v0 )"), "{}", canon_of(src3, "f"));
    }

    /// Fields, dict keys, named args, pattern paths — none are variables.
    #[test]
    fn namespaced_positions_keep_names() {
        // Named arg `n` isn't a local even when `n` is bound.
        let src = "fn f(n: int) -> int { n }\nfn g() -> int { let n = 1; f(n = n) }\n";
        let canon = canon_of(src, "g");
        assert!(canon.contains("@dep:"), "{canon}");
        assert!(canon.contains("( n = @v0 )"), "{canon}");
        // Dict keys stay literal.
        let src2 = "fn g() -> int { let x = 0; ~{ x = 1 } }\n";
        let canon2 = canon_of(src2, "g");
        assert!(canon2.contains("~{ x = 1 }"), "{canon2}");
    }

    /// Bindings in for/match/if-let/while-let/closures all alpha-rename.
    #[test]
    fn every_binding_form_alpha_renames() {
        let a = "fn f(xs: [int]) -> int { for x in xs { x } }\n";
        let b = "fn f(xs: [int]) -> int { for y in xs { y } }\n";
        assert_eq!(hash_of(&scoped_hashes(a), "f"), hash_of(&scoped_hashes(b), "f"));
        let canon = canon_of(a, "f");
        assert!(canon.contains("for @v1 in @v0"), "{canon}");

        let c = "fn f(m: int) -> int { match m { x => x } }\n";
        let d = "fn f(m: int) -> int { match m { q => q } }\n";
        assert_eq!(hash_of(&scoped_hashes(c), "f"), hash_of(&scoped_hashes(d), "f"));

        let e = "fn f(g: _) -> int { let h = |x| x + 1; h(1) }\n";
        let i = "fn f(g: _) -> int { let h = |y| y + 1; h(1) }\n";
        assert_eq!(hash_of(&scoped_hashes(e), "f"), hash_of(&scoped_hashes(i), "f"));
    }

    /// f-string interpolation: the interior idents carry real locations;
    /// renaming the local rewrites the token's text.
    #[test]
    fn fstring_interpolations_alpha_rename() {
        let a = "fn f(x: int) -> str { f\"{x}!\" }\n";
        let b = "fn f(q: int) -> str { f\"{q}!\" }\n";
        assert_eq!(hash_of(&scoped_hashes(a), "f"), hash_of(&scoped_hashes(b), "f"));
        let canon = canon_of(a, "f");
        assert!(canon.contains("@v0"), "{canon}");
        assert!(!canon.contains('x'), "{canon}");
    }

    /// A nested fn binds its name in the block scope: recursion inside it
    /// resolves through the same local slot, and renaming it is free.
    #[test]
    fn nested_fns_bind_and_alpha_rename() {
        let a = "fn f() -> int { fn inner(n: int) -> int { if n <= 0 { 0 } else { inner(n - 1) } } inner(3) }\n";
        let b = "fn f() -> int { fn helper(n: int) -> int { if n <= 0 { 0 } else { helper(n - 1) } } helper(3) }\n";
        assert_eq!(hash_of(&scoped_hashes(a), "f"), hash_of(&scoped_hashes(b), "f"));
        let canon = canon_of(a, "f");
        assert!(canon.contains("fn @v0"), "{canon}");
        assert!(!canon.contains("inner"), "{canon}");
    }

    /// Type params are bindings too: `fn id<T>(x: T) -> T` ≡ `fn id<U>(x: U)`.
    #[test]
    fn type_params_alpha_rename() {
        let a = "fn id<T>(x: T) -> T { x }\n";
        let b = "fn id<U>(y: U) -> U { y }\n";
        assert_eq!(hash_of(&scoped_hashes(a), "id"), hash_of(&scoped_hashes(b), "id"));
        let canon = canon_of(a, "id");
        assert_eq!(canon, "fn _ < @v0 > ( @v1 : @v0 ) -> @v0 { @v1 }");
    }

    /// `use`d names shadow same-source fns rather than dep-substituting:
    /// conservative — an ambiguous program keeps the written name.
    #[test]
    fn used_names_are_not_dep_substituted() {
        let src = "use a::square;\nfn square(n: int) -> int { n * n }\nfn f() -> int { square(2) }\n";
        let canon = canon_of(src, "f");
        assert!(canon.contains("square ( 2 )"), "{canon}");
        assert!(!canon.contains("@dep:"), "{canon}");
        // Without the `use`, the same call site dep-substitutes.
        let src2 = "fn square(n: int) -> int { n * n }\nfn f() -> int { square(2) }\n";
        assert!(canon_of(src2, "f").contains("@dep:"), "{}", canon_of(src2, "f"));
    }

    /// Unparseable source: `canonical_scoped` is `None`; `hash_scoped`
    /// falls back to the token-level hash — never a panic.
    #[test]
    fn unparseable_falls_back_to_token_hash() {
        let broken = "fn f( { let";
        let items = extract(broken);
        let globals = Globals::empty();
        if let Some(item) = items.first() {
            assert!(item.scoped_canonical(&globals).is_none());
            assert_eq!(item.scoped_hash(&globals), item.hash());
        }
        // And a non-fn slice has no scoped form at all.
        assert!(canonical_scoped("let x = 1;", &globals).is_none());
    }

    /// `canonical()`/`hash_item` are untouched — existing v1 hashes depend
    /// on their exact output.
    #[test]
    fn v1_token_hash_unchanged() {
        let src = "fn square(n: int) -> int { n * n }\n";
        assert_eq!(
            hash_item(src),
            blake3::hash(crate::canonical(src).unwrap().as_bytes())
                .to_hex()
                .to_string()
        );
        // Scoped and token-level forms are different namespaces of hash.
        assert_ne!(hash_of(&scoped_hashes(src), "square"), hash_item(src));
    }

    /// Repeating the whole pipeline is deterministic — span-keyed subs and
    /// sorted struct-pattern fields leave nothing to HashMap order.
    #[test]
    fn deterministic_across_runs() {
        let src = "fn a(x: int) -> int { let y = b(x); y }\nfn b(x: int) -> int { a(x) }\n";
        let mut seen = std::collections::HashSet::new();
        for _ in 0..5 {
            seen.insert(scoped_hashes(src));
        }
        assert_eq!(seen.len(), 1);
    }

    /// `deps` is the linking table: every `@dep:` site in canonical (source)
    /// order, carrying the written name and the callee's pass-1 hash.
    #[test]
    fn deps_lists_sites_in_order() {
        let src = "fn a() -> int { 1 }\nfn b() -> int { 2 }\nfn f() -> int { a() + b() + a() }\n";
        let items = extract(src);
        let globals = Globals::for_source(src);
        let f = items.iter().find(|i| i.name == "f").unwrap();
        let relocs = f.relocs(&globals).unwrap();
        let want_a = items[0].hash();
        let want_b = items[1].hash();
        assert_eq!(
            relocs
                .iter()
                .map(|r| (r.name(), r.hash()))
                .collect::<Vec<_>>(),
            vec![("a", Some(want_a.as_str())), ("b", Some(want_b.as_str())), ("a", Some(want_a.as_str()))]
        );
        // The site's span is the substituted ident's span in the item source.
        for reloc in &relocs {
            let (lo, hi) = reloc.span();
            assert_eq!(&f.source[lo..hi], reloc.name());
        }
        // The same sites, as markers, appear in the canonical form.
        let canon = f.scoped_canonical(&globals).unwrap();
        assert_eq!(canon.matches("@dep:").count(), 3, "{canon}");
        // `deps()` on the item slice agrees with the free function.
        assert_eq!(deps(&f.source, &globals), Some(relocs));
    }

    /// Recursion records `@self` sites — a reloc with no dep hash.
    #[test]
    fn deps_marks_self_recursion() {
        let src = "fn f(n: int) -> int { if n <= 0 { 0 } else { f(n - 1) } }\n";
        let items = extract(src);
        let globals = Globals::for_source(src);
        let relocs = items[0].relocs(&globals).unwrap();
        assert_eq!(relocs.len(), 1);
        assert!(matches!(&relocs[0], Reloc::SelfRef { name, .. } if name == "f"));
        assert_eq!(relocs[0].hash(), None);
        let (lo, hi) = relocs[0].span();
        assert_eq!(&items[0].source[lo..hi], "f");
        // Mutual recursion records one dep edge per cross reference.
        let cyc = "fn a(n: int) -> int { b(n) }\nfn b(n: int) -> int { a(n) }\n";
        let items = extract(cyc);
        let globals = Globals::for_source(cyc);
        let a = items.iter().find(|i| i.name == "a").unwrap();
        let b = items.iter().find(|i| i.name == "b").unwrap();
        assert_eq!(
            a.relocs(&globals).unwrap(),
            vec![Reloc::Dep {
                name: "b".into(),
                hash: b.hash(),
                start: a.source.find("b(n)").unwrap(),
                end: a.source.find("b(n)").unwrap() + 1,
            }]
        );
        assert_eq!(b.relocs(&globals).unwrap()[0].hash(), Some(a.hash().as_str()));
    }

    /// Calls that resolve to locals or `use`d names are not reloc sites —
    /// only same-source fn references and self-references are.
    #[test]
    fn deps_skips_shadowed_and_foreign_names() {
        let src = "fn f() -> int { 1 }\nfn g() -> int { let f = 0; f() }\n";
        let items = extract(src);
        let globals = Globals::for_source(src);
        let g = items.iter().find(|i| i.name == "g").unwrap();
        assert_eq!(g.relocs(&globals).unwrap(), vec![]);
        // `let f = f` binds the dep on the right, then shadows the name:
        // exactly one site — the RHS.
        let rhs = "fn f() -> int { 1 }\nfn g() -> int { let f = f; f() }\n";
        let items = extract(rhs);
        let globals = Globals::for_source(rhs);
        let g = items.iter().find(|i| i.name == "g").unwrap();
        let relocs = g.relocs(&globals).unwrap();
        assert_eq!(relocs.len(), 1, "{relocs:?}");
        let (lo, hi) = relocs[0].span();
        assert_eq!(&g.source[lo..hi], "f");
        assert!(lo < g.source.find("; f()").unwrap());
        // Unparseable source: `None`, same gate as `canonical_scoped`.
        assert!(deps("fn f( { let", &globals).is_none());
        assert!(deps("let x = 1;", &Globals::empty()).is_none());
    }

    /// `manifest` is the page-level form: names and hashes match
    /// `scoped_hashes`, and each entry's reloc sites come from `deps`.
    #[test]
    fn manifest_pairs_hashes_with_dep_edges() {
        let src = "fn square(n: int) -> int { n * n }\nfn twice(n: int) -> int { square(n) + square(n) }\nfn leaf() -> int { 0 }\n";
        let manifest = manifest(src);
        let hashes = scoped_hashes(src);
        assert_eq!(manifest.len(), 3);
        for (entry, (name, hash)) in manifest.iter().zip(&hashes) {
            assert_eq!(&entry.name, name);
            assert_eq!(&entry.hash, hash);
        }
        let twice = &manifest[1];
        // Dep edges carry the callee's pass-1 (token-level) hash — the same
        // value `dep_table` reports — not its scoped hash. Each entry also
        // publishes that token hash, so an edge target resolves to the
        // entry whose `token_hash` matches: the manifest is self-contained.
        let callee_token_hash = extract(src)[0].hash();
        assert_eq!(manifest[0].token_hash, callee_token_hash);
        assert_eq!(
            twice.dep_hashes().collect::<Vec<_>>(),
            vec![callee_token_hash.as_str(), callee_token_hash.as_str()]
        );
        for target in twice.dep_hashes() {
            let entry = manifest.iter().find(|m| m.token_hash == target);
            assert_eq!(entry.map(|m| m.name.as_str()), Some("square"));
        }
        assert!(manifest[0].relocs.is_empty());
        assert!(manifest[2].relocs.is_empty());
        // Extraction order is preserved — the manifest is positional.
        assert_eq!(
            manifest.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
            vec!["square", "twice", "leaf"]
        );
    }

    /// The manifest is rename-stable: renaming a callee (and its call
    /// sites) leaves every entry's hash and every dep edge untouched —
    /// the property link-by-hash exists for.
    #[test]
    fn manifest_is_rename_stable() {
        let a = "fn square(n: int) -> int { n * n }\nfn twice(n: int) -> int { square(n) }\n";
        let b = "fn sq(n: int) -> int { n * n }\nfn twice(n: int) -> int { sq(n) }\n";
        let ma = manifest(a);
        let mb = manifest(b);
        assert_eq!(ma[0].hash, mb[0].hash);
        assert_eq!(ma[1].hash, mb[1].hash);
        assert_eq!(
            ma[1].dep_hashes().collect::<Vec<_>>(),
            mb[1].dep_hashes().collect::<Vec<_>>()
        );
        // … while editing the callee's body moves the edge target.
        let c = "fn square(n: int) -> int { n * n + 1 }\nfn twice(n: int) -> int { square(n) }\n";
        let mc = manifest(c);
        assert_ne!(ma[1].dep_hashes().collect::<Vec<_>>(), mc[1].dep_hashes().collect::<Vec<_>>());
        assert_ne!(ma[1].hash, mc[1].hash);
    }

    /// Dep edges can point at idents inside f-string interpolation — the
    /// site records the inner ident's span even though the outer token is
    /// one lexer unit.
    #[test]
    fn deps_inside_fstrings() {
        let src = "fn who() -> str { \"world\" }\nfn greet() -> str { f\"hi {who()}\" }\n";
        let items = extract(src);
        let globals = Globals::for_source(src);
        let greet = items.iter().find(|i| i.name == "greet").unwrap();
        let relocs = greet.relocs(&globals).unwrap();
        assert_eq!(relocs.len(), 1);
        let (lo, hi) = relocs[0].span();
        assert_eq!(&greet.source[lo..hi], "who");
        assert!(greet.scoped_canonical(&globals).unwrap().contains("@dep:"));
    }

    /// `use "page"` — a call resolving to an included page's fn emits the
    /// same `@dep:<token-hash>` a same-source callee does: the edge names
    /// content, not location.
    #[test]
    fn used_page_fns_are_dep_edges() {
        let kit = "fn row() -> int { 42 }\n";
        let page = "use \"kit\";\nfn go() -> int { row() }\n";
        let globals = Globals::for_page(page, |n| (n == "kit").then(|| kit.to_string()));
        assert_eq!(globals.includes(), &["kit".to_string()]);
        let items = extract(page);
        let go = items.iter().find(|i| i.name == "go").unwrap();
        let relocs = go.relocs(&globals).unwrap();
        assert_eq!(relocs.len(), 1);
        assert_eq!(relocs[0].name(), "row");
        assert_eq!(relocs[0].hash(), Some(extract(kit)[0].hash().as_str()));
        // …and `go` hashes identically to the page where `row` is written
        // inline — content-addressed, not page-addressed.
        let inline = "fn go() -> int { row() }\nfn row() -> int { 42 }\n";
        assert_eq!(
            go.scoped_hash(&globals),
            hash_of(&scoped_hashes(inline), "go")
        );
    }

    /// Resolution is positional, matching the interpreter's module rib: an
    /// own `fn` declared *above* the call shadows the `use`d one; declared
    /// *below* it, the include's decl wins — the page's calls all sit before
    /// the spliced include, and a use site with no decl before it takes the
    /// last declaration.
    #[test]
    fn shadowing_a_used_fn_is_position_dependent() {
        let kit = "fn row() -> int { 99 }\n";
        let resolve = |n: &str| (n == "kit").then(|| kit.to_string());
        // Own decl above the call → own row wins.
        let above = "use \"kit\";\nfn row() -> int { 42 }\nfn go() -> int { row() }\n";
        let globals = Globals::for_page(above, resolve);
        let items = extract(above);
        let go = items.iter().find(|i| i.name == "go").unwrap();
        let own_row = items.iter().find(|i| i.name == "row").unwrap();
        assert_eq!(
            go.relocs(&globals).unwrap()[0].hash(),
            Some(own_row.hash().as_str())
        );
        // Own decl below the call → kit's row wins.
        let below = "use \"kit\";\nfn go() -> int { row() }\nfn row() -> int { 42 }\n";
        let globals = Globals::for_page(below, resolve);
        let go = extract(below)
            .into_iter()
            .find(|i| i.name == "go")
            .unwrap();
        assert_eq!(
            go.relocs(&globals).unwrap()[0].hash(),
            Some(extract(kit)[0].hash().as_str())
        );
    }

    /// Renaming a `use`d page's fn — and its call sites — leaves the
    /// caller's scoped hash untouched, same as same-page renames.
    #[test]
    fn renaming_a_used_page_fn_is_free() {
        let a = "use \"kit\";\nfn go() -> int { row() }\n";
        let b = "use \"kit\";\nfn go() -> int { r() }\n";
        let ga = Globals::for_page(a, |n| {
            (n == "kit").then(|| "fn row() -> int { 42 }\n".to_string())
        });
        let gb = Globals::for_page(b, |n| {
            (n == "kit").then(|| "fn r() -> int { 42 }\n".to_string())
        });
        let go_a = extract(a).into_iter().find(|i| i.name == "go").unwrap();
        let go_b = extract(b).into_iter().find(|i| i.name == "go").unwrap();
        assert_eq!(go_a.scoped_hash(&ga), go_b.scoped_hash(&gb));
        // … while editing the include's body moves the edge target.
        let gc = Globals::for_page(a, |n| {
            (n == "kit").then(|| "fn row() -> int { 43 }\n".to_string())
        });
        assert_ne!(go_a.scoped_hash(&ga), go_a.scoped_hash(&gc));
    }

    /// Two includes providing the same name: the last spliced decl wins —
    /// the interpreter's `table.get` fallback when no decl precedes the
    /// use. Same for a transitive include reached through another page.
    #[test]
    fn the_last_included_decl_wins() {
        let pages = [
            ("a", "fn row() -> int { 1 }\n"),
            ("b", "fn row() -> int { 2 }\n"),
        ];
        let page = "use \"a\";\nuse \"b\";\nfn go() -> int { row() }\n";
        let globals = Globals::for_page(page, |n| {
            pages
                .iter()
                .find(|(name, _)| *name == n)
                .map(|(_, s)| s.to_string())
        });
        assert_eq!(globals.includes(), &["a".to_string(), "b".to_string()]);
        let go = extract(page).into_iter().find(|i| i.name == "go").unwrap();
        assert_eq!(
            go.relocs(&globals).unwrap()[0].hash(),
            Some(extract(pages[1].1)[0].hash().as_str()),
            "b's row shadows a's"
        );
        // Transitive: kit uses img; img's decls splice after kit's, so
        // img's `row` — not kit's — is what the page's calls see.
        let transitive = [
            ("kit", "use \"img\";\nfn row() -> int { 1 }\n"),
            ("img", "fn row() -> int { 2 }\nfn shade() -> int { 7 }\n"),
        ];
        let page2 = "use \"kit\";\nfn go() -> int { row() + shade() }\n";
        let globals2 = Globals::for_page(page2, |n| {
            transitive
                .iter()
                .find(|(name, _)| *name == n)
                .map(|(_, s)| s.to_string())
        });
        assert_eq!(
            globals2.includes(),
            &["kit".to_string(), "img".to_string()]
        );
        let go2 = extract(page2).into_iter().find(|i| i.name == "go").unwrap();
        let relocs = go2.relocs(&globals2).unwrap();
        assert_eq!(relocs.len(), 2);
        assert_eq!(
            relocs[0].hash(),
            Some(extract(transitive[1].1)[0].hash().as_str()),
            "img's row, reached through kit"
        );
        assert_eq!(
            relocs[1].hash(),
            Some(extract(transitive[1].1)[1].hash().as_str())
        );
    }

    /// An unresolvable `use "…"` contributes nothing: the page's fns hash
    /// as if it weren't there, and a call to a name nobody declares stays
    /// literal — matching the assembler's `missing` list.
    #[test]
    fn missing_includes_contribute_no_names() {
        let page = "use \"ghost\";\nfn go() -> int { row() }\n";
        let globals = Globals::for_page(page, |_| None);
        assert!(globals.includes().is_empty());
        let go = extract(page).into_iter().find(|i| i.name == "go").unwrap();
        assert_eq!(go.relocs(&globals).unwrap(), vec![]);
        let canon = go.scoped_canonical(&globals).unwrap();
        assert!(canon.contains("row ( )"), "{canon}");
        // An include cycle terminates: a page using itself, or a↔b.
        let pages = [("a", "use \"b\";\nfn f() -> int { 1 }\n"), ("b", "use \"a\";\nfn g() -> int { 2 }\n")];
        let globals = Globals::for_page("use \"a\";\nfn go() -> int { f() }\n", |n| {
            pages
                .iter()
                .find(|(name, _)| *name == n)
                .map(|(_, s)| s.to_string())
        });
        assert_eq!(globals.includes(), &["a".to_string(), "b".to_string()]);
    }

    /// `manifest_with` carries the cross-page edges the same way it carries
    /// same-page ones — dep targets are token hashes either way.
    #[test]
    fn manifest_includes_cross_page_edges() {
        let kit = "fn row() -> int { 42 }\n";
        let page = "use \"kit\";\nfn go() -> int { row() }\nfn leaf() -> int { 0 }\n";
        let globals = Globals::for_page(page, |n| (n == "kit").then(|| kit.to_string()));
        let manifest = manifest_with(page, &globals);
        let kit_row = extract(kit)[0].hash();
        let go = &manifest[0];
        assert_eq!(go.dep_hashes().collect::<Vec<_>>(), vec![kit_row.as_str()]);
        assert!(manifest[1].relocs.is_empty());
        // The edge resolves through the *include's* manifest by token_hash,
        // exactly as same-page edges resolve through this one.
    }
}
