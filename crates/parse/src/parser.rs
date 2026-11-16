use crate::{components::*, errors::*, *};
use chompy::{
    lex::{Lex, Tok, Token},
    utils::Located as _,
};
use hashbrown::HashMap;
use lex::{Lexer, TokKind, interp_end};
use miette::NamedSource;
use shared::{FileId, Located, Location, Span, units::Dim};
use std::{cell::Cell, sync::Arc};

mod datalog;
pub(crate) mod macros;
mod reflect;
mod subscripts;
/// Forgets every declared macro, and every declaration the reflection
/// macros answer for: a new program starts with none.
pub fn clear_macros() {
    macros::clear();
    reflect::clear();
}

/// Recursively descends mimas source, incrementally returning various
/// statements and expressions.
/// What `x y` — an expression followed by an atom on the same line —
/// parses as. `Apply` is the language rule: `x y` is `x(y)`, and
/// `f (x)` already meant exactly that. `Off` is for regions where
/// whitespace already means "next item" (`table {}` cells today; a
/// space-separated row or a `2 x`-as-multiply mode would be another).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Juxta {
    /// `x y` parses as `x(y)` -- a same-line atom after an expr applies.
    Apply,
    /// Adjacency never applies -- the next token is the *next thing*:
    /// `if`/`while`/`for` heads, `table` cells, `query` verbs.
    Off,
    /// Inside a sequence (`[…]`, `f(…)`, `{…}`): adjacency ends the
    /// element, so commas are optional. Application needs `(…)` --
    /// `[f x]` is `[f, x]`; `[f(x)]` applies.
    Comma,
}

pub struct Parser<'s> {
    /// Always ends with an `Eof`.
    tokens: Vec<Tok<TokKind<'s>>>,
    /// Each doc comment, keyed by the index of the token after it.
    docs: HashMap<usize, &'s str>,
    /// Index of the next token in `tokens`.
    next: usize,
    /// Spent by looking at the next token and refilled by taking it. Running out means the
    /// parser is stuck.
    fuel: Cell<u32>,
    /// All errors the parser encounterse along its descent.
    errors: Vec<shared::Error>,
    /// The token the last error was reported at. Anything else that goes wrong there is
    /// fallout, and isn't reported.
    last_error: Option<usize>,
    /// The lexer already reported why the input ends where it does.
    cut_short: bool,
    /// The file id of this parser's source.
    file_id: FileId,
    /// The file name of the this parser's source. Todo: do we really need this _and_ the id?
    file_name: String,
    src: NamedSource<Arc<str>>,
    /// Off inside a condition, where a `{` opens the body.
    struct_literals: bool,
    /// What `x y` — an expression followed by an atom on the same line —
    /// parses as. `Apply` is the language rule: `x y` is `x(y)`. `Off`
    /// keeps whitespace as a separator, which is what `table {}` cells
    /// already mean — and what a juxtaposed argument parses under, so
    /// `f x y` stays `(f x) y` rather than `f (x y)`.
    juxta: Juxta,
    /// Inside a `(...)` group (call args, tuples, `#[tests]` lists), where an infix operator
    /// may bind across a newline.
    group_depth: usize,
    generator_type: Option<Annotation>,
    accessor_yield: bool,
    /// On while `±`'s tolerance parses, so a trailing `%` stays the percent marker
    /// (`x ± 10%`) instead of folding as postfix percent. Groups reset it -- inside
    /// `x ± (50%)` the `%` is postfix again.
    pct_marker: bool,
    /// Monotonic counter for `__assert_left_N` / `__assert_right_N` names minted by `assert!`
    /// rewriting, so nested asserts don't collide.
    assert_id: u32,
    /// Monotonic counter for `__with_N` names minted by the `with` desugar,
    /// so nested regions don't collide.
    with_id: u32,
    depth: usize,
    /// The dimension of each unit-suffixed literal (`25kW`), by the literal's node.
    quantities: HashMap<NodeId, shared::units::Dim>,
    /// `#1.10`: number literals marked exact, read as `dec`.
    exact_literals: std::collections::HashSet<NodeId>,
    /// The unit text as written for each literal in `quantities` (`5ft` -> `"ft"`).
    quantity_units: HashMap<NodeId, String>,
    /// The axis a direction suffix tagged an expression with (`50.0px↑`), by
    /// the wrapper node -- see `direction_suffix`.
    directions: HashMap<NodeId, shared::units::Dim>,
    /// The source's per-page shadow table (`Lexer::shadow_table`) — an
    /// f-string interpolation sublexer takes its page's set.
    shadow: Vec<(usize, [bool; 3])>,
    /// Top-level statements a declaration synthesized alongside itself
    /// (`opp`'s `impl Opp`), emitted right after it.
    pending: Vec<Stmt>,
    /// Attributes on functions that are not the parser's own (`#[machine]`):
    /// marks for `fns!` to read. One that no `fns!` call reads by the end
    /// of the source is an unknown attribute (`reflect::unread_marks`).
    pending_marks: Vec<(String, Location)>,
    /// One entry per token in `tokens`: how the token sits against the one
    /// before it, and which macro expansion wrote it. The parser asks this,
    /// not the spans, whether two tokens touch or a line breaks between
    /// them. A spliced macro expansion has spans from the call site.
    meta: Vec<TokMeta>,
    /// Macro hygiene and expansion state (`parser/macros.rs`).
    hygiene: macros::Hygiene,
}

/// How a token sits in the stream (see [`Parser::meta`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TokMeta {
    /// It starts exactly where the token before it ends.
    pub(crate) glued: bool,
    /// A line break comes between it and the token before it.
    pub(crate) newline: bool,
    /// The macro expansion whose template wrote it. 0 is source text,
    /// including a fragment the call site passed in.
    pub(crate) mark: u32,
}

/// The [`TokMeta`] of each token, read from the text between neighbours.
/// `offset` is where `source` starts in the spans (an f-string
/// interpolation lexes a slice).
fn token_meta(source: &str, offset: usize, tokens: &[Tok<TokKind<'_>>]) -> Vec<TokMeta> {
    let mut meta = Vec::with_capacity(tokens.len());
    let mut prev_end: Option<usize> = None;
    for tok in tokens {
        let start = tok.span().start();
        let entry = match prev_end {
            None => TokMeta {
                glued: true,
                newline: false,
                mark: 0,
            },
            Some(end) => {
                let gap = source
                    .get(end.saturating_sub(offset)..start.saturating_sub(offset))
                    .unwrap_or_default();
                TokMeta {
                    glued: end == start,
                    newline: gap.contains('\n'),
                    mark: 0,
                }
            }
        };
        meta.push(entry);
        prev_end = Some(tok.span().end());
    }
    meta
}

// Basic features
impl<'s> Parser<'s> {
    /// Creates a new parser.
    pub fn new(lexer: Lexer<'s>) -> Self {
        let src = NamedSource::new(lexer.file_name(), Arc::<str>::from(lexer.source()));
        // A fresh source is a fresh unit scope — `unit` decls never leak
        // between standalone parses.
        shared::units::clear();
        // and so is its macro table, and the declarations it can reflect on
        clear_macros();
        Self::with_src(lexer, src)
    }

    /// Like [`Self::new`], but keeps the ambient unit table — the files of
    /// one program parse through this so their `unit` decls accumulate.
    /// `load_files` clears once before the loop.
    pub fn new_extending(lexer: Lexer<'s>) -> Self {
        let src = NamedSource::new(lexer.file_name(), Arc::<str>::from(lexer.source()));
        Self::with_src(lexer, src)
    }

    /// Creates a parser whose diagnostics render against `src` rather than the lexer's own
    /// source (the lexer is on a slice of it).
    fn with_src(lexer: Lexer<'s>, src: NamedSource<Arc<str>>) -> Self {
        Self::with_src_at(lexer, src, None)
    }

    /// Like [`Self::with_src`]. With `relocate`, every token and lexer error
    /// takes that location instead of its own: a macro template's f-string
    /// has no place in the file, so its interpolation reports at the call.
    fn with_src_at(
        mut lexer: Lexer<'s>,
        src: NamedSource<Arc<str>>,
        relocate: Option<Location>,
    ) -> Self {
        let mut tokens: Vec<_> = lexer.by_ref().collect();
        // `unit` decls teach the lexer new suffixes — `18.0tile` only reads
        // as a quantity once `tile` is registered — so the token list gets
        // scanned for decls and the source relexed when a new name appears.
        // Top-level sources only: an f-string interpolation's sublexer
        // (offset > 0) inherits the file's table.
        if lexer.offset() == 0 && declare_units(lexer.source(), &tokens) {
            lexer = Lexer::new(
                lexer.source(),
                lexer.file_id(),
                lexer.file_name().to_string(),
            )
            .with_shadow_table(lexer.shadow_table().to_vec());
            tokens = lexer.by_ref().collect();
        }
        tokens.push(Tok::new(TokKind::Eof, lexer.end()));

        // Remove the comments from the stream, keep them for later
        let docs = tokens
            .iter()
            .enumerate()
            .filter(|(_, tok)| tok.kind().is_comment())
            .enumerate()
            .filter_map(|(removed, (og_index, tok))| match tok.kind() {
                // once the comments are out, the token after this one is at `og_index - removed`
                TokKind::DocComment(text) => Some((og_index - removed, text)),
                _ => None,
            })
            .collect();
        tokens.retain(|tok| !tok.kind().is_comment());
        let meta = token_meta(lexer.source(), lexer.offset(), &tokens);
        if let Some(at) = relocate {
            for tok in &mut tokens {
                tok.location = at.into();
            }
        }

        let errors = lexer
            .take_errors()
            .into_iter()
            .map(|(diag, location)| {
                let err = LexError {
                    src: src.clone(),
                    diag,
                    at: relocate.unwrap_or_else(|| Location::from(location)).into(),
                };
                err.into()
            })
            .collect();
        Self {
            tokens,
            docs,
            next: 0,
            fuel: Cell::new(FUEL),
            errors,
            last_error: None,
            cut_short: lexer.cut_short(),
            shadow: lexer.shadow_table().to_vec(),
            pending: Vec::new(),
            pending_marks: Vec::new(),
            file_id: lexer.file_id(),
            file_name: lexer.file_name().into(),
            src,
            struct_literals: true,
            juxta: Juxta::Apply,
            group_depth: 0,
            generator_type: None,
            accessor_yield: false,
            pct_marker: false,
            assert_id: 0,
            with_id: 0,
            depth: 0,
            quantities: HashMap::new(),
            exact_literals: Default::default(),
            directions: HashMap::new(),
            quantity_units: HashMap::new(),
            meta,
            hygiene: macros::Hygiene::default(),
        }
    }

    /// Parses the whole source into an Ast, along with every error found. Anything that failed
    /// to parse is left in the Ast as [Poison], which later stages treat as unreachable, so only
    /// pass the Ast on when the errors are empty.
    pub fn into_ast(mut self) -> (Ast, Vec<shared::Error>) {
        let mut statements = vec![];
        while !self.at(TokKind::Eof) {
            if self.peek().starts_stmt() {
                statements.push(self.stmt());
                statements.append(&mut self.pending);
            } else {
                self.reject_stmt(self.unexpected_token());
            }
        }
        self.unread_marks();
        // the lexer's errors went in first
        self.errors.sort_by_cached_key(|err| {
            let label = err.labels().and_then(|mut labels| labels.next());
            label.map_or(0, |label| label.offset())
        });
        let docs = self
            .docs
            .iter()
            .map(|(&at, doc)| {
                // a `/** … */` block doc loses its frame first; then each
                // line loses its margin — `*`s for a block, `/`s for a
                // `///` run, so `/// * bullet` keeps its markdown bullet
                let (body, block) = match doc.strip_prefix("/*") {
                    Some(rest) => {
                        let body = rest.strip_prefix('*').unwrap_or(rest);
                        (body.strip_suffix("*/").unwrap_or(body), true)
                    }
                    None => (*doc, false),
                };
                let lines: Vec<_> = body
                    .lines()
                    .map(|line| {
                        let line = line.trim_start();
                        let line = if block {
                            line.trim_start_matches('*')
                        } else {
                            line.trim_start_matches('/')
                        };
                        line.strip_prefix(' ').unwrap_or(line).trim_end()
                    })
                    .collect();
                (self.tokens[at].span().start(), lines.join("\n"))
            })
            .collect();
        (
            Ast::new(
                self.file_name,
                self.src,
                statements,
                docs,
                self.quantities,
                self.quantity_units,
                self.exact_literals,
                self.directions,
            ),
            self.errors,
        )
    }

    /// The Ast, if the source parsed cleanly. What most callers want -- see [Self::into_ast]
    /// for the errors themselves.
    pub fn try_into_ast(self) -> std::result::Result<Ast, Vec<shared::Error>> {
        let (ast, errors) = self.into_ast();
        if errors.is_empty() {
            Ok(ast)
        } else {
            Err(errors)
        }
    }

    /// Every error found so far. Check this after [Self::expr], which returns poison instead
    /// of failing.
    pub fn errors(&self) -> &[shared::Error] {
        &self.errors
    }

    /// Runs `body` one level deeper into the grammar, or gives up with `None` when that's too
    /// deep (the stack would overflow long before the input ran out). Every cycle in the
    /// grammar has to pass through a call to this.
    fn nested<T>(&mut self, body: impl FnOnce(&mut Self) -> T) -> Option<T> {
        const MAX_DEPTH: usize = 100;
        if self.depth >= MAX_DEPTH {
            self.error(NestingTooDeep {
                src: self.src(),
                at: self.next_location().into(),
            });
            // the rest of the statement would only land us back here
            self.skip();
            self.skip_stmt();
            return None;
        }
        self.depth += 1;
        let parsed = body(self);
        self.depth -= 1;
        Some(parsed)
    }

    /// Runs `body` with the group depth bumped -- the inside of a `(...)`-delimited list,
    /// where an infix operator may attach across a newline.
    fn in_group<T>(&mut self, body: impl FnOnce(&mut Self) -> T) -> T {
        self.with_group_depth(self.group_depth + 1, body)
    }

    /// Runs `body` at the given group depth, then puts it back. A `{` block resets to 0:
    /// statements inside must not inherit infix-across-newline permission from the parens
    /// the block sits in -- `({ c = a\n-b })` would otherwise still parse as `c = a - b`.
    fn with_group_depth<T>(&mut self, depth: usize, body: impl FnOnce(&mut Self) -> T) -> T {
        let saved = self.group_depth;
        let saved_pct = self.pct_marker;
        self.group_depth = depth;
        // `(50%)` inside a `±` tolerance is postfix percent, not the marker.
        self.pct_marker = false;
        let parsed = body(self);
        self.group_depth = saved;
        self.pct_marker = saved_pct;
        parsed
    }

    /// Runs `body` with `x y` resolved the given way, then puts it back.
    /// `Off` is what whitespace-separated regions (`table {}` cells) and
    /// a juxtaposed argument itself parse under — `f x y` is `(f x) y`,
    /// left-associative, never `f (x y)`.
    fn with_juxta<T>(&mut self, mode: Juxta, body: impl FnOnce(&mut Self) -> T) -> T {
        let saved = std::mem::replace(&mut self.juxta, mode);
        let parsed = body(self);
        self.juxta = saved;
        parsed
    }

    /// Whether an infix operator at the peek position may bind to the expression we just
    /// finished. A newline before the operator ends the expression (the semicolon-insertion
    /// rule), so `a\n+ b` is not addition; `a +\n b` still is, because the operator itself is
    /// on the same line as `a`. A `(...)` group opts back into newline-crossing.
    /// `binary_ops` also permits a leading logical OR after a left operand.
    fn infix_binds(&self) -> bool {
        self.group_depth > 0 || !self.at_line_start()
    }

    /// Runs `body` with struct literals allowed or not, then puts that back as it was. A
    /// condition turns them off, and any brackets inside it turn them back on (nothing in
    /// there can be mistaken for the body).
    fn struct_literals<T>(&mut self, allowed: bool, body: impl FnOnce(&mut Self) -> T) -> T {
        let outer = std::mem::replace(&mut self.struct_literals, allowed);
        let parsed = body(self);
        self.struct_literals = outer;
        parsed
    }

    /// Clone the per-file `NamedSource` for embedding in a diagnostic. Cheap (Arc + String).
    fn src(&self) -> NamedSource<Arc<str>> {
        self.src.clone()
    }

    /// Creates a new expression.
    fn new_expr(&self, expr: impl Into<ExprKind>, start: usize) -> Expr {
        Expr::new(expr.into(), self.location(start))
    }

    /// Creates a new statement.
    fn new_stmt(&self, stmt: impl Into<StmtKind>, start: usize) -> Stmt {
        Stmt::new(stmt.into(), NodeId::new(), self.location(start))
    }

    /// Creates a new pattern.
    fn new_pat(&self, pat: PatKind, start: usize) -> Pat {
        Pat::new(pat, self.location(start))
    }

    /// Creates a poison expression covering whatever was consumed since `start`.
    fn poison_expr(&self, start: usize) -> Expr {
        self.new_expr(Poison, start)
    }

    /// Creates a [Location] from the given position up until our current position (empty at the
    /// cursor if nothing was consumed since `start`).
    fn location(&self, start: usize) -> Location {
        let end = self.cursor();
        Location::new(self.file_id, Span::new(start.min(end), end))
    }
}

// Recursive descent (mimas grammar)
impl<'s> Parser<'s> {
    pub(crate) fn stmt(&mut self) -> Stmt {
        let start = self.next_start();
        match self.node() {
            BlockElement::Stmt(stmt) => stmt,
            BlockElement::MaybeYield(expr) => self.expr_stmt(expr, start),
        }
    }

    /// Routes the next token to the right parse function and reports back what kind of thing came
    /// out: a fully-formed statement, or a bare expression whose role (yielded value vs expression
    /// statement) the caller decides. Assignments finish here as `Stmt`; only true bare exprs
    /// surface as `MaybeYield`.
    fn node(&mut self) -> BlockElement {
        // a macro declaration holds no statement: read past it to the next one
        if self.macro_intro() {
            let start = self.next_start();
            while self.macro_intro() {
                self.macro_decl();
            }
            if !self.peek().starts_stmt() {
                return BlockElement::Stmt(self.empty_stmt(start));
            }
        }
        if self.stmt_macro_call() {
            return self.stmt_macro();
        }
        let start = self.next_start();
        match self.peek() {
            TokKind::Let => BlockElement::Stmt(self.let_stmt()),
            TokKind::Pub if self.nth(1) == TokKind::Ident("subscript") => {
                self.advance();
                let mut items = self.accessor_items(false, true).into_iter();
                let Some(item) = items.next() else { return BlockElement::Stmt(self.empty_stmt(start)); };
                for item in items { let stmt = self.new_stmt(item, start); self.pending.push(stmt); }
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            TokKind::Ident("subscript") if self.accessor_intro() => {
                let mut items = self.accessor_items(false, false).into_iter();
                let Some(item) = items.next() else { return BlockElement::Stmt(self.empty_stmt(start)); };
                for item in items { let stmt = self.new_stmt(item, start); self.pending.push(stmt); }
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            TokKind::Ident("yield") if self.accessor_yield => {
                self.advance();
                let target = self.expr();
                let call = self.generator_call("__accessor_yield", vec![target], start);
                let stmt = self.new_stmt(StmtKind::Expr(call), start);
                self.end_stmt(stmt.location());
                BlockElement::Stmt(stmt)
            }
            // r[impl mimas.generator.decl]
            TokKind::Ident("subscript") if self.notation_intro() => {
                self.advance();
                let (name, type_params, parameters, element) = self.function_sig(true, Some(""));
                let element = element.unwrap_or_else(|| {
                    self.expected("a yielded type after `->`");
                    Annotation::Unit
                });
                if matches!(element, Annotation::Unit | Annotation::Option(_)) {
                    self.error_here(Misdirection {
                        src: self.src(),
                        at: self.next_location().into(),
                        msg: "a generator must yield a non-null value".into(),
                        label: "wrap optional values in a struct so null can mark exhaustion"
                            .into(),
                    });
                }
                let wheres = self.where_clause();
                let saved = self.generator_type.replace(element.clone());
                let body = self.block();
                self.generator_type = saved;
                let mut parameter = Binding::new(self.ident_at("__generator_type", start));
                parameter.annotation = Some(element.clone());
                let closure = self.new_expr(
                    Closure {
                        parameters: vec![parameter],
                        body,
                        return_type: Some(Annotation::Unit),
                    },
                    start,
                );
                let body = self.generator_call("__generator", vec![closure], start);
                let f = Function {
                    name,
                    role: FnRole::Fn,
                    type_params,
                    parameters,
                    wheres,
                    return_type: Some(Annotation::Function(
                        vec![],
                        Box::new(Annotation::Option(Box::new(element))),
                    )),
                    body,
                };
                let item = Item::new(f.into(), self.location(start), false);
                self.end_item(&item);
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            TokKind::Ident("yield")
                if self.generator_type.is_some() && self.nth(1).starts_expr() =>
            {
                self.advance();
                let value = self.expr();
                let name = self.ident_at("__yield_value", start);
                let bind = self.new_stmt(
                    Let {
                        left: name.clone().into(),
                        annotation: self.generator_type.clone(),
                        right: value,
                        else_branch: None,
                    },
                    start,
                );
                let value = self.new_expr(name, start);
                let call = self.generator_call("__generator_yield", vec![value], start);
                let emit = self.new_stmt(StmtKind::Expr(call), start);
                self.end_stmt(self.location(start));
                BlockElement::Stmt(self.new_stmt(
                    StmtKind::Expr(self.new_expr(
                        Block {
                            body: vec![bind, emit],
                            yielded_expr: None,
                        },
                        start,
                    )),
                    start,
                ))
            }

            TokKind::Module => BlockElement::Stmt(self.module_stmt()),
            // `check <expr>` -- the literate spelling of a `#[tests]` case. The word stays
            // contextual: it only opens a check when a same-line expr follows, so `check(x)`
            // or `check = 5` still parse as ordinary expressions.
            // r[impl check.line]
            TokKind::Ident("check") if self.check_intro() => {
                let item = Item::new(self.check_block().into(), self.location(start), false);
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            // `axis wealth (poorer, richer);` names an axis and its two direction
            // words. Contextual like `unit`.
            TokKind::Ident("axis") if self.unit_intro() => {
                BlockElement::Stmt(self.axis_stmt(start))
            }
            // `unit tile` declares a base dimension; `unit league = 3mi`
            // a scale of known units. Contextual like `check`: `unit(x)`,
            // `unit = 4`, `unit.foo` stay expressions.
            TokKind::Ident("unit") if self.unit_intro() => {
                BlockElement::Stmt(self.unit_stmt(start))
            }
            // `needs { doc, net }` declares the page's effect authority.
            // Top-level only: inside a block `needs {x}` stays an apply.
            TokKind::Ident("needs") if self.needs_intro() => {
                BlockElement::Stmt(self.needs_stmt(start))
            }
            // `opp Sky { Sun, Moon }` declares a dichotomy. Contextual: `opp`
            // is a name anywhere a type name and `{` don't follow it.
            TokKind::Ident("opp") if self.opp_intro() => BlockElement::Stmt(self.opp_stmt(start)),
            // r[impl mimas.notation.user]
            // `notation chords(s: str) -> Chords { … }` -- the page's own tagged
            // literal: `chords"…"` lowers to `__lit_chords("…")`, and this is
            // that fn under its working name. Contextual like `opp`: a name
            // and a `(` after the word.
            TokKind::Ident("notation") if self.notation_intro() => {
                self.advance();
                let f = self.function_with(Some("__lit_"));
                let item = Item::new(f.into(), self.location(start), false);
                self.end_item(&item);
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            // `space Politics { econ: Dial, power: Dial }` -- axes taken together
            TokKind::Ident("space") if self.opp_intro() => {
                BlockElement::Stmt(self.space_stmt(start))
            }
            // `multi fn f(a, b) { … }` declares a multiple-dispatch generic:
            // its own body is the fallback every unmatched call reaches.
            // Contextual like `notation`: only `multi fn` reads this way.
            TokKind::Ident("multi") if self.nth(1) == TokKind::Fn => {
                self.advance();
                let mut f = self.function();
                f.role = FnRole::Multi;
                if f.is_method() {
                    self.error_here(Misdirection {
                        src: self.src(),
                        at: self.next_location().into(),
                        msg: "a `multi fn` takes no `self` -- it's a free generic, not an impl method".into(),
                        label: "drop `self`".into(),
                    });
                }
                if self.depth > 0 {
                    self.error(Misdirection {
                        src: self.src(),
                        at: self.location(start).into(),
                        msg: "`multi fn` decls are top-level only".into(),
                        label: "declare the generic at the top of the file".into(),
                    });
                }
                let item = Item::new(f.into(), self.location(start), false);
                self.end_item(&item);
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            // `method f(a: V::Num, b: V::Num) { … }` adds an arm to the `multi
            // fn` named `f`. Contextual like `notation`: a name and a `(`.
            TokKind::Ident("method") if self.method_intro() => {
                self.advance();
                // `Some("")` parses `name(params) -> ret { body }` without a
                // `fn` token -- same trick `notation` uses, with an empty prefix
                let mut f = self.function_with(Some(""));
                f.role = FnRole::Method;
                if f.is_method() {
                    self.error_here(Misdirection {
                        src: self.src(),
                        at: self.next_location().into(),
                        msg: "a `method` takes no `self` -- annotate dispatch types on the ordinary params".into(),
                        label: "drop `self`".into(),
                    });
                }
                if self.depth > 0 {
                    self.error(Misdirection {
                        src: self.src(),
                        at: self.location(start).into(),
                        msg: "`method` decls are top-level only".into(),
                        label: "declare the method at the top of the file".into(),
                    });
                }
                let item = Item::new(f.into(), self.location(start), false);
                self.end_item(&item);
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            kind if kind.starts_item() && !self.at_exact_literal() => {
                let first = self.next;
                let item = self.item();
                self.record_decl(&item, first);
                self.end_item(&item);
                BlockElement::Stmt(self.new_stmt(item, start))
            }
            _ => {
                let expr = self.expr();

                // `^=` went the way of `^` — retired spelling, teach `⊕=`
                // and recover as if it were written.
                if self.at(TokKind::CaretEqual) {
                    self.error_here(Misdirection {
                        src: self.src(),
                        at: self.next_location().into(),
                        msg: "xor-assign is `x ⊕= y` — `^` superscripts a name now, as in `x^2`"
                            .into(),
                        label: "write `⊕=` here".into(),
                    });
                }
                // TODO: assignment does not belong here... I think
                let operator: AssignmentOp = if self.at(TokKind::CaretEqual) {
                    AssignmentOp::XorEqual
                } else {
                    let Ok(operator) = self.peek().try_into() else {
                        return BlockElement::MaybeYield(expr);
                    };
                    operator
                };
                // `c\n= a` is not an assignment: a newline after the target ends the
                // expression, so the `=` belongs to whatever statement starts there.
                if !self.infix_binds() {
                    return BlockElement::MaybeYield(expr);
                }
                let left = if matches!(
                    expr.kind(),
                    ExprKind::Access(_) | ExprKind::Ident(_) | ExprKind::Poison(_)
                ) {
                    expr
                } else {
                    self.error(InvalidAssignmentTarget {
                        src: self.src(),
                        at: expr.location().into(),
                    });
                    Expr::new(Poison.into(), expr.location())
                };
                self.advance();
                let assignment = Assignment::new(left, operator, self.expr());
                let stmt = self.new_stmt(assignment, start);
                self.end_stmt(stmt.location());
                BlockElement::Stmt(stmt)
            }
        }
    }

    fn item(&mut self) -> Item {
        fn inner(parser: &mut Parser) -> ItemKind {
            match parser.peek() {
                TokKind::Fn => parser.function().into(),
                TokKind::Struct => parser.struct_decl().into(),
                TokKind::Pact => parser.pact_decl().into(),
                TokKind::Enum => {
                    let start = parser.next_start();
                    parser.enum_decl(start).into()
                }
                TokKind::Impl => parser.impl_decl().into(),
                TokKind::Const => parser.const_decl().into(),
                TokKind::Use => parser.use_decl().map_or(Poison.into(), Into::into),
                _ => {
                    parser.expected("item");
                    Poison.into()
                }
            }
        }
        let start = self.next_start();
        let attrs = self.attributes();
        let public = self.eat(TokKind::Pub);
        let has_tests = attrs.iter().any(|a| a.name.lexeme == "tests");
        if has_tests {
            if public {
                self.error(TestsNotPublic {
                    src: self.src(),
                    at: self.location(start).into(),
                });
            }
            if !self.at(TokKind::LeftSquare) {
                self.error_here(TestsNeedsList {
                    src: self.src(),
                    at: self.next_location().into(),
                });
            }
        }

        let kind = if has_tests {
            self.tests_list().into()
        } else {
            self.nested(inner).unwrap_or(Poison.into())
        };

        self.validate_attrs(&attrs, &kind, true);
        Item::new(kind, self.location(start), public).with_attrs(attrs)
    }

    /// Zero or more `#[name]` markers sitting immediately before an item.
    fn attributes(&mut self) -> Vec<Attribute> {
        let mut attrs = vec![];
        while self.at(TokKind::Hash) {
            attrs.push(self.attribute());
        }
        attrs
    }

    fn attribute(&mut self) -> Attribute {
        self.bump(TokKind::Hash);
        self.expect(TokKind::LeftSquare);
        let name = self.require_ident();
        self.expect(TokKind::RightSquare);
        Attribute { name }
    }

    fn validate_attrs(&mut self, attrs: &[Attribute], kind: &ItemKind, allow_test: bool) {
        let mut seen_test = false;
        let mut seen_tests = false;
        for attr in attrs {
            match attr.name.lexeme.as_str() {
                "test" => {
                    if seen_test {
                        self.error(DuplicateAttribute {
                            src: self.src(),
                            at: attr.name.location.into(),
                            name: attr.name.lexeme.clone(),
                        });
                    }
                    seen_test = true;
                    match kind {
                        ItemKind::Function(f) if allow_test => {
                            if f.is_method() {
                                self.error(TestOnMethod {
                                    src: self.src(),
                                    at: attr.name.location.into(),
                                });
                            }
                            if !f.parameters.is_empty() {
                                self.error(TestTakesParameters {
                                    src: self.src(),
                                    at: attr.name.location.into(),
                                });
                            }
                        }
                        ItemKind::Function(_) => self.error(TestOnMethod {
                            src: self.src(),
                            at: attr.name.location.into(),
                        }),
                        _ => self.error(AttributeNotOnFunction {
                            src: self.src(),
                            at: attr.name.location.into(),
                            name: attr.name.lexeme.clone(),
                        }),
                    }
                }
                "tests" => {
                    if seen_tests {
                        self.error(DuplicateAttribute {
                            src: self.src(),
                            at: attr.name.location.into(),
                            name: attr.name.lexeme.clone(),
                        });
                    }
                    seen_tests = true;
                    match kind {
                        ItemKind::Tests(_) if allow_test => {}
                        ItemKind::Tests(_) => self.error(TestOnMethod {
                            src: self.src(),
                            at: attr.name.location.into(),
                        }),
                        _ => self.error(TestsNeedsList {
                            src: self.src(),
                            at: attr.name.location.into(),
                        }),
                    }
                }
                // A mark on a top-level function, for `fns!(name)` to read.
                // It is judged at the end of the source: a mark nothing
                // reads is an unknown attribute.
                _ if allow_test && matches!(kind, ItemKind::Function(f) if !f.is_method()) => {
                    self.pending_marks
                        .push((attr.name.lexeme.clone(), attr.name.location));
                }
                _ => self.error(UnknownAttribute {
                    src: self.src(),
                    at: attr.name.location.into(),
                    name: attr.name.lexeme.clone(),
                }),
            }
        }
    }

    /// `#[tests] [ expr, expr, ... ]` -- each expression is a check the inspector runs.
    fn tests_list(&mut self) -> Tests {
        self.expect(TokKind::LeftSquare);
        self.in_group(|p| {
            if p.at(TokKind::RightSquare) {
                p.error(TestsEmpty {
                    src: p.src(),
                    at: p.next_location().into(),
                });
            }
            let mut cases = vec![];
            let mut names = std::collections::HashSet::new();
            loop {
                let inner = p.expr();
                let name = unique_test_name(&mut names, p.source_of(&inner));
                let location = inner.location();
                cases.push(TestCase {
                    id: NodeId::new(),
                    name: Ident::new(name, location),
                    expr: inner,
                });
                if !p.eat(TokKind::Comma) {
                    p.expect(TokKind::RightSquare);
                    break;
                }
                if p.eat(TokKind::RightSquare) {
                    break;
                }
            }
            Tests::new(cases)
        })
    }

    /// One-token lookahead for the contextual `check` keyword. `check` opens a check
    /// when something follows *on the same line* — tokens that could instead continue
    /// `check` as an ordinary expression don't qualify: `check(x)` is a call,
    /// `check - 1` a subtraction, `check == y` an equality. A next-line token is
    /// never taken — `check` alone on a line is an identifier, so `check\n    f()`
    /// doesn't steal the following statement. (A same-line `:`/`{` also routes here;
    /// the retired list forms get a pointed error, not a silent parse.)
    // r[impl check.contextual]
    fn check_intro(&self) -> bool {
        let after = self.meta_at(1);
        if after.newline {
            return false;
        }
        match self.nth(1) {
            TokKind::Colon | TokKind::LeftBrace => true,
            TokKind::Ident(_)
            | TokKind::FString(_)
            | TokKind::Tagged(..)
            | TokKind::Bang
            | TokKind::NotWord => true,
            // `check ∑ xs == 6` -- a prefix reduction opens one like a name does
            TokKind::Sum | TokKind::Prod | TokKind::Inf | TokKind::Sup => true,
            TokKind::RightArrow | TokKind::LeftArrow | TokKind::UpArrow | TokKind::DownArrow
            | TokKind::CircledDot | TokKind::CircledTimes
            | TokKind::Rewind | TokKind::FastForward => true,
            // `check (a * b).same(c)`, `check [1 2] == xs`, `check -x < 0`:
            // a gap after the word opens a check, as it does before a name;
            // `check(x)` and `check[i]` touching the word stay a call and
            // an index
            TokKind::LeftParenthesis
            | TokKind::LeftSquare
            | TokKind::Minus => !after.glued,
            kind => Literal::try_from(kind).is_ok(),
        }
    }

    /// Inside a pact, `check` opens a law when an expression follows on the
    /// same line -- a closure (`check |x| …`), or anything else, which
    /// `pact_law` then explains.
    fn law_intro(&self) -> bool {
        !self.meta_at(1).newline
            && (matches!(self.nth(1), TokKind::Pipe | TokKind::DoublePipe) || self.check_intro())
    }

    /// One-token lookahead for the contextual `needs` keyword: `needs {`
    /// at top level opens the page's effect-authority declaration. Inside
    /// a block `needs` stays an ordinary name — `needs {x}` applies it.
    fn needs_intro(&self) -> bool {
        self.depth == 0
            && matches!(self.nth(1), TokKind::LeftBrace)
            && !self.meta_at(1).newline
    }

    /// `needs { doc, net, … };` — the page's declared effect authority, the
    /// same names `granting {…}` writes (`Fx::named` validates). `needs {}`
    /// declares the page pure. The grades pass checks the page's inferred
    /// `Fx` against this set at solve and bounds `granting` clauses by it.
    // r[impl mimas.fx.needs]
    fn needs_stmt(&mut self, start: usize) -> Stmt {
        self.advance(); // `needs`
        self.bump(TokKind::LeftBrace);
        let names = self.list(TokKind::RightBrace, TokKind::is_ident, Self::require_ident);
        for name in &names {
            if shared::Fx::named(&name.lexeme).is_none() {
                self.error(Expected {
                    src: self.src(),
                    at: name.location.into(),
                    what: "an effect name (`doc`, `net`, `rng`, `yield`, `io`, `time`, `any`, `unaudited`)",
                });
            }
        }
        let stmt = self.new_stmt(Needs::new(names), start);
        self.end_stmt(stmt.location());
        stmt
    }

    /// One-token lookahead for the contextual `with` keyword: `with <name> =`
    /// on one line opens a scoped effect region (`with rng = 7 { … }`); a
    /// newline or any other shape (`with(x)`, `with = 4`, `with` alone) keeps
    /// `with` an ordinary expression.
    // r[impl mimas.fx.with]
    fn with_intro(&self) -> bool {
        !self.meta_at(1).newline
            && self.nth(1).is_ident()
            && self.nth(2) == TokKind::Equal
            && !self.meta_at(2).newline
    }

    /// `with rng = <spec> { … }` — installs the spec's state on the `rng`
    /// fixture for the body's dynamic extent and restores the enclosing one
    /// after. Desugars at parse to ordinary calls, so no new syntax node
    /// reaches solve or codegen:
    ///
    /// ```mimas
    /// { __fx_rng_push(<spec>); let __with_N = { … }; __fx_rng_pop(); __with_N }
    /// ```
    ///
    /// The `#[effects(rng)]` push/pop keep `rng` in the caller's inferred set —
    /// a handler channels the authority, it doesn't launder it. The fixture's
    /// scope stack restores across a hard fault, but an exit that skips the
    /// pop *without* unwinding a frame — `return`, `raise`, `?`, `break`,
    /// `continue` — is refused here: a scoped region a jump can escape
    /// silently isn't scoped.
    // r[impl mimas.fx.with]
    fn with_expr(&mut self, start: usize) -> Expr {
        self.advance(); // `with`
        let name = self.require_ident();
        self.expect(TokKind::Equal);
        // like a loop head: `{` after the spec opens the body, never a literal
        let spec = self.with_juxta(Juxta::Off, |p| p.struct_literals(false, Self::expr));
        if !self.at(TokKind::LeftBrace) {
            self.error_here(Misdirection {
                src: self.src(),
                at: self.next_location().into(),
                msg: format!("`with {} = <spec>` wants a `{{` body", name.lexeme),
                label: "the scoped region goes here".into(),
            });
            return self.poison_expr(start);
        }
        let body = self.block();
        match shared::Fx::named(&name.lexeme) {
            Some(fx) if fx == shared::Fx::RNG => {}
            Some(_) => self.error(Misdirection {
                src: self.src(),
                at: name.location.into(),
                msg: format!(
                    "`with` can't install `{}` yet — only `rng` has a scoped state",
                    name.lexeme
                ),
                label: "`with rng = …` is the working spelling".into(),
            }),
            None => self.error(Misdirection {
                src: self.src(),
                at: name.location.into(),
                msg: format!(
                    "`{}` isn't an effect — `needs`/`granting` write `doc`, `net`, `rng`, `yield`, `io`, `time`",
                    name.lexeme
                ),
                label: "the effect the `with` installs".into(),
            }),
        }
        if let Some((at, exit)) = with_region_escape(&body) {
            self.error(Misdirection {
                src: self.src(),
                at: at.into(),
                msg: format!(
                    "`{exit}` can't leave a `with` region — the handler's restore would be skipped"
                ),
                label: format!("handle it inside, or restructure so the `{exit}` sits outside"),
            });
        }
        self.with_desugar(&name.lexeme, spec, body, start)
    }

    /// `{ __fx_<fx>_push(spec); let __with_N = body; __fx_<fx>_pop(); __with_N }`
    /// — see [`Self::with_expr`].
    fn with_desugar(&mut self, fx: &str, spec: Expr, body: Expr, start: usize) -> Expr {
        self.with_id += 1;
        let temp = self.ident_at(format!("__with_{}", self.with_id), start);
        let call = |p: &Self, name: &str, args: Vec<Expr>| {
            p.new_expr(
                Call::new(
                    p.new_expr(p.ident_at(name.to_string(), start), start),
                    args.into_iter()
                        .map(|value| Argument { name: None, value })
                        .collect(),
                ),
                start,
            )
        };
        let push = self.new_stmt(
            StmtKind::Expr(call(self, &format!("__fx_{fx}_push"), vec![spec])),
            start,
        );
        let bind = self.synthetic_let(temp.clone(), body, start);
        let pop = self.new_stmt(
            StmtKind::Expr(call(self, &format!("__fx_{fx}_pop"), vec![])),
            start,
        );
        self.new_expr(
            Block {
                body: vec![push, bind, pop],
                yielded_expr: Some(self.new_expr(temp, start)),
            },
            start,
        )
    }

    /// One-token lookahead for the contextual `unit` keyword: `unit` opens
    /// a declaration when a plain name follows on the same line — `unit(x)`,
    /// `unit = 4`, `unit.foo`, and `unit` alone stay ordinary expressions.
    fn unit_intro(&self) -> bool {
        if self.meta_at(1).newline {
            return false;
        }
        matches!(self.nth(1), TokKind::Ident(_))
    }

    /// `unit tile;` claims a new base dimension; `unit league = 3mi;` a
    /// scale of known units. The pre-parse scan already registered the
    /// name so `18tile` could lex as a quantity — this reports bad specs
    /// and name collisions for real.
    fn unit_stmt(&mut self, start: usize) -> Stmt {
        let kw = self.advance(); // `unit`
        if self.depth > 0 {
            self.error(Misdirection {
                src: self.src(),
                at: Location::from(kw.location()).into(),
                msg: "`unit` decls are top-level only".into(),
                label: "declare units at the top of the file".into(),
            });
        }
        let name = self.require_ident();
        let name_loc: Location = name.location;
        if !valid_unit_name(&name.lexeme) {
            self.error(Misdirection {
                src: self.src(),
                at: name_loc.into(),
                msg: format!("`{}` can't be a unit", name.lexeme),
                label: "unit names are one word — letters, digits, `_`".into(),
            });
        }
        let mut spec_text = None;
        if self.at(TokKind::Equal) && !self.at_line_start() {
            self.advance();
            let rhs_start = self.next_start();
            let mut rhs = vec![];
            while !self.at_line_start() && self.peek().is_unit_spec() {
                rhs.push(self.advance().kind());
            }
            spec_text = Some(
                self.src
                    .inner()
                    .get(rhs_start..self.cursor())
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
            );
            match unit_spec(&rhs) {
                Some(spec) => {
                    if let Err(msg) = shared::units::declare(&name.lexeme, Some(spec)) {
                        self.error(Misdirection {
                            src: self.src(),
                            at: name_loc.into(),
                            msg,
                            label: name.lexeme.clone(),
                        });
                    }
                }
                None => self.error(Misdirection {
                    src: self.src(),
                    at: self.location(rhs_start).into(),
                    msg: format!(
                        "`{}` isn't a quantity",
                        spec_text.clone().unwrap_or_default()
                    ),
                    label: "a number and unit names — `3mi`, `1852 m/h`, `12`".into(),
                }),
            }
        } else if let Err(msg) = shared::units::declare(&name.lexeme, None) {
            self.error(Misdirection {
                src: self.src(),
                at: name_loc.into(),
                msg,
                label: name.lexeme.clone(),
            });
        }
        let stmt = self.new_stmt(Unit::new(name, spec_text), start);
        self.end_stmt(stmt.location());
        stmt
    }

    /// `axis NAME (NEG, POS);` -- an axis and its direction words, negative end
    /// first. The pre-parse scan already registered it so `richer 5usd` parses
    /// anywhere in the file; this reports a bad declaration for real.
    fn axis_stmt(&mut self, start: usize) -> Stmt {
        let kw = self.advance(); // `axis`
        if self.depth > 0 {
            self.error(Misdirection {
                src: self.src(),
                at: Location::from(kw.location()).into(),
                msg: "`axis` decls are top-level only".into(),
                label: "declare axes at the top of the file".into(),
            });
        }
        let name = self.require_ident();
        let name_loc: Location = name.location;
        let mut words = None;
        if self.eat(TokKind::LeftParenthesis) {
            let neg = self.require_ident();
            self.expect(TokKind::Comma);
            let pos = self.require_ident();
            self.expect(TokKind::RightParenthesis);
            words = Some((neg, pos));
        } else {
            self.error_here(Misdirection {
                src: self.src(),
                at: self.next_location().into(),
                msg: format!("`axis {}` needs its two directions", name.lexeme),
                label: "write `(negative, positive)` here, as in `axis y (up, down)`".into(),
            });
        }
        let spec = words.as_ref().map(|(n, p)| format!("({}, {})", n.lexeme, p.lexeme));
        if let Some((neg, pos)) = &words
            && let Err(msg) = shared::units::declare_axis(&name.lexeme, &neg.lexeme, &pos.lexeme)
        {
            self.error(Misdirection {
                src: self.src(),
                at: name_loc.into(),
                msg,
                label: name.lexeme.clone(),
            });
        }
        let stmt = self.new_stmt(Unit::new(name, spec), start);
        self.end_stmt(stmt.location());
        stmt
    }

    /// A `check` is one line: `check <expr>` — `check f(2) == 3` — becoming one
    /// [TestCase] in a [Tests] item, so `Vm::run_tests` sees it exactly like a
    /// `#[tests]` entry. `,`/`;` chain more checks on the same line. The
    /// expression may be a block: `check { let a = 2; a * 3 } == 6`.
    fn check_block(&mut self) -> Tests {
        self.advance();
        let mut names = std::collections::HashSet::new();
        let mut cases = vec![self.check_case(&mut names)];
        // `,`/`;` chain more checks on the same line.
        while self.eat(TokKind::Comma) || self.eat(TokKind::SemiColon) {
            if !self.check_chains() {
                break;
            }
            cases.push(self.check_case(&mut names));
        }
        Tests::new(cases)
    }

    /// One check: the `expr`, wrapped in `assert`'s operand-reporting rewrite.
    fn check_case(&mut self, names: &mut std::collections::HashSet<String>) -> TestCase {
        // a check is always an expression: a block that opens it can take
        // an operator after it (`check { let a = 2; a * 3 } == 6`), where a
        // statement-start block ends the expression
        let expr = if self.at(TokKind::LeftBrace) { self.null_coalecence() } else { self.expr() };
        let name = unique_test_name(names, self.source_of(&expr));
        let location = expr.location();
        // `check` gets `assert`'s operand rewrite: a failed `check x == 40`
        // panics "assertion failed: 39 == 40" -- the got value, not just
        // the check's expected text.
        let start = expr.span().start();
        let asserted = self.rewrite_assert(start, expr);
        let body = self.new_expr(
            Block {
                body: vec![self.new_stmt(StmtKind::Expr(asserted), start)],
                yielded_expr: Some(self.new_expr(Literal::True, start)),
            },
            start,
        );
        TestCase {
            id: NodeId::new(),
            name: Ident::new(name, location),
            expr: body,
        }
    }

    /// Whether a `,`/`;` on a `check` line chains another check — the next token
    /// must sit on the same line and be able to open one.
    fn check_chains(&mut self) -> bool {
        if self.at(TokKind::Eof) {
            return false;
        }
        !self.at_line_start() && self.peek().starts_expr()
    }

    /// The source text an expr was parsed from, for a check's display name.
    fn source_of(&self, expr: &Expr) -> String {
        let span = expr.span();
        self.src
            .inner()
            .get(span.start()..span.end())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map_or_else(|| expr.to_string(), str::to_string)
    }

    fn use_decl(&mut self) -> Option<Use> {
        self.bump(TokKind::Use);

        // detailed error, mostly as a humorous nod to rustc
        if self.at(TokKind::Star) {
            self.reject(UseAllModules {
                src: self.src(),
                at: self.next_location().into(),
            });
            return None;
        }

        // `use "page-name";` asks the host for another script (see `Use::Host`)
        if let TokKind::String(lit) = self.peek() {
            self.advance();
            // r[impl mimas.use.page]
            let name = lit.trim_matches('"').to_string();
            // r[impl mimas.use.page.granting]
            let grants = self.granting_clause();
            return Some(Use::Host(HostUse { name, grants }));
        }

        let mut path = vec![self.require_ident()];
        while self.eat(TokKind::DoubleColon) {
            match self.peek() {
                TokKind::Ident(_) => path.push(self.require_ident()),
                TokKind::Star => {
                    self.bump(TokKind::Star);
                    return Some(Use::All(path));
                }
                TokKind::LeftBrace => {
                    self.bump(TokKind::LeftBrace);
                    let items =
                        self.list(TokKind::RightBrace, TokKind::is_ident, Self::require_ident);
                    return Some(Use::Multi(path, items));
                }
                _ => {
                    self.expected("import");
                    return None;
                }
            }
        }
        // the path ran out, so its last segment is the import
        let item = path.pop()?;
        Some(Use::Singular(path, item))
    }

    /// The `granting { doc, net, … }` clause a host `use` may carry on
    /// the same line — `None` when no spaced `granting` word follows the
    /// quoted name. `granting {}` grants nothing: it asserts the
    /// include is pure.
    // r[impl mimas.use.page.granting]
    fn granting_clause(&mut self) -> Option<Vec<Ident>> {
        if self.at_line_start() || self.glued() {
            return None;
        }
        let TokKind::Ident(word) = self.peek() else {
            return None;
        };
        if word != "granting" {
            return None;
        }
        self.advance();
        if !self.at(TokKind::LeftBrace) {
            self.expected("a `{ … }` grant set");
            return Some(vec![]);
        }
        self.bump(TokKind::LeftBrace);
        let grants = self.list(TokKind::RightBrace, TokKind::is_ident, Self::require_ident);
        for grant in &grants {
            if shared::Fx::named(&grant.lexeme).is_none() {
                self.error(Expected {
                    src: self.src(),
                    at: grant.location.into(),
                    what: "an effect name (`doc`, `net`, `rng`, `yield`, `io`, `time`, `any`, `unaudited`)",
                });
            }
        }
        Some(grants)
    }

    fn module_stmt(&mut self) -> Stmt {
        let start = self.next_start();
        self.bump(TokKind::Module);

        // We either are given an ident or an @, which maps to the file name. The lexer's name
        // can be a full path (that's what diagnostics render), so take the stem here.
        let module = if self.eat(TokKind::At) {
            let stem = std::path::Path::new(&self.file_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&self.file_name)
                .to_string();
            Module::new(Ident::new(stem, self.location(start)))
        } else {
            // `module a::b;` -- nested path, flattened into one `::`-joined name (the solver
            // splits it back out through ensure_module_path)
            let mut ident = self.require_ident();
            while self.eat(TokKind::DoubleColon) {
                let segment = self.require_ident();
                ident.lexeme = format!("{}::{}", ident.lexeme, segment.lexeme);
            }
            ident.location = self.location(start);
            Module::new(ident)
        };

        let stmt = self.new_stmt(module, start);
        self.end_stmt(stmt.location());
        stmt
    }

    fn let_stmt(&mut self) -> Stmt {
        let start = self.next_start();
        self.bump(TokKind::Let);
        // `let mut x = ...` -- a Rust habit. Every `let` is already reassignable, so teach
        // that and recover as if `mut` weren't there. (`let mut = 1` still binds `mut`.)
        if matches!(self.peek(), TokKind::Ident("mut")) && self.nth(1).starts_pattern() {
            self.error(Misdirection {
                src: self.src(),
                at: self.next_location().into(),
                msg: "mimas has no `mut` -- every `let` binding is reassignable".into(),
                label: "drop `mut`: write `let x = ...`".into(),
            });
            self.advance();
        }
        let left = self.pattern();
        let annotation = self.eat(TokKind::Colon).then(|| self.annotation());
        self.expect(TokKind::Equal);
        let right = self.expr();
        let else_branch = self.eat(TokKind::Else).then(|| self.expr());
        let stmt = self.new_stmt(
            Let {
                left,
                annotation,
                right,
                else_branch,
            },
            start,
        );
        self.end_stmt(stmt.location());
        stmt
    }

    fn const_decl(&mut self) -> Const {
        self.bump(TokKind::Const);
        let left = self.pattern();
        let annotation = self.eat(TokKind::Colon).then(|| self.annotation());
        self.expect(TokKind::Equal);
        let right = self.expr();
        Const {
            left,
            annotation,
            right,
        }
    }

    /// Parses one expression. Anything that fails to parse comes back as [Poison], with its
    /// error in [Self::errors]. See [Self::into_ast] for what that means for the caller.
    pub fn expr(&mut self) -> Expr {
        let expr = self.block_body();
        self.chain_after_block(expr)
    }

    /// The expression dispatch without chaining a trailing block. Control-flow *bodies* parse
    /// through here so postfix after the body (`if c {...} else {...}.foo()`) attaches to the whole
    /// construct, not the inner block; `expr` wraps this with `chain_after_block`.
    fn block_body(&mut self) -> Expr {
        fn inner(parser: &mut Parser) -> Expr {
            match parser.peek() {
                TokKind::Pipe | TokKind::DoublePipe => parser.closure(),
                TokKind::Loop => parser.loop_expr(),
                TokKind::While => parser.while_expr(),
                TokKind::For => parser.for_in(),
                TokKind::If => parser.if_expr(),
                TokKind::Match => parser.match_expr(),
                TokKind::LeftBrace if parser.record_start() => parser.record_literal(),
                TokKind::LeftBrace => parser.block(),
                TokKind::Return => parser.return_expr(),
                TokKind::Raise => parser.raise_expr(),
                TokKind::Break => parser.break_stmt(),
                TokKind::Collect => parser.collect(),
                TokKind::Continue => parser.continue_expr(),

                // No keyword expressions found, so start recursive descent
                _ => parser.null_coalecence(),
            }
        }
        let start = self.next_start();
        if !self.peek().starts_expr() {
            self.expected("expression");
            return self.poison_expr(start);
        }
        self.nested(inner)
            .unwrap_or_else(|| self.poison_expr(start))
    }

    /// An optional `<T, U>` type parameter list after a declaration name. The `<` is
    /// unambiguous here: nothing else can follow the name in decl position.
    /// Each param may carry pact bounds: `<T: Draw, U: Draw + Vary>`.
    fn type_params(&mut self) -> Vec<TypeParam> {
        if !self.eat(TokKind::Less) {
            return vec![];
        }
        self.list(TokKind::Greater, TokKind::is_ident, |p| {
            let name = p.require_ident();
            let mut bounds = vec![];
            if p.eat(TokKind::Colon) {
                bounds.push(p.require_ident());
                while p.eat(TokKind::Plus) {
                    bounds.push(p.require_ident());
                }
            }
            TypeParam::new(name, bounds)
        })
    }

    /// The span covering a `<...>` list already parsed, for rejecting it in positions that
    /// don't take generics.
    fn reject_type_params(&mut self, params: &[TypeParam]) {
        if let (Some(first), Some(last)) = (params.first(), params.last()) {
            self.error(GenericsNotAllowed {
                src: self.src(),
                at: first.location.span.until(last.location.span).into(),
            });
        }
    }

    fn struct_decl(&mut self) -> Struct {
        self.bump(TokKind::Struct);
        let name = self.require_ident();
        let type_params = self.type_params();
        let fields = if self.eat(TokKind::SemiColon) {
            vec![]
        } else if self.eat(TokKind::LeftParenthesis) {
            let mut index = 0;
            self.list(
                TokKind::RightParenthesis,
                |kind| kind == TokKind::Pub || kind.starts_annotation(),
                |p| {
                    let start = p.next_start();
                    let public = p.eat(TokKind::Pub);
                    let annotation = p.annotation();
                    index += 1;
                    StructField {
                        name: FieldKey::Int(index - 1),
                        annotation,
                        public,
                        location: p.location(start),
                    }
                },
            )
        } else if self.expect(TokKind::LeftBrace) {
            self.list(
                TokKind::RightBrace,
                |kind| kind == TokKind::Pub || kind.is_ident(),
                Self::named_field,
            )
        } else {
            vec![]
        };
        Struct {
            name,
            type_params,
            fields,
        }
    }

    /// A `name: Type` field, `pub` or not.
    fn named_field(&mut self) -> StructField {
        let start = self.next_start();
        let public = self.eat(TokKind::Pub);
        let name = self.require_ident();
        self.expect(TokKind::Colon);
        let annotation = self.annotation();
        StructField {
            name: FieldKey::Ident(name),
            annotation,
            public,
            location: self.location(start),
        }
    }

    fn pact_decl(&mut self) -> Pact {
        self.bump(TokKind::Pact);
        let pact_name = self.require_ident();
        // No `<T>` production follows the name: a pact never takes type
        // parameters. Relations between two parameters are declined; the
        // spellings that cover them are a concrete operand, an overload,
        // `rmul`, or a dedicated solver.
        // r[impl mimas.pact.no-params]
        // `pact A: B + C` -- A requires B and C: every implementer of A must
        // also implement them, and their members are reachable through an
        // `A` bound. Same `+` spelling as a param bound.
        // r[impl mimas.pact.requires]
        let mut requires = vec![];
        if self.eat(TokKind::Colon) {
            requires.push(self.require_ident());
            while self.eat(TokKind::Plus) {
                requires.push(self.require_ident());
            }
        }
        let mut items = vec![];
        let mut laws = vec![];
        if !self.expect(TokKind::LeftBrace) {
            return Pact::new(pact_name, requires, items);
        }
        while !self.eat(TokKind::RightBrace) {
            let start = self.next_start();
            match self.peek() {
                TokKind::Pub => self.reject(InvalidPubMarker {
                    src: self.src(),
                    at: self.next_location().into(),
                }),
                TokKind::Const => {
                    self.bump(TokKind::Const);
                    let name = self.require_ident();
                    let annotation = if self.eat(TokKind::Colon) {
                        self.annotation()
                    } else {
                        self.error(PactConstAnnotationRequired {
                            src: self.src(),
                            at: self.next_location().into(),
                        });
                        Annotation::Poison(Poison)
                    };
                    self.end_stmt(self.location(start));
                    items.push(PactItem::Const {
                        name,
                        annotation,
                        location: self.location(start),
                    });
                }
                TokKind::Fn => {
                    let (name, generics, parameters, return_type) = self.function_sig(false, None);
                    self.reject_type_params(&generics);
                    // optional default body -- `{ ... }` after the sig. when present, the trailing
                    // `;` is dropped (block-terminated, like normal fns).
                    let default = self.at(TokKind::LeftBrace).then(|| self.block());
                    if default.is_some() {
                        self.eat(TokKind::SemiColon);
                    } else {
                        self.end_stmt(self.location(start));
                    }
                    items.push(PactItem::Fn {
                        name,
                        parameters,
                        return_type,
                        default,
                        location: self.location(start),
                    });
                }
                // r[impl mimas.pact.law]
                TokKind::Ident("check") if self.law_intro() => {
                    self.advance();
                    if let Some(law) = self.pact_law(&pact_name, laws.len(), start) {
                        laws.push(law);
                    }
                    self.eat(TokKind::SemiColon);
                }
                TokKind::Eof => {
                    self.expect(TokKind::RightBrace);
                    break;
                }
                _ => self.reject_stmt(InvalidPactItem {
                    src: self.src(),
                    at: self.next_location().into(),
                }),
            }
        }
        let mut pact = Pact::new(pact_name, requires, items);
        pact.laws = laws;
        pact
    }

    /// A pact's `check |x, y| body` law, after the `check`: lowered to the
    /// top-level `fn __law_<Pact>_<i><__Law: Pact>(x: __Law, y: __Law) -> bool
    /// { body }` (queued in `pending`), so the body is checked once, against
    /// the pact, and runs on any one implementer's values.
    fn pact_law(&mut self, pact: &Ident, index: usize, start: usize) -> Option<Law> {
        let expr = self.expr();
        let text = self.source_of(&expr);
        let ExprKind::Closure(closure) = expr.kind() else {
            self.error(Misdirection {
                src: self.src(),
                at: expr.location().into(),
                msg: "a pact's `check` is a law over the pact's values".into(),
                label: "write it as a closure, one parameter per value: `check |x| x.turned(↻).turned(↺) == x`"
                    .into(),
            });
            return None;
        };
        let closure = closure.clone();
        let body = self.source_of(&closure.body);
        let at = self.location(start);
        let mut params = vec![];
        let mut bindings = vec![];
        for b in closure.parameters {
            let PatKind::Ident(name) = b.left.kind() else {
                self.error(Misdirection {
                    src: self.src(),
                    at: b.left.location().into(),
                    msg: "a law's parameters are plain names".into(),
                    label: "each one stands for a value of the implementing type".into(),
                });
                return None;
            };
            if b.annotation.is_some() {
                self.error(Misdirection {
                    src: self.src(),
                    at: b.left.location().into(),
                    msg: "a law's parameters take no type".into(),
                    label: format!("each one is a `{}` -- a value of the implementing type", pact.lexeme),
                });
            }
            params.push(name.lexeme.clone());
            let mut binding = Binding::new(name.clone());
            binding.annotation = Some(Annotation::Ty(Ident::new("__Law", at)));
            bindings.push(binding);
        }
        if params.is_empty() {
            self.error(Misdirection {
                src: self.src(),
                at: expr.location().into(),
                msg: "a law needs at least one value to talk about".into(),
                label: "name it: `check |x| …`".into(),
            });
            return None;
        }
        let fn_name = format!("__law_{}_{index}", pact.lexeme);
        // generic over one implementer, so a law may pass one value to
        // another's method (`a.merge(b)` takes a `Self`)
        let func = Function {
            name: Ident::new(fn_name.clone(), at),
            role: FnRole::Fn,
            type_params: vec![TypeParam::new(
                Ident::new("__Law", at),
                vec![Ident::new(pact.lexeme.clone(), at)],
            )],
            parameters: bindings,
            wheres: vec![],
            return_type: Some(Annotation::Kw(lex::TyKw::Bool)),
            body: closure.body,
        };
        self.pending
            .push(self.new_stmt(Item::new(func.into(), at, false), start));
        Some(Law {
            text,
            body,
            params,
            fn_name,
            location: expr.location(),
        })
    }

    fn enum_decl(&mut self, start: usize) -> Enum {
        self.bump(TokKind::Enum);
        let head = self.require_ident();
        let type_params = self.type_params();
        let (members, glyphs) = if self.expect(TokKind::LeftBrace) {
            self.enum_members()
        } else {
            (vec![], vec![])
        };
        // r[impl mimas.enum.glyph]
        // `enum Suit { Spades "♠", … }`: every plain variant carries a glyph,
        // or none does -- an inherent `fn glyph(self) -> str` is synthesized
        let given = glyphs.iter().filter(|g| g.is_some()).count();
        if given > 0 {
            let all_plain = members
                .iter()
                .all(|(_, m)| matches!(m, Member::Struct(f) if f.is_empty()));
            if given != members.len() || !all_plain {
                self.error(Misdirection {
                    src: self.src(),
                    at: self.location(start).into(),
                    msg: "an enum's glyphs cover every plain variant or none".into(),
                    label: "give each variant a glyph: `enum Suit { Spades \"♠\", Hearts \"♥\" }`"
                        .into(),
                });
            } else if self.depth == 1 {
                // `item()` parses its body one level under `nested`, so a
                // top-level enum sees depth 1; a nested one keeps no impl
                let arms: Vec<(Ident, String)> = members
                    .iter()
                    .zip(&glyphs)
                    .map(|((n, _), g)| (n.clone(), g.clone().unwrap()))
                    .collect();
                let at = self.location(start);
                let item = self.glyph_fn(&head, &arms, start);
                let imp = Impl::new(Ident::new(head.lexeme.clone(), at), None, None, vec![item]);
                self.pending
                    .push(self.new_stmt(Item::new(imp.into(), at, false), start));
            }
        }
        Enum {
            head,
            type_params,
            members,
        }
    }

    /// `fn glyph(self) -> str { match self { T::A => "…", … } }` for the
    /// synthesized impl of an `opp` or `enum` whose variants carry glyphs.
    fn glyph_fn(&self, head: &Ident, arms: &[(Ident, String)], start: usize) -> Item {
        let at = self.location(start);
        let id = |name: &str| Ident::new(name, at);
        let cases = arms
            .iter()
            .map(|(from, text)| {
                MatchCase::new(
                    Pat::new(
                        PatKind::Variant(Box::new(self.opp_path(head, from, start))),
                        at,
                    ),
                    None,
                    self.new_expr(Literal::String(text.clone()), start),
                )
            })
            .collect();
        let pick = Match::new(self.new_expr(id("self"), start), cases, false);
        let body = self.new_expr(
            Block {
                body: vec![],
                yielded_expr: Some(self.new_expr(pick, start)),
            },
            start,
        );
        Item::new(
            Function {
                name: id("glyph"),
                role: FnRole::Fn,
                type_params: vec![],
                parameters: vec![Binding::new(id("self"))],
                wheres: vec![],
                return_type: Some(Annotation::Kw(lex::TyKw::Str)),
                body,
            }
            .into(),
            at,
            false,
        )
    }

    /// An enum's `{ … }` members, after the `{`, plus the glyph string a
    /// plain variant may carry (`Spades "♠"`), parallel to the members.
    fn enum_members(&mut self) -> (Vec<(Ident, Member)>, Vec<Option<String>>) {
        let mut glyphs = Vec::new();
        let members = self.list(TokKind::RightBrace, TokKind::is_ident, |p| {
            let name = p.require_ident();
            let glyph = match p.peek() {
                TokKind::String(s) => {
                    p.advance();
                    Some(chompy::utils::unescape(s, &['\\'], &['"']))
                }
                _ => None,
            };
            glyphs.push(glyph);
            let member = if p.eat(TokKind::LeftParenthesis) {
                Member::Tuple(p.list(
                    TokKind::RightParenthesis,
                    TokKind::starts_annotation,
                    Self::annotation,
                ))
            } else if p.eat(TokKind::LeftBrace) {
                Member::Struct(
                    p.list(TokKind::RightBrace, TokKind::is_ident, |p| StructField {
                        public: true,
                        ..p.named_field()
                    }),
                )
            } else {
                Member::Struct(vec![])
            };
            (name, member)
        });
        (members, glyphs)
    }

    /// `notation name(` — a name and a paren after the word.
    fn notation_intro(&self) -> bool {
        matches!(self.nth(1), TokKind::Ident(_)) && matches!(self.nth(2), TokKind::LeftParenthesis)
    }

    /// `method name(` — the same shape `notation` opens on: the word, a name,
    /// a `(`. `method` stays a name everywhere else.
    fn method_intro(&self) -> bool {
        self.notation_intro()
    }

    /// `opp Name {` — a type name and a brace after the word.
    fn opp_intro(&self) -> bool {
        matches!(self.nth(1), TokKind::Ident(_)) && matches!(self.nth(2), TokKind::LeftBrace)
    }

    /// `opp Sky { Sun, Moon }` — a dichotomy, as basic as an enum: two ends
    /// and a flip. It is the enum of its two ends plus a synthesized `impl
    /// Opp for Sky` (the pact every program has) whose `inv` swaps
    /// them, so `!Sky::Sun == Sky::Moon` and `involutive`-style code over
    /// `Opp` takes it. `bool` is the first opp. Top-level only.
    // r[impl mimas.opp.decl]
    fn opp_stmt(&mut self, start: usize) -> Stmt {
        let kw = self.advance(); // `opp`
        if self.depth > 0 {
            self.error(Misdirection {
                src: self.src(),
                at: Location::from(kw.location()).into(),
                msg: "`opp` decls are top-level only".into(),
                label: "declare the pair at the top of the file".into(),
            });
        }
        let head = self.require_ident();
        // r[impl mimas.opp.glyph]
        // each end is a plain name, optionally followed by the one-character
        // glyph its pictures draw with: `opp Sky { Sun "☀", Moon "☾" }`
        let ends: Vec<(Ident, Option<String>)> = if self.expect(TokKind::LeftBrace) {
            self.list(TokKind::RightBrace, TokKind::is_ident, |p| {
                let name = p.require_ident();
                let glyph = match p.peek() {
                    TokKind::String(s) => {
                        p.advance();
                        Some(chompy::utils::unescape(s, &['\\'], &['"']))
                    }
                    _ => None,
                };
                (name, glyph)
            })
        } else {
            vec![]
        };
        let pair = ends.len() == 2;
        if !pair {
            self.error(Misdirection {
                src: self.src(),
                at: self.location(start).into(),
                msg: "an `opp` has exactly two ends".into(),
                label: "two plain names, opposite each other: `opp Sky { Sun, Moon }`".into(),
            });
        }
        let glyphs = match ends.as_slice() {
            [(_, Some(a)), (_, Some(b))] => Some((a.clone(), b.clone())),
            [(_, Some(_)), (_, None)] | [(_, None), (_, Some(_))] => {
                self.error(Misdirection {
                    src: self.src(),
                    at: self.location(start).into(),
                    msg: "an `opp`'s glyphs come in a pair".into(),
                    label: "give both ends a glyph or neither: `opp Sky { Sun \"☀\", Moon \"☾\" }`"
                        .into(),
                });
                None
            }
            _ => None,
        };
        if pair && self.depth == 0 {
            let imp = self.opp_impl(&head, &ends[0].0, &ends[1].0, glyphs, start);
            self.pending.push(imp);
        }
        let members: Vec<(Ident, Member)> = ends
            .into_iter()
            .map(|(name, _)| (name, Member::Struct(vec![])))
            .collect();
        let item = Item::new(
            Enum {
                head,
                type_params: vec![],
                members,
            }
            .into(),
            self.location(start),
            false,
        );
        self.new_stmt(item, start)
    }

    /// `impl Opp for T { fn inv(self) -> T { match self { T::A => T::B,
    /// T::B => T::A } } }`, every node located at the `opp` declaration --
    /// plus `fn glyph(self) { match self { T::A => "…", T::B => "…" } }`
    /// when the ends carry glyphs (else the pact's default, `""`, stands).
    fn opp_impl(
        &self,
        head: &Ident,
        a: &Ident,
        b: &Ident,
        glyphs: Option<(String, String)>,
        start: usize,
    ) -> Stmt {
        let at = self.location(start);
        let id = |name: &str| Ident::new(name, at);
        let case = |from: &Ident, to: &Ident| {
            MatchCase::new(
                Pat::new(
                    PatKind::Variant(Box::new(self.opp_path(head, from, start))),
                    at,
                ),
                None,
                self.opp_path(head, to, start),
            )
        };
        let flip = Match::new(
            self.new_expr(id("self"), start),
            vec![case(a, b), case(b, a)],
            false,
        );
        let body = self.new_expr(
            Block {
                body: vec![],
                yielded_expr: Some(self.new_expr(flip, start)),
            },
            start,
        );
        let inv = Function {
            name: id("inv"),
            role: FnRole::Fn,
            type_params: vec![],
            parameters: vec![Binding::new(id("self"))],
            wheres: vec![],
            return_type: Some(Annotation::Ty(id(&head.lexeme))),
            body,
        };
        let mut items = vec![Item::new(inv.into(), at, false)];
        if let Some((ga, gb)) = glyphs {
            items.push(self.glyph_fn(head, &[(a.clone(), ga), (b.clone(), gb)], start));
        }
        let imp = Impl::new(id(&head.lexeme), Some(id("Opp")), None, items);
        self.new_stmt(Item::new(imp.into(), at, false), start)
    }

    /// `space Politics { econ: Dial, power: Dial }` -- axes taken together. It is
    /// the struct of its fields plus a synthesized `impl Opp` that flips every
    /// field (`Politics { econ = self.econ.inv(), … }`), so a space is an axis
    /// itself and spaces nest. The struct carries `#[space]`: the solver checks
    /// every field is an axis -- a type implementing `Opp`: an opp, `bool`, a
    /// number (flip is negation), a `Dial`, another space. Top-level only.
    // r[impl mimas.space.decl]
    fn space_stmt(&mut self, start: usize) -> Stmt {
        let kw = self.advance(); // `space`
        if self.depth > 0 {
            self.error(Misdirection {
                src: self.src(),
                at: Location::from(kw.location()).into(),
                msg: "`space` decls are top-level only".into(),
                label: "declare the space at the top of the file".into(),
            });
        }
        let name = self.require_ident();
        let fields = if self.expect(TokKind::LeftBrace) {
            self.list(
                TokKind::RightBrace,
                |kind| kind == TokKind::Pub || kind.is_ident(),
                Self::named_field,
            )
        } else {
            vec![]
        };
        if fields.is_empty() {
            self.error(Misdirection {
                src: self.src(),
                at: self.location(start).into(),
                msg: "a `space` needs at least one axis".into(),
                label: "name its axes: `space Compass { ns: Dial, ew: Dial }`".into(),
            });
        } else if self.depth == 0 {
            let imp = self.space_impl(&name, &fields, start);
            self.pending.push(imp);
        }
        let at = self.location(start);
        let item = Item::new(
            Struct {
                name: name.clone(),
                type_params: vec![],
                fields,
            }
            .into(),
            at,
            false,
        )
        .with_attrs(vec![Attribute {
            name: Ident::new("space", at),
        }]);
        self.new_stmt(item, start)
    }

    /// `impl Opp for S { fn inv(self) -> S { S { f = self.f.inv(), … } } }`, every
    /// node located at the `space` declaration.
    fn space_impl(&self, name: &Ident, fields: &[StructField], start: usize) -> Stmt {
        let at = self.location(start);
        let id = |n: &str| Ident::new(n, at);
        let dot = |left: Expr, right: &str| {
            self.new_expr(
                Access::Dot {
                    left,
                    right: self.new_expr(id(right), start),
                    kind: AccessKind::Direct,
                },
                start,
            )
        };
        let flipped: Vec<(FieldKey, Expr)> = fields
            .iter()
            .filter_map(|f| match &f.name {
                FieldKey::Ident(field) => {
                    let value = dot(self.new_expr(id("self"), start), &field.lexeme);
                    let call = self.new_expr(Call::new(dot(value, "inv"), vec![]), start);
                    Some((FieldKey::Ident(id(&field.lexeme)), call))
                }
                FieldKey::Int(_) => None,
            })
            .collect();
        let literal = self.new_expr(
            Literal::Struct(StructLiteral {
                name: self.new_expr(id(&name.lexeme), start),
                fields: flipped,
                base: None,
            }),
            start,
        );
        let body = self.new_expr(
            Block {
                body: vec![],
                yielded_expr: Some(literal),
            },
            start,
        );
        let inv = Function {
            name: id("inv"),
            role: FnRole::Fn,
            type_params: vec![],
            parameters: vec![Binding::new(id("self"))],
            wheres: vec![],
            return_type: Some(Annotation::Ty(id(&name.lexeme))),
            body,
        };
        let imp = Impl::new(
            id(&name.lexeme),
            Some(id("Opp")),
            None,
            vec![Item::new(inv.into(), at, false)],
        );
        self.new_stmt(Item::new(imp.into(), at, false), start)
    }

    /// `T::V` as an expression, located at `start`'s declaration.
    fn opp_path(&self, head: &Ident, member: &Ident, start: usize) -> Expr {
        let at = self.location(start);
        self.new_expr(
            Access::DoubleColon {
                left: self.new_expr(Ident::new(head.lexeme.clone(), at), start),
                right: Ident::new(member.lexeme.clone(), at),
            },
            start,
        )
    }

    fn impl_decl(&mut self) -> Impl {
        self.bump(TokKind::Impl);
        let first = self.require_ident();
        // `impl Foo<T>` would mean specializing one instantiation -- not supported. `impl Foo`
        // on a generic type implements all of them at once, with the adt's params in scope.
        let generics = self.type_params();
        self.reject_type_params(&generics);
        // `impl Foo {}` (inherent) vs `impl Pact for Foo {}` (pact impl). After the first ident,
        // a `for` token disambiguates -- first becomes the pact, second becomes the target.
        let (pact, target) = if self.eat(TokKind::For) {
            // a pact may be implemented for a primitive (`impl Opp for bool`)
            let target = match self.peek() {
                TokKind::TyKw(ty) => {
                    let at = self.next_location();
                    self.advance();
                    Ident::new(ty.to_string(), at)
                }
                _ => self.require_ident(),
            };
            let generics = self.type_params();
            self.reject_type_params(&generics);
            (Some(first), target)
        } else {
            (None, first)
        };
        // `impl X for T via Y` -- the impl forwards members it doesn't write
        // to `T`'s `Y` impl of the same name. `via` is
        // contextual: an impl target could itself be named `via`, so it's
        // only read when it sits right after the target of a pact impl.
        // r[impl mimas.impl.via]
        let via = if pact.is_some() && self.at(TokKind::Ident("via")) {
            self.advance();
            Some(self.require_ident())
        } else {
            None
        };
        let mut items = vec![];
        // a via impl may carry no braces at all
        if via.is_some() && !self.at(TokKind::LeftBrace) {
            return Impl::new(target, pact, via, items);
        }
        if !self.expect(TokKind::LeftBrace) {
            return Impl::new(target, pact, via, items);
        }
        while !self.eat(TokKind::RightBrace) {
            let kind = if self.at(TokKind::Pub) {
                self.nth(1)
            } else {
                self.peek()
            };
            match kind {
                TokKind::Ident("subscript") => {
                    let public = self.eat(TokKind::Pub);
                    items.extend(self.accessor_items(true, public));
                }
                TokKind::Const | TokKind::Fn => {
                    let item = self.item();
                    self.end_item(&item);
                    items.push(item);
                }
                TokKind::Eof => {
                    self.expect(TokKind::RightBrace);
                    break;
                }
                _ => self.reject_stmt(InvalidImplItem {
                    src: self.src(),
                    at: self.next_location().into(),
                }),
            }
        }
        Impl::new(target, pact, via, items)
    }

    /// A function up to its body. A pact's signatures don't get `defaults`. Returns the
    /// name, any `<T, U>` type parameters, the value parameters, and the return annotation.
    fn function_sig(
        &mut self,
        defaults: bool,
        prefix: Option<&str>,
    ) -> (Ident, Vec<TypeParam>, Vec<Binding>, Option<Annotation>) {
        if prefix.is_none() {
            self.bump(TokKind::Fn);
        }
        let mut name = self.require_ident();
        if let Some(prefix) = prefix {
            name = Ident::new(format!("{prefix}{}", name.lexeme), name.location);
        }
        let type_params = self.type_params();
        let parameters = if self.expect(TokKind::LeftParenthesis) {
            let mut first = true;
            self.list(
                TokKind::RightParenthesis,
                |kind| kind == TokKind::SelfKeyword || kind.is_ident(),
                |p| {
                    // only the first parameter can be the receiver
                    let receiver = std::mem::replace(&mut first, false);
                    if receiver && p.at(TokKind::SelfKeyword) {
                        let location = p.next_location();
                        p.bump(TokKind::SelfKeyword);
                        return Binding::new(Ident::new("self", location));
                    }
                    let mut binding = p.binding();
                    binding.right = p.eat(TokKind::Equal).then(|| p.expr());
                    if let Some(default) = binding.right.as_ref().filter(|_| !defaults) {
                        p.error(PactSigDefaultParam {
                            src: p.src(),
                            at: default.location().into(),
                        });
                    }
                    binding
                },
            )
        } else {
            vec![]
        };
        let return_type = self.eat(TokKind::Arrow).then(|| self.annotation());

        (name, type_params, parameters, return_type)
    }

    /// A parameter's name, and maybe its type.
    fn binding(&mut self) -> Binding {
        let name = self.require_ident();
        let mut binding = Binding::new(self.hygiene_bind(name));
        binding.annotation = self.eat(TokKind::Colon).then(|| self.annotation());
        binding
    }

    fn generator_call(&self, name: &str, values: Vec<Expr>, start: usize) -> Expr {
        self.new_expr(
            Call::new(
                self.new_expr(self.ident_at(name, start), start),
                values
                    .into_iter()
                    .map(|value| Argument { name: None, value })
                    .collect(),
            ),
            start,
        )
    }

    fn function(&mut self) -> Function {
        self.function_with(None)
    }

    /// A `fn` item -- or, with `prefix`, the same shape after a contextual
    /// keyword already consumed, its name prefixed (`notation x` is
    /// `fn __lit_x`).
    fn function_with(&mut self, prefix: Option<&str>) -> Function {
        let generator = self.generator_type.take();
        let f = self.ordinary_function_with(prefix);
        self.generator_type = generator;
        f
    }

    fn ordinary_function_with(&mut self, prefix: Option<&str>) -> Function {
        let start = self.next_start();
        let (name, type_params, parameters, return_type) = self.function_sig(true, prefix);
        let wheres = self.where_clause();
        let body = if self.at(TokKind::LeftBrace) {
            self.block()
        } else {
            self.error_here(MissingFunctionBody {
                src: self.src(),
                at: self.location(start).into(),
            });
            self.poison_expr(self.cursor())
        };

        Function {
            name,
            role: FnRole::Fn,
            type_params,
            parameters,
            wheres,
            return_type,
            body,
        }
    }

    /// An optional `where` clause on a `fn` declaration: a comma-separated run of bool
    /// predicates over the parameters (`fn f(xs: [int]) where xs.len() > 2 { ... }`),
    /// checked statically where provable and again on entry at runtime. `where` is a
    /// word only in this position -- everywhere else it stays an ordinary identifier.
    fn where_clause(&mut self) -> Vec<Expr> {
        if !self.at(TokKind::Ident("where")) {
            return vec![];
        }
        self.advance();
        let mut wheres = vec![];
        loop {
            // a `{` where a predicate should be is the function's body arriving early
            if self.at(TokKind::LeftBrace) || !self.peek().starts_expr() {
                self.expected("where predicate");
                break;
            }
            // `where xs { }` -- like an `if` condition, a bare `{` is the body, not a
            // struct literal tail on the predicate
            wheres.push(self.struct_literals(false, Self::expr));
            if !self.eat(TokKind::Comma) {
                break;
            }
        }
        wheres
    }

    fn closure(&mut self) -> Expr {
        let generator = self.generator_type.take();
        let closure = self.ordinary_closure();
        self.generator_type = generator;
        closure
    }

    fn ordinary_closure(&mut self) -> Expr {
        let start = self.next_start();
        if self.eat(TokKind::DoublePipe) {
            let return_type = self.eat(TokKind::Arrow).then(|| self.annotation());
            let body = self.expr();
            return self.new_expr(
                Closure {
                    parameters: vec![],
                    body,
                    return_type,
                },
                start,
            );
        }
        self.bump(TokKind::Pipe);
        let parameters = self.list(TokKind::Pipe, TokKind::is_ident, Self::binding);
        let return_type = self.eat(TokKind::Arrow).then(|| self.annotation());
        let body = self.expr();
        self.new_expr(
            Closure {
                parameters,
                body,
                return_type,
            },
            start,
        )
    }

    fn break_stmt(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::Break);
        let expr = self.optional_expr();
        self.new_expr(Break::new(expr), start)
    }

    fn collect(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::Collect);
        let expr = self.expr();
        self.new_expr(Collect::new(expr), start)
    }

    fn continue_expr(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::Continue);
        self.new_expr(ExprKind::Continue(Continue), start)
    }

    fn return_expr(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::Return);
        let expr = self.optional_expr();
        self.new_expr(Return::new(expr), start)
    }

    fn raise_expr(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::Raise);
        let value = self.expr();
        self.new_expr(Raise::new(value), start)
    }

    fn loop_expr(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::Loop);
        let body = self.block_body();
        self.new_expr(Loop::new(body), start)
    }

    fn while_expr(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::While);
        let (binding, header) = self.condition();
        let body = self.block_body();
        self.new_expr(
            While {
                header,
                body,
                binding,
            },
            start,
        )
    }

    fn for_in(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::For);

        let binding = self.pattern();
        self.expect(TokKind::In);
        let iterator = self.iter_target();
        let body = self.block_body();
        self.new_expr(For::new(binding, iterator, body), start)
    }

    /// The `for x in <this>` — an expr plus an optional `..`/`..=` range
    /// tail, with juxtaposition disarmed (`for x in xs y` -- `y` is the
    /// body, not `xs(y)`) and struct literals off (`for x in P {` --
    /// `{` opens the body, not `P`'s fields). Shared by `for` loops and
    /// the clauses of a `[e for x in xs]` comprehension.
    fn iter_target(&mut self) -> Expr {
        self.with_juxta(Juxta::Off, |p| {
            p.struct_literals(false, |p| {
                let iterator = p.expr();
                let inclusive = p.at(TokKind::DoubleDotEqual);
                if !p.infix_binds() || (!inclusive && !p.at(TokKind::DoubleDot)) {
                    return iterator;
                }
                p.advance();
                let start = iterator.span().start();
                let end = p.expr();
                p.new_expr(Range::new(iterator, end, inclusive), start)
            })
        })
    }

    // r[impl mimas.array.comprehension]
    /// The `for x in xs …` tail of a `[head for x in xs]` comprehension
    /// — the head expr is already parsed. Clauses apply in order, each
    /// `for` nesting a loop and each `if` guarding what follows. The
    /// desugar rides `collect`: `[e for x in xs if p]` is
    /// `for x in xs { if p { collect e } }`; a second `for` collects
    /// the inner loop's list and flattens one level per extra clause —
    /// `for x in a { collect (for y in b { collect e }) }.flatten()`,
    /// which is Python's nesting order.
    ///
    /// `start` is the opening `[` — the loop the comprehension becomes
    /// spans the whole bracket, not just its head, so a `check` named
    /// by its expression's snippet reads `[e for x in xs].all(..)` and
    /// a hover on `for` lights the bracket. The caller re-spans the
    /// node once the `]` is eaten.
    fn list_comprehension(&mut self, head: Expr, start: usize) -> Expr {
        enum Clause {
            For(Pat, Expr),
            If(Option<Pat>, Expr),
        }
        let mut clauses = vec![];
        loop {
            if self.eat(TokKind::For) {
                let binding = self.pattern();
                self.expect(TokKind::In);
                clauses.push(Clause::For(binding, self.iter_target()));
            } else if self.eat(TokKind::If) {
                let (binding, condition) = self.condition();
                clauses.push(Clause::If(binding, condition));
            } else {
                break;
            }
        }
        if !self.at(TokKind::RightSquare) {
            // `[e for x in xs, y]` — clauses end the bracket; report
            // and let the caller's sequence recover the rest.
            self.expected("]");
        }
        let hloc = head.location();
        let mut body = Expr::new(Collect::new(head).into(), hloc);
        // `emits` — the folded body's leaf `collect`s into the
        // enclosing loop. A `for` clause consumes the emitter as its
        // own body, or — once a `for` has wrapped it — `collect`s the
        // produced list (threaded under `if`s by `collect_at_leaf`).
        let mut emits = true;
        let mut depth = 0usize;
        for clause in clauses.into_iter().rev() {
            body = match clause {
                Clause::For(binding, iterator) => {
                    if !emits {
                        body = self.collect_at_leaf(body);
                    }
                    emits = false;
                    depth += 1;
                    self.new_expr(For::new(binding, iterator, body), start)
                }
                Clause::If(binding, condition) => self.new_expr(
                    If {
                        condition,
                        main_body: body,
                        else_expr: None,
                        binding,
                    },
                    start,
                ),
            };
        }
        // each `for` past the first wrapped a produced list — flatten
        // it back out, one `.flatten()` per extra clause.
        for _ in 1..depth {
            let at = body.location();
            let dot = Expr::new(
                Access::Dot {
                    left: body,
                    right: Expr::new(Ident::new("flatten", at).into(), at),
                    kind: AccessKind::Direct,
                }
                .into(),
                at,
            );
            body = Expr::new(Call::new(dot, vec![]).into(), at);
        }
        body
    }

    /// `collect <body>` with the `collect` pushed to the leaf of any
    /// `if` chain — `for x if p` collects the produced list only when
    /// `p` holds, never an `Option`.
    fn collect_at_leaf(&mut self, body: Expr) -> Expr {
        let at = body.location();
        match body.into_kind() {
            ExprKind::If(mut if_) => {
                if_.main_body = self.collect_at_leaf(if_.main_body);
                Expr::new(if_.into(), at)
            }
            kind => Expr::new(Collect::new(Expr::new(kind, at)).into(), at),
        }
    }

    fn if_expr(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::If);
        let (binding, condition) = self.condition();
        let main_body = self.block_body();
        let else_expr = self.eat(TokKind::Else).then(|| self.block_body());
        self.new_expr(
            If {
                condition,
                main_body,
                else_expr,
                binding,
            },
            start,
        )
    }

    /// What an `if` or a `while` tests, along with the pattern when it's a `let` binding.
    fn condition(&mut self) -> (Option<Pat>, Expr) {
        let binding = self.eat(TokKind::Let).then(|| {
            let binding = self.pattern();
            self.expect(TokKind::Equal);
            binding
        });
        // what follows the condition is the body, never an argument —
        // `if c "heads" else "tails"` can't read `"heads"` as `c("heads")`
        let condition = self.with_juxta(Juxta::Off, |p| p.struct_literals(false, Self::expr));

        // a trailing `= 5` means the user reached for another language's syntax --
        // catch it before the body parse swallows the token. the rest of the condition gets
        // taken too (so the body still parses) and the whole thing is poison
        let (msg, label) = match self.peek() {
            TokKind::Equal if binding.is_none() => {
                ("invalid assignment in a condition", "did you mean `==`?")
            }
            _ => return (binding, condition),
        };
        self.reject(Misdirection {
            src: self.src(),
            at: self.next_location().into(),
            msg: msg.into(),
            label: label.into(),
        });
        self.struct_literals(false, Self::expr);
        (binding, self.poison_expr(condition.span().start()))
    }

    /// After a `match` head: a `{ name = value` right where the arms should start is a struct
    /// literal that needs parentheses (`match (P { a = 1 }) { .. }`) -- a bare `P { .. }` there
    /// would be ambiguous with the arms' own `{`, as in Rust. (Only for `match`: an `if`/`while`
    /// body may legitimately begin `name = value`.)
    fn head_struct_literal_hint(&mut self, from: Option<usize>) {
        if self.at(TokKind::LeftBrace)
            && matches!(self.nth(1), TokKind::Ident(_))
            && self.nth(2) == TokKind::Equal
        {
            self.error(Misdirection {
                src: self.src(),
                // from the keyword when we have it, so this reads before the
                // statement-level errors the stray braces set off
                at: from
                    .map_or_else(|| self.next_location(), |f| self.location(f))
                    .into(),
                msg: "struct literals need parentheses here".into(),
                label: "wrap the struct literal in `( )`; a `{` here starts the body".into(),
            });
        }
    }

    fn match_expr(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::Match);
        let expr = self.struct_literals(false, Self::expr);
        self.head_struct_literal_hint(Some(start));
        let mut panic_terminator = false;
        let members = if self.expect(TokKind::LeftBrace) {
            self.sequence(
                TokKind::RightBrace,
                |kind| kind == TokKind::Bang || kind.starts_pattern(),
                |p| {
                    if p.eat(TokKind::Bang) {
                        panic_terminator = true;
                        p.eat(TokKind::Comma);
                        // nothing comes after the `!`
                        if !p.at(TokKind::RightBrace) {
                            p.expect(TokKind::RightBrace);
                        }
                        return None;
                    }
                    let pattern = p.pattern();
                    let guard = p.eat(TokKind::If).then(|| p.expr());
                    p.expect(TokKind::FatArrow);
                    let body_is_block = p.at(TokKind::LeftBrace);
                    let body = p.expr();
                    // block-bodied arms can omit the trailing comma
                    if !p.eat(TokKind::Comma) && !body_is_block && !p.at(TokKind::RightBrace) {
                        p.expect(TokKind::Comma);
                    }
                    Some(MatchCase::new(pattern, guard, body))
                },
            )
        } else {
            vec![]
        };
        let members = members.into_iter().flatten().collect();
        self.new_expr(Match::new(expr, members, panic_terminator), start)
    }

    fn null_coalecence(&mut self) -> Expr {
        let start = self.next_start();
        let mut expr = self.binary(0);
        loop {
            if !self.infix_binds() {
                return expr;
            }
            expr = if self.eat(TokKind::DoubleHook) {
                // `a ?? b ?? c` stays right-nested: the second `??` is
                // inside the fallback's own expr
                let value = self.expr();
                self.new_expr(Coalescence::new(expr, value), start)
            } else if self.eat(TokKind::Absolve) {
                let handler = self.expr();
                self.new_expr(Absolve::new(expr, handler), start)
            } else {
                return expr;
            };
            // A block-ended trailer hands the chain back to `binary_ops`, so an
            // infix after the `}` binds the trailer's value, not the handler:
            // `f() absolve |e| { d } == v` is `(f() absolve |e| { d }) == v`.
            expr = self.binary_ops(expr, 0, start);
        }
    }

    /// Precedence climbing over the binary operators. `min_power` is the loosest operator this
    /// call may take; anything looser belongs to the caller.
    fn binary(&mut self, min_power: u8) -> Expr {
        let start = self.next_start();
        let left = self.unary();
        self.binary_ops(left, min_power, start)
    }

    /// The infix loop of `binary`, split out so a `??`/`absolve` trailer can hand
    /// its result back to the chain once its handler ends.
    fn binary_ops(&mut self, mut left: Expr, min_power: u8, start: usize) -> Expr {
        // only looser operators can follow an `in`, which doesn't chain
        let mut max_power = u8::MAX;
        loop {
            // A spaced `%` reaching here can only be old-style modulo — postfix `x%` hugs
            // its operand and already folded in `chain_accesses`, and a `±` tolerance's
            // `%` is claimed by `plus_minus`. Teach `mod` and keep parsing as if written.
            if self.at(TokKind::Percent) && !self.pct_marker && self.infix_binds() {
                self.error_here(Misdirection {
                    src: self.src(),
                    at: self.next_location().into(),
                    msg: "modulo is `a mod b` now — `%` spells percent, as in `50%`".into(),
                    label: "write `mod` here".into(),
                });
                self.advance();
                let right = self.binary(8);
                left = self.op_expr(BinaryOp::Eval(EvaluationOp::Modulo), left, right, start);
                continue;
            }
            // A `^` reaching here is the retired xor — contiguous `x^2`
            // already folded into the name at lex time, so this is always
            // spaced (or after a number, where the intent was `pow`). Teach
            // `⊕`/`xor`/`.pow` and keep parsing as if xor were written.
            if self.at(TokKind::Caret) && self.infix_binds() {
                let caret = self.next_location();
                let pow = matches!(
                    left.kind(),
                    ExprKind::Literal(Literal::Int(_) | Literal::Float(_))
                );
                self.advance();
                let right = self.binary(5);
                // `2^10` is spelled like math, not xor — both sides are
                // const, so the value's known at desugar: fold it. `x ^ 2`
                // with a name stays the misdirection (`x^2` superscripts).
                if let Some(folded) = caret_pow(&left, &right) {
                    left = self.new_expr(folded, start);
                    continue;
                }
                // `error`, not `error_here`: the right side already parsed,
                // so at EOF the Misdirection itself would never emit.
                self.error(Misdirection {
                    src: self.src(),
                    at: caret.into(),
                    msg: if pow {
                        "`^` isn't an operator — exponents are `x.pow(n)` and xor is `a ⊕ b`; `x^2` superscripts a name".into()
                    } else {
                        "xor is `a ⊕ b` (or `a xor b`) — `^` superscripts a name, as in `x^2`".into()
                    },
                    label: if pow {
                        "write `.pow(…)`".into()
                    } else {
                        "write `⊕` here".into()
                    },
                });
                left = self.op_expr(BinaryOp::Eval(EvaluationOp::Xor), left, right, start);
                continue;
            }
            // in a comma-optional sequence a `-` glued to a number opens
            // the next element: `[0 -1]` is `[0, -1]`, while `a - 1`
            // and `a -x` keep the infix read
            if self.juxta == Juxta::Comma && self.glued_minus() {
                break;
            }
            let Some((op, power)) = BinaryOp::of(self.peek()) else {
                break;
            };
            // r[impl mimas.stmt.double-pipe]
            // After a left operand, `||` continues the expression across a newline.
            // Where a new expression is expected, prefix `||` still opens a closure.
            let continues = self.infix_binds() || self.at(TokKind::DoublePipe);
            if !(min_power..max_power).contains(&power) || !continues {
                break;
            }
            self.advance();
            let right = if matches!(op, BinaryOp::PlusMinus) {
                // `±`'s tolerance parses like a normal right side except a `%`
                // there is the literal's percent marker, not `mod` -- `x ± 10%`
                // is "give or take 10%", and `x ± (a mod b)` keeps infix `mod`.
                self.plus_minus_rhs()
            } else {
                self.binary(power + 1)
            };
            if matches!(op, BinaryOp::In(_)) {
                max_power = power;
            }
            left = self.op_expr(op, left, right, start);
        }
        left
    }

    /// Builds the expr `left op right`, `op` as [`BinaryOp::of`] classified it.
    fn op_expr(&mut self, op: BinaryOp, left: Expr, right: Expr, start: usize) -> Expr {
        match op {
            BinaryOp::Logical(op) => self.new_expr(Logical::new(left, op, right), start),
            BinaryOp::Equality(op) => self.new_expr(Equality::new(left, op, right), start),
            BinaryOp::In(condition) => self.new_expr(In::new(left, right, condition), start),
            BinaryOp::Eval(op) => self.new_expr(Evaluation::new(left, op, right), start),
            BinaryOp::PlusMinus => self.plus_minus(left, right, start),
        }
    }

    /// The `d` in `x ± d` binds at the multiplicative level, with one exception: a `%` at its
    /// top level is the literal's percent marker (`x ± 10%`), not `mod` -- write
    /// `x ± (a mod b)` for a `mod` inside the tolerance.
    fn plus_minus_rhs(&mut self) -> Expr {
        let start = self.next_start();
        let saved_pct = std::mem::replace(&mut self.pct_marker, true);
        let mut left = self.unary();
        while let Some((op, power)) = BinaryOp::of(self.peek())
            && power >= 7
            && !matches!(op, BinaryOp::Eval(EvaluationOp::Modulo))
            && self.infix_binds()
        {
            self.advance();
            let right = self.binary(power + 1);
            left = self.op_expr(op, left, right, start);
        }
        self.pct_marker = saved_pct;
        left
    }

    /// `x ± d` is `Interval::within(x * 1.0, d * 1.0)` — "give or take `d`" in `x`'s unit.
    /// `x ± d%` is relative: `Interval::pm(x * 1.0, d * 0.01)`, the `%` marking percent-of-x.
    /// r[impl mimas.interval.literal]
    /// The `* 1.0` widens an int operand to the float `Interval` works in; floats and
    /// quantities pass through unchanged.
    fn plus_minus(&mut self, left: Expr, right: Expr, start: usize) -> Expr {
        let times = |e: Expr, f: f64, p: &mut Self| {
            p.new_expr(
                Evaluation::new(
                    e,
                    EvaluationOp::Multiply,
                    p.new_expr(Literal::Float(f), start),
                ),
                start,
            )
        };
        let (ctor, tolerance) = if self.eat(TokKind::Percent) {
            ("pm", times(right, 0.01, self))
        } else {
            ("within", times(right, 1.0, self))
        };
        let mid = times(left, 1.0, self);
        let interval = self.new_expr(Ident::synthetic("Interval"), start);
        let path = self.new_expr(
            Access::DoubleColon {
                left: interval,
                right: Ident::synthetic(ctor),
            },
            start,
        );
        self.new_expr(
            Call::new(
                path,
                vec![
                    Argument {
                        name: None,
                        value: mid,
                    },
                    Argument {
                        name: None,
                        value: tolerance,
                    },
                ],
            ),
            start,
        )
    }

    fn unary(&mut self) -> Expr {
        let start = self.next_start();
        if self.eat(TokKind::Ampersand) {
            let place = self.unary();
            return self.generator_call("__borrow_place", vec![place], start);
        }

        // `∑ xs` -- a reduction is the big form of its operator, and desugars to
        // the method it names on the operand: `∑` `.sum()`, `∏` `.product()`,
        // `⨅` `.inf()` (meet: all/min), `⨆` `.sup()` (join: any/max). It binds
        // like a unary minus, so `∑ xs.map(f)` reduces the mapped list and
        // `⨆⨅ (m == p)` nests. A direction after the glyph reduces a tensor
        // along it: `∑→ m` is `m.sum_dim(1)` (each row to one number), `∑↓ m`
        // is `m.sum_dim(0)`.
        if let Some(method) = reduce_method(self.peek()) {
            self.advance();
            let along = match self.peek() {
                TokKind::RightArrow => Some(1),
                TokKind::DownArrow => Some(0),
                _ => None,
            };
            if along.is_some() {
                self.advance();
            }
            // `⨆ for ln in LINES { … collect … }`: a keyword expression is the whole
            // operand; anything else binds like a unary minus does
            let operand = if matches!(
                self.peek(),
                TokKind::For | TokKind::If | TokKind::Match | TokKind::Loop | TokKind::While | TokKind::LeftBrace
            ) {
                self.expr()
            } else {
                self.nested(Self::unary)
                    .unwrap_or_else(|| self.poison_expr(self.cursor()))
            };
            let at = self.location(start);
            let (name, args) = match along {
                Some(dim) => (
                    format!("{method}_dim"),
                    vec![Argument {
                        name: None,
                        value: self.new_expr(ExprKind::Literal(Literal::Int(dim)), start),
                    }],
                ),
                None => (method.to_string(), vec![]),
            };
            let callee = self.new_expr(
                Access::Dot {
                    left: operand,
                    right: self.new_expr(Ident::new(&name, at), start),
                    kind: AccessKind::Direct,
                },
                start,
            );
            return self.new_expr(Call::new(callee, args), start);
        }
        // `↑ x` — the old prefix spelling. Directions follow their
        // operand now (`x↑`), so a direction glyph here is a mistake to
        // name, not a form to parse.
        if matches!(
            self.peek(),
            TokKind::RightArrow
                | TokKind::LeftArrow
                | TokKind::UpArrow
                | TokKind::DownArrow
                | TokKind::CircledDot
                | TokKind::CircledTimes
                | TokKind::Rewind
                | TokKind::FastForward
        ) {
            self.error(Misdirection {
                src: self.src(),
                at: self.next_location().into(),
                msg: "a direction follows its operand".into(),
                label: "write `x↑`, not `↑x`".into(),
            });
            self.advance();
            return self
                .nested(Self::unary)
                .unwrap_or_else(|| self.poison_expr(self.cursor()));
        }
        let operator = if self.at(TokKind::NotWord) {
            UnaryOp::Not
        } else if let Ok(op) = self.peek().try_into() {
            op
        } else {
            return self.fstring();
        };
        self.advance();
        // `----...` recurses here without passing back through `expr`
        let right = self
            .nested(Self::unary)
            .unwrap_or_else(|| self.poison_expr(self.cursor()));
        self.new_expr(Unary::new(operator, right), start)
    }

    fn fstring(&mut self) -> Expr {
        /// Parses the expression inside a `{...}`. It gets a parser of its own, on the
        /// interpolation's slice of the real source.
        fn interpolation<'s>(
            outer: &mut Parser<'s>,
            body: &'s str,
            offset: usize,
            template: Option<(Location, u32)>,
        ) -> Expr {
            let lexer = Lexer::new(body, outer.file_id, outer.file_name.clone())
                .with_shadowed(crate::lex::shadow_at(&outer.shadow, offset));
            let relocate = template.map(|(at, _)| at);
            let mut parser = Parser::with_src_at(lexer.with_offset(offset), outer.src(), relocate);
            // nested f-strings would otherwise restart the depth guard
            parser.depth = outer.depth;
            // r[impl mimas.macro.hygiene]
            // a macro template's f-string names the template's own locals
            if let Some((_, mark)) = template {
                parser.adopt_template(&outer.hygiene, mark);
            }
            let expr = parser.expr();
            if !parser.at(TokKind::Eof) {
                parser.error(parser.unexpected_token());
            }
            outer.errors.extend(parser.errors);
            // units written inside the interpolation belong to the whole file's ast
            outer.quantities.extend(parser.quantities);
            outer.quantity_units.extend(parser.quantity_units);
            outer.directions.extend(parser.directions);
            expr
        }
        // `{` and `}` join the quote-chars so chompy's unescape resolves `\{` -> `{` and
        // `\}` -> `}` (the brace-escape form, sibling to `{{`/`}}`).
        fn unescape(literal: &str) -> FStringPart {
            FStringPart::Literal(chompy::utils::unescape(literal, &['\\'], &['"', '{', '}']))
        }

        let start = self.next_start();
        if let TokKind::Tagged(tag, body) = self.peek() {
            // r[impl mimas.str.tagged]
            // `tag"…"` lowers to `__lit_tag("…")`: a prelude native the host
            // registers per tag (`pat`, `keys`, …), so the literal is checked at
            // compile time by that native's validator and typed by its return.
            // The string argument keeps the quoted span, so a validator's fault
            // lands inside the literal; the callee spans the whole literal.
            let (tag, body) = (tag.to_string(), body.to_string());
            self.advance();
            let end = self.cursor();
            let quote = start + tag.len();
            let callee = self.new_expr(
                Ident::new(
                    format!("__lit_{tag}"),
                    Location::new(self.file_id, Span::new(start, quote)),
                ),
                start,
            );
            let value = Expr::new(
                ExprKind::Literal(Literal::String(chompy::utils::unescape(
                    &body,
                    &['\\'],
                    &['"'],
                ))),
                Location::new(self.file_id, Span::new(quote, end)),
            );
            let expr = self.new_expr(
                Call::new(callee, vec![Argument { name: None, value }]),
                start,
            );
            return self.chain_accesses(expr);
        }
        let TokKind::FString(content) = self.peek() else {
            return self.literal();
        };
        // an f-string a macro template wrote has no text in this file
        let template =
            (self.meta_at(0).mark != 0).then(|| (self.next_location(), self.meta_at(0).mark));
        self.advance();
        // the token covers `f"..."`, so the content begins two bytes in
        let content_start = start + 2;

        let mut parts = vec![];
        let mut literal = String::new();
        let mut at = 0;
        while let Some(c) = content[at..].chars().next() {
            at += c.len_utf8();
            let rest = &content[at..];
            match c {
                '\\' => {
                    literal.push(c);
                    if let Some(escaped) = rest.chars().next() {
                        literal.push(escaped);
                        at += escaped.len_utf8();
                    }
                }
                // `{{` and `}}` are a literal `{` and `}`
                '{' | '}' if rest.starts_with(c) => {
                    literal.push(c);
                    at += 1;
                }
                '{' => {
                    if !literal.is_empty() {
                        parts.push(unescape(&literal));
                        literal.clear();
                    }
                    let Some(end) = interp_end(rest) else {
                        let open = content_start + at - 1;
                        let at = template.map_or_else(
                            || Location::new(self.file_id, Span::new(open, open + 1)),
                            |(at, _)| at,
                        );
                        self.error(UnterminatedFStringExpr {
                            src: self.src(),
                            at: at.into(),
                        });
                        break;
                    };
                    let expr = interpolation(self, &rest[..end], content_start + at, template);
                    parts.push(FStringPart::Expr(expr));
                    at += end + 1;
                }
                _ => literal.push(c),
            }
        }
        if !literal.is_empty() {
            parts.push(unescape(&literal));
        }

        let expr = self.new_expr(FString::new(parts), start);
        self.chain_accesses(expr)
    }

    fn literal(&mut self) -> Expr {
        let start = self.next_start();
        if let Ok(literal) = Literal::try_from(self.peek()) {
            let dim = match self.peek() {
                TokKind::Quantity(_, unit) => {
                    shared::units::parse(unit).map(|(dim, _)| (dim, unit))
                }
                _ => None,
            };
            self.advance();
            let expr = self.new_expr(literal, start);
            if let Some((dim, unit)) = dim {
                self.quantities.insert(expr.id(), dim);
                self.quantity_units.insert(expr.id(), unit.to_string());
            }

            // todo: this might allow "hello"() or true[]" etc. only dot accesses are okay on lits"
            self.chain_accesses(expr)
        } else if self.eat(TokKind::LeftSquare) {
            // `[e for x in xs]` — a comprehension desugars to the
            // `for`/`collect` loop; anything else is a plain literal.
            let mut comp = false;
            let mut first = true;
            let elements = self.with_juxta(Juxta::Comma, |p| {
                p.list(TokKind::RightSquare, TokKind::starts_expr, |p| {
                    let mut element = p.struct_literals(true, Self::expr);
                    if first && p.at(TokKind::For) {
                        comp = true;
                        element = p.list_comprehension(element, start);
                    }
                    first = false;
                    element
                })
            });
            let expr = if comp {
                // the For node `list_comprehension` built — later
                // elements, if any, already drew their own errors.
                // Re-spanned now that the `]` is behind the cursor.
                let comp = elements.into_iter().next().unwrap();
                self.new_expr(comp.into_kind(), start)
            } else {
                self.new_expr(Literal::Array(elements), start)
            };
            self.chain_accesses(expr)
        } else if self.struct_literals
            && self.peek() == TokKind::Ident("table")
            && self.nth(1) == TokKind::LeftBrace
        {
            let expr = self.table_literal();
            self.chain_accesses(expr)
        } else if self.struct_literals
            && self.peek() == TokKind::Ident("query")
            && self.nth(1) == TokKind::LeftBrace
        {
            let expr = self.query_literal();
            self.chain_accesses(expr)
        } else if self.struct_literals
            && self.peek() == TokKind::Ident("datalog")
            && self.nth(1) == TokKind::LeftBrace
        {
            let expr = self.datalog_literal();
            self.chain_accesses(expr)
        } else if self.struct_literals
            && self.peek() == TokKind::Ident("data")
            && matches!(self.nth(1), TokKind::String(_))
            && !self.meta_at(1).newline
        {
            // `data "name-or-url"` — an external data source (see data_source)
            let expr = self.data_source();
            self.chain_accesses(expr)
        } else if self.peek() == TokKind::Ident("with") && self.with_intro() {
            // `with rng = <spec> { … }` — a scoped effect state for the
            // body's extent; `with` stays a name everywhere else
            // r[impl mimas.fx.with]
            let expr = self.with_expr(start);
            self.chain_accesses(expr)
        } else if self.at(TokKind::Dollar) {
            // `$expr` — a `query { }` splice: `expr` is a value from the
            // surrounding scope, not a column. Emitted as `__q_splice(expr)`;
            // `q_lower` unwraps the marker untouched. Outside a query the call
            // reaches the `__q_splice` native, which explains the escape only
            // applies inside a query.
            self.advance();
            let inner = self
                .nested(Self::unary)
                .unwrap_or_else(|| self.poison_expr(start));
            let callee = self.new_expr(Ident::new("__q_splice", self.location(start)), start);
            self.new_expr(
                Call::new(
                    callee,
                    vec![Argument {
                        name: None,
                        value: inner,
                    }],
                ),
                start,
            )
        } else {
            let name = self.supreme();
            if !self.struct_literals || !self.eat(TokKind::LeftBrace) {
                return name;
            }
            // r[impl mimas.struct.update]
            // fields, then optionally `..base` last: every field not listed
            // is read from `base`, a value of the same struct type
            let mut fields = Vec::new();
            let mut base = None;
            loop {
                if self.at(TokKind::RightBrace) {
                    break;
                }
                if self.eat(TokKind::DoubleDot) {
                    base = Some(self.expr());
                    self.eat(TokKind::Comma);
                    break;
                }
                if !self.peek().is_ident() {
                    self.expected("a field name or `..base`");
                    break;
                }
                let field = self.require_ident();
                // r[impl mimas.struct.shorthand]
                // shorthand: `Foo { x }` is `Foo { x = x }`
                let value = if self.at(TokKind::Comma) || self.at(TokKind::RightBrace) {
                    let local = self.hygiene_use(field.clone());
                    self.new_expr(local, field.location.span().start())
                } else {
                    // r[impl mimas.record.separator] `x: 1` or `x = 1`
                    if !self.eat(TokKind::Colon) {
                        self.expect(TokKind::Equal);
                    }
                    self.expr()
                };
                fields.push((FieldKey::Ident(field), value));
                if !self.eat(TokKind::Comma) {
                    break;
                }
            }
            self.expect(TokKind::RightBrace);
            let expr = self.new_expr(Literal::Struct(StructLiteral { name, fields, base }), start);
            self.chain_accesses(expr)
        }
    }

    /// `data "name-or-url"` -- an external data source: a page's `data`
    /// card by name, or a URL the host fetches. Lowers to `__data("…")`;
    /// the host binds the name at runtime and hands the checker its column
    /// schema, so `pull`/`query` code over the result checks by name (see
    /// docs/type-providers.md). `data("name")`, the host's untyped
    /// accessor, is unaffected -- the form needs a bare string.
    fn data_source(&mut self) -> Expr {
        let start = self.next_start();
        self.advance(); // `data`
        let str_start = self.next_start();
        let TokKind::String(raw) = self.peek() else {
            self.expected("a data source name or url (a quoted string)");
            return self.poison_expr(start);
        };
        let name = chompy::utils::unescape(raw, &['\\'], &['"']);
        self.advance();
        let value = self.new_expr(Literal::String(name), str_start);
        let callee = self.new_expr(Ident::new("__data", self.location(start)), start);
        self.new_expr(
            Call::new(
                callee,
                vec![Argument {
                    name: None,
                    value,
                }],
            ),
            start,
        )
    }

    /// `table { name age \n "Denmark" 25 \n … }` -- a table literal. The first line names the
    /// columns (identifiers or strings); each later line is one row, its cells separated by
    /// whitespace/tabs (commas are optional). A cell is one unary expression, so wrap anything
    /// bigger in parentheses. It lowers to `__table_col(… __table_col(__table_new(), "a", [..]) …)`
    /// -- one call per column, so columns of different types need no shared element type.
    fn table_literal(&mut self) -> Expr {
        // Cells are whitespace-separated — `1 2` is two cells, not
        // `1(2)` — so juxtaposition stays a separator inside `table {}`.
        self.with_juxta(Juxta::Off, Self::table_cells)
    }

    fn table_cells(&mut self) -> Expr {
        let start = self.next_start();
        self.advance(); // `table`
        self.bump(TokKind::LeftBrace);
        let here = self.location(start);

        let mut names: Vec<String> = vec![];
        let mut first = true;
        while !self.at(TokKind::RightBrace)
            && !self.at(TokKind::Eof)
            && (first || !self.at_line_start())
        {
            first = false;
            let cell = self.unary();
            match cell.kind() {
                ExprKind::Ident(id) => names.push(id.lexeme.clone()),
                ExprKind::Literal(Literal::String(s)) => names.push(s.clone()),
                _ => self.error(Misdirection {
                    src: self.src(),
                    at: cell.location().into(),
                    msg: "a table's first line names its columns".into(),
                    label: "use a plain name or a \"string\" here".into(),
                }),
            }
            self.eat(TokKind::Comma);
        }

        let mut rows: Vec<Vec<Expr>> = vec![];
        while !self.at(TokKind::RightBrace) && !self.at(TokKind::Eof) {
            let mut row = vec![];
            let mut first = true;
            while !self.at(TokKind::RightBrace)
                && !self.at(TokKind::Eof)
                && (first || !self.at_line_start())
            {
                first = false;
                if !self.peek().starts_expr() {
                    self.expected("a table cell");
                    self.advance();
                    continue;
                }
                row.push(self.unary());
                self.eat(TokKind::Comma);
            }
            if !row.is_empty() {
                if row.len() != names.len() {
                    let at = row[0].location();
                    self.error(Misdirection {
                        src: self.src(),
                        at: at.into(),
                        msg: format!(
                            "this row has {} cells but the table names {} columns",
                            row.len(),
                            names.len()
                        ),
                        label: "every row needs one cell per column".into(),
                    });
                }
                rows.push(row);
            }
        }
        self.expect(TokKind::RightBrace);

        // build the column-by-column call chain
        let ident = |p: &Self, name: &str| p.new_expr(Ident::new(name.to_string(), here), start);
        let arg = |value: Expr| Argument { name: None, value };
        let mut table = {
            let callee = ident(self, "__table_new");
            self.new_expr(Call::new(callee, vec![]), start)
        };
        for (j, name) in names.iter().enumerate() {
            let cells: Vec<Expr> = rows.iter().filter_map(|r| r.get(j).cloned()).collect();
            let column = self.new_expr(Literal::Array(cells), start);
            let title = self.new_expr(Literal::String(name.clone()), start);
            let callee = ident(self, "__table_col");
            table = self.new_expr(
                Call::new(callee, vec![arg(table), arg(title), arg(column)]),
                start,
            );
        }
        table
    }

    // ---- `query { … }` ----------------------------------------------------------------------
    // A PRQL-flavored block over a table:
    //
    //     query {
    //         events()
    //         filter delivery == "email" && numtix > 2
    //         derive cost = numtix * 25
    //         group delivery { aggregate { tickets = sum(numtix), orders = count() } }
    //         sort -tickets
    //         take 5
    //     }
    //
    // The first line is an ordinary expression (optionally `from <expr>`) that yields the
    // table; each later line is a verb. Inside a verb's expressions a bare name is a *column*,
    // `&&`/`||` combine column tests, `if c { a } else { b }` is a conditional column, and a few
    // functions (`sum`, `mean`, `count`, `to_lower`, `contains`, …) apply to columns. To reach a
    // value from the surrounding scope -- a `let` or a parameter -- write it `$name`d: the `$`
    // splice evaluates its expression outside the column namespace (see `q_lower`).
    // It lowers to one `__q_*` native call per verb (see `library::std_lib::dataframe`).
    fn query_literal(&mut self) -> Expr {
        let start = self.next_start();
        self.advance(); // `query`
        self.bump(TokKind::LeftBrace);
        // every verb's operands are whitespace-separated -- `join fees
        // delivery` must not read `fees(delivery)`; the block runs with
        // juxtaposition disarmed, like `table` cells
        self.with_juxta(Juxta::Off, |p| p.query_body(start))
    }

    fn query_body(&mut self, start: usize) -> Expr {
        if self.peek() == TokKind::Ident("from") {
            self.advance();
        }
        let mut table = self.struct_literals(true, Self::expr);

        while !self.at(TokKind::RightBrace) && !self.at(TokKind::Eof) {
            let verb_start = self.next_start();
            let TokKind::Ident(verb) = self.peek() else {
                self.expected("a query verb (filter, derive, select, sort, take, group, rename, distinct, join)");
                self.advance();
                continue;
            };
            self.advance();
            table = match verb {
                "filter" => {
                    let cond = self.q_column_expr();
                    self.q_call("__q_filter", vec![table, cond], verb_start)
                }
                "derive" => {
                    let cols = self.q_named_columns();
                    let list = self.new_expr(Literal::Array(cols), verb_start);
                    self.q_call("__q_mutate", vec![table, list], verb_start)
                }
                "select" => {
                    let names = self.q_names(false);
                    let list =
                        self.q_str_list(names.iter().map(|(n, _)| n.clone()).collect(), verb_start);
                    self.q_call("__q_select", vec![table, list], verb_start)
                }
                "sort" => {
                    let names = self.q_names(true);
                    let cols =
                        self.q_str_list(names.iter().map(|(n, _)| n.clone()).collect(), verb_start);
                    let dirs = names
                        .iter()
                        .map(|(_, desc)| {
                            self.new_expr(
                                if *desc { Literal::True } else { Literal::False },
                                verb_start,
                            )
                        })
                        .collect();
                    let dirs = self.new_expr(Literal::Array(dirs), verb_start);
                    self.q_call("__q_sort", vec![table, cols, dirs], verb_start)
                }
                "take" => {
                    let n = self.expr();
                    self.q_call("__q_take", vec![table, n], verb_start)
                }
                "distinct" => {
                    let names = self.q_names(false);
                    let list =
                        self.q_str_list(names.into_iter().map(|(n, _)| n).collect(), verb_start);
                    self.q_call("__q_distinct", vec![table, list], verb_start)
                }
                "rename" => {
                    let names = self.q_names(false);
                    if names.len() != 2 {
                        self.error(Misdirection {
                            src: self.src(),
                            at: self.location(verb_start).into(),
                            msg: "rename takes an old name and a new name".into(),
                            label: "`rename old new`".into(),
                        });
                        table
                    } else {
                        let a = self.new_expr(Literal::String(names[0].0.clone()), verb_start);
                        let b = self.new_expr(Literal::String(names[1].0.clone()), verb_start);
                        self.q_call("__q_rename", vec![table, a, b], verb_start)
                    }
                }
                "group" => {
                    let names = self.q_names_until_brace();
                    let list = self.q_str_list(names, verb_start);
                    let grouped = self.q_call("__q_group", vec![table, list], verb_start);
                    self.expect(TokKind::LeftBrace);
                    if self.peek() == TokKind::Ident("aggregate") {
                        self.advance();
                    } else {
                        self.expected("`aggregate { … }` inside `group`");
                    }
                    let aggs = self.q_named_columns();
                    self.expect(TokKind::RightBrace);
                    let list = self.new_expr(Literal::Array(aggs), verb_start);
                    self.q_call("__q_agg", vec![grouped, list], verb_start)
                }
                "join" => {
                    // `join other key [key…] ["kind"]`
                    let other = self.unary();
                    let mut keys = vec![];
                    let mut kind = "inner".to_string();
                    while !self.at_line_start()
                        && !self.at(TokKind::RightBrace)
                        && !self.at(TokKind::Eof)
                    {
                        match self.peek() {
                            TokKind::Ident(n) => {
                                keys.push(n.to_string());
                                self.advance();
                            }
                            TokKind::String(s) => {
                                kind = s.trim_matches('"').to_string();
                                self.advance();
                            }
                            _ => {
                                self.expected("a join key");
                                self.advance();
                            }
                        }
                    }
                    let keys = self.q_str_list(keys, verb_start);
                    let kind = self.new_expr(Literal::String(kind), verb_start);
                    self.q_call("__q_join", vec![table, other, keys, kind], verb_start)
                }
                _ => {
                    self.error(Misdirection {
                        src: self.src(),
                        at: self.location(verb_start).into(),
                        msg: format!("`{verb}` isn't a query verb"),
                        label: "try filter, derive, select, sort, take, group, rename, distinct or join".into(),
                    });
                    // skip the rest of this line
                    while !self.at_line_start()
                        && !self.at(TokKind::RightBrace)
                        && !self.at(TokKind::Eof)
                    {
                        self.advance();
                    }
                    table
                }
            };
        }
        self.expect(TokKind::RightBrace);
        // keep the whole block's span on the result for error labels
        let _ = start;
        table
    }

    fn q_call(&self, name: &str, args: Vec<Expr>, start: usize) -> Expr {
        let callee = self.new_expr(Ident::new(name.to_string(), self.location(start)), start);
        let args = args
            .into_iter()
            .map(|value| Argument { name: None, value })
            .collect();
        self.new_expr(Call::new(callee, args), start)
    }

    fn q_str_list(&self, names: Vec<String>, start: usize) -> Expr {
        let items = names
            .into_iter()
            .map(|n| self.new_expr(Literal::String(n), start))
            .collect();
        self.new_expr(Literal::Array(items), start)
    }

    /// A verb's column expression: parsed as normal Mimas, then its bare names become columns.
    fn q_column_expr(&mut self) -> Expr {
        let e = self.struct_literals(true, Self::expr);
        self.q_lower(&e)
    }

    /// Names (identifiers or strings) up to the end of the line; with `desc`, `-name` is
    /// descending. Returns `(name, descending)`.
    fn q_names(&mut self, desc: bool) -> Vec<(String, bool)> {
        let mut out = vec![];
        while !self.at_line_start() && !self.at(TokKind::RightBrace) && !self.at(TokKind::Eof) {
            let mut down = false;
            if desc && self.eat(TokKind::Minus) {
                down = true;
            }
            match self.peek() {
                TokKind::Ident(n) => {
                    out.push((n.to_string(), down));
                    self.advance();
                }
                TokKind::String(s) => {
                    out.push((s.trim_matches('"').to_string(), down));
                    self.advance();
                }
                _ => {
                    self.expected("a column name");
                    self.advance();
                }
            }
            self.eat(TokKind::Comma);
        }
        out
    }

    /// Names up to a `{` (the `group` verb's keys).
    fn q_names_until_brace(&mut self) -> Vec<String> {
        let mut out = vec![];
        while !self.at(TokKind::LeftBrace) && !self.at(TokKind::Eof) && !self.at_line_start() {
            match self.peek() {
                TokKind::Ident(n) => {
                    out.push(n.to_string());
                    self.advance();
                }
                TokKind::String(s) => {
                    out.push(s.trim_matches('"').to_string());
                    self.advance();
                }
                _ => {
                    self.expected("a column name");
                    self.advance();
                }
            }
            self.eat(TokKind::Comma);
        }
        out
    }

    /// `name = expr` on one line, or `{ name = expr, name = expr }`: each becomes a
    /// `__q_named(expr, "name")` call.
    fn q_named_columns(&mut self) -> Vec<Expr> {
        let braced = self.eat(TokKind::LeftBrace);
        let mut out = vec![];
        loop {
            if braced && (self.at(TokKind::RightBrace) || self.at(TokKind::Eof)) {
                break;
            }
            let start = self.next_start();
            let TokKind::Ident(name) = self.peek() else {
                self.expected("a column name");
                break;
            };
            self.advance();
            self.expect(TokKind::Equal);
            let value = self.q_column_expr();
            let title = self.new_expr(Literal::String(name.to_string()), start);
            out.push(self.q_call("__q_named", vec![value, title], start));
            self.eat(TokKind::Comma);
            if !braced
                && (self.at_line_start() || self.at(TokKind::RightBrace) || self.at(TokKind::Eof))
            {
                break;
            }
        }
        if braced {
            self.expect(TokKind::RightBrace);
        }
        out
    }

    /// Rewrites an expression so bare names mean columns: names -> `__q_col("x")`, `&&`/`||` ->
    /// `&`/`|` (column tests don't short-circuit), `if c { a } else { b }` -> `__q_when`, and the
    /// column functions -> `__q_apply(col, "fn", extra)`. Literals and operators carry over, and
    /// `$expr` splices pass through untouched -- the escape back to the surrounding scope.
    fn q_lower(&self, e: &Expr) -> Expr {
        let start = e.span().start();
        match e.kind() {
            ExprKind::Ident(id) => {
                let name = self.new_expr(Literal::String(id.lexeme.clone()), start);
                self.q_call("__q_col", vec![name], start)
            }
            ExprKind::Grouping(g) => self.new_expr(Grouping::new(self.q_lower(&g.inner)), start),
            ExprKind::Evaluation(ev) => self.new_expr(
                Evaluation::new(self.q_lower(&ev.left), ev.op, self.q_lower(&ev.right)),
                start,
            ),
            ExprKind::Equality(eq) => self.new_expr(
                Equality::new(self.q_lower(&eq.left), eq.op, self.q_lower(&eq.right)),
                start,
            ),
            ExprKind::Logical(l) => {
                let op = match l.op {
                    LogicalOp::And => EvaluationOp::And,
                    LogicalOp::Or => EvaluationOp::Or,
                };
                self.new_expr(
                    Evaluation::new(self.q_lower(&l.left), op, self.q_lower(&l.right)),
                    start,
                )
            }
            ExprKind::Unary(u) => self.new_expr(Unary::new(u.op, self.q_lower(&u.right)), start),
            ExprKind::If(i) => {
                let tail = |b: &Expr| match b.kind() {
                    ExprKind::Block(blk) if blk.body.is_empty() => blk.yielded_expr.clone(),
                    _ => Some(b.clone()),
                };
                match (tail(&i.main_body), i.else_expr.as_ref().and_then(tail)) {
                    (Some(a), Some(b)) => self.q_call(
                        "__q_when",
                        vec![
                            self.q_lower(&i.condition),
                            self.q_lower(&a),
                            self.q_lower(&b),
                        ],
                        start,
                    ),
                    _ => e.clone(),
                }
            }
            ExprKind::Call(c) => {
                // `$expr` — the splice marker from `literal`: emit the inside
                // untouched so it evaluates as an ordinary value in scope.
                if c.left.as_ident().is_some_and(|i| i.lexeme == "__q_splice") {
                    return c
                        .arguments
                        .first()
                        .map_or_else(|| e.clone(), |a| a.value.clone());
                }
                let args: Vec<Expr> = c.arguments.iter().map(|a| self.q_lower(&a.value)).collect();
                match c.left.as_ident().map(|i| i.lexeme.as_str()) {
                    Some("count") if args.is_empty() => self.q_call("__q_n", vec![], start),
                    Some(
                        f @ ("sum" | "mean" | "average" | "median" | "min" | "max" | "count"
                        | "n_unique" | "first" | "last" | "is_null" | "is_not_null"
                        | "to_upper" | "to_lower" | "len" | "contains" | "starts_with"
                        | "ends_with" | "fill_null" | "is_in" | "eq" | "neq" | "cast_int"
                        | "cast_float" | "cast_str" | "lag" | "lead" | "difference" | "abs"
                        | "floor" | "ceil" | "round" | "to_datetime" | "epoch_ms" | "strftime"
                        | "truncate" | "year" | "month" | "day" | "hour" | "minute" | "second"
                        | "weekday" | "ordinal_day" | "days_in_month" | "date" | "month_start"
                        | "month_end" | "list_len" | "list_get" | "list_first" | "list_last"
                        | "list_sum" | "list_mean" | "explode"),
                    ) if !args.is_empty() => {
                        let mut it = args.into_iter();
                        let first = it.next().unwrap();
                        let name = self.new_expr(Literal::String(f.to_string()), start);
                        // a second argument lowers like the first — `is_in(discount, $codes)`
                        let extra = match c.arguments.get(1) {
                            Some(a) => self.q_lower(&a.value),
                            None => self.new_expr(Literal::Null, start),
                        };
                        self.q_call("__q_apply", vec![first, name, extra], start)
                    }
                    _ => e.clone(),
                }
            }
            _ => e.clone(),
        }
    }

    fn supreme(&mut self) -> Expr {
        let expr = self.primary();
        self.chain_accesses(expr)
    }

    /// An expression with no postfix on it yet. [Self::chain_accesses] adds those.
    fn primary(&mut self) -> Expr {
        let start = self.next_start();
        // `int::random` etc. -- synthesize an Ident at the expr head so the regular library map
        // lookup handles it. TyKw is only legal here in `TyKw ::` shape.
        if let TokKind::TyKw(ty) = self.peek() {
            self.advance();
            if !self.at(TokKind::DoubleColon) {
                self.expect(TokKind::DoubleColon);
            }
            return self.new_expr(Ident::new(ty.to_string(), self.location(start)), start);
        }
        // `#1.10`: an exact decimal, a `dec` wherever it stands. The `#`
        // hugs the number, as Scheme's `#e0.3` does (r[impl mimas.dec.exact-literal])
        if self.at_exact_literal() {
            self.advance();
            let lit = self.literal();
            if matches!(lit.kind(), ExprKind::Literal(Literal::Int(_) | Literal::Float(_))) {
                self.exact_literals.insert(lit.id());
            }
            return lit;
        }
        self.parentheticals()
    }

    /// Chains any trailing access expressions (`.field`, `[key]`, `::member`,
    /// `()` calls, `!` unwraps, bare `?` postfix) onto an already-parsed expression.
    fn chain_accesses(&mut self, expr: Expr) -> Expr {
        // postfix on a failed expression only ever earns a second error for the same mistake
        if matches!(expr.kind(), ExprKind::Poison(_)) {
            return expr;
        }
        let mut expr = expr;
        loop {
            // `(` / `[` / `!` opening a line start the next expression (`!x` is a prefix not), the same newline rule
            // `infix_binds` applies to operators -- so `2 * 6 is 12` followed by a line
            // `(3 + 4) * 5 is 35` is two checks, not a call on `12`. Inside a group the
            // newline doesn't matter.
            let fresh_line = self.group_depth == 0 && self.at_line_start();
            expr = match self.peek() {
                TokKind::LeftParenthesis | TokKind::LeftSquare | TokKind::Bang if fresh_line => {
                    break expr;
                }
                TokKind::LeftParenthesis => self.call(expr),
                TokKind::LeftSquare | TokKind::HookLeftSquare => self.square_access(expr),
                TokKind::DoubleColon => self.colon_access(expr),
                TokKind::Dot | TokKind::HookDot => self.dot_access(expr),
                // postfix `!` hugs its operand like `x%` — a spaced `!`
                // is *not* an unwrap, it's the next thing: `[!a !b]` is
                // `[!a, !b]` (the `!` starts a fresh element), and
                // `f !x` errors at the statement boundary rather than
                // silently meaning `(f!)(x)`. The `x !`-reads-unwrap
                // spelling is `x!`.
                // r[impl mimas.macro.call]
                TokKind::Bang if self.expr_macro_call(&expr) => self.expr_macro(expr),
                TokKind::Bang
                    if self.glued() && expr.as_ident().is_some_and(|i| i.lexeme == "assert") =>
                {
                    self.assert_or_unwrap(expr)
                }
                TokKind::Bang if self.glued() => self.unwrap(expr),
                // `?` hugs like `!`: `x?` propagates, a spaced `?` starts
                // nothing (`?` opens no expression), so `f ?x` errors
                // rather than silently reading `(f?)(x)`.
                TokKind::Hook if self.glued() => self.propagate(expr),
                // `x%` hugs its operand and spells percent -- `50%` is `50 * 0.01`, the
                // `pct` unit by another name. A spaced `%` is neither postfix nor modulo
                // (`a mod b`); `binary` reports it. While a `±` tolerance parses the
                // marker claim takes precedence (`x ± 10%`).
                TokKind::Percent if !self.pct_marker && self.glued() => {
                    let at = expr.span().start();
                    self.advance();
                    let scale = self.new_expr(Literal::Float(0.01), at);
                    self.new_expr(Evaluation::new(expr, EvaluationOp::Multiply, scale), at)
                }
                // `x↑` — a direction follows its operand and folds the
                // sign in: `v↑` is `+v` along y (up is the positive end)
                // or `-v` in a page that renamed the axis. `x@down` is
                // the word spelling; `x@y` an axis tag without a sign.
                TokKind::RightArrow
                | TokKind::LeftArrow
                | TokKind::UpArrow
                | TokKind::DownArrow
                | TokKind::CircledDot
                | TokKind::CircledTimes
                | TokKind::Rewind
                | TokKind::FastForward
                    if !fresh_line =>
                {
                    self.direction_suffix(expr)
                }
                TokKind::At if !fresh_line => self.at_suffix(expr),
                // `x y` — juxtaposition applies: `f x` is `f(x)`, the same
                // call `f (x)` already was. Same-line only (the `(`/`[`/`!`
                // rule), atoms only (`f - x` stays subtraction), and never
                // inside a whitespace-separated region like `table {}`.
                tok if !fresh_line
                    && self.juxta == Juxta::Apply
                    && tok.starts_juxta_arg()
                    // `1 b => 2`: an atom right before `=>` is the next
                    // match arm's pattern (a missing `,`), not an argument
                    && self.nth(1) != TokKind::FatArrow
                    && !expr
                        .as_ident()
                        .is_some_and(|i| lex::is_operator_word(&i.lexeme)) =>
                {
                    self.juxta_apply(expr)
                }
                _ => break expr,
            }
        }
    }

    /// `x y` — `x` applied to the atom that follows: `f x` is `f(x)`.
    /// The argument is a unary atom (a name with its own postfix chain,
    /// a literal, a group) parsed with juxtaposition off, so `f x y` is
    /// `(f x) y`.
    fn juxta_apply(&mut self, left: Expr) -> Expr {
        let start = left.span().start();
        let arg = self.with_juxta(Juxta::Off, Self::unary);
        let arg = Argument {
            name: None,
            value: arg,
        };
        self.new_expr(Call::new(left, vec![arg]), start)
    }

    /// `x↑` — a direction glyph after its operand. Same desugar the old
    /// `↑x` prefix had: the operand is wrapped in a `+`/`-` whose sign is
    /// the direction's end on its axis, and the wrapper remembers the
    /// axis for the dimension checker.
    fn direction_suffix(&mut self, operand: Expr) -> Expr {
        let start = operand.span().start();
        let tok = self.advance();
        let word = match tok.kind() {
            TokKind::RightArrow => "→",
            TokKind::LeftArrow => "←",
            TokKind::UpArrow => "↑",
            TokKind::DownArrow => "↓",
            TokKind::CircledDot => "⊙",
            TokKind::CircledTimes => "⊗",
            TokKind::Rewind => "⏪",
            TokKind::FastForward => "⏩",
            _ => unreachable!(),
        };
        // the builtin glyphs always resolve — `direction` only fails
        // on words, and a glyph's word is builtin
        let Some((axis, negative)) = shared::units::direction(word) else {
            unreachable!()
        };
        let op = if negative {
            UnaryOp::Negative
        } else {
            UnaryOp::Positive
        };
        let wrapped = self.new_expr(Unary::new(op, operand), start);
        self.directions.insert(wrapped.id(), axis);
        wrapped
    }

    /// `5usd@richer`, `x@y` — a named direction or axis after its
    /// operand. A direction word signs like the glyph; an axis name
    /// tags without signing (`Positive` is the no-op wrap). `@word` is
    /// for directions with no glyph (`richer`, `inward`): a word that
    /// has one spells it — `x↑`, not `x@up`.
    fn at_suffix(&mut self, operand: Expr) -> Expr {
        let start = operand.span().start();
        self.advance(); // `@`
        let name = self.require_ident();
        // r[impl mimas.unit.direction.postfix]
        if let Some(glyph) = shared::units::builtin_direction_glyph(&name.lexeme) {
            self.error(Misdirection {
                src: self.src(),
                at: self.location(start).into(),
                msg: format!("`{}` is a direction, and it has a glyph", name.lexeme),
                label: format!("write `{glyph}` after the value, not `@{name}`", name = name.lexeme),
            });
        }
        if let Some((axis, negative)) = shared::units::direction(&name.lexeme) {
            let op = if negative {
                UnaryOp::Negative
            } else {
                UnaryOp::Positive
            };
            let wrapped = self.new_expr(Unary::new(op, operand), start);
            self.directions.insert(wrapped.id(), axis);
            return wrapped;
        }
        if let Some(axis) = shared::units::axis_lookup(&name.lexeme) {
            let wrapped = self.new_expr(Unary::new(UnaryOp::Positive, operand), start);
            self.directions.insert(wrapped.id(), axis);
            return wrapped;
        }
        self.error(Misdirection {
            src: self.src(),
            at: self.location(start).into(),
            msg: format!("`@{}` is neither an axis nor a direction", name.lexeme),
            label: "an `axis` decl names its two directions: `axis y (down, up)`".into(),
        });
        operand
    }

    /// Postfix chaining for block expressions. `(` and `[` directly after a block are ambiguous
    /// with a following statement (`if c {}` newline `[x]`), so they need parens; `.`/`::`/`!`/`?`
    /// can't start a statement, so once one attaches we hand off to the full `chain_accesses`.
    fn chain_after_block(&mut self, expr: Expr) -> Expr {
        match self.peek() {
            TokKind::Dot
            | TokKind::HookDot
            | TokKind::DoubleColon
            | TokKind::Bang
            | TokKind::Hook => self.chain_accesses(expr),
            _ => expr,
        }
    }

    fn unwrap(&mut self, left: Expr) -> Expr {
        let start = left.span().start();
        self.bump(TokKind::Bang);
        self.new_expr(Unwrap { expr: left }, start)
    }

    /// `expr?` — `expr`'s value, or the enclosing function hands back its
    /// null or its raise (`mimas.propagate`).
    fn propagate(&mut self, left: Expr) -> Expr {
        let start = left.span().start();
        self.bump(TokKind::Hook);
        self.new_expr(Propagate { expr: left }, start)
    }

    /// `assert!(expr)` rewrites to an `if`/`panic` block. Bare `assert!` (no following `(`) is
    /// still unwrap of a binding named `assert`.
    fn assert_or_unwrap(&mut self, assert_ident: Expr) -> Expr {
        let start = assert_ident.span().start();
        self.bump(TokKind::Bang);
        if !self.at(TokKind::LeftParenthesis) {
            return self.new_expr(Unwrap { expr: assert_ident }, start);
        }
        self.bump(TokKind::LeftParenthesis);
        let inner = self.in_group(|p| {
            if p.at(TokKind::RightParenthesis) {
                p.error(AssertArity {
                    src: p.src(),
                    at: p.next_location().into(),
                });
                return p.poison_expr(p.next_start());
            }
            let expr = p.expr();
            if p.eat(TokKind::Comma) && !p.at(TokKind::RightParenthesis) {
                p.error(AssertArity {
                    src: p.src(),
                    at: p.next_location().into(),
                });
            }
            p.expect(TokKind::RightParenthesis);
            expr
        });
        self.rewrite_assert(start, inner)
    }

    fn rewrite_assert(&mut self, start: usize, inner: Expr) -> Expr {
        match Self::skip_groups(&inner).kind().clone() {
            ExprKind::Equality(eq) => {
                let op_text = eq.op.to_string();
                self.rewrite_assert_cmp(start, eq.left, eq.right, eq.op.invert(), op_text)
            }
            ExprKind::In(inn) => {
                let op_text = if inn.condition { "in" } else { "!in" };
                self.rewrite_assert_in(start, inn.left, inn.right, !inn.condition, op_text)
            }
            // `check a ≈ b` reports both sides, as `==` does (r[impl mimas.op.approx])
            ExprKind::Evaluation(ev) if ev.op == EvaluationOp::Approx => {
                self.rewrite_assert_approx(start, ev.left, ev.right)
            }
            // r[impl check.report.call]
            ExprKind::Call(call) => self.rewrite_assert_call(start, call, false),
            ExprKind::Unary(u) if u.op == UnaryOp::Not => {
                match Self::skip_groups(&u.right).kind().clone() {
                    ExprKind::Call(call) => self.rewrite_assert_call(start, call, true),
                    _ => self.rewrite_assert_bool(start, inner),
                }
            }
            _ => self.rewrite_assert_bool(start, inner),
        }
    }

    /// A predicate call: each argument is evaluated once into a temp so a
    /// failure can say what the call saw — `is_root_of(2.0, 1.414…) is false`
    /// rather than the source text alone. Literals and closures stay inline
    /// (a closure's parameter types come from the call) and print as written.
    fn rewrite_assert_call(&mut self, start: usize, call: Call, negated: bool) -> Expr {
        let callee = self.source_of(&call.left);
        let mut lets = Vec::new();
        let mut args = Vec::new();
        let mut parts = vec![FStringPart::Literal(format!("assertion failed: {callee}("))];
        for (i, arg) in call.arguments.into_iter().enumerate() {
            if i > 0 {
                parts.push(FStringPart::Literal(", ".into()));
            }
            if let Some(name) = &arg.name {
                parts.push(FStringPart::Literal(format!("{name} = ")));
            }
            let value = match arg.value.kind() {
                ExprKind::Literal(_) | ExprKind::Closure(_) => {
                    parts.push(FStringPart::Literal(self.source_of(&arg.value)));
                    arg.value
                }
                ExprKind::Ident(_) => {
                    parts.push(FStringPart::Expr(arg.value.clone()));
                    arg.value
                }
                _ => {
                    self.assert_id += 1;
                    let ident = self.ident_at(format!("__assert_arg_{}", self.assert_id), start);
                    lets.push(self.synthetic_let(ident.clone(), arg.value, start));
                    let use_site = self.new_expr(ident, start);
                    parts.push(FStringPart::Expr(use_site.clone()));
                    use_site
                }
            };
            args.push(Argument {
                name: arg.name,
                value,
            });
        }
        parts.push(FStringPart::Literal(
            if negated { ") is true" } else { ") is false" }.into(),
        ));
        // the values the call saw, in argument order — what the host
        // opens when this check fails. A closure sits out: its parameter
        // types come from the call itself, so a bare `__check_operand`
        // can't type it, and an opaque `fn` tree adds nothing over the
        // source text the message already shows.
        let witnesses = args
            .iter()
            .filter(|arg| !matches!(arg.value.kind(), ExprKind::Closure(_)))
            .map(|arg| arg.value.clone())
            .collect();
        let rebuilt = self.new_expr(Call::new(call.left, args), start);
        let condition = if negated {
            rebuilt
        } else {
            self.new_expr(Unary::new(UnaryOp::Not, rebuilt), start)
        };
        let message = self.new_expr(FString::new(parts), start);
        self.assert_lets_if(start, lets, condition, message, witnesses)
    }

    fn skip_groups(mut expr: &Expr) -> &Expr {
        while let ExprKind::Grouping(grouping) = expr.kind() {
            expr = &grouping.inner;
        }
        expr
    }

    fn rewrite_assert_cmp(
        &mut self,
        start: usize,
        left: Expr,
        right: Expr,
        inverted: EqualityOp,
        op_text: String,
    ) -> Expr {
        let (left_ident, right_ident) = self.next_assert_temps(start);
        let (left_expr, left_let) = self.assert_operand(left, left_ident, start);
        let (right_expr, right_let) = self.assert_operand(right, right_ident, start);
        let condition = self.new_expr(
            Equality::new(left_expr.clone(), inverted, right_expr.clone()),
            start,
        );
        let message =
            self.assert_pair_message(left_expr.clone(), right_expr.clone(), &op_text, start);
        self.assert_lets_if(
            start,
            [left_let, right_let].into_iter().flatten().collect(),
            condition,
            message,
            vec![left_expr, right_expr],
        )
    }

    /// `check a ≈ b`: each side evaluated once, and a failure says both
    /// values, `assertion failed: 21.77 ≈ 30.0`.
    fn rewrite_assert_approx(&mut self, start: usize, left: Expr, right: Expr) -> Expr {
        let (left_ident, right_ident) = self.next_assert_temps(start);
        let (left_expr, left_let) = self.assert_operand(left, left_ident, start);
        let (right_expr, right_let) = self.assert_operand(right, right_ident, start);
        let close = self.new_expr(
            Evaluation::new(left_expr.clone(), EvaluationOp::Approx, right_expr.clone()),
            start,
        );
        let condition = self.new_expr(Unary::new(UnaryOp::Not, close), start);
        let message = self.assert_pair_message(left_expr.clone(), right_expr.clone(), "≈", start);
        self.assert_lets_if(
            start,
            [left_let, right_let].into_iter().flatten().collect(),
            condition,
            message,
            vec![left_expr, right_expr],
        )
    }

    /// The use-site operand for an assert pair: literals don't need the
    /// evaluate-once temp (and `let __n = null` would trip the bare-null
    /// lint), so they inline; anything else gets `let __tmp = expr`.
    fn assert_operand(&mut self, expr: Expr, ident: Ident, start: usize) -> (Expr, Option<Stmt>) {
        if matches!(expr.kind(), ExprKind::Literal(_)) {
            return (expr, None);
        }
        (
            self.new_expr(ident.clone(), start),
            Some(self.synthetic_let(ident, expr, start)),
        )
    }

    fn rewrite_assert_in(
        &mut self,
        start: usize,
        left: Expr,
        right: Expr,
        fail_when: bool,
        op_text: &str,
    ) -> Expr {
        let (left_ident, right_ident) = self.next_assert_temps(start);
        let (left_expr, left_let) = self.assert_operand(left, left_ident, start);
        let (right_expr, right_let) = self.assert_operand(right, right_ident, start);
        let condition = self.new_expr(
            In::new(left_expr.clone(), right_expr.clone(), fail_when),
            start,
        );
        let message = self.assert_pair_message(left_expr.clone(), right_expr.clone(), op_text, start);
        self.assert_lets_if(
            start,
            [left_let, right_let].into_iter().flatten().collect(),
            condition,
            message,
            vec![left_expr, right_expr],
        )
    }

    /// The fallback `check <expr>`: the checked value itself is what a
    /// failure should open, so it goes through `assert_operand` like a
    /// comparison side — evaluated once into a temp, then both the
    /// condition and the witness read the temp. An ident or literal
    /// needs no temp: reading it again costs nothing.
    fn rewrite_assert_bool(&mut self, start: usize, inner: Expr) -> Expr {
        let snippet = self.source_of(&inner);
        let (value, value_let) = match inner.kind() {
            ExprKind::Literal(_) | ExprKind::Ident(_) => (inner, None),
            _ => {
                self.assert_id += 1;
                let ident = self.ident_at(format!("__assert_arg_{}", self.assert_id), start);
                self.assert_operand(inner, ident, start)
            }
        };
        let condition = self.new_expr(Unary::new(UnaryOp::Not, value.clone()), start);
        let message = self.new_expr(
            Literal::String(format!("assertion failed: {snippet}")),
            start,
        );
        self.assert_lets_if(
            start,
            value_let.into_iter().collect(),
            condition,
            message,
            vec![value],
        )
    }

    fn next_assert_temps(&mut self, start: usize) -> (Ident, Ident) {
        self.assert_id += 1;
        let n = self.assert_id;
        (
            self.ident_at(format!("__assert_left_{n}"), start),
            self.ident_at(format!("__assert_right_{n}"), start),
        )
    }

    fn ident_at(&self, name: impl Into<String>, start: usize) -> Ident {
        Ident::new(name, self.location(start))
    }

    /// `let __assert_* = value` gets the operand's own span, not the whole
    /// check's: a failed check's tape (r[impl check.tape] in emit) records
    /// each operand let at the statement's location, so the strip names the
    /// subexpression that produced the value.
    fn synthetic_let(&self, name: Ident, value: Expr, start: usize) -> Stmt {
        let loc = value.location();
        Stmt::new(
            Let {
                left: Pat::new(PatKind::Ident(name), self.location(start)),
                annotation: None,
                right: value,
                else_branch: None,
            }
            .into(),
            NodeId::new(),
            loc,
        )
    }

    fn assert_pair_message(&self, left: Expr, right: Expr, op: &str, start: usize) -> Expr {
        self.new_expr(
            FString::new(vec![
                FStringPart::Literal("assertion failed: ".into()),
                FStringPart::Expr(left),
                FStringPart::Literal(format!(" {op} ")),
                FStringPart::Expr(right),
            ]),
            start,
        )
    }

    /// The assert desugar's tail: `body` (the operand lets), then
    /// `if (condition) { __check_operand(w0); __check_operand(w1); …;
    /// panic(message) }`. `witnesses` are the operand values the message
    /// names — the use-site exprs, temps or inlined literals, so nothing
    /// with an effect runs twice. One call each, not one `[…]` list:
    /// each keeps its operand's own type where a list would force them
    /// to share one. `__check_operand` hands them to the host before
    /// `panic` unwinds: a failed check keeps its operands as inspectable
    /// trees, not just inside the message's text.
    fn assert_lets_if(
        &self,
        start: usize,
        mut body: Vec<Stmt>,
        condition: Expr,
        message: Expr,
        witnesses: Vec<Expr>,
    ) -> Expr {
        let mut then = Vec::with_capacity(witnesses.len() + 1);
        for witness in witnesses {
            let record = self.new_expr(
                Call::new(
                    self.new_expr(self.ident_at("__check_operand", start), start),
                    vec![Argument {
                        name: None,
                        value: witness,
                    }],
                ),
                start,
            );
            then.push(self.new_stmt(StmtKind::Expr(record), start));
        }
        let panic_call = self.new_expr(
            Call::new(
                self.new_expr(self.ident_at("panic", start), start),
                vec![Argument {
                    name: None,
                    value: message,
                }],
            ),
            start,
        );
        then.push(self.new_stmt(StmtKind::Expr(panic_call), start));
        let then_body = self.new_expr(
            Block {
                body: then,
                yielded_expr: None,
            },
            start,
        );
        let iff = self.new_expr(
            If {
                condition,
                main_body: then_body,
                else_expr: None,
                binding: None,
            },
            start,
        );
        body.push(self.new_stmt(StmtKind::Expr(iff), start));
        self.new_expr(
            Block {
                body,
                yielded_expr: None,
            },
            start,
        )
    }

    fn call(&mut self, left: Expr) -> Expr {
        let start = left.span().start();
        // `f_(x)` isn't a call anyone means to write -- `name_(…)` only
        // exists as destructure sugar in patterns, so treat it as a mistake.
        let tail = match left.kind() {
            ExprKind::Ident(id) => Some(id),
            ExprKind::Access(Access::DoubleColon { right, .. }) => Some(right),
            ExprKind::Access(Access::Dot { right, .. }) => match right.kind() {
                ExprKind::Ident(id) => Some(id),
                _ => None,
            },
            _ => None,
        };
        if let Some(id) = tail {
            if id.lexeme.len() > 1 && id.lexeme.ends_with('_') {
                self.error(Misdirection {
                    src: self.src(),
                    at: left.location().into(),
                    msg: format!(
                        "`{}(…)` reads as prefix destructuring, which only exists in pattern position",
                        id.lexeme
                    ),
                    label: format!(
                        "calling a `{}` name isn't valid -- rename it to drop the trailing `_`",
                        id.lexeme
                    ),
                });
            }
        }
        self.bump(TokKind::LeftParenthesis);
        let mut named = false;
        let arguments = self.in_group(|p| {
            p.with_juxta(Juxta::Comma, |p| {
                p.list(TokKind::RightParenthesis, TokKind::starts_expr, |p| {
                    let value = p.struct_literals(true, Self::expr);
                    if !p.at(TokKind::Equal) {
                        if named {
                            p.error(NamedBeforePositional {
                                src: p.src(),
                                at: value.location().into(),
                            });
                        }
                        return Argument { name: None, value };
                    }
                    named = true;
                    let name = match value.kind() {
                        // a parameter's name, never a template's renamed local
                        ExprKind::Ident(ident) => Some(macros::unhygienic(ident)),
                        _ => {
                            p.error(p.unexpected_token());
                            None
                        }
                    };
                    p.bump(TokKind::Equal);
                    let value = p.expr();
                    Argument { name, value }
                })
            })
        });
        self.new_expr(Call::new(left, arguments), start)
    }

    fn dot_access(&mut self, left: Expr) -> Expr {
        let start = left.span().start();
        let kind = AccessKind::try_from(self.advance().kind()).expect("dispatched on a dot");
        // the member spans just its own token, not back to the start of the receiver
        let member_start = self.next_start();
        let right = match self.peek() {
            TokKind::Ident(_) => {
                let ident = self.require_ident();
                self.new_expr(ident, member_start)
            }
            // take the int token directly -- `parser.literal()` would chain further accesses,
            // which would steal a trailing `.foo()` from the *outer* dot (`a.0.pairs()` would
            // misparse as `a . (0.pairs())`). The outer chain_accesses loop owns chaining.
            TokKind::Int(index) => {
                self.advance();
                self.new_expr(Literal::Int(index), member_start)
            }
            _ => {
                self.error_here(InvalidDotAccess {
                    src: self.src(),
                    at: self.location(start).into(),
                });
                self.poison_expr(start)
            }
        };
        self.new_expr(Access::Dot { left, right, kind }, start)
    }

    fn colon_access(&mut self, left: Expr) -> Expr {
        let start = left.span().start();
        self.bump(TokKind::DoubleColon);
        let right = self.require_member_name();
        self.new_expr(Access::DoubleColon { left, right }, start)
    }

    fn square_access(&mut self, left: Expr) -> Expr {
        let start = left.span().start();
        let kind = AccessKind::try_from(self.advance().kind()).expect("dispatched on a square");
        // `xs[..b]`: a slice from the start (r[impl mimas.slice])
        if kind == AccessKind::Direct && (self.at(TokKind::DoubleDot) || self.at(TokKind::DoubleDotEqual)) {
            let inclusive = self.at(TokKind::DoubleDotEqual);
            self.advance();
            let end = self.struct_literals(true, Self::expr);
            self.expect(TokKind::RightSquare);
            let zero = self.new_expr(Literal::Int(0), start);
            return self.slice_call(left, zero, Some(end), inclusive, start);
        }
        let key = self.struct_literals(true, Self::expr);
        // `xs[a..b]`, `xs[a..=b]`, `xs[a..]`
        if kind == AccessKind::Direct && (self.at(TokKind::DoubleDot) || self.at(TokKind::DoubleDotEqual)) {
            let inclusive = self.at(TokKind::DoubleDotEqual);
            self.advance();
            let end = if self.at(TokKind::RightSquare) {
                None
            } else {
                Some(self.struct_literals(true, Self::expr))
            };
            self.expect(TokKind::RightSquare);
            return self.slice_call(left, key, end, inclusive, start);
        }
        let key = if self.eat(TokKind::Comma) {
            let mut keys = vec![key];
            while !self.at(TokKind::RightSquare) && !self.at(TokKind::Eof) {
                keys.push(self.struct_literals(true, Self::expr));
                if !self.eat(TokKind::Comma) { break; }
            }
            self.new_expr(Literal::Tuple(keys), start)
        } else { key };
        self.expect(TokKind::RightSquare);
        self.new_expr(Access::Square { left, key, kind }, start)
    }

    /// A slice as the method it means: `xs[a..b]` is `xs.slice(a, b)`,
    /// `xs[a..=b]` is `xs.slice(a, b + 1)`, and `xs[a..]` is
    /// `xs.slice_from(a)`. `xs` is evaluated once. r[impl mimas.slice]
    fn slice_call(&self, left: Expr, from: Expr, end: Option<Expr>, inclusive: bool, start: usize) -> Expr {
        let at = self.location(start);
        let method = |name: &str| Ident::new(name, at);
        let arg = |value: Expr| Argument { name: None, value };
        let (name, args) = match end {
            None => ("slice_from", vec![arg(from)]),
            Some(end) => {
                let end = if inclusive {
                    let one = self.new_expr(Literal::Int(1), start);
                    self.new_expr(Evaluation::new(end, EvaluationOp::Plus, one), start)
                } else {
                    end
                };
                ("slice", vec![arg(from), arg(end)])
            }
        };
        let callee = self.new_expr(
            Access::Dot { left, right: self.new_expr(method(name), start), kind: AccessKind::Direct },
            start,
        );
        self.new_expr(Call::new(callee, args), start)
    }

    fn parentheticals(&mut self) -> Expr {
        let start = self.next_start();
        if !self.eat(TokKind::LeftParenthesis) {
            return if self.record_start() { self.record_literal() } else { self.block() };
        }
        // `(...)` is a fresh expression boundary: modes disarmed by an
        // enclosing head (`if`, `for .. in`, `table`) re-arm inside
        self.in_group(|p| {
            p.with_juxta(Juxta::Apply, |p| {
                if p.eat(TokKind::RightParenthesis) {
                    return p.new_expr(Literal::Unit, start);
                }
                let first = p.struct_literals(true, Self::expr);
                if !p.eat(TokKind::Comma) {
                    p.expect(TokKind::RightParenthesis);
                    return p.new_expr(Grouping::new(first), start);
                }
                let mut members = vec![first];
                members.extend(
                    p.list(TokKind::RightParenthesis, TokKind::starts_expr, |p| {
                        p.struct_literals(true, Self::expr)
                    }),
                );
                p.new_expr(Literal::Tuple(members), start)
            })
        })
    }

    /// At a `{` in expression position: does a bare-brace literal open here rather
    /// than a block? `{ name: …` and `{ "key": …` always do (no statement starts
    /// with a name and one colon), `{ a, b }` does (a comma never follows a block's
    /// first statement), and `{ name = …, …` does once a comma follows the first
    /// value on its line — a lone `{ x = 1 }` stays the block that assigns `x`.
    /// r[impl mimas.record.literal] r[impl mimas.record.separator]
    fn record_start(&self) -> bool {
        if self.peek() != TokKind::LeftBrace {
            return false;
        }
        let key = self.nth(1);
        let named = key.is_ident();
        if !named && !matches!(key, TokKind::String(_)) {
            return false;
        }
        match self.nth(2) {
            TokKind::Colon => true,
            TokKind::Comma => named,
            TokKind::Equal if named => {
                // scan the first value at bracket depth 0: a comma on its line
                // makes it an entry; `;`, `}` or a line break first, a block
                let mut depth = 0i32;
                let mut i = self.next + 3;
                while i < self.tokens.len() {
                    let tok = &self.tokens[i];
                    if depth == 0 && self.meta[i].newline {
                        return false;
                    }
                    match tok.kind() {
                        TokKind::LeftParenthesis | TokKind::LeftSquare | TokKind::LeftBrace => depth += 1,
                        TokKind::RightParenthesis | TokKind::RightSquare => depth -= 1,
                        TokKind::RightBrace => {
                            if depth == 0 {
                                return false;
                            }
                            depth -= 1
                        }
                        TokKind::Comma if depth == 0 => return true,
                        TokKind::SemiColon | TokKind::Eof if depth == 0 => return false,
                        _ => {}
                    }
                    i += 1;
                }
                false
            }
            _ => false,
        }
    }

    /// `{ a: 1, b = 2, c, "d e": 3 }` — entries take `:` or `=`, a bare name is
    /// shorthand for `name: name`, a quoted key is text. Commas are optional
    /// between entries. r[impl mimas.record.literal]
    fn record_literal(&mut self) -> Expr {
        let start = self.next_start();
        self.bump(TokKind::LeftBrace);
        let elements = self.with_juxta(Juxta::Comma, |p| {
            p.list(
                TokKind::RightBrace,
                |k| k.is_ident() || matches!(k, TokKind::String(_)),
                |p| {
                    let location = p.next_location();
                    let key = match p.peek() {
                        TokKind::String(_) => {
                            let lit = Literal::try_from(p.peek()).ok();
                            p.advance();
                            match lit {
                                Some(Literal::String(text)) => Ident::new(&text, location),
                                _ => Ident::new("", location),
                            }
                        }
                        _ => p.require_ident(),
                    };
                    if p.at(TokKind::Comma) || p.at(TokKind::RightBrace) {
                        // shorthand: `{ qty, price }` is `{ qty: qty, price: price }`
                        let local = p.hygiene_use(key.clone());
                        let value = p.new_expr(local, key.location.span().start());
                        return (key, value);
                    }
                    if !p.eat(TokKind::Colon) {
                        p.expect(TokKind::Equal);
                    }
                    (key, p.struct_literals(true, Self::expr))
                },
            )
        });
        let expr = self.new_expr(Literal::Record(elements), start);
        self.chain_accesses(expr)
    }

    fn block(&mut self) -> Expr {
        let start = self.next_start();
        if !self.eat(TokKind::LeftBrace) {
            return self.ident();
        }
        let mut body: Vec<Stmt> = vec![];
        // A block is a fresh statement list. Even when the block sits inside `(...)`,
        // statements inside must not inherit infix-across-newline permission -- otherwise
        // `({ c = a\n-b })` would still parse as `c = a - b`.
        let yielded_expr = self.with_group_depth(0, |p| {
            loop {
                if p.eat(TokKind::RightBrace) {
                    break None;
                }
                if p.at(TokKind::Eof) {
                    p.expect(TokKind::RightBrace);
                    break None;
                }
                if !p.peek().starts_stmt() {
                    p.reject_stmt(p.unexpected_token());
                    continue;
                }
                let node_start = p.next_start();
                match p.node() {
                    BlockElement::Stmt(stmt) => body.push(stmt),
                    // last node in the block -- this is the yield, not a new stmt
                    BlockElement::MaybeYield(expr) if p.eat(TokKind::RightBrace) => {
                        break Some(expr);
                    }
                    BlockElement::MaybeYield(expr) => body.push(p.expr_stmt(expr, node_start)),
                }
            }
        });
        self.new_expr(Block { body, yielded_expr }, start)
    }

    /// The bottom of the descent. Anything that isn't an identifier by now isn't an expression.
    fn ident(&mut self) -> Expr {
        let start = self.next_start();
        match self.peek() {
            // we're just gonna hijack this guy...
            TokKind::SelfKeyword => {
                let lexeme = self.advance().to_string();
                self.new_expr(Ident::new(lexeme, self.location(start)), start)
            }
            TokKind::Ident(_) => {
                let ident = self.require_ident();
                let ident = self.hygiene_use(ident);
                self.new_expr(ident, start)
            }
            _ => {
                self.expected("expression");
                self.poison_expr(start)
            }
        }
    }
}

// General/helpers
impl<'s> Parser<'s> {
    /// Turns a bare expression into a statement, which has to end like one.
    fn expr_stmt(&mut self, expr: Expr, start: usize) -> Stmt {
        // Expr's that end with blocks do not need semicolons.
        let semicolon_optional = matches!(
            expr.kind(),
            ExprKind::If(_)
                | ExprKind::Match(_)
                | ExprKind::For(_)
                | ExprKind::Block(_)
                | ExprKind::Loop(_)
                | ExprKind::While(_)
                | ExprKind::Break(_)
                | ExprKind::Continue(_)
                | ExprKind::Collect(_)
                | ExprKind::Return(_)
                | ExprKind::Raise(_),
        );

        // a keyword from another language (the `elif` in `if a {} elif b {}`) parses as a
        // statement of its own, so the next token is no help in spotting it
        if let ExprKind::Ident(ident) = expr.kind()
            && let Some((msg, label)) = foreign_spelling(&ident.lexeme)
            && !self.at(TokKind::SemiColon)
        {
            self.error(Misdirection {
                src: self.src(),
                at: ident.location.into(),
                msg: msg.into(),
                label: label.into(),
            });
        }

        let stmt = self.new_stmt(StmtKind::Expr(expr), start);
        if semicolon_optional || self.at(TokKind::Eof) {
            // The end of input ends an expression statement (Go's rule):
            // a file — or a page's last card — may close on a bare
            // expression, whose value a host echoes. A `let` or an
            // assignment still wants its `;`.
            // r[impl mimas.stmt.trailing]
            self.eat(TokKind::SemiColon);
        } else {
            self.end_stmt(stmt.location());
        }
        stmt
    }

    /// Ends an item the way its kind does: a `const` or a `use` needs its `;`, and the rest
    /// may have one.
    fn end_item(&mut self, item: &Item) {
        if matches!(item.kind(), ItemKind::Const(_) | ItemKind::Use(_)) {
            self.end_stmt(item.location());
        } else {
            self.eat(TokKind::SemiColon);
        }
    }

    /// Ends a statement on its `;` -- or on the newline itself (semicolon insertion, Go's
    /// rule: a newline where the grammar wants a `;` counts as one), or right before the
    /// block's closing `}` (Go's other rule, and Rust's: `{ i += 1 }` needs no `;`). The `}`
    /// is left for the block, which then has no yielded value -- the same as if the `;` were
    /// written. When something the statement couldn't use is in the way, the rest of the
    /// line goes with it.
    fn end_stmt(&mut self, stmt: Location) {
        if self.eat(TokKind::SemiColon) || self.at_line_start() || self.at(TokKind::RightBrace) {
            return;
        }
        let stray = !self.peek().ends_stmt() && !self.at_line_start();
        if let TokKind::Ident(name) = self.peek()
            && let Some((msg, label)) = foreign_spelling(name)
        {
            self.error(Misdirection {
                src: self.src(),
                at: self.next_location().into(),
                msg: msg.into(),
                label: label.into(),
            });
        } else if stray {
            self.error(self.unexpected_token());
        } else {
            self.error(MissingSemiColon {
                src: self.src(),
                at: stmt.into(),
            });
        }
        if stray {
            self.skip_stmt();
            self.eat(TokKind::SemiColon);
        }
    }

    /// The value of a `return` or a `break`, when there is one.
    fn optional_expr(&mut self) -> Option<Expr> {
        self.peek().starts_expr().then(|| self.expr())
    }

    fn annotation(&mut self) -> Annotation {
        fn inner(parser: &mut Parser) -> Annotation {
            let atom = parser.annotation_atom();

            if parser.at(TokKind::Plus) {
                let Annotation::Ty(first) = atom else {
                    parser.reject(NonPactInBound {
                        src: parser.src(),
                        at: parser.next_location().into(),
                    });
                    return Annotation::Poison(Poison);
                };
                let mut idents = vec![first];
                while parser.eat(TokKind::Plus) {
                    idents.push(parser.require_ident());
                }
                if parser.at(TokKind::Hook) || parser.at(TokKind::Bang) {
                    parser.reject(PactBoundConstraint {
                        src: parser.src(),
                        at: parser.next_location().into(),
                    });
                }
                return Annotation::Bounds(idents);
            }

            let mut annotation = atom;
            loop {
                if parser.eat(TokKind::Hook) {
                    annotation = Annotation::Option(Box::new(annotation));
                } else if parser.eat(TokKind::Bang) {
                    annotation = Annotation::Result(Box::new(annotation));
                } else if parser.at(TokKind::DoubleHook) {
                    // `T??` lexes as one DoubleHook (the coalesce operator), so without this
                    // arm it dies on a generic "expected token"
                    parser.error(DoubledOption {
                        src: parser.src(),
                        at: parser.next_location().into(),
                    });
                    parser.bump(TokKind::DoubleHook);
                    annotation = Annotation::Option(Box::new(annotation));
                } else {
                    break;
                }
            }
            if parser.at(TokKind::Plus) {
                parser.reject(PactBoundConstraint {
                    src: parser.src(),
                    at: parser.next_location().into(),
                });
            }
            annotation
        }
        // `{V}` is the dict type (r[impl mimas.dict.type]); `{str: int}` is
        // Rust's `HashMap<str, int>` habit — keys are always `str`, so the value
        // type alone goes in the braces. `reject` skips the whole brace group.
        // `str` lexes as a type keyword, not a name -- any token then `:` is the habit
        let key_colon = self.at(TokKind::LeftBrace) && self.nth(2) == TokKind::Colon;
        if key_colon || !self.peek().starts_annotation() {
            if key_colon {
                self.reject(Misdirection {
                    src: self.src(),
                    at: self.next_location().into(),
                    msg: "`{str: V}` isn't a type -- a dict type is `{V}` (keys are always `str`)"
                        .into(),
                    label: "not a type -- write `{V}`".into(),
                });
            } else {
                self.expected("type");
            }
            return Annotation::Poison(Poison);
        }
        self.nested(inner).unwrap_or(Annotation::Poison(Poison))
    }

    fn annotation_atom(&mut self) -> Annotation {
        match self.peek() {
            TokKind::Ident(_) => {
                let mut segments = vec![self.require_ident()];
                while self.eat(TokKind::DoubleColon) {
                    segments.push(self.require_ident());
                }
                match segments.len() {
                    // `m^2` arrives inside the ident's lexeme now, so a
                    // `^n` tail on a lone segment means quantity too.
                    1 if matches!(self.peek(), TokKind::Slash | TokKind::Star | TokKind::Caret)
                        || segments[0].lexeme.rsplit_once('^').is_some_and(|(_, t)| {
                            !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
                        }) =>
                    {
                        self.quantity_annotation(segments.remove(0))
                    }
                    // `List<T>`, `mod::Pair<A, B>`, or `Interval<kW>` (a unit arg -- the
                    // checker knows whether the head is generic or takes a unit)
                    _ if self.at(TokKind::Less) => {
                        self.bump(TokKind::Less);
                        let args = self.list(TokKind::Greater, TokKind::starts_annotation, |p| {
                            p.annotation()
                        });
                        Annotation::Applied(segments, args)
                    }
                    // `px↕`, `usd@wealth`: a lone unit followed by its axis
                    1 if matches!(
                        self.peek(),
                        TokKind::LeftRightArrow | TokKind::UpDownArrow | TokKind::At
                    ) =>
                    {
                        self.quantity_annotation(segments.remove(0))
                    }
                    1 => Annotation::Ty(segments.remove(0)),
                    _ => Annotation::Path(segments),
                }
            }
            TokKind::TyKw(tykw) => {
                self.advance();
                Annotation::Kw(tykw)
            }
            TokKind::LeftParenthesis => self.paren_annotation(),
            TokKind::LeftSquare => {
                self.bump(TokKind::LeftSquare);
                let inner = self.annotation();
                // `[T; n]` pins an exact length, and `[T; m, n]` nests: the first dim
                // is the outermost (`[f32; 4, 8]` is 4 rows of `[f32; 8]`).
                let mut dims = vec![];
                if self.eat(TokKind::SemiColon) {
                    loop {
                        match self.peek() {
                            TokKind::Int(n) => {
                                self.advance();
                                dims.push(usize::try_from(n).unwrap_or(usize::MAX));
                            }
                            _ => {
                                self.expected("array size");
                                break;
                            }
                        }
                        if !self.eat(TokKind::Comma) {
                            break;
                        }
                    }
                }
                self.expect(TokKind::RightSquare);
                Annotation::Array(Box::new(inner), dims)
            }
            // r[impl mimas.dict.type] `{float}` is the dict type
            TokKind::LeftBrace => {
                self.advance();
                let inner = self.annotation();
                self.expect(TokKind::RightBrace);
                Annotation::Dictionary(Box::new(inner))
            }
            _ => {
                self.expected("type");
                Annotation::Poison(Poison)
            }
        }
    }

    /// The rest of a compound unit type (`usd/kWh`, `m/s^2`, `W*h`) after its first name: signed
    /// exponents, `/` flipping the next term.
    fn quantity_annotation(&mut self, first: Ident) -> Annotation {
        let mut parts = vec![self.unit_name_exp(first)];
        loop {
            let sign = match self.peek() {
                TokKind::Star => 1,
                TokKind::Slash => -1,
                _ => break,
            };
            self.advance();
            let unit = self.require_ident();
            let (unit, exp) = self.unit_name_exp(unit);
            parts.push((unit, sign * exp));
        }
        // a trailing axis: `px/s↕`, `index@wealth` -- an axis glyph is a unit
        // name to `units::lookup`, and `@name` the same for a declared axis
        match self.peek() {
            TokKind::LeftRightArrow | TokKind::UpDownArrow => {
                let tok = self.advance();
                let glyph = if tok.kind() == TokKind::LeftRightArrow { "↔" } else { "↕" };
                parts.push((Ident::new(glyph, Location::from(tok.location())), 1));
            }
            TokKind::At => {
                let at = self.advance();
                let name = self.require_ident();
                parts.push((Ident::new(format!("@{}", name.lexeme), Location::from(at.location())), 1));
            }
            _ => {}
        }
        Annotation::Quantity(parts)
    }

    /// A unit name and its exponent. `s^2` lexes as one ident now (`^`
    /// joins a name like `_` does), so a `^n` tail splits off the lexeme;
    /// `s^-2` or spaced `s ^ 2` still come through as a Caret token.
    fn unit_name_exp(&mut self, unit: Ident) -> (Ident, i8) {
        let Some(caret) = unit.lexeme.rfind('^') else {
            let exp = self.unit_exponent();
            return (unit, exp);
        };
        let tail = &unit.lexeme[caret + 1..];
        // `x^foo` stays one name — only a digits tail is an exponent.
        match tail.parse::<i8>() {
            Ok(n) if !tail.is_empty() => (Ident::new(&unit.lexeme[..caret], unit.location), n),
            _ => {
                let exp = self.unit_exponent();
                (unit, exp)
            }
        }
    }

    /// An optional `^n` (or `^-n`) after a unit name.
    fn unit_exponent(&mut self) -> i8 {
        if !self.eat(TokKind::Caret) {
            return 1;
        }
        let neg = self.eat(TokKind::Minus);
        match self.peek() {
            TokKind::Int(n) if n <= i8::MAX as i64 => {
                self.advance();
                if neg { -(n as i8) } else { n as i8 }
            }
            _ => {
                self.expected("exponent");
                1
            }
        }
    }

    /// Everything that opens with a `(`: unit, a tuple, a function, or a parenthesized bound.
    fn paren_annotation(&mut self) -> Annotation {
        let open = self.next_location();
        self.bump(TokKind::LeftParenthesis);
        let mut members = vec![];
        let mut tuple = false;
        if !self.eat(TokKind::RightParenthesis) {
            members.push(self.annotation());
            tuple = self.eat(TokKind::Comma);
            if tuple {
                members.extend(self.list(
                    TokKind::RightParenthesis,
                    TokKind::starts_annotation,
                    Self::annotation,
                ));
            } else {
                self.expect(TokKind::RightParenthesis);
            }
        }

        if self.eat(TokKind::Arrow) {
            return Annotation::Function(members, Box::new(self.annotation()));
        }
        if tuple {
            return Annotation::Tuple(members);
        }
        match members.pop() {
            None => Annotation::Unit,
            // no use complaining about the parens when what's inside them didn't parse
            Some(inner @ (Annotation::Bounds(_) | Annotation::Poison(_))) => inner,
            Some(single) => {
                self.error(SingleTypeParens {
                    src: self.src(),
                    at: open.into(),
                });
                single
            }
        }
    }

    /// Parses `element`s up to `closer` (whose opener is already taken), with a `,` between
    /// them and maybe one after the last.
    fn list<T>(
        &mut self,
        closer: TokKind<'s>,
        starts: impl Fn(TokKind<'s>) -> bool,
        mut element: impl FnMut(&mut Self) -> T,
    ) -> Vec<T> {
        self.sequence(closer, &starts, |p| {
            let parsed = element(p);
            // a missing `,` right before another element is reported, and the list carries on --
            // unless the sequence is comma-optional (`Juxta::Comma`), where it just separates
            if !p.eat(TokKind::Comma) && starts(p.peek()) && p.juxta != Juxta::Comma {
                p.expect(TokKind::Comma);
            }
            parsed
        })
    }

    /// Parses `element`s up to `closer`. An element begins on a token `starts` accepts, and has
    /// to take it. Anything else is reported and skipped through the next `,`. A token that
    /// something further out is waiting for ends the sequence early.
    fn sequence<T>(
        &mut self,
        closer: TokKind<'s>,
        starts: impl Fn(TokKind<'s>) -> bool,
        mut element: impl FnMut(&mut Self) -> T,
    ) -> Vec<T> {
        let mut elements = vec![];
        loop {
            // `Outer<Inner<T>>` lexes its tail as one `>>`: a list closed by `>`
            // takes the first half and leaves a `>` for the list around it
            if closer == TokKind::Greater && self.at(TokKind::DoubleRightCaret) {
                self.split_double_caret();
            }
            if self.eat(closer) {
                break;
            }
            let kind = self.peek();
            if starts(kind) {
                let at = self.next;
                elements.push(element(self));
                // A `starts`-accepted token the element left behind
                // feeds the next iteration forever (`#[` inside call
                // args was one: `Hash` starts an expr for the `#1.10`
                // literal but `#[name]` parses nothing). Report it and
                // step past.
                if self.next == at {
                    self.error(self.unexpected_token());
                    self.skip();
                }
            } else if kind.ends_list() {
                self.expect(closer);
                break;
            } else {
                self.error(self.unexpected_token());
                while !self.at(TokKind::Comma) && !self.peek().is_boundary() {
                    self.skip();
                }
                self.eat(TokKind::Comma);
            }
        }
        elements
    }

    /// Reports that `what` was expected where the next token is. The token is skipped, unless
    /// something further out is waiting for it.
    fn expected(&mut self, what: &'static str) {
        self.error_here(Expected {
            src: self.src(),
            at: self.next_location().into(),
            what,
        });
        if !self.peek().is_anchor() {
            self.skip();
        }
    }

    /// Reports the next token with `err`, then skips it and the rest of its statement.
    fn reject_stmt(&mut self, err: impl Into<shared::Error>) {
        self.reject(err);
        self.skip_stmt();
        self.eat(TokKind::SemiColon);
    }

    /// Reports the next token with `err`, then skips it.
    fn reject(&mut self, err: impl Into<shared::Error>) {
        self.error_here(err);
        self.skip();
    }

    /// Skips what's left of a statement that stopped making sense, up to its `;` or the end of
    /// its line.
    fn skip_stmt(&mut self) {
        while !self.peek().ends_stmt() && !self.at_line_start() {
            self.skip();
        }
    }

    /// Skips the next token, along with everything up to its matching closer if it opens a
    /// group. A closer of the wrong kind stops the skip (it belongs to something further out).
    /// Whatever goes wrong where the skip ends is fallout.
    fn skip(&mut self) {
        let mut open: Vec<TokKind<'s>> = vec![];
        loop {
            let kind = self.peek();
            let mismatched = kind.is_closer() && open.last().is_some_and(|open| *open != kind);
            if kind == TokKind::Eof || mismatched {
                break;
            }
            self.advance();
            match kind {
                TokKind::LeftParenthesis => open.push(TokKind::RightParenthesis),
                TokKind::LeftSquare | TokKind::HookLeftSquare => open.push(TokKind::RightSquare),
                TokKind::LeftBrace => open.push(TokKind::RightBrace),
                kind if kind.is_closer() => {
                    open.pop();
                }
                _ => {}
            }
            if open.is_empty() {
                break;
            }
        }
        self.last_error = Some(self.next);
    }

    /// Creates the error for a token that doesn't belong where it is.
    fn unexpected_token(&self) -> UnexpectedToken {
        UnexpectedToken {
            src: self.src(),
            at: self.next_location().into(),
            tok: self.peek().to_string(),
        }
    }

    /// Records an error about the next token, or the end of input if that's where we are.
    fn error_here(&mut self, err: impl Into<shared::Error>) {
        if self.at(TokKind::Eof) {
            self.error(UnexpectedEnd {
                src: self.src(),
                at: self.location(self.cursor()).into(),
            });
        } else {
            self.error(err);
        }
    }

    /// Records an error, unless one was already reported at this token.
    fn error(&mut self, err: impl Into<shared::Error>) {
        let explained = self.cut_short && self.at(TokKind::Eof);
        if self.last_error != Some(self.next) && !explained {
            self.errors.push(err.into());
        }
        self.last_error = Some(self.next);
    }
}

// Patterns
impl<'s> Parser<'s> {
    fn pattern(&mut self) -> Pat {
        fn inner(parser: &mut Parser) -> Pat {
            let first = parser.match_pat_atom();
            if !parser.at(TokKind::Pipe) {
                return first;
            }
            let location = first.location();
            let mut alts = vec![first];
            while parser.eat(TokKind::Pipe) {
                alts.push(parser.match_pat_atom());
            }
            Pat::new(PatKind::Or(alts), location)
        }
        let start = self.next_start();
        if !self.peek().starts_pattern() {
            self.expected("pattern");
            return self.new_pat(PatKind::Poison(Poison), start);
        }
        self.nested(inner)
            .map(|pat| desugar_prefix_pat(&pat, ""))
            .unwrap_or_else(|| self.new_pat(PatKind::Poison(Poison), start))
    }

    fn match_pat_atom(&mut self) -> Pat {
        let start = self.next_start();
        let base = self.match_pat_base();
        if self.eat(TokKind::Hook) {
            return self.new_pat(PatKind::NullBind(Box::new(base)), start);
        }
        base
    }

    fn match_pat_base(&mut self) -> Pat {
        let start = self.next_start();
        let kind = self.peek();
        match kind {
            TokKind::LeftParenthesis => {
                self.bump(TokKind::LeftParenthesis);
                let members = self.list(
                    TokKind::RightParenthesis,
                    TokKind::starts_pattern,
                    Self::pattern,
                );
                self.new_pat(PatKind::Tuple(members), start)
            }
            TokKind::Minus => {
                self.bump(TokKind::Minus);
                let literal = match self.peek() {
                    TokKind::Int(i) => Literal::Int(-i),
                    TokKind::Float(f) => Literal::Float(-f),
                    _ => {
                        self.expected("number");
                        return self.new_pat(PatKind::Poison(Poison), start);
                    }
                };
                self.advance();
                self.new_pat(PatKind::Literal(literal), start)
            }
            TokKind::Ident(_) => {
                // walk path: ident (:: ident)*
                let first = self.require_ident();
                let mut head_expr = self.new_expr(first.clone(), start);
                let mut is_path = false;
                while self.eat(TokKind::DoubleColon) {
                    let right = self.require_member_name();
                    let left = head_expr;
                    head_expr = self.new_expr(Access::DoubleColon { left, right }, start);
                    is_path = true;
                }

                match self.peek() {
                    // `Foo(...)` tuple-variant (`Name_(…)` becomes prefix
                    // sugar in `pattern()`'s desugar pass)
                    TokKind::LeftParenthesis => {
                        self.bump(TokKind::LeftParenthesis);
                        let pats = self.list(
                            TokKind::RightParenthesis,
                            TokKind::starts_pattern,
                            Self::pattern,
                        );
                        self.new_pat(PatKind::TupleVariant(Box::new(head_expr), pats), start)
                    }
                    // `Foo { ... }` struct
                    TokKind::LeftBrace => {
                        self.bump(TokKind::LeftBrace);
                        // a trailing `..` says the rest of the fields are
                        // deliberately unnamed -- a struct pattern already
                        // matches any field it doesn't mention, so it binds
                        // nothing; it just reads as the omission it is
                        let starts = |k: TokKind<'s>| k.is_ident() || k == TokKind::DoubleDot;
                        let fields = self.list(TokKind::RightBrace, starts, |p| {
                            if p.eat(TokKind::DoubleDot) {
                                return None;
                            }
                            let name = p.require_ident();
                            let sub = if p.eat(TokKind::Equal) {
                                p.pattern()
                            } else {
                                // r[impl mimas.struct.shorthand]
                                // shorthand: `Foo { x }` is `Foo { x = x }`
                                Pat::from(p.hygiene_bind(name.clone()))
                            };
                            Some((name.lexeme, sub))
                        });
                        let fields = fields.into_iter().flatten().collect();
                        self.new_pat(PatKind::Struct(Box::new(head_expr), fields), start)
                    }
                    // bare path -- `Foo::Bar` with no payload
                    _ if is_path => self.new_pat(PatKind::Variant(Box::new(head_expr)), start),
                    // bare identifier -- binding/wildcard
                    _ => {
                        let first = self.hygiene_bind(first);
                        self.new_pat(PatKind::Ident(first), start)
                    }
                }
            }
            _ => match Literal::try_from(kind) {
                Ok(literal) => {
                    self.advance();
                    self.new_pat(PatKind::Literal(literal), start)
                }
                Err(()) => {
                    self.expected("pattern");
                    self.new_pat(PatKind::Poison(Poison), start)
                }
            },
        }
    }
}

// Lexing tools
impl<'s> Parser<'s> {
    /// Returns the next tok as an Identifier, or poison (after reporting) if it isn't one.
    fn require_ident(&mut self) -> Ident {
        let location = self.next_location();
        if let TokKind::Ident(lexeme) = self.peek() {
            let mark = self.meta_at(0).mark;
            self.advance();
            let ident = Ident::new(lexeme, location);
            if mark != 0 {
                self.hygiene.marks.insert(ident.id, mark);
            }
            return ident;
        }
        self.expected("identifier");
        // `Ident` is a plain struct with nowhere to put a variant, so its poison is a lexeme no
        // real identifier can have
        Ident::new(POISON, self.location(self.next_start()))
    }

    // member-access position (right of `::`): also accepts type keywords as plain names so
    // namespaces like `std::random::int` work without colliding with the type system.
    fn require_member_name(&mut self) -> Ident {
        let location = self.next_location();
        if let TokKind::TyKw(kw) = self.peek() {
            self.advance();
            return Ident::new(kw.to_string(), location);
        }
        self.require_ident()
    }

    /// A grammar-mandated token the user must supply next (separator/opener/closer). If it's
    /// absent the span sits at the end of the last consumed token, where it belongs -- not on the
    /// stray one. Nothing is consumed when it's missing.
    fn expect(&mut self, kind: TokKind<'s>) -> bool {
        let found = self.eat(kind);
        if !found {
            self.error_here(ExpectedToken {
                src: self.src(),
                at: self.location(self.cursor()).into(),
                expected: kind.to_string(),
            });
        }
        found
    }

    /// Takes the next token if it's a `kind`, and returns whether it was.
    fn eat(&mut self, kind: TokKind<'s>) -> bool {
        let found = self.at(kind);
        if found {
            self.advance();
        }
        found
    }

    /// The `>>` ahead becomes two `>` tokens (same location: a `>` only ever
    /// errors as "expected `>`", never by its own span).
    fn split_double_caret(&mut self) {
        debug_assert!(self.at(TokKind::DoubleRightCaret));
        let location = self.tokens[self.next].location;
        self.tokens[self.next] = Tok::new(TokKind::Greater, location);
        self.tokens
            .insert(self.next, Tok::new(TokKind::Greater, location));
        let first = self.meta[self.next];
        self.meta.insert(self.next, first);
        self.meta[self.next + 1] = TokMeta {
            glued: true,
            newline: false,
            ..first
        };
        // Everything after the split sits one token later. A doc comment
        // is kept by the index of the token it hangs on (`docs`), and so
        // is the last error: both move with their tokens, as a macro's
        // splice moves them.
        let at = self.next;
        self.docs = std::mem::take(&mut self.docs)
            .into_iter()
            .map(|(i, doc)| (if i > at { i + 1 } else { i }, doc))
            .collect();
        if let Some(e) = self.last_error
            && e > at
        {
            self.last_error = Some(e + 1);
        }
    }

    /// Takes a token the grammar already dispatched on.
    fn bump(&mut self, kind: TokKind<'s>) {
        debug_assert!(self.at(kind), "dispatched on {kind}");
        self.advance();
    }

    /// Takes the next token. The `Eof` stays put.
    fn advance(&mut self) -> Tok<TokKind<'s>> {
        let tok = self.tokens[self.next];
        if tok.kind() != TokKind::Eof {
            self.next += 1;
            self.fuel.set(FUEL);
        }
        tok
    }

    /// Returns if the next token is a `kind`.
    fn at(&self, kind: TokKind<'s>) -> bool {
        self.peek() == kind
    }

    /// Kind of the next token.
    fn peek(&self) -> TokKind<'s> {
        self.nth(0)
    }

    /// Kind of the token `n` past the next one.
    fn nth(&self, n: usize) -> TokKind<'s> {
        let fuel = self.fuel.get();
        assert!(
            fuel > 0,
            "the parser is stuck at byte {}",
            self.next_start()
        );
        self.fuel.set(fuel - 1);
        let last = self.tokens.len() - 1;
        self.tokens[(self.next + n).min(last)].kind()
    }

    /// Location of the next token.
    fn next_location(&self) -> Location {
        self.tokens[self.next].location().into()
    }

    /// Start byte of the next token.
    fn next_start(&self) -> usize {
        self.tokens[self.next].span().start()
    }

    /// Whether the next token is the first on its line.
    fn at_line_start(&self) -> bool {
        self.meta_at(0).newline
    }

    /// Whether the next token touches the last one taken, as `x!` and `50%` do.
    fn glued(&self) -> bool {
        self.meta_at(0).glued
    }

    /// The [`TokMeta`] of the token `n` past the next one.
    fn meta_at(&self, n: usize) -> TokMeta {
        self.meta[(self.next + n).min(self.meta.len() - 1)]
    }

    /// `#1.10`: a `#` glued to a number opens an exact literal, not an
    /// attribute (r[impl mimas.dec.exact-literal]).
    fn at_exact_literal(&self) -> bool {
        self.peek() == TokKind::Hash
            && matches!(self.nth(1), TokKind::Int(_) | TokKind::Float(_) | TokKind::Quantity(..))
            && self.next + 1 < self.tokens.len()
            && self.meta_at(1).glued
    }

    /// In a comma-optional sequence, a `-` glued to a numeric literal
    /// starts the next element: `[0 -1]` is `[0, -1]`. A `-` spaced on
    /// the right (`a - 1`) or glued on the left (`a -x`, `a-1`) keeps
    /// the infix read.
    fn glued_minus(&self) -> bool {
        if self.peek() != TokKind::Minus {
            return false;
        }
        let glued_right = self.next + 1 < self.tokens.len() && self.meta_at(1).glued;
        glued_right
            && matches!(
                self.nth(1),
                TokKind::Int(_) | TokKind::Float(_) | TokKind::Quantity(..)
            )
            && !self.glued()
    }

    /// End byte of the last token taken.
    fn cursor(&self) -> usize {
        match self.next.checked_sub(1) {
            Some(last) => self.tokens[last].span().end(),
            None => self.next_start(),
        }
    }
}

/// How many times the parser can look at a token before it has to take it. Nearly everything
/// stays in the dozens. The exception is backing out of the depth guard, where each of the 100
/// levels looks at whatever stopped it about a dozen times on its way out.
const FUEL: u32 = 4096;

/// Operators and keywords from other languages, with what to say about them.
fn foreign_spelling(name: &str) -> Option<(&'static str, &'static str)> {
    match name {
        "elif" => Some(("unknown keyword `elif`", "mimas spells this `else if`")),
        "as" => Some((
            "mimas has no `as` casts",
            "convert with a method instead, e.g. `.to_float()`",
        )),
        _ => None,
    }
}

/// Whether a name can follow a number — what `scan_suffix` will ever read.
/// `foo-bar` lexes `foo`, `bar`, and `-` separately, so it can never be a
/// suffix; ASCII letters, digits, `_`, starting with a letter.
fn valid_unit_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The token scan that registers `unit` decls before the real lex — the
/// lexer needs custom names in the table to read `18tile` as a quantity.
/// Walks the probe tokens, finds `unit <name> [= spec]` at statement
/// position, evaluates the spec, and declares. Returns whether the table
/// gained a new name — that's what calls for a relex.
fn declare_units(source: &str, tokens: &[Tok<TokKind<'_>>]) -> bool {
    let toks: Vec<&Tok<TokKind>> = tokens.iter().filter(|t| !t.kind().is_comment()).collect();
    // (name, spec tokens) at each top-level `unit` decl site; `None` for a
    // bare `unit name;`.
    let mut sites: Vec<(&str, Option<Vec<TokKind>>)> = vec![];
    let mut fresh_axes = false;
    let mut i = 0;
    let mut depth = 0usize;
    while i + 1 < toks.len() {
        match toks[i].kind() {
            TokKind::LeftBrace
            | TokKind::LeftParenthesis
            | TokKind::LeftSquare
            | TokKind::HookLeftSquare => depth += 1,
            TokKind::RightBrace | TokKind::RightParenthesis | TokKind::RightSquare => {
                depth = depth.saturating_sub(1)
            }
            _ => {}
        }
        let stmt_pos = i == 0
            || matches!(toks[i - 1].kind(), TokKind::SemiColon)
            || source[toks[i - 1].span().end()..toks[i].span().start()].contains('\n');
        // `axis NAME (NEG, POS)` registers its words here too, so a direction
        // written above its declaration still parses as one
        if depth == 0
            && stmt_pos
            && matches!(toks[i].kind(), TokKind::Ident("axis"))
            && i + 6 < toks.len()
            && let (
                TokKind::Ident(name),
                TokKind::LeftParenthesis,
                TokKind::Ident(neg),
                TokKind::Comma,
                TokKind::Ident(pos),
                TokKind::RightParenthesis,
            ) = (
                toks[i + 1].kind(),
                toks[i + 2].kind(),
                toks[i + 3].kind(),
                toks[i + 4].kind(),
                toks[i + 5].kind(),
                toks[i + 6].kind(),
            )
            && !source[toks[i].span().end()..toks[i + 1].span().start()].contains('\n')
        {
            fresh_axes |= shared::units::axis_lookup(name).is_none();
            let _ = shared::units::declare_axis(name, neg, pos);
        }
        let is_decl = depth == 0
            && stmt_pos
            && matches!(toks[i].kind(), TokKind::Ident("unit"))
            && matches!(toks[i + 1].kind(), TokKind::Ident(_))
            && !source[toks[i].span().end()..toks[i + 1].span().start()].contains('\n');
        if !is_decl {
            i += 1;
            continue;
        }
        let TokKind::Ident(name) = toks[i + 1].kind() else {
            unreachable!()
        };
        i += 2;
        let mut spec = None;
        if i < toks.len()
            && matches!(toks[i].kind(), TokKind::Equal)
            && !source[toks[i - 1].span().end()..toks[i].span().start()].contains('\n')
        {
            i += 1;
            let mut rhs = vec![];
            while i < toks.len()
                && toks[i].kind().is_unit_spec()
                && !source[toks[i - 1].span().end()..toks[i].span().start()].contains('\n')
            {
                rhs.push(toks[i].kind());
                i += 1;
            }
            spec = Some(rhs);
        }
        if valid_unit_name(name) {
            sites.push((name, spec));
        }
    }
    // Base decls first so a spec can name a `unit` declared after it
    // (`unit speed = 1tile/s; unit tile;` resolves); spec'd decls retry to
    // a fixpoint so `unit a = 2tile; unit b = 2a;` works in either order.
    // Whatever never evaluates stays undeclared — the stmt parser reports
    // the bad spec.
    let mut fresh = false;
    let mut pending = vec![];
    for (name, spec) in sites {
        match spec {
            None => {
                fresh |= shared::units::lookup(name).is_none();
                let _ = shared::units::declare(name, None);
            }
            Some(rhs) => pending.push((name, rhs)),
        }
    }
    while !pending.is_empty() {
        let mut progress = false;
        pending.retain(|(name, rhs)| match unit_spec(rhs) {
            Some(spec) => {
                fresh |= shared::units::lookup(name).is_none();
                let _ = shared::units::declare(name, Some(spec));
                progress = true;
                false
            }
            None => true,
        });
        if !progress {
            break;
        }
    }
    fresh || fresh_axes
}

/// One factor of a unit spec: a unit name (bare ident or a quantity
/// token's suffix) contributes its dim and scale; a bare number is a
/// dimensionless scale.
fn unit_factor(tok: TokKind<'_>, num: &mut f64, dim: &mut Dim) -> Option<()> {
    match tok {
        TokKind::Quantity(v, u) => {
            // the suffix is a full unit expression — `1tile/s` lexes as
            // one token whose unit text is "tile/s"
            let (d, s) = shared::units::parse(u)?;
            *num *= v * s;
            *dim = dim.mul(d);
        }
        TokKind::Ident(u) => {
            let (d, s) = shared::units::lookup(u)?;
            *num *= s;
            *dim = dim.mul(d);
        }
        TokKind::Int(v) => *num *= v as f64,
        TokKind::Float(v) => *num *= v,
        _ => return None,
    }
    Some(())
}

/// Evaluates a `unit` decl's spec tokens into (dim, scale): a leading
/// number, then unit-name factors under `*`, `/`, and `^n`. `None` when
/// the tokens aren't a quantity — the stmt reports the diagnostic.
fn unit_spec(toks: &[TokKind<'_>]) -> Option<(Dim, f64)> {
    let mut num = match toks.first().copied()? {
        TokKind::Int(v) => v as f64,
        TokKind::Float(v) => v,
        TokKind::Quantity(v, u) => {
            let (d, s) = shared::units::parse(u)?;
            let mut num = v * s;
            let mut dim = d;
            spec_tail(&toks[1..], &mut num, &mut dim)?;
            return Some((dim, num));
        }
        _ => return None,
    };
    let mut dim = Dim::NONE;
    spec_tail(&toks[1..], &mut num, &mut dim)?;
    Some((dim, num))
}

/// The `op factor` chain after the leading number of a unit spec.
fn spec_tail(toks: &[TokKind<'_>], num: &mut f64, dim: &mut Dim) -> Option<()> {
    let mut i = 0;
    let mut divide = false;
    while i < toks.len() {
        match toks[i] {
            TokKind::Star => {
                divide = false;
                i += 1;
            }
            TokKind::Slash => {
                divide = true;
                i += 1;
            }
            TokKind::Minus | TokKind::Caret => return None,
            _ => {
                let mut f_num = 1.0;
                let mut f_dim = Dim::NONE;
                unit_factor(toks[i], &mut f_num, &mut f_dim)?;
                i += 1;
                if matches!(toks.get(i), Some(TokKind::Caret)) {
                    i += 1;
                    let neg = matches!(toks.get(i), Some(TokKind::Minus));
                    if neg {
                        i += 1;
                    }
                    let Some(TokKind::Int(v)) = toks.get(i).copied() else {
                        return None;
                    };
                    let exp = v.clamp(i8::MIN.into(), i8::MAX.into()) as i8;
                    let exp = if neg { exp.wrapping_neg() } else { exp };
                    f_dim = f_dim.powi(exp);
                    f_num = f_num.powi(exp.into());
                    i += 1;
                }
                if divide {
                    *num /= f_num;
                    *dim = dim.div(f_dim);
                } else {
                    *num *= f_num;
                    *dim = dim.mul(f_dim);
                }
                divide = false;
            }
        }
    }
    Some(())
}

/// What `node()` parsed: either a fully-formed stmt, or a bare expression whose role
/// (block-trailing yield vs expression statement) is decided by the caller. Assignments
/// commit to `Stmt` inside `node()` so they never surface as `MaybeYield`.
enum BlockElement {
    Stmt(Stmt),
    MaybeYield(Expr),
}

/// A binary operator, paired with how tightly it binds (loosest is 1).
enum BinaryOp {
    Logical(LogicalOp),
    Equality(EqualityOp),
    /// `in` when true, `!in` when false.
    In(bool),
    Eval(EvaluationOp),
    /// `±` — binds like `+`/`-`; desugars to an `Interval::` constructor call.
    PlusMinus,
}

impl BinaryOp {
    /// Returns the operator `kind` is, if it is one.
    fn of(kind: TokKind) -> Option<(Self, u8)> {
        if kind == TokKind::PlusMinus {
            return Some((Self::PlusMinus, 6));
        }
        if let Ok(op) = LogicalOp::try_from(kind) {
            return Some((Self::Logical(op), 1));
        }
        if let Ok(op) = EqualityOp::try_from(kind) {
            return Some((Self::Equality(op), 2));
        }
        // `in` doesn't chain — `binary` caps `max_power` after taking one
        if matches!(kind, TokKind::In | TokKind::NotIn) {
            return Some((Self::In(kind == TokKind::In), 3));
        }
        // `a xor b` — the word spellings of `⊕`/`⍲`/`⍱`, contextual like
        // `mod`: ordinary names everywhere else.
        let word_op = match kind {
            TokKind::Ident("xor") => Some(EvaluationOp::Xor),
            TokKind::Ident("nand") => Some(EvaluationOp::Nand),
            TokKind::Ident("nor") => Some(EvaluationOp::Nor),
            _ => None,
        };
        if let Some(op) = word_op {
            return Some((Self::Eval(op), 4));
        }
        let op = EvaluationOp::try_from(kind).ok()?;
        // `a ≈ b` compares, so it binds like `==` (r[impl mimas.op.approx])
        if op == EvaluationOp::Approx {
            return Some((Self::Eval(op), 2));
        }
        let power = if op.is_binary() {
            4
        } else if op.is_bit_shift() {
            5
        } else if op.is_additive() || op.is_named_additive() {
            6
        } else {
            // the multiplicative ops (and the named ones that bind like
            // `*`: `∧ ⋅ × ∘ ⊠ ∩`), which bind tightest
            7
        };
        Some((Self::Eval(op), power))
    }
}

/// The method a prefix reduction glyph names, if `kind` is one.
fn reduce_method(kind: TokKind<'_>) -> Option<&'static str> {
    Some(match kind {
        TokKind::Sum => "sum",
        TokKind::Prod => "product",
        TokKind::Inf => "inf",
        TokKind::Sup => "sup",
        _ => return None,
    })
}

/// Makes a test name unique within its list by appending ` #n`.
fn unique_test_name(seen: &mut std::collections::HashSet<String>, name: String) -> String {
    if seen.insert(name.clone()) {
        return name;
    }
    let mut n = 2;
    loop {
        let candidate = format!("{name} #{n}");
        if seen.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}

/// A signed number literal — `2`, `0.5`, `-3` — as `caret_pow`'s operand.
#[derive(Clone, Copy)]
enum SignedLit {
    Int(i64),
    Float(f64),
}
impl SignedLit {
    fn as_f64(&self) -> f64 {
        match self {
            SignedLit::Int(n) => *n as f64,
            SignedLit::Float(n) => *n,
        }
    }
}

/// Reads `2`, `0.5`, `-3` (and `+3`, parenthesized forms) as a number;
/// anything else is `None`.
fn num_lit(e: &Expr) -> Option<SignedLit> {
    match e.kind() {
        ExprKind::Literal(Literal::Int(n)) => Some(SignedLit::Int(*n)),
        ExprKind::Literal(Literal::Float(n)) => Some(SignedLit::Float(*n)),
        ExprKind::Unary(Unary {
            op: UnaryOp::Negative,
            right,
        }) => match num_lit(right)? {
            SignedLit::Int(n) => n.checked_neg().map(SignedLit::Int),
            SignedLit::Float(n) => Some(SignedLit::Float(-n)),
        },
        ExprKind::Unary(Unary {
            op: UnaryOp::Positive,
            right,
        }) => num_lit(right),
        ExprKind::Grouping(g) => num_lit(&g.inner),
        _ => None,
    }
}

/// `lit ^ lit` folds to the literal's power at desugar — `2^10` is `1024`.
/// A bare signed base keeps the math reading — `-2^2` is `-(2^2)`, while
/// `(-2)^2` squares the negative. Ints stay ints when the exponent is
/// non-negative and the value fits; a float side, negative exponent, or
/// overflow falls back to float.
fn caret_pow(left: &Expr, right: &Expr) -> Option<Literal> {
    let (negated, base) = match left.kind() {
        ExprKind::Unary(Unary {
            op: UnaryOp::Negative,
            right,
        }) => (true, num_lit(right)?),
        ExprKind::Unary(Unary {
            op: UnaryOp::Positive,
            right,
        }) => (false, num_lit(right)?),
        _ => (false, num_lit(left)?),
    };
    let exp = num_lit(right)?;
    let sign = if negated { -1.0 } else { 1.0 };
    Some(
        if let (SignedLit::Int(b), SignedLit::Int(e)) = (base, exp) {
            match u32::try_from(e).ok().and_then(|e| b.checked_pow(e)) {
                // a `checked_pow` result can't be `i64::MIN`, so `-n` can't overflow
                Some(n) => Literal::Int(if negated { -n } else { n }),
                None => Literal::Float((b as f64).powf(e as f64) * sign),
            }
        } else {
            Literal::Float(base.as_f64().powf(exp.as_f64()) * sign)
        },
    )
}

/// `FIELD_(X, Y, W, H)` desugars to `(FIELD_X, FIELD_Y, FIELD_W, FIELD_H)`: a
/// `_`-tailed `name(..)` pattern binds `name`-prefixed leaves, and nested
/// `name_(..)` nodes compose (`A_(X, B_(Y))` binds `A_X` and `A_B_Y`). Applied
/// at `pattern()`'s exit, so every pattern position gets it -- `const`,
/// `let`, `match`, `for`, `if let`/`while let`. (In expression position `f_(x)`
/// is a call on a `_`-tailed name, which `call()` rejects as a mistake.)
fn desugar_prefix_pat(pat: &Pat, prefix: &str) -> Pat {
    let kind = match pat.kind() {
        PatKind::Ident(id) if !prefix.is_empty() && id.lexeme != "_" => {
            let mut id = id.clone();
            id.lexeme = format!("{prefix}{id}");
            PatKind::Ident(id)
        }
        PatKind::Tuple(pats) => {
            PatKind::Tuple(pats.iter().map(|p| desugar_prefix_pat(p, prefix)).collect())
        }
        PatKind::TupleVariant(head, pats) => {
            // only `Name_(…)` is sugar -- a real path head (`A::B(..)`) or a
            // plain `Name(..)` variant pattern is left for the solver to reject
            let ExprKind::Ident(head) = head.kind() else {
                return pat.clone();
            };
            if head.lexeme.len() <= 1 || !head.lexeme.ends_with('_') {
                return pat.clone();
            }
            let prefix = format!("{prefix}{head}");
            PatKind::Tuple(
                pats.iter()
                    .map(|p| desugar_prefix_pat(p, &prefix))
                    .collect(),
            )
        }
        PatKind::NullBind(inner) => PatKind::NullBind(Box::new(desugar_prefix_pat(inner, prefix))),
        PatKind::Or(pats) => {
            PatKind::Or(pats.iter().map(|p| desugar_prefix_pat(p, prefix)).collect())
        }
        PatKind::Struct(head, fields) => PatKind::Struct(
            head.clone(),
            fields
                .iter()
                .map(|(k, v)| (k.clone(), desugar_prefix_pat(v, prefix)))
                .collect(),
        ),
        _ => return pat.clone(),
    };
    Pat::new(kind, pat.location())
}

/// The first `return`/`raise`/`?`/`break`/`continue` in a `with` region's
/// body that would leave the region without running the desugared restore.
/// These are structured exits — the frame returns early or the loop jumps;
/// neither unwinds the Vm, so the `__fx_*_pop` at the region's tail never
/// runs and the installed state leaks to the caller. Loops and closures
/// inside the body bind their own exits, so only exits aimed *past* the
/// `with` count. (A hard fault is different: it unwinds, and the fixture's
/// `reconcile` heals the scope at the truncation boundary.)
// r[impl mimas.fx.with.exits]
fn with_region_escape(body: &Expr) -> Option<(Location, &'static str)> {
    let ExprKind::Block(block) = body.kind() else {
        return None;
    };
    block
        .body
        .iter()
        .find_map(|stmt| with_stmt_escape(stmt, false, false))
        .or_else(|| {
            block
                .yielded_expr
                .as_ref()
                .and_then(|expr| with_expr_escape(expr, false, false))
        })
}

fn with_stmt_escape(stmt: &Stmt, in_loop: bool, in_fn: bool) -> Option<(Location, &'static str)> {
    match stmt.kind() {
        StmtKind::Let(l) => with_pat_escape(&l.left, in_loop, in_fn)
            .or_else(|| with_expr_escape(&l.right, in_loop, in_fn))
            .or_else(|| {
                l.else_branch
                    .as_ref()
                    .and_then(|e| with_expr_escape(e, in_loop, in_fn))
            }),
        StmtKind::Assignment(a) => with_expr_escape(&a.left, in_loop, in_fn)
            .or_else(|| with_expr_escape(&a.right, in_loop, in_fn)),
        StmtKind::Expr(e) => with_expr_escape(e, in_loop, in_fn),
        // a nested item (`fn`, `check`, `const`, …) binds its own exits
        StmtKind::Item(_) | StmtKind::Module(_) | StmtKind::Unit(_) | StmtKind::Needs(_) => None,
    }
}

fn with_expr_escape(expr: &Expr, in_loop: bool, in_fn: bool) -> Option<(Location, &'static str)> {
    let scan = |e: &Expr| with_expr_escape(e, in_loop, in_fn);
    let scan_opt = |e: &Option<Expr>| e.as_ref().and_then(|e| scan(e));
    match expr.kind() {
        // exits aimed past the region — the desugared pop never runs
        ExprKind::Call(c) if !in_fn && matches!(c.left.kind(),
            ExprKind::Ident(i) if i.lexeme == "__generator_yield") => Some((expr.location(), "yield")),
        ExprKind::Return(_) if !in_fn => Some((expr.location(), "return")),
        ExprKind::Raise(_) if !in_fn => Some((expr.location(), "raise")),
        ExprKind::Propagate(_) if !in_fn => Some((expr.location(), "`?`")),
        ExprKind::Break(_) if !(in_loop || in_fn) => Some((expr.location(), "break")),
        ExprKind::Continue(_) if !(in_loop || in_fn) => Some((expr.location(), "continue")),
        // exits that stay inside — bound, but their payloads still get scanned
        ExprKind::Return(r) => scan_opt(&r.value),
        ExprKind::Raise(r) => scan(&r.value),
        ExprKind::Propagate(p) => scan(&p.expr),
        ExprKind::Break(b) => scan_opt(&b.value),
        ExprKind::Continue(_) | ExprKind::Ident(_) | ExprKind::Poison(_) => None,
        // a closure binds its own exits — a `return`/`?` inside one leaves
        // the closure, not this region; parameter defaults likewise
        ExprKind::Closure(c) => c
            .parameters
            .iter()
            .filter_map(|b| b.right.as_ref())
            .find_map(|e| with_expr_escape(e, false, true))
            .or_else(|| with_expr_escape(&c.body, false, true)),
        ExprKind::For(f) => with_pat_escape(&f.binding, in_loop, in_fn)
            .or_else(|| scan(&f.iterator))
            .or_else(|| with_expr_escape(&f.body, true, in_fn)),
        ExprKind::While(w) => w
            .binding
            .as_ref()
            .and_then(|b| with_pat_escape(b, in_loop, in_fn))
            .or_else(|| scan(&w.header))
            .or_else(|| with_expr_escape(&w.body, true, in_fn)),
        ExprKind::Loop(l) => with_expr_escape(&l.body, true, in_fn),
        ExprKind::Collect(c) => scan(&c.value),
        ExprKind::Block(b) => b
            .body
            .iter()
            .find_map(|s| with_stmt_escape(s, in_loop, in_fn))
            .or_else(|| scan_opt(&b.yielded_expr)),
        ExprKind::If(i) => i
            .binding
            .as_ref()
            .and_then(|b| with_pat_escape(b, in_loop, in_fn))
            .or_else(|| scan(&i.condition))
            .or_else(|| scan(&i.main_body))
            .or_else(|| scan_opt(&i.else_expr)),
        ExprKind::Match(m) => scan(&m.identity).or_else(|| {
            m.cases.iter().find_map(|case| {
                with_pat_escape(case.pat(), in_loop, in_fn)
                    .or_else(|| case.guard().and_then(|g| scan(g)))
                    .or_else(|| scan(case.body()))
            })
        }),
        ExprKind::Absolve(a) => scan(&a.left).or_else(|| scan(&a.handler)),
        ExprKind::Call(c) => scan(&c.left).or_else(|| {
            c.arguments.iter().find_map(|a| scan(&a.value))
        }),
        ExprKind::Access(a) => match a {
            Access::Dot { left, right, .. } => scan(left).or_else(|| scan(right)),
            Access::DoubleColon { left, .. } => scan(left),
            Access::Square { left, key, .. } => scan(left).or_else(|| scan(key)),
            Access::Identity { .. } => None,
        },
        ExprKind::Equality(e) => scan(&e.left).or_else(|| scan(&e.right)),
        ExprKind::Evaluation(e) => scan(&e.left).or_else(|| scan(&e.right)),
        ExprKind::Logical(e) => scan(&e.left).or_else(|| scan(&e.right)),
        ExprKind::Coalescence(e) => scan(&e.left).or_else(|| scan(&e.right)),
        ExprKind::In(e) => scan(&e.left).or_else(|| scan(&e.right)),
        ExprKind::FString(f) => f.parts.iter().find_map(|p| match p {
            FStringPart::Expr(e) => scan(e),
            _ => None,
        }),
        ExprKind::Grouping(g) => scan(&g.inner),
        ExprKind::Literal(l) => match l {
            Literal::Array(exprs) | Literal::Tuple(exprs) => {
                exprs.iter().find_map(|e| scan(e))
            }
            Literal::Record(fields) => fields.iter().find_map(|(_, v)| scan(v)),
            Literal::Struct(s) => scan(&s.name)
                .or_else(|| s.fields.iter().find_map(|(_, v)| scan(v)))
                .or_else(|| scan_opt(&s.base)),
            _ => None,
        },
        ExprKind::Range(r) => scan(&r.start).or_else(|| scan(&r.end)),
        ExprKind::Unary(u) => scan(&u.right),
        ExprKind::Unwrap(u) => scan(&u.expr),
    }
}

fn with_pat_escape(pat: &Pat, in_loop: bool, in_fn: bool) -> Option<(Location, &'static str)> {
    match pat.kind() {
        PatKind::Tuple(pats) | PatKind::Or(pats) => pats
            .iter()
            .find_map(|p| with_pat_escape(p, in_loop, in_fn)),
        PatKind::Struct(head, fields) => {
            with_expr_escape(head, in_loop, in_fn).or_else(|| {
                fields
                    .values()
                    .find_map(|p| with_pat_escape(p, in_loop, in_fn))
            })
        }
        PatKind::TupleVariant(head, pats) => {
            with_expr_escape(head, in_loop, in_fn).or_else(|| {
                pats.iter()
                    .find_map(|p| with_pat_escape(p, in_loop, in_fn))
            })
        }
        PatKind::Variant(head) => with_expr_escape(head, in_loop, in_fn),
        PatKind::NullBind(inner) => with_pat_escape(inner, in_loop, in_fn),
        PatKind::Ident(_) | PatKind::Literal(_) | PatKind::Poison(_) => None,
    }
}
