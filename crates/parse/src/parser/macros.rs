//! Macros. `macro name(pattern) { template }` declares one, and `name!(…)`
//! expands it before anything is checked.
//!
//! The design is Rust's `macro_rules!`, in Mimas words. A pattern matches the
//! call's tokens. `$x:expr` takes one expression, `$x:ident` one name, and
//! `$( … ),*` repeats. The template is tokens too, with the same `$x` and
//! `$( … )` forms. Expansion writes those tokens into the parser's stream in
//! place of the call. The ordinary grammar then reads them, so a macro can
//! expand to an expression, statements or items.
//!
//! Hygiene follows Rust's rule for locals. A name the template binds (a
//! `let`, a parameter, a pattern) is renamed per expansion to `name·N`. No
//! one can type that name, so it cannot capture a name the call site passed
//! in, and the call site cannot see it. Every other template name means what
//! it means where the macro is used: functions, types, fields and globals.
//!
//! The parser asks [`TokMeta`] whether tokens touch or a line breaks between
//! them. A spliced token keeps the meta it had in the template (or in the
//! call, for a fragment). Its span is the call's, so a later error points at
//! the macro use. A fragment keeps its own span inside the call.

use super::*;
use std::{
    cell::RefCell,
    collections::{HashMap as StdMap, HashSet},
    rc::Rc,
};

/// How deep one expansion may nest inside another before the parser gives
/// up. A macro that calls itself with the same input never stops by itself.
const MAX_DEPTH: u32 = 64;
/// How many expansions one source may make in all.
const MAX_EXPANSIONS: u32 = 20_000;

thread_local! {
    /// The macros declared so far, by name. Like the unit table, it is
    /// ambient: a fresh [`Parser::new`] clears it, and the files of one
    /// program, parsed with [`Parser::new_extending`], share it.
    static MACROS: RefCell<StdMap<String, Rc<MacroDef>>> = RefCell::default();
    /// Interned macro definition text. A macro outlives the source it was
    /// written in (a later file uses it), so its tokens borrow from here.
    static TEXTS: RefCell<HashSet<&'static str>> = RefCell::default();
}

/// Forgets every macro. A new program starts with none.
pub fn clear() {
    MACROS.with(|m| m.borrow_mut().clear());
}

/// The text, kept for the life of the thread. Each distinct text is kept once.
fn intern(text: &str) -> &'static str {
    TEXTS.with(|texts| {
        let mut texts = texts.borrow_mut();
        if let Some(&kept) = texts.get(text) {
            return kept;
        }
        let kept: &'static str = Box::leak(text.to_string().into_boxed_str());
        texts.insert(kept);
        kept
    })
}

fn lookup(name: &str) -> Option<Rc<MacroDef>> {
    MACROS.with(|m| m.borrow().get(name).cloned())
}

/// The name as the call site wrote it: a template's renamed local
/// `total·3` reads back as `total`. Source text can never hold a `·` in
/// a name (it lexes as the `⋅` operator).
pub(crate) fn unhygienic(ident: &Ident) -> Ident {
    let mut plain = ident.clone();
    if let Some((name, _)) = ident.lexeme.split_once('·') {
        plain.lexeme = name.to_string();
    }
    plain
}

/// The parser's macro state.
#[derive(Default)]
pub(crate) struct Hygiene {
    /// The expansion that wrote each identifier, for identifiers a template
    /// wrote. Read where a name binds or is used.
    pub(crate) marks: hashbrown::HashMap<NodeId, u32>,
    /// The template locals bound so far, as (name, expansion).
    bound: HashSet<(String, u32)>,
    /// How many expansions this parser made. The next one is this plus 1.
    expansions: u32,
    /// The nesting depth of each expansion, by mark - 1.
    depths: Vec<u32>,
    /// Above 0 while a fragment is parsed only to find where it ends. A macro
    /// call inside it is then skipped, not expanded.
    speculating: u32,
}

/// One declared macro.
struct MacroDef {
    arms: Vec<Arm>,
}

/// One `(pattern) => { template }` rule.
struct Arm {
    pattern: Vec<Matcher>,
    template: Vec<Trans>,
}

/// A template token and how it sits against the one before it.
#[derive(Clone, Copy)]
struct MTok {
    kind: TokKind<'static>,
    meta: TokMeta,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Frag {
    Expr,
    Ident,
    Ty,
    Pat,
    Block,
    Literal,
    Tt,
}

impl Frag {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "expr" => Self::Expr,
            "ident" => Self::Ident,
            "ty" => Self::Ty,
            "pat" => Self::Pat,
            "block" => Self::Block,
            "literal" => Self::Literal,
            "tt" => Self::Tt,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rep {
    /// `*`
    Any,
    /// `+`
    Some,
    /// `?`
    Maybe,
}

/// One piece of a pattern.
enum Matcher {
    /// This exact token.
    Tok(TokKind<'static>),
    /// `$name:kind`
    Frag(&'static str, Frag),
    /// `$( … ) sep? op`
    Repeat(Vec<Matcher>, Option<TokKind<'static>>, Rep),
}

/// One piece of a template.
enum Trans {
    Tok(MTok),
    /// `$name`, with the meta of its `$`.
    Var(&'static str, TokMeta),
    /// `$( … ) sep? op`. The operator only matters in the pattern.
    Repeat(Vec<Trans>, Option<MTok>),
}

/// What a fragment matched: the call's tokens, or one match per repetition.
enum Bind<'s> {
    One(Vec<(Tok<TokKind<'s>>, TokMeta)>, bool),
    Many(Vec<Bind<'s>>),
}

type Binds<'s> = StdMap<&'static str, Bind<'s>>;

/// A definition error: what, and the token it is about.
struct DefError {
    at: Location,
    msg: String,
    label: String,
}

/// Reads a definition's relexed tokens.
struct DefCursor<'a> {
    toks: &'a [Tok<TokKind<'static>>],
    meta: &'a [TokMeta],
    i: usize,
    end: usize,
}

impl DefCursor<'_> {
    fn peek(&self) -> TokKind<'static> {
        if self.i < self.end {
            self.toks[self.i].kind
        } else {
            TokKind::Eof
        }
    }

    fn peek_at(&self, n: usize) -> TokKind<'static> {
        if self.i + n < self.end {
            self.toks[self.i + n].kind
        } else {
            TokKind::Eof
        }
    }

    fn location(&self) -> Location {
        let i = self.i.min(self.toks.len() - 1);
        self.toks[i].location.into()
    }

    fn error(&self, msg: impl Into<String>, label: impl Into<String>) -> DefError {
        DefError {
            at: self.location(),
            msg: msg.into(),
            label: label.into(),
        }
    }

    /// The index of the closer that matches the opener at `open`.
    fn closer(&self, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        for i in open..self.end {
            match self.toks[i].kind {
                k if opens(k) => depth += 1,
                k if k.is_closer() => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// The `$( … )` group at the cursor (on its `(`): the inner range, then
    /// the separator and the operator after it.
    fn repetition(
        &mut self,
    ) -> std::result::Result<(usize, usize, Option<TokKind<'static>>, Rep), DefError> {
        let open = self.i;
        let Some(close) = self.closer(open) else {
            return Err(self.error("this `$(` never closes", "close it with `)`"));
        };
        self.i = close + 1;
        let rep = |k| match k {
            TokKind::Star => Some(Rep::Any),
            TokKind::Plus => Some(Rep::Some),
            TokKind::Hook => Some(Rep::Maybe),
            _ => None,
        };
        if let Some(op) = rep(self.peek()) {
            self.i += 1;
            return Ok((open + 1, close, None, op));
        }
        let sep = self.peek();
        if let Some(op) = rep(self.peek_at(1))
            && op != Rep::Maybe
            && !opens(sep)
            && !sep.is_closer()
            && sep != TokKind::Eof
        {
            self.i += 2;
            return Ok((open + 1, close, Some(sep), op));
        }
        Err(self.error(
            "a `$( … )` repetition needs `*`, `+` or `?` after it",
            "write `$( … ),*` for a list with commas",
        ))
    }

    fn pattern(&mut self, end: usize) -> std::result::Result<Vec<Matcher>, DefError> {
        let mut out = vec![];
        while self.i < end {
            let kind = self.peek();
            if kind != TokKind::Dollar {
                out.push(Matcher::Tok(kind));
                self.i += 1;
                continue;
            }
            match self.peek_at(1) {
                TokKind::Ident(name) => {
                    self.i += 2;
                    if self.peek() != TokKind::Colon {
                        return Err(self.error(
                            format!("`${name}` needs a kind in a pattern"),
                            format!("write `${name}:expr`, `${name}:ident` or another kind"),
                        ));
                    }
                    self.i += 1;
                    let kind = match self.peek() {
                        TokKind::Ident(k) => Frag::from_name(k),
                        _ => None,
                    };
                    let Some(kind) = kind else {
                        return Err(self.error(
                            "unknown fragment kind",
                            "use expr, ident, ty, pat, block, literal or tt",
                        ));
                    };
                    self.i += 1;
                    out.push(Matcher::Frag(name, kind));
                }
                TokKind::LeftParenthesis => {
                    self.i += 1;
                    let (from, to, sep, rep) = self.repetition()?;
                    let after = self.i;
                    self.i = from;
                    let inner = self.pattern(to)?;
                    self.i = after;
                    out.push(Matcher::Repeat(inner, sep, rep));
                }
                _ => {
                    return Err(self.error(
                        "a `$` in a pattern starts `$name:kind` or `$( … )`",
                        "name the fragment after the `$`",
                    ));
                }
            }
        }
        Ok(out)
    }

    fn template(&mut self, end: usize) -> std::result::Result<Vec<Trans>, DefError> {
        let mut out = vec![];
        while self.i < end {
            let kind = self.peek();
            let meta = self.meta[self.i];
            if kind != TokKind::Dollar {
                out.push(Trans::Tok(MTok { kind, meta }));
                self.i += 1;
                continue;
            }
            match self.peek_at(1) {
                TokKind::Ident(name) => {
                    self.i += 2;
                    out.push(Trans::Var(name, meta));
                }
                TokKind::LeftParenthesis => {
                    self.i += 1;
                    let (from, to, sep, _) = self.repetition()?;
                    let sep_meta = self.meta[self.i - 2];
                    let after = self.i;
                    self.i = from;
                    let inner = self.template(to)?;
                    self.i = after;
                    let sep = sep.map(|kind| MTok {
                        kind,
                        meta: sep_meta,
                    });
                    out.push(Trans::Repeat(inner, sep));
                }
                _ => {
                    return Err(self.error(
                        "a `$` in a template starts `$name` or `$( … )`",
                        "name a fragment of the pattern after the `$`",
                    ));
                }
            }
        }
        Ok(out)
    }
}

/// Whether the token opens a group a closer ends.
fn opens(kind: TokKind<'_>) -> bool {
    matches!(
        kind,
        TokKind::LeftParenthesis
            | TokKind::LeftSquare
            | TokKind::HookLeftSquare
            | TokKind::LeftBrace
            | TokKind::TildeLeftBrace
    )
}

/// Each fragment a pattern names, with how many repetitions deep it sits.
fn pattern_depths(pattern: &[Matcher], depth: usize, out: &mut StdMap<&'static str, usize>) {
    for m in pattern {
        match m {
            Matcher::Tok(_) => {}
            Matcher::Frag(name, _) => {
                out.insert(name, depth);
            }
            Matcher::Repeat(inner, ..) => pattern_depths(inner, depth + 1, out),
        }
    }
}

/// Checks that a template names only the pattern's fragments, and no fragment
/// at fewer repetitions than its pattern gives it. Each `$( … )` must hold a
/// fragment that repeats at its depth, or nothing says how often to repeat.
/// Returns the deepest pattern depth of any fragment the template names.
fn check_template(
    template: &[Trans],
    depths: &StdMap<&'static str, usize>,
    depth: usize,
) -> std::result::Result<usize, String> {
    let mut deepest = 0;
    for t in template {
        match t {
            Trans::Tok(_) => {}
            Trans::Var(name, _) => {
                let Some(&want) = depths.get(name) else {
                    return Err(format!("`${name}` is not in the pattern"));
                };
                if want > depth {
                    return Err(format!(
                        "`${name}` repeats in the pattern, so the template must use it inside `$( … )`"
                    ));
                }
                deepest = deepest.max(want);
            }
            Trans::Repeat(inner, ..) => {
                let inner = check_template(inner, depths, depth + 1)?;
                if inner <= depth {
                    return Err("this `$( … )` holds no fragment that repeats".into());
                }
                deepest = deepest.max(inner);
            }
        }
    }
    Ok(deepest)
}

/// The fragment names a pattern binds directly or in nested repetitions.
fn pattern_names(pattern: &[Matcher], out: &mut Vec<&'static str>) {
    for m in pattern {
        match m {
            Matcher::Tok(_) => {}
            Matcher::Frag(name, _) => out.push(name),
            Matcher::Repeat(inner, ..) => pattern_names(inner, out),
        }
    }
}

/// Whether an expression fragment must be put in parentheses to keep its
/// meaning once spliced: `$a * 2` with `1 + 1` must stay `(1 + 1) * 2`. A
/// single name, a literal, or a postfix chain (`a.b[0]`, `f(x)`) needs none,
/// and stays assignable. A struct literal always gets them, so that it still
/// reads as one inside `if` or `match`.
fn needs_parens(tokens: &[(Tok<TokKind<'_>>, TokMeta)]) -> bool {
    if tokens.len() <= 1 {
        return false;
    }
    let mut depth = 0usize;
    for (tok, meta) in tokens {
        let kind = tok.kind;
        // a struct literal needs them where a `{` would open a body
        if depth == 0 && matches!(kind, TokKind::LeftBrace | TokKind::TildeLeftBrace) {
            return true;
        }
        if opens(kind) {
            depth += 1;
            continue;
        }
        if kind.is_closer() {
            depth = depth.saturating_sub(1);
            continue;
        }
        if depth > 0 {
            continue;
        }
        let atom = matches!(
            kind,
            TokKind::Ident(_)
                | TokKind::SelfKeyword
                | TokKind::TyKw(_)
                | TokKind::Int(_)
                | TokKind::Float(_)
                | TokKind::Quantity(..)
                | TokKind::String(_)
                | TokKind::FString(_)
                | TokKind::Tagged(..)
                | TokKind::Hex(_)
                | TokKind::True
                | TokKind::False
                | TokKind::Null
                | TokKind::Dot
                | TokKind::HookDot
                | TokKind::DoubleColon
        ) || (matches!(kind, TokKind::Bang | TokKind::Hook) && meta.glued);
        if !atom {
            return true;
        }
    }
    false
}

/// The `Bind` a fragment name reaches at the repetition indices `at`.
fn reach<'b, 's>(bind: &'b Bind<'s>, at: &[usize]) -> Option<&'b Bind<'s>> {
    match (bind, at.split_first()) {
        (Bind::One(..), _) => Some(bind),
        (Bind::Many(binds), Some((&i, rest))) => binds.get(i).and_then(|b| reach(b, rest)),
        (Bind::Many(_), None) => Some(bind),
    }
}

/// How the parser read a macro call.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Spot {
    /// In an expression: the expansion parses as a block.
    Expr,
    /// As a statement of its own: the expansion's statements and items join
    /// the enclosing list.
    Stmt,
}

impl<'s> Parser<'s> {
    /// Whether the next tokens are a macro declaration: `macro name(` or
    /// `macro name {`, the name on the word's line. `macro` stays a name
    /// anywhere else.
    // r[impl mimas.macro.declare]
    pub(super) fn macro_intro(&self) -> bool {
        matches!(self.peek(), TokKind::Ident("macro"))
            && matches!(self.nth(1), TokKind::Ident(_))
            && !self.meta_at(1).newline
            && matches!(self.nth(2), TokKind::LeftParenthesis | TokKind::LeftBrace)
    }

    /// The index of the closer that matches the opener at `open`, in the
    /// parser's own tokens.
    fn closer_of(&self, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        for i in open..self.tokens.len() {
            match self.tokens[i].kind {
                k if opens(k) => depth += 1,
                k if k.is_closer() => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                TokKind::Eof => return None,
                _ => {}
            }
        }
        None
    }

    fn macro_error(&mut self, at: Location, msg: impl Into<String>, label: impl Into<String>) {
        let err = Misdirection {
            src: self.src(),
            at: at.into(),
            msg: msg.into(),
            label: label.into(),
        };
        // `error` drops a second error at one token. A macro reports at
        // the call, where nothing else has, so it goes in directly.
        self.errors.push(err.into());
    }

    /// Reads a declaration and adds it to the table. On a mistake, it
    /// reports and skips the declaration.
    pub(super) fn macro_decl(&mut self) {
        let kw = self.next;
        self.advance(); // `macro`
        let name = match self.peek() {
            TokKind::Ident(name) => name.to_string(),
            _ => unreachable!("macro_intro saw a name"),
        };
        let name_at = self.next_location();
        self.advance();
        let first = self.next;
        let Some(mut last) = self.closer_of(first) else {
            self.reject_stmt(self.unexpected_token());
            return;
        };
        let single = self.tokens[first].kind == TokKind::LeftParenthesis;
        if single {
            if self.tokens[last + 1].kind != TokKind::LeftBrace {
                self.next = last + 1;
                self.expected("`{` and the macro's template");
                return;
            }
            let Some(body_end) = self.closer_of(last + 1) else {
                self.next = last + 1;
                self.reject_stmt(self.unexpected_token());
                return;
            };
            last = body_end;
        }
        self.next = last + 1;
        if (kw..=last).any(|i| self.meta[i].mark != 0) {
            self.macro_error(
                name_at,
                "a macro can't be declared inside a macro's expansion",
                "declare it at the top of the page",
            );
            return;
        }
        // The declaration relexes from its own text so its tokens outlive
        // this source. The offset keeps their spans on the real file.
        let start = self.tokens[first].span().start();
        let end = self.tokens[last].span().end();
        let Some(text) = self.src.inner().get(start..end) else {
            return;
        };
        let text = intern(text);
        let mut lexer = Lexer::new(text, self.file_id, self.file_name.clone())
            .with_shadowed(crate::lex::shadow_at(&self.shadow, start))
            .with_offset(start);
        let mut toks: Vec<_> = lexer.by_ref().collect();
        toks.retain(|t| !t.kind.is_comment());
        let meta = token_meta(text, start, &toks);
        match parse_def(&toks, &meta, single) {
            Ok(def) => {
                MACROS.with(|m| m.borrow_mut().insert(name, Rc::new(def)));
            }
            Err(e) => self.macro_error(e.at, e.msg, e.label),
        }
    }

    /// Whether the next tokens call a declared macro as a statement of its
    /// own: `name!(…)` with a line break, `;`, `}` or the end after it.
    pub(super) fn stmt_macro_call(&self) -> bool {
        let TokKind::Ident(name) = self.peek() else {
            return false;
        };
        if self.nth(1) != TokKind::Bang || !self.meta_at(1).glued || !opens(self.nth(2)) {
            return false;
        }
        if lookup(name).is_none() {
            return false;
        }
        let Some(close) = self.closer_of(self.next + 2) else {
            return false;
        };
        let after = (close + 1).min(self.tokens.len() - 1);
        self.meta[after].newline
            || matches!(
                self.tokens[after].kind,
                TokKind::SemiColon | TokKind::RightBrace | TokKind::Eof
            )
    }

    /// Whether the name just taken and the `!` next start a macro call.
    pub(super) fn expr_macro_call(&self, expr: &Expr) -> bool {
        if self.next == 0 || !self.glued() || !opens(self.nth(1)) {
            return false;
        }
        let TokKind::Ident(name) = self.tokens[self.next - 1].kind else {
            return false;
        };
        expr.as_ident().is_some() && lookup(name).is_some()
    }

    /// A macro call as a statement. The expansion's tokens replace the call
    /// and the parser reads on from them.
    // r[impl mimas.macro.stmt]
    pub(super) fn stmt_macro(&mut self) -> BlockElement {
        let start = self.next_start();
        let at = self.next;
        if self.hygiene.speculating > 0 {
            self.advance();
            self.advance();
            self.skip();
            return BlockElement::Stmt(self.empty_stmt(start));
        }
        if !self.expand(at, Spot::Stmt) {
            return BlockElement::Stmt(
                self.new_stmt(StmtKind::Expr(self.poison_expr(start)), start),
            );
        }
        if !self.peek().starts_stmt() || self.at(TokKind::RightBrace) {
            // it expanded to nothing
            return BlockElement::Stmt(self.empty_stmt(start));
        }
        self.nested(Self::node)
            .unwrap_or_else(|| BlockElement::Stmt(self.empty_stmt(start)))
    }

    /// A macro call in an expression: `name` was just taken, `!` is next.
    // r[impl mimas.macro.expr]
    pub(super) fn expr_macro(&mut self, name: Expr) -> Expr {
        let start = name.span().start();
        let at = self.next - 1;
        if self.hygiene.speculating > 0 {
            self.advance();
            self.skip();
            return self.new_expr(
                Block {
                    body: vec![],
                    yielded_expr: None,
                },
                start,
            );
        }
        if !self.expand(at, Spot::Expr) {
            return self.poison_expr(start);
        }
        self.block()
    }

    /// A statement that does nothing: where a declaration or an empty
    /// expansion stood.
    pub(super) fn empty_stmt(&self, start: usize) -> Stmt {
        let block = Expr::new(
            Block {
                body: vec![],
                yielded_expr: None,
            }
            .into(),
            Location::new(self.file_id, Span::new(start, start)),
        );
        Stmt::new(
            StmtKind::Expr(block),
            NodeId::new(),
            Location::new(self.file_id, Span::new(start, start)),
        )
    }

    /// Expands the call whose name is token `at`. On success the expansion
    /// replaces the call's tokens and the parser stands at its first token.
    /// On a mistake it reports at the call, skips it, and returns false.
    fn expand(&mut self, at: usize, spot: Spot) -> bool {
        let TokKind::Ident(name) = self.tokens[at].kind else {
            unreachable!("a macro call starts with its name");
        };
        let def = lookup(name).expect("the caller looked the macro up");
        let open = at + 2;
        let Some(close) = self.closer_of(open) else {
            self.next = open;
            self.reject_stmt(self.unexpected_token());
            return false;
        };
        // r[impl mimas.macro.call-site-errors]
        // every token the template writes is located at the whole call
        let call = Location::new(
            self.file_id,
            Span::new(
                self.tokens[at].span().start(),
                self.tokens[close]
                    .span()
                    .end()
                    .max(self.tokens[at].span().start()),
            ),
        );
        // r[impl mimas.macro.recursion]
        // a call a template wrote nests one deeper than the expansion it sits in
        let outer = self.meta[at].mark;
        let depth = if outer == 0 {
            1
        } else {
            self.hygiene.depths[outer as usize - 1] + 1
        };
        if depth > MAX_DEPTH || self.hygiene.expansions >= MAX_EXPANSIONS {
            self.next = close + 1;
            // r[impl mimas.macro.recursion-limit]
            self.macro_error(
                call,
                format!("`{name}!` expands too deep (more than {MAX_DEPTH} levels)"),
                "this call never stops expanding",
            );
            return false;
        }

        // r[impl mimas.macro.match]
        // The first rule whose pattern takes every token of the call wins.
        let tail = self.tokens.len() - close;
        let mut found = None;
        for arm in &def.arms {
            self.next = open + 1;
            let mut binds = Binds::new();
            if self.match_seq(&arm.pattern, tail, &mut binds)
                && self.next == self.tokens.len() - tail
            {
                found = Some((arm, binds));
                break;
            }
        }
        let close = self.tokens.len() - tail;
        let Some((arm, binds)) = found else {
            self.next = close + 1;
            // r[impl mimas.macro.no-match]
            self.macro_error(
                call,
                format!("no rule of `{name}!` matches this call"),
                format!("`{name}!` can't take these arguments"),
            );
            return false;
        };

        self.hygiene.expansions += 1;
        let mark = self.hygiene.expansions;
        self.hygiene.depths.push(depth);
        let mut out = vec![];
        if spot == Spot::Expr {
            out.push((
                Tok::new(TokKind::LeftBrace, call.into()),
                TokMeta {
                    mark,
                    ..TokMeta::default()
                },
            ));
        }
        if let Err(msg) = self.transcribe(&arm.template, &binds, &[], call, mark, &mut out) {
            self.next = close + 1;
            self.macro_error(call, msg, format!("in this use of `{name}!`"));
            return false;
        }
        if spot == Spot::Expr {
            out.push((
                Tok::new(TokKind::RightBrace, call.into()),
                TokMeta {
                    mark,
                    ..TokMeta::default()
                },
            ));
        }
        // the expansion sits where the call's name did
        if let Some((_, first)) = out.first_mut() {
            first.glued = self.meta[at].glued;
            first.newline = self.meta[at].newline;
        }
        self.splice(at, close, out);
        self.next = at;
        true
    }

    /// Puts `out` in place of tokens `from..=to`.
    fn splice(&mut self, from: usize, to: usize, out: Vec<(Tok<TokKind<'s>>, TokMeta)>) {
        let added = out.len();
        let removed = to + 1 - from;
        let (toks, meta): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        self.tokens.splice(from..=to, toks);
        self.meta.splice(from..=to, meta);
        let shift = |i: usize| i + added - removed;
        self.docs = std::mem::take(&mut self.docs)
            .into_iter()
            .filter(|(i, _)| *i <= from || *i > to)
            .map(|(i, doc)| (if i > to { shift(i) } else { i }, doc))
            .collect();
        if let Some(e) = self.last_error
            && e > to
        {
            self.last_error = Some(shift(e));
        }
    }

    /// Matches `pattern` from the cursor. The call's tokens end `tail`
    /// tokens before the end of the stream.
    fn match_seq(&mut self, pattern: &[Matcher], tail: usize, binds: &mut Binds<'s>) -> bool {
        for m in pattern {
            let end = self.tokens.len() - tail;
            match m {
                Matcher::Tok(kind) => {
                    if self.next >= end || self.peek() != *kind {
                        return false;
                    }
                    self.advance();
                }
                Matcher::Frag(name, kind) => {
                    let from = self.next;
                    if from >= end || !self.fragment(*kind, tail) {
                        return false;
                    }
                    let tokens: Vec<_> = (from..self.next)
                        .map(|i| (self.tokens[i], self.meta[i]))
                        .collect();
                    let wrap = *kind == Frag::Expr && needs_parens(&tokens);
                    binds.insert(name, Bind::One(tokens, wrap));
                }
                // r[impl mimas.macro.repeat]
                Matcher::Repeat(inner, sep, rep) => {
                    let mut names = vec![];
                    pattern_names(inner, &mut names);
                    let mut rounds: Vec<Binds<'s>> = vec![];
                    loop {
                        let before = self.next;
                        if !rounds.is_empty()
                            && let Some(sep) = sep
                        {
                            if self.next >= self.tokens.len() - tail || self.peek() != *sep {
                                break;
                            }
                            self.advance();
                        }
                        let mut round = Binds::new();
                        let from = self.next;
                        if from < self.tokens.len() - tail
                            && self.match_seq(inner, tail, &mut round)
                            && self.next > from
                        {
                            rounds.push(round);
                        } else {
                            self.next = before;
                            break;
                        }
                        if *rep == Rep::Maybe {
                            break;
                        }
                    }
                    if *rep == Rep::Some && rounds.is_empty() {
                        return false;
                    }
                    for name in names {
                        let each = rounds
                            .iter_mut()
                            .filter_map(|round| round.remove(name))
                            .collect();
                        binds.insert(name, Bind::Many(each));
                    }
                }
            }
        }
        true
    }

    /// Parses one fragment only to find where it ends, then leaves the
    /// cursor there. Nothing it reports stays reported. Returns whether it
    /// parsed cleanly inside the call.
    fn fragment(&mut self, kind: Frag, tail: usize) -> bool {
        let from = self.next;
        let errors = self.errors.len();
        let last_error = self.last_error;
        let pending = self.pending.len();
        let bound = self.hygiene.bound.clone();
        self.hygiene.speculating += 1;
        let parsed = match kind {
            Frag::Ident => {
                let ok = matches!(self.peek(), TokKind::Ident(_));
                if ok {
                    self.advance();
                }
                ok
            }
            Frag::Tt => {
                let ok = !self.peek().is_closer() && self.peek() != TokKind::Eof;
                if ok {
                    self.skip();
                }
                ok
            }
            Frag::Literal => {
                if self.at(TokKind::Minus)
                    && matches!(self.nth(1), TokKind::Int(_) | TokKind::Float(_))
                {
                    self.advance();
                }
                let ok = Literal::try_from(self.peek()).is_ok();
                if ok {
                    self.advance();
                }
                ok
            }
            Frag::Expr => {
                // read like a call's argument: commas or adjacency end it
                self.in_group(|p| {
                    p.with_juxta(Juxta::Comma, |p| p.struct_literals(true, Self::expr))
                });
                true
            }
            Frag::Ty => {
                self.annotation();
                true
            }
            Frag::Pat => {
                self.pattern();
                true
            }
            Frag::Block => {
                let ok = self.at(TokKind::LeftBrace);
                if ok {
                    self.block();
                }
                ok
            }
        };
        self.hygiene.speculating -= 1;
        let clean = parsed
            && self.errors.len() == errors
            && self.next > from
            && self.next <= self.tokens.len() - tail;
        self.errors.truncate(errors);
        self.last_error = last_error;
        self.pending.truncate(pending);
        self.hygiene.bound = bound;
        if !clean {
            self.next = from;
        }
        clean
    }

    /// Writes the template's tokens, with each fragment's tokens in place of
    /// its `$name`. `at` holds the index of each `$( … )` round we are in.
    fn transcribe(
        &self,
        template: &[Trans],
        binds: &Binds<'s>,
        at: &[usize],
        call: Location,
        mark: u32,
        out: &mut Vec<(Tok<TokKind<'s>>, TokMeta)>,
    ) -> std::result::Result<(), String> {
        for t in template {
            match t {
                Trans::Tok(tok) => out.push((
                    Tok::new(tok.kind, call.into()),
                    TokMeta { mark, ..tok.meta },
                )),
                Trans::Var(name, meta) => {
                    let bind = binds.get(name).and_then(|b| reach(b, at));
                    let Some(Bind::One(tokens, wrap)) = bind else {
                        return Err(format!(
                            "`${name}` repeats a different number of times here"
                        ));
                    };
                    let place = |m: TokMeta| TokMeta {
                        glued: meta.glued,
                        newline: meta.newline,
                        ..m
                    };
                    if *wrap {
                        let (first, last) =
                            (tokens[0].0.location, tokens[tokens.len() - 1].0.location);
                        out.push((
                            Tok::new(TokKind::LeftParenthesis, first),
                            place(TokMeta::default()),
                        ));
                        let inside = TokMeta {
                            glued: true,
                            newline: false,
                            mark: 0,
                        };
                        for (i, (tok, m)) in tokens.iter().enumerate() {
                            let m = if i == 0 {
                                TokMeta {
                                    mark: m.mark,
                                    ..inside
                                }
                            } else {
                                *m
                            };
                            out.push((*tok, m));
                        }
                        out.push((Tok::new(TokKind::RightParenthesis, last), inside));
                    } else {
                        for (i, (tok, m)) in tokens.iter().enumerate() {
                            out.push((*tok, if i == 0 { place(*m) } else { *m }));
                        }
                    }
                }
                Trans::Repeat(inner, sep) => {
                    let mut vars = vec![];
                    trans_vars(inner, &mut vars);
                    let mut count: Option<(usize, &str)> = None;
                    for name in vars {
                        if let Some(Bind::Many(each)) = binds.get(name).and_then(|b| reach(b, at)) {
                            match count {
                                Some((n, other)) if n != each.len() => {
                                    return Err(format!(
                                        "`${other}` repeats {n} times but `${name}` repeats {}",
                                        each.len()
                                    ));
                                }
                                _ => count = Some((each.len(), name)),
                            }
                        }
                    }
                    let n = count.map_or(0, |(n, _)| n);
                    for i in 0..n {
                        if i > 0
                            && let Some(sep) = sep
                        {
                            out.push((
                                Tok::new(sep.kind, call.into()),
                                TokMeta { mark, ..sep.meta },
                            ));
                        }
                        let mut deeper = at.to_vec();
                        deeper.push(i);
                        self.transcribe(inner, binds, &deeper, call, mark, out)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// A name a template binds: renamed for this expansion, so it can't
    /// capture or be seen by the call site's names. `_` binds nothing and
    /// stays as it is.
    // r[impl mimas.macro.hygiene]
    pub(super) fn hygiene_bind(&mut self, ident: Ident) -> Ident {
        let Some(&mark) = self.hygiene.marks.get(&ident.id) else {
            return ident;
        };
        if ident.lexeme == "_" {
            return ident;
        }
        self.hygiene.bound.insert((ident.lexeme.clone(), mark));
        self.renamed(ident, mark)
    }

    /// A name used in an expression. A template's name for one of its own
    /// locals takes that local's renamed form. Any other name is left alone.
    pub(super) fn hygiene_use(&self, ident: Ident) -> Ident {
        let Some(&mark) = self.hygiene.marks.get(&ident.id) else {
            return ident;
        };
        if !self.hygiene.bound.contains(&(ident.lexeme.clone(), mark)) {
            return ident;
        }
        self.renamed(ident, mark)
    }

    fn renamed(&self, ident: Ident, mark: u32) -> Ident {
        // the file id keeps two files' top-level expansions apart
        let lexeme = if self.file_id == 0 {
            format!("{}·{mark}", ident.lexeme)
        } else {
            format!("{}·{}.{mark}", ident.lexeme, self.file_id)
        };
        Ident { lexeme, ..ident }
    }

    /// An f-string interpolation parser takes the template's mark and the
    /// locals bound so far, when a macro template wrote the f-string.
    pub(super) fn adopt_template(&mut self, outer: &Hygiene, mark: u32) {
        for m in &mut self.meta {
            m.mark = mark;
        }
        self.hygiene.bound = outer.bound.clone();
    }
}

fn trans_vars(template: &[Trans], out: &mut Vec<&'static str>) {
    for t in template {
        match t {
            Trans::Tok(_) => {}
            Trans::Var(name, _) => out.push(name),
            Trans::Repeat(inner, ..) => trans_vars(inner, out),
        }
    }
}

/// Reads a declaration's relexed tokens: `(pattern) { template }` when
/// `single`, or `{ (pattern) => { template } … }`.
fn parse_def(
    toks: &[Tok<TokKind<'static>>],
    meta: &[TokMeta],
    single: bool,
) -> std::result::Result<MacroDef, DefError> {
    let mut c = DefCursor {
        toks,
        meta,
        i: 0,
        end: toks.len(),
    };
    let mut arms = vec![];
    if single {
        arms.push(parse_arm(&mut c)?);
    } else {
        let close = c.closer(0).unwrap_or(toks.len() - 1);
        c.i = 1;
        c.end = close;
        while c.i < close {
            if matches!(c.peek(), TokKind::Comma | TokKind::SemiColon) {
                c.i += 1;
                continue;
            }
            arms.push(parse_arm(&mut c)?);
        }
        if arms.is_empty() {
            c.i = 0;
            return Err(c.error("this macro has no rules", "add `(pattern) => { template }`"));
        }
    }
    Ok(MacroDef { arms })
}

/// One `(pattern) { template }` (or with `=>` between), at the cursor.
fn parse_arm(c: &mut DefCursor) -> std::result::Result<Arm, DefError> {
    if c.peek() != TokKind::LeftParenthesis {
        return Err(c.error(
            "a macro rule starts with its `(pattern)`",
            "expected `(` here",
        ));
    }
    let open = c.i;
    let Some(close) = c.closer(open) else {
        return Err(c.error("this pattern never closes", "close it with `)`"));
    };
    let end = c.end;
    c.i = open + 1;
    let pattern = c.pattern(close)?;
    c.i = close + 1;
    if c.peek() == TokKind::FatArrow {
        c.i += 1;
    }
    if c.peek() != TokKind::LeftBrace {
        return Err(c.error("a macro rule's template goes in `{ }`", "expected `{` here"));
    }
    let body = c.i;
    let Some(body_close) = c.closer(body) else {
        return Err(c.error("this template never closes", "close it with `}`"));
    };
    c.i = body + 1;
    let template = c.template(body_close)?;
    c.end = end;
    let mut depths = StdMap::new();
    pattern_depths(&pattern, 0, &mut depths);
    if let Err(msg) = check_template(&template, &depths, 0) {
        c.i = body;
        return Err(c.error(msg, "in this template"));
    }
    c.i = body_close + 1;
    Ok(Arm { pattern, template })
}
