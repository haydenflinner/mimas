//! Reflection on declarations. A program can ask what it declared.
//!
//! Declaration queries answer in two forms. The plain form
//! expands to a value. The `=>` form hands the declaration to a macro
//! of yours as tokens, so a macro can write code from it.
//!
//! ```mimas
//! /// Twice a number.
//! fn double(x: float) -> float { x * 2.0 }
//!
//! sig!(double)            // FnSig { name = "double", params = [FnParam { name = "x",
//!                         //   ty = "float" }], ret = "float", doc = "Twice a number.", marks = [] }
//! sig!(double => m)       // m!(double; (x: float) -> float; "Twice a number.")
//!
//! fns!(machine)           // [FnSig] of the functions marked `#[machine]`
//! fns!(machine => m)      // [m!(…), m!(…)], one call for each of them
//! fns!()                  // every function declared so far
//!
//! fields!(Ball)           // [StructField { name = "id", ty = "float" }, …]
//! fields!(Ball => m)      // m!(Ball; id: float, v: Val)
//!
//! variants!(Val)          // [EnumVariant { name = "Num", fields = [StructField { name = "0", …
//! variants!(Val => m)     // m!(Val; Num(float), Bit(bool), Nothing)
//!
//! graph!(full_adder)      // FnGraph { name = "full_adder", ins = ["a", "b", "c"], … }
//! graph!(full_adder => m) // m!(full_adder; FnGraph { … })
//! ```
//!
//! `graph!` lowers a function's body to a dataflow: one `in` node per
//! parameter, a `const` per literal, a `call` per call, a `sel` for one
//! part of a call's result. `rets` names the nodes the result is made
//! of. A body that is not calls, tuples, `let`, parameters and literals
//! (a `match`, a loop, an operator) is not dataflow: the answer's
//! `opaque` is true and `why` says what the body was instead.
//!
//! A call sees every declaration of its own file, above it or below it,
//! and every declaration of the files of the program parsed before it.
//! Types are text, as written. A method counts as `Type::name`:
//! `sig!(Ball::step)`, and `fns!` lists marked methods by that name.
//!
//! Seeing below works in two passes. The first pass answers from what it
//! has read so far, and remembers each answer. At the end of the file it
//! asks again with the whole file read. When an answer changed, or a name
//! was missing, the file parses a second time with the first pass's table
//! as the declarations "ahead", and only then is a missing name an error.
//! A file whose calls all look upward parses once.
//!
//! A mark on a function is an attribute nobody else knows: `#[machine]`.
//! It is legal when some `fns!(machine …)` in the program reads it, so a
//! misspelled `#[tets]` is still an error (see `Parser::unread_marks`).
//!
//! The expansion is source text, lexed and put in place of the call the
//! way a declared macro's template is. The struct types it names
//! (`FnSig`, `FnParam`, `StructField`, `EnumVariant`) are declared in
//! std (`library/src/reflect.mim`).

mod for_fields;

use super::macros::Spot;
use super::*;
use std::{cell::RefCell, collections::HashSet, rc::Rc};

/// A name and its type as written: a parameter, or a field.
#[derive(Clone)]
struct Named {
    name: String,
    ty: String,
    /// `x: inout T` -- the param's call convention lends the caller's slot.
    /// Only `FnParam`s carry it; fields and variants always read `false`.
    inout: bool,
}

#[derive(Clone)]
struct FnRec {
    name: String,
    /// The `//# page:` region the function is declared in (`""` for the
    /// page's own text). One name may be declared once per region.
    page: String,
    params: Vec<Named>,
    ret: String,
    doc: String,
    /// Each mark's name and its arguments as text: `#[command("f s")]` is
    /// `("command", ["f s"])`.
    marks: Vec<(String, Vec<String>)>,
    /// The body as parsed. `graph!` lowers it to a dataflow; the other
    /// macros never look at it. Shared: a table snapshot clones every rec.
    body: Option<Rc<Expr>>,
}

#[derive(Clone)]
struct VariantRec {
    name: String,
    fields: Vec<Named>,
    /// The variant as written: `Num(float)`, `Won { t: Team }`, `Nothing`.
    text: String,
}

#[derive(Clone, Default)]
pub(super) struct Decls {
    /// In declaration order. A later declaration of a name replaces the
    /// earlier one in place.
    fns: Vec<FnRec>,
    structs: Vec<(String, Vec<Named>)>,
    enums: Vec<(String, Vec<VariantRec>)>,
    impls: Vec<(String, String)>,
    pacts: Vec<(String, Vec<String>)>,
    /// Every mark some `fns!` call has asked for.
    read_marks: HashSet<String>,
    /// In a file's second pass: the table its first pass ended with, so a
    /// call sees the declarations below it.
    ahead: Option<Rc<Decls>>,
    /// A top-level parse is in its first pass and can run again: a missing
    /// name answers `()` for now instead of failing.
    armed: bool,
    /// What each call answered in the first pass, to ask again at its end.
    asks: Vec<Ask>,
}

/// One reflection call and what it answered: `None` for a missing name.
#[derive(Clone)]
struct Ask {
    name: String,
    target: Option<String>,
    mac: Option<String>,
    answer: Option<String>,
}

/// What a reflection call expands to.
enum Answer {
    Text(String),
    /// The name it asks about is not declared (yet): (message, label).
    Miss(String, String),
    /// The call itself is wrong: (message, label).
    Bad(String, String),
}

impl Decls {
    /// The function `name`: the page's own first, then any region's.
    /// `kit_a::f` names region `kit-a`'s `f`.
    fn find_fn(&self, name: &str) -> Option<&FnRec> {
        let hit = |f: &&FnRec| {
            f.name == name
                || (!f.page.is_empty()
                    && name.strip_suffix(f.name.as_str()).and_then(|p| p.strip_suffix("::"))
                        == Some(f.page.replace('-', "_").as_str()))
        };
        let find = |fns: &[FnRec]| -> Option<usize> {
            fns.iter()
                .position(|f| hit(&f) && f.page.is_empty())
                .or_else(|| fns.iter().position(|f| hit(&f)))
        };
        match find(&self.fns) {
            Some(i) => Some(&self.fns[i]),
            None => self.ahead.as_ref().and_then(|a| find(&a.fns).map(|i| &a.fns[i])),
        }
    }

    /// Every function, in declaration order. In a second pass that is the
    /// whole file's order, each one as this pass has it when it has it.
    fn all_fns(&self) -> Vec<&FnRec> {
        match &self.ahead {
            None => self.fns.iter().collect(),
            Some(a) => {
                let mut out: Vec<&FnRec> = a
                    .fns
                    .iter()
                    .map(|f| {
                        self.fns
                            .iter()
                            .find(|g| g.name == f.name && g.page == f.page)
                            .unwrap_or(f)
                    })
                    .collect();
                for f in &self.fns {
                    if !a.fns.iter().any(|g| g.name == f.name && g.page == f.page) {
                        out.push(f);
                    }
                }
                out
            }
        }
    }

    fn find_struct(&self, name: &str) -> Option<&Vec<Named>> {
        let find = |d: &Decls| -> Option<usize> { d.structs.iter().position(|(n, _)| n == name) };
        match find(self) {
            Some(i) => Some(&self.structs[i].1),
            None => self.ahead.as_ref().and_then(|a| find(a).map(|i| &a.structs[i].1)),
        }
    }

    fn find_enum(&self, name: &str) -> Option<&Vec<VariantRec>> {
        let find = |d: &Decls| -> Option<usize> { d.enums.iter().position(|(n, _)| n == name) };
        match find(self) {
            Some(i) => Some(&self.enums[i].1),
            None => self.ahead.as_ref().and_then(|a| find(a).map(|i| &a.enums[i].1)),
        }
    }
}

/// What a top-level parse keeps so it can run its file a second time.
pub(super) struct Rerun<'s> {
    pub(super) source: &'s str,
    pub(super) file_id: FileId,
    pub(super) file_name: String,
    pub(super) shadow: Vec<(usize, [bool; 3])>,
    /// The macro and declaration tables as the file found them.
    pub(super) start: Option<(StdMacros, Decls)>,
}

type StdMacros = std::collections::HashMap<String, Rc<macros::MacroDef>>;

/// The start of a top-level parse's first pass: remember the tables as
/// the file finds them, and let a missing name wait for the file's end.
pub(super) fn begin_first_pass(rerun: &mut Rerun<'_>) {
    rerun.start = Some((macros::snapshot(), snapshot()));
    DECLS.with(|d| {
        let mut d = d.borrow_mut();
        d.armed = true;
        d.asks.clear();
    });
}

/// The end of a first pass: whether some call would answer differently
/// with the whole file read, or named something missing.
// r[impl mimas.reflect.below]
pub(super) fn finish_first_pass() -> bool {
    DECLS.with(|d| {
        let mut d = d.borrow_mut();
        d.armed = false;
        let asks = std::mem::take(&mut d.asks);
        asks.iter().any(|a| {
            match answer(&d, &a.name, a.target.as_deref(), a.mac.as_deref()) {
                Answer::Text(t) => a.answer.as_ref() != Some(&t),
                Answer::Miss(..) | Answer::Bad(..) => true,
            }
        })
    })
}

/// Put the tables back as the file found them, with the first pass's
/// table ahead: the second pass starts.
pub(super) fn begin_second_pass(start: (StdMacros, Decls)) {
    let ahead = Rc::new(snapshot());
    let (macs, mut decls) = start;
    macros::install(macs);
    decls.ahead = Some(ahead);
    decls.armed = false;
    install(decls);
}

/// The second pass is over: the table it built holds the whole file.
pub(super) fn end_second_pass() {
    DECLS.with(|d| d.borrow_mut().ahead = None);
}

thread_local! {
    /// The declarations read so far. Ambient like the macro table: a fresh
    /// [`Parser::new`] clears it, and the files of one program share it.
    static DECLS: RefCell<Decls> = RefCell::default();
}

/// Forgets every declaration. A new program starts with none.
pub(super) fn clear() {
    DECLS.with(|d| *d.borrow_mut() = Decls::default());
}

/// The declaration table as it stands, for a subordinate parse (the
/// prelude) to put back — its own items are not the program's to reflect.
pub(super) fn snapshot() -> Decls {
    DECLS.with(|d| d.borrow().clone())
}

/// Makes `decls` the live table — see [`snapshot`].
pub(super) fn install(decls: Decls) {
    DECLS.with(|d| *d.borrow_mut() = decls);
}

/// Whether `name!(…)` is one of the reflection macros.
pub(super) fn is_builtin(name: &str) -> bool {
    matches!(name, "sig" | "fns" | "fields" | "variants" | "graph" | "for_fields")
}

/// Whether some `fns!` call has read functions marked `#[name]`.
pub(super) fn mark_is_read(name: &str) -> bool {
    DECLS.with(|d| {
        let d = d.borrow();
        d.read_marks.contains(name)
            || d.ahead.as_ref().is_some_and(|a| a.read_marks.contains(name))
    })
}

/// A string literal that holds `text`.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '{' => out.push_str("{{"),
            '}' => out.push_str("}}"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn field_value(kind: &str, f: &Named) -> String {
    if kind == "FnParam" {
        format!(
            "FnParam {{ name = {}, ty = {}, inout = {} }}",
            quote(&f.name),
            quote(&f.ty),
            f.inout,
        )
    } else {
        format!("{kind} {{ name = {}, ty = {} }}", quote(&f.name), quote(&f.ty))
    }
}

fn list_value(elem: &str, items: Vec<String>) -> String {
    if items.is_empty() {
        // an empty list literal has no type of its own
        format!("{{ let none: [{elem}] = []; none }}")
    } else {
        format!("[{}]", items.join(", "))
    }
}

impl FnRec {
    fn value(&self) -> String {
        let params: Vec<_> = self.params.iter().map(|p| field_value("FnParam", p)).collect();
        let marks: Vec<_> = self
            .marks
            .iter()
            .map(|(name, args)| {
                format!(
                    "FnMark {{ name = {}, args = {} }}",
                    quote(name),
                    list_value("str", args.iter().map(|a| quote(a)).collect())
                )
            })
            .collect();
        format!(
            "FnSig {{ name = {}, page = {}, params = {}, ret = {}, doc = {}, marks = {}, writes = __fn_writes({}) }}",
            quote(&self.name),
            quote(&self.page),
            list_value("FnParam", params),
            quote(&self.ret),
            quote(&self.doc),
            list_value("FnMark", marks),
            if self.page.is_empty() {
                self.name.clone()
            } else {
                format!("{}::{}", self.page.replace('-', "_"), self.name)
            },
        )
    }

    /// `m!(name; (a: T, b: U) -> R; "doc")`.
    fn hand_to(&self, mac: &str) -> String {
        let params: Vec<_> = self
            .params
            .iter()
            .map(|p| {
                if p.ty.is_empty() {
                    p.name.clone()
                } else {
                    format!("{}: {}", p.name, p.ty)
                }
            })
            .collect();
        format!(
            "{mac}!({}; ({}) -> {}; {})",
            self.name,
            params.join(", "),
            self.ret,
            quote(&self.doc)
        )
    }
}

impl<'s> Parser<'s> {
    /// Remembers a top-level declaration, so the reflection macros can
    /// answer for it. `first` is the index of the item's first token,
    /// where its doc comment hangs.
    // r[impl mimas.reflect.visible]
    pub(super) fn record_decl(&mut self, item: &Item, first: usize) {
        match item.kind() {
            ItemKind::Function(f) if !f.is_method() => {
                let name = macros::unhygienic(&f.name).lexeme;
                self.record_fn(name, f, item.attrs(), first);
            }
            ItemKind::Impl(i) => {
                if let Some(pact) = &i.pact {
                    DECLS.with(|d| d.borrow_mut().impls.push((i.target.lexeme.clone(), pact.lexeme.clone())));
                }
                self.record_impl_members(i);
            }
            ItemKind::Struct(s) => {
                let fields = s.fields.iter().map(named_field).collect();
                let name = s.name.lexeme.clone();
                DECLS.with(|d| {
                    let structs = &mut d.borrow_mut().structs;
                    structs.retain(|(n, _)| *n != name);
                    structs.push((name, fields));
                });
            }
            ItemKind::Enum(e) => {
                let variants = e
                    .members
                    .iter()
                    .map(|(name, member)| {
                        let (fields, text): (Vec<Named>, String) = match member {
                            Member::Tuple(tys) if tys.is_empty() => (vec![], name.lexeme.clone()),
                            Member::Tuple(tys) => {
                                let fields: Vec<_> = tys
                                    .iter()
                                    .enumerate()
                                    .map(|(i, t)| Named {
                                        name: i.to_string(),
                                        ty: t.to_string(),
                                        inout: false,
                                    })
                                    .collect();
                                let inside: Vec<_> = fields.iter().map(|f| f.ty.clone()).collect();
                                let text = format!("{}({})", name.lexeme, inside.join(", "));
                                (fields, text)
                            }
                            Member::Struct(fs) => {
                                let fields: Vec<_> = fs.iter().map(named_field).collect();
                                let inside: Vec<_> = fields
                                    .iter()
                                    .map(|f| format!("{}: {}", f.name, f.ty))
                                    .collect();
                                let text = format!("{} {{ {} }}", name.lexeme, inside.join(", "));
                                (fields, text)
                            }
                        };
                        VariantRec {
                            name: name.lexeme.clone(),
                            fields,
                            text,
                        }
                    })
                    .collect();
                let name = e.head.lexeme.clone();
                DECLS.with(|d| {
                    let enums = &mut d.borrow_mut().enums;
                    enums.retain(|(n, _)| *n != name);
                    enums.push((name, variants));
                });
            }
            ItemKind::Pact(p) => {
                DECLS.with(|d| d.borrow_mut().pacts.push((p.name.lexeme.clone(), p.requires.iter().map(|r| r.lexeme.clone()).collect())));
            }
            _ => {}
        }
    }

    /// Remembers each `fn` member of an impl (or of a struct body) as
    /// `Type::name`. r[impl mimas.reflect.methods]
    pub(super) fn record_impl_members(&mut self, i: &Impl) {
        for member in &i.items {
            if let ItemKind::Function(f) = member.kind() {
                let name = format!("{}::{}", i.target.lexeme, macros::unhygienic(&f.name).lexeme);
                let at = member.location().span.start;
                let first = self
                    .tokens
                    .iter()
                    .position(|t| t.span().start() >= at)
                    .unwrap_or(0);
                self.record_fn(name, f, member.attrs(), first);
            }
        }
    }

    fn record_fn(&mut self, name: String, f: &Function, attrs: &[Attribute], first: usize) {
        let params = f
            .parameters
            .iter()
            .map(|b| Named {
                name: b
                    .left
                    .as_ident()
                    .map(|i| macros::unhygienic(i).lexeme)
                    .unwrap_or_else(|| "_".to_string()),
                ty: b.annotation.as_ref().map(|a| a.to_string()).unwrap_or_default(),
                inout: matches!(b.annotation, Some(Annotation::Inout(..))),
            })
            .collect();
        let doc = self
            .docs
            .get(&first)
            .map(|doc| {
                doc.lines()
                    .map(|line| {
                        let line = line.trim_start().trim_start_matches('/');
                        line.strip_prefix(' ').unwrap_or(line).trim_end()
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        let page = self
            .tokens
            .get(first)
            .map(|t| self.region_at(t.span().start()))
            .unwrap_or_default();
        let rec = FnRec {
            name,
            page,
            params,
            ret: f
                .return_type
                .as_ref()
                .map_or_else(|| "()".to_string(), |a| a.to_string()),
            doc,
            marks: attrs
                .iter()
                .filter(|a| a.name.lexeme != "test")
                .map(|a| (a.name.lexeme.clone(), a.args.clone()))
                .collect(),
            body: Some(Rc::new(f.body.clone())),
        };
        DECLS.with(|d| {
            let fns = &mut d.borrow_mut().fns;
            match fns.iter_mut().find(|f| f.name == rec.name && f.page == rec.page) {
                Some(old) => *old = rec,
                None => fns.push(rec),
            }
        });
    }

    /// Reports each mark on a function that no `fns!` call read. Called
    /// once the whole source is parsed.
    // r[impl mimas.reflect.mark]
    pub(super) fn unread_marks(&mut self) {
        for (name, at) in std::mem::take(&mut self.pending_marks) {
            if !mark_is_read(&name) {
                self.error(UnknownAttribute {
                    src: self.src(),
                    at: at.into(),
                    name,
                });
            }
        }
    }

    /// Expands the reflection call whose name is token `at`: the same
    /// contract as a declared macro's `expand`.
    pub(super) fn expand_reflect(&mut self, at: usize, spot: Spot) -> bool {
        let TokKind::Ident(name) = self.tokens[at].kind else {
            unreachable!("a macro call starts with its name");
        };
        if name == "for_fields" { return self.expand_for_fields(at, spot); }
        let open = at + 2;
        let Some(close) = self.closer_of(open) else {
            self.next = open;
            self.reject_stmt(self.unexpected_token());
            return false;
        };
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
        // The call's own tokens: `target`, `target => mac`, `=> mac`, or none.
        let args: Vec<TokKind<'s>> = self.tokens[open + 1..close].iter().map(|t| t.kind).collect();
        // `Type::method` names a method (`sig!` and `graph!` only)
        let path;
        let (target, mac) = match args.as_slice() {
            [] => (None, None),
            [TokKind::Ident(t)] => (Some(*t), None),
            [TokKind::Ident(t), TokKind::FatArrow, TokKind::Ident(m)] => (Some(*t), Some(*m)),
            [TokKind::FatArrow, TokKind::Ident(m)] => (None, Some(*m)),
            [TokKind::Ident(t), TokKind::DoubleColon, TokKind::Ident(m), rest @ ..]
                if matches!(name, "sig" | "graph")
                    && matches!(rest, [] | [TokKind::FatArrow, TokKind::Ident(_)]) =>
            {
                path = format!("{t}::{m}");
                let mac = match rest {
                    [_, TokKind::Ident(mac)] => Some(*mac),
                    _ => None,
                };
                (Some(path.as_str()), mac)
            }
            _ => {
                self.next = close + 1;
                self.macro_error(
                    call,
                    format!("`{name}!` takes a name, or a name and `=> macro`"),
                    format!("write `{name}!(name)` or `{name}!(name => macro)`"),
                );
                return false;
            }
        };
        let text = match self.reflect_text(name, target, mac) {
            Ok(text) => text,
            Err((msg, label)) => {
                self.next = close + 1;
                self.macro_error(call, msg, label);
                return false;
            }
        };

        // the same nesting count a declared macro's expansion keeps
        let outer = self.meta[at].mark;
        let depth = if outer == 0 {
            1
        } else {
            self.hygiene_depth(outer) + 1
        };
        let Some(mark) = self.hygiene_enter(depth) else {
            self.next = close + 1;
            self.macro_error(
                call,
                format!("`{name}!` expands too deep"),
                "this call never stops expanding",
            );
            return false;
        };

        let start = self.tokens[at].span().start();
        let text = macros::intern(&text);
        let mut lexer = Lexer::new(text, self.file_id, self.file_name.clone())
            .with_shadowed(crate::lex::shadow_at(&self.shadow, start))
            .with_offset(start);
        let mut toks: Vec<_> = lexer.by_ref().collect();
        toks.retain(|t| !t.kind.is_comment());
        let meta = token_meta(text, start, &toks);
        let mut out: Vec<(Tok<TokKind<'s>>, TokMeta)> = Vec::with_capacity(toks.len() + 2);
        let open_brace = spot == Spot::Expr;
        if open_brace {
            out.push((
                Tok::new(TokKind::LeftBrace, call.into()),
                TokMeta {
                    mark,
                    ..TokMeta::default()
                },
            ));
        }
        for (tok, m) in toks.into_iter().zip(meta) {
            // r[impl mimas.macro.call-site-errors]
            out.push((Tok::new(tok.kind, call.into()), TokMeta { mark, ..m }));
        }
        if open_brace {
            out.push((
                Tok::new(TokKind::RightBrace, call.into()),
                TokMeta {
                    mark,
                    ..TokMeta::default()
                },
            ));
        }
        if let Some((_, first)) = out.first_mut() {
            first.glued = self.meta[at].glued;
            first.newline = self.meta[at].newline;
        }
        self.splice(at, close, out);
        self.next = at;
        true
    }

    /// The source text one reflection call expands to, or the error
    /// (message, label) to report at the call. In a first pass a missing
    /// name answers `()` and the file runs again (see the module doc).
    fn reflect_text(
        &mut self,
        name: &str,
        target: Option<&str>,
        mac: Option<&str>,
    ) -> std::result::Result<String, (String, String)> {
        DECLS.with(|d| {
            let mut d = d.borrow_mut();
            if name == "fns"
                && let Some(wanted) = target
            {
                d.read_marks.insert(wanted.to_string());
            }
            let ask = |answer: Option<String>| Ask {
                name: name.to_string(),
                target: target.map(str::to_string),
                mac: mac.map(str::to_string),
                answer,
            };
            match answer(&d, name, target, mac) {
                Answer::Text(text) => {
                    if d.armed {
                        d.asks.push(ask(Some(text.clone())));
                    }
                    Ok(text)
                }
                Answer::Miss(..) if d.armed => {
                    d.asks.push(ask(None));
                    Ok("()".to_string())
                }
                Answer::Miss(msg, label) | Answer::Bad(msg, label) => Err((msg, label)),
            }
        })
    }
}

/// What one reflection call answers from the table `d`.
// r[impl mimas.reflect.sig]
// r[impl mimas.reflect.fns]
// r[impl mimas.reflect.fields]
// r[impl mimas.reflect.variants]
// r[impl mimas.reflect.hand]
fn answer(d: &Decls, name: &str, target: Option<&str>, mac: Option<&str>) -> Answer {
    let need = |what: &str| -> std::result::Result<&str, Answer> {
        target.ok_or_else(|| {
            Answer::Bad(
                format!("`{name}!` needs the name of {what}"),
                format!("write `{name}!(name)`"),
            )
        })
    };
    let miss = |kind: &str, target: &str, plural: &str| {
        Answer::Miss(
            format!("no {kind} `{target}` is declared for this `{name}!`"),
            format!("`{name}!` sees the {plural} of this file and of the files before it; `{target}` is not one"),
        )
    };
    let target = match name {
        "fns" => target,
        "sig" | "graph" => match need("a function") {
            Ok(t) => Some(t),
            Err(e) => return e,
        },
        "fields" => match need("a struct") {
            Ok(t) => Some(t),
            Err(e) => return e,
        },
        "variants" => match need("an enum") {
            Ok(t) => Some(t),
            Err(e) => return e,
        },
        _ => unreachable!("`is_builtin` names five macros"),
    };
    match name {
        "sig" => {
            let target = target.unwrap();
            let Some(f) = d.find_fn(target) else {
                return miss("function", target, "functions and methods");
            };
            Answer::Text(match mac {
                Some(m) => f.hand_to(m),
                None => f.value(),
            })
        }
        "fns" => {
            let picked: Vec<&FnRec> = d
                .all_fns()
                .into_iter()
                .filter(|f| target.is_none_or(|t| f.marks.iter().any(|(x, _)| x == t)))
                .collect();
            Answer::Text(match mac {
                Some(m) => format!(
                    "[{}]",
                    picked.iter().map(|f| f.hand_to(m)).collect::<Vec<_>>().join(", ")
                ),
                None => list_value("FnSig", picked.iter().map(|f| f.value()).collect()),
            })
        }
        "fields" => {
            let target = target.unwrap();
            let Some(fields) = d.find_struct(target) else {
                return miss("struct", target, "structs");
            };
            Answer::Text(match mac {
                Some(m) => format!(
                    "{m}!({target}; {})",
                    fields
                        .iter()
                        .map(|f| format!("{}: {}", f.name, f.ty))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                None => list_value(
                    "StructField",
                    fields.iter().map(|f| field_value("StructField", f)).collect(),
                ),
            })
        }
        "variants" => {
            let target = target.unwrap();
            let Some(variants) = d.find_enum(target) else {
                return miss("enum", target, "enums");
            };
            Answer::Text(match mac {
                Some(m) => format!(
                    "{m}!({target}; {})",
                    variants.iter().map(|v| v.text.clone()).collect::<Vec<_>>().join(", ")
                ),
                None => list_value(
                    "EnumVariant",
                    variants
                        .iter()
                        .map(|v| {
                            format!(
                                "EnumVariant {{ name = {}, fields = {} }}",
                                quote(&v.name),
                                list_value(
                                    "StructField",
                                    v.fields
                                        .iter()
                                        .map(|f| field_value("StructField", f))
                                        .collect()
                                ),
                            )
                        })
                        .collect(),
                ),
            })
        }
        "graph" => {
            let target = target.unwrap();
            let Some(f) = d.find_fn(target) else {
                return miss("function", target, "functions and methods");
            };
            let text = f.graph_value();
            Answer::Text(match mac {
                Some(m) => format!("{m}!({target}; {text})"),
                None => text,
            })
        }
        _ => unreachable!("`is_builtin` names five macros"),
    }
}

fn named_field(f: &StructField) -> Named {
    Named {
        name: match &f.name {
            FieldKey::Ident(i) => i.lexeme.clone(),
            FieldKey::Int(i) => i.to_string(),
        },
        ty: f.annotation.to_string(),
        inout: false,
    }
}

// ---- `graph!`: a body as dataflow --------------------------------------

/// One node of a lowered body. `op` is `in` (a parameter), `const` (a
/// literal), `call` (a call by name) or `sel` (one part of a call's
/// result). `args` are node indexes: the call's argument wires, or the
/// node a `sel` picks from.
struct GNode {
    op: &'static str,
    name: String,
    args: Vec<usize>,
}

/// The dataflow a body lowers to, or the reason it is not dataflow.
struct Lowered {
    nodes: Vec<GNode>,
    rets: Vec<usize>,
    why: Option<String>,
}

/// A name and the wires it stands for. A call with several results is
/// several wires: a `let (a, b)` or a `.0` picks one with a `sel`.
struct Lower {
    nodes: Vec<GNode>,
    env: Vec<(String, Vec<usize>)>,
}

type LResult<T> = std::result::Result<T, String>;

/// How many results a return type names: `()` none, `(A, B)` two, the
/// rest one.
fn out_names(ret: &str) -> Vec<String> {
    let t = ret.trim();
    if t == "()" {
        return vec![];
    }
    let Some(inner) = t.strip_prefix('(').and_then(|s| s.strip_suffix(')')) else {
        return vec!["out".to_string()];
    };
    let mut depth = 0i32;
    let mut n = 1;
    for c in inner.chars() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            ',' if depth == 0 => n += 1,
            _ => {}
        }
    }
    (0..n).map(|i| i.to_string()).collect()
}

/// A word for the expression kind, for `why`.
fn kind_word(kind: &ExprKind) -> &'static str {
    match kind {
        ExprKind::Absolve(_) => "absolve",
        ExprKind::Access(_) => "access",
        ExprKind::Atom(_) => "atom",
        ExprKind::Block(_) => "block",
        ExprKind::Break(_) => "break",
        ExprKind::Call(_) => "call",
        ExprKind::Closure(_) => "closure",
        ExprKind::Collect(_) => "collect",
        ExprKind::Continue(_) => "continue",
        ExprKind::Equality(_) => "equality",
        ExprKind::Evaluation(_) => "operator",
        ExprKind::For(_) => "for",
        ExprKind::FString(_) => "interpolated string",
        ExprKind::Grouping(_) => "grouping",
        ExprKind::Ident(_) => "name",
        ExprKind::If(_) => "if",
        ExprKind::In(_) => "in",
        ExprKind::Literal(_) => "literal",
        ExprKind::Logical(_) => "logical",
        ExprKind::Loop(_) => "loop",
        ExprKind::Match(_) => "match",
        ExprKind::Coalescence(_) => "else",
        ExprKind::Propagate(_) => "propagate",
        ExprKind::Raise(_) => "raise",
        ExprKind::Range(_) => "range",
        ExprKind::Return(_) => "return",
        ExprKind::Unary(_) => "unary",
        ExprKind::Unwrap(_) => "unwrap",
        ExprKind::While(_) => "while",
        ExprKind::Poison(_) => "poison",
    }
}

impl Lower {
    fn node(&mut self, op: &'static str, name: String, args: Vec<usize>) -> usize {
        self.nodes.push(GNode { op, name, args });
        self.nodes.len() - 1
    }

    fn bind(&mut self, name: &str, wires: Vec<usize>) {
        self.env.push((name.to_string(), wires));
    }

    fn lookup(&self, name: &str) -> Option<Vec<usize>> {
        self.env
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, w)| w.clone())
    }

    /// The wires one expression is made of: one node for most, one per
    /// part for a tuple.
    fn wires(&mut self, e: &Expr) -> LResult<Vec<usize>> {
        match e.kind() {
            ExprKind::Block(b) => self.block(b),
            ExprKind::Grouping(g) => self.wires(&g.inner),
            ExprKind::Ident(i) => self
                .lookup(&i.lexeme)
                .ok_or_else(|| format!("`{}` is a name nothing is bound to", i.lexeme)),
            ExprKind::Literal(Literal::Tuple(xs)) => {
                let mut out = vec![];
                for x in xs {
                    out.extend(self.wires(x)?);
                }
                Ok(out)
            }
            ExprKind::Literal(l) => Ok(vec![self.node("const", l.to_string(), vec![])]),
            ExprKind::Call(c) => {
                let Some(name) = c.left.as_ident().map(|i| i.lexeme.clone()) else {
                    return Err("a call of something that is not a name".to_string());
                };
                if self.lookup(&name).is_some() {
                    return Err(format!("a call of `{name}`, which is a wire here"));
                }
                let mut args = vec![];
                for a in &c.arguments {
                    if let Some(n) = a.name.as_ref() {
                        return Err(format!("a named argument `{}`", n.lexeme));
                    }
                    let w = self.wires(&a.value)?;
                    if w.len() != 1 {
                        return Err(format!("an argument of `{name}` that is not one wire"));
                    }
                    args.push(w[0]);
                }
                Ok(vec![self.node("call", name, args)])
            }
            ExprKind::Access(Access::Dot {
                left,
                right,
                kind: AccessKind::Direct,
            }) => {
                let w = self.wires(left)?;
                if w.len() != 1 {
                    return Err("a `.` on more than one wire".to_string());
                }
                let name = match right.kind() {
                    ExprKind::Ident(i) => i.lexeme.clone(),
                    _ => right.to_string(),
                };
                Ok(vec![self.node("sel", name, vec![w[0]])])
            }
            ExprKind::Access(Access::DoubleColon { .. }) => {
                Ok(vec![self.node("const", e.to_string(), vec![])])
            }
            kind => Err(format!("a {} expression", kind_word(kind))),
        }
    }

    /// A `let` binds names to wires. Any other statement is not
    /// dataflow.
    fn stmt(&mut self, s: &Stmt) -> LResult<()> {
        let StmtKind::Let(l) = s.kind() else {
            return Err("a statement that is not a let".to_string());
        };
        if l.else_branch.is_some() {
            return Err("a `let` with an `else`".to_string());
        }
        let right = self.wires(&l.right)?;
        match l.left.kind() {
            PatKind::Ident(i) => self.bind(&i.lexeme, right),
            PatKind::Tuple(pats) => {
                let mut names = vec![];
                for p in pats {
                    let Some(i) = p.as_ident() else {
                        return Err("a pattern that is not names".to_string());
                    };
                    names.push(i.lexeme.clone());
                }
                if right.len() == names.len() {
                    for (n, w) in names.into_iter().zip(right) {
                        self.bind(&n, vec![w]);
                    }
                } else if right.len() == 1 {
                    for (k, n) in names.into_iter().enumerate() {
                        let sel = self.node("sel", k.to_string(), vec![right[0]]);
                        self.bind(&n, vec![sel]);
                    }
                } else {
                    return Err("a destructure and its value are different widths".to_string());
                }
            }
            _ => return Err("a pattern that is not names".to_string()),
        }
        Ok(())
    }

    fn block(&mut self, b: &Block) -> LResult<Vec<usize>> {
        let scope = self.env.len();
        let out = (|| {
            for s in &b.body {
                self.stmt(s)?;
            }
            match b.yielded_expr.as_ref() {
                Some(e) => self.wires(e),
                None => Ok(vec![]),
            }
        })();
        self.env.truncate(scope);
        out
    }
}

/// Lower the body of `f` to nodes, or the reason it is not dataflow.
fn lower_fn(f: &FnRec) -> Lowered {
    let mut lower = Lower {
        nodes: vec![],
        env: vec![],
    };
    for p in f.params.iter() {
        let n = lower.node("in", p.name.clone(), vec![]);
        lower.env.push((p.name.clone(), vec![n]));
    }
    let out = f.body.as_ref().map(|b| lower.wires(b)).unwrap_or(Ok(vec![]));
    match out {
        Ok(rets) => Lowered {
            nodes: lower.nodes,
            rets,
            why: None,
        },
        Err(why) => Lowered {
            nodes: vec![],
            rets: vec![],
            why: Some(why),
        },
    }
}

impl FnRec {
    /// The `FnGraph` value `graph!(name)` expands to.
    // r[impl mimas.reflect.graph]
    fn graph_value(&self) -> String {
        let lowered = lower_fn(self);
        let nodes: Vec<String> = lowered
            .nodes
            .iter()
            .map(|n| {
                format!(
                    "FnNode {{ op = {}, name = {}, args = {} }}",
                    quote(n.op),
                    quote(&n.name),
                    list_value(
                        "int",
                        n.args.iter().map(|i| i.to_string()).collect()
                    )
                )
            })
            .collect();
        let (opaque, why) = match lowered.why {
            Some(w) => ("true", w),
            None => ("false", String::new()),
        };
        format!(
            "FnGraph {{ name = {}, ins = {}, outs = {}, nodes = {}, rets = {}, opaque = {}, why = {} }}",
            quote(&self.name),
            list_value("str", self.params.iter().map(|p| quote(&p.name)).collect()),
            list_value("str", out_names(&self.ret).iter().map(|n| quote(n)).collect()),
            list_value("FnNode", nodes),
            list_value("int", lowered.rets.iter().map(|i| i.to_string()).collect()),
            opaque,
            quote(&why),
        )
    }
}
