//! Reflection on declarations. A program can ask what it declared.
//!
//! Four built-in macros answer, each in two forms. The plain form
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
//! ```
//!
//! The answer is what the parser has read so far: a declaration is
//! visible after it, in the same file and in every file of the program
//! parsed after it, as a macro is. Types are text, as written. Only
//! top-level functions count, not methods.
//!
//! A mark on a function is an attribute nobody else knows: `#[machine]`.
//! It is legal when some `fns!(machine …)` in the program reads it, so a
//! misspelled `#[tets]` is still an error (see `Parser::unread_marks`).
//!
//! The expansion is source text, lexed and put in place of the call the
//! way a declared macro's template is. The struct types it names
//! (`FnSig`, `FnParam`, `StructField`, `EnumVariant`) are declared in
//! std (`library/src/reflect.mim`).

use super::macros::Spot;
use super::*;
use std::{cell::RefCell, collections::HashSet};

/// A name and its type as written: a parameter, or a field.
#[derive(Clone)]
struct Named {
    name: String,
    ty: String,
}

#[derive(Clone)]
struct FnRec {
    name: String,
    params: Vec<Named>,
    ret: String,
    doc: String,
    marks: Vec<String>,
}

#[derive(Clone)]
struct VariantRec {
    name: String,
    fields: Vec<Named>,
    /// The variant as written: `Num(float)`, `Won { t: Team }`, `Nothing`.
    text: String,
}

#[derive(Default)]
struct Decls {
    /// In declaration order. A later declaration of a name replaces the
    /// earlier one in place.
    fns: Vec<FnRec>,
    structs: Vec<(String, Vec<Named>)>,
    enums: Vec<(String, Vec<VariantRec>)>,
    /// Every mark some `fns!` call has asked for.
    read_marks: HashSet<String>,
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

/// Whether `name!(…)` is one of the reflection macros.
pub(super) fn is_builtin(name: &str) -> bool {
    matches!(name, "sig" | "fns" | "fields" | "variants")
}

/// Whether some `fns!` call has read functions marked `#[name]`.
pub(super) fn mark_is_read(name: &str) -> bool {
    DECLS.with(|d| d.borrow().read_marks.contains(name))
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
    format!("{kind} {{ name = {}, ty = {} }}", quote(&f.name), quote(&f.ty))
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
        let marks: Vec<_> = self.marks.iter().map(|t| quote(t)).collect();
        format!(
            "FnSig {{ name = {}, params = {}, ret = {}, doc = {}, marks = {} }}",
            quote(&self.name),
            list_value("FnParam", params),
            quote(&self.ret),
            quote(&self.doc),
            list_value("str", marks),
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
                let rec = FnRec {
                    name: macros::unhygienic(&f.name).lexeme,
                    params,
                    ret: f
                        .return_type
                        .as_ref()
                        .map_or_else(|| "()".to_string(), |a| a.to_string()),
                    doc,
                    marks: item
                        .attrs()
                        .iter()
                        .map(|a| a.name.lexeme.clone())
                        .filter(|n| n != "test")
                        .collect(),
                };
                DECLS.with(|d| {
                    let fns = &mut d.borrow_mut().fns;
                    match fns.iter_mut().find(|f| f.name == rec.name) {
                        Some(old) => *old = rec,
                        None => fns.push(rec),
                    }
                });
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
            _ => {}
        }
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
        let (target, mac) = match args.as_slice() {
            [] => (None, None),
            [TokKind::Ident(t)] => (Some(*t), None),
            [TokKind::Ident(t), TokKind::FatArrow, TokKind::Ident(m)] => (Some(*t), Some(*m)),
            [TokKind::FatArrow, TokKind::Ident(m)] => (None, Some(*m)),
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
    /// (message, label) to report at the call.
    // r[impl mimas.reflect.sig]
    // r[impl mimas.reflect.fns]
    // r[impl mimas.reflect.fields]
    // r[impl mimas.reflect.variants]
    // r[impl mimas.reflect.hand]
    fn reflect_text(
        &mut self,
        name: &str,
        target: Option<&str>,
        mac: Option<&str>,
    ) -> std::result::Result<String, (String, String)> {
        let need = |what: &str| -> std::result::Result<&str, (String, String)> {
            target.ok_or_else(|| {
                (
                    format!("`{name}!` needs the name of {what}"),
                    format!("write `{name}!(name)`"),
                )
            })
        };
        DECLS.with(|d| {
            let mut d = d.borrow_mut();
            match name {
                "sig" => {
                    let target = need("a function")?;
                    let f = d.fns.iter().find(|f| f.name == target).ok_or_else(|| {
                        (
                            format!("no function `{target}` is declared before this `sig!`"),
                            format!("`sig!` sees the top-level functions above it; `{target}` is not one"),
                        )
                    })?;
                    Ok(match mac {
                        Some(m) => f.hand_to(m),
                        None => f.value(),
                    })
                }
                "fns" => {
                    if let Some(wanted) = target {
                        d.read_marks.insert(wanted.to_string());
                    }
                    let picked: Vec<&FnRec> = d
                        .fns
                        .iter()
                        .filter(|f| target.is_none_or(|t| f.marks.iter().any(|x| x == t)))
                        .collect();
                    Ok(match mac {
                        Some(m) => format!(
                            "[{}]",
                            picked.iter().map(|f| f.hand_to(m)).collect::<Vec<_>>().join(", ")
                        ),
                        None => list_value("FnSig", picked.iter().map(|f| f.value()).collect()),
                    })
                }
                "fields" => {
                    let target = need("a struct")?;
                    let (_, fields) = d.structs.iter().find(|(n, _)| n == target).ok_or_else(|| {
                        (
                            format!("no struct `{target}` is declared before this `fields!`"),
                            format!("`fields!` sees the structs above it; `{target}` is not one"),
                        )
                    })?;
                    Ok(match mac {
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
                    let target = need("an enum")?;
                    let (_, variants) = d.enums.iter().find(|(n, _)| n == target).ok_or_else(|| {
                        (
                            format!("no enum `{target}` is declared before this `variants!`"),
                            format!("`variants!` sees the enums above it; `{target}` is not one"),
                        )
                    })?;
                    Ok(match mac {
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
                _ => unreachable!("`is_builtin` names four macros"),
            }
        })
    }
}

fn named_field(f: &StructField) -> Named {
    Named {
        name: match &f.name {
            FieldKey::Ident(i) => i.lexeme.clone(),
            FieldKey::Int(i) => i.to_string(),
        },
        ty: f.annotation.to_string(),
    }
}
