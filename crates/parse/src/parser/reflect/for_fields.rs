use super::*;

fn compact(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}
fn item_type(ty: &str) -> Option<&str> {
    let inner = ty.strip_prefix('[')?.strip_suffix(']')?;
    let mut depth = 0;
    for (at, c) in inner.char_indices() {
        match c {
            '[' | '(' | '<' => depth += 1,
            ']' | ')' | '>' => depth -= 1,
            ';' if depth == 0 => return Some(&inner[..at]),
            _ => {}
        }
    }
    Some(inner)
}
fn type_base(ty: &str) -> &str {
    ty.split('<').next().unwrap_or(ty)
}
fn type_key(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in type_base(name).chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}
impl Decls {
    fn implements(&self, ty: &str, pact: &str) -> bool {
        let mut todo: Vec<&str> = self
            .impls
            .iter()
            .filter(|(t, _)| t == type_base(ty))
            .map(|(_, p)| p.as_str())
            .collect();
        if self.pacts.iter().any(|(name, _)| name == type_base(ty)) {
            todo.push(type_base(ty));
        }
        let mut seen = HashSet::new();
        while let Some(next) = todo.pop() {
            if next == pact {
                return true;
            }
            if !seen.insert(next) {
                continue;
            }
            if let Some((_, parents)) = self.pacts.iter().find(|(p, _)| p == next) {
                todo.extend(parents.iter().map(String::as_str));
            }
        }
        false
    }
    // r[impl mimas.reflect.for-fields-filter]
    fn matching_fields(
        &self,
        owner: &str,
        filter: &str,
    ) -> std::result::Result<Vec<Named>, (String, String)> {
        let Some((_, fields)) = self.structs.iter().find(|(name, _)| name == owner) else {
            return Err((
                format!("no struct `{owner}` is declared before this `for_fields!`"),
                "put `for_fields!` after its struct declaration".into(),
            ));
        };
        let wanted = item_type(filter).unwrap_or(filter);
        let known = wanted == "_"
            || self.pacts.iter().any(|(p, _)| p == wanted)
            || self.structs.iter().any(|(s, _)| s == type_base(wanted))
            || self.enums.iter().any(|(s, _)| s == wanted)
            || matches!(
                wanted,
                "int" | "float" | "bool" | "str" | "dec" | "index" | "()"
            )
            || fields.iter().any(|f| {
                let ty = compact(&f.ty);
                ty == wanted || item_type(&ty) == Some(wanted)
            });
        if !known {
            return Err((
                format!("unknown `for_fields!` field filter `{filter}`"),
                "use a declared type, pact, or `_`".into(),
            ));
        }
        Ok(fields
            .iter()
            .filter(|f| {
                let ty = compact(&f.ty);
                let actual = if item_type(filter).is_some() {
                    let Some(inner) = item_type(&ty) else {
                        return false;
                    };
                    inner
                } else {
                    ty.as_str()
                };
                wanted == "_" || actual == wanted || self.implements(actual, wanted)
            })
            .cloned()
            .collect())
    }
}

impl<'s> Parser<'s> {
    // r[impl mimas.reflect.for-fields]
    // r[impl mimas.reflect.for-fields-cost]
    // r[impl mimas.reflect.for-fields-once]
    // r[impl mimas.reflect.for-fields-order]
    pub(super) fn expand_for_fields(&mut self, at: usize, spot: Spot) -> bool {
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
                self.tokens[close].span().end(),
            ),
        );
        let fail = |p: &mut Self, msg: String| {
            p.next = close + 1;
            p.macro_error(call, msg, "write `for_fields!(Type, value, [Pact] => |field| { ... })` or omit the value to declare items");
            false
        };
        let mut depth = 0;
        let mut cuts = vec![open + 1];
        let mut arrow = None;
        for i in open + 1..close {
            match self.tokens[i].kind() {
                TokKind::LeftBrace | TokKind::LeftSquare | TokKind::LeftParenthesis => depth += 1,
                TokKind::RightBrace | TokKind::RightSquare | TokKind::RightParenthesis => {
                    depth -= 1
                }
                TokKind::Comma if depth == 0 => cuts.push(i + 1),
                TokKind::FatArrow if depth == 0 => {
                    arrow = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let Some(arrow) = arrow else {
            return fail(self, "`for_fields!` needs `=> |field| { ... }`".into());
        };
        if !(cuts.len() == 2 || cuts.len() == 3 || cuts.len() == 4) {
            return fail(
                self,
                "`for_fields!` takes a type, an optional receiver, and a field filter".into(),
            );
        }
        let owner_end = cuts[1] - 1;
        if owner_end != cuts[0] + 1 {
            return fail(self, "the first `for_fields!` argument is a struct name".into());
        }
        let TokKind::Ident(owner) = self.tokens[cuts[0]].kind() else {
            return fail(self, "the first `for_fields!` argument is a struct name".into());
        };
        let body_name = arrow + 2;
        if self.tokens.get(arrow + 1).map(|t| t.kind()) != Some(TokKind::Pipe)
            || !matches!(
                self.tokens.get(body_name).map(|t| t.kind()),
                Some(TokKind::Ident(_))
            )
            || self.tokens.get(arrow + 3).map(|t| t.kind()) != Some(TokKind::Pipe)
            || self.tokens.get(arrow + 4).map(|t| t.kind()) != Some(TokKind::LeftBrace)
            || self.closer_of(arrow + 4) != Some(close - 1)
        {
            return fail(self, "a `for_fields!` body is `|field| { ... }`".into());
        }
        let TokKind::Ident(binding) = self.tokens[body_name].kind() else {
            unreachable!()
        };
        let filter_start = if cuts.len() >= 3 { cuts[2] } else { cuts[1] };
        let filter_end = if cuts.len() == 4 { cuts[3] - 1 } else { arrow };
        let filter = (filter_start..filter_end)
            .map(|i| self.tokens[i].kind().to_string())
            .collect::<String>();
        let mut fields = match DECLS.with(|d| d.borrow().matching_fields(owner, &filter)) {
            Ok(fields) => fields,
            Err((msg, label)) => {
                self.next = close + 1;
                self.macro_error(call, msg, label);
                return false;
            }
        };
        if cuts.len() == 4 {
            let ordering = &self.tokens[cuts[3]..arrow];
            if ordering.first().map(|t| t.kind()) != Some(TokKind::LeftSquare)
                || ordering.last().map(|t| t.kind()) != Some(TokKind::RightSquare)
            {
                return fail(
                    self,
                    "`for_fields!` ordering is a list of field-name strings".into(),
                );
            }
            let mut order = vec![];
            let mut want_name = true;
            for token in &ordering[1..ordering.len() - 1] {
                match (want_name, token.kind()) {
                    (true, TokKind::String(name)) => {
                        order.push(name);
                        want_name = false;
                    }
                    (false, TokKind::Comma) => want_name = true,
                    _ => {
                        return fail(
                            self,
                            "`for_fields!` ordering is a list of field-name strings".into(),
                        );
                    }
                }
            }
            let mut seen = HashSet::new();
            for name in &order {
                if !seen.insert(*name) || !fields.iter().any(|f| f.name == *name) {
                    return fail(
                        self,
                        format!("unknown or repeated `for_fields!` field in ordering: `{name}`"),
                    );
                }
            }
            fields.sort_by_key(|f| {
                order
                    .iter()
                    .position(|n| *n == f.name)
                    .unwrap_or(order.len())
            });
        }
        let receiver = (cuts.len() >= 3).then(|| cuts[1]..cuts[2] - 1);
        if receiver.is_none() && spot == Spot::Expr {
            return fail(self, "a declaration `for_fields!` belongs at statement position".into());
        }
        let owner_name = format!("__for_fields_owner_{}", NodeId::new().index());
        let outer = self.meta[at].mark;
        let depth = if outer == 0 {
            1
        } else {
            self.hygiene_depth(outer) + 1
        };
        let Some(mark) = self.hygiene_enter(depth) else {
            return fail(self, "`for_fields!` expansion is too deep".into());
        };
        let file_id = self.file_id;
        let file_name = self.file_name.clone();
        let generated = |text: &str| -> Vec<(Tok<TokKind<'s>>, TokMeta)> {
            let text = macros::intern(text);
            let toks: Vec<_> = Lexer::new(text, file_id, file_name.clone())
                .filter(|t| !t.kind().is_comment())
                .collect();
            let meta = token_meta(text, 0, &toks);
            toks.into_iter()
                .zip(meta)
                .map(|(t, m)| {
                    (
                        Tok::new(t.kind, call.into()),
                        TokMeta {
                            mark: if matches!(t.kind, TokKind::Ident(n) if n == owner_name) {
                                0
                            } else {
                                mark
                            },
                            ..m
                        },
                    )
                })
                .collect()
        };
        let mut out = vec![];
        if let Some(receiver) = receiver.clone() {
            out.extend(generated(&format!("{{ let {owner_name}: {owner} =")));
            out.extend(receiver.map(|i| (self.tokens[i].clone(), self.meta[i].clone())));
            out.extend(generated(";"));
        }
        for field in fields {
            let ty = compact(&field.ty);
            let inner = item_type(&ty);
            let key = type_key(inner.unwrap_or(&ty));
            if receiver.is_some() {
                out.extend(generated("{"));
            }
            let property = |property: &str| -> std::result::Result<String, String> {
                Ok(match property {
                    "name" => quote(&field.name),
                    "ident" => field.name.clone(),
                    "type" => field.ty.clone(),
                    "type_name" => quote(&field.ty),
                    "item_type" => inner.ok_or("`item_type` needs a list field")?.to_string(),
                    "item_name" => quote(inner.ok_or("`item_name` needs a list field")?),
                    "key" => quote(&key),
                    "key_ident" => key.clone(),
                    "value" if receiver.is_some() => format!("{owner_name}.{}", field.name),
                    "value" => return Err("`field.value` needs a `for_fields!` receiver".into()),
                    _ => return Err(format!("unknown `for_fields!` field property `{property}`")),
                })
            };
            let mut i = arrow + 5;
            while i < close - 1 {
                if self.tokens[i].kind() == TokKind::Pipe
                    && self.tokens.get(i + 1).map(|t| t.kind()) == Some(TokKind::Ident(binding))
                    && self.tokens.get(i + 2).map(|t| t.kind()) == Some(TokKind::Pipe)
                    && self.tokens.get(i + 3).map(|t| t.kind()) == Some(TokKind::LeftBrace)
                    && let Some(end) = self.closer_of(i + 3)
                {
                    out.extend((i..=end).map(|j| (self.tokens[j].clone(), self.meta[j].clone())));
                    i = end + 1;
                } else if self.tokens[i].kind() == TokKind::Ident(binding)
                    && self.tokens.get(i + 1).map(|t| t.kind()) == Some(TokKind::Dot)
                    && let Some(TokKind::Ident(name)) = self.tokens.get(i + 2).map(|t| t.kind())
                {
                    match property(name) {
                        Ok(value) => out.extend(generated(&value)),
                        Err(msg) => return fail(self, msg),
                    }
                    i += 3;
                } else if let TokKind::FString(content) = self.tokens[i].kind() {
                    let text =
                        match rewrite_fstring(content, binding, &property, file_id, &file_name) {
                            Ok(text) => text,
                            Err(msg) => return fail(self, msg),
                        };
                    let mut meta = self.meta[i].clone();
                    if meta.mark == 0 {
                        meta.mark = mark;
                    }
                    out.push((
                        Tok::new(TokKind::FString(macros::intern(&text)), call.into()),
                        meta,
                    ));
                    i += 1;
                } else {
                    // Keep call-site hygiene on user code, including captured locals.
                    out.push((self.tokens[i].clone(), self.meta[i].clone()));
                    i += 1;
                }
            }
            if receiver.is_some() {
                out.extend(generated("}"));
            }
        }
        if receiver.is_some() {
            out.extend(generated("}"));
        }
        if out.is_empty() {
            out.extend(generated("{}"));
        }
        if let Some((_, first)) = out.first_mut() {
            first.glued = self.meta[at].glued;
            first.newline = self.meta[at].newline;
        }
        self.splice(at, close, out);
        self.next = at;
        true
    }
}

// Only interpolation expressions are rewritten. Literal text, escaped braces,
// string literals inside expressions, and their operator characters stay intact.
fn rewrite_fstring(
    content: &str,
    binding: &str,
    property: &impl Fn(&str) -> std::result::Result<String, String>,
    file_id: FileId,
    file_name: &str,
) -> std::result::Result<String, String> {
    let mut out = String::new();
    let mut at = 0;
    while at < content.len() {
        let c = content[at..].chars().next().unwrap();
        at += c.len_utf8();
        out.push(c);
        if c == '\\' {
            if let Some(next) = content[at..].chars().next() {
                out.push(next);
                at += next.len_utf8();
            }
        } else if matches!(c, '{' | '}') && content[at..].starts_with(c) {
            out.push(c);
            at += c.len_utf8();
        } else if c == '{' {
            let end = crate::lex::interp_end(&content[at..])
                .ok_or("unterminated `for_fields!` interpolation")?;
            let text = &content[at..at + end];
            let tokens: Vec<_> = Lexer::new(text, file_id, file_name.to_string()).collect();
            let mut from = 0;
            let mut i = 0;
            while i < tokens.len() {
                if tokens[i].kind() == TokKind::Ident(binding)
                    && tokens.get(i + 1).map(|t| t.kind()) == Some(TokKind::Dot)
                    && let Some(TokKind::Ident(name)) = tokens.get(i + 2).map(|t| t.kind())
                {
                    out.push_str(&text[from..tokens[i].span().start()]);
                    out.push_str(&property(name)?);
                    from = tokens[i + 2].span().end();
                    i += 3;
                } else if let TokKind::FString(inner) = tokens[i].kind() {
                    out.push_str(&text[from..tokens[i].span().start()]);
                    out.push_str("f\"");
                    out.push_str(&rewrite_fstring(
                        inner, binding, property, file_id, file_name,
                    )?);
                    out.push('"');
                    from = tokens[i].span().end();
                    i += 1;
                } else {
                    i += 1;
                }
            }
            out.push_str(&text[from..]);
            out.push('}');
            at += end + 1;
        }
    }
    Ok(out)
}
