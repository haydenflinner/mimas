use super::*;

impl Parser<'_> {
    pub(super) fn accessor_intro(&self) -> bool {
        if self.nth(1) == TokKind::LeftParenthesis {
            return true;
        }
        let mut depth = 0;
        for (index, token) in self.tokens.iter().enumerate().skip(self.next + 1) {
            match token.kind() {
                TokKind::LeftParenthesis => depth += 1,
                TokKind::RightParenthesis => depth -= 1,
                TokKind::LeftBrace if depth == 0 => {
                    return matches!(
                        self.tokens.get(index + 1).map(|t| t.kind()),
                        Some(TokKind::Ident("get" | "set" | "inout" | "remove"))
                    ) && self
                        .tokens
                        .get(index + 2)
                        .is_some_and(|t| t.kind() == TokKind::LeftBrace);
                }
                TokKind::Eof => break,
                _ => {}
            }
        }
        false
    }

    fn accessor_header(&mut self, member: bool) -> (String, Vec<Binding>, Annotation) {
        self.advance(); // contextual `subscript`
        let name = if self.at(TokKind::LeftParenthesis) {
            String::new()
        } else {
            self.require_ident().lexeme
        };
        self.expect(TokKind::LeftParenthesis);
        let mut parameters = self.list(
            TokKind::RightParenthesis,
            |k| k.is_ident() || k == TokKind::SelfKeyword,
            |p| {
                if p.at(TokKind::SelfKeyword) {
                    let at = p.next_location();
                    p.advance();
                    Binding::new(Ident::new("self", at))
                } else {
                    p.binding()
                }
            },
        );
        if member
            && !parameters
                .first()
                .is_some_and(|p| p.left.as_ident().is_some_and(|i| i.is_receiver()))
        {
            parameters.insert(0, Binding::new(self.ident_at("self", self.next_start())));
        }
        self.expect(TokKind::Arrow);
        let element = self.annotation();
        (name, parameters, element)
    }

    fn skip_accessor_block(&mut self) {
        if !self.expect(TokKind::LeftBrace) {
            return;
        }
        let mut depth = 1;
        while depth > 0 && !self.at(TokKind::Eof) {
            let token = self.advance();
            match token.kind() {
                TokKind::LeftBrace => depth += 1,
                TokKind::RightBrace => depth -= 1,
                _ => {}
            }
        }
    }

    // r[impl mimas.sub.decl]
    // r[impl mimas.sub.arms]
    pub(super) fn accessor_items(&mut self, member: bool, public: bool) -> Vec<Item> {
        let header = self.next;
        let start = self.next_start();
        let (_, _, _) = self.accessor_header(member);
        let mut arms = Vec::new();
        if self.at(TokKind::LeftBrace)
            && !(matches!(
                self.nth(1),
                TokKind::Ident("get" | "set" | "inout" | "remove")
            ) && self.nth(2) == TokKind::LeftBrace)
        {
            arms.push(("get".to_string(), self.next));
            self.skip_accessor_block();
        } else {
            if !self.expect(TokKind::LeftBrace) {
                return vec![];
            }
            while !self.eat(TokKind::RightBrace) && !self.at(TokKind::Eof) {
                let before = self.next;
                let at = self.next_start();
                let arm = self.require_ident().lexeme;
                if !matches!(arm.as_str(), "get" | "set" | "inout" | "remove")
                    || arms.iter().any(|(a, _)| *a == arm)
                {
                    self.error(Misdirection {
                        src: self.src(),
                        at: self.location(at).into(),
                        msg: format!("invalid or repeated subscript arm `{arm}`"),
                        label: "use each of `get`, `set`, `inout`, and `remove` at most once"
                            .into(),
                    });
                }
                arms.push((arm.clone(), self.next));
                self.skip_accessor_block();
                self.eat(TokKind::SemiColon);
                if self.next == before && !self.at(TokKind::Eof) {
                    self.advance();
                }
            }
        }
        let mut end = self.next;
        let explicit_get = arms.iter().any(|(a, _)| a == "get");
        let explicit_set = arms.iter().any(|(a, _)| a == "set");
        let arms_remove = arms.iter().any(|(a, _)| a == "remove");
        let mut items = vec![];
        for arm_index in 0..arms.len() {
            let (arm, body_at) = arms[arm_index].clone();
            let operations: &[&str] = match arm.as_str() {
                "inout" => &["get", "set", "modify", "borrow"],
                "get" => &["get"],
                "set" => &["set"],
                "remove" => &["remove"],
                _ => &[],
            };
            for (ordinal, &op) in operations.iter().enumerate() {
                if arm == "inout"
                    && ((op == "get" && explicit_get) || (op == "set" && explicit_set))
                {
                    continue;
                }
                self.next = header;
                let (name, mut parameters, element) = self.accessor_header(member);
                self.next = body_at;
                let saved = std::mem::replace(&mut self.accessor_yield, arm == "inout");
                let before = self.tokens.len();
                let body = self.block();
                // Body macros splice tokens. Keep later arm offsets and the
                // declaration's end on the expanded stream before reparsing.
                let shift = self.tokens.len() as isize - before as isize;
                end = end.checked_add_signed(shift).unwrap();
                for (_, at) in &mut arms[arm_index + 1..] {
                    *at = at.checked_add_signed(shift).unwrap();
                }
                self.accessor_yield = saved;
                let body = if arm == "inout" {
                    self.accessor_body(body, op, &element, &name, &parameters, arms_remove, start)
                } else {
                    body
                };
                let return_type = if op == "get" {
                    element.clone()
                } else if op == "borrow" {
                    Annotation::Applied(
                        vec![self.ident_at(
                            if arms_remove {
                                "__RemovablePlace"
                            } else {
                                "__Place"
                            },
                            start,
                        )],
                        vec![element.clone()],
                    )
                } else {
                    Annotation::Unit
                };
                if op == "set" || op == "modify" {
                    let mut parameter = Binding::new(
                        self.ident_at(if op == "set" { "new_value" } else { "__change" }, start),
                    );
                    parameter.annotation = Some(if op == "set" {
                        element.clone()
                    } else {
                        Annotation::Function(vec![element.clone()], Box::new(element.clone()))
                    });
                    parameters.push(parameter);
                }
                let body_start = self.tokens[if arm == "inout" { body_at - 1 } else { body_at }]
                    .span()
                    .start();
                let at = Location::new(
                    self.file_id,
                    Span::new(body_start + ordinal, body_start + ordinal + 1),
                );
                let f = Function {
                    name: Ident::new(format!("__sub_{name}_{op}"), at),
                    role: FnRole::Fn,
                    type_params: vec![],
                    parameters,
                    wheres: vec![],
                    return_type: Some(return_type),
                    body,
                };
                items.push(Item::new(f.into(), at, public));
            }
        }
        self.next = end;
        items
    }

    // r[impl mimas.generator.member]
    /// A named member subscript with a plain (non-arm) body: a lazy pull
    /// iterator over the receiver. `subscript live() -> T { yield v; }` in an
    /// impl lowers to the `get` slot of the subscript's method table —
    /// `recv.live` resolves through the usual subscript dispatch to
    /// `recv.__sub_live_get()`, which returns `() -> T?` exactly like the
    /// free form's `live()`.
    pub(super) fn member_generator_item(&mut self, public: bool) -> Item {
        let start = self.next_start();
        self.advance(); // contextual `subscript`
        let (name, type_params, mut parameters, element) =
            self.function_sig(true, Some(""));
        let element = element.unwrap_or_else(|| {
            self.expected("a yielded type after `->`");
            Annotation::Unit
        });
        if matches!(element, Annotation::Unit | Annotation::Option(_)) {
            self.error_here(Misdirection {
                src: self.src(),
                at: self.next_location().into(),
                msg: "a generator must yield a non-null value".into(),
                label: "wrap optional values in a struct so null can mark exhaustion".into(),
            });
        }
        // the receiver rides first, as accessor arms' `self` does.
        if !parameters
            .first()
            .is_some_and(|p| p.left.as_ident().is_some_and(|i| i.is_receiver()))
        {
            parameters.insert(0, Binding::new(self.ident_at("self", self.next_start())));
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
            name: Ident::new(
                format!("__sub_{}_get", name.lexeme),
                name.location,
            ),
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
        let item = Item::new(f.into(), self.location(start), public);
        self.end_item(&item);
        item
    }

    fn fresh_place(&self, e: &Expr) -> Expr {
        let kind = match e.kind() {
            ExprKind::Access(Access::Square { left, key, kind }) => Access::Square {
                left: self.fresh_place(left),
                key: self.fresh_place(key),
                kind: *kind,
            }
            .into(),
            ExprKind::Access(Access::Dot { left, right, kind }) => Access::Dot {
                left: self.fresh_place(left),
                right: self.fresh_place(right),
                kind: *kind,
            }
            .into(),
            kind => kind.clone(),
        };
        Expr::new(kind, e.location())
    }

    fn guarded_place(&self, value: Expr, scoped: bool, start: usize) -> Expr {
        if !scoped {
            return value;
        }
        let guard = self.generator_call(
            "__place_live",
            vec![self.new_expr(self.ident_at("__place_lifetime", start), start)],
            start,
        );
        self.new_expr(
            Block {
                body: vec![self.new_stmt(StmtKind::Expr(guard), start)],
                yielded_expr: Some(value),
            },
            start,
        )
    }

    fn accessor_body(
        &mut self,
        body: Expr,
        op: &str,
        element: &Annotation,
        name: &str,
        parameters: &[Binding],
        removable: bool,
        start: usize,
    ) -> Expr {
        let ExprKind::Block(mut block) = body.into_kind() else {
            return self.poison_expr(start);
        };
        let mut found = None;
        for (i, statement) in block.body.iter().enumerate() {
            if let StmtKind::Expr(e) = statement.kind()
                && let ExprKind::Call(c) = e.kind()
                && c.left
                    .as_ident()
                    .is_some_and(|i| i.lexeme == "__accessor_yield")
            {
                if found.is_some() {
                    self.expected("exactly one top-level place yield in an inout arm");
                }
                found = Some((i, c.arguments[0].value.clone()));
            }
        }
        let Some((index, target)) = found else {
            self.error(Misdirection {
                src: self.src(),
                at: self.location(start).into(),
                msg: "an inout arm needs one top-level `yield place`".into(),
                label: "yield a field or a list slot".into(),
            });
            return self.poison_expr(start);
        };
        let mut prefix = Vec::new();
        // Freeze the final container and key before reading or writing the slot.
        let target = match target.into_kind() {
            ExprKind::Access(Access::Square { left, key, kind }) => {
                let container = self.ident_at("__place_container", start);
                let slot = self.ident_at("__place_key", start);
                prefix.push(self.synthetic_let(container.clone(), left, start));
                prefix.push(self.synthetic_let(slot.clone(), key, start));
                self.new_expr(
                    Access::Square {
                        left: self.new_expr(container, start),
                        key: self.new_expr(slot, start),
                        kind,
                    },
                    start,
                )
            }
            ExprKind::Access(Access::Dot { left, right, kind }) => {
                let container = self.ident_at("__place_container", start);
                prefix.push(self.synthetic_let(container.clone(), left, start));
                self.new_expr(
                    Access::Dot {
                        left: self.new_expr(container, start),
                        right,
                        kind,
                    },
                    start,
                )
            }
            ExprKind::Ident(_) => {
                self.error(Misdirection {
                    src: self.src(),
                    at: self.location(start).into(),
                    msg: "inout binding slots are not implemented".into(),
                    label: "yield a field or a list slot".into(),
                });
                return self.poison_expr(start);
            }
            _ => {
                self.expected("a writable place after `yield`");
                return self.poison_expr(start);
            }
        };
        let validation = self.generator_call(
            "__validate_place_target",
            vec![self.fresh_place(&target)],
            start,
        );
        prefix.push(self.new_stmt(StmtKind::Expr(validation), start));
        let mut tail = block.body.split_off(index);
        tail.remove(0);
        if let Some(last) = block.yielded_expr.take() {
            tail.push(self.new_stmt(StmtKind::Expr(last), start));
        }
        block.body.extend(prefix);
        let scoped = !tail.is_empty() || op == "borrow";
        if scoped {
            if op == "borrow" {
                let lifetime = self.ident_at("__place_lifetime", start);
                block.body.push(self.synthetic_let(
                    lifetime.clone(),
                    self.new_expr(
                        Literal::Array(vec![self.new_expr(Literal::True, start)]),
                        start,
                    ),
                    start,
                ));
                let slot = self.new_expr(
                    Access::Square {
                        left: self.new_expr(lifetime, start),
                        key: self.new_expr(Literal::Int(0), start),
                        kind: AccessKind::Direct,
                    },
                    start,
                );
                tail.insert(
                    0,
                    self.new_stmt(
                        Assignment::new(
                            slot,
                            AssignmentOp::Identity,
                            self.new_expr(Literal::False, start),
                        ),
                        start,
                    ),
                );
            }
            let cleanup = self.new_expr(
                Closure {
                    parameters: vec![],
                    return_type: Some(Annotation::Unit),
                    body: self.new_expr(
                        Block {
                            body: tail,
                            yielded_expr: None,
                        },
                        start,
                    ),
                },
                start,
            );
            let register = self.generator_call("__cleanup_register", vec![cleanup], start);
            block
                .body
                .push(self.new_stmt(StmtKind::Expr(register), start));
        }
        if op == "borrow" {
            let read = self.new_expr(
                Closure {
                    parameters: vec![],
                    body: self.guarded_place(self.fresh_place(&target), scoped, start),
                    return_type: Some(element.clone()),
                },
                start,
            );
            let value = self.ident_at("__place_value", start);
            let mut parameter = Binding::new(value.clone());
            parameter.annotation = Some(element.clone());
            let store = self.new_stmt(
                Assignment::new(target, AssignmentOp::Identity, self.new_expr(value, start)),
                start,
            );
            let write = self.new_expr(
                Closure {
                    parameters: vec![parameter],
                    return_type: Some(Annotation::Unit),
                    body: self.guarded_place(
                        self.new_expr(
                            Block {
                                body: vec![store],
                                yielded_expr: None,
                            },
                            start,
                        ),
                        scoped,
                        start,
                    ),
                },
                start,
            );
            let mut fields = Vec::new();
            fields.push((FieldKey::Ident(self.ident_at("read", start)), read));
            fields.push((FieldKey::Ident(self.ident_at("write", start)), write));
            if removable {
                let member = parameters
                    .first()
                    .is_some_and(|p| p.left.as_ident().is_some_and(|i| i.is_receiver()));
                let method =
                    self.new_expr(self.ident_at(format!("__sub_{name}_remove"), start), start);
                let callee = if member {
                    self.new_expr(
                        Access::Dot {
                            left: self.new_expr(self.ident_at("self", start), start),
                            right: method,
                            kind: AccessKind::Direct,
                        },
                        start,
                    )
                } else {
                    method
                };
                let args = parameters
                    .iter()
                    .skip(usize::from(member))
                    .map(|p| Argument {
                        name: None,
                        value: self.new_expr(p.left.as_ident().unwrap().clone(), start),
                    })
                    .collect();
                let remove = self.new_expr(
                    Closure {
                        parameters: vec![],
                        return_type: Some(Annotation::Unit),
                        body: self.guarded_place(
                            self.new_expr(Call::new(callee, args), start),
                            scoped,
                            start,
                        ),
                    },
                    start,
                );
                fields.push((FieldKey::Ident(self.ident_at("remove", start)), remove));
            }
            let literal = self.new_expr(
                Literal::Struct(StructLiteral {
                    name: self.new_expr(
                        self.ident_at(
                            if removable {
                                "__RemovablePlace"
                            } else {
                                "__Place"
                            },
                            start,
                        ),
                        start,
                    ),
                    fields,
                    base: None,
                }),
                start,
            );
            let promote = self.generator_call("__cleanup_promote", vec![], start);
            block
                .body
                .push(self.new_stmt(StmtKind::Expr(promote), start));
            block.yielded_expr = Some(literal);
        } else if op == "get" {
            let result = self.ident_at("__place_result", start);
            block
                .body
                .push(self.synthetic_let(result.clone(), target, start));
            block.yielded_expr = Some(self.new_expr(result, start));
        } else {
            let value = if op == "set" {
                self.new_expr(self.ident_at("new_value", start), start)
            } else {
                self.generator_call("__change", vec![target.clone()], start)
            };
            block.body.push(self.new_stmt(
                Assignment::new(target, AssignmentOp::Identity, value),
                start,
            ));
            block.yielded_expr = None;
        }
        self.new_expr(block, start)
    }
}
