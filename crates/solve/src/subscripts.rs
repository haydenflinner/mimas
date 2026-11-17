use crate::{
    Result, Solver,
    components::{DecKind, Ty, TyExt},
    traits::Query,
};
use parse::{components::*, *};
use shared::{Located, Location};

#[derive(Clone)]
pub(crate) struct Target {
    receiver: Option<Expr>,
    name: String,
    keys: Vec<Expr>,
    available: Vec<String>,
    at: Location,
}
fn expr(kind: impl Into<ExprKind>, at: Location) -> Expr {
    Expr::new(kind.into(), at)
}
fn ident(name: &str, at: Location) -> Expr {
    expr(Ident::new(name, at), at)
}
fn call(left: Expr, values: Vec<Expr>, at: Location) -> Expr {
    expr(
        Call {
            left,
            arguments: values
                .into_iter()
                .map(|value| Argument { name: None, value })
                .collect(),
        },
        at,
    )
}
fn bind(name: &str, value: Expr, at: Location) -> Stmt {
    Stmt::new(
        Let {
            left: Ident::new(name, at).into(),
            annotation: None,
            right: value,
            else_branch: None,
        }
        .into(),
        NodeId::new(),
        at,
    )
}
impl Target {
    fn has(&self, arm: &str) -> bool {
        self.available.iter().any(|a| a == arm)
    }
    fn invocation(&self, arm: &str, mut extra: Vec<Expr>) -> Expr {
        let method = format!("__sub_{}_{}", self.name, arm);
        let callee = match &self.receiver {
            Some(receiver) => expr(
                Access::Dot {
                    left: receiver.clone(),
                    right: ident(&method, self.at),
                    kind: AccessKind::Direct,
                },
                self.at,
            ),
            None => ident(&method, self.at),
        };
        let mut values = self.keys.clone();
        values.append(&mut extra);
        call(callee, values, self.at)
    }
    fn freeze(&mut self, id: NodeId) -> Vec<Stmt> {
        let mut body = vec![];
        if let Some(receiver) = self.receiver.take() {
            let name = format!("__access_{}_receiver", id.index());
            body.push(bind(&name, receiver, self.at));
            self.receiver = Some(ident(&name, self.at));
        }
        for (i, key) in self.keys.iter_mut().enumerate() {
            let name = format!("__access_{}_key_{i}", id.index());
            body.push(bind(&name, key.clone(), self.at));
            *key = ident(&name, self.at);
        }
        body
    }
}
impl Solver {
    fn accessor_error(&self, at: Location, arm: &str) -> shared::Error {
        crate::errors::SubscriptUse {
            src: self.src(at),
            at: at.into(),
            message: format!("this subscript has no `{arm}` arm for this use"),
        }
        .into()
    }
    fn accessor_arms(&mut self, receiver: &Expr, name: &str) -> Result<Vec<String>> {
        let ty = receiver.query(self)?.normalized(self);
        let Some((aid, _)) = ty.applied() else {
            return Ok(vec![]);
        };
        Ok(["get", "set", "modify", "remove", "borrow"]
            .iter()
            .filter(|arm| {
                self.adts[aid]
                    .impls
                    .contains_key(&format!("__sub_{name}_{arm}"))
            })
            .map(|s| s.to_string())
            .collect())
    }
    pub(crate) fn subscript_target(&mut self, e: &Expr) -> Result<Option<Target>> {
        let (left, keys, _property) = match e.kind() {
            ExprKind::Access(Access::Square {
                left,
                key,
                kind: AccessKind::Direct,
            }) => {
                let keys = match key.kind() {
                    ExprKind::Literal(Literal::Tuple(keys))
                        if key.span().start() == e.span().start() =>
                    {
                        keys.clone()
                    }
                    _ => vec![key.clone()],
                };
                (left, keys, false)
            }
            ExprKind::Access(Access::Dot {
                left,
                right,
                kind: AccessKind::Direct,
            }) => {
                let Some(name) = right.as_ident() else {
                    return Ok(None);
                };
                let available = self.accessor_arms(left, &name.lexeme)?;
                return Ok((!available.is_empty()).then(|| Target {
                    receiver: Some(left.clone()),
                    name: name.lexeme.clone(),
                    keys: vec![],
                    available,
                    at: e.location(),
                }));
            }
            _ => return Ok(None),
        };
        if let ExprKind::Access(Access::Dot {
            left: receiver,
            right,
            ..
        }) = left.kind()
            && let Some(name) = right.as_ident()
        {
            let available = self.accessor_arms(receiver, &name.lexeme)?;
            if !available.is_empty() {
                return Ok(Some(Target {
                    receiver: Some(receiver.clone()),
                    name: name.lexeme.clone(),
                    keys,
                    available,
                    at: e.location(),
                }));
            }
        }
        if let Some(name) = left.as_ident() {
            let available: Vec<String> = ["get", "set", "modify", "remove", "borrow"]
                .iter()
                .filter(|arm| {
                    self.ribs
                        .resolve(&Ident::new(
                            format!("__sub_{}_{}", name.lexeme, arm),
                            e.location(),
                        ))
                        .is_some()
                })
                .map(|s| s.to_string())
                .collect();
            if !available.is_empty() {
                return Ok(Some(Target {
                    receiver: None,
                    name: name.lexeme.clone(),
                    keys,
                    available,
                    at: e.location(),
                }));
            }
        }
        let available = self.accessor_arms(left, "")?;
        Ok((!available.is_empty()).then(|| Target {
            receiver: Some(left.clone()),
            name: String::new(),
            keys,
            available,
            at: e.location(),
        }))
    }
    // r[impl mimas.sub.call]
    pub(crate) fn subscript_read(
        &mut self,
        id: NodeId,
        access: &Access,
        at: Location,
    ) -> Result<Option<Ty>> {
        let source = Expr::with_id(access.clone().into(), id, at);
        let Some(target) = self.subscript_target(&source)? else {
            return Ok(None);
        };
        if !target.has("get") {
            return Err(self.accessor_error(at, "get"));
        }
        let read = target.invocation("get", vec![]);
        let ty = read.query(self)?;
        self.rewrites.insert(id, read);
        Ok(Some(ty))
    }
    // r[impl mimas.sub.select]
    pub(crate) fn subscript_assignment(&mut self, id: NodeId, a: &Assignment) -> Result<bool> {
        self.check_place_write(&a.left)?;
        if self.place_assignment(id, a)? {
            return Ok(true);
        }
        let Some((mut target, root)) = self.assignment_projection(&a.left)? else {
            return Ok(false);
        };
        if let Some((root, _)) = crate::root_and_path(&a.left)
            && let Some(dec) = self.ribs.resolve(root)
            && matches!(self.decs[dec].kind, DecKind::Constant(_))
        {
            return Err(crate::errors::AssignToConst {
                src: self.src(a.left.location()),
                at: a.left.location().into(),
                name: root.lexeme.clone(),
            }
            .into());
        }
        let at = a.left.location();
        if target.has("get") {
            a.left.query(self)?;
        }
        let mut body = target.freeze(id);
        let result = if a.op == AssignmentOp::Identity && root == a.left.id() {
            if !target.has("set") {
                return Err(self.accessor_error(at, "set"));
            }
            target.invocation("set", vec![a.right.clone()])
        } else {
            let name = format!("__access_{}_old", id.index());
            let update = Stmt::new(
                Assignment {
                    left: rebase(&a.left, root, &name),
                    op: a.op,
                    right: a.right.clone(),
                }
                .into(),
                NodeId::new(),
                at,
            );
            if target.has("modify") {
                let closure = expr(
                    Closure {
                        parameters: vec![Binding {
                            left: Ident::new(&name, at).into(),
                            annotation: None,
                            right: None,
                        }],
                        return_type: None,
                        body: expr(
                            Block {
                                body: vec![update],
                                yielded_expr: Some(ident(&name, at)),
                            },
                            at,
                        ),
                    },
                    at,
                );
                target.invocation("modify", vec![closure])
            } else {
                if !target.has("get") || !target.has("set") {
                    return Err(self.accessor_error(at, "inout or get/set"));
                }
                body.push(bind(&name, target.invocation("get", vec![]), at));
                body.push(update);
                target.invocation("set", vec![ident(&name, at)])
            }
        };
        let rewritten = expr(
            Block {
                body,
                yielded_expr: Some(result),
            },
            at,
        );
        rewritten.query(self)?;
        self.rewrites.insert(id, rewritten);
        Ok(true)
    }
    fn assignment_projection(&mut self, e: &Expr) -> Result<Option<(Target, NodeId)>> {
        if let Some(target) = self.subscript_target(e)? {
            return Ok(Some((target, e.id())));
        }
        match e.kind() {
            ExprKind::Access(Access::Dot { left, .. } | Access::Square { left, .. }) => {
                self.assignment_projection(left)
            }
            _ => Ok(None),
        }
    }

    // r[impl mimas.sub.remove]
    // r[impl mimas.place.list-remove]
    pub(crate) fn subscript_removal(
        &mut self,
        id: NodeId,
        c: &Call,
        at: Location,
    ) -> Result<Option<Ty>> {
        if self.synth_removal.contains(&id) {
            return Ok(None);
        }
        let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind() else {
            return Ok(None);
        };
        if !right.as_ident().is_some_and(|i| i.lexeme == "remove") {
            return Ok(None);
        }
        if let Some((raw, removable)) = self.place_binding(left)? {
            if self.raw_places.contains(&left.id()) {
                return Ok(None);
            }
            if !removable {
                return Err(self.accessor_error(at, "remove"));
            }
            self.check_place_write(left)?;
            let rewritten = call(
                expr(
                    Access::Dot {
                        left: raw,
                        right: ident("remove", at),
                        kind: AccessKind::Direct,
                    },
                    at,
                ),
                c.arguments.iter().map(|a| a.value.clone()).collect(),
                at,
            );
            let ty = rewritten.query(self)?;
            self.rewrites.insert(id, rewritten);
            return Ok(Some(ty));
        }
        let Some(target) = self.subscript_target(left)? else {
            if let ExprKind::Access(Access::Square {
                left: collection,
                key,
                kind: AccessKind::Direct,
            }) = left.kind()
                && matches!(collection.query(self)?.normalized(self), Ty::Array(..))
            {
                self.check_place_write(left)?;
                if !c.arguments.is_empty() {
                    return Err(self.accessor_error(at, "zero-argument removal"));
                }
                let rewritten = call(
                    expr(
                        Access::Dot {
                            left: collection.clone(),
                            right: ident("remove", at),
                            kind: AccessKind::Direct,
                        },
                        at,
                    ),
                    vec![key.clone()],
                    at,
                );
                self.synth_removal.insert(rewritten.id());
                let ty = rewritten.query(self)?;
                self.rewrites.insert(id, rewritten);
                return Ok(Some(ty));
            }
            return Ok(None);
        };
        self.check_place_write(left)?;
        if !target.has("remove") {
            return Err(self.accessor_error(at, "remove"));
        }
        if !c.arguments.is_empty() {
            return Err(crate::errors::SubscriptUse {
                src: self.src(at),
                at: at.into(),
                message: "place removal takes no arguments".into(),
            }
            .into());
        }
        let rewritten = target.invocation("remove", vec![]);
        let ty = rewritten.query(self)?;
        self.rewrites.insert(id, rewritten);
        Ok(Some(ty))
    }
}

impl Solver {
    pub(crate) fn is_place_type(&self, ty: &Ty) -> Option<bool> {
        let (aid, _) = ty.applied()?;
        match self.adts[aid].name.as_str() {
            "__Place" => Some(false),
            "__RemovablePlace" => Some(true),
            _ => None,
        }
    }

    /// `(removable, element)` for a place descriptor's `__Place<T>` /
    /// `__RemovablePlace<T>` type. An unremovable place unifies with either
    /// param side; a removable one only lends where removal stays on offer.
    pub(crate) fn place_element(&mut self, ty: &Ty) -> Option<(bool, Ty)> {
        let ty = ty.clone().normalized(self);
        let removable = self.is_place_type(&ty)?;
        let (_, args) = ty.applied()?;
        let element = args.first().cloned().unwrap_or(Ty::Unit);
        Some((removable, element))
    }
    fn raw_place(&mut self, name: &Ident) -> Expr {
        let e = expr(name.clone(), name.location());
        self.raw_places.insert(e.id());
        e
    }
    fn place_binding(&mut self, e: &Expr) -> Result<Option<(Expr, bool)>> {
        if self.raw_places.contains(&e.id()) {
            return Ok(None);
        }
        let Some(name) = e.as_ident() else {
            return Ok(None);
        };
        let Some(dec) = self.ribs.resolve(name) else {
            return Ok(None);
        };
        let ty = Ty::Vid(self.decs[dec].vid).normalized(self);
        Ok(self
            .is_place_type(&ty)
            .map(|removable| (self.raw_place(name), removable)))
    }
    pub(crate) fn place_read(&mut self, id: NodeId, name: &Ident, ty: &Ty) -> Result<Option<Ty>> {
        if self.raw_places.contains(&id)
            || self.is_place_type(&ty.clone().normalized(self)).is_none()
        {
            return Ok(None);
        }
        let raw = self.raw_place(name);
        let rewritten = call(
            expr(
                Access::Dot {
                    left: raw,
                    right: ident("read", name.location()),
                    kind: AccessKind::Direct,
                },
                name.location(),
            ),
            vec![],
            name.location(),
        );
        let value = rewritten.query(self)?;
        self.rewrites.insert(id, rewritten);
        Ok(Some(value))
    }
    // r[impl mimas.place.marker]
    // r[impl mimas.place.let]
    pub(crate) fn borrow_place(
        &mut self,
        id: NodeId,
        c: &Call,
        at: Location,
    ) -> Result<Option<Ty>> {
        if !c
            .left
            .as_ident()
            .is_some_and(|i| i.lexeme == "__borrow_place")
        {
            return Ok(None);
        }
        let [argument] = c.arguments.as_slice() else {
            return Err(self.accessor_error(at, "one place argument"));
        };
        // a `&` has no value of its own -- only a context that can lend or
        // keep a place may hold one. r[impl mimas.place.marker]
        if !self.allowed_borrows.contains(&id) {
            return Err(crate::errors::SubscriptUse {
                src: self.src(at),
                at: at.into(),
                message: "`&` stands only where a place can be lent or kept: an `inout` argument, a `let`'s value, or a `yield`'s place".into(),
            }
            .into());
        }
        self.check_place_write(&argument.value)?;
        argument.value.query(self)?;
        self.place_sources.insert(id, argument.value.clone());
        let rewritten = if let Some((raw, _)) = self.place_binding(&argument.value)? {
            raw
        } else if let Some(target) = self.subscript_target(&argument.value)? {
            if !target.has("borrow") {
                return Err(crate::errors::SubscriptUse {
                    src: self.src(at),
                    at: at.into(),
                    message: "borrowing needs an inout arm ending with `yield place`".into(),
                }
                .into());
            }
            target.invocation("borrow", vec![])
        } else {
            self.builtin_place(id, &argument.value)?
        };
        let ty = rewritten.query(self)?;
        self.rewrites.insert(id, rewritten);
        Ok(Some(ty))
    }
    fn assignment_place_binding(&mut self, e: &Expr) -> Result<Option<(Expr, NodeId)>> {
        if let Some((raw, _)) = self.place_binding(e)? {
            return Ok(Some((raw, e.id())));
        }
        match e.kind() {
            ExprKind::Access(Access::Dot { left, .. } | Access::Square { left, .. }) => {
                self.assignment_place_binding(left)
            }
            _ => Ok(None),
        }
    }

    fn place_assignment(&mut self, id: NodeId, a: &Assignment) -> Result<bool> {
        let Some((raw, root)) = self.assignment_place_binding(&a.left)? else {
            return Ok(false);
        };
        let at = a.left.location();
        a.left.query(self)?;
        let mut body = vec![];
        let value = if a.op == AssignmentOp::Identity && root == a.left.id() {
            a.right.clone()
        } else {
            let name = format!("__place_{}_old", id.index());
            let read = call(
                expr(
                    Access::Dot {
                        left: raw.clone(),
                        right: ident("read", at),
                        kind: AccessKind::Direct,
                    },
                    at,
                ),
                vec![],
                at,
            );
            body.push(bind(&name, read, at));
            body.push(Stmt::new(
                Assignment {
                    left: rebase(&a.left, root, &name),
                    right: a.right.clone(),
                    op: a.op,
                }
                .into(),
                NodeId::new(),
                at,
            ));
            ident(&name, at)
        };
        let write = call(
            expr(
                Access::Dot {
                    left: raw,
                    right: ident("write", at),
                    kind: AccessKind::Direct,
                },
                at,
            ),
            vec![value],
            at,
        );
        let rewritten = expr(
            Block {
                body,
                yielded_expr: Some(write),
            },
            at,
        );
        rewritten.query(self)?;
        self.rewrites.insert(id, rewritten);
        Ok(true)
    }
}

impl Solver {
    fn check_place_write(&self, e: &Expr) -> Result<()> {
        if let Some((root, _)) = crate::root_and_path(e)
            && let Some(dec) = self.ribs.resolve(root)
            && matches!(self.decs[dec].kind, DecKind::Constant(_))
        {
            return Err(crate::errors::AssignToConst {
                src: self.src(e.location()),
                at: e.location().into(),
                name: root.lexeme.clone(),
            }
            .into());
        }
        Ok(())
    }
    fn builtin_place(&mut self, id: NodeId, e: &Expr) -> Result<Expr> {
        let at = e.location();
        let receiver = format!("__slot_{}_container", id.index());
        let key_name = format!("__slot_{}_key", id.index());
        let mut body = vec![];
        let (target, removal) = match e.kind() {
            ExprKind::Access(Access::Square {
                left,
                key,
                kind: AccessKind::Direct,
            }) if matches!(left.query(self)?.normalized(self), Ty::Array(..))
                && key.query(self)?.normalized(self) == Ty::Int =>
            {
                if let Some((root, path)) = crate::root_and_path(left)
                    && let Some(dec) = self.ribs.resolve(root)
                {
                    self.check_iter_guard(dec, &path, at)?;
                }
                body.push(bind(&receiver, left.clone(), at));
                body.push(bind(&key_name, key.clone(), at));
                let target = expr(
                    Access::Square {
                        left: ident(&receiver, at),
                        key: ident(&key_name, at),
                        kind: AccessKind::Direct,
                    },
                    at,
                );
                let sized = matches!(
                    left.query(self)?.normalized(self),
                    Ty::Array(_, shared::Len::Const(_))
                );
                let removal = (!sized).then(|| {
                    call(
                        expr(
                            Access::Dot {
                                left: ident(&receiver, at),
                                right: ident("remove", at),
                                kind: AccessKind::Direct,
                            },
                            at,
                        ),
                        vec![ident(&key_name, at)],
                        at,
                    )
                });
                (target, removal)
            }
            ExprKind::Access(Access::Dot {
                left,
                right,
                kind: AccessKind::Direct,
            }) => {
                e.query(self)?;
                body.push(bind(&receiver, left.clone(), at));
                (
                    expr(
                        Access::Dot {
                            left: ident(&receiver, at),
                            right: right.clone(),
                            kind: AccessKind::Direct,
                        },
                        at,
                    ),
                    None,
                )
            }
            _ => {
                return Err(crate::errors::SubscriptUse {
                    src: self.src(at),
                    at: at.into(),
                    message: "`&` needs a field, a list slot, or a mutable accessor".into(),
                }
                .into());
            }
        };
        fn fresh(e: &Expr) -> Expr {
            let kind = match e.kind() {
                ExprKind::Access(Access::Square { left, key, kind }) => Access::Square {
                    left: fresh(left),
                    key: fresh(key),
                    kind: *kind,
                }
                .into(),
                ExprKind::Access(Access::Dot { left, right, kind }) => Access::Dot {
                    left: fresh(left),
                    right: fresh(right),
                    kind: *kind,
                }
                .into(),
                kind => kind.clone(),
            };
            Expr::new(kind, e.location())
        }
        let read = expr(
            Closure {
                parameters: vec![],
                return_type: None,
                body: fresh(&target),
            },
            at,
        );
        let parameter = Binding {
            left: Ident::new("__slot_value", at).into(),
            annotation: None,
            right: None,
        };
        let store = Stmt::new(
            Assignment {
                left: target,
                op: AssignmentOp::Identity,
                right: ident("__slot_value", at),
            }
            .into(),
            NodeId::new(),
            at,
        );
        let write = expr(
            Closure {
                parameters: vec![parameter],
                return_type: None,
                body: expr(
                    Block {
                        body: vec![store],
                        yielded_expr: None,
                    },
                    at,
                ),
            },
            at,
        );
        let mut fields = vec![
            (FieldKey::Ident(Ident::new("read", at)), read),
            (FieldKey::Ident(Ident::new("write", at)), write),
        ];
        let removable = removal.is_some();
        if let Some(removal) = removal {
            let action = Stmt::new(StmtKind::Expr(removal), NodeId::new(), at);
            fields.push((
                FieldKey::Ident(Ident::new("remove", at)),
                expr(
                    Closure {
                        parameters: vec![],
                        return_type: None,
                        body: expr(
                            Block {
                                body: vec![action],
                                yielded_expr: None,
                            },
                            at,
                        ),
                    },
                    at,
                ),
            ));
        }
        let descriptor = expr(
            Literal::Struct(StructLiteral {
                name: ident(
                    if removable {
                        "__RemovablePlace"
                    } else {
                        "__Place"
                    },
                    at,
                ),
                fields,
                base: None,
            }),
            at,
        );
        Ok(expr(
            Block {
                body,
                yielded_expr: Some(descriptor),
            },
            at,
        ))
    }
}

fn rebase(e: &Expr, root: NodeId, name: &str) -> Expr {
    if e.id() == root {
        return ident(name, e.location());
    }
    let kind = match e.kind() {
        ExprKind::Access(Access::Dot { left, right, kind }) => Access::Dot {
            left: rebase(left, root, name),
            right: right.clone(),
            kind: *kind,
        },
        ExprKind::Access(Access::Square { left, key, kind }) => Access::Square {
            left: rebase(left, root, name),
            key: key.clone(),
            kind: *kind,
        },
        _ => unreachable!("projection assignment follows an access spine"),
    };
    expr(kind, e.location())
}

impl Solver {
    pub(crate) fn validate_place_target(
        &mut self,
        id: NodeId,
        c: &Call,
        at: Location,
    ) -> Result<Option<Ty>> {
        if !c
            .left
            .as_ident()
            .is_some_and(|i| i.lexeme == "__validate_place_target")
        {
            return Ok(None);
        }
        let [argument] = c.arguments.as_slice() else {
            return Err(self.accessor_error(at, "place target"));
        };
        let target = &argument.value;
        let valid = match target.kind() {
            ExprKind::Access(Access::Square { left, key, .. }) => {
                matches!(left.query(self)?.normalized(self), Ty::Array(..))
                    && key.query(self)?.normalized(self) == Ty::Int
            }
            ExprKind::Access(Access::Dot { .. }) => self.subscript_target(target)?.is_none(),
            _ => false,
        };
        if !valid {
            return Err(crate::errors::SubscriptUse {
                src: self.src(at),
                at: at.into(),
                message: "an inout arm must yield a built-in field or a single list slot".into(),
            }
            .into());
        }
        target.query(self)?;
        self.rewrites.insert(id, expr(Literal::Unit, at));
        Ok(Some(Ty::Unit))
    }
}

impl Solver {
    // r[impl mimas.sub.same-name]
    pub(crate) fn check_subscript_names(&self) -> Result<()> {
        for (_, adt) in self.adts.iter() {
            for (key, member) in &adt.impls {
                let Some(rest) = key.strip_prefix("__sub_") else {
                    continue;
                };
                let Some((name, _)) = rest.rsplit_once('_') else {
                    continue;
                };
                if name.is_empty() {
                    continue;
                }
                let field_exists = adt.variants.values().any(|v| match v {
                    crate::components::Variant::Struct(s) => s.fields.contains_key(name),
                    _ => false,
                });
                if field_exists || adt.impls.contains_key(name) {
                    let at = self.decs[member.dec].location;
                    return Err(crate::errors::SubscriptUse {
                        src: self.src(at),
                        at: at.into(),
                        message: format!(
                            "subscript `{name}` conflicts with a field or method of `{}`",
                            adt.name
                        ),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
}
