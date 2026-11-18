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
/// The ident a first-level member write stores into: `x.f`, `x[i]` and `x.1` all
/// yield `x`. `x` itself, `x.f.g` and `x[i].f` yield `None` -- those store into an
/// object `x` only references, which can be shared even when `x` is fresh.
fn member_write_root(e: &Expr) -> Option<&Ident> {
    let ExprKind::Access(access) = e.kind() else {
        return None;
    };
    match access {
        Access::Dot { left, .. } | Access::Square { left, .. } => left.as_ident(),
        _ => None,
    }
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
        self.check_dead_elem_write(a);
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
                // `xs[i].remove()` shifts the iterated collection -- the
                // same freeze `xs.push` gets, at or above the lent place.
                // r[impl mimas.for.inplace-freeze]
                if let Some((root, path)) = crate::root_and_path(collection)
                    && let Some(dec) = self.ribs.resolve(root)
                {
                    self.check_iter_lend(dec, &path, at)?;
                }
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
        if let Some((root, cell)) = self.binding_borrow_cells.remove(&id) {
            let cell = self.node_decs[&cell.id];
            self.binding_borrows.insert(id, (root, cell));
        }
        self.rewrites.insert(id, rewritten);
        Ok(Some(ty))
    }

    /// `yield &e` inside a generator body hands out the place `e` names --
    /// the same descriptor a `&` lend mints, minus `remove`: pulling a row
    /// out from under a suspended walk is the length mutation the loop's
    /// freeze forbids, so the yielded place is `__Place` only.
    /// r[impl mimas.generator.place]
    pub(crate) fn yield_place(
        &mut self,
        id: NodeId,
        c: &Call,
        at: Location,
    ) -> Result<Option<Ty>> {
        if !c
            .left
            .as_ident()
            .is_some_and(|i| i.lexeme == "__yield_place")
        {
            return Ok(None);
        }
        let [argument] = c.arguments.as_slice() else {
            return Err(self.accessor_error(at, "one place argument"));
        };
        // the `&` under a `yield` is claimed by the yield -- it lends
        // nowhere else.
        if !self.allowed_borrows.contains(&id) {
            return Err(crate::errors::SubscriptUse {
                src: self.src(at),
                at: at.into(),
                message: "`yield &` belongs to a generator body -- the `&` marks the place that `yield` hands out".into(),
            }
            .into());
        }
        self.check_place_write(&argument.value)?;
        argument.value.query(self)?;
        self.element_borrows.insert(id);
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
        // a removable place can't be yielded -- `remove` mid-walk would
        // shift the very rows the generator is pulling over.
        if matches!(self.place_element(&ty), Some((true, _))) {
            return Err(crate::errors::SubscriptUse {
                src: self.src(at),
                at: at.into(),
                message: "a place yield can't hand out a removable place -- `remove` mid-walk would shift the rows under it".into(),
            }
            .into());
        }
        // a place lent to this call (`inout` param or a re-lend of one)
        // can't be yielded -- it would outlive the loan.
        // r[impl mimas.inout.escape]
        if let Some((root, _)) = crate::root_and_path(&argument.value)
            && let Some(dec) = self.ribs.resolve(root)
            && self.inout_decs.contains(&dec)
        {
            return Err(crate::errors::InoutEscape {
                src: self.src(at),
                at: at.into(),
                name: root.lexeme.clone(),
                via: "yielded by a generator".to_string(),
            }
            .into());
        }
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
        // `v = x` where `v` is a place-bound `for` variable stays the
        // loop-variable error: the binding is the loop's, and letting the
        // place path swallow it would make `for v in s.live { v = m }` a
        // write-through while `for m in xs { m = m2 }` stays rejected.
        // r[impl mimas.for.elem-bind]
        if let ExprKind::Ident(name) = a.left.kind()
            && let Some(dec) = self.ribs.resolve(name)
            && matches!(self.decs[dec].kind, DecKind::LoopVar)
        {
            return Ok(false);
        }
        let Some((raw, root)) = self.assignment_place_binding(&a.left)? else {
            return Ok(false);
        };
        // a write through a place-typed lhs routes around the Assign
        // solve's guard, but it stales an in-place `for` loan the same
        // way: writing `g` or `g.xs` under `for r in &g`/`&g.xs` replaces
        // the slot the element places resolve through.
        // r[impl mimas.for.inplace-freeze]
        if let Some((root_name, path)) = crate::root_and_path(&a.left)
            && let Some(dec) = self.ribs.resolve(root_name)
            && let Some(guard) = self.iter_guards.iter().find(|g| {
                g.inplace && g.dec == dec && g.path.starts_with(&path)
            })
        {
            return Err(crate::errors::MutateWhileIterating {
                src: self.src(a.left.location()),
                at: a.left.location().into(),
                name: guard.name.clone(),
                iterating: vec![crate::errors::IteratingHere {
                    src: self.src(guard.iterated_at),
                    at: guard.iterated_at.into(),
                    name: guard.name.clone(),
                }],
            }
            .into());
        }
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

    /// `for x in <computed>` binds a fresh value each pass: a first-level member write
    /// (`x.f`, `x[i]`, `x.1`) lands on it and dies with the iteration, so warn. Deeper
    /// writes (`x.f.g`) reach objects the value only references and stay live, as does
    /// every member write over a place iterator (`for x in xs` shares `xs`'s element).
    /// r[impl mimas.for.elem-write]
    fn check_dead_elem_write(&mut self, a: &Assignment) {
        let Some(root) = member_write_root(&a.left) else {
            return;
        };
        let Some(dec) = self.ribs.resolve(root) else {
            return;
        };
        let Some(fresh) = self.ephemeral_elems.get(&dec) else {
            return;
        };
        let (iter, help) = match fresh {
            crate::FreshElem::DictPair(d) => (
                d.clone(),
                format!("write `{d}[{}.0]` to change the entry", root.lexeme),
            ),
            crate::FreshElem::Iterated(i) => (
                i.clone(),
                concat!(
                    "write the source place instead: bind the iterator's result ",
                    "(`let ys = ...`) and iterate it, or loop an index (`for i in 0 .. n` ",
                    "writes `xs[i].f`); `for x in &xs` is planned for in-place iteration"
                )
                .to_string(),
            ),
        };
        self.warnings.push(std::rc::Rc::new(
            crate::errors::DeadElemWrite {
                src: self.src(a.left.location()),
                at: a.left.location().into(),
                name: root.lexeme.clone(),
                iter,
                help,
            }
            .into(),
        ));
    }
    fn builtin_place(&mut self, id: NodeId, e: &Expr) -> Result<Expr> {
        let at = e.location();
        let receiver = format!("__slot_{}_container", id.index());
        let key_name = format!("__slot_{}_key", id.index());
        let gen_name = format!("__slot_{}_gen", id.index());
        let mut body = vec![];
        let mut stamp_container: Option<Expr> = None;
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
                    self.check_iter_lend(dec, &path, at)?;
                }
                body.push(bind(&receiver, left.clone(), at));
                body.push(bind(&key_name, key.clone(), at));
                // the slot's shape at the lend: a later grow/shrink or a
                // wholesale swap of the container faults on the next use
                // instead of writing a shifted or detached slot.
                // r[impl mimas.place.stale]
                body.push(bind(
                    &gen_name,
                    call(
                        expr(
                            Access::Dot {
                                left: ident(&receiver, at),
                                right: ident("len", at),
                                kind: AccessKind::Direct,
                            },
                            at,
                        ),
                        vec![],
                        at,
                    ),
                    at,
                ));
                stamp_container = Some(left.clone());
                let target = expr(
                    Access::Square {
                        left: ident(&receiver, at),
                        key: ident(&key_name, at),
                        kind: AccessKind::Direct,
                    },
                    at,
                );
                // `&` on the iterated place itself (`&xs[i]` inside
                // `for x in xs[i]`) refuses the loan too -- the Square arm
                // below keys on `left`, so check the whole path here.
                if let Some((root, path)) = crate::root_and_path(e)
                    && let Some(dec) = self.ribs.resolve(root)
                {
                    self.check_iter_lend(dec, &path, at)?;
                }
                let sized = matches!(
                    left.query(self)?.normalized(self),
                    Ty::Array(_, shared::Len::Const(_))
                );
                // element places the `for r in &xs` desugar mints never get
                // `remove` -- see `element_borrows`.
                // r[impl mimas.for.inplace.use]
                let removal = (!sized && !self.element_borrows.contains(&id)).then(|| {
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
                // `&s.f` lends `s.f` itself -- the same refusal `&xs[i]`
                // gets when a `for` loop is iterating that exact place.
                if let Some((root, path)) = crate::root_and_path(e)
                    && let Some(dec) = self.ribs.resolve(root)
                {
                    self.check_iter_lend(dec, &path, at)?;
                }
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
            ExprKind::Ident(name) => {
                // `&c` lends a whole binding: the descriptor's closures
                // capture copies, so `c` itself is out of reach. The place
                // reads/writes a one-slot cell instead, and the call that
                // holds the loan stores `__cell[0]` back into `c` when it
                // returns -- a kept `&c` has no call boundary to write back
                // at, so it can't exist. r[impl mimas.inout.binding]
                if self.kept_borrows.contains(&id) {
                    return Err(crate::errors::SubscriptUse {
                        src: self.src(at),
                        at: at.into(),
                        message: "a `&` on a bare binding lends to an `inout` call only -- a kept or yielded place needs a field, a list slot, or a mutable accessor".into(),
                    }
                    .into());
                }
                e.query(self)?;
                let Some((dec, crossed)) = self.ribs.resolve_with_closures(name) else {
                    unreachable!("a `&` operand the solver queried resolves");
                };
                if matches!(self.decs[dec].kind, DecKind::LoopVar) {
                    return Err(crate::errors::SubscriptUse {
                        src: self.src(at),
                        at: at.into(),
                        message: "a `&` can't lend a `for` variable -- its binding is the loop's, not the body's".into(),
                    }
                    .into());
                }
                // a captured `c` is the closure's copy: the write-back would
                // land there and the original's slot would never see it
                if !crossed.is_empty() {
                    return Err(crate::errors::SubscriptUse {
                        src: self.src(at),
                        at: at.into(),
                        message: "a `&` can't lend a captured binding -- the loan can't reach the original's slot".into(),
                    }
                    .into());
                }
                self.check_iter_lend(dec, &[], at)?;
                let cell = Ident::new(&receiver, at);
                body.push(Stmt::new(
                    Let {
                        left: cell.clone().into(),
                        annotation: None,
                        right: expr(Literal::Array(vec![e.clone()]), at),
                        else_branch: None,
                    }
                    .into(),
                    NodeId::new(),
                    at,
                ));
                self.binding_borrow_cells.insert(id, (dec, cell));
                (
                    expr(
                        Access::Square {
                            left: ident(&receiver, at),
                            key: expr(Literal::Int(0), at),
                            kind: AccessKind::Direct,
                        },
                        at,
                    ),
                    None,
                )
            }
            _ => {
                // an `inout self` call mints its receiver's `&` itself, so the
                // user sees no `&` to point at: `get_box().reset()` fails here
                // because a temporary has no slot to write back.
                // r[impl mimas.inout.self]
                let message = if self.implicit_self_borrows.contains(&id) {
                    "`self` is lent to this method's `inout self` -- the receiver needs a field, a list slot, or a named variable to write back".into()
                } else {
                    "`&` needs a field, a list slot, or a mutable accessor".into()
                };
                return Err(crate::errors::SubscriptUse {
                    src: self.src(at),
                    at: at.into(),
                    message,
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
        // `&xs[i]` stamps the captured container's length at the lend: the
        // closures re-check it per use so a grow/shrink of the very array
        // the slot lives in faults instead of hitting a shifted slot. A
        // rebound *binding* detaches the place by spec -- only the
        // captured container's own shape is checked. Built fresh per
        // closure: a cloned stmt shares NodeIds, and the second user's
        // idents then miss the capture pass and read Null.
        // r[impl mimas.place.stale]
        let gen_check = || {
            Stmt::new(
                StmtKind::Expr(call(
                    ident("__seq_fresh", at),
                    vec![ident(&receiver, at), ident(&gen_name, at)],
                    at,
                )),
                NodeId::new(),
                at,
            )
        };
        let read = expr(
            Closure {
                parameters: vec![],
                return_type: None,
                body: if stamp_container.is_some() {
                    expr(
                        Block {
                            body: vec![gen_check()],
                            yielded_expr: Some(fresh(&target)),
                        },
                        at,
                    )
                } else {
                    fresh(&target)
                },
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
                        body: stamp_container
                            .is_some()
                            .then(gen_check)
                            .into_iter()
                            .chain(std::iter::once(store))
                            .collect(),
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
                                body: stamp_container
                                    .is_some()
                                    .then(gen_check)
                                    .into_iter()
                                    .chain(std::iter::once(action))
                                    .collect(),
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
    /// `for r in &xs` — in-place iteration. The `&` claims the iterator the
    /// way an `inout` argument claims its place, but the loan is the loop
    /// itself: each iteration binds `r` to the element's own place, so
    /// `r = v`, `r.f = v` and `r[i] = v` write `xs[i]` directly while
    /// `for r in xs` keeps copying.
    ///
    /// The desugar reuses `let p = &s[i]` wholesale:
    ///
    /// ```text
    /// {
    ///   let __forin_seq_N = <operand>            // evaluated once
    ///   for __forin_i_N in 0..__forin_seq_N.len() {
    ///     let r = &__forin_seq_N[__forin_i_N]    // fresh element place
    ///     <body>
    ///   }
    /// }
    /// ```
    ///
    /// The `&` node minted for the element goes into `element_borrows` so
    /// `builtin_place` builds a `__Place` without `remove` -- pulling an
    /// element out from under the loop is the same length mutation the
    /// root freeze forbids.
    /// r[impl mimas.for.inplace]
    pub(crate) fn solve_for_in_place(
        &mut self,
        id: NodeId,
        for_expr: &For,
        iterator: &Expr,
        location: Location,
    ) -> Result<Ty> {
        let ExprKind::Call(c) = iterator.kind() else {
            unreachable!("for-in-place solve runs only on a `__borrow_place` iterator")
        };
        let at = iterator.location();
        let [argument] = c.arguments.as_slice() else {
            return Err(self.accessor_error(at, "one place to iterate"));
        };
        let operand = &argument.value;

        // the operand names the place the elements live in: `&` on a const
        // fails as its assignment does, and a place binding (`let p = &..`,
        // an `inout` param) is refused below -- its elements would write
        // through fine, but the loan's real root lives behind the place and
        // the freeze can only hold names in this scope.
        // r[impl mimas.for.inplace.operand]
        self.check_place_write(operand)?;
        let iter_ty = operand.query(self)?.normalized(self);
        if self.is_place_type(&iter_ty).is_some() {
            return Err(crate::errors::SubscriptUse {
                src: self.src(at),
                at: at.into(),
                message: "`for .. in &` can't lend through a place binding -- the loop's freeze \
                          can't name the slots behind it"
                    .into(),
            }
            .into());
        }
        // `for v in &s.live` where `live` is a place generator: the pull
        // iterator already hands out `__Place`s, so the `&` affirms the
        // binding's place-ness rather than minting anything -- solve as
        // the plain `for`, whose element binding then holds the yielded
        // place itself. r[impl mimas.for.place-iter]
        if let Ty::Fn(header) = &iter_ty
            && header.parameters.is_empty()
        {
            let yields_place = matches!(
                header.return_ty.as_ref(),
                Ty::Option(element)
                    if self
                        .is_place_type(&element.clone().normalized(self))
                        .is_some()
            );
            if !yields_place {
                return Err(crate::errors::SubscriptUse {
                    src: self.src(at),
                    at: at.into(),
                    message: "`for .. in &` on a pull iterator needs a place generator -- `yield &` in its body is what makes one".into(),
                }
                .into());
            }
            let rewritten = expr(
                For {
                    binding: for_expr.binding.clone(),
                    iterator: operand.clone(),
                    body: for_expr.body.clone(),
                },
                location,
            );
            let ty = rewritten.query(self)?;
            self.rewrites.insert(id, rewritten);
            return Ok(ty);
        }
        if !matches!(iter_ty, Ty::Array(..)) {
            return Err(crate::errors::InvalidIterTarget {
                src: self.src(at),
                at: at.into(),
                ty: iter_ty.to_string(),
            }
            .into());
        }
        // a computed subscript (`s.live`) yields its place through `inout`
        // arms, whose elements aren't built-in slots -- the machinery for
        // lending elements through a yielded place isn't there yet.
        if self.subscript_target(operand)?.is_some() {
            return Err(crate::errors::SubscriptUse {
                src: self.src(at),
                at: at.into(),
                message: "`for .. in &` can't lend elements of a computed subscript's place".into(),
            }
            .into());
        }
        // one place per element: destructuring would read the place apart.
        if !matches!(for_expr.binding.kind(), PatKind::Ident(_)) {
            return Err(crate::errors::SubscriptUse {
                src: self.src(for_expr.binding.location()),
                at: for_expr.binding.location().into(),
                message: "`for .. in &` binds a single name -- a pattern can't lend one place"
                    .into(),
            }
            .into());
        }

        // the iterated root freezes for the loop's extent: same guard a
        // plain `for` pushes (mutating calls, index writes, `&` lends) plus
        // rebinding the name itself, since `r`'s places write through it.
        // r[impl mimas.for.inplace-freeze]
        let guard = crate::root_and_path(operand).and_then(|(root, path)| {
            self.ribs.resolve(root).map(|dec| crate::IterGuard {
                dec,
                path,
                inplace: true,
                iterated_at: at,
                name: operand.to_string(),
            })
        });
        if let Some(g) = &guard {
            // `for r in &p` / `&p.xs` on a *kept* place writes through
            // fine, but the loan's real slots sit behind `p`'s own root,
            // which stays writable and unguarded in this scope. An `inout`
            // param is different: the caller's storage has no name here, so
            // guarding `g` covers every path the loop's writes could take.
            let dec_ty = Ty::Vid(self.decs[g.dec].vid).normalized(self);
            if self.is_place_type(&dec_ty).is_some()
                && !self.inout_decs.contains(&g.dec)
            {
                return Err(crate::errors::SubscriptUse {
                    src: self.src(at),
                    at: at.into(),
                    message: "`for .. in &` can't lend through a place binding -- the loop's \
                              freeze can't name the slots behind it"
                        .into(),
                }
                .into());
            }
            // a second `for .. in &` overlapping this root nests two live
            // loans of element places on it -- reject it like a `&` lend
            // would, above or under the first loan's path alike.
            // r[impl mimas.inout.exclusive]
            self.check_iter_lend(g.dec, &g.path, at)?;
        }

        self.allowed_borrows.insert(iterator.id());
        self.place_sources.insert(iterator.id(), operand.clone());

        let seq_name = format!("__forin_seq_{}", id.index());
        let idx_name = format!("__forin_i_{}", id.index());

        let bind_seq = bind(&seq_name, operand.clone(), at);
        let borrow = expr(
            Call {
                left: ident("__borrow_place", at),
                arguments: vec![Argument {
                    name: None,
                    value: expr(
                        Access::Square {
                            left: ident(&seq_name, at),
                            key: ident(&idx_name, at),
                            kind: AccessKind::Direct,
                        },
                        at,
                    ),
                }],
            },
            at,
        );
        self.element_borrows.insert(borrow.id());
        let bind_elem = Stmt::new(
            Let {
                left: for_expr.binding.clone(),
                annotation: None,
                right: borrow,
                else_branch: None,
            }
            .into(),
            NodeId::new(),
            at,
        );
        let body = match for_expr.body.kind() {
            ExprKind::Block(b) => {
                let mut body = vec![bind_elem];
                body.extend(b.body.iter().cloned());
                expr(
                    Block {
                        body,
                        yielded_expr: b.yielded_expr.clone(),
                    },
                    for_expr.body.location(),
                )
            }
            _ => expr(
                Block {
                    body: vec![bind_elem],
                    yielded_expr: Some(for_expr.body.clone()),
                },
                for_expr.body.location(),
            ),
        };
        let len = call(
            expr(
                Access::Dot {
                    left: ident(&seq_name, at),
                    right: ident("len", at),
                    kind: AccessKind::Direct,
                },
                at,
            ),
            vec![],
            at,
        );
        let range = expr(
            Range {
                start: expr(Literal::Int(0), at),
                end: len,
                inclusive: false,
            },
            at,
        );
        let idx_pat = Pat::new(PatKind::Ident(Ident::new(&idx_name, at)), at);
        let rewritten = expr(
            Block {
                body: vec![bind_seq],
                yielded_expr: Some(expr(
                    For {
                        binding: idx_pat,
                        iterator: range,
                        body,
                    },
                    location,
                )),
            },
            location,
        );

        let guarded = guard.is_some();
        if let Some(guard) = guard {
            self.iter_guards.push(guard);
        }
        let ty = rewritten.query(self)?;
        if guarded {
            self.iter_guards.pop();
        }
        self.rewrites.insert(id, rewritten);
        Ok(ty)
    }

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
