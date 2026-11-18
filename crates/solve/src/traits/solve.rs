use crate::FxHashMap;
use crate::{
    Error, FnRun, LoopRun, Result, Solver, Unification, array,
    components::{
        Adt, AdtFlags, AdtId, DecId, DecKind, Field, Flow, FnHeader, FnParam, Len, Quantification,
        Ty, TyExt, Variant,
    },
    errors::*,
    option,
    resolutions::OpDispatch,
    traits::Query,
};
use parse::{components::Binding, *};
use shared::{Located, Location};

/// Pick which native impl on `aid` named `name` dispatches for a receiver typed as `actual`.
///
/// Returns the unique matching candidate, or `None` if there's no method at all. With a single
/// candidate we skip the work and return it directly so non-overloaded methods keep their existing
/// behavior. With multiple, we freshen each candidate's recv pattern and dry-unify it against the
/// actual receiver: the substitution is computed but discarded (never `commit`ted), so testing a
/// loser only leaks unused vids. The unique success wins; zero matches surface as a type-mismatch
/// against the first candidate, multiple as ambiguity.
fn pick_method_overload(
    solver: &mut Solver,
    aid: AdtId,
    name: &str,
    actual: &Ty,
    location: Location,
) -> Result<Option<Field>> {
    let primary = solver.adts[aid].impls.get(name).cloned();
    let overloads = solver.adts[aid]
        .native_overloads
        .get(name)
        .cloned()
        .unwrap_or_default();
    let mut candidates: Vec<Field> = primary.into_iter().chain(overloads).collect();
    if candidates.len() <= 1 {
        return Ok(candidates.pop());
    }

    let actual_norm = actual.clone().normalized(solver);
    let mut matches: Vec<Field> = Vec::new();
    let mut first_pat_str = String::new();
    for cand in &candidates {
        let recv_pat = solver
            .dec_to_native
            .get(&cand.dec)
            .and_then(|b| b.sig.recv.clone());
        // shouldn't happen for the overload path (only natives go through native_overloads), but
        // if a non-native or recv-less candidate sneaks in we have no recv to test, so keep it.
        let Some(recv_pat) = recv_pat else {
            matches.push(cand.clone());
            continue;
        };
        let mut memo = FxHashMap::default();
        let mut pat = solver.substitute_anons(&recv_pat, &mut memo);
        if first_pat_str.is_empty() {
            first_pat_str = pat.to_string();
        }
        let mut actual_clone = actual_norm.clone();
        if Unification::unify(&mut actual_clone, &mut pat, &*solver).is_ok() {
            matches.push(cand.clone());
        }
    }

    match matches.len() {
        0 => Err(TypeMismatch {
            src: solver.src(location),
            at: location.into(),
            expected: first_pat_str,
            found: actual_norm.to_string(),
        }
        .into()),
        1 => Ok(Some(matches.into_iter().next().unwrap())),
        _ => Err(AmbiguousPactMember {
            src: solver.src(location),
            at: location.into(),
            member: name.to_string(),
        }
        .into()),
    }
}

pub trait Solve {
    /// Evaluates this T and its inner nodes on the provided Session to discover its Ty.
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty>;
}

/// The method names an infix `a <op> b` desugars to — `a.<name>(b)` when the left
/// operand's type implements it, else `b.<rname>(a)` on the right's, the same
/// left-to-right order the native `add_bin_op` dispatch uses. `self` always owns
/// the impl, so `s - v` finds `v.rsub(s)` for `s - self`. The dodged spellings
/// (`rem`/`bxor`/`band`/`bor`/`inv`) keep call-space and op-space separate —
/// `mod`/`xor` could name methods (`fn xor` is an ordinary declaration),
/// `and`/`or`/`not` are shadowable words, and `inv` dodges `not`.
/// `nand`/`nor` deliberately *don't* dodge: the word, the method and the op
/// unify, so `a.nand(b)` and `a nand b` name the same call.
fn op_method_names(op: EvaluationOp) -> (&'static str, &'static str) {
    match op {
        EvaluationOp::Plus => ("add", "radd"),
        EvaluationOp::Minus => ("sub", "rsub"),
        EvaluationOp::Multiply => ("mul", "rmul"),
        EvaluationOp::Divide => ("div", "rdiv"),
        EvaluationOp::Div => ("idiv", "ridiv"),
        EvaluationOp::Modulo => ("rem", "rrem"),
        EvaluationOp::And => ("band", "rband"),
        EvaluationOp::Or => ("bor", "rbor"),
        EvaluationOp::Xor => ("bxor", "rbxor"),
        EvaluationOp::Nand => ("nand", "rnand"),
        EvaluationOp::Nor => ("nor", "rnor"),
        EvaluationOp::BitShiftLeft => ("shl", "rshl"),
        EvaluationOp::BitShiftRight => ("shr", "rshr"),
        // the named operators (r[impl mimas.op.named]): the glyph is the
        // method, and nothing else gives it meaning
        EvaluationOp::Wedge => ("wedge", "rwedge"),
        EvaluationOp::Vee => ("vee", "rvee"),
        EvaluationOp::Dot => ("dot", "rdot"),
        EvaluationOp::Cross => ("cross", "rcross"),
        EvaluationOp::Compose => ("compose", "rcompose"),
        EvaluationOp::Tensor => ("tensor", "rtensor"),
        EvaluationOp::Odot => ("odot", "rodot"),
        EvaluationOp::Boxplus => ("boxplus", "rboxplus"),
        EvaluationOp::Boxminus => ("boxminus", "rboxminus"),
        EvaluationOp::Intersect => ("intersect", "rintersect"),
        EvaluationOp::Union => ("union", "runion"),
        EvaluationOp::Ominus => ("ominus", "rominus"),
        EvaluationOp::Uplus => ("uplus", "ruplus"),
        EvaluationOp::Approx => ("approx", "rapprox"),
    }
}

/// `x <op>= y` — `x = x.<name>(y)` under the same dispatch as the infix form.
pub(crate) fn op_assign_names(op: AssignmentOp) -> Option<(&'static str, &'static str)> {
    Some(match op {
        AssignmentOp::PlusEqual => ("add", "radd"),
        AssignmentOp::MinusEqual => ("sub", "rsub"),
        AssignmentOp::StarEqual => ("mul", "rmul"),
        AssignmentOp::SlashEqual => ("div", "rdiv"),
        AssignmentOp::DivEqual => ("idiv", "ridiv"),
        AssignmentOp::ModEqual => ("rem", "rrem"),
        AssignmentOp::AndEqual => ("band", "rband"),
        AssignmentOp::OrEqual => ("bor", "rbor"),
        AssignmentOp::XorEqual => ("bxor", "rbxor"),
        AssignmentOp::Identity | AssignmentOp::NullCoalescenceEqual => return None,
    })
}

/// `<` `<=` `>` `>=` — `a < b` ⇒ `a.lt(b)` / `b.rlt(a)`. `==`/`!=` never dispatch:
/// instance equality is structural, matching the `add_bin_op` contract.
fn cmp_method_names(op: EqualityOp) -> Option<(&'static str, &'static str)> {
    Some(match op {
        EqualityOp::Less => ("lt", "rlt"),
        EqualityOp::LessOrEqual => ("le", "rle"),
        EqualityOp::Greater => ("gt", "rgt"),
        EqualityOp::GreaterOrEqual => ("ge", "rge"),
        EqualityOp::Equal | EqualityOp::NotEqual => return None,
    })
}

/// `-a`/`+a`/`~a`/`!a` ⇒ `a.neg()`/`a.pos()`/`a.bnot()`/`a.inv()`. `!`'s name is
/// `inv`, not `not`: an ordinary identifier, so overloading `!` doesn't shadow
/// the `not` operator word for the whole program the way a `fn not` would.
/// Types without `inv` keep the bool-only rule — `!true` never dispatches.
fn un_method_name(op: UnaryOp) -> Option<&'static str> {
    match op {
        UnaryOp::Negative => Some("neg"),
        UnaryOp::Positive => Some("pos"),
        UnaryOp::BitwiseNot => Some("bnot"),
        UnaryOp::Not => Some("inv"),
    }
}

/// The shared lookup for operator dispatch: `recv_ty`'s adt (builtin containers
/// included — a native `rem` on `array` would dispatch the same way) offering a
/// method called `name`, specialized to the receiver's instantiation.
fn op_method_sig(
    solver: &mut Solver,
    aid: AdtId,
    name: &str,
    recv_ty: &Ty,
    args: &[Ty],
    location: Location,
) -> Result<Option<(crate::components::DecId, FnHeader)>> {
    let Some(field) = pick_method_overload(solver, aid, name, recv_ty, location)? else {
        return Ok(None);
    };
    solver.check_vis(field.dec, location)?;
    let sig = match solver.dec_to_native.get(&field.dec).map(|b| b.sig.clone()) {
        Some(sig) => solver.instantiate_native(&sig, Some((recv_ty, location)))?,
        None => solver.specialize(&field.ty, aid, args),
    };
    let Ty::Fn(f) = sig else { return Ok(None) };
    Ok(Some((field.dec, f)))
}

// r[impl mimas.op.method]
/// `a <op> b` → `a.<name>(b)` when `a`'s type implements an operator-shaped
/// `<name>` method, else `b.<rname>(a)` when `b`'s does. "Operator-shaped" is
/// `self` plus exactly one operand — a same-named method of another arity just
/// isn't an operator impl and the search continues. Records the hit in
/// `op_decs` for codegen and returns the method's return type. `None` leaves
/// the caller's built-in rules — and their diagnostics — in place.
pub(crate) fn op_dispatch(
    solver: &mut Solver,
    id: NodeId,
    names: (&'static str, &'static str),
    left: &Expr,
    right: &Expr,
    lhs: &Ty,
    rhs: &Ty,
    location: Location,
) -> Result<Option<Ty>> {
    for (recv_ty, name, arg, swap) in [(lhs, names.0, right, false), (rhs, names.1, left, true)] {
        // a primitive's `+` is built in; a named operator has no built-in
        // meaning, so on a primitive it is the method an impl gives it
        // (`impl Approx for float` is what makes `0.1 + 0.2 ≈ 0.3` work)
        if is_primitive(recv_ty) && !is_named_method(names.0) {
            continue;
        }
        let (aid, args) = match recv_ty {
            Ty::Adt(a, args) | Ty::Identity(a, args) => (*a, args.clone()),
            other => match solver.builtin_adt(other) {
                Some(a) => (a, vec![]),
                None => continue,
            },
        };
        // the primary method first, then its operand-typed overloads
        // (r[impl mimas.op.overload]) — the first whose operand the arg fits wins
        let mut candidates: Vec<(crate::components::DecId, FnHeader)> = Vec::new();
        if let Some(primary) = op_method_sig(solver, aid, name, recv_ty, &args, location)? {
            candidates.push(primary);
        }
        for field in solver.adts[aid]
            .overloads
            .get(name)
            .cloned()
            .unwrap_or_default()
        {
            solver.check_vis(field.dec, location)?;
            if let Ty::Fn(f) = solver.specialize(&field.ty, aid, &args) {
                candidates.push((field.dec, f));
            }
        }
        for (dec, f) in candidates {
            // source methods carry `self` in parameters[0]; a native sig bound its
            // receiver during `instantiate_native`, so its params are all operands
            let operand = match f.parameters.as_slice() {
                [_, operand] if f.is_method => operand.ty.clone(),
                [operand] if !f.is_method => operand.ty.clone(),
                _ => continue,
            };
            let ret = f.return_ty.as_ref().clone();
            // an arg that doesn't fit this candidate isn't an error — the
            // search continues (`xs + ys` passes a list to `[T]::add`'s
            // element slot and still has `union`/`radd` to try). Whatever
            // nothing claims falls back to `eval`'s own diagnostics.
            if arg.fulfill_ty(operand, solver).is_err() {
                continue;
            }
            solver.op_decs.insert(id, OpDispatch { dec, swap });
            return Ok(Some(ret));
        }
    }
    Ok(None)
}

/// `xs + ys`/`xs - ys` — the Z-set list forms, dispatched to `[T]`'s
/// `union`/`minus` natives when the rhs is a same-element list (the
/// element shapes `add`/`sub` are `op_dispatch`'s business). `None`
/// leaves `eval`'s diagnostics — `[int] - "a"` still errors.
pub(crate) fn seq_op_dispatch(
    solver: &mut Solver,
    id: NodeId,
    op: EvaluationOp,
    right: &Expr,
    lhs: &Ty,
    rhs: &Ty,
    location: Location,
) -> Result<Option<Ty>> {
    let name = match op {
        EvaluationOp::Plus => "union",
        EvaluationOp::Minus => "minus",
        _ => return Ok(None),
    };
    let (Ty::Array(inner, _), Ty::Array(other, _)) = (lhs, rhs) else {
        return Ok(None);
    };
    if other != inner {
        return Ok(None);
    }
    let Some(aid) = solver.builtin_adt(lhs) else {
        return Ok(None);
    };
    let Some((dec, f)) = op_method_sig(solver, aid, name, lhs, &[], location)? else {
        return Ok(None);
    };
    let Some(operand) = f.parameters.first().map(|p| p.ty.clone()) else {
        return Ok(None);
    };
    right.fulfill_ty(operand, solver)?;
    solver.op_decs.insert(id, OpDispatch { dec, swap: false });
    Ok(Some(Ty::array(inner.as_ref().clone())))
}

/// `int`/`float`/`str`/`bool`: their operators are built in, never dispatched
/// to a method, even when a pact impl gives them one of those names.
/// Whether `name` is the method of a named operator (`∧ ⋅ × … ≈`).
fn is_named_method(name: &str) -> bool {
    matches!(
        name,
        "wedge"
            | "vee"
            | "dot"
            | "cross"
            | "compose"
            | "tensor"
            | "odot"
            | "intersect"
            | "union"
            | "ominus"
            | "uplus"
            | "approx"
    )
}

fn is_primitive(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float | Ty::Str | Ty::Bool)
}

/// `<op> a` → `a.<name>()` — the unary half of `op_dispatch`.
fn unary_dispatch(
    solver: &mut Solver,
    id: NodeId,
    name: &'static str,
    expr: &Expr,
    ty: &Ty,
    location: Location,
) -> Result<Option<Ty>> {
    if let Ty::Pacts(pids) = ty {
        return Ok(pact_unary_dispatch(solver, id, name, ty, pids));
    }
    // a primitive's operators are built in -- `impl Opp for bool` gives `bool`
    // an `inv`, but `!b` stays the bool op (and `!self` inside that `inv`
    // can't recurse)
    if is_primitive(ty) {
        return Ok(None);
    }
    let (aid, args) = match ty {
        Ty::Adt(a, args) | Ty::Identity(a, args) => (*a, args.clone()),
        other => match solver.builtin_adt(other) {
            Some(a) => (a, vec![]),
            None => return Ok(None),
        },
    };
    let Some((dec, f)) = op_method_sig(solver, aid, name, ty, &args, location)? else {
        return Ok(None);
    };
    let nullary = f.parameters.len() == usize::from(f.is_method);
    if !nullary {
        return Ok(None);
    }
    let ret = f.return_ty.as_ref().clone();
    let _ = expr;
    solver.op_decs.insert(id, OpDispatch { dec, swap: false });
    Ok(Some(ret))
}

/// `<op> a` with `a` pact-typed (`!x`, `x: Dichotomy`): the pact declares the
/// operator's method (`fn inv(self) -> Self`), so the operator is a call
/// through the pact — every implementer has it. Recorded in `pact_ops`; emit
/// dispatches at runtime. A `Self` return widens to the bound, as it does for
/// `x.inv()`.
fn pact_unary_dispatch(
    solver: &mut Solver,
    id: NodeId,
    name: &'static str,
    ty: &Ty,
    pids: &[shared::PactId],
) -> Option<Ty> {
    let header = pids.iter().find_map(|pid| {
        solver.pacts[pid]
            .functions
            .get(name)
            .map(|(h, _)| h.clone())
    })?;
    if header.parameters.len() != usize::from(header.is_method) {
        return None;
    }
    let Ty::Fn(f) = solver.instantiate_pact_fn(&header, ty) else {
        return None;
    };
    solver.pact_ops.insert(
        id,
        crate::PactOp {
            pids: pids.to_vec(),
            method: name.to_string(),
            swap: false,
        },
    );
    Some(f.return_ty.as_ref().clone())
}

impl Solve for Access {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        if let Some(ty) = solver.subscript_read(id, self, location)? { return Ok(ty); }

        fn handle_adt_access(
            id: NodeId,
            solver: &mut Solver,
            left: &Expr,
            right: &Ident,
        ) -> Result<Ty> {
            let aid = adt_from_type_path(left, solver)?;
            let adt = &solver.adts[aid];

            // variants live in `adt.variants`, not `adt.impls` -- try them first. an empty variant
            // is an instance value at the access site; a non-empty one is only accessible through a
            // constructor (`Foo::Bar(..)` or `Foo::Bar { .. }`) handled in Call / Literal::Struct.
            if let Some(variant) = adt.variants.get(&right.lexeme) {
                return if variant.is_empty() {
                    // resolve to the variant's first-class dec: the IR matches on
                    // `DeclKind::Variant { layout, .. }` to know what struct_id to build.
                    let dec = variant
                        .dec()
                        .expect("enum variant must have a dec set during hoist");
                    solver.node_decs.insert(id, dec);
                    // `L::E` on a generic enum instantiates its params fresh, the same as a
                    // ctor call would -- each `E` can land on a different `List<T>`
                    let ty = solver.instantiated_adt(aid);
                    solver.note(right, ty.clone(), Some(dec));
                    Ok(ty)
                } else {
                    let ctor = variant
                        .layout()
                        .map(|l| format!("`{}`", variant.ctor(&solver.adts[l].name)));
                    Err(MissingTupleMembers {
                        src: solver.src(right.location()),
                        at: right.location().into(),
                        ctor,
                    })?
                };
            }

            // modules expose their items as fields, so fall back to fields when the ADT was
            // declared as a module
            let field = adt
                .impls
                .get(&right.lexeme)
                .or_else(|| {
                    if adt.flags.contains(AdtFlags::IS_MODULE) {
                        adt.as_struct().fields.get(&right.lexeme)
                    } else {
                        None
                    }
                })
                .cloned()
                .ok_or_else(|| {
                    Error::from(FieldNotFound {
                        src: solver.src(right.location()),
                        at: right.location().into(),
                        field_name: right.lexeme.clone(),
                    })
                })?;

            // a path that stops on a module hasn't named a value yet
            if solver.non_value != Some(id)
                && matches!(
                    field.ty.clone().normalized(solver),
                    Ty::Adt(inner, _) if solver.adts[&inner]
                        .flags
                        .intersects(AdtFlags::IS_MODULE | AdtFlags::IS_BUILTIN)
                )
            {
                Err(NotAValue {
                    src: solver.src(right.location()),
                    at: right.location().into(),
                    name: right.lexeme.clone(),
                })?
            }

            let dec = field.dec;
            // `mod::Pact` names a pact, not a value -- the same hole a bare
            // `P` hits through `Ident::solve`, one path further along.
            if solver.non_value != Some(id)
                && matches!(solver.decs[dec].kind, DecKind::Pact(_))
            {
                Err(crate::errors::PactIsNotAValue {
                    src: solver.src(right.location()),
                    at: right.location().into(),
                })?;
            }
            solver.check_vis(dec, right.location())?;
            solver.node_decs.insert(id, dec);
            // module-nested natives need fresh type vars per call site, same as free natives.
            // generic items (fns with `<T>` params, generic adt assoc items) do too. module
            // fields store a bare `Ty::Vid` (see `sync_module_adt`), so normalize first to
            // reach any `Ty::Param`s underneath.
            let ty = if let Some(binding) = solver.dec_to_native.get(&dec) {
                let sig = binding.sig.clone();
                solver.instantiate_native(&sig, None)?
            } else {
                let stored = field.ty.normalized(solver);
                let prev = solver.obligation_at.replace(right.location());
                let ty = solver.instantiate_params(&stored);
                solver.obligation_at = prev;
                ty
            };
            solver.note(right, ty.clone(), Some(dec));
            Ok(ty)
        }
        // can a plain `[i]` / `.x` ride the short-circuit on `e`, or does `e` leave a *fresh*
        // option that needs its own `?`? the mental model is "one `?` per option you reach
        // through": a `?` rides cleanly through accesses that introduce no new option
        // (array index, field), but a dict index is a fresh option (the key might miss), so
        // the next access spends its own `?`. that's why `f["x"]?["y"][0]` is rejected --
        // the `["y"]` miss isn't acknowledged -- while `a[0]["foo"]?[0][0]` is fine, the
        // trailing accesses being array drills. a dict never rides, even as `?["y"]`: the
        // `?` consumes the *incoming* option but the key's own miss is still unhandled. a
        // plain index of a fresh option just falls through to the normal path below and
        // fails to unify (option vs array/dict), so no dedicated error is needed. only the solver
        // needs this; codegen still null-checks anything `?`-tainted, which stays correct for what
        // the solver lets through.
        fn rides(solver: &mut Solver, e: &Expr) -> bool {
            match e.kind() {
                ExprKind::Access(Access::Square { left, key, kind }) => {
                    let dict = key
                        .query(solver)
                        .is_ok_and(|t| matches!(t.normalized(solver), Ty::Str));
                    !dict && (*kind == AccessKind::Option || rides(solver, left))
                }
                ExprKind::Access(Access::Dot { left, kind, .. }) => {
                    *kind == AccessKind::Option || rides(solver, left)
                }
                _ => false,
            }
        }
        match self {
            Access::Square { left, key, kind } => {
                let k = key.query(solver)?;
                let key_src = solver.src(location);

                // strings index by char position and never unify against the array/dict shape
                // below; a non-int key gets its own error so the shape's fresh vid never leaks
                // into a user-facing message
                let receiver = left.query(solver)?.normalized(solver);
                let optional = *kind == AccessKind::Option || rides(solver, left);
                if receiver == Ty::Str || (optional && receiver == option!(Ty::Str)) {
                    if k.normalized(solver) != Ty::Int {
                        return Err(StringIndexing {
                            src: solver.src(key.location()),
                            at: key.location().into(),
                        }
                        .into());
                    }
                    return Ok(if optional { option!(Ty::Str) } else { Ty::Str });
                }

                // `xs[idxs]` -- a list of ints gathers: `b[[0, 4, 8]]` is `[b[0], b[4], b[8]]`,
                // and a nested index list gathers a nested result (`b[LINES]` is `[[T]]`)
                let mut depth = 0;
                let mut key_inner = k.clone().normalized(solver);
                while let Ty::Array(inner, _) = key_inner {
                    depth += 1;
                    key_inner = inner.as_ref().clone().normalized(solver);
                }
                if depth > 0 && key_inner == Ty::Int && receiver != Ty::Str {
                    let value = solver.vid();
                    let optional = *kind == AccessKind::Option || rides(solver, left);
                    let seq = Ty::array(Ty::from(value));
                    left.fulfill_ty(if optional { option!(seq) } else { seq }, solver)?;
                    let mut out = Ty::from(value).normalized(solver);
                    for _ in 0..depth {
                        out = Ty::array(out);
                    }
                    return Ok(if optional { option!(out) } else { out });
                }

                let shape = |inner: Ty| -> Result<Ty> {
                    match &k {
                        Ty::Int => Ok(Ty::array(inner)),
                        Ty::Str => Ok(Ty::Dict(Box::new(inner))),
                        _ => Err(InvalidAccess {
                            src: key_src.clone(),
                            at: location.into(),
                            ty: k.to_string(),
                        })?,
                    }
                };
                if *kind == AccessKind::Option || rides(solver, left) {
                    let dummy_var = solver.vid();
                    let ty = option!(shape(Ty::Vid(dummy_var))?);
                    left.fulfill_ty(ty, solver)?;
                    Ok(option!(Ty::Vid(dummy_var).normalized(solver)))
                } else {
                    let value = solver.vid();
                    let ty = shape(Ty::from(value))?;
                    let is_dict = matches!(ty, Ty::Dict(_));
                    left.fulfill_ty(ty, solver)?;

                    let out = if is_dict {
                        option!(Ty::from(value))
                    } else {
                        Ty::from(value)
                    };

                    Ok(out)
                }
            }
            Access::Dot { left, right, kind } => {
                let opt = kind == &AccessKind::Option || rides(solver, left);
                let lhs_outer = left.query(solver)?;
                let lhs = if opt {
                    match lhs_outer {
                        Ty::Option(inner) => inner.as_ref().clone(),
                        ref l => Err(InvalidUnwrap {
                            src: solver.src(left.location()),
                            at: left.location().into(),
                            ty: l.to_string(),
                        })?,
                    }
                } else {
                    lhs_outer
                };

                let result = match &lhs {
                    Ty::Tuple(members) => {
                        // `pt.0`..`pt.3` index by slot; the axis names `x` `y`
                        // `z` `t` spell the same slots, so a `(px↔, px↕)`
                        // point reads `pt.x`/`pt.y` the way it means.
                        // r[impl mimas.tuple.axis]
                        let i = match right.kind() {
                            ExprKind::Literal(Literal::Int(i)) => *i as usize,
                            ExprKind::Ident(id) => match id.lexeme.as_str() {
                                "x" => 0,
                                "y" => 1,
                                "z" => 2,
                                "t" => 3,
                                _ => Err(InvalidTupleIndex {
                                    src: solver.src(right.location()),
                                    at: right.location().into(),
                                })?,
                            },
                            _ => Err(InvalidTupleIndex {
                                src: solver.src(right.location()),
                                at: right.location().into(),
                            })?,
                        };
                        members.get(i).cloned().ok_or_else(|| {
                            Error::from(OutOfBoundsTupleRead {
                                src: solver.src(location),
                                at: location.into(),
                                tried_index: i,
                                max_index: members.len(),
                                ty: Ty::Tuple(members.clone()).to_string(),
                            })
                        })?
                    }
                    // a bound annotation, `Self` in a default body, or a bounded type
                    // param (`x: T` under `<T: Draw>`): members come from the pacts
                    // r[impl mimas.generic.bound]
                    Ty::Pacts(_) | Ty::Skolem(_) | Ty::Param(_)
                        if solver.pacts_of(&lhs).is_some() =>
                    {
                        if let ExprKind::Literal(Literal::Int(_)) = right.kind() {
                            return Err(NotATuple {
                                src: solver.src(location),
                                at: location.into(),
                            }
                            .into());
                        }

                        // saftey: parser rejects any dot that is not ident or int
                        let name = &right.as_ident().unwrap().lexeme;
                        let mut con = None;
                        let mut fun = None;
                        let mut matches = 0;
                        // `self` in a default body is one implementer of its pact, not yet known
                        // which, so it reaches members the same way a bound does
                        for pid in &solver.pacts_of(&lhs).expect("narrowed by the outer arm") {
                            let pact = &solver.pacts[pid];
                            if let Some(c) = pact.constants.get(name) {
                                con = Some(c.clone());
                                matches += 1;
                            } else if let Some((header, _)) = pact.functions.get(name) {
                                fun = Some((*pid, header.clone()));
                                matches += 1;
                            }
                        }

                        if matches > 1 {
                            return Err(AmbiguousPactMember {
                                src: solver.src(right.location()),
                                at: right.location().into(),
                                member: name.clone(),
                            }
                            .into());
                        }
                        if con.is_some() {
                            // methods dispatch through `pact_impls`, but constants have no
                            // runtime dispatch -- every impl declares its own value and the
                            // receiver's concrete type isn't known here
                            return Err(PactConstantNotDispatchable {
                                src: solver.src(right.location()),
                                at: right.location().into(),
                                member: name.clone(),
                                via: lhs.to_string(),
                            }
                            .into());
                        } else if let Some((pid, header)) = fun {
                            // `Self` in a parameter is the receiver's concrete type (Skolem). a
                            // bound doesn't pin that down, so any implementer would satisfy it and
                            // the callee would read another type's fields by slot. only a concrete
                            // receiver may call it -- or `self` in a default body, where every
                            // `Self` is the same (unknown) type.
                            if let Ty::Pacts(_) = &lhs
                                && let Some(param) = header
                                    .parameters
                                    .iter()
                                    .skip(usize::from(header.is_method))
                                    .find(|p| p.ty.contains_skolem())
                            {
                                let param = match &param.name {
                                    Some(name) => format!("{name}: {}", param.ty),
                                    None => param.ty.to_string(),
                                };
                                return Err(PactMethodNotDispatchable {
                                    src: solver.src(right.location()),
                                    at: right.location().into(),
                                    member: name.clone(),
                                    param,
                                    via: lhs.to_string(),
                                }
                                .into());
                            }

                            if solver.non_value != Some(id) {
                                return Err(MethodIsNotAValue {
                                    src: solver.src(right.location()),
                                    at: right.location().into(),
                                }
                                .into());
                            }

                            // freshen so dispatch can't bind the pact header's shared self-slot
                            // vid, which would pollute every later use
                            // and impl signature check. through a bound, a `Self` return is some
                            // implementer of it, so it widens to the bound and the skolem never
                            // escapes its pact; on `self` in a default body it stays `Self`.
                            let ty = solver.instantiate_pact_fn(&header, &lhs);
                            let member = solver.pact_members.get(&(pid, name.clone())).copied();
                            solver.note(right.as_ident().unwrap(), ty.clone(), member);
                            ty
                        } else {
                            return Err(FieldNotFound {
                                src: solver.src(location),
                                at: location.into(),
                                field_name: name.clone(),
                            }
                            .into());
                        }
                    }
                    other => {
                        // adts/identity resolve directly. primitives and collections route
                        // through their library adt, which is where their methods live. `args`
                        // are the receiver's actual type args -- a generic struct's fields and a
                        // generic impl's methods are stored in terms of `Ty::Param`s and get
                        // `specialize`d onto them (`p.first` on `Pair<int, str>` gives `int`).
                        let (aid, args) = match other {
                            Ty::Adt(aid, args) | Ty::Identity(aid, args) => (*aid, args.clone()),
                            // an unbounded param is any type at all -- nothing to look
                            // a member up on. the fix is a bound, so say so.
                            Ty::Param(_) if right.as_ident().is_some() => {
                                return Err(ParamHasNoBound {
                                    src: solver.src(left.location()),
                                    at: left.location().into(),
                                    param: other.to_string(),
                                    member: right.as_ident().unwrap().lexeme.clone(),
                                }
                                .into());
                            }
                            l => (
                                solver.builtin_adt(l).ok_or_else(|| {
                                    Error::from(TypeHasNoFields {
                                        src: solver.src(left.location()),
                                        at: left.location().into(),
                                        ty: l.to_string(),
                                    })
                                })?,
                                vec![],
                            ),
                        };

                        if let ExprKind::Literal(Literal::Int(i)) = right.kind() {
                            let i = *i as usize;
                            let adt = &solver.adts[aid];
                            let tup_info = match adt.as_singular() {
                                Some(Variant::Tuple(t)) => Some((
                                    t.members.get(i).cloned(),
                                    t.members.len(),
                                    adt.name.clone(),
                                )),
                                _ => None,
                            };
                            match tup_info {
                                Some((Some(t), _, _)) => solver.specialize(&t, aid, &args),
                                Some((None, max, name)) => Err(OutOfBoundsTupleRead {
                                    src: solver.src(location),
                                    at: location.into(),
                                    tried_index: i,
                                    max_index: max,
                                    ty: name,
                                })?,
                                None => Err(NotATuple {
                                    src: solver.src(location),
                                    at: location.into(),
                                })?,
                            }
                        } else {
                            // saftey: parser rejects any dot that is not ident or int
                            let name = &right.as_ident().unwrap().lexeme;

                            // fields win over impls. method dispatch happens @ call
                            let field = match solver.adts[aid].as_singular() {
                                Some(Variant::Struct(s)) => {
                                    s.fields.get(name).map(|f| (f.ty.clone(), f.dec))
                                }
                                _ => None,
                            };

                            if let Some((ty, dec)) = field {
                                let ty = solver.specialize(&ty, aid, &args);
                                solver.note(right.as_ident().unwrap(), ty.clone(), Some(dec));
                                ty
                            } else {
                                let impl_field = pick_method_overload(
                                    solver,
                                    aid,
                                    name,
                                    &lhs,
                                    right.location(),
                                )?;
                                let Some(impl_field) = impl_field else {
                                    Err(FieldNotFound {
                                        src: solver.src(location),
                                        at: location.into(),
                                        field_name: name.clone(),
                                    })?
                                };
                                let dec = impl_field.dec;
                                solver.check_vis(dec, right.location())?;
                                solver.node_decs.insert(id, dec);
                                let member = if let Some(binding) = solver.dec_to_native.get(&dec) {
                                    let sig = binding.sig.clone();
                                    solver
                                        .instantiate_native(&sig, Some((&lhs, left.location())))?
                                } else {
                                    solver.specialize(&impl_field.ty, aid, &args)
                                };
                                // an associated const reads fine through a value, but a
                                // method only exists as a call's callee
                                if solver.non_value != Some(id)
                                    && matches!(member.clone().normalized(solver), Ty::Fn(_))
                                {
                                    return Err(MethodIsNotAValue {
                                        src: solver.src(right.location()),
                                        at: right.location().into(),
                                    }
                                    .into());
                                }
                                solver.note(right.as_ident().unwrap(), member.clone(), Some(dec));
                                member
                            }
                        }
                    }
                };

                if opt && !matches!(result, Ty::Fn(_)) {
                    Ok(Ty::Option(Box::new(result)))
                } else {
                    Ok(result)
                }
            }

            Access::Identity { .. } => {
                unreachable!("Access::Identity is never produced by the parser")
            }
            Access::DoubleColon { left, right } => {
                // `Self::member` inside a pact default resolves against the contextual pact.
                // There's no explicit `Pact::member` form -- concrete access goes
                // through the adt name.
                if let ExprKind::Ident(ident) = left.kind()
                    && ident.is_identity()
                    && let Some(pid) = solver.pact_self
                {
                    let pact = &solver.pacts[pid];
                    if pact.constants.contains_key(&right.lexeme) {
                        // a default body is compiled once and shared by every impl, so `Self`
                        // is still the abstract pact here and there's no single constant to
                        // point at -- same limitation as reaching one through a value
                        Err(PactConstantNotDispatchable {
                            src: solver.src(right.location),
                            at: right.location.into(),
                            member: right.lexeme.clone(),
                            via: "Self".into(),
                        }
                        .into())
                    } else if let Some((header, _)) = pact.functions.get(&right.lexeme) {
                        let header = header.clone();
                        Ok(solver.instantiate_pact_fn(&header, &Ty::Skolem(pid)))
                    } else {
                        Err(FieldNotFound {
                            src: solver.src(right.location),
                            at: right.location.into(),
                            field_name: right.lexeme.clone(),
                        }
                        .into())
                    }
                } else {
                    handle_adt_access(id, solver, left, right)
                }
            }
        }
    }
}

impl Solve for Block {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        solver.ribs.push_block();
        solver.hoist_block_consts(&self.body);
        let mut diverged = false;
        for stmt in &self.body {
            solver.visit_stmt(stmt)?;
            if let StmtKind::Expr(e) = stmt.kind()
                && matches!(e.query(solver)?, Ty::Never)
            {
                diverged = true;
            }
        }
        let yielded = self
            .yielded_expr
            .as_ref()
            .map(|v| v.query(solver))
            .transpose()?;
        let ty = if diverged {
            Ty::Never
        } else {
            yielded.unwrap_or(Ty::Unit)
        };
        solver.ribs.pop();
        Ok(ty)
    }
}

impl Solve for Break {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let ty = self
            .value
            .as_ref()
            .map_or(Ok(Ty::Unit), |v| v.query(solver))?;
        let src = solver.src(location);
        solver.with_loop_mut(
            || {
                BreakOutsideLoop {
                    src: src.clone(),
                    at: location.into(),
                }
                .into()
            },
            |solver, loop_data| {
                solver
                    .regsiter_loop_control_flow_ty(ty, &mut loop_data.break_ty)
                    .map_err(|e| e.into_type_mismatch(solver, location))
            },
        )?;
        Ok(Ty::Never)
    }
}

impl Solve for Call {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        if let Some(ty) = solver.validate_place_target(id, self, location)? { return Ok(ty); }
        if let Some(ty) = solver.borrow_place(id, self, location)? { return Ok(ty); }
        if let Some(ty) = solver.subscript_removal(id, self, location)? { return Ok(ty); }

        let adt = match self.left.kind() {
            ExprKind::Access(Access::DoubleColon { left, right }) => {
                let adt = match adt_from_type_path(left, solver) {
                    Ok(adt) => adt,
                    Err(_) => {
                        // an unknown head (`missing::f()`) reports itself as undefined
                        let ty = left.query(solver)?.to_string();
                        return Err(NotAStruct {
                            src: solver.src(location),
                            at: location.into(),
                            ty,
                        }
                        .into());
                    }
                };

                let lock = &solver.adts[adt];
                if lock.flags.contains(AdtFlags::IS_MODULE) {
                    // field.ty is the bare vid for a module item, so normalize to chase it to the
                    // actual adt id before we treat it as a constructor target.
                    let field_ty = lock
                        .as_struct()
                        .fields
                        .get(&right.lexeme)
                        .map(|f| f.ty.clone());
                    field_ty.and_then(|ty| {
                        let layout = *ty.normalized(solver).as_adt()?;
                        Some((
                            layout,
                            solver.adts[layout].as_singular().unwrap().clone(),
                            layout,
                        ))
                    })
                } else {
                    let variant = lock.variants.get(&right.lexeme).cloned();
                    if variant.is_none()
                        && right.lexeme == "default"
                        && !lock.impls.contains_key(&right.lexeme)
                    {
                        // `S::default()` — every struct implicitly carries a
                        // zero-arg constructor filling each member with its
                        // type's default. A variant or impl member named
                        // `default` shadows it.
                        return default_ctor(self, id, right, adt, solver);
                    }
                    if let Some(variant) = &variant {
                        solver.note(right, Ty::adt(adt), variant.dec());
                    }
                    variant.map(|v| (v.layout().unwrap(), v, adt))
                }
            }
            ExprKind::Ident(ident) => match ident.query(solver)? {
                Ty::Adt(aid, _) | Ty::Identity(aid, _) => {
                    let (is_enum, name, singular, variants) = {
                        let adt = &solver.adts[aid];
                        let variants = if adt.flags.contains(AdtFlags::IS_ENUM) {
                            let shown = adt
                                .variants
                                .iter()
                                .take(4)
                                .map(|(n, v)| {
                                    format!("`{}`", v.ctor(&format!("{}::{n}", adt.name)))
                                })
                                .collect::<Vec<_>>()
                                .join(", ");
                            if adt.variants.len() > 4 {
                                format!("{shown}, …")
                            } else {
                                shown
                            }
                        } else {
                            String::new()
                        };
                        (
                            adt.flags.contains(AdtFlags::IS_ENUM),
                            adt.name.clone(),
                            adt.as_singular().cloned(),
                            variants,
                        )
                    };
                    if is_enum {
                        Err(EnumNotConstructable {
                            src: solver.src(location),
                            at: location.into(),
                            name,
                            variants,
                        })?
                    }
                    singular.map(|v| (aid, v, aid))
                }
                _ => None,
            },
            _ => None,
        };

        if let Some((layout, variant, adt)) = adt {
            // the ctor spelling the mismatch errors suggest (`Stack(<int>)`,
            // `List::Cons { head = <int> }`) -- computed while `variant` is
            // still the enum; the arms destructure it
            let ctor_spelling = variant.ctor(&solver.adts[layout].name);
            match variant {
                Variant::Tuple(variant) => {
                    if let Some(named) = self.arguments.iter().find(|a| a.name.is_some()) {
                        let name = named.name.as_ref().unwrap();
                        Err(UnknownNamedArgument {
                            src: solver.src(name.location),
                            at: name.location.into(),
                            name: name.lexeme.clone(),
                        })?
                    }
                    // a ctor on a generic adt gets a fresh instantiation of its params --
                    // `List::C(1, ...)` and `List::C("x", ...)` pick different `T`s, and the
                    // member types are specialized with the same args the result reports
                    let adt_ty = solver.instantiated_adt_at(adt, location);
                    let ctor_args: Vec<Ty> = match &adt_ty {
                        Ty::Adt(_, args) => args.clone(),
                        _ => vec![],
                    };
                    let mut args = self.arguments.clone().into_iter().peekable();
                    let mut params = variant
                        .members
                        .iter()
                        .map(|m| solver.specialize(m, adt, &ctor_args))
                        .collect::<Vec<_>>()
                        .into_iter();

                    loop {
                        match (args.next(), params.next()) {
                            (Some(arg), Some(param)) => arg.value.fulfill_ty(param, solver)?,
                            (Some(_), None) => Err(ExtraTupleMembers {
                                src: solver.src(location),
                                at: location.into(),
                                ctor: Some(format!("`{ctor_spelling}`")),
                            })?,
                            (None, Some(_)) => Err(MissingTupleMembers {
                                src: solver.src(location),
                                at: location.into(),
                                ctor: Some(format!("`{ctor_spelling}`")),
                            })?,
                            _ => break,
                        }
                    }

                    solver.shadow_expr_ty(self.left.id(), Ty::adt(layout), self.left.location())?;
                    Ok(adt_ty.normalized(solver))
                }
                // r[impl mimas.struct.positional]
                // `Vec2(0, -1)`: the arguments fill the fields in declaration
                // order; a named one (`Vec2(y = -1, x = 0)`) fills its field by name
                Variant::Struct(variant) if !solver.adts[adt].flags.contains(AdtFlags::IS_ENUM) => {
                    let names: Vec<String> = variant
                        .fields
                        .0
                        .iter()
                        .filter(|(_, f)| !f.constant)
                        .map(|(n, _)| n.clone())
                        .collect();
                    let positional = self.arguments.iter().filter(|a| a.name.is_none()).count();
                    if positional > names.len() {
                        Err(ExtraTupleMembers {
                            src: solver.src(location),
                            at: location.into(),
                            ctor: Some(format!("`{ctor_spelling}`")),
                        })?
                    }
                    let mut keyed: Vec<(FieldKey, &Expr)> = vec![];
                    let mut order: Vec<String> = vec![];
                    for (i, arg) in self.arguments.iter().enumerate() {
                        let key = match &arg.name {
                            Some(name) => name.clone(),
                            None => {
                                solver
                                    .field_hints
                                    .push((arg.value.location(), names[i].clone()));
                                Ident::new(names[i].clone(), arg.value.location())
                            }
                        };
                        order.push(key.lexeme.clone());
                        keyed.push((FieldKey::Ident(key), &arg.value));
                    }
                    let ty = struct_fields(solver, adt, &variant, layout, &keyed, None, location)?;
                    // the field each argument fills, for the emitter
                    let slots = order
                        .iter()
                        .map(|n| names.iter().position(|m| m == n).unwrap_or(usize::MAX))
                        .collect();
                    solver.ctor_slots.insert(id, slots);
                    solver.ctor_names.insert(id, order);
                    solver.shadow_expr_ty(self.left.id(), Ty::adt(layout), self.left.location())?;
                    Ok(ty)
                }
                v @ Variant::Struct(_) => Err(ExpectedTupleStruct {
                    src: solver.src(location),
                    at: location.into(),
                    ty: v.to_string(),
                    ctor: format!("construct it with braces: `{ctor_spelling}`"),
                })?,
            }
        } else {
            // methods win over same-named fields at call sites -- Access::Dot is field-first
            // for value reads, so short-circuit here to make the builder pattern work
            // (`s.power(5)` where `power` is also a field).
            fn method_intercept(call: &Call, solver: &mut Solver) -> Result<Option<Ty>> {
                let ExprKind::Access(Access::Dot { left, right, kind }) = call.left.kind() else {
                    return Ok(None);
                };
                let Some(ident) = right.as_ident() else {
                    return Ok(None);
                };
                let lhs = match (kind, left.query(solver)?.normalized(solver)) {
                    (AccessKind::Option, Ty::Option(inner)) => *inner,
                    (AccessKind::Option, other) => Err(InvalidUnwrap {
                        src: solver.src(left.location()),
                        at: left.location().into(),
                        ty: other.to_string(),
                    })?,
                    (_, other) => other,
                };
                // a pact-typed or bounded receiver calling an overloaded member
                // (`x.mul(3.0)` on `x: T` under `<T: Num>`): the key is chosen by
                // the one argument, and the noted dec carries it for emit
                // r[impl mimas.pact.overload]
                if let Some(pids) = solver.pacts_of(&lhs)
                    && let [arg] = call.arguments.as_slice()
                    && arg.name.is_none()
                    && let Some((pid, keys)) = pids.iter().find_map(|pid| {
                        solver
                            .pact_groups
                            .get(&(*pid, ident.lexeme.clone()))
                            .map(|keys| (*pid, keys.clone()))
                    })
                {
                    for key in keys {
                        let (header, _) = solver.pacts[&pid].functions[&key].clone();
                        let ty = solver.instantiate_pact_fn(&header, &lhs);
                        let Ty::Fn(f) = &ty else { continue };
                        let [_, operand] = f.parameters.as_slice() else {
                            continue;
                        };
                        if arg.value.fulfill_ty(operand.ty.clone(), solver).is_err() {
                            continue;
                        }
                        let member = solver.pact_members[&(pid, key.clone())];
                        solver.check_vis(member, ident.location)?;
                        solver.node_decs.insert(call.left.id(), member);
                        solver.note(ident, ty.clone(), Some(member));
                        return Ok(Some(ty));
                    }
                    return Ok(None);
                }
                let (aid, recv_args) = match &lhs {
                    Ty::Adt(a, args) | Ty::Identity(a, args) => (*a, args.clone()),
                    l => match solver.builtin_adt(l) {
                        Some(a) => (a, vec![]),
                        None => return Ok(None),
                    },
                };
                let Some(field) =
                    pick_method_overload(solver, aid, &ident.lexeme, &lhs, ident.location)?
                else {
                    return Ok(None);
                };
                let dec = field.dec;
                solver.check_vis(dec, ident.location)?;
                solver.node_decs.insert(call.left.id(), dec);
                // `push`/`pop`/`extend` on a `[T; n]` -- the type pins the length,
                // so length-changing methods are rejected outright
                if crate::lens::LEN_MUTATING.contains(&ident.lexeme.as_str())
                    && let Ty::Array(_, Len::Const(n)) = lhs
                {
                    Err(crate::errors::SizedArrayMutation {
                        src: solver.src(ident.location),
                        at: ident.location.into(),
                        method: ident.lexeme.clone(),
                        ty: format!("[_; {n}]"),
                    })?
                }
                // a `&mut`-receiver method on a collection a `for` loop is iterating
                // (`xs.push(..)` inside `for x in xs`) -- see `check_iter_guard`
                if solver
                    .dec_to_native
                    .get(&dec)
                    .is_some_and(|b| b.mutates_recv)
                    && let Some((root, path)) = crate::root_and_path(left)
                    && let Some(root_dec) = solver.ribs.resolve(root)
                {
                    solver.check_iter_guard(root_dec, &path, left.location())?;
                }
                let ty = if let Some(binding) = solver.dec_to_native.get(&dec) {
                    let sig = binding.sig.clone();
                    solver.instantiate_native(&sig, Some((&lhs, left.location())))?
                } else {
                    // a method stored on a generic adt holds `Ty::Param`s; plug in the
                    // receiver's own args (`l.push(5)` on `List<int>` sees `x: int`)
                    solver.specialize(&field.ty, aid, &recv_args)
                };
                // `v.mul(2.0)` beside `fn mul(self, o: V)` and `fn mul(self, k: float)`:
                // the call picks the overload its one argument fits, the same
                // choice `v * 2.0` makes (r[impl mimas.op.overload])
                let (dec, ty) =
                    pick_source_overload(solver, aid, &ident.lexeme, &recv_args, call, dec, ty)?;
                solver.node_decs.insert(call.left.id(), dec);
                solver.note(ident, ty.clone(), Some(dec));
                Ok(Some(ty))
            }
            let ty = match method_intercept(self, solver)? {
                Some(t) => {
                    solver.shadow_expr_ty(self.left.id(), t.clone(), self.left.location())?;
                    t.normalized(solver)
                }
                // the callee is the one place a method access is legal
                None => solver.query_non_value(&self.left)?.normalized(solver),
            };
            let fn_data = match ty {
                Ty::Fn(fn_data) => fn_data,
                // unbound callee (e.g. `fn apply(f, x) { f(x) }` -- `f` has no annotation, so
                // its type is still an unresolved vid when we solve the body). synthesize a
                // fn shape with fresh vids for each arg + return and unify it back onto the
                // callee. cascading constraints fill the params/return when the outer call
                // (`apply(double, 5)`) supplies a concrete `double`.
                //
                // named args aren't supported in this path: the synthesized params have no
                // source names. anything that tried `f(b=1)` falls through to NotCallable
                // since we can't decide what `b` refers to without a real signature.
                Ty::Vid(_) if self.arguments.iter().all(|a| a.name.is_none()) => {
                    let synth_params: Vec<FnParam> = self
                        .arguments
                        .iter()
                        .map(|_| FnParam::new(None, Ty::Vid(solver.vid()), false))
                        .collect();
                    let synth_return = Ty::Vid(solver.vid());
                    let synth =
                        FnHeader::new(synth_params, synth_return, /* is_method */ false);
                    self.left.fulfill_ty(Ty::Fn(synth.clone()), solver)?;
                    synth
                }
                _ => Err(NotCallable {
                    src: solver.src(location),
                    at: location.into(),
                    ty: ty.to_string(),
                })?,
            };
            // the callee names a `multi fn`: emit lowers this call as runtime
            // dispatch over the method arms, not a direct call. the header's
            // own params still check the arguments against the bound types.
            // r[impl mimas.multi]
            if let Some(&gdec) = solver.node_decs.get(&self.left.id())
                && solver.multis.contains_key(&gdec)
            {
                solver.multi_calls.insert(id, gdec);
            }
            let params: Vec<FnParam> = fn_data.parameters;
            let mut filled = vec![false; params.len()];
            let mut cursor = 0usize;

            if fn_data.is_method
                && let ExprKind::Access(Access::Dot { left, kind, .. }) = self.left.kind()
                && let Some(first) = params.first()
            {
                // a `?.` receiver is option-typed; dispatch is on the unwrapped inner
                // (method_intercept already vetted it), so expect Option(self) here
                let expected = if *kind == AccessKind::Option {
                    Ty::Option(Box::new(first.ty.clone()))
                } else {
                    first.ty.clone()
                };
                left.fulfill_ty(expected, solver)?;
                filled[0] = true;
                cursor = 1;
            }

            // slot resolution + duplicate checks run in arg order first; the
            // fulfill_ty pass below is reordered so closures check last -- their
            // expected param types may hold still-unbound vids that a later
            // positional arg pins down (`under(at(2), |x| x * 10, xs)` -- `xs`
            // binds the lens's element vid before `|x|`'s body is checked).
            let mut args_by_slot: Vec<(usize, &parse::Argument)> = Vec::new();
            for arg in &self.arguments {
                let slot = match &arg.name {
                    None => {
                        if cursor >= params.len() {
                            Err(ExtraArguments {
                                src: solver.src(location),
                                at: location.into(),
                            })?
                        }
                        let s = cursor;
                        cursor += 1;
                        s
                    }
                    Some(name) => params
                        .iter()
                        .position(|p| p.name.as_deref() == Some(name.lexeme.as_str()))
                        .ok_or_else(|| UnknownNamedArgument {
                            src: solver.src(name.location),
                            at: name.location.into(),
                            name: name.lexeme.clone(),
                        })?,
                };
                if filled[slot] {
                    let name = arg.name.as_ref().unwrap();
                    Err(DuplicateNamedArgument {
                        src: solver.src(name.location),
                        at: name.location.into(),
                        name: name.lexeme.clone(),
                    })?
                }
                filled[slot] = true;
                args_by_slot.push((slot, arg));
            }
            let (closures, rest): (Vec<_>, Vec<_>) = args_by_slot
                .iter()
                .partition(|(_, arg)| matches!(arg.value.kind(), ExprKind::Closure(_)));
            // `&` loans collect their root decs so exclusivity can check them
            // pairwise and against the roots a sibling argument reads.
            // r[impl mimas.inout.exclusive]
            let mut lent: Vec<(DecId, String, Location)> = Vec::new();
            let mut plain_roots: Vec<(DecId, String, Location)> = Vec::new();
            let borrow_root = |solver: &Solver, arg: &parse::Argument| {
                solver
                    .place_sources
                    .get(&arg.value.id())
                    .and_then(crate::root_and_path)
                    .and_then(|(root, _)| {
                        solver
                            .ribs
                            .resolve(root)
                            .map(|dec| (dec, root.lexeme.clone(), root.location()))
                    })
            };
            for &(slot, arg) in rest.iter().chain(&closures) {
                if matches!(arg.value.kind(), ExprKind::Closure(_))
                    && let Ty::Fn(header) = params[slot].ty.clone().normalized(solver)
                {
                    solver
                        .expected_closures
                        .insert(arg.value.id(), header.parameters);
                }
                let is_borrow = matches!(arg.value.kind(), ExprKind::Call(c) if c
                    .left
                    .as_ident()
                    .is_some_and(|i| i.lexeme == "__borrow_place"));
                match solver.place_element(&params[slot].ty) {
                    Some((removable, mut elem)) => {
                        // an `inout` parameter takes a place, spelled `&` at the
                        // call -- a bare value has no slot to write back through.
                        // r[impl mimas.inout.param]
                        if !is_borrow {
                            Err(crate::errors::InoutArgument {
                                src: solver.src(arg.value.location()),
                                at: arg.value.location().into(),
                                param: params[slot]
                                    .name
                                    .clone()
                                    .unwrap_or_else(|| "parameter".to_string()),
                            })?
                        }
                        solver.allowed_borrows.insert(arg.value.id());
                        let mut arg_ty = arg.value.query(solver)?;
                        match solver.place_element(&arg_ty) {
                            // a removable place still lends where removal is
                            // not on offer; the reverse would fabricate a
                            // `remove` the source place doesn't have.
                            Some((arg_removable, _)) if removable && !arg_removable => {
                                let mut want = params[slot].ty.clone();
                                arg_ty
                                    .fulfill_ty(&mut want, solver)
                                    .map_err(|e| {
                                        e.into_type_mismatch(solver, arg.value.location())
                                    })?;
                            }
                            Some((_, mut arg_elem)) => {
                                arg_elem
                                    .fulfill_ty(&mut elem, solver)
                                    .map_err(|e| {
                                        e.into_type_mismatch(solver, arg.value.location())
                                    })?;
                            }
                            None => {
                                arg.value.fulfill_ty(params[slot].ty.clone(), solver)?;
                            }
                        }
                        if let Some(rooted) = borrow_root(solver, arg) {
                            lent.push(rooted);
                        }
                        solver.inout_calls.insert(id);
                        // a `&binding` loans through a cell: the call writes
                        // `__cell[0]` back into the binding when it returns.
                        // r[impl mimas.inout.binding]
                        if let Some(&writeback) = solver.binding_borrows.get(&arg.value.id()) {
                            solver.inout_writebacks.entry(id).or_default().push(writeback);
                        }
                    }
                    None => {
                        // a `&` here was never claimed by an inout slot, so the
                        // borrow errors inside `fulfill_ty` -- only plain args
                        // reach the roots below.
                        arg.value.fulfill_ty(params[slot].ty.clone(), solver)?;
                        if let Some((root, _)) = crate::root_and_path(&arg.value)
                            && let Some(dec) = solver.ribs.resolve(root)
                        {
                            plain_roots.push((dec, root.lexeme.clone(), root.location()));
                        }
                    }
                }
            }
            // r[impl mimas.inout.exclusive]
            for (i, (dec, name, at)) in lent.iter().enumerate() {
                let overlap = lent[..i]
                    .iter()
                    .chain(&plain_roots)
                    .find(|(d, ..)| d == dec);
                if overlap.is_some() {
                    Err(crate::errors::InoutAlias {
                        src: solver.src(*at),
                        at: (*at).into(),
                        root: name.clone(),
                    })?
                }
            }
            if params
                .iter()
                .zip(&filled)
                .any(|(p, &f)| !f && !p.has_default)
            {
                Err(MissingArguments {
                    src: solver.src(location),
                    at: location.into(),
                })?;
            }

            // `where` predicates on the callee: prove what the call site's args can
            // prove -- a predicate that's false-forever fails here, anything unknown
            // defers to the callee's entry check (see lens.rs).
            if let Some(&dec) = solver.node_decs.get(&self.left.id()) {
                let receiver = match self.left.kind() {
                    ExprKind::Access(Access::Dot { left, .. }) if fn_data.is_method => Some(left),
                    _ => None,
                };
                solver.check_call_wheres(dec, &params, &args_by_slot, receiver, location)?;
            }

            let ty = fn_data.return_ty.as_ref().clone().normalized(solver);
            // dataframe schemas ride beside the type system: `pull` gets the column's
            // concrete element type here, and `schema`/`table{}`/`__q_*` calls record
            // column maps for downstream calls (see frames.rs)
            let ty = solver.frame_call(id, self, ty)?;
            if let ExprKind::Access(Access::Dot {
                kind: AccessKind::Option,
                ..
            }) = self.left.kind()
            {
                Ok(Ty::Option(Box::new(ty)))
            } else {
                Ok(ty)
            }
        }
    }
}

impl Solve for Closure {
    fn solve(&self, id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        solver.ribs.push_closure(id);

        // a call site may have stashed the fn-typed parameter it expects here -- seed each closure
        // param's vid with it before the body solves, so `|n| n + 1` learns `n: int` from the
        // annotation rather than erroring on an unresolved vid. only when arities match; a mismatch
        // falls through to the call-site unification, which reports it.
        let expected = solver
            .expected_closures
            .swap_remove(&id)
            .filter(|e| e.len() == self.parameters.len());

        let mut parameters = vec![];
        for (i, binding) in self.parameters.iter().enumerate() {
            let mut param = solver.bind_parameter(binding)?;
            if let Some(expected) = &expected {
                let mut want = expected[i].ty.clone();
                param
                    .ty
                    .fulfill_ty(&mut want, solver)
                    .map_err(|e| e.into_type_mismatch(solver, binding.location()))?;
                param.ty = param.ty.normalized(solver);
            }
            // closures have no hoist phase, so the real per-body declare happens here. mirrors
            // the loop in `Function::solve`. `bind_parameter` itself no longer declares -- it
            // just resolves the param's type into an `FnParam`.
            solver.declare(
                binding.left.as_ident().unwrap(),
                binding.left.id(),
                param.ty.clone(),
            )?;
            if solver.inout_params.contains(&binding.left.id())
                && let Some(&dec) = solver.node_decs.get(&binding.left.id())
            {
                solver.inout_decs.insert(dec);
            }
            parameters.push(param);
        }

        // a `return` in here exits the closure (not the enclosing fn). without an annotation,
        // the first return or the body's value settles the type.
        let expected_ty = match &self.return_type {
            Some(annotation) => Ty::from_annotation(annotation.clone(), solver)?,
            None => Ty::Vid(solver.vid()),
        };
        solver.fn_stack.push(FnRun {
            expected_ty: expected_ty.clone(),
        });
        // `break`, `continue` and `collect` can't target a loop outside the closure (the body
        // compiles on its own)
        let outer_loops = std::mem::take(&mut solver.loop_stack);
        solver.control_flow.enter();
        solver.hint_records(&self.body, &expected_ty);
        let body_ty = self.body.query(solver)?;
        solver.loop_stack = outer_loops;
        solver.fn_stack.pop().unwrap();
        let flow = solver.control_flow.exit();

        let expected_ty = expected_ty.normalized(solver);
        if body_ty != Ty::Unit || matches!(expected_ty, Ty::Vid(_)) {
            self.body.fulfill_ty(expected_ty.clone(), solver)?;
        } else if expected_ty != Ty::Unit
            && solver.control_flow.quantify(&flow) != Quantification::Universal
        {
            Err(NotAllPathsReturn {
                src: solver.src(self.body.location()),
                at: self.body.location().into(),
                ty: expected_ty.to_string(),
            })?;
        }

        solver.ribs.pop();

        let return_ty = expected_ty.normalized(solver);
        Ok(Ty::Fn(FnHeader::new(parameters, return_ty, false)))
    }
}

impl Solve for Coalescence {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        let lhs = self.left.query(solver)?.normalized(solver);
        match lhs {
            Ty::Option(inner) => {
                let inner = *inner;
                self.right.fulfill_ty(inner.clone(), solver)?;
                Ok(inner)
            }
            Ty::Null => self.right.query(solver),
            other => Err(CoalesceNonOptional {
                src: solver.src(self.left.location()),
                at: self.left.location().into(),
                ty: other.to_string(),
            })?,
        }
    }
}

impl Solve for Collect {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let ty = self.value.query(solver)?;
        let src = solver.src(location);
        solver.with_loop_mut(
            || {
                CollectOutsideLoop {
                    src: src.clone(),
                    at: location.into(),
                }
                .into()
            },
            |solver, loop_data| {
                solver
                    .regsiter_loop_control_flow_ty(ty, &mut loop_data.collect_ty)
                    .map_err(|e| e.into_type_mismatch(solver, location))
            },
        )?;
        Ok(Ty::Never)
    }
}

impl Solve for Continue {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        if solver.loop_stack.last().is_none() {
            Err(ContinueOutOfLoop {
                src: solver.src(location),
                at: location.into(),
            })?
        }
        Ok(Ty::Never)
    }
}

impl Solve for Enum {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        let Some(adt) = self.head.query(solver)?.as_adt().copied() else {
            unreachable!()
        };

        // self-referential variants (`enum List { Cons(int, List), Nil }`) need filter_adt to
        // break the recursion check on field types -- same pattern as Struct::solve.
        solver.ribs.push_impl(adt);
        // `enum List<T> { .. }`: member annotations resolve `T` (and `List<T>`) through the
        // declared params -- stored as `Ty::Param`s for per-use instantiation
        let generic = solver.adts[adt].is_generic();
        if generic {
            solver.push_adt_params(adt);
        }

        let result = self
            .members
            .iter()
            .try_for_each(|(member_name, member)| match member {
                Member::Tuple(members) => {
                    members.iter().enumerate().try_for_each(|(i, annotation)| {
                        let mut ty = Ty::from_annotation(annotation.clone(), solver)?;
                        let mut field_ty = solver.adts[adt]
                            .variants
                            .get_mut(&member_name.lexeme)
                            .unwrap()
                            .as_tuple()
                            .members
                            .get(i)
                            .unwrap()
                            .clone();

                        field_ty
                            .fulfill_ty(&mut ty, solver)
                            .map_err(|v| v.into_type_mismatch(solver, member_name.location()))
                    })
                }
                Member::Struct(fields) => fields.iter().try_for_each(|field| {
                    let mut ty = Ty::from_annotation(field.annotation.clone(), solver)?;
                    let mut field_ty = solver.adts[adt]
                        .variants
                        .get_mut(&member_name.lexeme)
                        .unwrap()
                        .as_struct()
                        .ty(&field.name.to_string())
                        .unwrap();
                    field_ty
                        .fulfill_ty(&mut ty, solver)
                        .map_err(|v| v.into_type_mismatch(solver, field.location))
                }),
            });

        if generic {
            solver.pop_type_params();
        }
        solver.ribs.pop();
        result?;

        // todo: helper method on solver to avoid this clone
        let mut normalized = solver.adts[adt].clone();
        normalized.normalize_members(solver);
        let layouts: Vec<_> = normalized
            .variants
            .values()
            .filter_map(|variant| {
                variant.layout().map(|layout| {
                    let name = solver.adts[layout].name.clone();
                    (layout, Adt::new_variant_layout(name, variant))
                })
            })
            .collect();
        solver.adts[adt] = normalized;
        for (layout, mut layout_adt) in layouts {
            // the rebuild drops what hoist recorded: the layout still
            // belongs to this enum, or `enum_parent` can't answer and the
            // variant-is-a-enum widening below has nothing to find.
            // r[impl mimas.multi]
            layout_adt.enum_of = Some(adt);
            solver.adts[layout] = layout_adt;
        }

        Ok(Ty::adt(adt))
    }
}

/// `col("age") > 30` (and `==`/`!=`/`<`/`<=`/`>=`, and `+ - * /` in [`Evaluation::solve`]) build a
/// real `polars::prelude::Expr` tree at runtime instead of evaluating anything -- see the
/// `to_pl_expr`-gated block at the top of `bin()` in `crates/vm/src/val.rs`. That's a dynamic
/// (VM-level) overload with no matching static concept, so the two solve impls special-case it
/// here by name: whenever either operand's type is the native `PlExpr` adt, the whole expression's
/// type is `PlExpr` too (an int/float/bool/str on the other side is fine -- the VM auto-promotes it
/// via `lit(..)`; anything else is a runtime error, same tradeoff the VM side already made). This
/// can't reach for a real vid-level "supports overload" trait bound; native adts are opaque to the
/// solver, so matching by name against the one adt that opts in is what's available.
fn plexpr_overload_ty(lhs: &Ty, rhs: &Ty, solver: &Solver) -> Option<Ty> {
    let is_plexpr = |ty: &Ty| matches!(ty, Ty::Adt(id, _) if solver.adts[*id].name == "PlExpr");
    if is_plexpr(lhs) {
        Some(lhs.clone())
    } else if is_plexpr(rhs) {
        Some(rhs.clone())
    } else {
        None
    }
}

/// `tensor + tensor` / `tensor * 0.5` / `2 - tensor` -- the same kind of dynamic VM-level
/// overload as `PlExpr`'s, resolved through the `Val::Tensor` arms of `bin()` (broadcast
/// elementwise on the burn backend). Either operand being the native `Tensor` adt types the
/// whole expr `Tensor`. In `Evaluation` that's arithmetic; in `Equality` the six comparisons
/// are pervasive too (`t > 0` is the 0/1 mask, NumPy/Uiua-style -- `.all()`/`.any()` collapse
/// it back to bool). The other side still has to be broadcastable -- numeric, tensor, or
/// still-unresolved (`Vid`/`Param`) -- so `t == "x"` keeps its real error rather than typing
/// a mask that can never run.
/// `dec` arithmetic: with a `dec` or an `int` the answer is a `dec`, exact;
/// with a `float` it is a `float` (docs/decimal.md). r[impl mimas.dec]
pub(crate) fn dec_overload_ty(lhs: &Ty, rhs: &Ty, solver: &Solver) -> Option<Ty> {
    let is_dec = |ty: &Ty| matches!(ty, Ty::Adt(id, _) if solver.adts[*id].name == "dec");
    match (is_dec(lhs), is_dec(rhs)) {
        (true, true) => Some(lhs.clone()),
        (true, false) if matches!(rhs, Ty::Int) => Some(lhs.clone()),
        (false, true) if matches!(lhs, Ty::Int) => Some(rhs.clone()),
        (true, false) | (false, true) if matches!(lhs, Ty::Float) || matches!(rhs, Ty::Float) => {
            Some(Ty::Float)
        }
        _ => None,
    }
}

fn tensor_overload_ty(lhs: &Ty, rhs: &Ty, solver: &Solver) -> Option<Ty> {
    let is_tensor = |ty: &Ty| matches!(ty, Ty::Adt(id, _) if solver.adts[*id].name == "Tensor");
    let broadcastable =
        |ty: &Ty| matches!(ty, Ty::Int | Ty::Float | Ty::Vid(_) | Ty::Param(_)) || is_tensor(ty);
    if is_tensor(lhs) && broadcastable(rhs) {
        Some(lhs.clone())
    } else if is_tensor(rhs) && broadcastable(lhs) {
        Some(rhs.clone())
    } else {
        None
    }
}

/// `xs == 3`, `b[ln] == p`, `xs < 0` -- a list against a scalar compares pervasively and
/// answers a list of bools, as deep as the list nests (`[[T]] == T` is `[[bool]]`), the way
/// a tensor does. Two lists still compare structurally, a list against `null` is still the
/// option check, and an unresolved scalar keeps its inference (no mask is claimed for it).
fn seq_cmp_overload_ty(lhs: &Ty, rhs: &Ty, solver: &mut Solver) -> Option<Ty> {
    fn mask(seq: &Ty, scalar: &Ty, solver: &mut Solver) -> Option<Ty> {
        let Ty::Array(inner, _) = seq else {
            return None;
        };
        let inner = inner.as_ref().clone().normalized(solver);
        if matches!(inner, Ty::Array(..)) {
            return mask(&inner, scalar, solver).map(Ty::array);
        }
        if matches!(
            scalar,
            Ty::Array(..) | Ty::Null | Ty::Option(_) | Ty::Vid(_) | Ty::Dict(_) | Ty::Tuple(_)
        ) {
            return None;
        }
        // a `null` element is simply unequal to the scalar: compare the inner type
        let mut a = match inner {
            Ty::Option(inner) => inner.as_ref().clone(),
            inner => inner,
        };
        let mut b = scalar.clone();
        a.fulfill_ty(&mut b, solver).ok()?;
        Some(Ty::array(Ty::Bool))
    }
    mask(lhs, rhs, solver).or_else(|| mask(rhs, lhs, solver))
}

impl Solve for Equality {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        solver.dec_literal_beside(&self.left, &self.right)?;
        let lhs = self.left.query(solver)?;
        let rhs = self.right.query(solver)?;
        let lhs_n = lhs.clone().normalized(solver);
        let rhs_n = rhs.clone().normalized(solver);
        if let Some(ty) = plexpr_overload_ty(&lhs_n, &rhs_n, solver)
            .or_else(|| tensor_overload_ty(&lhs_n, &rhs_n, solver))
            .or_else(|| seq_cmp_overload_ty(&lhs_n, &rhs_n, solver))
        {
            return Ok(ty);
        }
        // a `dec` compares with a `dec`, an `int` or a `float` by value (r[impl mimas.dec])
        if dec_overload_ty(&lhs_n, &rhs_n, solver).is_some() {
            return Ok(Ty::Bool);
        }
        // `<` `<=` `>` `>=` on an adt with registered operator impls (`Api::add_bin_op`)
        // dispatch at runtime and answer a bool -- `==`/`!=` stay structural
        if !matches!(self.op, EqualityOp::Equal | EqualityOp::NotEqual) {
            let has_ops = |t: &Ty| matches!(t, Ty::Adt(id, _) if solver.adts[*id].flags.contains(AdtFlags::HAS_OPS));
            if has_ops(&lhs_n) || has_ops(&rhs_n) {
                return Ok(Ty::Bool);
            }
            if let Some(names) = cmp_method_names(self.op)
                && let Some(ty) = op_dispatch(
                    solver,
                    id,
                    names,
                    &self.left,
                    &self.right,
                    &lhs_n,
                    &rhs_n,
                    location,
                )?
            {
                return Ok(ty);
            }
            // unresolved operands on a relational op (`r.x <= r.w` inside `Rect<T>`):
            // anywhere the comparison could run both sides already agree, so a shared
            // var settles to `bool` and a lone vid pins to the concrete operand's type
            let mut lhs_n = lhs_n.clone();
            let mut rhs_n = rhs_n.clone();
            loop {
                match (&lhs_n, &rhs_n) {
                    (Ty::Vid(a), Ty::Vid(b)) if a == b => return Ok(Ty::Bool),
                    (Ty::Param(a), Ty::Param(b)) if a == b => return Ok(Ty::Bool),
                    (Ty::Vid(_), _) | (_, Ty::Vid(_)) => {
                        let mut l = lhs_n.clone();
                        let mut r = rhs_n.clone();
                        if l.fulfill_ty(&mut r, solver).is_err() {
                            break;
                        }
                        lhs_n = l.normalized(solver);
                        rhs_n = r.normalized(solver);
                    }
                    _ => {
                        return if (lhs_n.is_numeric() && rhs_n.is_numeric())
                            || (lhs_n == Ty::Str && rhs_n == Ty::Str)
                        {
                            Ok(Ty::Bool)
                        } else {
                            Err(InvalidComparison {
                                src: solver.src(location),
                                at: location.into(),
                            }
                            .into())
                        };
                    }
                }
            }
        }
        // `==`/`!=` on `any` (a `raise` payload): structural eq is total at
        // runtime, no static claim needed — `e == AscDesc::Desc` just asks.
        if matches!(self.op, EqualityOp::Equal | EqualityOp::NotEqual)
            && (lhs_n == Ty::Any || rhs_n == Ty::Any)
        {
            return Ok(Ty::Bool);
        }
        match (&lhs, &rhs) {
            (Ty::Null, Ty::Option(_)) | (Ty::Option(_), Ty::Null) => Ok(()),
            (l, r) if l.is_numeric() && r.is_numeric() => Ok(()),
            _ if matches!(self.op, EqualityOp::Equal | EqualityOp::NotEqual) => {
                // the LHS-driven rule lets `T? == T` coerce; mirror it when only the rhs
                // is optional so `T == T?` is equally legal (compare-inner -- null vs a
                // value is just false). the mirror needs a settled non-option lhs, so
                // inference for unresolved operands keeps pinning rhs to lhs as before.
                let l = lhs.clone().normalized(solver);
                let r = rhs.clone().normalized(solver);
                // `x == null` where x can never be null: the declaration is
                // what's wrong, not the `null` -- label the operand and name
                // the `?` it wants rather than a bare type mismatch.
                let settled = |t: &Ty| {
                    !matches!(
                        t,
                        Ty::Option(_) | Ty::Null | Ty::Any | Ty::Never | Ty::Vid(_)
                    )
                };
                let never_null = if matches!(r, Ty::Null) && settled(&l) {
                    Some((self.left.location(), l.to_string()))
                } else if matches!(l, Ty::Null) && settled(&r) {
                    Some((self.right.location(), r.to_string()))
                } else {
                    None
                };
                if let Some((at, ty)) = never_null {
                    let (op, answer) = match self.op {
                        EqualityOp::Equal => ("==", "false"),
                        _ => ("!=", "true"),
                    };
                    return Err(NeverNull {
                        src: solver.src(location),
                        at: at.into(),
                        ty,
                        op,
                        answer,
                    }
                    .into());
                }
                if matches!(r, Ty::Option(_)) && !matches!(l, Ty::Option(_) | Ty::Vid(_)) {
                    self.left.fulfill_ty(self.right.query(solver)?, solver)
                } else {
                    self.right.fulfill_ty(self.left.query(solver)?, solver)
                }
            }
            _ if lhs.clone().normalized(solver) == Ty::Str
                && rhs.clone().normalized(solver) == Ty::Str =>
            {
                Ok(())
            }
            _ => Err(InvalidComparison {
                src: solver.src(location),
                at: location.into(),
            })?,
        }?;

        Ok(Ty::Bool)
    }
}

impl Solve for Evaluation {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        solver.dec_literal_beside(&self.left, &self.right)?;
        fn eval(
            op: EvaluationOp,
            lhs: &Ty,
            rhs: &Ty,
            location: Location,
            solver: &Solver,
        ) -> Result<Ty> {
            // a named operator reaching here found no method on either side:
            // say which methods would have caught it (r[impl mimas.op.named])
            if op.is_named() {
                let (method, rmethod) = op_method_names(op);
                return Err(UnboundOperator {
                    src: solver.src(location),
                    at: location.into(),
                    lhs: lhs.to_string(),
                    rhs: rhs.to_string(),
                    op: op.to_string(),
                    method,
                    rmethod,
                }
                .into());
            }
            match (lhs, rhs) {
                (Ty::Str, Ty::Str) if op == EvaluationOp::Plus => Ok(Ty::Str),
                (Ty::Bool, Ty::Bool) if op.is_binary() => Ok(Ty::Bool),
                (Ty::Int, Ty::Int) if op == EvaluationOp::Divide => Ok(Ty::Float),
                (Ty::Int, Ty::Int) => Ok(Ty::Int),
                (Ty::Float, Ty::Float) | (Ty::Int, Ty::Float) | (Ty::Float, Ty::Int)
                    if !op.is_bitwise() =>
                {
                    Ok(Ty::Float)
                }

                (Ty::Tuple(a), Ty::Tuple(b)) if a.len() == b.len() => a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| eval(op, a, b, location, solver))
                    .collect::<Result<Vec<Ty>>>()
                    .map(Ty::Tuple),

                (Ty::Tuple(elems), s) if matches!(s, Ty::Int | Ty::Float) => elems
                    .iter()
                    .map(|e| eval(op, e, s, location, solver))
                    .collect::<Result<Vec<Ty>>>()
                    .map(Ty::Tuple),

                (s, Ty::Tuple(elems)) if matches!(s, Ty::Int | Ty::Float) => elems
                    .iter()
                    .map(|e| eval(op, s, e, location, solver))
                    .collect::<Result<Vec<Ty>>>()
                    .map(Ty::Tuple),

                // adts with registered operator impls (`Api::add_bin_op`) typecheck
                // here and dispatch at runtime — the result is the flagged operand's
                // type by convention (LHS wins when both are flagged).
                (Ty::Adt(l, _), _) if solver.adts[*l].flags.contains(AdtFlags::HAS_OPS) => {
                    Ok(lhs.clone())
                }
                (_, Ty::Adt(r, _)) if solver.adts[*r].flags.contains(AdtFlags::HAS_OPS) => {
                    Ok(rhs.clone())
                }

                (lhs, rhs) => Err(InvalidEvaluation {
                    src: solver.src(location),
                    at: location.into(),
                    lhs: lhs.to_string(),
                    op: op.to_string(),
                    rhs: rhs.to_string(),
                }
                .into()),
            }
        }

        let lhs = self.left.query(solver)?;
        let rhs = self.right.query(solver)?;
        let mut lhs_n = lhs.clone().normalized(solver);
        let mut rhs_n = rhs.clone().normalized(solver);
        // an operand typed by a pact (`x: Num`, or `x: T` under `<T: Num>`):
        // the operator is the pact's method, dispatched at runtime
        if let Some(ty) = pact_binary_dispatch(
            solver,
            id,
            op_method_names(self.op),
            &self.left,
            &self.right,
            &lhs_n,
            &rhs_n,
            location,
        )? {
            return Ok(ty);
        }
        // a pact-typed operand whose pacts declare neither method: nothing
        // could run this at runtime, so say so now r[impl mimas.pact.binary]
        if solver.pacts_of(&lhs_n).is_some() || solver.pacts_of(&rhs_n).is_some() {
            let (method, rmethod) = op_method_names(self.op);
            return Err(UnboundOperator {
                src: solver.src(location),
                at: location.into(),
                lhs: lhs_n.to_string(),
                rhs: rhs_n.to_string(),
                op: self.op.to_string(),
                method,
                rmethod,
            }
            .into());
        }
        // the VM-level overloads of `PlExpr`/`Tensor` cover the arithmetic
        // operators only; a named operator on them is its method or nothing
        if !self.op.is_named()
            && let Some(ty) = plexpr_overload_ty(&lhs_n, &rhs_n, solver)
                .or_else(|| tensor_overload_ty(&lhs_n, &rhs_n, solver))
                .or_else(|| dec_overload_ty(&lhs_n, &rhs_n, solver))
        {
            return Ok(ty);
        }
        // unresolved operands -- `r.x + r.w` on a `Rect<T>` receiver, or two params of an
        // unannotated `fn` -- share one type wherever `op` could run, so unify them (or pin
        // a lone vid to the concrete side) and answer that type. `/` never rides the
        // operand var: `int / int` already widens to `float`. settling a vid can reveal a
        // `Param` pair, so loop until no vid remains.
        loop {
            match (&lhs_n, &rhs_n) {
                (Ty::Vid(a), Ty::Vid(b)) if a == b => {
                    return Ok(match self.op {
                        EvaluationOp::Divide => Ty::Float,
                        _ => lhs_n,
                    });
                }
                (Ty::Param(a), Ty::Param(b)) if a == b => {
                    return Ok(match self.op {
                        EvaluationOp::Divide => Ty::Float,
                        _ => lhs_n,
                    });
                }
                // `a.w * 0.5`-style scalar math: `int` keeps the param's slot (an `int`
                // instantiation stays `int`, `float` stays `float`), while a `float`
                // operand widens the answer the same way it widens `int`s
                (Ty::Param(_), t) | (t, Ty::Param(_)) if t.is_numeric() => {
                    return Ok(if self.op == EvaluationOp::Divide || *t == Ty::Float {
                        Ty::Float
                    } else if matches!(lhs_n, Ty::Param(_)) {
                        lhs_n.clone()
                    } else {
                        rhs_n.clone()
                    });
                }
                (Ty::Vid(_), Ty::Vid(_)) => {
                    let mut l = lhs_n.clone();
                    let mut r = rhs_n.clone();
                    l.fulfill_ty(&mut r, solver)
                        .map_err(|e| e.into_type_mismatch(solver, location))?;
                    lhs_n = l.normalized(solver);
                    rhs_n = r.normalized(solver);
                }
                (Ty::Vid(_), _) | (_, Ty::Vid(_)) => {
                    let mut l = lhs_n.clone();
                    let mut r = rhs_n.clone();
                    if l.fulfill_ty(&mut r, solver).is_err() {
                        break;
                    }
                    lhs_n = l.normalized(solver);
                    rhs_n = r.normalized(solver);
                }
                _ => break,
            }
        }
        if let Some(ty) = op_dispatch(
            solver,
            id,
            op_method_names(self.op),
            &self.left,
            &self.right,
            &lhs_n,
            &rhs_n,
            location,
        )?
        .or(seq_op_dispatch(
            solver,
            id,
            self.op,
            &self.right,
            &lhs_n,
            &rhs_n,
            location,
        )?) {
            return Ok(ty);
        }
        eval(self.op, &lhs_n, &rhs_n, location, solver)
    }
}

impl Solve for For {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        // `for r in &xs` iterates element places, not element copies -- the
        // `&` is the same marker an `inout` argument wears, scoped to a
        // loop's own freeze instead of a call's loan.
        // r[impl mimas.for.inplace]
        let mut iterator = &self.iterator;
        while let ExprKind::Grouping(g) = iterator.kind() {
            iterator = &g.inner;
        }
        if let ExprKind::Call(c) = iterator.kind()
            && c.left
                .as_ident()
                .is_some_and(|i| i.lexeme == "__borrow_place")
        {
            return solver.solve_for_in_place(id, self, iterator, location);
        }

        // Run the loop
        solver.ribs.push_block();

        let iter_ty = self.iterator.query(solver)?.normalized(solver);
        match &iter_ty {
            Ty::Fn(header) if header.parameters.is_empty() => {
                if let Ty::Option(element) = header.return_ty.as_ref() {
                    solver.solve_pat(&self.binding, element.as_ref().clone())?;
                } else {
                    return Err(InvalidIterTarget {
                        src: solver.src(self.iterator.location()),
                        at: self.iterator.location().into(),
                        ty: iter_ty.to_string(),
                    }
                    .into());
                }
            }
            Ty::Str => {
                solver.solve_pat(&self.binding, Ty::Str)?;
            }
            Ty::Int => {
                solver.solve_pat(&self.binding, Ty::Int)?;
            }
            Ty::Array(value, _) => {
                solver.solve_pat(&self.binding, value.as_ref().clone())?;
            }
            Ty::Dict(value) => {
                solver.solve_pat(
                    &self.binding,
                    Ty::Tuple(vec![Ty::Str, value.as_ref().clone()]),
                )?;
            }
            other => Err(InvalidIterTarget {
                src: solver.src(self.iterator.location()),
                at: self.iterator.location().into(),
                ty: other.to_string(),
            })?,
        }

        // every name a `for` pattern binds is a loop variable -- `x`, and each member
        // of `for (i, v)` too: the member holds the pass's own copy, so `v = ...` /
        // `v += ...` must hit AssignToLoopVar exactly like `x = ...` does.
        // r[impl mimas.for.elem-bind]
        for leaf in self.binding.bound_leaves() {
            if let Some(&dec_id) = solver.node_decs.get(&leaf.id()) {
                solver.decs[dec_id].kind = DecKind::LoopVar;
            }
        }

        // `for x in <not a place>` binds a fresh value every pass, so `x.f =` /
        // `x[i] =` writes on the element var die with it -- the assignment path
        // warns (r[mimas.for.elem-write]). Dict iteration yields a fresh
        // `(key, value)` pair even when the dict itself is a place, so `p.1 =`
        // is dead there too.
        let fresh = if matches!(iter_ty, Ty::Dict(_)) {
            Some(crate::FreshElem::DictPair(self.iterator.to_string()))
        } else if crate::root_and_path(&self.iterator)
            .and_then(|(root, _)| solver.ribs.resolve(root))
            .is_none()
        {
            Some(crate::FreshElem::Iterated(self.iterator.to_string()))
        } else {
            None
        };
        if let Some(fresh) = fresh
            && let Some(&dec_id) = solver.node_decs.get(&self.binding.id())
        {
            solver.ephemeral_elems.insert(dec_id, fresh);
        }

        // guard the iterated place against in-loop mutation -- `for x in xs { xs.push(0) }`
        // would grow the bound mid-loop. only arrays/dicts can be mutated at all, and only a
        // name-rooted iterator (`xs`, `s.arr`, `xs[0]` -- not `f()`) names a place to guard.
        let guard = match &iter_ty {
            Ty::Array(..) | Ty::Dict(_) => {
                crate::root_and_path(&self.iterator).and_then(|(root, path)| {
                    solver.ribs.resolve(root).map(|dec| crate::IterGuard {
                        dec,
                        path,
                        inplace: false,
                        iterated_at: self.iterator.location(),
                        name: self.iterator.to_string(),
                    })
                })
            }
            _ => None,
        };
        let guarded = guard.is_some();
        if let Some(guard) = guard {
            solver.iter_guards.push(guard);
        }
        let run = solver.run_loop_body(&self.body);
        if guarded {
            solver.iter_guards.pop();
        }
        let LoopRun {
            break_ty,
            collect_ty,
            ..
        } = run?;
        solver.ribs.pop();

        // Validate the types
        let break_ty = break_ty.unwrap_or_default();
        Ok(match (break_ty, collect_ty) {
            (Ty::Unit, None) => Ty::Unit,
            (bty, None) => Ty::Option(Box::new(bty)),
            (Ty::Unit, Some(cty)) => array!(cty),
            (_, Some(_)) => Err(BreakValueWhileCollection {
                src: solver.src(location),
                at: location.into(),
            })?,
        })
    }
}

impl Solve for FString {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        for part in &self.parts {
            if let FStringPart::Expr(expr) = part {
                expr.query(solver)?;
            }
        }
        Ok(Ty::Str)
    }
}

impl Solve for Function {
    fn solve(&self, id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        let vid = solver.node_vid(id);

        #[cfg(feature = "logging")]
        println!("{}", crate::utils::Printer::vid(&vid));

        // hoist only seeds the vid for top-level / impl-body fns. an unseeded vid here means
        // we're a `fn` inside a fn body -- mimas has no support for that scope, only closures.
        let Some(ty) = solver.sub(vid).cloned() else {
            Err(NestedFn {
                src: solver.src(self.name.location),
                at: self.name.location.into(),
                name: self.name.lexeme.clone(),
            })?
        };
        let Ty::Fn(fn_data) = ty else {
            unreachable!("hoist seeded the vid with something other than Ty::Fn")
        };

        // declared type params are rigid names inside the body (`let y: T = x` is legal and
        // means exactly the caller's `T`). popped after the body's been checked
        solver.push_type_params(&self.type_params);

        // one rib holds the params and doubles as the capture barrier; the body's block pushes its
        // own scope on top.
        solver.ribs.push_function();
        let param_tys = fn_data.parameters.iter();
        let parse_params = self.parameters.iter();

        for (param, Binding { left, .. }) in param_tys.zip(parse_params) {
            solver.declare(left.as_ident().unwrap(), left.id(), param.ty.clone())?;
            // an `inout` param's dec holds a lent place: captures and returns
            // of it are escapes, so the name is worth remembering.
            // r[impl mimas.inout.escape]
            if solver.inout_params.contains(&left.id())
                && let Some(&dec) = solver.node_decs.get(&left.id())
            {
                solver.inout_decs.insert(dec);
            }
        }

        // `where` predicates solve in the parameter scope -- `xs.len() > 2` sees
        // `xs` bound to its declared type. Each must be a bool; whether it provably
        // holds is a call-site question (see lens.rs), not a body-condition one.
        for w in &self.wheres {
            w.fulfill_ty(Ty::Bool, solver)?;
        }

        let expected_ty = fn_data.return_ty.as_ref().clone();
        solver.fn_stack.push(FnRun {
            expected_ty: expected_ty.clone(),
        });
        solver.control_flow.enter();
        solver.hint_records(&self.body, &expected_ty);
        let body_ty = self.body.query(solver)?;
        solver.fn_stack.pop().unwrap();
        let flow = solver.control_flow.exit();

        if body_ty != Ty::Unit {
            self.body.fulfill_ty(expected_ty.clone(), solver)?;
        } else {
            let expected = expected_ty.clone().normalized(solver);
            // a still-open `->` isn't "known non-unit" -- skip NotAllPathsReturn
            // for it and let the settle below give it `()`
            if !matches!(expected, Ty::Vid(_))
                && expected != Ty::Unit
                && solver.control_flow.quantify(&flow) != Quantification::Universal
            {
                solver.pop_type_params();
                solver.ribs.pop();
                Err(NotAllPathsReturn {
                    src: solver.src(self.body.location()),
                    at: self.body.location().into(),
                    ty: expected.to_string(),
                })?;
            }
        }

        // r[impl mimas.fn.inferred] an inferred `->` that stayed isolated --
        // never unified into any param's type, at any depth -- settles to
        // `()`: a call site must not get to pin the return of a body that
        // produced nothing for it. A `->` that reaches a param stays open
        // (`fn id(x) { x }` is `?x -> ?x`, `fn first(a) { a[0] }` is
        // `[?e] -> ?e`), and one that only aliases another fn's unsettled `->`
        // (`fn a() { b() }`) waits for that fn's own solve.
        if let Ty::Vid(v) = expected_ty
            && solver.sub(v).is_none()
            && !fn_data.parameters.iter().any(|p| p.ty.occurs(v, solver))
        {
            solver
                .register_sub(v, Ty::Unit)
                .map_err(|e| e.into_type_mismatch(solver, self.body.location()))?;
        }

        solver.pop_type_params();
        solver.ribs.pop();

        // a `const` may call this fn if its body folds: record the body (its
        // node_decs are now in place) keyed by the fn's own dec so a call
        // site can bind params to literal args and fold (r[mimas.const.eval])
        if let Some(&dec) = solver.node_decs.get(&id) {
            let params = self
                .parameters
                .iter()
                .filter_map(|b| solver.node_decs.get(&b.left.id()).copied())
                .collect();
            solver.const_items.insert(
                dec,
                crate::solver::ConstItem {
                    params,
                    body: self.body.clone(),
                    method: self.is_method(),
                },
            );
        }

        Ok(Ty::Fn(fn_data.clone()))
    }
}

impl Solve for Grouping {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        self.inner.query(solver)
    }
}

impl Solve for Ident {
    fn solve(&self, id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        let ty = self.query(solver)?;
        let resolved = solver.ribs.resolve_with_closures(self);
        // a unit constant (`kWh` reads as `1kWh`) resolved through `resolve_name`'s
        // fallback, not the ribs -- it counts as bound so the use site links its dec
        let resolved = resolved.or_else(|| {
            solver
                .unit_consts
                .get(&self.lexeme)
                .map(|&(dec, _)| (dec, Vec::new()))
        });
        let bound = resolved.is_some();

        if let Some(value) = solver.place_read(id, self, &ty)? { return Ok(value); }

        // mark this usage; every closure boundary crossed on the way to the binding captures it
        if let Some((dec_id, crossed)) = resolved {
            solver.node_decs.insert(id, dec_id);
            if matches!(solver.decs[dec_id].kind, DecKind::Local | DecKind::LoopVar) {
                for closure in &crossed {
                    solver
                        .closure_captures
                        .entry(*closure)
                        .or_default()
                        .insert(dec_id);
                }
                // an inout param's place is lent for the call it arrived on --
                // a closure that keeps it would outlive the loan.
                // r[impl mimas.inout.escape]
                if solver.inout_decs.contains(&dec_id) && !crossed.is_empty() {
                    Err(crate::errors::InoutEscape {
                        src: solver.src(self.location()),
                        at: self.location().into(),
                        name: self.lexeme.clone(),
                        via: "captured by a closure".to_string(),
                    })?;
                }
            }
            // a pact name resolves so `P::CONST` paths have a head type, but
            // it has no value form -- a bare use ICEs at codegen. The `let`
            // check covers only ident RHSes; expression statements like the
            // stray second name of `impl P for S via Q Q` land here.
            if matches!(solver.decs[dec_id].kind, DecKind::Pact(_)) {
                Err(crate::errors::PactIsNotAValue {
                    src: solver.src(self.location()),
                    at: self.location().into(),
                })?;
            }
        }

        if let Some((adt, _)) = ty.applied() {
            let members = {
                let lock = &solver.adts[&adt];
                (self.lexeme == lock.name && lock.is_tuple_struct())
                    .then(|| lock.as_tuple().members.clone())
            };
            if let Some(members) = members {
                // `let make = Point;` on `struct Point<T>(T, T)`: the ctor value instantiates
                // the adt's params fresh, so each `make(..)` call picks its own `T`
                let ret = solver.instantiated_adt(adt);
                let ctor_args: Vec<Ty> = match &ret {
                    Ty::Adt(_, args) => args.clone(),
                    _ => vec![],
                };
                let params = members
                    .into_iter()
                    .map(|m| {
                        FnParam::new(
                            None,
                            solver.specialize(&m, adt, &ctor_args).normalized(solver),
                            false,
                        )
                    })
                    .collect();
                return Ok(Ty::Fn(FnHeader::new(params, ret, false).ctor()));
            }
        }

        // modules and builtin namespaces have a type for `::` paths to resolve against (see
        // [Solver::resolve_path_head]), but they aren't values
        if let Ty::Adt(adt, _) = &ty {
            let flags = solver.adts[adt].flags;
            if !bound || flags.intersects(AdtFlags::IS_MODULE | AdtFlags::IS_BUILTIN) {
                Err(NotAValue {
                    src: solver.src(self.location()),
                    at: self.location().into(),
                    name: self.lexeme.clone(),
                })?
            }
        }
        Ok(ty)
    }
}

impl Solve for If {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        if let Some(binding) = self.binding.as_ref() {
            let scrut_ty = self.condition.query(solver)?.normalized(solver);
            solver.ribs.push_block();
            // `if let` over an option binds the unwrapped value (implicit null test). an explicit
            // `?` pattern does its own unwrapping, so leave the option intact for it.
            solver.solve_match_pat(binding, scrut_ty, false)?;
        } else {
            self.condition.fulfill_ty(Ty::Bool, solver)?;
        }

        solver.control_flow.enter();
        let positive_ty = self.main_body.query(solver)?;
        let positive = solver.control_flow.exit();

        // Ensure the branches match, or coercse an option.
        let (ty, negative) = if let Some(else_expr) = self.else_expr.as_ref() {
            solver.control_flow.enter();
            // an error *inside* the else arm is real -- only a failure to unify the
            // arm's type with the main body's may fall back to option coercion
            // (otherwise the retry below re-reads a half-solved cache and swallows
            // the error, leaving unresolved nodes for the emitter to panic on)
            else_expr.query(solver)?;
            let ty = if let Err(e) = else_expr.fulfill_ty(positive_ty.clone(), solver) {
                if let Some(ty) = Ty::coerce_option(else_expr.query(solver)?, positive_ty, solver) {
                    ty
                } else {
                    Err(e)?
                }
            } else {
                positive_ty.clone()
            };
            (ty, solver.control_flow.exit())
        } else {
            // If there's no else then the main body must be ()
            self.main_body
                .fulfill_ty(Ty::Unit, solver)
                .map_err(|_| IfNeedsElse {
                    src: solver.src(location),
                    at: location.into(),
                })?;
            (Ty::Unit, solver.control_flow.push(Flow::Block(vec![])))
        };

        solver
            .control_flow
            .push(Flow::Potential { positive, negative });

        if self.binding.is_some() {
            solver.ribs.pop();
        }

        Ok(ty)
    }
}

impl Solve for Impl {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        // Find our target -- a pact impl may name a primitive (`impl Opp for
        // bool`), which lands on that primitive's library adt
        let primitive = match self.target.lexeme.as_str() {
            // `int`/`float` impls serve static calls (a space's flip calls a number
            // field's `inv` directly); only a `bool` may also travel as a pact-typed
            // value, since dispatch through a pact tests the value's kind at runtime
            // and only a bool has a cheap test (`x == true || x == false`)
            "bool" | "int" | "float" if self.pact.is_some() => {
                solver.library.get(self.target.lexeme.as_str()).copied()
            }
            _ => None,
        };
        let resolved = match primitive {
            Some(aid) => Some(aid),
            None => solver
                .resolve_name(&self.target, self.target.location())?
                .as_adt()
                .copied(),
        };
        let Some(aid) = resolved else {
            Err(InvalidImplTarget {
                src: solver.src(self.target.location()),
                at: self.target.location().into(),
            })?
        };

        let mut pact = if let Some(name) = self.pact.as_ref() {
            let ty = solver.resolve_name(name, name.location)?;
            let Some(pid) = ty.as_single_pact() else {
                return Err(NotAPact {
                    src: solver.src(name.location),
                    at: name.location.into(),
                    ty: ty.to_string(),
                }
                .into());
            };

            // we clone because we're going to remove the fields to ensure they are all covered
            Some((solver.pacts[pid].clone(), pid))
        } else {
            None
        };

        solver.ribs.push_impl(aid);
        // member signatures/annotations see the target's own declared params (`impl List` is
        // `impl List<T>` -- there's no specialization syntax)
        let generic = solver.adts[aid].is_generic();
        if generic {
            solver.push_adt_params(aid);
        }

        // todo, I am not positive that these need to be in separate passes, it may be okay to do
        // them concurrently

        // First resolve all constants
        for (con, id) in self.items.iter().filter_map(|v| {
            if let ItemKind::Const(con) = v.kind() {
                Some((con, v.id()))
            } else {
                None
            }
        }) {
            let mut ty = solver.solve_const(con, id)?;

            if let Some((pact, _)) = &mut pact {
                // a destructuring const can never satisfy a pact's `const NAME` requirement
                let name = con
                    .left
                    .as_ident()
                    .map(|i| i.lexeme.clone())
                    .unwrap_or_else(|| con.left.to_string());
                let mut pact_ty = pact.constants.swap_remove(&name).ok_or_else(|| {
                    miette::Error::from(PactConstNotFound {
                        src: solver.src(con.left.location()),
                        at: con.left.location().into(),
                        name,
                        pact: pact.name.clone(),
                    })
                })?;

                ty.fulfill_ty(&mut pact_ty, solver)
                    .map_err(|e| e.into_type_mismatch(solver, con.left.location()))?;
            }
        }

        // Run again to fire off fn impls
        let result: Result<()> = self.items.iter().try_for_each(|item| {
            let ItemKind::Function(function) = item.kind() else {
                return Ok(());
            };

            // memoized solve, mirrors `Query for Expr`. top-level fns get this for free via
            // visit_stmt, but impl method items only flow through here so we have to do it
            // ourselves.
            if solver.touch_node(item.id()) {
                let ty = function.solve(item.id(), item.location(), solver)?;
                let vid = solver.node_vid(item.id());
                solver
                    .register_sub(vid, ty)
                    .map_err(|e| e.into_type_mismatch(solver, item.location()))?;
            }
            let ty = Ty::Vid(solver.node_vid(item.id())).normalized(solver);

            // inherent methods are concrete fields; refresh the stored type with the solved one.
            // pact methods aren't fields (they dispatch via `pact_impls`), so this is a no-op
            // there.
            // an overload refreshes its own slot, never the primary's: match by dec
            let dec = solver.node_decs.get(&item.id()).copied();
            let name = &function.name.lexeme;
            if let Some(field) = solver.adts[aid]
                .impls
                .get_mut(name)
                .filter(|f| dec.is_none_or(|d| f.dec == d))
            {
                field.ty = ty.clone();
            } else if let Some(field) = solver.adts[aid]
                .overloads
                .get_mut(name)
                .and_then(|fs| fs.iter_mut().find(|f| Some(f.dec) == dec))
            {
                field.ty = ty.clone();
            }

            if let Some((pact, pid)) = &mut pact {
                // r[impl mimas.pact.overload]
                // the member this fn implements: by name when the name is declared once;
                // among an overload group, the key whose operand is this fn's operand
                let keys: Vec<String> = pact
                    .functions
                    .keys()
                    .filter(|k| crate::pact_key_base(k) == function.name.lexeme)
                    .cloned()
                    .collect();
                // every overload whose operand instantiates to this fn's operand:
                // for `impl Num for float`, `mul(self, o: Self)` and `mul(self, k:
                // float)` both read `float`, and one fn implements both
                // r[impl mimas.pact.primitive]
                let chosen: Vec<String> = match keys.as_slice() {
                    [one] => vec![one.clone()],
                    [] => vec![],
                    many => {
                        let target = solver.impl_target_ty().unwrap();
                        let mine = overload_operand_of(&ty).map(|o| o.normalized(solver));
                        many.iter()
                            .filter(|k| {
                                let (h, _) = &pact.functions[*k];
                                let inst = solver.instantiate_pact_fn(h, &target);
                                overload_operand_of(&inst)
                                    .map(|o| o.normalized(solver))
                                    .is_some_and(|o| {
                                        mine.as_ref().is_some_and(|m| same_operand(&o, m))
                                    })
                            })
                            .cloned()
                            .collect()
                    }
                };
                let mut found = None;
                for k in &chosen {
                    if let Some(f) = pact.functions.swap_remove(k) {
                        if let (Some(dec), Some(&member)) = (
                            solver.node_decs.get(&item.id()).copied(),
                            solver.pact_members.get(&(*pid, k.clone())),
                        ) {
                            solver.impl_member.entry(dec).or_default().push(member);
                        }
                        found.get_or_insert(f);
                    }
                }
                let (pact_header, _default) = found.ok_or_else(|| {
                    miette::Error::from(PactFunctionNotFound {
                        src: solver.src(function.name.location),
                        at: function.name.location.into(),
                        name: function.name.lexeme.clone(),
                        pact: pact.name.clone(),
                    })
                })?;

                // unify the whole function type against the (freshened) pact requirement --
                // covers param count, per-param types, and the return type in one move.
                let mut impl_ty = ty;
                // `Self` is this impl's target: `other: Self` has to be exactly this adt, not
                // just any implementer
                let mut pact_ty =
                    solver.instantiate_pact_fn(&pact_header, &solver.impl_target_ty().unwrap());
                impl_ty
                    .fulfill_ty(&mut pact_ty, solver)
                    .map_err(|e| e.into_type_mismatch(solver, function.name.location))?;
            }

            Ok(())
        });

        result?;
        if generic {
            solver.pop_type_params();
        }
        solver.ribs.pop();

        if let Some((pact, pid)) = pact {
            // anything left in the cloned pact is unimplemented, except functions carrying a
            // default body, which the impl may legally omit (the dispatch table was filled at
            // hoist) -- or a member a `via` impl forwards by name
            // r[impl mimas.impl.via]
            let missing: Vec<String> = pact
                .functions
                .iter()
                .filter(|(key, (_, default))| {
                    default.is_none() && !solver.via_covered.contains(&(aid, pid, (*key).clone()))
                })
                .map(
                    |(key, (h, _))| match overload_operand_of(&Ty::Fn(h.clone())) {
                        Some(o)
                            if key.contains('#')
                                || pact.functions.keys().any(|k| {
                                    k != key && crate::pact_key_base(k) == crate::pact_key_base(key)
                                }) =>
                        {
                            format!("{}(self, _: {})", crate::pact_key_base(key), o)
                        }
                        _ => key.clone(),
                    },
                )
                .chain(
                    pact.constants
                        .keys()
                        .filter(|k| !solver.via_covered.contains(&(aid, pid, (*k).clone())))
                        .cloned(),
                )
                .collect();
            if !missing.is_empty() {
                Err(PactImplIncomplete {
                    src: solver.src(self.target.location()),
                    at: self.target.location().into(),
                    target: self.target.lexeme.clone(),
                    pact: pact.name,
                    missing: missing.join(", "),
                })?;
            }
        }

        Ok(Ty::Unit)
    }
}

impl Solve for In {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let rhs = self.right.query(solver)?;
        match rhs {
            Ty::Array(ref inner, _) => {
                self.left.fulfill_ty(*inner.clone(), solver)?;
            }
            Ty::Dict(_) => {
                self.left.fulfill_ty(Ty::Str, solver)?;
            }
            Ty::Str => {
                self.left.fulfill_ty(Ty::Str, solver)?;
            }
            _ => Err(InvalidInTarget {
                src: solver.src(location),
                at: location.into(),
                ty: rhs.to_string(),
            })?,
        }
        Ok(Ty::Bool)
    }
}

impl Solve for Literal {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        fn assemble(exprs: Vec<&Expr>, solver: &mut Solver) -> Result<Ty> {
            let ty = if let Some(expr) = exprs.first() {
                let mut first_ty = expr.query(solver)?;
                for expr in exprs.iter().skip(1) {
                    // an error *inside* the element is real -- only a failure to unify the
                    // element's type with the accumulated one may fall back to coercion
                    // (otherwise the retry below re-reads a half-solved cache and swallows
                    // the error, leaving unresolved nodes for the emitter to panic on)
                    expr.query(solver)?;
                    first_ty = if let Err(e) = expr.fulfill_ty(first_ty.clone(), solver) {
                        let found = expr.query(solver)?;
                        if let Some(ty) = Ty::coerce_option(found.clone(), first_ty.clone(), solver)
                        {
                            ty
                        } else if let Some(ty) = Ty::coerce_pacts(found, first_ty, solver) {
                            // mixed concrete types that share a pact widen to it, so
                            // `[Square, Circle]` can be a `[Draw]`
                            ty
                        } else {
                            Err(e)?
                        }
                    } else {
                        first_ty.clone()
                    };
                }
                first_ty
            } else {
                Ty::Vid(solver.vid())
            };
            Ok(ty)
        }
        Ok(match self {
            // literals stay growable `[T]` in the type; their exact length is a
            // `lens` fact consulted where a sized slot wants it
            Literal::Array(exprs) => Ty::array(assemble(exprs.iter().collect(), solver)?),
            Literal::Struct(StructLiteral { name, fields, base }) => {
                let (adt, variant, layout) = match name.kind() {
                    ExprKind::Access(Access::DoubleColon { left, right }) => {
                        let lhs = solver.resolve_path_head(left)?;
                        let adt = match lhs {
                            Ty::Adt(adt, _) | Ty::Identity(adt, _) => adt,
                            _ => Err(NotAStruct {
                                src: solver.src(location),
                                at: location.into(),
                                ty: lhs.to_string(),
                            })?,
                        };

                        if solver.adts[adt].flags.contains(AdtFlags::IS_MODULE) {
                            let lhs = name.query(solver)?;
                            let Ty::Adt(layout, _) = lhs else {
                                Err(NotAStruct {
                                    src: solver.src(location),
                                    at: location.into(),
                                    ty: lhs.to_string(),
                                })?
                            };
                            let variant = solver.adts[layout]
                                .try_as_struct()
                                .ok_or_else(|| {
                                    Error::from(NotAStruct {
                                        src: solver.src(location),
                                        at: location.into(),
                                        ty: solver.adts[layout].name.clone(),
                                    })
                                })?
                                .clone();
                            (layout, variant, layout)
                        } else if !solver.adts[adt].flags.contains(AdtFlags::IS_ENUM) {
                            let name = solver.adts[adt].name.clone();
                            Err(NotAnEnum {
                                src: solver.src(location),
                                at: location.into(),
                                ty: name,
                            })?
                        } else {
                            let variant = solver.adts[adt]
                                .variants
                                .get(&right.lexeme)
                                .cloned()
                                .ok_or_else(|| {
                                    Error::from(VariantNotFound {
                                        src: solver.src(location),
                                        at: location.into(),
                                        name: right.lexeme.clone(),
                                    })
                                })?;

                            let layout = variant.layout().unwrap();
                            solver.note(right, Ty::adt(adt), variant.dec());
                            let Variant::Struct(struct_variant) = variant else {
                                Err(NotAStruct {
                                    src: solver.src(location),
                                    at: location.into(),
                                    ty: right.lexeme.clone(),
                                })?
                            };

                            (adt, struct_variant, layout)
                        }
                    }
                    ExprKind::Ident(ident) if ident.lexeme == "Self" => {
                        let adt = *solver.impl_target().as_ref().ok_or_else(|| {
                            Error::from(SelfOutOfContext {
                                src: solver.src(ident.location()),
                                at: ident.location().into(),
                            })
                        })?;
                        let variant = solver.adts[adt]
                            .try_as_struct()
                            .ok_or_else(|| {
                                Error::from(NotAStruct {
                                    src: solver.src(location),
                                    at: location.into(),
                                    ty: solver.adts[adt].name.clone(),
                                })
                            })?
                            .clone();
                        (adt, variant, adt)
                    }
                    ExprKind::Ident(ident) => {
                        let lhs = solver.resolve_path_head(name)?;
                        let dec = solver.ribs.resolve(ident).ok_or_else(|| NotFound {
                            src: solver.src(ident.location),
                            at: ident.location.into(),
                            name: ident.lexeme.clone(),
                            help: None,
                        })?;
                        if solver.decs[dec].kind.as_adt().is_none() {
                            Err(NotAStruct {
                                src: solver.src(location),
                                at: location.into(),
                                ty: ident.lexeme.clone(),
                            })?
                        }
                        // `Ty::Identity` shows up when the named struct is the current impl
                        // target -- filter_adt rewrote it. treat it as the same adt.
                        let adt = match lhs {
                            Ty::Adt(adt, _) | Ty::Identity(adt, _) => adt,
                            _ => Err(NotAStruct {
                                src: solver.src(location),
                                at: location.into(),
                                ty: lhs.to_string(),
                            })?,
                        };
                        let variant = solver.adts[adt]
                            .try_as_struct()
                            .ok_or_else(|| {
                                Error::from(NotAStruct {
                                    src: solver.src(location),
                                    at: location.into(),
                                    ty: solver.adts[adt].name.clone(),
                                })
                            })?
                            .clone();
                        (adt, variant, adt)
                    }
                    _kind => {
                        let ty = name.query(solver)?.to_string();
                        Err(Error::from(NotAStruct {
                            src: solver.src(name.location()),
                            at: name.location().into(),
                            ty,
                        }))?
                    }
                };

                let keyed: Vec<(FieldKey, &Expr)> =
                    fields.iter().map(|(k, v)| (k.clone(), v)).collect();
                let ty = struct_fields(
                    solver,
                    adt,
                    &variant,
                    layout,
                    &keyed,
                    base.as_ref(),
                    location,
                )?;
                solver.shadow_expr_ty(name.id(), Ty::adt(layout), name.location())?;
                ty
            }
            // r[impl mimas.record.context]
            Literal::Record(fields) => {
                // the struct the slot expects, when it expects one
                let hint = solver
                    .record_hints
                    .get(&_id)
                    .cloned()
                    .map(|t| t.normalized(solver));
                let hint = match hint {
                    Some(Ty::Option(inner)) => Some(*inner),
                    other => other,
                };
                let target = match hint {
                    Some(Ty::Adt(adt, _) | Ty::Identity(adt, _))
                        if !solver.adts[adt].flags.contains(AdtFlags::IS_ENUM)
                            && !solver.adts[adt].flags.contains(AdtFlags::IS_MODULE) =>
                    {
                        solver.adts[adt].try_as_struct().cloned().map(|v| (adt, v))
                    }
                    _ => None,
                };
                if let Some((adt, variant)) = target {
                    let keyed: Vec<(FieldKey, &Expr)> = fields
                        .iter()
                        .map(|(k, v)| (FieldKey::Ident(k.clone()), v))
                        .collect();
                    struct_fields(solver, adt, &variant, adt, &keyed, None, location)?
                } else {
                    let values: Vec<&Expr> = fields.iter().map(|(_, v)| v).collect();
                    match assemble(values, solver) {
                        Ok(t) => Ty::Dict(Box::new(t)),
                        Err(_) => Err(UntypedRecord {
                            src: solver.src(location),
                            at: location.into(),
                        })?,
                    }
                }
            }
            Literal::Tuple(members) => {
                // r[impl mimas.struct.positional]
                // a tuple where a named-field struct is expected fills its
                // fields in declaration order: `cells: [(0, -1), (0, 0)]`
                let hint = solver
                    .record_hints
                    .get(&_id)
                    .cloned()
                    .map(|t| t.normalized(solver));
                let hint = match hint {
                    Some(Ty::Option(inner)) => Some(*inner),
                    other => other,
                };
                let target = match hint {
                    Some(Ty::Adt(adt, _) | Ty::Identity(adt, _))
                        if !solver.adts[adt]
                            .flags
                            .intersects(AdtFlags::IS_ENUM | AdtFlags::IS_MODULE)
                            && !solver.adts[adt].is_tuple_struct() =>
                    {
                        solver.adts[adt].try_as_struct().cloned().map(|v| (adt, v))
                    }
                    _ => None,
                };
                match target {
                    Some((adt, variant)) => {
                        let names: Vec<String> = variant
                            .fields
                            .0
                            .iter()
                            .filter(|(_, f)| !f.constant)
                            .map(|(n, _)| n.clone())
                            .collect();
                        if members.len() != names.len() {
                            Err(if members.len() > names.len() {
                                Error::from(ExtraTupleMembers {
                                    src: solver.src(location),
                                    at: location.into(),
                                    ctor: Some(format!("`{}`", solver.adts[adt].name)),
                                })
                            } else {
                                Error::from(MissingTupleMembers {
                                    src: solver.src(location),
                                    at: location.into(),
                                    ctor: Some(format!("`{}`", solver.adts[adt].name)),
                                })
                            })?
                        }
                        for (v, n) in members.iter().zip(&names) {
                            solver.field_hints.push((v.location(), n.clone()));
                        }
                        let keyed: Vec<(FieldKey, &Expr)> = members
                            .iter()
                            .zip(&names)
                            .map(|(v, n)| (FieldKey::Ident(Ident::new(n.clone(), v.location())), v))
                            .collect();
                        struct_fields(solver, adt, &variant, adt, &keyed, None, location)?
                    }
                    None => Ty::Tuple(
                        members
                            .iter()
                            .map(|v| v.query(solver))
                            .collect::<Result<_>>()?,
                    ),
                }
            }
            _ => unreachable!(),
        })
    }
}

/// A struct literal's fields checked against `adt`'s declaration — shared by
/// `Foo { … }` and a bare `{ … }` whose slot expects `Foo`.
fn struct_fields(
    solver: &mut Solver,
    adt: AdtId,
    variant: &crate::components::StructVariant,
    layout: AdtId,
    fields: &[(FieldKey, &Expr)],
    base: Option<&Expr>,
    location: Location,
) -> Result<Ty> {
    // like a ctor call, the literal gets a fresh instantiation of the adt's
    // params; each field's stored `Ty::Param`s are specialized with those args so
    // what the fields end up as is exactly what the literal's type reports
    let adt_ty = solver.instantiated_adt_at(adt, location);
    let ctor_args: Vec<Ty> = match &adt_ty {
        Ty::Adt(_, args) => args.clone(),
        _ => vec![],
    };

    let mut prototype_fields = variant.fields.clone();
    for (field_name, value) in fields {
        let mut target = prototype_fields
            .swap_remove(&field_name.to_string())
            .ok_or_else(|| {
                Error::from(FieldNotFound {
                    src: solver.src(location),
                    at: location.into(),
                    field_name: field_name.to_string(),
                })
            })?;
        target.ty = solver.specialize(&target.ty, adt, &ctor_args);

        if let FieldKey::Ident(key) = field_name {
            solver.note(key, target.ty.clone(), Some(target.dec));
        }
        solver
            .settle_field_ty(&target.ty, target.dec)
            .map_err(|e| e.into_type_mismatch(solver, value.location()))?;
        value.fulfill_ty(target.ty, solver)?;
    }

    // r[impl mimas.struct.update]
    // `..base`: the rest of the fields come from a value of this very
    // type -- a struct, not an enum variant, whose layout the base's
    // own variant might not share
    if let Some(base) = base {
        if layout != adt {
            Err(UpdateNeedsStruct {
                src: solver.src(base.location()),
                at: base.location().into(),
            })?;
        }
        base.fulfill_ty(adt_ty.clone(), solver)?;
    }

    let remaining: Vec<String> = prototype_fields
        .iter()
        .filter(|(_, field)| !field.constant)
        .map(|v| v.0.clone())
        .collect();

    if base.is_none() && !remaining.is_empty() {
        Err(MissingStructFields {
            src: solver.src(location),
            at: location.into(),
            fields: remaining,
        })?;
    }

    Ok(adt_ty.normalized(solver))
}

impl Solve for Logical {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        self.left.fulfill_ty(Ty::Bool, solver)?;
        self.right.fulfill_ty(Ty::Bool, solver)?;
        Ok(Ty::Bool)
    }
}

impl Solve for Loop {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let LoopRun {
            break_ty,
            collect_ty,
            ..
        } = solver.run_loop_body(&self.body)?;
        let ty = match (break_ty, collect_ty) {
            (None, _) => Ty::Never,
            (Some(Ty::Unit), None) => Ty::Unit,
            (Some(bty), None) => bty,
            (Some(Ty::Unit), Some(cty)) => array!(cty),
            (_, Some(_)) => Err(BreakValueWhileCollection {
                src: solver.src(location),
                at: location.into(),
            })?,
        };
        Ok(ty)
    }
}

impl Solve for Match {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let scrut_ty = self.identity.query(solver)?.normalized(solver);

        let mut result: Option<Ty> = None;
        for case in self.cases.iter() {
            solver.ribs.push_block();
            solver.solve_match_pat(case.pat(), scrut_ty.clone(), false)?;
            if let Some(guard) = case.guard() {
                guard.fulfill_ty(Ty::Bool, solver)?;
            }
            let body = case.body();
            result = Some(match result.take() {
                None => body.query(solver)?,
                Some(acc) => {
                    // an error *inside* the arm is real -- only a failure to unify the arm's
                    // type with the earlier arms' may fall back to option coercion (otherwise
                    // the retry below re-reads a half-solved cache and swallows the error,
                    // leaving unresolved nodes for the emitter to panic on)
                    body.query(solver)?;
                    if let Err(e) = body.fulfill_ty(acc.clone(), solver) {
                        Ty::coerce_option(body.query(solver)?, acc, solver).ok_or(e)?
                    } else {
                        acc
                    }
                }
            });
            solver.ribs.pop();
        }

        let body_ty = result.unwrap_or(Ty::Unit);
        if !self.panic_terminator
            && let Some(report) =
                crate::exhaustion::missing_witnesses(&self.cases, &scrut_ty, solver)
        {
            let summary = report.summary();
            let help = report.variant_defs.is_empty().then(|| {
                "add the missing arms, an `_` wildcard arm, or a `!` panic terminator".to_string()
            });
            Err(NonExhaustiveMatch {
                src: solver.src(location),
                at: location.into(),
                summary,
                help,
                variant_defs: report.variant_defs,
            })?
        }
        Ok(body_ty)
    }
}

impl Solve for Return {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let mut expected_ty = solver
            .fn_stack
            .last()
            .ok_or_else(|| ReturnOutOfFunction {
                src: solver.src(location),
                at: location.into(),
            })?
            .expected_ty
            .clone();
        // the returned value fills the function's return slot
        if let Some(value) = &self.value {
            solver.hint_records(value, &expected_ty);
        }
        let mut this_ty = self
            .value
            .as_ref()
            .map_or(Ok(Ty::Unit), |e| e.query(solver))?;

        // `return &x` on an inout param: the lent place would outlive the call.
        // r[impl mimas.inout.escape]
        if let Some(value) = &self.value
            && let Some(source) = solver.place_sources.get(&value.id())
            && let Some((root, _)) = crate::root_and_path(source)
        {
            let root_dec = solver.ribs.resolve(root);
            if root_dec.is_some_and(|d| solver.inout_decs.contains(&d)) {
                Err(crate::errors::InoutEscape {
                    src: solver.src(root.location()),
                    at: root.location().into(),
                    name: root.lexeme.clone(),
                    via: "returned".to_string(),
                })?;
            }
        }

        this_ty
            .fulfill_ty(&mut expected_ty, solver)
            .map_err(|e| e.into_type_mismatch(solver, location))?;

        solver.control_flow.push(Flow::Return);

        Ok(Ty::Never)
    }
}

impl Solve for Struct {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        let Some(adt) = self.name.query(solver)?.as_adt().copied() else {
            unreachable!()
        };

        // push an impl scope so field types that reference this struct (e.g. `struct Foo { f: Foo
        // }`) get filter_adt-ed to `Ty::Identity` -- breaks the recursion check that would
        // otherwise refuse the self-referential adt.
        solver.ribs.push_impl(adt);
        // `struct Pair<A, B>`: field annotations resolve `A`/`B`/`Pair<A, B>` through the
        // declared params
        let generic = solver.adts[adt].is_generic();
        if generic {
            solver.push_adt_params(adt);
        }

        let is_tuple = solver.adts[adt].is_tuple_struct();

        let result = self.fields.iter().enumerate().try_for_each(|(i, field)| {
            let mut member_ty = if is_tuple {
                solver.adts[adt].as_tuple().members[i].clone()
            } else {
                solver.adts[adt]
                    .as_struct_mut()
                    .fields
                    .get_mut(&field.name.to_string())
                    .unwrap()
                    .ty
                    .clone()
            };

            let mut ty = Ty::from_annotation(field.annotation.clone(), solver)?;
            member_ty
                .fulfill_ty(&mut ty, solver)
                .map_err(|v| v.into_type_mismatch(solver, field.location))
        });

        if generic {
            solver.pop_type_params();
        }
        solver.ribs.pop();
        result?;

        let mut normalized = solver.adts[adt].clone();
        normalized.normalize_members(solver);
        solver.adts[adt] = normalized;

        Ok(Ty::adt(adt))
    }
}

impl Solve for Unary {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        fn unary(op: UnaryOp, ty: &Ty, location: Location, solver: &Solver) -> Result<Ty> {
            match (op, ty) {
                (UnaryOp::Not, Ty::Bool) => Ok(Ty::Bool),
                (UnaryOp::BitwiseNot, Ty::Int) => Ok(Ty::Int),
                (UnaryOp::Negative | UnaryOp::Positive, Ty::Int) => Ok(Ty::Int),
                (UnaryOp::Negative | UnaryOp::Positive, Ty::Float) => Ok(Ty::Float),
                // `-price` on a `dec` stays exact (r[impl mimas.dec])
                (UnaryOp::Negative | UnaryOp::Positive, Ty::Adt(id, _))
                    if solver.adts[*id].name == "dec" =>
                {
                    Ok(ty.clone())
                }
                (op, Ty::Tuple(elems)) => elems
                    .iter()
                    .map(|e| unary(op, e, location, solver))
                    .collect::<Result<Vec<Ty>>>()
                    .map(Ty::Tuple),
                // adts with registered operator impls (`Api::add_unary_op`)
                // typecheck here and dispatch at runtime, same as `eval`'s.
                (_, Ty::Adt(id, _)) if solver.adts[*id].flags.contains(AdtFlags::HAS_OPS) => {
                    Ok(ty.clone())
                }
                _ => Err(InvalidUnary {
                    src: solver.src(location),
                    at: location.into(),
                    ty: ty.to_string(),
                    op: op.to_string(),
                })?,
            }
        }

        // for an unresolved operand, pin it to the scalar shape this op needs before checking
        let ty = self.right.query(solver)?.normalized(solver);
        if matches!(ty, Ty::Vid(_)) {
            let pin = match self.op {
                UnaryOp::Not => Ty::Bool,
                UnaryOp::BitwiseNot => Ty::Int,
                UnaryOp::Negative | UnaryOp::Positive => return Ok(ty),
            };
            self.right.fulfill_ty(pin.clone(), solver)?;
            return Ok(pin);
        }
        if let Some(name) = un_method_name(self.op)
            && let Some(t) = unary_dispatch(solver, id, name, &self.right, &ty, location)?
        {
            return Ok(t);
        }
        unary(self.op, &ty, location, solver)
    }
}

impl Solve for Unwrap {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        let ty = self.expr.query(solver)?;
        let new_ty = match ty.normalized(solver) {
            Ty::Option(inner) | Ty::Result(inner) => *inner,
            ty => Err(Error::from(InvalidUnwrap {
                src: solver.src(self.expr.location()),
                at: self.expr.location().into(),
                ty: ty.to_string(),
            }))?,
        };
        Ok(new_ty)
    }
}

// r[impl mimas.propagate] r[impl mimas.propagate.where]
impl Solve for parse::Propagate {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        // `expr?` is `expr`'s value, or an early exit from the enclosing
        // function: `return null` from a `U?` function, the same raise (or a
        // raise saying what was null) from a `U!` one.
        let ty = self.expr.query(solver)?;
        let (inner, raised) = match ty.normalized(solver) {
            Ty::Option(inner) => (*inner, false),
            Ty::Result(inner) => (*inner, true),
            ty => Err(PropagateOnPlain {
                src: solver.src(self.expr.location()),
                at: self.expr.location().into(),
                ty: ty.to_string(),
            })?,
        };
        let Some(ret) = solver.fn_stack.last().map(|run| run.expected_ty.clone()) else {
            Err(PropagateOutsideFallible {
                src: solver.src(location),
                at: location.into(),
                why: "there's no function here to return from".into(),
                help:
                    "at the top level, `x!` stops with an error and `x ?? fallback` picks a value"
                        .into(),
            })?
        };
        let mut ret = ret.normalized(solver);
        // a closure with no return annotation: `?` makes it return an option
        if let Ty::Vid(v) = ret {
            let opt = Ty::Option(Box::new(Ty::Vid(solver.vid())));
            solver
                .register_sub(v, opt.clone())
                .map_err(|e| e.into_type_mismatch(solver, location))?;
            ret = opt;
        }
        let how = match ret {
            Ty::Option(_) => crate::Propagation::ReturnNull,
            Ty::Result(_) if raised => crate::Propagation::Reraise,
            Ty::Result(_) => crate::Propagation::RaiseNull(format!("`{}` was null", self.expr)),
            ret => Err(PropagateOutsideFallible {
                src: solver.src(location),
                at: location.into(),
                why: format!("this function returns `{ret}`, which can't carry a null or a raise"),
                help: format!(
                    "return `{ret}?` to pass the null on, or `{ret}!` to pass the raise on"
                ),
            })?,
        };
        solver.propagates.insert(id, how);
        Ok(inner)
    }
}

impl Solve for parse::Raise {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let expected_ty = solver
            .fn_stack
            .last()
            .ok_or_else(|| RaiseOutsideResult {
                src: solver.src(location),
                at: location.into(),
            })?
            .expected_ty
            .clone();

        if !matches!(expected_ty.normalized(solver), Ty::Result(_)) {
            Err(RaiseOutsideResult {
                src: solver.src(location),
                at: location.into(),
            })?
        }

        // `raise` carries any value — the payload stays structured until a
        // boundary stringifies it (`!` faults) or an `absolve` handler
        // narrows it by `match`.
        self.value.query(solver)?;
        solver.control_flow.push(Flow::Return);
        Ok(Ty::Never)
    }
}

impl Solve for parse::Range {
    fn solve(&self, _: NodeId, _: Location, solver: &mut Solver) -> Result<Ty> {
        self.start.fulfill_ty(Ty::Int, solver)?;
        self.end.fulfill_ty(Ty::Int, solver)?;
        Ok(Ty::Int)
    }
}

impl Solve for parse::Absolve {
    fn solve(&self, _id: NodeId, _location: Location, solver: &mut Solver) -> Result<Ty> {
        let lhs = self.left.query(solver)?.normalized(solver);
        let Ty::Result(inner) = lhs else {
            Err(AbsolveNonResult {
                src: solver.src(self.left.location()),
                at: self.left.location().into(),
                ty: lhs.to_string(),
            })?
        };
        let inner = *inner;
        // the handler sees the raised payload itself, not a stringified
        // note — `e: any`, narrowed by `match` or formatted on use. seed the
        // expected params before the closure solves (same as call-site args)
        // or the body's own use would pin `e` to something concrete first.
        let handler_ty = Ty::Fn(FnHeader::new(
            vec![FnParam::new(None, Ty::Any, false)],
            inner.clone(),
            false,
        ));
        if let ExprKind::Closure(_) = self.handler.kind()
            && let Ty::Fn(h) = &handler_ty
        {
            solver
                .expected_closures
                .insert(self.handler.id(), h.parameters.clone());
        }
        self.handler.fulfill_ty(handler_ty, solver)?;
        Ok(inner)
    }
}

impl Solve for While {
    fn solve(&self, _id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let mut pushed_rib = false;
        if let Some(binding) = self.binding.as_ref() {
            solver.ribs.push_block();
            pushed_rib = true;
            let header_ty = self.header.query(solver)?;
            let binding_ty = if let Ty::Option(ty) = header_ty {
                *ty
            } else {
                header_ty
            };
            solver.solve_pat(binding, binding_ty)?;
        } else {
            self.header.fulfill_ty(Ty::Bool, solver)?;
        };

        let LoopRun {
            break_ty,
            collect_ty,
            ..
        } = solver.run_loop_body(&self.body)?;

        if pushed_rib {
            solver.ribs.pop();
        }

        // Validate the types
        let break_ty = break_ty.unwrap_or_default();
        let ty = match (break_ty, collect_ty) {
            (Ty::Unit, None) => Ty::Unit,
            (bty, None) => Ty::Option(Box::new(bty)),
            (Ty::Unit, Some(cty)) => array!(cty),
            (_, Some(_)) => Err(BreakValueWhileCollection {
                src: solver.src(location),
                at: location.into(),
            })?,
        };
        Ok(ty)
    }
}

/// Pulls an `AdtId` out of an expression that names a type (the left of a `::`). A bare tuple
/// struct ident resolves to its constructor `Ty::Fn` via `Ident::solve`, so look through that to
/// reach the underlying adt.
fn adt_from_type_path(left: &Expr, solver: &mut Solver) -> Result<AdtId> {
    let ty = solver.resolve_path_head(left)?;
    match ty {
        Ty::Adt(adt, _) | Ty::Identity(adt, _) => Ok(adt),
        Ty::Fn(ref f) => match f.return_ty.as_ref() {
            Ty::Adt(adt, _) | Ty::Identity(adt, _) => Ok(*adt),
            _ => Err(TypeHasNoFields {
                src: solver.src(left.location()),
                at: left.location().into(),
                ty: ty.to_string(),
            })?,
        },
        _ => Err(TypeHasNoFields {
            src: solver.src(left.location()),
            at: left.location().into(),
            ty: ty.to_string(),
        })?,
    }
}

/// `S::default()` — the implicit memberwise-default constructor every
/// non-enum adt carries. No fn is minted: the call node lands in
/// `default_ctors` and codegen emits the `NewInstance` itself.
fn default_ctor(
    call: &Call,
    id: NodeId,
    right: &Ident,
    aid: AdtId,
    solver: &mut Solver,
) -> Result<Ty> {
    let name = solver.adts[aid].name.clone();
    if !call.arguments.is_empty() {
        Err(NoDefault {
            src: solver.src(right.location),
            at: right.location.into(),
            name: name.clone(),
            reason: "it takes no arguments".into(),
        })?
    }
    let ty = solver.instantiated_adt(aid);
    let args = match &ty {
        Ty::Adt(_, args) | Ty::Identity(_, args) => args.clone(),
        _ => vec![],
    };
    check_defaultable(solver, &name, aid, &args, right.location, &mut vec![aid])?;
    solver.default_ctors.insert(id, aid);
    solver.shadow_expr_ty(call.left.id(), Ty::adt(aid), call.left.location())?;
    Ok(ty.normalized(solver))
}

/// Every member of `aid` must admit a default value for `::default()` to
/// exist. `call_name` names the type in errors (the call-site spelling,
/// not a nested member's owner); `at` labels the `default` in the source.
fn check_defaultable(
    solver: &mut Solver,
    call_name: &str,
    aid: AdtId,
    args: &[Ty],
    at: Location,
    visiting: &mut Vec<AdtId>,
) -> Result<()> {
    let adt = &solver.adts[aid];
    if adt.flags.contains(AdtFlags::IS_ENUM) {
        let name = adt.name.clone();
        Err(NoDefault {
            src: solver.src(at),
            at: at.into(),
            name: call_name.into(),
            reason: format!("`{name}` is an enum -- a default can't pick a variant"),
        })?
    }
    if adt
        .flags
        .intersects(AdtFlags::IS_MODULE | AdtFlags::IS_BUILTIN)
    {
        let name = adt.name.clone();
        Err(NoDefault {
            src: solver.src(at),
            at: at.into(),
            name: call_name.into(),
            reason: format!("`{name}` is a built-in -- no memberwise default exists"),
        })?
    }
    let members: Vec<(String, Location, Ty)> = match adt.as_singular() {
        Some(Variant::Struct(v)) => v
            .fields
            .iter()
            .map(|(n, f)| (n.clone(), f.declaration_location, f.ty.clone()))
            .collect(),
        Some(Variant::Tuple(v)) => v
            .members
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("member {i}"), at, t.clone()))
            .collect(),
        None => vec![],
    };
    for (member, loc, ty) in members {
        let spec = solver.specialize(&ty, aid, args).normalized(solver);
        check_member_defaultable(solver, call_name, spec, &member, loc, visiting)?;
    }
    Ok(())
}

/// One member's defaultability. Containers always default (`[]`, `dict::new()`,
/// `null`) without ever materializing an element, so their inners aren't
/// visited; a nested struct member recurses; anything opaque reports.
fn check_member_defaultable(
    solver: &mut Solver,
    call_name: &str,
    ty: Ty,
    member: &str,
    loc: Location,
    visiting: &mut Vec<AdtId>,
) -> Result<()> {
    let fail = |solver: &Solver, reason: String| -> Error {
        NoDefault {
            src: solver.src(loc),
            at: loc.into(),
            name: call_name.into(),
            reason,
        }
        .into()
    };
    match ty {
        Ty::Unit | Ty::Null | Ty::Bool | Ty::Int | Ty::Float | Ty::Str => Ok(()),
        Ty::Array(..) | Ty::Dict(_) | Ty::Option(_) => Ok(()),
        Ty::Tuple(members) => {
            for m in members {
                check_member_defaultable(solver, call_name, m, member, loc, visiting)?;
            }
            Ok(())
        }
        Ty::Adt(inner, iargs) | Ty::Identity(inner, iargs) => {
            if visiting.contains(&inner) {
                Err(fail(
                    solver,
                    format!(
                        "`{member}` recurses through `{}` -- no default",
                        solver.adts[inner].name
                    ),
                ))?
            }
            visiting.push(inner);
            let r = check_defaultable(solver, call_name, inner, &iargs, loc, visiting);
            visiting.pop();
            r
        }
        Ty::Param(_) | Ty::Vid(_) | Ty::Anon(_) => Err(fail(
            solver,
            format!("`{member}`'s type is generic -- write `default` in an `impl`"),
        ))?,
        other => Err(fail(
            solver,
            format!("`{member}: {other}` has no default value"),
        ))?,
    }
}

/// Among `name`'s primary (`dec`, typed `ty`) and its operand-typed overloads
/// on `aid`, the one whose operand `call`'s single positional argument fits.
/// The primary stands when there are no overloads, the call isn't
/// single-argument, or nothing fits (the call's own checking then reports
/// the mismatch against the primary). r[impl mimas.op.overload]
fn pick_source_overload(
    solver: &mut Solver,
    aid: AdtId,
    name: &str,
    recv_args: &[Ty],
    call: &Call,
    dec: crate::components::DecId,
    ty: Ty,
) -> Result<(crate::components::DecId, Ty)> {
    let overloads = solver.adts[aid]
        .overloads
        .get(name)
        .cloned()
        .unwrap_or_default();
    let [arg] = call.arguments.as_slice() else {
        return Ok((dec, ty));
    };
    if overloads.is_empty() || arg.name.is_some() {
        return Ok((dec, ty));
    }
    let mut candidates = vec![(dec, ty.clone())];
    for field in overloads {
        solver.check_vis(field.dec, call.left.location())?;
        let sig = solver.specialize(&field.ty, aid, recv_args);
        candidates.push((field.dec, sig));
    }
    for (dec, sig) in &candidates {
        let Ty::Fn(f) = sig else { continue };
        let [_, operand] = f.parameters.as_slice() else {
            continue;
        };
        if arg.value.fulfill_ty(operand.ty.clone(), solver).is_ok() {
            return Ok((*dec, sig.clone()));
        }
    }
    Ok((dec, ty))
}

/// The declared type of an operator-shaped method's one operand (`self` plus
/// one written parameter), or `None`. The pact overload pairing's key.
fn overload_operand_of(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Fn(h) if h.is_method && h.parameters.len() == 2 => Some(h.parameters[1].ty.clone()),
        _ => None,
    }
}

/// `a <op> b` with `a` (or `b`) pact-typed or a bounded type param: the pact
/// that declares the operator's method (`mul`, or `rmul` on the right) owns
/// the operator, and the call dispatches at runtime over the bound's
/// implementers. Among an overload group the first member whose operand the
/// other side fits wins. `Self` in the signature reads as the operand's own
/// type, so `x * x` on `x: T` stays `T`. Recorded in `pact_ops`.
/// r[impl mimas.pact.binary] r[impl mimas.pact.overload]
#[allow(clippy::too_many_arguments)]
fn pact_binary_dispatch(
    solver: &mut Solver,
    id: NodeId,
    names: (&'static str, &'static str),
    left: &Expr,
    right: &Expr,
    lhs: &Ty,
    rhs: &Ty,
    location: Location,
) -> Result<Option<Ty>> {
    for (recv_ty, name, arg, swap) in [(lhs, names.0, right, false), (rhs, names.1, left, true)] {
        let Some(pids) = solver.pacts_of(recv_ty) else {
            continue;
        };
        // the pact that declares the name, and its overload group
        let Some((pid, keys)) = pids.iter().find_map(|pid| {
            let pact = &solver.pacts[pid];
            if !pact.functions.contains_key(name) {
                return None;
            }
            let keys = solver
                .pact_groups
                .get(&(*pid, name.to_string()))
                .cloned()
                .unwrap_or_else(|| vec![name.to_string()]);
            Some((*pid, keys))
        }) else {
            continue;
        };
        for key in keys {
            let (header, _) = solver.pacts[&pid].functions[&key].clone();
            let Ty::Fn(f) = solver.instantiate_pact_fn(&header, recv_ty) else {
                continue;
            };
            let [_, operand] = f.parameters.as_slice() else {
                continue;
            };
            if arg.fulfill_ty(operand.ty.clone(), solver).is_err() {
                continue;
            }
            solver.check_vis(solver.pact_members[&(pid, key.clone())], location)?;
            solver.pact_ops.insert(
                id,
                crate::PactOp {
                    pids: pids.clone(),
                    method: key,
                    swap,
                },
            );
            return Ok(Some(f.return_ty.as_ref().clone()));
        }
    }
    Ok(None)
}

/// Two operand types that name the same thing: the impl target written as
/// itself (`o: R` inside `impl Num for R` is `Identity(R)`) matches the pact's
/// `Self` instantiated to the target (`Adt(R)`).
fn same_operand(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Adt(x, xa) | Ty::Identity(x, xa), Ty::Adt(y, ya) | Ty::Identity(y, ya)) => {
            x == y && xa == ya
        }
        _ => a == b,
    }
}
