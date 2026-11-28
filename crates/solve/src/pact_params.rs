//! Pacts with type parameters, `pact Add<Rhs = Self> { type Out; … }`.
//!
//! A pact that declares parameters is a template. Each list of arguments a
//! bound, an impl or an annotation gives it (`Add<float>`, or `Add` with its
//! defaults) is one *instance*: a pact of its own, minted once per argument
//! list, whose signatures read the arguments in place of the parameters.
//! Everything keyed by a `PactId` (impls, associated-type bindings, member
//! keys, runtime dispatch) then works per instance, so a type may implement
//! `Mul<Vec2>` and `Mul<float>` side by side.
//!
//! An argument that names the implementing type is written `Self` in the
//! instance's key, so `impl Add for Vec2`, `impl Add<Vec2> for Vec2` and a
//! bound `T: Add` all meet at the one instance `Add<Self>`.
//! r[impl mimas.pact.params]

use crate::{
    FxHashMap, Result, Solver,
    components::*,
    errors::{NotAPact, PactArgs},
};
use parse::{
    Ident,
    components::{Annotation, Bound},
};
use shared::{Location, PactId, ParamId};

/// One declared parameter of a pact: its name, its stand-in param in the
/// template's signatures, and its default (`Self` as the template's
/// `Skolem`).
#[derive(Debug, Clone)]
pub(crate) struct PactParam {
    pub name: String,
    pub stand_in: ParamId,
    pub default: Option<Ty>,
}

impl Solver {
    /// Replace every `Skolem(from)` with `to`, and every param in `map`.
    pub(crate) fn subst_pact_ty(t: &Ty, map: &FxHashMap<ParamId, Ty>, from: PactId, to: &Ty) -> Ty {
        let go = |t: &Ty| Self::subst_pact_ty(t, map, from, to);
        match t {
            Ty::Param(p) => map.get(p).cloned().unwrap_or_else(|| t.clone()),
            Ty::Skolem(p) if *p == from => to.clone(),
            Ty::Array(v, len) => Ty::Array(Box::new(go(v)), *len),
            Ty::Dict(k, v) => Ty::Dict(Box::new(go(k)), Box::new(go(v))),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(go).collect()),
            Ty::Option(v) => Ty::Option(Box::new(go(v))),
            Ty::Result(v) => Ty::Result(Box::new(go(v))),
            Ty::Adt(a, args) => Ty::Adt(*a, args.iter().map(go).collect()),
            Ty::Identity(a, args) => Ty::Identity(*a, args.iter().map(go).collect()),
            Ty::Fn(h) => {
                let mut h2 = FnHeader::new(
                    h.parameters
                        .iter()
                        .map(|p| FnParam::new(p.name.clone(), go(&p.ty), p.has_default))
                        .collect(),
                    go(&h.return_ty),
                    h.is_method,
                );
                h2.is_ctor = h.is_ctor;
                Ty::Fn(h2)
            }
            other => other.clone(),
        }
    }

    /// `t` with each occurrence of `self_ty` written as `Skolem(base)`, the
    /// key's `Self`.
    fn canon_self(&self, t: &Ty, self_ty: &Ty, base: PactId) -> Ty {
        if same_ty(t, self_ty) {
            return Ty::Skolem(base);
        }
        let go = |t: &Ty| self.canon_self(t, self_ty, base);
        match t {
            Ty::Array(v, len) => Ty::Array(Box::new(go(v)), *len),
            Ty::Dict(k, v) => Ty::Dict(Box::new(go(k)), Box::new(go(v))),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(go).collect()),
            Ty::Option(v) => Ty::Option(Box::new(go(v))),
            Ty::Result(v) => Ty::Result(Box::new(go(v))),
            Ty::Adt(a, args) => Ty::Adt(*a, args.iter().map(go).collect()),
            Ty::Identity(a, args) => Ty::Adt(*a, args.iter().map(go).collect()),
            other => other.clone(),
        }
    }

    /// The instance of `base` for the full, canonical argument list `args`
    /// (`Self` as `Skolem(base)`), minted on first use.
    pub(crate) fn pact_instance(&mut self, base: PactId, args: Vec<Ty>) -> PactId {
        let key = (base, format!("{args:?}"));
        if let Some(p) = self.pact_insts.get(&key) {
            return *p;
        }
        let bp = self.pacts[base].clone();
        let shown: Vec<String> = args
            .iter()
            .map(|a| match a {
                Ty::Skolem(p) if *p == base => "Self".to_string(),
                other => other.to_string(),
            })
            .collect();
        let name = format!("{}<{}>", bp.name, shown.join(", "));
        let inst = self.pacts.push(shared::Pact {
            name: name.clone(),
            requires: bp.requires.clone(),
            functions: indexmap::IndexMap::default(),
            constants: indexmap::IndexMap::default(),
        });
        shared::name_pact(inst.index(), &name);
        let me = Ty::Skolem(inst);
        let args: Vec<Ty> = args
            .iter()
            .map(|a| Self::subst_pact_ty(a, &FxHashMap::default(), base, &me))
            .collect();
        let params = self.pact_params.get(&base).cloned().unwrap_or_default();
        let mut map: FxHashMap<ParamId, Ty> = params
            .iter()
            .zip(&args)
            .map(|(p, a)| (p.stand_in, a.clone()))
            .collect();
        // each instance has its own associated-type stand-ins: a binding is
        // per (instance, type), and so is a bounded param's opaque `T::Out`
        let assoc = self.pact_assoc.get(&base).cloned().unwrap_or_default();
        for item in &assoc {
            let old = self
                .assoc_params
                .iter()
                .find(|(_, (p, n))| *p == base && n == item)
                .map(|(k, _)| *k);
            // a fresh node id keeps it clear of every declared param's id
            // (and small: param names live in a vector indexed by id)
            let fresh = ParamId::from(parse::NodeId::new().index() as u32);
            shared::name_param(fresh.index(), item);
            self.assoc_params.insert(fresh, (inst, item.clone()));
            if let Some(old) = old {
                map.insert(old, Ty::Param(fresh));
            }
        }
        if !assoc.is_empty() {
            self.pact_assoc.insert(inst, assoc);
        }
        for (k, (h, default)) in &bp.functions {
            let Ty::Fn(h2) = Self::subst_pact_ty(&Ty::Fn(h.clone()), &map, base, &me) else {
                unreachable!("a header substitutes to a header")
            };
            let base_dec = self.pact_members[&(base, k.clone())];
            // a default body is solved once, against the template; every
            // instance's graft runs that one body
            let dec = if default.is_some() {
                base_dec
            } else {
                let at = self.decs[base_dec].location;
                self.dec_id(
                    &Ident::new(format!("{k}@{name}"), at),
                    Ty::Fn(h2.clone()),
                    DecKind::Item {
                        defaults: vec![],
                        wheres: vec![],
                    },
                    Vis::Public,
                )
            };
            self.pact_members.insert((inst, k.clone()), dec);
            self.pacts[inst].functions.insert(k.clone(), (h2, default.clone()));
        }
        for (k, t) in &bp.constants {
            let t2 = Self::subst_pact_ty(t, &map, base, &me);
            let base_dec = self.pact_members[&(base, k.clone())];
            self.pact_members.insert((inst, k.clone()), base_dec);
            self.pacts[inst].constants.insert(k.clone(), t2);
        }
        let groups: Vec<(String, Vec<String>)> = self
            .pact_groups
            .iter()
            .filter(|((p, _), _)| *p == base)
            .map(|((_, n), g)| (n.clone(), g.clone()))
            .collect();
        for (n, g) in groups {
            self.pact_groups.insert((inst, n), g);
        }
        let req_pins: Vec<(PactId, Vec<(String, Ty)>)> = self
            .require_assoc
            .iter()
            .filter(|((p, _), _)| *p == base)
            .map(|((_, q), pins)| (*q, pins.clone()))
            .collect();
        for (q, pins) in req_pins {
            let pins = pins
                .iter()
                .map(|(n, t)| (n.clone(), Self::subst_pact_ty(t, &map, base, &me)))
                .collect();
            self.require_assoc.insert((inst, q), pins);
        }
        self.inst_of.insert(inst, (base, args));
        self.pact_insts.insert(key, inst);
        inst
    }

    /// `base` applied to `explicit` (already resolved), the rest from the
    /// defaults, with `self_ty` (the implementer, or the bounded param)
    /// written as `Self`. A pact with no parameters is itself.
    pub(crate) fn apply_pact(
        &mut self,
        base: PactId,
        explicit: Vec<Ty>,
        self_ty: Option<&Ty>,
        at: Location,
    ) -> Result<PactId> {
        let params = self.pact_params.get(&base).cloned().unwrap_or_default();
        let name = self.pacts[base].name.clone();
        if params.is_empty() {
            if !explicit.is_empty() {
                return Err(PactArgs {
                    src: self.src(at),
                    at: at.into(),
                    msg: format!("`{name}` takes no type arguments"),
                    label: "declared as `pact` with no `<…>`".into(),
                }
                .into());
            }
            return Ok(base);
        }
        if explicit.len() > params.len() {
            return Err(PactArgs {
                src: self.src(at),
                at: at.into(),
                msg: format!(
                    "`{name}` takes {} type argument(s), not {}",
                    params.len(),
                    explicit.len()
                ),
                label: "too many".into(),
            }
            .into());
        }
        let mut args: Vec<Ty> = explicit
            .iter()
            .map(|a| match self_ty {
                Some(s) => self.canon_self(a, s, base),
                None => a.clone(),
            })
            .collect();
        for p in &params[args.len()..] {
            let Some(d) = &p.default else {
                return Err(PactArgs {
                    src: self.src(at),
                    at: at.into(),
                    msg: format!("`{name}` needs an argument for `{}`", p.name),
                    label: format!("write `{name}<…>`"),
                }
                .into());
            };
            // a default may name an earlier parameter (`pact P<A, B = A>`)
            let map: FxHashMap<ParamId, Ty> = params
                .iter()
                .zip(&args)
                .map(|(q, a)| (q.stand_in, a.clone()))
                .collect();
            let d = Self::subst_pact_ty(d, &map, PactId::from(u32::MAX), &Ty::Unit);
            args.push(d);
        }
        Ok(self.pact_instance(base, args))
    }

    /// The pact a bound names, applied to its arguments, with `self_ty` as
    /// `Self`, and its associated-type pins (`Add<Out = T>`). `None` when the
    /// name does not resolve yet (the pact is declared later).
    pub(crate) fn resolve_bound(
        &mut self,
        bound: &Bound,
        self_ty: &Ty,
    ) -> Result<Option<(PactId, Vec<(String, Ty)>)>> {
        let Some(base) = self
            .resolve_name(bound, bound.location)
            .ok()
            .and_then(|t| t.as_single_pact())
        else {
            return Ok(None);
        };
        self.apply_bound(base, bound, self_ty).map(Some)
    }

    pub(crate) fn apply_bound(
        &mut self,
        base: PactId,
        bound: &Bound,
        self_ty: &Ty,
    ) -> Result<(PactId, Vec<(String, Ty)>)> {
        let explicit = bound
            .args
            .iter()
            .map(|a| Ty::from_annotation(a.clone(), self))
            .collect::<Result<Vec<_>>>()?;
        let pid = self.apply_pact(base, explicit, Some(self_ty), bound.location)?;
        let mut pins = vec![];
        let declared: Vec<String> = self
            .pact_closure(pid)
            .into_iter()
            .flat_map(|p| self.pact_assoc.get(&p).cloned().unwrap_or_default())
            .collect();
        for (n, ann) in &bound.assoc {
            if !declared.contains(&n.lexeme) {
                return Err(crate::errors::NotFound {
                    src: self.src(n.location),
                    at: n.location.into(),
                    name: n.lexeme.clone(),
                    help: Some(format!(
                        "`{}` declares no `type {}`",
                        self.pacts[pid].name, n.lexeme
                    )),
                }
                .into());
            }
            pins.push((n.lexeme.clone(), Ty::from_annotation(ann.clone(), self)?));
        }
        Ok((pid, pins))
    }

    /// The pact an `impl` implements, applied to its arguments with the
    /// target as `Self`. Resolves the arguments with the target's own params
    /// in scope.
    pub(crate) fn impl_pact(&mut self, imp: &parse::Impl, adt: AdtId) -> Result<Option<PactId>> {
        let Some(name) = imp.pact.as_ref() else {
            return Ok(None);
        };
        let ty = self.resolve_name(name, name.location)?;
        let Some(base) = ty.as_single_pact() else {
            return Err(NotAPact {
                src: self.src(name.location),
                at: name.location.into(),
                ty: ty.to_string(),
            }
            .into());
        };
        if imp.pact_args.is_empty() && !self.pact_params.contains_key(&base) {
            return Ok(Some(base));
        }
        self.ribs.push_impl(adt);
        self.push_adt_params(adt);
        let args = imp
            .pact_args
            .iter()
            .map(|a| Ty::from_annotation(a.clone(), self))
            .collect::<Result<Vec<_>>>();
        self.pop_type_params();
        self.ribs.pop();
        let args = args?;
        let self_ty = self.adt_self_ty(adt);
        self.apply_pact(base, args, Some(&self_ty), name.location).map(Some)
    }

    /// Do `ty`'s impls cover `pact` with the associated types `pins`? An
    /// instance matches an impl of a sibling instance whose arguments name
    /// the target's own params (`impl Add<T> for Stream<T>` covers
    /// `Stream<int>: Add<int>`).
    pub(crate) fn implements(&mut self, ty: &Ty, pact: PactId, pins: &[(String, Ty)]) -> bool {
        let ty = ty.clone().normalized(self);
        let (aid, args) = match &ty {
            Ty::Vid(_) | Ty::Never => return true,
            Ty::Pacts(pids) => return pids.iter().any(|p| self.pact_closure(*p).contains(&pact)),
            Ty::Skolem(p) => return self.pact_closure(*p).contains(&pact),
            Ty::Param(p) => {
                return self.param_bounds.get(p).is_some_and(|b| {
                    b.iter().any(|bound| self.pact_closure(*bound).contains(&pact))
                });
            }
            Ty::Adt(a, args) | Ty::Identity(a, args) => (*a, args.clone()),
            other => match self.builtin_adt(other) {
                Some(a) => (a, vec![]),
                None => return false,
            },
        };
        let mut candidates = vec![];
        if self.pact_impls.contains(&(pact, aid)) {
            candidates.push(pact);
        } else {
            candidates.extend(self.sibling_impls(pact, aid, &args, &ty));
        }
        // a sibling impl answers this instance: record it, so runtime
        // dispatch through the bound finds the impl's members
        // (r[impl mimas.pact.params.generic-target])
        if let Some(&i) = candidates.first()
            && i != pact
            && !self.pact_impls.contains(&(pact, aid))
        {
            self.inst_alias.insert((pact, aid), i);
        }
        candidates.into_iter().any(|c| {
            pins.iter().all(|(n, want)| {
                let Some(item) = self
                    .assoc_params
                    .iter()
                    .find(|(_, (p, m))| *p == c && m == n)
                    .map(|(k, _)| *k)
                else {
                    return false;
                };
                match self.assoc_for(item, &ty) {
                    Some(have) => loose_eq(self, &have, want),
                    None => false,
                }
            })
        })
    }

    /// The instances of `pact`'s template that `aid` implements and whose
    /// arguments, read for `ty` (an `aid` over `args`), equal `pact`'s.
    pub(crate) fn sibling_impls(&mut self, pact: PactId, aid: AdtId, args: &[Ty], ty: &Ty) -> Vec<PactId> {
        let Some((base, want)) = self.inst_of.get(&pact).cloned() else {
            return vec![];
        };
        let siblings: Vec<(PactId, Vec<Ty>)> = self
            .inst_of
            .iter()
            .filter(|(i, (b, _))| *b == base && **i != pact && self.pact_impls.contains(&(**i, aid)))
            .map(|(i, (_, a))| (*i, a.clone()))
            .collect();
        let mut out = vec![];
        for (i, have) in siblings {
            let fits = have.iter().zip(&want).all(|(h, w)| {
                let h = Self::subst_pact_ty(h, &FxHashMap::default(), i, ty);
                let h = self.specialize(&h, aid, args);
                let w = Self::subst_pact_ty(w, &FxHashMap::default(), pact, ty);
                loose_eq(self, &h, &w)
            });
            if fits {
                out.push(i);
            }
        }
        out.sort_by_key(|p| p.index());
        out
    }

    /// The pins on `owner`'s associated type `name` of `pact`: a bound's
    /// own (`T: Add<Out = T>`), or one a required pact states (`pact G:
    /// Add<Out = Self>`), with `Self` read as `owner`.
    pub(crate) fn assoc_pin(&self, owner: &Ty, pact: PactId, name: &str) -> Option<Ty> {
        let pick = |pins: &Vec<(String, Ty)>| {
            pins.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone())
        };
        let mut roots: Vec<PactId> = vec![];
        match owner {
            Ty::Param(p) => {
                for b in self.param_bounds.get(p).into_iter().flatten() {
                    if let Some(t) = self.bound_assoc.get(&(*p, *b)).filter(|_| *b == pact).and_then(pick) {
                        return Some(t);
                    }
                    roots.push(*b);
                }
            }
            Ty::Skolem(q) => roots.push(*q),
            _ => return None,
        }
        for r in roots {
            for c in self.pact_closure(r) {
                if let Some(t) = self.require_assoc.get(&(c, pact)).and_then(pick) {
                    // the pin's `Self` is the requiring pact's: read it as the owner
                    return Some(Self::subst_pact_ty(&t, &FxHashMap::default(), c, owner));
                }
            }
        }
        None
    }
}

/// Two types that name the same thing, where `Identity` and `Adt` agree.
pub(crate) fn same_ty(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Adt(x, xa) | Ty::Identity(x, xa), Ty::Adt(y, ya) | Ty::Identity(y, ya)) => {
            x == y && xa.len() == ya.len() && xa.iter().zip(ya).all(|(p, q)| same_ty(p, q))
        }
        _ => a == b,
    }
}

/// Structural equality where an open var or a param matches anything: a
/// bound obligation is held to what is known.
pub(crate) fn loose_eq(solver: &Solver, a: &Ty, b: &Ty) -> bool {
    let a = a.clone().normalized(solver);
    let b = b.clone().normalized(solver);
    match (&a, &b) {
        (Ty::Vid(_) | Ty::Param(_), _) | (_, Ty::Vid(_) | Ty::Param(_)) => true,
        (Ty::Adt(x, xa) | Ty::Identity(x, xa), Ty::Adt(y, ya) | Ty::Identity(y, ya)) => {
            x == y && xa.len() == ya.len() && xa.iter().zip(ya).all(|(p, q)| loose_eq(solver, p, q))
        }
        (Ty::Array(x, _), Ty::Array(y, _))
        | (Ty::Option(x), Ty::Option(y))
        | (Ty::Result(x), Ty::Result(y)) => loose_eq(solver, x, y),
        (Ty::Dict(k1, v1), Ty::Dict(k2, v2)) => loose_eq(solver, k1, k2) && loose_eq(solver, v1, v2),
        (Ty::Tuple(x), Ty::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| loose_eq(solver, p, q))
        }
        _ => a == b,
    }
}

impl Solver {
    /// `pact Add<Rhs = Self>`: register the parameters' stand-ins, in scope
    /// (with `assoc_scope`) while the member signatures resolve. The
    /// defaults resolve with `Self` as the pact's own.
    pub(crate) fn declare_pact_params(
        &mut self,
        pact: PactId,
        params: &[(Ident, Option<Annotation>)],
    ) -> Result<FxHashMap<String, Ty>> {
        let mut scope = FxHashMap::default();
        let mut out = vec![];
        for (name, default) in params {
            let p = Solver::param_id_of(name);
            scope.insert(name.lexeme.clone(), Ty::Param(p));
            self.assoc_scope.push(scope.clone());
            let default = default
                .as_ref()
                .map(|d| Ty::from_annotation(d.clone(), self));
            self.assoc_scope.pop();
            out.push(PactParam {
                name: name.lexeme.clone(),
                stand_in: p,
                default: default.transpose()?,
            });
        }
        if !out.is_empty() {
            self.pact_params.insert(pact, out);
        }
        Ok(scope)
    }
}
