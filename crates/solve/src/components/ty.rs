use crate::{
    Result, Solver, Unification, UnificationError,
    components::AdtFlags,
    errors::{FieldNotFound, NotAPact, NotAStruct, TypeHasNoFields, UnknownUnit},
    traits::Query,
};
use parse::{components::Annotation, lex::TyKw};
pub use shared::{AdtId, FnHeader, FnParam, PactId, Ty, Vid};

pub trait TyExt: Sized {
    fn occurs(&self, other: Vid, solver: &Solver) -> bool;
    fn from_annotation(annotation: Annotation, solver: &mut Solver) -> Result<Ty>;
    fn coerce_option(a: Ty, b: Ty, solver: &mut Solver) -> Option<Ty>;
    fn coerce_pacts(a: Ty, b: Ty, solver: &mut Solver) -> Option<Ty>;
    fn fulfill_ty(
        &mut self,
        other: &mut Ty,
        solver: &mut Solver,
    ) -> std::result::Result<(), UnificationError>;
    fn normalized(self, solver: &Solver) -> Ty;
    fn filter_adt(&self, adt: AdtId) -> Ty;
}

impl TyExt for Ty {
    fn occurs(&self, other: Vid, solver: &Solver) -> bool {
        match self {
            Ty::Vid(vid) if *vid == other => true,
            Ty::Vid(vid) => solver.sub(*vid).is_some_and(|v| v.occurs(other, solver)),
            Ty::Array(ty) | Ty::Dict(ty) => ty.occurs(other, solver),
            Ty::Fn(fn_data) => {
                fn_data
                    .parameters
                    .iter()
                    .any(|p| p.ty.occurs(other, solver))
                    || fn_data.return_ty.occurs(other, solver)
            }
            Ty::Tuple(members) => members.iter().any(|v| v.occurs(other, solver)),
            // adt bodies are nominal: the type args are part of the *head*, so `X = List<X>`
            // must fail the occurs check (it would form an infinite type), while a vid inside a
            // field type can never form one -- the adt itself doesn't unfold.
            Ty::Adt(_, args) | Ty::Identity(_, args) => {
                args.iter().any(|a| a.occurs(other, solver))
            }
            Ty::Option(inner) | Ty::Result(inner) => inner.occurs(other, solver),
            Ty::Anon(_)
            | Ty::Param(_)
            | Ty::Pacts(_) // annotations always required, never holds vids
            | Ty::Skolem(_)
            | Ty::Unit
            | Ty::Never
            | Ty::Null
            | Ty::Bool
            | Ty::Int
            | Ty::Float
            | Ty::Str => false,
        }
    }

    fn coerce_option(a: Ty, b: Ty, solver: &mut Solver) -> Option<Ty> {
        match (a, b) {
            (Ty::Null, ty) | (ty, Ty::Null) => {
                if ty.clone().normalized(solver) != Ty::Null {
                    Some(Ty::Option(Box::new(ty)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Widen two otherwise-incompatible types to the pacts they share, if any. Without
    /// generics, this is the only way to build a collection of "things that implement X":
    /// `[Square, Circle]` settles on `[Draw]` rather than forcing every element into the
    /// first one's concrete type. Only consulted after unification has already failed, so
    /// it can turn an error into a success but never change a program that already checked.
    fn coerce_pacts(a: Ty, b: Ty, solver: &mut Solver) -> Option<Ty> {
        fn bounds(ty: &Ty, solver: &Solver) -> Option<Vec<PactId>> {
            match ty {
                Ty::Adt(aid, _) | Ty::Identity(aid, _) => Some(
                    solver
                        .pact_impls
                        .iter()
                        .filter(|(_, impl_adt)| impl_adt == aid)
                        .map(|(pid, _)| *pid)
                        .collect(),
                ),
                _ => ty.as_pacts(),
            }
        }

        let a = a.normalized(solver);
        let b = b.normalized(solver);
        let (lhs, rhs) = (bounds(&a, solver)?, bounds(&b, solver)?);
        let shared: Vec<_> = lhs.into_iter().filter(|pid| rhs.contains(pid)).collect();
        // `Ty::pacts` sorts and dedups -- `Ty::Pacts` compares as a plain Vec, so that
        // normalization is what lets two independently derived bounds come out equal
        (!shared.is_empty()).then(|| Ty::pacts(shared))
    }

    fn from_annotation(annotation: Annotation, solver: &mut Solver) -> Result<Ty> {
        match annotation {
            Annotation::Unit => Ok(Ty::Unit),
            Annotation::Poison(poison) => poison.escaped(),
            Annotation::Kw(kw) => Ok(ty_from_kw(kw)),
            Annotation::Option(ty) => Ok(Ty::Option(Box::new(Ty::from_annotation(*ty, solver)?))),
            Annotation::Result(ty) => Ok(Ty::Result(Box::new(Ty::from_annotation(*ty, solver)?))),
            Annotation::Array(ty) => Ok(Ty::Array(Box::new(Ty::from_annotation(*ty, solver)?))),
            Annotation::Dictionary(ty) => Ok(Ty::Dict(Box::new(Ty::from_annotation(*ty, solver)?))),
            Annotation::Function(params, ret) => {
                let parameters = params
                    .into_iter()
                    .map(|p| Ty::from_annotation(p, solver).map(|t| FnParam::new(None, t, false)))
                    .collect::<Result<_>>()?;
                let return_ty = Ty::from_annotation(*ret, solver)?;
                Ok(Ty::Fn(FnHeader::new(parameters, return_ty, false)))
            }
            Annotation::Tuple(members) => Ok(Ty::Tuple(
                members
                    .into_iter()
                    .map(|v| Ty::from_annotation(v, solver))
                    .collect::<Result<_>>()?,
            )),
            // a declared type param (`fn map<T>`, `struct Pair<A>`) -- looked up before
            // anything else so a param named like a unit or a builtin still wins
            Annotation::Ty(ident) if solver.type_param(&ident).is_some() => {
                let ty = Ty::Param(solver.type_param(&ident).unwrap());
                solver.note(&ident, ty.clone(), None);
                Ok(ty)
            }
            // a unit written where a type goes (`kW`, `usd`) is a float whose dimension the
            // checker tracks -- unless the program declares a type by that name, which wins
            Annotation::Ty(ident)
                if solver.ribs.resolve(&ident).is_none()
                    && shared::units::lookup(&ident.lexeme).is_some() =>
            {
                Ok(Ty::Float)
            }
            // `Pair<int, str>` (or `mod::Pair<int, str>`) applies a generic adt;
            // `Interval<kW>` is an `Interval` to the type checker -- the unit is the
            // dimension pass's
            Annotation::Applied(ty, args) => {
                let head_ann = if ty.len() == 1 {
                    Annotation::Ty(ty[0].clone())
                } else {
                    Annotation::Path(ty.clone())
                };
                let name_ident = ty.last().expect("applied annotation has a head");
                let head = Ty::from_annotation(head_ann, solver)?.normalized(solver);
                let Some((aid, head_args)) = head.applied() else {
                    Err(crate::errors::NotGeneric {
                        src: solver.src(name_ident.location),
                        at: name_ident.location.into(),
                        name: name_ident.lexeme.clone(),
                    })?
                };
                let expected = solver.adts[aid].type_params.len();
                if expected == 0 {
                    // a `<...>` on a non-generic type is the old unit-annotation syntax --
                    // only allowed when every argument is a unit (`Interval<kW>`), not a type
                    let all_units = args.iter().all(|a| match a {
                        Annotation::Ty(ident) => {
                            solver.ribs.resolve(&ident).is_none()
                                && shared::units::lookup(&ident.lexeme).is_some()
                        }
                        Annotation::Quantity(_) => true,
                        _ => false,
                    });
                    if all_units {
                        return Ok(head);
                    }
                    Err(crate::errors::NotGeneric {
                        src: solver.src(name_ident.location),
                        at: name_ident.location.into(),
                        name: name_ident.lexeme.clone(),
                    })?
                }
                if args.len() != expected || !head_args.is_empty() {
                    Err(crate::errors::GenericArity {
                        src: solver.src(name_ident.location),
                        at: name_ident.location.into(),
                        name: solver.adts[aid].name.clone(),
                        expected,
                        found: args.len(),
                    })?
                }
                let arg_tys = args
                    .into_iter()
                    .map(|a| Ty::from_annotation(a, solver))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Ty::Adt(aid, arg_tys))
            }
            Annotation::Quantity(parts) => {
                for (unit, _) in &parts {
                    if shared::units::lookup(&unit.lexeme).is_none() {
                        Err(UnknownUnit {
                            src: solver.src(unit.location),
                            at: unit.location.into(),
                            name: unit.lexeme.clone(),
                        })?;
                    }
                }
                Ok(Ty::Float)
            }
            Annotation::Ty(ident) => {
                let ty = ident.query(solver)?;
                let dec = solver.ribs.resolve(&ident);
                solver.note(&ident, ty.clone(), dec);
                Ok(ty)
            }
            Annotation::Path(segments) => {
                let mut iter = segments.into_iter();
                let head = iter
                    .next()
                    .expect("path annotation has at least two segments");
                let mut ty = head.query(solver)?;
                let dec = solver.ribs.resolve(&head);
                solver.note(&head, ty.clone(), dec);
                for segment in iter {
                    let adt = match ty.clone().normalized(solver) {
                        Ty::Adt(adt, _) | Ty::Identity(adt, _) => adt,
                        other => Err(TypeHasNoFields {
                            src: solver.src(segment.location),
                            at: segment.location.into(),
                            ty: other.to_string(),
                        })?,
                    };
                    if !solver.adts[adt].flags.contains(AdtFlags::IS_MODULE) {
                        Err(NotAStruct {
                            src: solver.src(segment.location),
                            at: segment.location.into(),
                            ty: solver.adts[adt].name.clone(),
                        })?
                    }
                    let field = solver.adts[adt]
                        .as_struct()
                        .fields
                        .get(&segment.lexeme)
                        .cloned()
                        .ok_or_else(|| FieldNotFound {
                            src: solver.src(segment.location),
                            at: segment.location.into(),
                            field_name: segment.lexeme.clone(),
                        })?;
                    solver.note(&segment, field.ty.clone(), Some(field.dec));
                    ty = field.ty;
                }
                Ok(ty.normalized(solver))
            }
            Annotation::Bounds(idents) => {
                let mut pacts = Vec::with_capacity(idents.len());
                for ident in idents {
                    let ty = ident.query(solver)?;
                    let dec = solver.ribs.resolve(&ident);
                    solver.note(&ident, ty.clone(), dec);
                    let Some(pid) = ty.as_single_pact() else {
                        return Err(NotAPact {
                            src: solver.src(ident.location),
                            at: ident.location.into(),
                            ty: ty.to_string(),
                        }
                        .into());
                    };
                    pacts.push(pid);
                }
                Ok(Ty::pacts(pacts))
            }
        }
    }

    fn fulfill_ty(
        &mut self,
        other: &mut Ty,
        solver: &mut Solver,
    ) -> std::result::Result<(), UnificationError> {
        Unification::unify(self, other, solver).and_then(|v| v.commit(solver))
    }

    fn normalized(self, solver: &Solver) -> Ty {
        match self {
            Ty::Unit | Ty::Never | Ty::Null | Ty::Bool | Ty::Int | Ty::Float | Ty::Str => self,
            Ty::Vid(vid) => solver
                .sub(vid)
                .map(|ty| ty.clone().normalized(solver))
                .unwrap_or(Ty::Vid(vid)),
            Ty::Array(ty) => Ty::Array(Box::new(ty.normalized(solver))),
            Ty::Dict(ty) => Ty::Dict(Box::new(ty.normalized(solver))),
            Ty::Tuple(members) => {
                Ty::Tuple(members.into_iter().map(|v| v.normalized(solver)).collect())
            }
            Ty::Fn(f) => {
                let was_ctor = f.is_ctor;
                let mut h = FnHeader::new(
                    f.parameters
                        .into_iter()
                        .map(|p| FnParam::new(p.name, p.ty.normalized(solver), p.has_default))
                        .collect(),
                    f.return_ty.normalized(solver),
                    f.is_method,
                );
                h.is_ctor = was_ctor;
                Ty::Fn(h)
            }
            Ty::Adt(adt, args) => Ty::Adt(
                adt,
                args.into_iter().map(|a| a.normalized(solver)).collect(),
            ),
            Ty::Pacts(pacts) => Ty::Pacts(pacts),
            Ty::Skolem(pid) => Ty::Skolem(pid),
            Ty::Anon(n) => Ty::Anon(n),
            Ty::Param(pid) => Ty::Param(pid),
            Ty::Identity(adt, args) => Ty::Identity(
                adt,
                args.into_iter().map(|a| a.normalized(solver)).collect(),
            ),
            Ty::Option(inner) => {
                if let Ty::Option(nested_inner) = *inner {
                    Ty::Option(Box::new(nested_inner.normalized(solver)))
                } else {
                    Ty::Option(Box::new(inner.normalized(solver)))
                }
            }
            Ty::Result(inner) => Ty::Result(Box::new(inner.normalized(solver))),
        }
    }

    fn filter_adt(&self, adt: AdtId) -> Ty {
        match self {
            Ty::Array(ty) => Ty::Array(Box::new(ty.filter_adt(adt))),
            Ty::Dict(ty) => Ty::Dict(Box::new(ty.filter_adt(adt))),
            Ty::Tuple(members) => Ty::Tuple(members.iter().map(|m| m.filter_adt(adt)).collect()),
            Ty::Adt(this_adt, args) if *this_adt == adt => Ty::Identity(
                adt,
                args.iter().map(|a| a.filter_adt(adt)).collect(),
            ),
            Ty::Adt(id, args) => Ty::Adt(
                *id,
                args.iter().map(|a| a.filter_adt(adt)).collect(),
            ),
            Ty::Identity(id, args) => Ty::Identity(
                *id,
                args.iter().map(|a| a.filter_adt(adt)).collect(),
            ),
            Ty::Fn(f) => {
                let parameters: Vec<FnParam> = f
                    .parameters
                    .iter()
                    .map(|p| FnParam::new(p.name.clone(), p.ty.filter_adt(adt), p.has_default))
                    .collect();
                let mut h = FnHeader::new(parameters, f.return_ty.filter_adt(adt), f.is_method);
                h.is_ctor = f.is_ctor;
                Ty::Fn(h)
            }
            Ty::Option(inner) => Ty::Option(Box::new(inner.filter_adt(adt))),
            Ty::Result(inner) => Ty::Result(Box::new(inner.filter_adt(adt))),
            Ty::Pacts(_)
            | Ty::Skolem(_)
            | Ty::Anon(_)
            | Ty::Param(_)
            | Ty::Vid(_)
            | Ty::Unit
            | Ty::Never
            | Ty::Null
            | Ty::Bool
            | Ty::Int
            | Ty::Float
            | Ty::Str => self.clone(),
        }
    }
}

pub(crate) fn ty_from_kw(kw: TyKw) -> Ty {
    match kw {
        TyKw::Int => Ty::Int,
        TyKw::Float => Ty::Float,
        TyKw::Str => Ty::Str,
        TyKw::Bool => Ty::Bool,
    }
}
