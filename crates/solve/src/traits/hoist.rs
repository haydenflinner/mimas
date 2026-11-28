use crate::FxHashSet;
use crate::errors::{DuplicatePactFunction, PactOverloadDefault};

use super::Query;
use crate::{
    Result,
    components::{
        Adt, AdtId, DecKind, Field, Fields, FnHeader, StructVariant, TupleVariant, Ty, TyExt,
        Variant, Vis,
    },
    errors::{DuplicateImplDeclaration, DuplicatePactImpl, InvalidImplTarget},
    *,
};
use parse::{
    components::{Annotation, Binding},
    *,
};
use shared::{Located, Location};

/// Utility passed around by expressions/statements in the declaration pass of the type checker.
pub(crate) struct HoistCtx<'a> {
    solver: &'a mut Solver,
    target: HoistTarget,
    node_id: Option<NodeId>,
    location: Location,
    vis: Vis,
}

impl<'a> HoistCtx<'a> {
    /// Creates a new ctx. `vis` is the visibility of the item being hoisted -- used by `write`
    /// and `write_adt` for the resulting dec.
    pub(crate) fn new(
        solver: &'a mut Solver,
        target: HoistTarget,
        node_id: Option<NodeId>,
        location: Location,
        vis: Vis,
    ) -> Self {
        Self {
            solver,
            target,
            node_id,
            location,
            vis,
        }
    }

    pub(crate) fn write_adt(&mut self, ident: &Ident, adt_id: AdtId) -> Result<()> {
        debug_assert!(
            matches!(self.target, HoistTarget::Scope),
            "adt decls only hoist at scope level"
        );
        let ty = Ty::adt(adt_id);
        // r[impl mimas.prelude.shadow]
        if self.solver.in_prelude {
            self.solver.prelude_types.insert(ident.lexeme.clone());
            self.solver.prelude_adts.insert(ident.lexeme.clone(), adt_id);
        } else if self.solver.prelude_types.contains(&ident.lexeme) {
            self.solver.warnings.push(std::rc::Rc::new(
                crate::errors::ShadowedBuiltinType {
                    src: self.solver.src(ident.location),
                    at: ident.location.into(),
                    name: ident.lexeme.clone(),
                }
                .into(),
            ));
        }
        // a second type of one name in one file's region is an error (C2);
        // across regions each is its own. r[impl mimas.region.type-dup]
        let region = crate::components::regions::of(ident.location);
        if crate::components::regions::any()
            && !self.solver.in_prelude
            && let Some((prev_at, prev)) = self.solver.ribs.module().in_region(&ident.lexeme, &region)
            && prev_at.file_id() == ident.location.file_id()
            && prev.index() >= self.solver.declare_floor
            && matches!(self.solver.decs[prev].kind, DecKind::Adt(_))
        {
            return Err(crate::errors::MultipleTypeDeclarations {
                src: self.solver.src(ident.location),
                at: ident.location.into(),
                name: ident.lexeme.clone(),
                original: vec![crate::errors::ConstDefinedHere {
                    src: self.solver.src(prev_at),
                    at: prev_at.into(),
                    name: ident.lexeme.clone(),
                }],
            }
            .into());
        }
        let dec_id = self
            .solver
            .dec_id(ident, ty.clone(), DecKind::Adt(adt_id), self.vis);
        self.solver.ribs.module_mut().insert(ident.clone(), dec_id);
        self.solver.export_item(&ident.lexeme, &region, dec_id);
        if let Some(node_id) = self.node_id {
            self.solver.node_decs.insert(node_id, dec_id);
            let vid = self.solver.node_vid(node_id);
            let loc = self.location;
            self.solver
                .register_sub(vid, ty)
                .map_err(|e| e.into_type_mismatch(self.solver, loc))?;
        }
        Ok(())
    }

    /// Completes a hoist by writing it to the needed targets.
    pub(crate) fn write(&mut self, ident: &Ident, mut ty: Ty, constant: bool) -> Result<()> {
        if let HoistTarget::Adt(adt) = &self.target {
            ty = ty.filter_adt(*adt);
        }

        match &self.target {
            HoistTarget::Scope => {
                self.solver
                    .declare_item(ident, ty.clone(), self.node_id, constant, self.vis)?;
            }
            HoistTarget::Adt(adt) => {
                let kind = if constant {
                    DecKind::Constant(None)
                } else {
                    DecKind::Item {
                        defaults: vec![],
                        wheres: vec![],
                    }
                };

                let existing = self.solver.adts[*adt].impls.get(&ident.lexeme).cloned();
                // r[impl mimas.op.overload]
                // a second `fn mul` is an overload when both are operator-shaped
                // (`self` + one operand) and the operands' declared types differ
                // from each other's and from every overload already there;
                // anything else with the name is the duplicate it always was
                let overload = match &existing {
                    Some(first) => {
                        let siblings = self.solver.adts[*adt]
                            .overloads
                            .get(&ident.lexeme)
                            .cloned()
                            .unwrap_or_default();
                        overload_operand(&ty).is_some_and(|mine| {
                            std::iter::once(first)
                                .chain(&siblings)
                                .all(|f| overload_operand(&f.ty).is_some_and(|o| o != mine))
                        })
                    }
                    None => false,
                };
                // C3: impl members key by (pact, name). A pact member whose name
                // the type already holds, or an inherent member whose name a pact
                // member holds, keeps both: the pact's lives under `name@Pact`.
                // r[impl mimas.pact.member-key]
                if let Some(existing) = &existing
                    && !overload
                    && (self.solver.hoisting_pact.is_some()
                        || self.solver.member_pact.contains_key(&existing.dec))
                {
                    let adt = *adt;
                    let name = ident.lexeme.clone();
                    let mine = self.solver.hoisting_pact;
                    let theirs = self.solver.member_pact.get(&existing.dec).copied();
                    if mine.is_some() && mine == theirs {
                        Err(DuplicateImplDeclaration {
                            src: self.solver.src(ident.location),
                            at: ident.location.into(),
                            original_at: Some(self.solver.decs[existing.dec].location.into()),
                            name: name.clone(),
                        })?;
                    }
                    let dec = self.solver.dec_id(ident, ty.clone(), kind, self.vis);
                    let field = Field::new_constant(ty.clone(), self.location, dec);
                    match (mine, theirs) {
                        // a pact member beside an inherent or another pact's member
                        (Some(p), _) => {
                            let key = format!("{name}@{}", self.solver.pacts[p].name);
                            self.solver.decs[dec].name = key.clone();
                            self.solver.member_pact.insert(dec, p);
                            self.solver.adts[adt].impls.insert(key, field);
                            if let Some(t) = theirs {
                                let e = self.solver.ambiguous_members.entry((adt, name)).or_default();
                                for q in [t, p] {
                                    if !e.contains(&q) {
                                        e.push(q);
                                    }
                                }
                            }
                        }
                        // an inherent member takes the bare name; the pact's moves aside
                        (None, Some(t)) => {
                            let key = format!("{name}@{}", self.solver.pacts[t].name);
                            let moved = self.solver.adts[adt].impls.remove(&name).unwrap();
                            self.solver.decs[moved.dec].name = key.clone();
                            self.solver.adts[adt].impls.insert(key, moved);
                            self.solver.adts[adt].impls.insert(name.clone(), field);
                            self.solver.ambiguous_members.remove(&(adt, name));
                        }
                        (None, None) => unreachable!("guarded above"),
                    }
                    if let Some(node_id) = self.node_id {
                        self.solver.node_decs.insert(node_id, dec);
                    }
                    if let Some(node_id) = self.node_id {
                        let vid = self.solver.node_vid(node_id);
                        let loc = self.location;
                        self.solver
                            .register_sub(vid, ty)
                            .map_err(|e| e.into_type_mismatch(self.solver, loc))?;
                    }
                    return Ok(());
                }
                if let Some(existing) = &existing
                    && !overload
                {
                    let prev = &self.solver.decs[existing.dec];
                    // a member the page's own source declared points back at it;
                    // one a native adt carried in has no source to show
                    let original_at = (!prev.location.span().is_synthetic())
                        .then(|| prev.location.into());
                    Err(DuplicateImplDeclaration {
                        src: self.solver.src(ident.location),
                        at: ident.location.into(),
                        original_at,
                        name: ident.lexeme.clone(),
                    })?;
                }

                let dec = self.solver.dec_id(ident, ty.clone(), kind, self.vis);
                if let Some(p) = self.solver.hoisting_pact {
                    self.solver.member_pact.insert(dec, p);
                }
                let field = Field::new_constant(ty.clone(), self.location, dec);
                if overload {
                    self.solver.adts[adt]
                        .overloads
                        .entry(ident.lexeme.clone())
                        .or_default()
                        .push(field);
                } else {
                    self.solver.adts[adt]
                        .impls
                        .insert(ident.lexeme.clone(), field);
                }
                // def-site mapping: lets the item's own body lower via node_dec(expr.id())
                if let Some(node_id) = self.node_id {
                    self.solver.node_decs.insert(node_id, dec);
                }
            }
        }

        if let Some(node_id) = self.node_id {
            let vid = self.solver.node_vid(node_id);
            let loc = self.location;
            self.solver
                .register_sub(vid, ty)
                .map_err(|e| e.into_type_mismatch(self.solver, loc))?;
        }

        Ok(())
    }
}

/// The target for a given hoist (where it should be written).
#[derive(Debug, Clone)]
pub(crate) enum HoistTarget {
    /// This hoist will get written into the current scope.
    Scope,
    /// This hoist will get insderted into a struct.
    Adt(AdtId),
}

/// Used for expressions/statements that need to define named values in the first pass of the type
/// checker such as structs, traits, impls, and functions.
pub(crate) trait Hoist {
    /// Declares this type into the solver.
    fn hoist(&self, ctx: HoistCtx) -> Result<()>;
}

impl Hoist for parse::item::Enum {
    fn hoist(&self, mut ctx: HoistCtx) -> Result<()> {
        let variants = self
            .members
            .iter()
            .map(|(name, member)| {
                let variant = match member {
                    Member::Tuple(members) => Variant::Tuple(TupleVariant {
                        members: members.iter().map(|_| ctx.solver.vid().into()).collect(),
                        location: Some(name.location),
                        ..Default::default()
                    }),
                    Member::Struct(fields) => Variant::Struct(StructVariant {
                        fields: Fields(
                            fields
                                .iter()
                                .map(|StructField { name, location, .. }| {
                                    let ty: Ty = ctx.solver.vid().into();
                                    let ident = field_ident(name, *location);
                                    let dec = ctx.solver.dec_id(
                                        &ident,
                                        ty.clone(),
                                        DecKind::Local,
                                        Vis::Public,
                                    );
                                    (name.to_string(), Field::new(ty, *location, dec))
                                })
                                .collect(),
                        ),
                        location: Some(name.location),
                        ..Default::default()
                    }),
                };

                (name.to_string(), variant)
            })
            .collect();

        let mut adt = crate::components::Adt::new_enum(self.head.to_string(), variants);
        adt.type_params = ctx.solver.declare_type_params(&self.type_params);
        let id = ctx.solver.push_adt(adt);

        let variant_names: Vec<_> = ctx.solver.adts[id].variants.keys().cloned().collect();
        for name in variant_names {
            let qualified = format!("{}::{}", self.head, name);

            let layout_adt = {
                let variant = &ctx.solver.adts[id].variants[&name];
                let mut adt = Adt::new_variant_layout(qualified.clone(), variant);
                adt.enum_of = Some(id);
                adt
            };
            let layout = ctx.solver.push_adt(layout_adt);

            // the parser's ident under the qualified name (tooling keys off its id)
            let (written, _) = self
                .members
                .iter()
                .find(|(written, _)| written.lexeme == name)
                .expect("every variant comes from a member");
            let variant_ident = parse::Ident {
                lexeme: qualified,
                ..written.clone()
            };
            let dec = ctx.solver.dec_id(
                &variant_ident,
                Ty::adt(id),
                DecKind::Variant { parent: id, layout },
                Vis::Private,
            );

            let v = &mut ctx.solver.adts[id].variants[&name];
            v.set_layout(layout);
            v.set_dec(dec);
        }

        ctx.write_adt(&self.head, id)
    }
}

impl Hoist for Struct {
    fn hoist(&self, mut ctx: HoistCtx) -> Result<()> {
        let is_tuple = !self.fields.is_empty()
            && self
                .fields
                .iter()
                .all(|f| matches!(f.name, FieldKey::Int(_)));

        let adt = if is_tuple {
            let members = self
                .fields
                .iter()
                .map(|_| Ty::Vid(ctx.solver.vid()))
                .collect();

            Adt::new_tuple_struct(self.name.to_string(), members)
        } else {
            let mut adt = Adt::new_struct(self.name.to_string());

            let generic = !self.type_params.is_empty();
            self.fields.iter().for_each(
                |StructField {
                     name,
                     location,
                     public,
                     annotation,
                     ..
                 }| {
                    let ty = Ty::Vid(ctx.solver.vid());
                    let ident = field_ident(name, *location);
                    let dec = ctx.solver.dec_id(
                        &ident,
                        ty.clone(),
                        DecKind::Local,
                        if *public { Vis::Public } else { Vis::Private },
                    );
                    if !generic {
                        ctx.solver.field_annotations.insert(dec, annotation.clone());
                    }
                    adt.as_struct_mut()
                        .insert(name.to_string(), Field::new(ty, *location, dec));
                },
            );
            adt
        };

        let id = ctx.solver.push_adt(adt);
        ctx.solver.adts[id].type_params = ctx.solver.declare_type_params(&self.type_params);
        ctx.write_adt(&self.name, id)
    }
}

impl Hoist for Impl {
    fn hoist(&self, ctx: HoistCtx) -> Result<()> {
        // Find our target
        let Some(adt) = self.target.query(ctx.solver)?.as_adt().copied() else {
            Err(InvalidImplTarget {
                src: ctx.solver.src(self.target.location()),
                at: self.target.location().into(),
            })?
        };

        // `impl S<T: Bound>`: the bound holds `S`'s own `T` (r[impl mimas.generic.impl-bound])
        for tp in &self.bounds {
            let Some(pid) = ctx.solver.adts[adt]
                .type_params
                .iter()
                .find(|(n, _)| *n == tp.name.lexeme)
                .map(|(_, p)| *p)
            else {
                return Err(crate::errors::NotFound {
                    src: ctx.solver.src(tp.name.location),
                    at: tp.name.location.into(),
                    name: tp.name.lexeme.clone(),
                    help: Some(format!(
                        "`impl {}<{}: …>` bounds a type parameter `{}` declares",
                        self.target.lexeme, tp.name.lexeme, self.target.lexeme
                    )),
                }
                .into());
            };
            for bound in &tp.bounds {
                ctx.solver.bind_param(pid, bound);
            }
        }

        // Record pact membership up front (before any body solves) so coercions resolve regardless
        // of source order. Implementing the same pact twice is rejected here; member-name clashes
        // with anything else already on the adt surface as the members land in `impls` below.
        // `impl Add<T> for Stream<T>`: the restated params are the target's own
        if !self.target_params.is_empty() {
            let declared: Vec<String> = ctx.solver.adts[adt]
                .type_params
                .iter()
                .map(|(n, _)| n.clone())
                .collect();
            let written: Vec<String> = self.target_params.iter().map(|i| i.lexeme.clone()).collect();
            if declared != written {
                let at = self.target_params[0].location;
                return Err(crate::errors::PactArgs {
                    src: ctx.solver.src(at),
                    at: at.into(),
                    msg: format!(
                        "`{}` declares <{}>, so the impl restates them as written",
                        self.target.lexeme,
                        declared.join(", ")
                    ),
                    label: format!("write `{}<{}>`", self.target.lexeme, declared.join(", ")),
                }
                .into());
            }
        }
        // `impl Mul<float> for Vec2` implements that instance of `Mul`
        // (r[impl mimas.pact.params]); an error in the arguments reports at solve
        let pid = ctx.solver.impl_pact(self, adt).ok().flatten();
        if let Some(pid) = pid
            && !ctx.solver.pact_impls.insert((pid, adt))
        {
            Err(DuplicatePactImpl {
                src: ctx.solver.src(self.target.location()),
                at: self.target.location().into(),
                target: self.target.lexeme.clone(),
                pact: ctx.solver.pacts[pid].name.clone(),
            })?;
        }
        if let Some(pid) = pid {
            // the impl's required pacts (`pact X: Y`) check out once every impl
            // is registered -- `resolve_pending_impls`
            ctx.solver
                .pending_required
                .push((adt, pid, self.target.location()));
            // `impl X for T via Y`: member coverage waits for the same phase.
            // the default graft below doesn't run for it -- `via` covers by
            // name, so a member the block omits lands from `Y` or reports
            if let Some(via) = self.via.as_ref() {
                let provided: FxHashSet<String> = self
                    .items
                    .iter()
                    .filter_map(|i| match i.kind() {
                        ItemKind::Function(f) => Some(f.name.lexeme.clone()),
                        ItemKind::Const(c) => Some(
                            c.left
                                .as_ident()
                                .map(|i| i.lexeme.clone())
                                .unwrap_or_else(|| c.left.to_string()),
                        ),
                        _ => None,
                    })
                    .collect();
                ctx.solver.pending_vias.push(PendingVia {
                    adt,
                    pact: pid,
                    via: via.clone(),
                    provided,
                    target: self.target.clone(),
                    location: self.target.location(),
                });
            }
        }

        ctx.solver.ribs.push_impl(adt);
        // the impl target's own params are in scope for member signatures (`impl List` can
        // write `x: T` / `rest: List<T>` -- `impl List<int>`-style specializations are rejected
        // by the parser)
        ctx.solver.push_adt_params(adt);
        // `impl Iter for [T]`: the shape's params and `Self`
        // (r[impl mimas.impl.builtin-shape])
        if let Some(shape) = ctx.solver.enter_impl_shape(&self.shape)?
            && let Some(pid) = pid
        {
            ctx.solver.impl_shapes.insert((pid, adt), shape);
        }
        // `type Item = Card;`: the impl fixes its pact's associated types,
        // and names them in its own members (r[impl mimas.pact.assoc-type])
        let fixed = match crate::traits::assoc_fixes(ctx.solver, self, adt, pid) {
            Ok(fixed) => fixed,
            Err(e) => {
                ctx.solver.pop_type_params();
                ctx.solver.ribs.pop();
                return Err(e);
            }
        };
        ctx.solver.assoc_scope.push(fixed);

        let prev_pact = std::mem::replace(&mut ctx.solver.hoisting_pact, pid);
        let hoist_result: Result<()> = (|| {
            for item in self.items.iter() {
                let vis = if item.public() {
                    Vis::Public
                } else {
                    Vis::Private
                };
                let hoist_ctx = HoistCtx::new(
                    ctx.solver,
                    HoistTarget::Adt(adt),
                    Some(item.id()),
                    ctx.location,
                    vis,
                );
                match item.kind() {
                    ItemKind::Const(con) => con.hoist(hoist_ctx)?,
                    ItemKind::Function(function) => function.hoist(hoist_ctx)?,
                    // parser rejects everything else in impl-body position
                    _ => unreachable!("parser only allows fn/const in impl bodies"),
                };
            }

            // graft methods the impl omitted but the pact defaults, so they become real callable
            // methods on the adt (and collide like everything else if another pact already provides
            // the name). freshen the header so each impl's `self` slot is independent. a `via`
            // impl skips this -- its omitted members land from the via pact instead
            if let Some(pid) = pid
                && self.via.is_none()
            {
                let provided: FxHashSet<String> = self
                    .items
                    .iter()
                    .filter_map(|i| match i.kind() {
                        ItemKind::Function(f) => Some(f.name.lexeme.clone()),
                        _ => None,
                    })
                    .collect();
                let omitted: Vec<String> = ctx.solver.pacts[pid]
                    .functions
                    .iter()
                    .filter(|&(name, (_, default))| {
                        default.is_some() && !provided.contains(name.as_str())
                    })
                    .map(|(name, _)| name.clone())
                    .collect();
                for name in omitted {
                    let header = ctx.solver.pacts[pid].functions[&name].0.clone();
                    let ty = ctx
                        .solver
                        .graft_pact_fn(&header, &ctx.solver.impl_target_ty().unwrap());
                    let dec = ctx.solver.pact_members[&(pid, name.clone())];
                    // the name is taken (an inherent member or another pact's): the
                    // default lands under `name@Pact` (C3) r[impl mimas.pact.member-key]
                    let key = match ctx.solver.adts[adt].impls.get(&name) {
                        Some(existing) => {
                            if let Some(&t) = ctx.solver.member_pact.get(&existing.dec) {
                                let e = ctx.solver.ambiguous_members.entry((adt, name.clone())).or_default();
                                for q in [t, pid] {
                                    if !e.contains(&q) {
                                        e.push(q);
                                    }
                                }
                            }
                            format!("{name}@{}", ctx.solver.pacts[pid].name)
                        }
                        None => name,
                    };
                    ctx.solver.member_pact.insert(dec, pid);
                    ctx.solver.adts[adt]
                        .impls
                        .insert(key, Field::new_constant(ty, ctx.location, dec));
                }
            }
            Ok(())
        })();

        ctx.solver.hoisting_pact = prev_pact;
        ctx.solver.assoc_scope.pop();
        ctx.solver.impl_shape = None;
        ctx.solver.pop_type_params();
        ctx.solver.ribs.pop();

        hoist_result
    }
}

impl Hoist for Pact {
    fn hoist(&self, ctx: HoistCtx) -> Result<()> {
        // push the empty pact and set it as `pact_self` before resolving signatures, so `Self`
        // and the pact's own name resolve to it within its own method types (see `resolve_name`)
        let pact_id = ctx.solver.pacts.push(shared::Pact {
            name: self.name.lexeme.clone(),
            // `pact A: B` bounds -- a later pact's name may not resolve yet, so
            // the idents wait for `resolve_pending_impls` to fill this
            requires: vec![],
            // `shared::Pact` keeps indexmap's own hasher: it is shared's type
            functions: indexmap::IndexMap::default(),
            constants: indexmap::IndexMap::default(),
        });
        for req in self.requires.iter() {
            ctx.solver
                .pending_pact_requires
                .push((pact_id, req.clone()));
        }
        shared::name_pact(pact_id.index(), &self.name.lexeme);
        let prev_self = ctx.solver.pact_self.replace(pact_id);
        // `pact Add<Rhs = Self>`: the parameters read as stand-ins in the
        // member signatures (r[impl mimas.pact.params])
        let param_scope = match ctx.solver.declare_pact_params(pact_id, &self.params) {
            Ok(scope) => scope,
            Err(e) => {
                ctx.solver.pact_self = prev_self;
                return Err(e);
            }
        };
        let has_params = !param_scope.is_empty();
        if has_params {
            ctx.solver.assoc_scope.push(param_scope);
        }

        // `type Item;`: a stand-in vid each impl's binding replaces
        // (r[impl mimas.pact.assoc-type])
        let mut assoc = FxHashMap::default();
        for item in self.items.iter() {
            if let PactItem::Type { name, .. } = item {
                let p = Solver::param_id_of(name);
                ctx.solver.assoc_params.insert(p, (pact_id, name.lexeme.clone()));
                assoc.insert(name.lexeme.clone(), Ty::Param(p));
            }
        }
        let scoped = !assoc.is_empty();
        if scoped {
            ctx.solver.pact_assoc.insert(pact_id, assoc.keys().cloned().collect());
            ctx.solver.assoc_scope.push(assoc);
        }

        for item in self.items.iter() {
            match item {
                PactItem::Type { .. } => {}
                PactItem::Const {
                    name, annotation, ..
                } => {
                    let ty = Ty::from_annotation(annotation.clone(), ctx.solver)?;
                    // Local rather than Constant, which has to carry a value by the IR boundary
                    let dec = ctx.solver.dec_id(name, ty.clone(), DecKind::Local, ctx.vis);
                    ctx.solver
                        .pact_members
                        .insert((pact_id, name.lexeme.clone()), dec);
                    ctx.solver.pacts[pact_id]
                        .constants
                        .insert(name.lexeme.clone(), ty);
                }
                PactItem::Fn {
                    name,
                    generics,
                    parameters,
                    return_type,
                    default: body,
                    ..
                } => {
                    // the member's own `<U>`: params a call instantiates
                    // fresh (r[impl mimas.pact.member-generic])
                    for (_, p) in ctx.solver.declare_type_params(generics) {
                        ctx.solver.member_params.insert(p);
                    }
                    ctx.solver.push_type_params(generics);
                    let is_method = parameters
                        .first()
                        .and_then(|b| b.left.as_ident())
                        .is_some_and(|i| i.is_receiver());

                    let default = body.as_ref().map(|v| Ty::Vid(ctx.solver.node_vid(v.id())));

                    let header =
                        hoist_fn_header(ctx.solver, parameters, return_type, is_method, false);
                    ctx.solver.pop_type_params();
                    let header = header?;
                    // r[impl mimas.pact.overload]
                    // A second fn of one name is an overload when both are operator-shaped
                    // (`self` + one operand with a written type) and the operands differ —
                    // it lives under the key `name#k`, and `pact_groups` lists the group. An
                    // overload can't carry a default: the impl pairing is by operand, not name.
                    let base = name.lexeme.clone();
                    let key = if ctx.solver.pacts[pact_id].functions.contains_key(&base) {
                        let group = ctx
                            .solver
                            .pact_groups
                            .get(&(pact_id, base.clone()))
                            .cloned()
                            .unwrap_or_else(|| vec![base.clone()]);
                        let mine = overload_operand(&Ty::Fn(header.clone()));
                        let distinct = mine.as_ref().is_some_and(|mine| {
                            group.iter().all(|k| {
                                let (h, _) = &ctx.solver.pacts[pact_id].functions[k];
                                overload_operand(&Ty::Fn(h.clone())).is_some_and(|o| o != *mine)
                            })
                        });
                        if !distinct {
                            let first = ctx.solver.pact_members[&(pact_id, base.clone())];
                            Err(DuplicatePactFunction {
                                src: ctx.solver.src(name.location),
                                at: name.location.into(),
                                original_at: ctx.solver.decs[first].location.into(),
                                name: base.clone(),
                            })?;
                        }
                        let key = format!("{base}#{}", group.len());
                        let mut group = group;
                        group.push(key.clone());
                        ctx.solver
                            .pact_groups
                            .insert((pact_id, base.clone()), group);
                        key
                    } else {
                        base.clone()
                    };
                    if key != base && body.is_some() {
                        Err(PactOverloadDefault {
                            src: ctx.solver.src(name.location),
                            at: name.location.into(),
                            name: base.clone(),
                        })?;
                    }
                    // Mint a Dec for each method (a defaulted one is what the dispatch table
                    // points at when an impl omits it). Done at hoist so it exists before any body
                    // solves, regardless of pact/impl source order. An overload's dec is named
                    // by its key, which is how emit finds its slot in each impl's table.
                    let dec_ident = if key == base {
                        name.clone()
                    } else {
                        Ident::new(key.clone(), name.location)
                    };
                    let dec = ctx.solver.dec_id(
                        &dec_ident,
                        Ty::Fn(header.clone()),
                        DecKind::Item {
                            defaults: vec![],
                            wheres: vec![],
                        },
                        ctx.vis,
                    );
                    ctx.solver.pact_members.insert((pact_id, key.clone()), dec);
                    if let Some(body) = body {
                        // def-site mapping: this is how compile finds the dec to lower the default
                        // body into (emit.rs's Pact arm)
                        ctx.solver.node_decs.insert(body.id(), dec);
                    }
                    ctx.solver.pacts[pact_id]
                        .functions
                        .insert(key, (header, default));
                }
            }
        }

        ctx.solver.pact_self = prev_self;
        if scoped {
            ctx.solver.assoc_scope.pop();
        }
        if has_params {
            ctx.solver.assoc_scope.pop();
        }

        let dec = ctx.solver.dec_id(
            &self.name,
            Ty::pacts(vec![pact_id]),
            DecKind::Pact(pact_id),
            ctx.vis,
        );
        ctx.solver.ribs.module_mut().insert(self.name.clone(), dec);

        Ok(())
    }
}

impl Hoist for Function {
    fn hoist(&self, mut ctx: HoistCtx) -> Result<()> {
        // a `method` arm declares no name -- it attaches to the `multi fn` its
        // name resolves to and gets a body dec of its own. r[impl mimas.multi]
        if self.role == FnRole::Method {
            return hoist_method(self, ctx);
        }

        // declared type params (`fn map<T, U>`) are in scope while the signature resolves --
        // they land in the stored `Ty::Fn` as `Ty::Param`s and get re-instantiated per use
        ctx.solver.push_type_params(&self.type_params);
        let fn_header = hoist_fn_header(
            ctx.solver,
            &self.parameters,
            &self.return_type,
            self.is_method(),
            true,
        );
        ctx.solver.pop_type_params();
        let mut fn_header = fn_header?;

        // a `multi fn` dispatches at runtime: an unannotated param accepts the
        // whole call space (misses land in the mandatory fallback), so it must
        // be `any` -- a Vid would pin to the first call site's type and a
        // bound annotation only describes the dispatchable space, not the
        // caller's obligation. the fallback narrows with `match` like any
        // `any`-typed value. r[impl mimas.multi.decl]
        if self.role == FnRole::Multi {
            for (param, binding) in fn_header.parameters.iter_mut().zip(&self.parameters) {
                if binding.annotation.is_none() {
                    param.ty = Ty::Any;
                }
            }
        }

        let ty = Ty::Fn(fn_header);
        let node_id = ctx.node_id;
        ctx.write(&self.name, ty, false)?;

        // populate per-parameter defaults onto the just-created Item kind. defaults must be
        // const-reducible; one that can't reduce yet (it names a `const` the fixpoint
        // hasn't solved) waits in `pending_defaults` and is rejected by
        // `resolve_pending_defaults` if it never does, so the IR never has to deal with a
        // `None` literal where it expected a value. names live on the callee's `Ty::Fn`,
        // not here.
        if let Some(node_id) = node_id
            && let Some(&dec) = ctx.solver.node_decs.get(&node_id)
        {
            let mut defaults: Vec<Option<parse::Literal>> =
                Vec::with_capacity(self.parameters.len());
            for (index, b) in self.parameters.iter().enumerate() {
                defaults.push(match b.right.as_ref() {
                    None => None,
                    // r[impl mimas.fn.default-const]
                    Some(expr) => match ctx.solver.default_literal(expr)? {
                        Some(lit) => Some(lit),
                        None => {
                            ctx.solver.pending_defaults.push((dec, index, expr.clone()));
                            None
                        }
                    },
                });
            }
            if let DecKind::Item {
                defaults: slot,
                wheres: where_slot,
            } = &mut ctx.solver.decs[dec].kind
            {
                *where_slot = self.wheres.clone();
                *slot = defaults;
            }

            // a `multi fn` registers its bound space (as raw annotations --
            // `check_multis` resolves them once every impl is known) so the
            // `method` arms that follow can find and key against it.
            // r[impl mimas.multi]
            if self.role == FnRole::Multi {
                ctx.solver.multis.insert(
                    dec,
                    crate::solver::MultiData {
                        bound_annotations: self
                            .parameters
                            .iter()
                            .map(|b| b.annotation.clone())
                            .collect(),
                        bounds: vec![],
                        methods: vec![],
                    },
                );
            }
        }

        Ok(())
    }
}

/// Hoist a `method` arm: resolve the name to an existing `multi fn`'s
/// dec, give the arm a body dec of its own (seeded exactly like
/// `ctx.write` does for a plain `fn`, minus the rib binding -- the name
/// stays the generic's), and record its key annotations on the generic.
/// r[impl mimas.multi]
fn hoist_method(this: &Function, ctx: HoistCtx) -> Result<()> {
    let Some(gdec) = ctx.solver.ribs.resolve(&this.name) else {
        Err(crate::errors::MultiNoGeneric {
            src: ctx.solver.src(this.name.location),
            at: this.name.location.into(),
            name: this.name.lexeme.clone(),
        })?
    };
    if !ctx.solver.multis.contains_key(&gdec) {
        Err(crate::errors::MultiNotGeneric {
            src: ctx.solver.src(this.name.location),
            at: this.name.location.into(),
            name: this.name.lexeme.clone(),
        })?
    }
    let expected = ctx.solver.multis[&gdec].bound_annotations.len();
    if expected != this.parameters.len() {
        Err(crate::errors::MultiArityMismatch {
            src: ctx.solver.src(this.name.location),
            at: this.name.location.into(),
            expected,
            found: this.parameters.len(),
        })?
    }

    // the arm's params type from their key annotations (`V::Num` reads as
    // the variant's layout type); no `self` -- the parser rejected it.
    ctx.solver.push_type_params(&this.type_params);
    let fn_header = hoist_fn_header(ctx.solver, &this.parameters, &this.return_type, false, true);
    ctx.solver.pop_type_params();
    let ty = Ty::Fn(fn_header?);

    let dec = ctx.solver.dec_id(
        &this.name,
        ty.clone(),
        DecKind::Item {
            defaults: vec![],
            wheres: this.wheres.clone(),
        },
        ctx.vis,
    );
    if let Some(node_id) = ctx.node_id {
        ctx.solver.node_decs.insert(node_id, dec);
        let vid = ctx.solver.node_vid(node_id);
        let loc = ctx.location;
        ctx.solver
            .register_sub(vid, ty)
            .map_err(|e| e.into_type_mismatch(ctx.solver, loc))?;
    }
    ctx.solver
        .multis
        .get_mut(&gdec)
        .unwrap()
        .methods
        .push(crate::solver::PendingMethod {
            dec,
            key_annotations: this
                .parameters
                .iter()
                .map(|b| b.annotation.clone())
                .collect(),
            keys: vec![],
            location: ctx.location,
        });
    Ok(())
}

impl Hoist for Tests {
    fn hoist(&self, ctx: HoistCtx) -> Result<()> {
        let HoistCtx { solver, vis, .. } = ctx;
        for case in &self.cases {
            let mut case_ctx = HoistCtx::new(
                solver,
                HoistTarget::Scope,
                Some(case.id),
                case.expr.location(),
                vis,
            );
            let ty = Ty::Fn(FnHeader::new(vec![], Ty::Bool, false));
            case_ctx.write(&case.name, ty, false)?;
        }
        Ok(())
    }
}

impl Hoist for Const {
    fn hoist(&self, mut ctx: HoistCtx) -> Result<()> {
        let vid = ctx.solver.node_vid(self.right.id());
        if let Some(ident) = self.left.as_ident() {
            return ctx.write(ident, Ty::Vid(vid), true);
        }

        // destructuring const -- one Constant dec per bound name; the leaf types are fresh
        // vids that `solve_const`'s `bind_const_pat` pins to the rhs's element types.
        for leaf in self.left.bound_leaves() {
            let ident = leaf.as_ident().expect("bound_leaves only yields idents");
            let mut leaf_ctx = HoistCtx::new(
                ctx.solver,
                ctx.target.clone(),
                Some(leaf.id()),
                ctx.location,
                ctx.vis,
            );
            let leaf_vid = leaf_ctx.solver.vid();
            leaf_ctx.write(ident, Ty::Vid(leaf_vid), true)?;
        }
        // the item node itself still types as the whole tuple
        if let Some(node_id) = ctx.node_id {
            let item_vid = ctx.solver.node_vid(node_id);
            let location = ctx.location;
            ctx.solver
                .register_sub(item_vid, Ty::Vid(vid))
                .map_err(|e| e.into_type_mismatch(ctx.solver, location))?;
        }
        Ok(())
    }
}

fn hoist_fn_header(
    solver: &mut Solver,
    parameters: &[Binding],
    return_type: &Option<Annotation>,
    is_method: bool,
    infer_return_ty: bool,
) -> Result<FnHeader> {
    // `fn f(x) { x }` -- a `fn` with a body but no `->` leaves the return an
    // open Vid the body's check settles: one monotype per hole, shared across
    // call sites (`mimas.fn.inferred`). A signature with no body to read (a
    // pact item) keeps `-> ()`.
    // r[impl mimas.fn.inferred]
    let expected_ty = match return_type {
        Some(a) => Ty::from_annotation(a.clone(), solver)?,
        None if infer_return_ty => Ty::Vid(solver.vid()),
        None => Ty::Unit,
    };

    // Resolve each parameter's type into an `FnParam` for the `FnHeader`. `bind_parameter`
    // allocates one `Vid` per param and unifies it directly (no `Dec`, no rib insertion) --
    // the real per-body binding happens in `Function::solve`.
    let solved_parameters = parameters
        .iter()
        .map(|b| solver.bind_parameter(b))
        .collect::<Result<Vec<_>>>()?;

    // once a param has a default, every later one must too -- otherwise positional call sites
    // become ambiguous about where the default starts. `parameters` and `self.parameters`
    // are parallel, so an index serves both.
    if let Some(first) = solved_parameters.iter().position(|p| p.has_default) {
        for (i, param) in solved_parameters.iter().enumerate().skip(first + 1) {
            if !param.has_default {
                let loc = parameters[i].location();
                Err(crate::errors::RequiredAfterOptional {
                    src: solver.src(loc),
                    at: loc.into(),
                })?;
            }
        }
    }

    Ok(FnHeader::new(solved_parameters, expected_ty, is_method))
}

/// A field's name as an ident. A named field hands back the one the parser made (tooling keys
/// off its id), while a tuple field has only a number to go on.
/// `(name, ParamId)` for each entry of a parsed `<T, U>` list -- stored on the adt as
/// `type_params` so `impl` blocks and `Self` can re-enter the scope with the same ids.
fn field_ident(name: &FieldKey, location: Location) -> Ident {
    match name {
        FieldKey::Ident(ident) => ident.clone(),
        key => Ident::new(key.to_string(), location),
    }
}

/// The declared type of a method's one operand, when the method is
/// operator-shaped: `self` plus exactly one parameter with a written type.
/// Two such methods may share a name (`mimas.op.overload`); an unannotated
/// operand can't tell them apart, so it never overloads.
fn overload_operand(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Fn(h) if h.is_method && h.parameters.len() == 2 => match &h.parameters[1].ty {
            Ty::Vid(_) => None,
            operand => Some(operand.clone()),
        },
        _ => None,
    }
}

/// The associated types an impl fixes, `type Item = Card;`, checked against
/// its pact's names and recorded for `fresh_vids` (r[impl mimas.pact.assoc-type]).
pub(crate) fn assoc_fixes(
    solver: &mut Solver,
    imp: &Impl,
    adt: AdtId,
    pid: Option<shared::PactId>,
) -> Result<crate::FxHashMap<String, Ty>> {
    let mut fixed = crate::FxHashMap::default();
    let Some(pid) = pid else { return Ok(fixed) };
    let wanted = solver.pact_assoc.get(&pid).cloned().unwrap_or_default();
    for (name, ann) in &imp.assoc {
        if !wanted.contains(&name.lexeme) {
            return Err(crate::errors::NotFound {
                src: solver.src(name.location),
                at: name.location.into(),
                name: name.lexeme.clone(),
                help: Some(format!("the pact declares no `type {}`", name.lexeme)),
            }
            .into());
        }
        let ty = Ty::from_annotation(ann.clone(), solver)?;
        solver.assoc_bindings.insert((pid, adt, name.lexeme.clone()), ty.clone());
        fixed.insert(name.lexeme.clone(), ty);
    }
    if let Some(missing) = wanted.iter().find(|w| !fixed.contains_key(*w)) {
        return Err(crate::errors::NotFound {
            src: solver.src(imp.target.location()),
            at: imp.target.location().into(),
            name: missing.clone(),
            help: Some(format!("this impl must fix the pact's `type {missing} = …;`")),
        }
        .into());
    }
    Ok(fixed)
}

