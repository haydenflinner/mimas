#![allow(unused)] // temp

use crate::FxHashMap;
use std::sync::Arc;

use crate::FxIndexMap;
use api::NativeId;
use miette::NamedSource;
use parse::{Literal, NodeId};
use shared::{FileId, IdVec, Location, PactId, ParamId, TyNames};

use crate::{
    Solver,
    components::{Adt, AdtFlags, AdtId, DecId, DecKind, Ty, TyExt, Variant, Vis},
    grades::UseInfo,
};

pub struct Resolutions {
    pub node_tys: FxIndexMap<NodeId, Ty>,
    /// The unit dimension the dims pass computed for each expr —
    /// `b.x + 2.0px` is `Float` in `node_tys` but `px` here.
    pub node_dims: FxIndexMap<NodeId, shared::units::Dim>,
    /// The unit dimension the expr's context demanded — `b.x + BALL` passed to a
    /// `px` param is `Any` in `node_dims` but `px` here.
    pub want_dims: FxIndexMap<NodeId, shared::units::Dim>,
    /// `print`/`dbg`/`warn` call site → the dimension of the reported value — codegen ships
    /// it on `Program` so the natives can print `25000 W` rather than `25000`.
    pub print_dims: FxIndexMap<Location, shared::units::Dim>,
    /// Top-level echo site → the unit its quantity shows in (the unit the program
    /// wrote for that dimension) — codegen ships it so `12lb` echoes `12 lb`.
    pub echo_units: FxIndexMap<Location, shared::units::Shown>,
    /// Display site (`print`/`dbg`/`warn` call, top-level echo) → the shown value's
    /// shape: tuple parentheses and units the runtime value cannot tell.
    pub print_shapes: FxIndexMap<Location, shared::units::Shape>,
    /// f-string hole → its value's shape; emit renders a non-plain one through
    /// `__shown`.
    pub hole_shapes: rustc_hash::FxHashMap<NodeId, shared::units::Shape>,
    /// The unit the page wrote for each dimension a display names.
    pub shown_units: FxIndexMap<shared::units::Dim, shared::units::Shown>,
    /// `__shown`: a value rendered by a shape, for an f-string hole.
    pub shown_native: Option<NativeId>,
    /// f-string holes inside `impl Show for T`'s `show`, with `T`
    /// (r[impl mimas.show.self]).
    pub built_in_holes: rustc_hash::FxHashMap<NodeId, AdtId>,
    /// `__shown_as`: an f-string hole rendered with one type's `show` held off.
    pub shown_as_native: Option<NativeId>,
    /// Values a failing check's message names, with the unit to name them in.
    pub check_units: rustc_hash::FxHashMap<NodeId, shared::units::Shown>,
    /// Numbers with units written in a check, as written (`47lb`).
    pub check_texts: rustc_hash::FxHashMap<NodeId, String>,
    /// A `dec` held in the page's unit for its dimension: a literal's
    /// factor from its written unit, or a product's correction
    /// (r[impl mimas.dec.units]).
    pub dec_scales: rustc_hash::FxHashMap<NodeId, f64>,
    /// Exprs emit negates — the dims pass marked them as crossing to the
    /// opposite axis convention where they land.
    pub sign_negs: rustc_hash::FxHashSet<NodeId>,
    pub node_decs: FxIndexMap<NodeId, DecId>,
    pub rewrites: FxIndexMap<NodeId, parse::Expr>,
    /// Read/write site counts per declaration -- the `{0,1,ω}` usage grades the
    /// lint suite is built on (`Use::Never` is what "unused" warnings report on).
    pub dec_uses: FxIndexMap<DecId, UseInfo>,
    /// Inferred effect set per `fn` dec: the union of what its body can do, seeded by
    /// `#[effects]` declarations on natives. `Fx::unknown()`-flagged sets carry
    /// `Fx::UNAUDITED` -- something in the call chain couldn't be graded.
    pub fn_effects: FxIndexMap<DecId, shared::Fx>,
    /// Effect set of each file's top-level statements, keyed by the ast's name. Hosts
    /// running peer pages gate on this: `fx.fits(granted)` or the code doesn't eval.
    pub script_effects: FxIndexMap<String, shared::Fx>,
    /// Per top-level statement: `(file name, site, inferred fx)` in source order.
    /// Spliced `use`-includes share one file name -- the site's byte span is what
    /// attributes each statement back to the include that wrote it.
    pub script_segments: Vec<(String, shared::Location, shared::Fx)>,
    /// Each page's declared `needs {…}` authority: `(page, decl site, set)`,
    /// `page` named by its `//# page:` mark or by the file for own text. A
    /// page absent here declared no `needs` -- it holds ambient authority
    /// (`Fx::ANY`) by default.
    pub page_needs: Vec<(String, shared::Location, shared::Fx)>,
    /// Non-fatal diagnostics from the grades pass's lint suite. The solve succeeded;
    /// these are severity-warning reports for the host to render.
    pub warnings: Vec<miette::Report>,
    pub decs: IdVec<DecId, ResolvedDecl>,
    pub adts: IdVec<AdtId, ResolvedAdt>,
    pub pact_names: IdVec<PactId, String>,
    pub module_paths: FxHashMap<AdtId, Vec<String>>,
    pub closure_captures: FxIndexMap<NodeId, Vec<DecId>>,
    /// The derived write sets: fn dec -> `(param position, path segments)`
    /// pairs the body may write through, calls included. `[]` names the
    /// param's whole storage; a path holds `Field` names and `Index`
    /// wildcards rendered as segments. r[impl mimas.fn.writes]
    pub fn_writes: FxHashMap<crate::components::DecId, Vec<(u32, Vec<String>)>>,
    /// Calls that lend a `&` argument to an `inout` parameter: emit scopes the
    /// borrow's pending cleanup to the call so the loan ends when it returns.
    pub inout_calls: rustc_hash::FxHashSet<NodeId>,
    /// Whole-binding `&` args (`f(&c)`) loan through a one-slot cell: call
    /// node → `(binding, cell)` pairs whose `binding = cell[0]` write-backs
    /// emit after the call returns.
    pub inout_writebacks: FxIndexMap<NodeId, Vec<(DecId, DecId)>>,
    /// `fn f(inout self)` call sites: call node → the `__borrow_place` marker
    /// emit lowers as arg-0 instead of the receiver's value.
    pub self_place_mints: FxIndexMap<NodeId, parse::Expr>,
    /// Exprs whose array lengths the solver couldn't prove against a `[T; n]`
    /// contract -- emit wraps each in a runtime dim check (see `lens.rs`).
    pub len_checks: FxIndexMap<NodeId, Vec<Option<usize>>>,
    /// `S::default()` calls the solver blessed: call node → the struct's adt.
    /// Emit lowers each to a `NewInstance` whose members are the types'
    /// defaults instead of a call.
    pub default_ctors: FxIndexMap<NodeId, AdtId>,
    /// The `__contract_fail` native's id, for contract-failure lowering at emit.
    /// `None` in embeddings that never installed the stdlib.
    pub contract_native: Option<NativeId>,
    /// The `echo` native's id: emit hands every top-level expression
    /// statement's value to it, so a host can show the value at its line.
    /// `None` without the stdlib — then nothing is emitted.
    pub echo_native: Option<NativeId>,
    pub cleanup_natives: Option<(NativeId, NativeId)>,
    /// `__in_unit`: a quantity in a named unit, for failing checks.
    pub in_unit_native: Option<NativeId>,
    /// `__freeze`: makes the value of a `const` read-only (`mimas.const.frozen`).
    /// `None` when std is not installed: a const is then built fresh at each use.
    pub freeze_native: Option<NativeId>,
    /// The `panic` native's id -- emit's contract-failure fallback when the
    /// prettier `__contract_fail` channel isn't installed.
    pub panic_native: Option<NativeId>,
    /// `a <op> b` / `<op> a` / `a <op>= b` nodes the checker resolved to an
    /// operator method on an operand's adt (`v.add(w)` for `v + w`, `v.rsub(s)`
    /// for `s - v`). Emit lowers them as ordinary method calls rather than
    /// `Bin`/`Unary` ops.
    pub op_decs: FxIndexMap<NodeId, OpDispatch>,
    /// `expr?` nodes → what a null or a raise does to the enclosing function
    /// (`mimas.propagate`); the operand's own type says which test to run.
    pub propagates: FxIndexMap<NodeId, Propagation>,
    /// Int literals that fill a `float` slot (`mimas.num.literal`): emit and
    /// fold them as floats.
    pub int_as_float: rustc_hash::FxHashSet<NodeId>,
    /// Numeric literals that are `dec` (`mimas.dec.literal`).
    pub dec_literals: rustc_hash::FxHashSet<NodeId>,
    /// Each `dec` literal as written, so `3.50` keeps its scale.
    pub dec_texts: rustc_hash::FxHashMap<NodeId, String>,
    /// `Name(args…)` on a named-field struct (`mimas.struct.positional`): the
    /// field each argument fills, so the emitter builds the instance in field
    /// order after evaluating the arguments in source order.
    pub ctor_slots: FxHashMap<NodeId, Vec<usize>>,
    /// Each atom literal's layout (`mimas.atom.literal`): the `atom` variant,
    /// or the declared enum's variant it fits. Emit builds an instance of it.
    pub atom_layouts: FxIndexMap<NodeId, AdtId>,
    /// `const` items computed before the program runs (`mimas.const.computed`),
    /// in source order: the emitter lowers each value at the top of the entry.
    pub computed_consts: rustc_hash::FxHashSet<NodeId>,
    /// Values filling a struct field without naming it, with that field
    /// (`app.field-names`): a host editor shows `V(x = 0, y = -1)`.
    pub field_hints: Vec<(shared::Location, String)>,
    /// `<op> a` nodes whose operand is pact-typed (`!x` with `x: Dichotomy`):
    /// no single impl, so emit dispatches at runtime over the bound's
    /// implementers, exactly like a method call through the pact.
    pub pact_ops: FxIndexMap<NodeId, PactOp>,
    /// `multi fn` generics: dec → dispatch table. Emit lowers a call in
    /// `multi_calls` as first-match dispatch over these arms, ending at the
    /// generic's own body (the mandatory fallback). r[impl mimas.multi]
    pub multis: FxHashMap<DecId, ResolvedMulti>,
    /// Call sites whose callee resolved to a `multi fn`'s dec.
    pub multi_calls: FxIndexMap<NodeId, DecId>,
    /// The pacts each declared type param is bounded by (`<T: Draw>`): a method call on a
    /// `T`-typed receiver dispatches through them at runtime, like a pact-typed one.
    pub param_bounds: FxIndexMap<ParamId, Vec<PactId>>,
    pub root: ResolvedModule,
    pub sources: FxHashMap<FileId, NamedSource<Arc<str>>>,
}

/// A solved `a <op> b` → method call: `dec` is the `fn` inside the operand's
/// `impl`; `swap` marks an `r`-prefixed method whose receiver is the *right*
/// operand (`s - v` ⇒ `v.rsub(s)` — `self` always owns the impl).
/// What `expr?` does when `expr` is null or raised: the enclosing function
/// returns `null` (it returns `U?`), raises the same payload again (it returns
/// `U!` and `expr` raised), or raises `msg` (it returns `U!` and `expr` was a
/// null option).
#[derive(Debug, Clone)]
pub enum Propagation {
    ReturnNull,
    Reraise,
    RaiseNull(String),
}

#[derive(Debug, Clone, Copy)]
pub struct OpDispatch {
    pub dec: DecId,
    pub swap: bool,
}

/// An operator on a pact-typed operand → the pact method `method` (`inv` for
/// `!`), dispatched at runtime over the implementers of `pids`.
#[derive(Debug, Clone)]
pub struct PactOp {
    pub pids: Vec<PactId>,
    /// The member key (`mul`, or an overload's `mul#1`).
    pub method: String,
    /// The receiver is the *right* operand (`3.0 * x` ⇒ `x.rmul(3.0)`).
    pub swap: bool,
}

impl Resolutions {
    /// An adt's name qualified by the module it's declared in, e.g. `one::two::Foo`.
    pub fn adt_path(&self, aid: AdtId) -> String {
        let adt = &self.adts[aid];
        let mut path = self.module_path(adt.module).to_vec();
        path.push(Ty::adt(aid).display(self));
        path.join("::")
    }

    /// What a dec sits in, as written before its name: the adt for a method or field, the
    /// module path for anything else (empty at the root).
    pub fn owner_path(&self, dec: DecId) -> String {
        let dec = &self.decs[dec];
        match dec.owner {
            Some(owner) => self.adt_path(owner),
            None => self.module_path(dec.module).join("::"),
        }
    }

    /// The segments leading to `module`. Empty for the root (or anything that isn't a module).
    pub fn module_path(&self, module: AdtId) -> &[String] {
        self.module_paths.get(&module).map_or(&[], Vec::as_slice)
    }
}

impl TyNames for Resolutions {
    fn adt(&self, id: AdtId) -> Option<String> {
        self.adts.get(id).map(|adt| adt.name.clone())
    }

    fn pact(&self, id: PactId) -> Option<String> {
        self.pact_names.get(id).cloned()
    }

    fn param(&self, id: ParamId) -> Option<String> {
        shared::ThreadNames.param(id)
    }
}

/// Everything a file or module declares, in declaration order.
#[derive(Debug, Clone, Default)]
pub struct ResolvedModule {
    pub items: FxIndexMap<String, DecId>,
    pub modules: FxIndexMap<String, ResolvedModule>,
}

impl ResolvedModule {
    fn new(solver: &Solver, adt: AdtId) -> Self {
        let root = adt == AdtId::DANGLING;
        let items = solver
            .module_items
            .get(&adt)
            .into_iter()
            .flatten()
            .map(|(name, &dec)| (name.clone(), dec))
            .collect();
        let children: Vec<(String, AdtId)> = if root {
            solver
                .root_modules
                .iter()
                .map(|(name, &adt)| (name.clone(), adt))
                .collect()
        } else {
            let fields = solver.adts[adt].as_struct().fields.iter();
            fields
                .filter_map(|(name, field)| {
                    let child = *field.ty.as_adt()?;
                    let is_module = solver.adts[child].flags.contains(AdtFlags::IS_MODULE);
                    is_module.then(|| (name.clone(), child))
                })
                .collect()
        };
        Self {
            items,
            modules: children
                .into_iter()
                .map(|(name, child)| (name, Self::new(solver, child)))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedDecl {
    pub name: String,
    pub ty: Ty,
    pub kind: ResolvedDeclKind,
    pub vis: Vis,
    pub location: Location,
    pub module: AdtId,
    pub owner: Option<AdtId>,
    pub implements: Option<DecId>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedDeclKind {
    Local,
    /// A top-level `let`: mutable storage living in the entry frame, visible to fns.
    Global,
    Item {
        defaults: Vec<Option<Literal>>,
        native: Option<NativeId>,
        takes_self: bool,
        /// Def site of the registered Rust fn for natives (`None` for user-defined items).
        src: Option<api::NativeSrc>,
        /// The built-in's `///` docstring, harvested by `#[native]`/`#[mimas]`
        /// (`Some` only when non-empty).
        doc: Option<String>,
    },
    Constant(Literal),
    Variant {
        parent: AdtId,
        layout: AdtId,
    },
    Adt(AdtId),
    Pact(PactId),
}

pub struct ResolvedAdt {
    pub name: String,
    pub module: AdtId,
    pub fields: Vec<String>,
    /// Declared member types in construction order (struct fields, tuple
    /// members); `Ty::Param`s stand for the adt's own params, which a
    /// `S::default()` lowering substitutes with the call site's args.
    /// Empty for enums and modules.
    pub member_tys: Vec<Ty>,
    /// Each member's declared unit, parallel to `fields` (through `?`/`[..]`):
    /// `next: index?` is `Some(INDEX)`. `None` for unitless members and tuples.
    pub member_dims: Vec<Option<shared::units::Dim>>,
    /// Each member's display shape from its annotation, parallel to
    /// `fields`, one unit per tuple slot. `None` when the member shows plain.
    pub member_shapes: Vec<Option<shared::units::Shape>>,
    /// The adt's declared type params (`struct P<T>` → `[T]`), positionally
    /// parallel to the `args` on a `Ty::Adt`/`Ty::Identity` instantiation.
    pub type_params: Vec<ParamId>,
    pub implements: Vec<PactId>,
    pub methods: FxIndexMap<String, DecId>,
    pub dispatch_ids: Vec<AdtId>,
}

impl ResolvedAdt {
    pub(crate) fn new(aid: AdtId, adt: &Adt, solver: &Solver) -> Self {
        let fields = match (adt.variants.len(), adt.variants.values().next()) {
            (1, Some(Variant::Struct(variant))) => variant.fields.keys().cloned().collect(),
            (1, Some(Variant::Tuple(variant))) => {
                (0..variant.members.len()).map(|i| i.to_string()).collect()
            }
            _ => Vec::new(),
        };

        let member_tys = match (adt.variants.len(), adt.variants.values().next()) {
            (1, Some(Variant::Struct(variant))) => variant
                .fields
                .values()
                .map(|f| f.ty.clone().normalized(solver))
                .collect(),
            (1, Some(Variant::Tuple(variant))) => variant
                .members
                .iter()
                .map(|m| m.clone().normalized(solver))
                .collect(),
            _ => Vec::new(),
        };
        let member_shapes = fields
            .iter()
            .map(|f| {
                solver
                    .member_shapes
                    .get(&(adt.name.clone(), f.clone()))
                    .cloned()
            })
            .collect();
        let member_dims = fields
            .iter()
            .map(|f| {
                solver
                    .member_units
                    .get(&(adt.name.clone(), f.clone()))
                    .copied()
            })
            .collect();
        let type_params = adt.type_params.iter().map(|(_, pid)| *pid).collect();

        let mut implements: Vec<PactId> = solver
            .pact_impls
            .iter()
            .filter_map(|(p, a)| (*a == aid).then_some(*p))
            .collect();
        // an instance's impl also answers its template, so a default body
        // (solved once, against the template) dispatches on the type
        // (r[impl mimas.pact.params])
        // an instance a sibling impl answers (`impl Add<T> for Box<T>` for a
        // bound's `Add<int>`) dispatches to that impl
        // (r[impl mimas.pact.params.generic-target])
        let aliased: Vec<(PactId, PactId)> = solver
            .inst_alias
            .iter()
            .filter(|((_, a), _)| *a == aid)
            .map(|((j, _), i)| (*j, *i))
            .collect();
        for (j, _) in &aliased {
            if !implements.contains(j) {
                implements.push(*j);
            }
        }
        let bases: Vec<PactId> = implements
            .iter()
            .filter_map(|p| solver.inst_of.get(p).map(|(b, _)| *b))
            .collect();
        for b in bases {
            if !implements.contains(&b) {
                implements.push(b);
            }
        }
        implements.sort_by_key(|p| p.index());

        // a fn that implements a pact member sits under the member's key (an
        // overload's `mul#1`), which is the name runtime dispatch looks up;
        // everything else under its own name. r[impl mimas.pact.overload]
        let slots = |name: &str, dec: DecId| -> Vec<(String, DecId)> {
            match solver.impl_member.get(&dec) {
                Some(members) if !members.is_empty() => members
                    .iter()
                    .map(|m| (solver.decs[*m].name.clone(), dec))
                    .collect(),
                _ => vec![(name.to_string(), dec)],
            }
        };
        let mut methods: FxIndexMap<String, DecId> = adt
            .impls
            .iter()
            .filter(|(_, field)| matches!(solver.decs[field.dec].kind, DecKind::Item { .. }))
            .flat_map(|(name, field)| {
                // `label@B` (C3) dispatches by its own key, never collapsing onto
                // another pact's `label`
                if name.contains('@') {
                    vec![(name.clone(), field.dec)]
                } else {
                    slots(name, field.dec)
                }
            })
            .chain(
                adt.overloads
                    .iter()
                    .flat_map(|(name, fields)| fields.iter().map(move |f| (name, f.dec)))
                    .flat_map(|(name, dec)| slots(name, dec)),
            )
            .collect();
        for (j, i) in aliased {
            for k in solver.pacts[j].functions.keys() {
                let (Some(mj), Some(mi)) = (
                    solver.pact_members.get(&(j, k.clone())),
                    solver.pact_members.get(&(i, k.clone())),
                ) else {
                    continue;
                };
                if let Some(&dec) = methods.get(&solver.decs[*mi].name) {
                    methods.insert(solver.decs[*mj].name.clone(), dec);
                }
            }
        }

        let dispatch_ids = if adt.flags.contains(AdtFlags::IS_ENUM) {
            adt.variants.values().filter_map(Variant::layout).collect()
        } else {
            vec![aid]
        };

        let module = solver
            .decs
            .iter()
            .find(|(_, dec)| dec.kind == DecKind::Adt(aid))
            .map_or(AdtId::DANGLING, |(_, dec)| dec.module);

        Self {
            name: adt.name.clone(),
            module,
            fields,
            member_tys,
            member_dims,
            member_shapes,
            type_params,
            implements,
            methods,
            dispatch_ids,
        }
    }
}

impl From<Solver> for Resolutions {
    fn from(solver: Solver) -> Self {
        let node_tys = solver
            .node_to_vid
            .iter()
            .map(|(node_id, vid)| (*node_id, Ty::Vid(*vid).normalized(&solver)))
            .collect();

        let mut resolved_adts = IdVec::new();
        for (aid, adt) in solver.adts.iter() {
            resolved_adts.push(ResolvedAdt::new(aid, adt, &solver));
        }

        let root = ResolvedModule::new(&solver, AdtId::DANGLING);

        fn module_paths(
            solver: &Solver,
            adt: AdtId,
            prefix: &[String],
            out: &mut FxHashMap<AdtId, Vec<String>>,
        ) {
            for (name, field) in solver.adts[adt].as_struct().fields.iter() {
                let Some(child) = field.ty.as_adt().copied() else {
                    continue;
                };
                if !solver.adts[child].flags.contains(AdtFlags::IS_MODULE) {
                    continue;
                }
                let mut path = prefix.to_vec();
                path.push(name.clone());
                out.insert(child, path.clone());
                module_paths(solver, child, &path, out);
            }
        }
        let mut paths = FxHashMap::default();
        for (name, &adt) in solver.root_modules.iter() {
            paths.insert(adt, vec![name.clone()]);
            module_paths(&solver, adt, std::slice::from_ref(name), &mut paths);
        }

        let mut owners: FxHashMap<DecId, AdtId> = FxHashMap::default();
        for (aid, adt) in solver.adts.iter() {
            for field in adt.impls.values() {
                owners.insert(field.dec, aid);
            }
            if let Some(Variant::Struct(variant)) = adt.variants.values().next() {
                for field in variant.fields.values() {
                    owners.insert(field.dec, aid);
                }
            }
        }

        // a pact impl's methods pair with the pact's members by name (a default the impl left out
        // is already the member's own dec)
        let mut implements: FxHashMap<DecId, DecId> = FxHashMap::default();
        for &(pid, aid) in solver.pact_impls.iter() {
            for (name, field) in &solver.adts[aid].impls {
                // a `name@Pact` key pairs with that pact's `name` only
                let base = match name.split_once('@') {
                    Some((b, p)) if p == solver.pacts[pid].name => b.to_string(),
                    Some(_) => continue,
                    None => name.clone(),
                };
                if solver.member_pact.get(&field.dec).is_some_and(|p| *p != pid) {
                    continue;
                }
                if let Some(&member) = solver.pact_members.get(&(pid, base))
                    && member != field.dec
                {
                    implements.insert(field.dec, member);
                }
            }
        }
        // overload pairings were settled by operand at impl solve
        for (&dec, members) in solver.impl_member.iter() {
            if let Some(&member) = members.first() {
                implements.insert(dec, member);
            }
        }
        for (aid, adt) in solver.adts.iter() {
            for field in adt.overloads.values().flatten() {
                owners.insert(field.dec, aid);
            }
        }

        // contract checks emit as native calls -- find the real natives even if
        // user decls shadow the names (theirs won't be in dec_to_native).
        // computed before `decs` is consumed below.
        let contract_native = solver
            .decs
            .iter()
            .filter(|(id, _)| solver.dec_to_native.contains_key(id))
            .find(|(_, dec)| dec.name == "__contract_fail")
            .map(|(id, _)| solver.dec_to_native[&id].id);
        let cleanup_natives = solver.decs.iter()
            .find(|(id, dec)| matches!(dec.name.as_str(), "__cleanup_register" | "__cleanup_promote")
                && solver.dec_to_native.contains_key(id)
                && solver.node_decs.values().any(|used| used == id))
            .and_then(|_| {
                let native = |name: &str| solver.decs.iter()
                    .find(|(id, dec)| dec.name == name && solver.dec_to_native.contains_key(id))
                    .map(|(id, _)| solver.dec_to_native[&id].id);
                Some((native("__cleanup_mark")?, native("__cleanup_take")?))
            });
        let echo_native = solver
            .decs
            .iter()
            .filter(|(id, _)| solver.dec_to_native.contains_key(id))
            .find(|(_, dec)| dec.name == "echo")
            .map(|(id, _)| solver.dec_to_native[&id].id);
        let in_unit_native = solver
            .decs
            .iter()
            .filter(|(id, _)| solver.dec_to_native.contains_key(id))
            .find(|(_, dec)| dec.name == "__in_unit")
            .map(|(id, _)| solver.dec_to_native[&id].id);
        let shown_as_native = solver
            .decs
            .iter()
            .filter(|(id, _)| solver.dec_to_native.contains_key(id))
            .find(|(_, dec)| dec.name == "__shown_as")
            .map(|(id, _)| solver.dec_to_native[&id].id);
        let shown_native = solver
            .decs
            .iter()
            .filter(|(id, _)| solver.dec_to_native.contains_key(id))
            .find(|(_, dec)| dec.name == "__shown")
            .map(|(id, _)| solver.dec_to_native[&id].id);
        let freeze_native = solver
            .decs
            .iter()
            .filter(|(id, _)| solver.dec_to_native.contains_key(id))
            .find(|(_, dec)| dec.name == "__freeze")
            .map(|(id, _)| solver.dec_to_native[&id].id);
        let panic_native = solver
            .decs
            .iter()
            .filter(|(id, _)| solver.dec_to_native.contains_key(id))
            .find(|(_, dec)| dec.name == "panic")
            .map(|(id, _)| solver.dec_to_native[&id].id);

        let tys: Vec<Ty> = solver
            .decs
            .iter()
            .map(|(_, dec)| Ty::Vid(dec.vid).normalized(&solver))
            .collect();

        let mut resolved_decs: IdVec<DecId, ResolvedDecl> = IdVec::new();
        for ((id, dec), ty) in solver.decs.into_iter().zip(tys) {
            let vis = dec.vis;
            let kind = match dec.kind {
                DecKind::Local | DecKind::LoopVar => ResolvedDeclKind::Local,
                DecKind::Global => ResolvedDeclKind::Global,
                DecKind::Item { defaults, .. } => ResolvedDeclKind::Item {
                    defaults,
                    native: solver.dec_to_native.get(&id).map(|b| b.id),
                    takes_self: match solver.dec_to_native.get(&id) {
                        Some(binding) => binding.takes_self,
                        None => matches!(&ty, Ty::Fn(header) if header.is_method),
                    },
                    src: solver.dec_to_native.get(&id).and_then(|b| b.src),
                    doc: solver
                        .dec_to_native
                        .get(&id)
                        .and_then(|b| (!b.doc.is_empty()).then(|| b.doc.clone())),
                },
                DecKind::Constant(Some(lit)) => ResolvedDeclKind::Constant(lit),
                DecKind::Constant(None) => panic!(
                    "dec {:?} ({}) reached IR boundary as `DeclKind::Constant(None)`",
                    id, dec.name
                ),
                DecKind::Variant { parent, layout } => ResolvedDeclKind::Variant { parent, layout },
                DecKind::Adt(a) => ResolvedDeclKind::Adt(a),
                DecKind::Pact(p) => ResolvedDeclKind::Pact(p),
            };
            resolved_decs.push(ResolvedDecl {
                name: dec.name,
                ty,
                kind,
                vis,
                location: dec.location,
                module: dec.module,
                owner: owners.get(&id).copied(),
                implements: implements.get(&id).copied(),
            });
        }

        let mut pact_names = IdVec::new();
        for (_, pact) in solver.pacts.iter() {
            pact_names.push(pact.name.clone());
        }

        let closure_captures = solver
            .closure_captures
            .into_iter()
            .map(|(node, decs)| (node, decs.into_iter().collect()))
            .collect();

        let multis: FxHashMap<DecId, ResolvedMulti> = solver
            .multis
            .iter()
            .map(|(gdec, data)| {
                (
                    *gdec,
                    ResolvedMulti {
                        methods: data
                            .methods
                            .iter()
                            .map(|m| ResolvedMultiMethod {
                                dec: m.dec,
                                keys: m
                                    .keys
                                    .iter()
                                    .map(|k| k.as_ref().map(|s| s.iter().copied().collect()))
                                    .collect(),
                            })
                            .collect(),
                    },
                )
            })
            .collect();

        // a `multi fn`'s callers dispatch through the generic's dec, so the
        // fixpoint never unioned the method arms' effects into it -- do that
        // here, before the callers' grades are read. r[impl mimas.multi]
        let mut fn_effects = solver.fn_effects.clone();
        for (gdec, data) in &solver.multis {
            let union = data
                .methods
                .iter()
                .filter_map(|m| fn_effects.get(&m.dec).copied())
                .fold(shared::Fx::empty(), |acc, fx| acc | fx);
            if union != shared::Fx::empty() {
                *fn_effects.entry(*gdec).or_insert(shared::Fx::empty()) |= union;
            }
        }

        Self {
            node_tys,
            node_dims: solver.node_dims,
            want_dims: solver.want_dims,
            print_dims: solver.print_dims,
            echo_units: solver.echo_units,
            print_shapes: solver.print_shapes,
            hole_shapes: solver.hole_shapes,
            shown_units: solver.shown_units,
            shown_native,
            built_in_holes: solver.built_in_holes,
            shown_as_native,
            check_units: solver.check_units,
            check_texts: solver.check_texts,
            dec_scales: solver.dec_scales,
            sign_negs: solver.sign_negs,
            node_decs: solver.node_decs,
            rewrites: solver.rewrites,
            dec_uses: solver.dec_uses,
            fn_effects,
            script_effects: solver.script_effects,
            script_segments: solver.script_segments,
            page_needs: solver.page_needs,
            warnings: crate::load::unwrap_warnings(solver.warnings),
            decs: resolved_decs,
            adts: resolved_adts,
            pact_names,
            module_paths: paths,
            closure_captures,
            fn_writes: solver
                .fn_writes
                .into_iter()
                .map(|(d, ws)| {
                    (
                        d,
                        ws.into_iter()
                            .map(|(p, path)| {
                                (
                                    p,
                                    path.into_iter()
                                        .map(|s| match s {
                                            crate::GuardSeg::Field(f) => f,
                                            crate::GuardSeg::Index => "[]".to_string(),
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            inout_calls: solver.inout_calls,
            inout_writebacks: solver.inout_writebacks,
            self_place_mints: solver.self_place_mints,
            len_checks: solver.len_checks,
            default_ctors: solver.default_ctors,
            contract_native,
            echo_native,
            cleanup_natives,
            in_unit_native,
            freeze_native,
            panic_native,
            op_decs: solver.op_decs,
            propagates: solver.propagates,
            int_as_float: solver.int_as_float,
            dec_literals: solver.dec_literals,
            dec_texts: solver.dec_texts,
            computed_consts: solver
                .computed_consts
                .iter()
                .map(|(id, _, _)| *id)
                .collect(),
            field_hints: solver.field_hints,
            ctor_slots: solver.ctor_slots,
            atom_layouts: solver.atoms.layouts,
            pact_ops: solver.pact_ops,
            multis,
            multi_calls: solver.multi_calls,
            param_bounds: solver
                .param_bounds
                .iter()
                .map(|(pid, pacts)| (*pid, pacts.clone()))
                .collect(),
            root,
            sources: solver.sources,
        }
    }
}

/// A `multi fn`'s dispatch table: its `method` arms in declaration order.
/// Emit's chain is first-match -- an arm wins when every position passes, and
/// the generic's own body is the final fallback. r[impl mimas.multi]
#[derive(Debug, Clone)]
pub struct ResolvedMulti {
    pub methods: Vec<ResolvedMultiMethod>,
}

/// One `method` arm: `dec` is its body's fn dec; `keys[i]` is the set of
/// runtime ids position `i` accepts (`None` = no test, matches anything).
#[derive(Debug, Clone)]
pub struct ResolvedMultiMethod {
    pub dec: DecId,
    pub keys: Vec<Option<Vec<AdtId>>>,
}

/// What the grades pass learned about a load, handed to hosts deciding whether
/// to run the code at all -- mobile-code proof-carrying, half one: the page's
/// `Fx` is a checked property before a single bytecode executes.
pub struct GradeAudit {
    /// Each file's inferred top-level effect set, keyed by the file name the
    /// caller gave `load_files`. `UNAUDITED`-flagged sets contain calls the
    /// inference couldn't grade (unannotated natives, pact dispatch, closure
    /// callees) -- treat them as "could do anything" when gating.
    pub script_effects: FxIndexMap<String, shared::Fx>,
    /// Per top-level statement: `(file name, site, inferred fx)` in source
    /// order -- a spliced `use`-closure attributes each include back to its
    /// byte range, so per-page verdicts exist even though the program is one file.
    pub script_segments: Vec<(String, shared::Location, shared::Fx)>,
    /// Lint-suite diagnostics (unused bindings/params, dead stores). Severity
    /// warning; the load succeeded -- the host renders or surfaces these.
    pub warnings: Vec<miette::Report>,
}

impl GradeAudit {
    /// Every file whose effect set escapes `allowed` -- the gate verdict.
    /// `fits` is deliberate about `UNAUDITED`: an ungraded file fits only when
    /// the host's grant also carries the flag.
    pub fn violations(&self, allowed: shared::Fx) -> Vec<(&str, shared::Fx)> {
        self.script_effects
            .iter()
            .filter(|(_, fx)| !fx.fits(allowed))
            .map(|(name, fx)| (name.as_str(), *fx))
            .collect()
    }
}
