use crate::{
    Error, Result, Unification, UnificationError,
    components::*,
    errors::{
        AssignToConst, AssignToLoopVar, AssignToStringIndex, BareNullBinding, BoundNotSatisfied,
        CheckIsNotCallable, ExtraTupleMembers, FieldNotFound, InvalidAssignTarget, InvalidPattern,
        InvalidUseTarget, IteratingHere, MissingTupleMembers, MultipleConstDeclarations,
        MutateWhileIterating, NonConstantValue, NotAPact, NotFound, SelfOutOfContext,
    },
    traits::*,
};
use crate::{FxHashMap, FxHashSet};
use crate::{FxIndexMap, FxIndexSet};
use api::{ApiAdt, ApiAdtKind, ApiEntry, ApiVariantFields, Library, NativeId};
use miette::NamedSource;
use parse::{
    ExprKind,
    components::{Annotation, Binding, Pat, PatKind, TypeParam},
    *,
};
use shared::{FileId, IdVec, Literal as RawLiteral, Located, Location, PactId, ParamId};
use std::{collections::BTreeSet, rc::Rc, sync::Arc};

/// A table the prepared state (`load::Prepared`) shares with every solve
/// cloned from it: reads go straight through, the first write copies
/// (`Arc::make_mut`). A solve that never writes one pays a pointer copy
/// for it instead of a deep clone — and since it is an `Arc`, solves on
/// other threads could share the same table.
#[derive(Clone, Default)]
pub(crate) struct Shared<T>(Arc<T>);

impl<T> Shared<T> {
    pub(crate) fn new(t: T) -> Self {
        Self(Arc::new(t))
    }
}

impl<T> std::ops::Deref for Shared<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Clone> std::ops::DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut T {
        Arc::make_mut(&mut self.0)
    }
}

#[derive(Clone)]
pub struct Solver {
    pub(crate) decs: IdVec<DecId, Dec>,
    pub(crate) adts: IdVec<AdtId, Adt>,
    pub(crate) pacts: Shared<IdVec<PactId, shared::Pact>>,
    pub(crate) pact_impls: Shared<FxHashSet<(PactId, AdtId)>>,
    pub(crate) pact_members: Shared<FxHashMap<(PactId, String), DecId>>,
    pub(crate) pact_self: Option<PactId>,
    pub(crate) node_to_vid: FxIndexMap<NodeId, Vid>,
    pub(crate) node_decs: FxIndexMap<NodeId, DecId>,
    pub(crate) op_decs: FxIndexMap<NodeId, crate::OpDispatch>,
    pub(crate) pact_ops: FxIndexMap<NodeId, crate::PactOp>,
    /// The pacts each declared type param is bounded by (`T: Draw`), by the param's id.
    /// Absent means unbounded. Fn and adt params share the map -- ids are global.
    pub(crate) param_bounds: FxHashMap<ParamId, Vec<PactId>>,
    /// A pact's overload groups: base name → the keys in `Pact::functions` that
    /// spell it (`mul` → `["mul", "mul#1"]`), first declared first. A name
    /// declared once has no entry. r[impl mimas.pact.overload]
    pub(crate) pact_groups: FxHashMap<(PactId, String), Vec<String>>,
    /// Which pact member an impl's fn implements, for the fns whose pairing
    /// isn't by name alone (an overload): impl dec → member dec.
    pub(crate) impl_member: FxHashMap<DecId, Vec<DecId>>,
    /// Bounds named before their pact was hoisted (a struct's `<T: Draw>` is read in
    /// `hoist_types`, ahead of `hoist_pacts`) -- settled by `resolve_pending_bounds`.
    pub(crate) pending_bounds: Vec<(ParamId, Ident)>,
    /// Parameter defaults that didn't reduce at `hoist_callables` -- a default naming a
    /// `const` the fixpoint hasn't solved yet -- settled by `resolve_pending_defaults`:
    /// the fn's dec, the parameter's index, the default's expr.
    pub(crate) pending_defaults: Vec<(DecId, usize, Expr)>,
    /// Every instantiation of a bounded param: the vid it became, the pact it owes, and the
    /// use that incurred it. Checked once bodies are solved (`check_bounds`).
    pub(crate) bound_obligations: Vec<(Ty, PactId, Location)>,
    /// Where the instantiation under way is written, so `substitute_params` can record an
    /// obligation against it. `None` when the caller has no use site to blame.
    pub(crate) obligation_at: Option<Location>,
    pub(crate) closure_captures: FxIndexMap<NodeId, FxIndexSet<DecId>>,
    pub(crate) expected_closures: FxIndexMap<NodeId, Vec<FnParam>>,
    vids: IdVec<Vid, ()>,
    subs: IdVec<Vid, Option<Ty>>,
    node_visits: FxHashSet<NodeId>,
    pub(crate) library: Shared<FxHashMap<String, AdtId>>,
    pub(crate) root_modules: Shared<FxIndexMap<String, AdtId>>,
    pub(crate) module_items: Shared<FxIndexMap<AdtId, FxIndexMap<String, DecId>>>,
    pub(crate) sources: FxHashMap<FileId, NamedSource<Arc<str>>>,

    pub(crate) dec_to_native: Shared<FxHashMap<DecId, NativeBinding>>,

    /// Mimas `fn`s recorded at solve (`Function::solve`), for the const call
    /// fold: dec → params + solved body. Populated per program — not `Shared`.
    pub(crate) const_items: FxHashMap<DecId, ConstItem>,
    /// `true` while the early solve_consts fixpoint runs: a `const` whose rhs
    /// holds a call to a Mimas fn defers rather than erroring — the callee's
    /// body isn't solved (and so not recorded for the fold) until
    /// `solve_bodies`. False everywhere else; the post-bodies retry errors.
    pub(crate) defer_call_consts: bool,

    // dataframe schemas (see frames.rs): a frame's Ty stays the nominal `DataFrame` while
    // these carry its column map -- keyed by producing expr and by `let` binding.
    pub(crate) frame_schemas: FxHashMap<NodeId, Rc<crate::frames::FrameSchema>>,
    pub(crate) frame_decs: FxHashMap<DecId, Rc<crate::frames::FrameSchema>>,
    /// `__q_group`/`group_by` results: `(input schema, key columns)` by expr/binding.
    pub(crate) group_schemas: FxHashMap<NodeId, crate::frames::GroupInfo>,
    pub(crate) group_decs: FxHashMap<DecId, crate::frames::GroupInfo>,
    /// Unit dims of `25kW`-style literals, merged from every solved ast (`NodeId`s are
    /// globally unique, so one map is collision-free).
    pub(crate) ast_quantities: FxHashMap<NodeId, shared::units::Dim>,
    /// The unit text as written for each literal in `ast_quantities`, so diagnostics can
    /// name `ft` rather than collapsing it to `length (m)`.
    pub(crate) ast_quantity_units: FxHashMap<NodeId, String>,
    /// Bare unit names resolved as constants (`kWh` reads as `1kWh`): unit text -> its
    /// synthetic `Constant` dec and dimension, deduped so every use points at one dec.
    /// r[impl mimas.unit.constant]
    pub(crate) unit_consts: FxHashMap<String, (DecId, shared::units::Dim)>,
    /// Library constants declared with a dimension (`PI` is `rad`): the
    /// constant's dec -> its dimension, seeded into the dims pass like
    /// `unit_consts` so the constant's uses type as quantities.
    pub(crate) const_dims: Vec<(DecId, shared::units::Dim)>,
    /// Every expression's computed dimension, recorded by the dims pass —
    /// `b.x + 2.0px` is `px` here while its `node_tys` stays `Float`.
    /// Surfaces on [`Resolutions::node_dims`](crate::Resolutions).
    pub(crate) node_dims: FxIndexMap<NodeId, shared::units::Dim>,
    /// The dimension each expr's context demanded (a call arg's declared param
    /// dim) — an expr whose own dim is `Any` still knows the unit its slot wanted.
    /// Surfaces on [`Resolutions::want_dims`](crate::Resolutions).
    pub(crate) want_dims: FxIndexMap<NodeId, shared::units::Dim>,
    /// `print`/`dbg`/`warn` call site → the dimension the pass computed for the printed
    /// value — so the natives can render `25000 W` rather than a bare `25000`.
    /// Keyed by call location, the same span the `Prints` fixture attributes.
    /// Surfaces on [`Resolutions::print_dims`](crate::Resolutions).
    pub(crate) print_dims: FxIndexMap<shared::Location, shared::units::Dim>,
    /// Top-level echo site → how its quantity shows (the unit as written).
    /// Surfaces on [`Resolutions::echo_units`](crate::Resolutions).
    pub(crate) echo_units: FxIndexMap<shared::Location, shared::units::Shown>,
    /// The type each bare-brace literal's slot expects (`{ x: 1 }` passed where a
    /// `Pt` goes), noted by `fulfill_ty` before the literal is solved — a struct
    /// hint makes it that struct, anything else leaves it a dict.
    /// r[impl mimas.record.context]
    pub(crate) record_hints: FxHashMap<NodeId, Ty>,
    /// Dimensions of api-adt fields (`Input.dt` is `s`): `(adt name, field name)`.
    /// Populated by `install_user_adt` from `ApiVariantFields::Named` dims; the dims pass
    /// consults it when a `.field` read isn't on a source-declared struct.
    pub(crate) field_dims: FxHashMap<(String, String), shared::units::Dim>,
    /// Declared unit of each source-struct member, through `?`/`[..]`:
    /// `(struct name, field name)`. `next: index?` and `kids: [idx]` both
    /// land as `INDEX`. Recorded by the dims pass; surfaces on
    /// [`ResolvedAdt::member_dims`](crate::ResolvedAdt).
    pub(crate) member_units: FxHashMap<(String, String), shared::units::Dim>,
    /// Exprs that fulfill a `[T; n]` position with a length the compiler couldn't
    /// prove -- emit wraps each in a runtime dim check (see `lens.rs`).
    /// Surfaces on [`Resolutions::len_checks`](crate::Resolutions).
    pub(crate) len_checks: FxIndexMap<NodeId, Vec<Option<usize>>>,
    /// `S::default()` calls the solver blessed: call node → the struct's adt.
    /// Emit lowers each to a `NewInstance` with every member set to its
    /// type's default rather than a call.
    pub(crate) default_ctors: FxIndexMap<NodeId, AdtId>,

    /// Per-declaration read/write site counts, filled by the grades pass -- the `{0,1,ω}`
    /// usage grades behind the lint suite. Surfaces on [`Resolutions::dec_uses`].
    pub(crate) dec_uses: FxIndexMap<DecId, crate::grades::UseInfo>,
    /// Inferred effect sets for each `fn` dec (`#[effects]`-declared natives seed the
    /// fixpoint; user fns are the union of their callees). `Resolutions::fn_effects`.
    pub(crate) fn_effects: FxIndexMap<DecId, shared::Fx>,
    /// Effect set of each file's top-level statements, keyed by the ast's name. Script code
    /// isn't a dec, so page-eval effects surface here for a host to check against its caps.
    pub(crate) script_effects: FxIndexMap<String, shared::Fx>,
    /// Per top-level statement: `(file name, site, inferred fx)` in source order. A host
    /// that splices `use`-includes into one file maps each site back onto its include's
    /// byte range -- how a peer page's contribution is attributed (and gated) separately.
    pub(crate) script_segments: Vec<(String, shared::Location, shared::Fx)>,
    /// Non-fatal diagnostics (the lint suite). Solve succeeds; hosts decide how to show them.
    /// Surfaces on `Loaded::warnings` and `Resolutions::warnings`.
    /// Lints from the grades pass, behind `Rc` so a prepared solver (see
    /// `load::Prepared`) clones; `Loaded` unwraps them on the way out.
    pub(crate) warnings: Vec<std::rc::Rc<miette::Report>>,
    /// Modules brought in by a bare `use music;` (which binds the name,
    /// not the members) -- an unbound-name error that names a member of
    /// one of these gets the "you wanted `::*`" hint.
    pub(crate) bare_module_uses: FxHashMap<String, AdtId>,
    /// Each `use` item's resolved bindings, by the item's node id. A use
    /// resolves once; every phase (and `visit_stmt`, for block-level uses)
    /// installs the same decs into the current ribs instead of re-walking
    /// the path and minting fresh decs each time.
    pub(crate) resolved_uses: FxHashMap<NodeId, ResolvedUse>,
    /// Set while the prelude group (the built-in source and the library's
    /// own sources) is being solved -- `load_files` solves that group to
    /// completion before any user file, so a user `enum PosNeg` can't catch
    /// the built-in `impl Opp for PosNeg` mid-hoist. Types hoisted under it
    /// land in `prelude_types`; a user type of the same name shadows it
    /// for user code, with a `shadowed_builtin` warning.
    pub(crate) in_prelude: bool,
    pub(crate) prelude_types: FxHashSet<String>,

    pub(crate) control_flow: ControlFlow,
    pub(crate) ribs: Ribs,
    pub(crate) loop_stack: Vec<LoopRun>,
    pub(crate) fn_stack: Vec<FnRun>,
    pub(crate) non_value: Option<NodeId>,
    /// Collections a `for` loop is iterating, pushed for the duration of each body's solve.
    pub(crate) iter_guards: Vec<IterGuard>,

    /// In-scope declared type params (`fn map<T, U>`, `impl Pair` re-entering `Pair<A, B>`),
    /// innermost last. `Ty::from_annotation` resolves a bare `T` through these before ribs,
    /// units, or builtins. Entries persist only for the duration of one item's solve.
    pub(crate) type_params: Vec<FxHashMap<String, ParamId>>,
}

// Public impls
impl Solver {
    pub fn new() -> Self {
        let mut solver = Self {
            vids: IdVec::new(),
            subs: IdVec::new(),
            decs: IdVec::new(),
            adts: IdVec::new(),
            pacts: Shared::new(IdVec::new()),
            pact_impls: Shared::default(),
            pact_members: Shared::default(),
            pact_self: None,
            node_to_vid: FxIndexMap::default(),
            node_decs: FxIndexMap::default(),
            op_decs: FxIndexMap::default(),
            pact_ops: FxIndexMap::default(),
            closure_captures: FxIndexMap::default(),
            expected_closures: FxIndexMap::default(),
            control_flow: ControlFlow::new(),
            node_visits: FxHashSet::default(),
            ribs: Ribs::default(),
            loop_stack: vec![],
            fn_stack: vec![],
            library: Shared::default(),
            root_modules: Shared::default(),
            module_items: Shared::default(),
            sources: FxHashMap::default(),
            dec_to_native: Shared::default(),
            const_items: FxHashMap::default(),
            defer_call_consts: false,
            frame_schemas: FxHashMap::default(),
            frame_decs: FxHashMap::default(),
            group_schemas: FxHashMap::default(),
            group_decs: FxHashMap::default(),
            ast_quantities: FxHashMap::default(),
            ast_quantity_units: FxHashMap::default(),
            unit_consts: FxHashMap::default(),
            const_dims: Vec::new(),
            node_dims: FxIndexMap::default(),
            want_dims: FxIndexMap::default(),
            print_dims: FxIndexMap::default(),
            echo_units: FxIndexMap::default(),
            record_hints: FxHashMap::default(),
            field_dims: FxHashMap::default(),
            member_units: FxHashMap::default(),
            len_checks: FxIndexMap::default(),
            default_ctors: FxIndexMap::default(),
            dec_uses: FxIndexMap::default(),
            fn_effects: FxIndexMap::default(),
            script_effects: FxIndexMap::default(),
            script_segments: Vec::new(),
            warnings: Vec::new(),
            bare_module_uses: FxHashMap::default(),
            resolved_uses: FxHashMap::default(),
            in_prelude: false,
            prelude_types: FxHashSet::default(),
            non_value: None,
            iter_guards: vec![],
            type_params: vec![],
            param_bounds: FxHashMap::default(),
            pact_groups: FxHashMap::default(),
            impl_member: FxHashMap::default(),
            pending_bounds: vec![],
            pending_defaults: vec![],
            bound_obligations: vec![],
            obligation_at: None,
        };
        // root frame: [Import, Module, Script] -- see `Ribs::default` for why
        // the import rib sits beneath the module rib.
        solver.ribs.push_script();

        // todo, this is a bit brittle
        for name in ["int", "float", "str", "bool", "array", "dict"] {
            let mut adt = Adt::new_struct(name.into());
            adt.flags |= AdtFlags::IS_BUILTIN;
            let id = solver.push_adt(adt);
            solver.library.insert(name.into(), id);
        }
        solver
    }

    /// Install the per-file source map for diagnostic rendering. Call before `solve_all`.
    pub fn set_sources(
        &mut self,
        sources: impl IntoIterator<Item = (FileId, NamedSource<Arc<str>>)>,
    ) {
        self.sources = sources.into_iter().collect();
    }

    /// Look up the `NamedSource` for a `Location`'s file. Falls back to an empty `<unknown>` source
    /// if the file isn't in the map. This is for the sake of tests, we could use cfgs and be more
    /// strict if we wanted
    pub(crate) fn src(&self, loc: Location) -> NamedSource<Arc<str>> {
        self.sources
            .get(&loc.file_id)
            .cloned()
            .unwrap_or_else(|| NamedSource::new("<unknown>", Arc::<str>::from("")))
    }

    /// Cross-module visibility check. A `Private` dec whose home module differs from the
    /// one currently being solved is unreachable -- errors at the access location. Decs
    /// homed at `AdtId::DANGLING` (builtins, natives, locals) are always reachable.
    pub(crate) fn check_vis(&self, dec: DecId, access_at: Location) -> Result<()> {
        let dec = &self.decs[dec];
        if dec.vis == Vis::Private
            && dec.module != AdtId::DANGLING
            && dec.module != self.ribs.current_module()
        {
            Err(crate::errors::PrivateAccess {
                src: self.src(access_at),
                name: dec.name.clone(),
                access_at: access_at.into(),
                declaration: vec![crate::errors::PrivateDeclaredHere {
                    src: self.src(dec.location),
                    at: dec.location.into(),
                    name: dec.name.clone(),
                }],
            })?
        }
        Ok(())
    }

    /// Errors if `path` off `dec` names a place a `for` loop is iterating -- the write or
    /// mutating call at `at` would change the collection mid-loop. Guards compare on the
    /// *resolved* root dec plus the access path, so shadowed names and sibling fields don't
    /// false-positive. Deliberately not an alias analysis (see `IterGuard`).
    pub(crate) fn check_iter_guard(
        &self,
        dec: DecId,
        path: &[GuardSeg],
        at: Location,
    ) -> Result<()> {
        if let Some(guard) = self
            .iter_guards
            .iter()
            .find(|g| g.dec == dec && g.path == *path)
        {
            Err(MutateWhileIterating {
                src: self.src(at),
                at: at.into(),
                name: guard.name.clone(),
                iterating: vec![IteratingHere {
                    src: self.src(guard.iterated_at),
                    at: guard.iterated_at.into(),
                    name: guard.name.clone(),
                }],
            })?
        }
        Ok(())
    }

    /// Every field of a `space` is an axis: its type implements `Opp` (an `opp`,
    /// `bool`, a number, a `Dial`, another space). A field that isn't gets the
    /// targeted error here, before the synthesized flip would trip over it.
    // r[impl mimas.space.axes]
    fn check_space_axes(&mut self, st: &parse::Struct) -> Result<()> {
        let opp = self
            .resolve_name(&Ident::synthetic("Opp"), st.name.location)?
            .as_single_pact();
        for field in &st.fields {
            let ty = Ty::from_annotation(field.annotation.clone(), self)?.normalized(self);
            let aid = match &ty {
                Ty::Adt(a, _) | Ty::Identity(a, _) => Some(*a),
                other => self.builtin_adt(other),
            };
            let axis = opp
                .zip(aid)
                .is_some_and(|(p, a)| self.pact_impls.contains(&(p, a)));
            if !axis {
                let name = match &field.name {
                    parse::FieldKey::Ident(i) => i.lexeme.clone(),
                    parse::FieldKey::Int(n) => n.to_string(),
                };
                Err(crate::errors::SpaceFieldNotAnAxis {
                    src: self.src(field.location),
                    at: field.location.into(),
                    space: st.name.lexeme.clone(),
                    field: name,
                    ty: ty.to_string(),
                })?;
            }
        }
        Ok(())
    }

    /// The primitive whose library adt `aid` is (`bool` for the `bool`
    /// namespace) -- the reverse of [`Self::builtin_adt`] for the scalars.
    pub(crate) fn primitive_of(&self, aid: AdtId) -> Option<Ty> {
        [
            ("int", Ty::Int),
            ("float", Ty::Float),
            ("str", Ty::Str),
            ("bool", Ty::Bool),
        ]
        .into_iter()
        .find(|(name, _)| self.library.get(*name) == Some(&aid))
        .map(|(_, ty)| ty)
    }

    pub(crate) fn builtin_adt(&self, ty: &Ty) -> Option<AdtId> {
        let name = match ty {
            Ty::Int => "int",
            Ty::Float => "float",
            Ty::Str => "str",
            Ty::Bool => "bool",
            Ty::Array(..) => "array",
            Ty::Dict(_) => "dict",
            _ => return None,
        };
        self.library.get(name).copied()
    }

    pub fn solve(&mut self, ast: &Ast) -> Result<()> {
        self.solve_all([ast])
    }

    pub fn solve_all<'a>(&mut self, asts: impl IntoIterator<Item = &'a Ast>) -> Result<()> {
        let asts: Vec<_> = asts.into_iter().collect();
        // `MIMAS_SOLVE_TIMING=1` prints each phase's time to stderr — a
        // probe for "where does a solve go", costing nothing when unset.
        let timing = std::env::var_os("MIMAS_SOLVE_TIMING").is_some();
        // `Instant::now` only when asked: on wasm32 it panics ("time not
        // implemented"), and the shipped solver runs there
        let mut last = timing.then(std::time::Instant::now);
        let mut lap = move |phase: &str| {
            if let Some(prev) = last {
                let now = std::time::Instant::now();
                eprintln!("SOLVE {phase}: {:?}", now - prev);
                last = Some(now);
            }
        };
        for ast in &asts {
            self.ast_quantities.extend(ast.quantities());
            self.ast_quantity_units.extend(
                ast.quantity_units()
                    .map(|(id, unit)| (id, unit.to_string())),
            );
        }
        let module_names: Vec<Option<String>> = asts
            .iter()
            .map(|ast| ast.module_name())
            .collect::<Result<_>>()?;

        // pre-allocate one Adt per named module so cross-file refs can resolve at hoist time.
        // nested decls (`module a::b;`) build the parent chain; the file's rib binds the leaf
        let mut module_adts: FxHashMap<String, AdtId> = FxHashMap::default();
        for name in module_names.iter().flatten() {
            let segments: Vec<String> = name.split("::").map(str::to_string).collect();
            let id = self.ensure_module_path(&segments);
            module_adts.insert(name.clone(), id);
            self.root_modules
                .insert(segments[0].clone(), self.library[&segments[0]]);
        }

        // each ast's module rib accumulates across phases -- saved between visits
        let mut saved_ribs: Vec<Option<Rib>> = vec![None; asts.len()];

        let run_phase = |solver: &mut Self,
                         phase: fn(&mut Self, &Ast) -> Result<()>,
                         saved_ribs: &mut Vec<Option<Rib>>|
         -> Result<()> {
            for (i, ast) in asts.iter().enumerate() {
                let module = module_names[i].as_ref();
                if let Some(name) = module {
                    let rib = saved_ribs[i]
                        .take()
                        .unwrap_or_else(|| Rib::new(RibKind::Module(module_adts[name])));
                    // [Import, Module, Script] like the root frame: the file's
                    // own items outrank what its `use`s pull in.
                    solver.ribs.push_import();
                    solver.ribs.push_rib(rib);
                    solver.ribs.push_script();
                }
                phase(solver, ast)?;
                if let Some(name) = module {
                    solver.ribs.pop_script();
                    let rib = solver.ribs.pop_module();
                    solver.ribs.pop_import();
                    solver.sync_module_adt(module_adts[name], &rib);
                    saved_ribs[i] = Some(rib);
                }
            }
            Ok(())
        };

        run_phase(self, Self::hoist_types, &mut saved_ribs)?;
        lap("hoist_types");
        run_phase(self, Self::hoist_pacts, &mut saved_ribs)?;
        lap("hoist_pacts");
        // constants are declared ahead of the callables so a parameter default can name
        // one (`fn panel(fill: str = G_DEEP)`); their values come later, in the fixpoint
        run_phase(self, Self::hoist_constants, &mut saved_ribs)?;
        lap("hoist_constants");
        run_phase(self, Self::hoist_callables, &mut saved_ribs)?;
        lap("hoist_callables");
        run_phase(self, Self::resolve_pending_bounds, &mut saved_ribs)?;
        lap("resolve_pending_bounds");

        // fixpoint: hoist_uses + solve_consts. each round may resolve more consts (e.g. a const
        // that references another module's const reduces only once that other const is bound).
        // re-run while the count of consts-with-known-values grows, capped as a safety net.
        const FIXPOINT_CAP: usize = 16;
        let mut prev = self.resolved_constants();
        self.defer_call_consts = true;
        for _ in 0..FIXPOINT_CAP {
            run_phase(self, Self::hoist_uses, &mut saved_ribs)?;
            lap("hoist_uses");
            run_phase(self, Self::solve_consts, &mut saved_ribs)?;
            lap("solve_consts");
            let curr = self.resolved_constants();
            if curr == prev {
                break;
            }
            prev = curr;
        }
        self.defer_call_consts = false;

        // the consts are as solved as they get before bodies: the parameter defaults that
        // waited on one reduce now
        run_phase(self, Self::resolve_pending_defaults, &mut saved_ribs)?;
        lap("resolve_pending_defaults");

        run_phase(self, Self::solve_types, &mut saved_ribs)?;
        lap("solve_types");
        run_phase(self, Self::solve_bodies, &mut saved_ribs)?;
        lap("solve_bodies");

        // a `const` that deferred on a Mimas-fn call now has the callee bodies
        // it needs — retry (r[mimas.const.eval]); anything still unresolvable
        // errors for real this time
        run_phase(self, Self::solve_consts, &mut saved_ribs)?;
        lap("solve_consts (deferred)");

        // body solving is the last chance for an impl-associated or block-scoped const to gain
        // its literal value. anything still at `Constant(None)` afterwards is unresolvable --
        // almost always a self/mutual cycle. surface the first such dec so it doesn't reach IR.
        if let Some((_, dec)) = self
            .decs
            .iter()
            .find(|(_, d)| matches!(d.kind, DecKind::Constant(None)))
        {
            let location = dec.location;
            let name = dec.name.clone();
            Err(crate::errors::UnresolvedConst {
                src: self.src(location),
                at: location.into(),
                name,
            })?;
        }

        self.control_flow = ControlFlow::default();
        assert!(self.fn_stack.is_empty());
        assert!(self.loop_stack.is_empty());
        // every bounded param that got filled: by a type that implements the pact?
        self.check_bounds()?;
        lap("check_bounds");
        // last, once every node has its type: do the units add up?
        crate::dims::check(self, &asts)?;
        lap("dims");
        // and, with decs resolved everywhere the walk reaches: the usage/effect grades
        crate::grades::check(self, &asts)?;
        lap("grades");
        Ok(())
    }

    fn for_each_item<F>(ast: &Ast, mut f: F) -> Result<()>
    where
        F: FnMut(&Item) -> Result<()>,
    {
        ast.stmts().iter().try_for_each(|stmt| {
            let StmtKind::Item(item) = stmt.kind() else {
                return Ok(());
            };
            f(item)
        })
    }

    fn hoist_pacts(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            let ItemKind::Pact(p) = item.kind() else {
                return Ok(());
            };
            let vis = if item.public() {
                Vis::Public
            } else {
                Vis::Private
            };
            p.hoist(HoistCtx::new(
                self,
                HoistTarget::Scope,
                Some(item.id()),
                item.location(),
                vis,
            ))
        })
    }

    fn hoist_types(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            let vis = if item.public() {
                Vis::Public
            } else {
                Vis::Private
            };
            let ctx = HoistCtx::new(
                self,
                HoistTarget::Scope,
                Some(item.id()),
                item.location(),
                vis,
            );
            match item.kind() {
                ItemKind::Struct(s) => s.hoist(ctx),
                ItemKind::Enum(e) => e.hoist(ctx),
                _ => Ok(()),
            }
        })
    }

    fn hoist_callables(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            if let ItemKind::Use(_) = item.kind() {
                // best effort: an imported type must land before the fns
                // below hoist so a signature can name it. Whatever can't
                // resolve yet (a use of a member hoisted in *this* pass)
                // retries -- and reports -- at `hoist_uses`.
                let _ = self.process_use(item);
            }
            Ok(())
        })?;
        Self::for_each_item(ast, |item| {
            let vis = if item.public() {
                Vis::Public
            } else {
                Vis::Private
            };
            let ctx = HoistCtx::new(
                self,
                HoistTarget::Scope,
                Some(item.id()),
                item.location(),
                vis,
            );
            match item.kind() {
                ItemKind::Impl(i) => i.hoist(ctx),
                ItemKind::Function(f) => f.hoist(ctx),
                ItemKind::Tests(tests) => tests.hoist(ctx),
                _ => Ok(()),
            }
        })
    }

    fn hoist_constants(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            let ItemKind::Const(con) = item.kind() else {
                return Ok(());
            };
            let vis = if item.public() {
                Vis::Public
            } else {
                Vis::Private
            };
            con.hoist(HoistCtx::new(
                self,
                HoistTarget::Scope,
                Some(item.id()),
                item.location(),
                vis,
            ))
        })
    }

    fn hoist_uses(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            let ItemKind::Use(_) = item.kind() else {
                return Ok(());
            };
            self.process_use(item)
        })
    }

    fn solve_bodies(&mut self, ast: &Ast) -> Result<()> {
        ast.stmts().iter().try_for_each(|v| self.visit_stmt(v))
    }

    fn solve_types(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            if let ItemKind::Use(_) = item.kind() {
                // repopulate the phase's import rib (all resolves are cached
                // by now -- `hoist_uses` reported anything that failed)
                self.process_use(item)?;
            }
            Ok(())
        })?;
        Self::for_each_item(ast, |item| {
            if !matches!(item.kind(), ItemKind::Struct(_) | ItemKind::Enum(_))
                || !self.touch_node(item.id())
            {
                return Ok(());
            }
            let ty = match item.kind() {
                ItemKind::Struct(s) => s.solve(item.id(), item.location(), self)?,
                ItemKind::Enum(e) => e.solve(item.id(), item.location(), self)?,
                _ => unreachable!(),
            };
            let vid = self.node_vid(item.id());
            self.register_sub(vid, ty)
                .map_err(|e| e.into_type_mismatch(self, item.location()))
        })
    }

    fn solve_consts(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            if let ItemKind::Const(con) = item.kind() {
                self.solve_const(con, item.id()).map(|_| ())
            } else {
                Ok(())
            }
        })
    }

    /// Number of `Constant` decls whose literal value has been reduced. Used by the hoist-uses /
    /// solve-consts fixpoint loop to detect "no progress this round -> stop."
    fn resolved_constants(&self) -> usize {
        self.decs
            .iter()
            .filter(|(_, d)| matches!(d.kind, DecKind::Constant(Some(_))))
            .count()
    }

    /// `let`/`const` annotation enforcement: unify the binding's ty with the annotation's.
    /// A unit annotation (`kg`, `usd/kWh`) is different from every other: it declares "a
    /// quantity in this dimension" -- a check the dims pass owns -- while its number may
    /// be an `int` (`5.5kg.to_int()` keeps the unit and is an `int`) as well as a `float`.
    /// So for unit annotations the rhs solves first and any numeric result stands;
    /// anything else gets the usual annotation mismatch.
    fn fulfill_annotation(
        &mut self,
        ty: &mut Ty,
        annotation: &Annotation,
        right: &Expr,
        at: Location,
    ) -> Result<()> {
        if Ty::is_unit_annotation(annotation, self) {
            right.fulfill_ty(ty.clone(), self)?;
            // `index` is the integer unit: an index that isn't an `int` can't index
            // anything, so a float rhs gets the ordinary int mismatch
            let integer_unit = matches!(annotation, Annotation::Ty(ident)
                if shared::units::lookup(&ident.lexeme).is_some_and(|(d, _)| d == shared::units::INDEX));
            return match ty.clone().normalized(self) {
                Ty::Int => Ok(()),
                _ if integer_unit => right.fulfill_ty(Ty::Int, self),
                // a numeric rhs stands (an int quantity is an `int` here); the dims
                // pass owns the dimension check itself
                Ty::Float => Ok(()),
                // anything else fails the annotation the ordinary way, at the rhs
                _ => right.fulfill_ty(Ty::Float, self),
            };
        }
        ty.fulfill_ty(&mut Ty::from_annotation(annotation.clone(), self)?, self)
            .map_err(|e| e.into_type_mismatch(self, at))
    }

    /// Solve a single `const` item's rhs and, if reducible, populate `DeclKind::Constant`'s
    /// payload with the literal. No-op for non-const items. Shared by the top-level fixpoint,
    /// impl associated consts, and block-scoped consts during body visits.
    pub(crate) fn solve_const(&mut self, con: &Const, id: NodeId) -> Result<Ty> {
        let Const {
            left,
            annotation,
            right,
        } = con;

        let vid = self.node_vid(right.id());
        let mut ty = Ty::Vid(vid);

        if let Some(annotation) = &annotation {
            self.fulfill_annotation(&mut ty, annotation, right, left.location())?;
        };

        // solve rhs first so inner idents land in `node_decs` before reduce_const_expr uses them
        right.fulfill_ty(ty.clone(), self)?;

        if annotation.is_none() && ty.clone().normalized(self) == Ty::Null {
            Err(BareNullBinding {
                src: self.src(right.location()),
                at: right.location().into(),
            })?;
        }

        if !self.is_const_expr(right) {
            // a call to a Mimas fn can't fold until `solve_bodies` records the
            // callee — defer to the retry pass instead of erroring early
            // (r[mimas.const.eval])
            if self.defer_call_consts && self.calls_unsolved_fn(right) {
                return Ok(ty);
            }
            Err(NonConstantValue {
                src: self.src(right.location()),
                at: right.location().into(),
                value: crate::errors::elide(right.to_string()),
            })?;
        }

        // a destructuring const's leaf vids unify with the rhs tuple's element types
        if left.as_ident().is_none() {
            let rhs_ty = ty.clone().normalized(self);
            self.bind_const_pat(left, rhs_ty)?;
        }

        if let Some(lit) = self.reduce_const_expr(right)? {
            if left.as_ident().is_some() {
                if let Some(dec) = self.node_decs.get(&id).copied()
                    && let DecKind::Constant(slot) = &mut self.decs[dec].kind
                {
                    *slot = Some(lit);
                }
            } else {
                self.fill_const_pat(left, &lit)?;
            }
        }

        Ok(ty)
    }

    /// The type half of a destructuring `const`: each leaf's vid unifies with the matching
    /// element of the rhs's tuple type. Only idents and tuples of them are irrefutable, same
    /// as `let` without an `else` -- anything else is an `InvalidPattern`.
    fn bind_const_pat(&mut self, pat: &Pat, ty: Ty) -> Result<()> {
        let ty = ty.normalized(self);
        match (pat.kind(), &ty) {
            (PatKind::Ident(_), _) => {
                let Some(&dec) = self.node_decs.get(&pat.id()) else {
                    return Ok(());
                };
                let mut leaf = Ty::Vid(self.decs[dec].vid);
                leaf.fulfill_ty(&mut ty.clone(), self)
                    .map_err(|e| e.into_type_mismatch(self, pat.location()))
            }
            (PatKind::Tuple(pats), Ty::Tuple(tys)) => match pats.len().cmp(&tys.len()) {
                std::cmp::Ordering::Less => Err(MissingTupleMembers {
                    src: self.src(pat.location()),
                    at: pat.location().into(),
                    ctor: None,
                })?,
                std::cmp::Ordering::Greater => Err(ExtraTupleMembers {
                    src: self.src(pat.location()),
                    at: pat.location().into(),
                    ctor: None,
                })?,
                std::cmp::Ordering::Equal => pats
                    .iter()
                    .zip(tys.iter())
                    .try_for_each(|(p, t)| self.bind_const_pat(p, t.clone())),
            },
            (PatKind::Poison(poison), _) => poison.escaped(),
            _ => Err(InvalidPattern {
                src: self.src(pat.location()),
                at: pat.location().into(),
                pattern: pat.to_string(),
                ty: ty.to_string(),
            })?,
        }
    }

    /// The value half of a destructuring `const`: distributes a reduced rhs literal into the
    /// leaf decs' `Constant` slots. Shape mismatches can't reach here -- `bind_const_pat`
    /// rejects them first.
    fn fill_const_pat(&mut self, pat: &Pat, lit: &Literal) -> Result<()> {
        match (pat.kind(), lit) {
            (PatKind::Ident(_), _) => {
                if let Some(&dec) = self.node_decs.get(&pat.id())
                    && let DecKind::Constant(slot) = &mut self.decs[dec].kind
                {
                    *slot = Some(lit.clone());
                }
                Ok(())
            }
            (PatKind::Tuple(pats), Literal::Tuple(exprs) | Literal::Array(exprs)) => {
                for (pat, expr) in pats.iter().zip(exprs) {
                    let Some(sub) = self.reduce_const_expr(expr)? else {
                        continue;
                    };
                    self.fill_const_pat(pat, &sub)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Brings a `use` item's names into scope. The path is resolved once per
    /// item (`resolved_uses`) and the result installed wherever this call finds
    /// scope: the phases each install into the file's import rib (module files
    /// get a fresh one per phase; a script's root rib just takes the same decs
    /// again), and a `use` inside a block binds in that block, scoped like a
    /// `let`. The import rib sits *below* the module rib in the stack, so the
    /// file's own declarations always win -- by lookup order, nothing to check.
    // r[impl mimas.use.forms]
    fn process_use(&mut self, item: &Item) -> Result<()> {
        let ItemKind::Use(us) = item.kind() else {
            return Ok(());
        };
        // `use "page";` is the host's business (it splices that script in first)
        if matches!(us, Use::Host(_)) {
            return Ok(());
        }
        let bindings = match self.resolve_use(us, item)? {
            ResolvedUse::Named(bindings) => bindings,
            ResolvedUse::Glob(adt) => self.visible_members(adt),
        };
        // r[impl mimas.use.shadow] -- the import rib sits beneath the module
        // rib; r[impl mimas.use.block] -- inside a block, the current rib.
        let rib = if self.ribs.current_kind() == RibKind::Script {
            self.ribs.import_mut()
        } else {
            self.ribs.current_mut()
        };
        for (name, dec) in bindings {
            rib.insert(Ident::synthetic(name), dec);
        }
        Ok(())
    }

    /// What `us` resolves to, computed once and kept on the item's id. Failures
    /// cache nothing -- the early best-effort pass can silently miss items a
    /// later phase hoists, and `hoist_uses` reports whatever never resolved.
    fn resolve_use(&mut self, us: &Use, item: &Item) -> Result<ResolvedUse> {
        if let Some(resolved) = self.resolved_uses.get(&item.id()) {
            return Ok(resolved.clone());
        }
        let resolved = self.compute_use(us, item)?;
        self.resolved_uses.insert(item.id(), resolved.clone());
        Ok(resolved)
    }

    /// Walks a `use`'s path to its target and returns the bindings it puts in
    /// scope, noting each segment's resolution for tooling as it goes.
    fn compute_use(&mut self, us: &Use, item: &Item) -> Result<ResolvedUse> {
        let walk_segments: Vec<&Ident> = match us {
            Use::Singular(path, item) => path.iter().chain(std::iter::once(item)).collect(),
            Use::Multi(path, _) | Use::All(path) => path.iter().collect(),
            Use::Host(_) => unreachable!("handled by process_use"),
        };

        let first = *walk_segments.first().unwrap();
        let mut target = (
            self.library
                .get(&first.lexeme)
                .cloned()
                .map(Ty::adt)
                .ok_or_else(|| NotFound {
                    src: self.src(first.location),
                    at: first.location.into(),
                    name: first.lexeme.clone(),
                })?,
            None,
        );
        self.note(first, target.0.clone(), None);

        for ident in walk_segments.iter().skip(1) {
            let Some(adt) = target.0.as_adt().copied() else {
                Err(InvalidUseTarget {
                    src: self.src(ident.location),
                    at: ident.location.into(),
                })?
            };
            let (ty, dec) = self.module_field(adt, ident)?;
            // r[impl mimas.module.privacy] -- naming a private member through
            // `use` fails here, on the `use`, not wherever it might be read.
            self.check_vis(dec, ident.location)?;
            self.note(ident, ty.clone(), Some(dec));
            target = (ty, Some(dec));
        }

        match us {
            Use::Host(_) => unreachable!("handled by process_use"),
            Use::Singular(_, item) => {
                // `use music` binds `music` -- the module itself, which
                // `music::` already names -- it does NOT import `pat`.
                // Remember it so an unbound `pat` can say why.
                if let Some(adt) = target.0.as_adt().copied()
                    && self.adts[adt].flags.contains(AdtFlags::IS_MODULE)
                {
                    self.bare_module_uses.insert(item.lexeme.clone(), adt);
                }
                let dec = match target.1 {
                    Some(dec) => dec,
                    // `use m;` -- the module's own name. Root modules live in
                    // `library`, not in a module's fields, so no field dec
                    // exists to reuse; mint the binding's dec here, once.
                    None => self.dec_id(
                        &Ident::synthetic(item.lexeme.clone()),
                        target.0.clone(),
                        DecKind::Item {
                            defaults: vec![],
                            wheres: vec![],
                        },
                        Vis::Public,
                    ),
                };
                Ok(ResolvedUse::Named(vec![(item.lexeme.clone(), dec)]))
            }
            Use::Multi(_, items) => {
                let Some(adt) = target.0.as_adt().copied() else {
                    Err(InvalidUseTarget {
                        src: self.src(item.location()),
                        at: item.location().into(),
                    })?
                };
                items
                    .iter()
                    .map(|i| {
                        let (ty, dec) = self.module_field(adt, i)?;
                        // r[impl mimas.module.privacy]
                        self.check_vis(dec, i.location)?;
                        self.note(i, ty.clone(), Some(dec));
                        Ok((i.lexeme.clone(), dec))
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(ResolvedUse::Named)
            }
            Use::All(_) => {
                let Some(adt) = target.0.as_adt().copied() else {
                    Err(InvalidUseTarget {
                        src: self.src(item.location()),
                        at: item.location().into(),
                    })?
                };
                if !self.adts[adt].flags.contains(AdtFlags::IS_MODULE) {
                    Err(InvalidUseTarget {
                        src: self.src(item.location()),
                        at: item.location().into(),
                    })?
                }
                Ok(ResolvedUse::Glob(adt))
            }
        }
    }

    /// The members `use m::*` puts in scope right now: every field and impl
    /// member of `adt` visible from the module being solved -- a `::*` never
    /// pulls a private name. Collected at each install so members hoisted in a
    /// later phase (a glob processed before the target's fns) still land.
    // r[impl mimas.use.forms]
    fn visible_members(&self, adt: AdtId) -> Vec<(String, DecId)> {
        let module = &self.adts[adt];
        module
            .as_struct()
            .fields
            .iter()
            .chain(module.impls.iter())
            .filter(|(_, field)| self.check_vis(field.dec, Location::default()).is_ok())
            .map(|(name, field)| (name.clone(), field.dec))
            .collect()
    }

    /// Pre-binds every `const` declared directly inside this block into the current rib so refs
    /// above the declaration line resolve, matching Rust's scope-hoisted-item semantics. Walks one
    /// level -- nested blocks hoist themselves when their own `Block::solve` runs.
    pub(crate) fn hoist_block_consts(&mut self, body: &[Stmt]) {
        for stmt in body {
            let StmtKind::Item(item) = stmt.kind() else {
                continue;
            };
            let ItemKind::Const(Const { left, right, .. }) = item.kind() else {
                continue;
            };
            let vid = self.node_vid(right.id());
            let vis = if item.public() {
                Vis::Public
            } else {
                Vis::Private
            };
            if let Some(ident) = left.as_ident() {
                let dec = self.dec_id(ident, Ty::Vid(vid), DecKind::Constant(None), vis);
                self.node_decs.insert(item.id(), dec);
                self.ribs.current_mut().insert(ident.clone(), dec);
                continue;
            }
            // destructuring const -- one Constant dec per bound name, keyed on the leaf pat
            for leaf in left.bound_leaves() {
                let ident = leaf.as_ident().expect("bound_leaves only yields idents");
                let leaf_ty = Ty::Vid(self.vid());
                let dec = self.dec_id(ident, leaf_ty, DecKind::Constant(None), vis);
                self.node_decs.insert(leaf.id(), dec);
                self.ribs.current_mut().insert(ident.clone(), dec);
            }
        }
    }

    // every adt allocation routes through here so `Display for Ty` can name it (the
    // shared-crate name table is what stands between users and `<adt>` placeholders)
    pub(crate) fn push_adt(&mut self, adt: Adt) -> AdtId {
        let name = adt.name.clone();
        let id = self.adts.push(adt);
        shared::name_adt(id.index(), &name);
        id
    }

    fn sync_module_adt(&mut self, adt: AdtId, module: &Rib) {
        let mut entries: Vec<_> = module
            .table
            .iter()
            .map(|(name, dec)| (name.clone(), *dec))
            .collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        // child modules (`module a::b;` registers `b` inside `a` at pre-alloc) live only in the
        // fields, not in any file's rib -- they must survive the rebuild
        let children: Vec<_> = self.adts[adt]
            .as_struct()
            .fields
            .iter()
            .filter(|(_, f)| {
                f.ty.as_adt()
                    .is_some_and(|id| self.adts[id].flags.contains(AdtFlags::IS_MODULE))
            })
            .map(|(n, f)| (n.clone(), f.clone()))
            .collect();
        self.adts[adt].as_struct_mut().fields.clear();
        for (name, field) in children {
            self.adts[adt].as_struct_mut().insert(name, field);
        }
        for (name, did) in entries {
            let dec = &self.decs[did];
            // store the bare Vid -- not its current substitution. readers always normalize, so
            // they pick up whatever the vid resolves to AT READ TIME. avoids snapshotting an
            // unresolved type when a later-phase or later-file body still has to bind it.
            let ty = Ty::Vid(dec.vid);
            self.adts[adt].as_struct_mut().insert(
                name,
                Field {
                    ty,
                    constant: dec.kind.is_constant(),
                    declaration_location: dec.location,
                    dec: did,
                },
            );
        }
    }

    /// One `use` path segment off module `adt`: the member's type (for `note`)
    /// and its dec (the binding and the visibility check).
    fn module_field(&self, adt: AdtId, ident: &Ident) -> Result<(Ty, DecId)> {
        let module = &self.adts[adt];
        if !module.flags.contains(AdtFlags::IS_MODULE) {
            Err(InvalidUseTarget {
                src: self.src(ident.location),
                at: ident.location.into(),
            })?
        }

        let field = module
            .as_struct()
            .fields
            .get(&ident.lexeme)
            .or_else(|| module.impls.get(&ident.lexeme))
            .ok_or_else(|| FieldNotFound {
                src: self.src(ident.location),
                at: ident.location.into(),
                field_name: ident.lexeme.clone(),
            })?;

        Ok((field.ty.clone(), field.dec))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn declare_native_fn(
        &mut self,
        name: String,
        params: Vec<Option<Ty>>,
        param_dims: Vec<Option<shared::units::Dim>>,
        return_ty: Option<Ty>,
        return_dim: Option<shared::units::Dim>,
        native_id: NativeId,
        validate: Option<api::LitValidator>,
        src: Option<api::NativeSrc>,
        doc: String,
        effects: Option<shared::Fx>,
        consumes: Vec<bool>,
        must_use: bool,
    ) -> DecId {
        let sig = NativeFnSig {
            params,
            param_dims,
            return_ty,
            return_dim,
            recv: None,
            effects,
            consumes,
            consumes_recv: false,
            must_use,
            ctor: None,
        };
        // first instantiation fills the dec's vid, later ident resolutions detect that this is
        // a native and re-instantiate so each call gets its own type vars.
        let initial = self
            .instantiate_native(&sig, None)
            .expect("free fn has no recv");
        let ident = Ident::synthetic(name);
        let dec_id = self.dec_id(
            &ident,
            initial,
            DecKind::Item {
                defaults: vec![],
                wheres: vec![],
            },
            Vis::Public,
        );
        self.ribs.module_mut().insert(ident, dec_id);
        self.dec_to_native.insert(
            dec_id,
            NativeBinding {
                id: native_id,
                sig,
                takes_self: false,
                mutates_recv: false,
                validate,
                src,
                doc,
            },
        );
        dec_id
    }

    /// Returns the `Ty::Fn` for a native callable, with every `Ty::Anon(N)` in the sig replaced
    /// by a fresh `Ty::Vid` (same `N` -> same vid, within this one call). If a receiver pattern
    /// is present and `recv_check` is provided, the actual receiver type is unified against the
    /// pattern -- failure means the method isn't applicable to this receiver.
    pub(crate) fn instantiate_native(
        &mut self,
        sig: &NativeFnSig,
        recv_check: Option<(&Ty, Location)>,
    ) -> Result<Ty> {
        let mut memo: FxHashMap<u32, Vid> = FxHashMap::default();
        let materialized: Vec<Ty> = sig
            .params
            .iter()
            .map(|p| self.materialize_slot(p, &mut memo))
            .collect();
        // trailing Option params are omittable: marking them defaulted lets the call-site
        // arity check pass, and the IR fills the empty slot with null (-> None natively)
        let mut optional_from = materialized.len();
        while optional_from > 0 && matches!(materialized[optional_from - 1], Ty::Option(_)) {
            optional_from -= 1;
        }
        let parameters: Vec<FnParam> = materialized
            .into_iter()
            .enumerate()
            .map(|(i, ty)| FnParam::new(None, ty, i >= optional_from))
            .collect();
        let return_ty = self.materialize_slot(&sig.return_ty, &mut memo);

        if let (Some(recv_pat), Some((actual, location))) = (sig.recv.as_ref(), recv_check) {
            let mut pat = self.substitute_anons(recv_pat, &mut memo);
            let mut actual = actual.clone();
            actual
                .fulfill_ty(&mut pat, self)
                .map_err(|e| e.into_type_mismatch(self, location))?;
        }

        Ok(Ty::Fn(FnHeader::new(parameters, return_ty, false)))
    }

    /// Produce a fresh `Ty::Fn` from a pact's stored header so unifying it against an impl can't
    /// pollute the shared signature. A pact header holds vids minted once at hoist (notably the
    /// `self` slot, left unbound); without freshening, the first `impl Pact for A` binds those
    /// vids globally and a later `impl Pact for B` would unify against the stale binding.
    ///
    /// The header's `Self` becomes `self_ty` -- the impl target when checking or grafting an impl,
    /// the bound when calling through one, or `Self` itself inside a default body.
    pub(crate) fn instantiate_pact_fn(&mut self, header: &FnHeader, self_ty: &Ty) -> Ty {
        let mut memo: FxHashMap<Vid, Vid> = FxHashMap::default();
        self.fresh_vids(&Ty::Fn(header.clone()), self_ty, &mut memo)
    }

    /// Normalize `t`, then replace any still-unbound vid with a fresh memoized one and any `Self`
    /// with `self_ty`. Bound vids resolve to their concrete types and stay put; only the open
    /// slots get fresh handles.
    fn fresh_vids(&mut self, t: &Ty, self_ty: &Ty, memo: &mut FxHashMap<Vid, Vid>) -> Ty {
        match t.clone().normalized(self) {
            Ty::Vid(v) => Ty::Vid(*memo.entry(v).or_insert_with(|| self.vid())),
            Ty::Skolem(_) => self_ty.clone(),
            Ty::Array(inner, len) => {
                Ty::Array(Box::new(self.fresh_vids(&inner, self_ty, memo)), len)
            }
            Ty::Dict(inner) => Ty::Dict(Box::new(self.fresh_vids(&inner, self_ty, memo))),
            Ty::Option(inner) => Ty::Option(Box::new(self.fresh_vids(&inner, self_ty, memo))),
            Ty::Result(inner) => Ty::Result(Box::new(self.fresh_vids(&inner, self_ty, memo))),
            Ty::Tuple(ts) => Ty::Tuple(
                ts.iter()
                    .map(|t| self.fresh_vids(t, self_ty, memo))
                    .collect(),
            ),
            Ty::Adt(aid, args) => Ty::Adt(
                aid,
                args.iter()
                    .map(|a| self.fresh_vids(a, self_ty, memo))
                    .collect(),
            ),
            Ty::Identity(aid, args) => Ty::Identity(
                aid,
                args.iter()
                    .map(|a| self.fresh_vids(a, self_ty, memo))
                    .collect(),
            ),
            Ty::Fn(h) => {
                let parameters = h
                    .parameters
                    .iter()
                    .map(|p| {
                        let ty = self.fresh_vids(&p.ty, self_ty, memo);
                        FnParam::new(p.name.clone(), ty, p.has_default)
                    })
                    .collect();
                let return_ty = self.fresh_vids(&h.return_ty, self_ty, memo);
                Ty::Fn(FnHeader::new(parameters, return_ty, h.is_method))
            }
            other => other,
        }
    }

    fn materialize_slot(&mut self, t: &Option<Ty>, memo: &mut FxHashMap<u32, Vid>) -> Ty {
        match t {
            Some(t) => self.substitute_anons(t, memo),
            None => Ty::Vid(self.vid()),
        }
    }

    pub(crate) fn substitute_anons(&mut self, t: &Ty, memo: &mut FxHashMap<u32, Vid>) -> Ty {
        match t {
            Ty::Anon(n) => Ty::Vid(*memo.entry(*n).or_insert_with(|| self.vid())),
            Ty::Array(v, len) => Ty::Array(Box::new(self.substitute_anons(v, memo)), *len),
            Ty::Dict(v) => Ty::Dict(Box::new(self.substitute_anons(v, memo))),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.substitute_anons(t, memo)).collect()),
            Ty::Option(inner) => Ty::Option(Box::new(self.substitute_anons(inner, memo))),
            Ty::Result(inner) => Ty::Result(Box::new(self.substitute_anons(inner, memo))),
            Ty::Fn(h) => Ty::Fn(FnHeader::new(
                h.parameters
                    .iter()
                    .map(|p| {
                        FnParam::new(
                            p.name.clone(),
                            self.substitute_anons(&p.ty, memo),
                            p.has_default,
                        )
                    })
                    .collect(),
                self.substitute_anons(&h.return_ty, memo),
                h.is_method,
            )),
            Ty::Adt(aid, args) => Ty::Adt(
                *aid,
                args.iter()
                    .map(|a| self.substitute_anons(a, memo))
                    .collect(),
            ),
            Ty::Identity(aid, args) => Ty::Identity(
                *aid,
                args.iter()
                    .map(|a| self.substitute_anons(a, memo))
                    .collect(),
            ),
            Ty::Unit
            | Ty::Never
            | Ty::Null
            | Ty::Bool
            | Ty::Int
            | Ty::Float
            | Ty::Str
            | Ty::Any
            | Ty::Vid(_)
            | Ty::Param(_)
            | Ty::Pacts(_)
            | Ty::Skolem(_) => t.clone(),
        }
    }

    /// Registers a native module (e.g. `std::fs`) so that `use` paths and `module::function` calls
    pub fn ensure_module_path(&mut self, path: &[String]) -> AdtId {
        let location = Location::default();
        let mut parent: Option<AdtId> = None;
        let mut leaf: Option<AdtId> = None;
        for segment in path.iter() {
            let existing: Option<AdtId> = match &parent {
                Some(parent_adt) => self.adts[parent_adt]
                    .as_struct()
                    .fields
                    .get(segment)
                    .and_then(|field| field.ty.as_adt().cloned()),
                None => self.library.get(segment).cloned(),
            };

            let adt_ref = if let Some(existing) = existing {
                existing
            } else {
                let mut adt = Adt::new_struct(format!("<module:{segment}>"));
                adt.flags |= AdtFlags::IS_MODULE;
                let id = self.push_adt(adt);
                if let Some(parent_adt) = &parent {
                    let seg_ident = Ident::synthetic(segment.clone());
                    let dec = self.dec_id(&seg_ident, Ty::adt(id), DecKind::Adt(id), Vis::Public);
                    self.adts[parent_adt].as_struct_mut().insert(
                        segment.clone(),
                        Field {
                            ty: Ty::adt(id),
                            constant: true,
                            declaration_location: location,
                            dec,
                        },
                    );
                } else {
                    self.library.insert(segment.clone(), id);
                }
                id
            };

            parent = Some(adt_ref);
            leaf = Some(adt_ref);
        }
        leaf.expect("module path must have at least one segment")
    }

    // adt ids arrive pre-allocated from the library's registry, so `self.adts.len()` must
    // equal `api_adt.adt_id.index()` -- library's allocator starts at the builtin count.
    fn install_user_adt(&mut self, api_adt: &ApiAdt) {
        debug_assert_eq!(self.adts.len(), api_adt.adt_id.index());

        let mut variants = FxIndexMap::default();
        for v in &api_adt.variants {
            let variant = self.build_variant(&v.fields);
            if let ApiVariantFields::Named(named) = &v.fields {
                for (name, _ty, dim) in named {
                    if let Some(dim) = dim {
                        self.field_dims
                            .insert((api_adt.name.clone(), name.clone()), *dim);
                    }
                }
            }
            let key = match api_adt.kind {
                ApiAdtKind::Enum => v.name.clone(),
                ApiAdtKind::Struct => Adt::STRUCT_VARIANT_NAME.to_string(),
            };
            variants.insert(key, variant);
        }
        let umbrella = Adt {
            name: api_adt.name.clone(),
            // host/native adts never declare type params; generic signatures live on the
            // host side as `Ty::Anon` slots instead
            type_params: vec![],
            variants,
            impls: FxHashMap::default(),
            native_overloads: FxHashMap::default(),
            overloads: FxHashMap::default(),
            flags: match api_adt.kind {
                ApiAdtKind::Enum => AdtFlags::IS_ENUM,
                ApiAdtKind::Struct => AdtFlags::empty(),
            } | if api_adt.op_overloads {
                AdtFlags::HAS_OPS
            } else {
                AdtFlags::empty()
            },
        };
        let pushed = self.push_adt(umbrella);
        debug_assert_eq!(pushed, api_adt.adt_id);

        if matches!(api_adt.kind, ApiAdtKind::Enum) {
            for v in &api_adt.variants {
                let qualified = format!("{}::{}", api_adt.name, v.name);
                let layout_adt = {
                    let variant = &self.adts[api_adt.adt_id].variants[&v.name];
                    Adt::new_variant_layout(qualified.clone(), variant)
                };
                let layout_pushed = self.push_adt(layout_adt);
                debug_assert_eq!(layout_pushed, v.layout_id);

                let variant_ident = Ident::synthetic(qualified);
                let dec = self.dec_id(
                    &variant_ident,
                    Ty::adt(api_adt.adt_id),
                    DecKind::Variant {
                        parent: api_adt.adt_id,
                        layout: v.layout_id,
                    },
                    Vis::Public,
                );
                let variant_ref = &mut self.adts[api_adt.adt_id].variants[&v.name];
                variant_ref.set_layout(v.layout_id);
                variant_ref.set_dec(dec);
            }
        }

        if api_adt.module.is_empty() {
            let ident = Ident::synthetic(api_adt.name.clone());
            let ty = Ty::adt(api_adt.adt_id);
            let dec_id = self.dec_id(&ident, ty, DecKind::Adt(api_adt.adt_id), Vis::Public);
            self.ribs.module_mut().insert(ident, dec_id);
            self.library.insert(api_adt.name.clone(), api_adt.adt_id);
        }
    }

    fn build_variant(&mut self, fields: &ApiVariantFields) -> Variant {
        match fields {
            ApiVariantFields::Unit => Variant::Tuple(TupleVariant {
                members: vec![],
                ..Default::default()
            }),
            ApiVariantFields::Tuple(members) => Variant::Tuple(TupleVariant {
                members: members.clone(),
                ..Default::default()
            }),
            ApiVariantFields::Named(named) => {
                let mut fields_map = FxIndexMap::default();
                for (name, ty, _dim) in named {
                    let ident = Ident::synthetic(name.clone());
                    let dec = self.dec_id(&ident, ty.clone(), DecKind::Local, Vis::Public);
                    fields_map.insert(
                        name.clone(),
                        Field::new(ty.clone(), Location::default(), dec),
                    );
                }
                Variant::Struct(StructVariant {
                    fields: Fields(fields_map),
                    ..Default::default()
                })
            }
        }
    }

    pub fn install_library<C>(&mut self, lib: &Library<C>) {
        for adt in lib.adts() {
            self.install_user_adt(adt);
        }
        for adt in lib.adts() {
            if adt.module.is_empty() {
                continue;
            }
            let leaf = self.ensure_module_path(&adt.module);
            let ident = Ident::synthetic(adt.name.clone());
            let dec = self.dec_id(
                &ident,
                Ty::adt(adt.adt_id),
                DecKind::Adt(adt.adt_id),
                Vis::Public,
            );
            self.adts[leaf].as_struct_mut().insert(
                adt.name.clone(),
                Field {
                    ty: Ty::adt(adt.adt_id),
                    constant: true,
                    declaration_location: Default::default(),
                    dec,
                },
            );
        }
        for (id, entry) in lib.natives() {
            match entry {
                // top-level native fn
                ApiEntry::Function(f) if f.module.is_empty() => {
                    self.declare_native_fn(
                        f.name.clone(),
                        f.parameters.clone(),
                        f.param_dims.clone(),
                        f.return_ty.clone(),
                        f.return_dim,
                        id,
                        f.validate,
                        f.src,
                        f.doc.clone(),
                        f.effects,
                        f.consumes.clone(),
                        f.must_use,
                    );
                }
                // module-nested native fn
                ApiEntry::Function(f) => {
                    let leaf = self.ensure_module_path(&f.module);
                    let sig = NativeFnSig {
                        params: f.parameters.clone(),
                        param_dims: f.param_dims.clone(),
                        return_ty: f.return_ty.clone(),
                        return_dim: f.return_dim,
                        recv: None,
                        effects: f.effects,
                        consumes: f.consumes.clone(),
                        consumes_recv: false,
                        must_use: f.must_use,
                        ctor: None,
                    };
                    let ident = Ident::synthetic(f.name.clone());
                    let ty = self.instantiate_native(&sig, None).expect("no recv");
                    let dec_id = self.dec_id(
                        &ident,
                        ty.clone(),
                        DecKind::Item {
                            defaults: vec![],
                            wheres: vec![],
                        },
                        Vis::Public,
                    );
                    self.dec_to_native.insert(
                        dec_id,
                        NativeBinding {
                            id,
                            sig,
                            takes_self: false,
                            mutates_recv: false,
                            validate: f.validate,
                            src: f.src,
                            doc: f.doc.clone(),
                        },
                    );
                    self.adts[leaf].as_struct_mut().insert(
                        f.name.clone(),
                        Field {
                            ty,
                            constant: true,
                            declaration_location: Default::default(),
                            dec: dec_id,
                        },
                    );
                }
                // native method or associated fn on a built-in
                ApiEntry::Method(m) => {
                    let adt = match &m.recv_ty {
                        Ty::Adt(id, _) => *id,
                        other => self.builtin_adt(other).unwrap_or_else(|| {
                            panic!("native method receiver must be a builtin Ty or registered adt; got {other:?}")
                        }),
                    };
                    // recv pattern carries the full receiver shape (e.g. `Ty::Array(Anon(0))`)
                    // so `instantiate_native` can unify it against the actual receiver per call.
                    // primitives like `Ty::Int` have nothing to unify, but it's harmless to pass.
                    let sig = NativeFnSig {
                        params: m.parameters.clone(),
                        param_dims: m.param_dims.clone(),
                        return_ty: m.return_ty.clone(),
                        return_dim: m.return_dim,
                        recv: Some(m.recv_ty.clone()),
                        effects: m.effects,
                        consumes: m.consumes.clone(),
                        consumes_recv: m.consumes_recv,
                        must_use: m.must_use,
                        ctor: m.ctor.clone(),
                    };
                    let ident = Ident::synthetic(m.name.clone());
                    let ty = self.instantiate_native(&sig, None).expect("no recv check");
                    let dec_id = self.dec_id(
                        &ident,
                        ty.clone(),
                        DecKind::Item {
                            defaults: vec![],
                            wheres: vec![],
                        },
                        Vis::Public,
                    );
                    self.dec_to_native.insert(
                        dec_id,
                        NativeBinding {
                            id,
                            sig,
                            takes_self: m.takes_self,
                            mutates_recv: m.mutates_recv,
                            validate: m.validate,
                            src: m.src,
                            doc: m.doc.clone(),
                        },
                    );
                    let field = Field {
                        ty,
                        constant: true,
                        declaration_location: Default::default(),
                        dec: dec_id,
                    };
                    // first registration for this name lands in `impls`; subsequent same-name
                    // natives are overloads (e.g. `max_int` + `max_float` both as `max`) and
                    // get dispatched by recv-pattern unification at the call site.
                    if self.adts[adt].impls.contains_key(&m.name) {
                        self.adts[adt]
                            .native_overloads
                            .entry(m.name.clone())
                            .or_default()
                            .push(field);
                    } else {
                        self.adts[adt].impls.insert(m.name.clone(), field);
                    }
                }
                ApiEntry::Constant(c) => {
                    let lit = match &c.value {
                        RawLiteral::Null => Literal::Null,
                        RawLiteral::Bool(true) => Literal::True,
                        RawLiteral::Bool(false) => Literal::False,
                        RawLiteral::Int(i) => Literal::Int(*i),
                        RawLiteral::Float(f) => Literal::Float(*f),
                        RawLiteral::Str(s) => Literal::String(s.clone()),
                    };
                    let ident = Ident::synthetic(c.name.clone());
                    let dec_id = self.dec_id(
                        &ident,
                        c.ty.clone(),
                        DecKind::Constant(Some(lit)),
                        Vis::Public,
                    );
                    if let Some(dim) = c.dim {
                        self.const_dims.push((dec_id, dim));
                    }
                    // an assoc constant (`Player::MAX_HEALTH`) lands in the adt's impls,
                    // same slot a lang `impl` const hoists into
                    if let Some(recv) = &c.recv_ty {
                        let adt = match recv {
                            Ty::Adt(id, _) => *id,
                            other => self.builtin_adt(other).unwrap_or_else(|| {
                                panic!("assoc constant receiver must be a builtin Ty or registered adt; got {other:?}")
                            }),
                        };
                        self.adts[adt].impls.insert(
                            c.name.clone(),
                            Field {
                                ty: c.ty.clone(),
                                constant: true,
                                declaration_location: Default::default(),
                                dec: dec_id,
                            },
                        );
                    // empty module = prelude: bind the name directly, same as a top-level
                    // native fn (ensure_module_path has no leaf for an empty path)
                    } else if c.module.is_empty() {
                        self.ribs.module_mut().insert(ident, dec_id);
                    } else {
                        let leaf = self.ensure_module_path(&c.module);
                        self.adts[leaf].as_struct_mut().insert(
                            c.name.clone(),
                            Field {
                                ty: c.ty.clone(),
                                constant: true,
                                declaration_location: Default::default(),
                                dec: dec_id,
                            },
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn register_sub(
        &mut self,
        vid: Vid,
        ty: Ty,
    ) -> std::result::Result<(), UnificationError> {
        // normalize first: a BOUND vid left raw would hit unify's vid arm below, blindly rebind,
        // and re-unify its old binding with the found/expected roles reversed (the
        // cross-module-const inverted-mismatch bug). normalized, the conflict unify below compares
        // concrete types with the caller's roles intact.
        let mut ty = ty.normalized(&*self);

        // trivial self-substitution `V := Vid(V)` is a no-op -- keep any existing sub intact.
        // shows up when an expression's solve returns its own expr_vid (e.g. `d["x"]` whose value
        // type is still an unresolved vid), and Query then folds that result back through
        // register_sub against the same vid.
        if let Ty::Vid(v) = &ty
            && *v == vid
        {
            return Ok(());
        }

        let ty = if let Some(mut previous_ty) = self.subs[vid].take() {
            if let Err(e) =
                Unification::unify(&mut ty, &mut previous_ty, self).and_then(|v| v.commit(self))
            {
                self.subs[vid] = Some(previous_ty);
                return Err(e);
            }
            // `ty` and `previous_ty` now denote the same type under the substitutions just
            // committed; store the resolved form so the binding is the most concrete view rather
            // than whichever one happened to be less inferred.
            previous_ty.normalized(&*self)
        } else {
            ty
        };

        let ty = if let Some(impl_target) = self.impl_target() {
            ty.filter_adt(impl_target)
        } else {
            ty.clone()
        };

        if ty.occurs(vid, self) {
            return Err(UnificationError::recursive(vid, &ty));
        }

        #[cfg(feature = "logging")]
        println!("{}", crate::utils::Printer::substitution(&vid, &ty));
        self.subs[vid] = Some(ty);
        Ok(())
    }
}

// some id utils
impl Solver {
    pub(crate) fn vid(&mut self) -> Vid {
        let vid = self.vids.push(());
        self.subs.push(None);
        vid
    }

    pub(crate) fn node_vid(&mut self, node_id: NodeId) -> Vid {
        if let Some(vid) = self.node_to_vid.get(&node_id) {
            return *vid;
        }
        let vid = self.vid();
        self.node_to_vid.insert(node_id, vid);
        vid
    }

    /// The type solving recorded for `node_id`, normalized — `None` when the
    /// node never got one (unreached, or an error stopped the solve first).
    /// Read-only counterpart of `Resolutions::node_tys` for callers that
    /// already hold the `Solver`.
    pub fn node_ty(&self, node_id: NodeId) -> Option<Ty> {
        self.node_to_vid
            .get(&node_id)
            .map(|vid| Ty::Vid(*vid).normalized(self))
    }

    /// This exists as a shim to provide information to the IR that it would otherwise be able to
    /// find. Take the given example:
    ///
    /// ```
    /// enum Foo {
    ///     Bar,
    /// }
    /// let a: Foo = Foo::Bar;
    /// ```
    ///
    /// The type of a is `Foo`, but to the IR that type is meaningless, as there's no way to
    /// construct `Foo` by itself. Instead, we mark specifically this expression as an Adt built
    /// directly off of the variant so that when the IR inspects this expression, it will know what
    /// it actually should build.
    ///
    /// Bit weird but works.
    pub(crate) fn shadow_expr_ty(
        &mut self,
        node_id: NodeId,
        ty: Ty,
        location: Location,
    ) -> Result<()> {
        let vid = self.node_vid(node_id);
        self.register_sub(vid, ty)
            .map_err(|e| e.into_type_mismatch(self, location))?;
        self.touch_node(node_id);
        Ok(())
    }

    pub(crate) fn sub(&self, vid: Vid) -> Option<&Ty> {
        self.subs[vid].as_ref()
    }

    pub(crate) fn dec_id(&mut self, ident: &Ident, ty: Ty, kind: DecKind, vis: Vis) -> DecId {
        // the binding's type is just a handle into `subs`. if we were handed a bare vid, that's
        // already the handle; otherwise mint one and point it at the type.
        let vid = match ty {
            Ty::Vid(v) => v,
            ty => {
                let v = self.vid();
                self.subs[v] = Some(ty);
                v
            }
        };
        let module = self.ribs.current_module();
        let dec = self.decs.push(Dec {
            name: ident.lexeme.clone(),
            location: ident.location,
            kind,
            vid,
            vis,
            module,
        });
        self.note(ident, Ty::Vid(vid), Some(dec));
        dec
    }

    /// `kWh` where nothing binds it: the quantity `1kWh`, as a synthetic `Constant` dec —
    /// codegen folds it like `PI`, `reduce_const_expr` proves it literal, and the dims
    /// pass reads its dimension from `unit_consts`. Deduped by name; an `index` unit is
    /// an `int`, everything else a `float`.
    fn unit_const(
        &mut self,
        ident: &Ident,
        dim: shared::units::Dim,
        scale: f64,
    ) -> (Ty, Option<DecId>) {
        if let Some(&(dec, _)) = self.unit_consts.get(&ident.lexeme) {
            let ty = Ty::Vid(self.decs[dec].vid);
            self.note(ident, ty.clone(), Some(dec));
            return (ty, Some(dec));
        }
        let (ty, lit) = if dim == shared::units::INDEX {
            (Ty::Int, Literal::Int(1))
        } else {
            (Ty::Float, Literal::Float(scale))
        };
        let dec = self.dec_id(ident, ty.clone(), DecKind::Constant(Some(lit)), Vis::Public);
        self.unit_consts.insert(ident.lexeme.clone(), (dec, dim));
        (ty, Some(dec))
    }
}

// generic type params
impl Solver {
    /// The `ParamId` for one declared param ident (`T` in `fn map<T>`). Derived from the
    /// ident's `NodeId` so hoisting and body-solving the same decl always agree on it; also
    /// registers the display name so `Ty::Param` renders as `T`, not `<param>`.
    pub(crate) fn param_id_of(ident: &Ident) -> ParamId {
        let pid = ParamId::from(ident.id.index() as u32);
        shared::name_param(pid.index(), &ident.lexeme);
        pid
    }

    /// Push a type-parameter scope parsed off a generic item (`fn map<T, U>`). Each `ParamId`
    /// is derived from the param ident's `NodeId`, so hoisting and body-solving the same
    /// decl resolve `T` to the *same* id -- that's what lets a signature written once carry
    /// rigid params through both phases.
    pub(crate) fn push_type_params(&mut self, params: &[TypeParam]) {
        let scope: FxHashMap<String, ParamId> =
            self.declare_type_params(params).into_iter().collect();
        self.type_params.push(scope);
    }

    /// The `ParamId` of each declared param, with its bounds registered in `param_bounds`.
    /// Hoisting and body-solving both pass through here for one decl; the ids agree (see
    /// `param_id_of`) and a bound registers once.
    pub(crate) fn declare_type_params(&mut self, params: &[TypeParam]) -> Vec<(String, ParamId)> {
        params
            .iter()
            .map(|p| {
                let pid = Self::param_id_of(&p.name);
                for bound in &p.bounds {
                    self.bind_param(pid, bound);
                }
                (p.lexeme.clone(), pid)
            })
            .collect()
    }

    // r[impl mimas.generic.bound]
    /// `T: Draw` -- hold `pid` to the pact `bound` names. A name that doesn't resolve yet
    /// (the pact is hoisted after the types) waits in `pending_bounds`.
    fn bind_param(&mut self, pid: ParamId, bound: &Ident) {
        let pact = self
            .resolve_name(bound, bound.location)
            .ok()
            .and_then(|ty| ty.as_single_pact());
        match pact {
            Some(pact) => {
                let bounds = self.param_bounds.entry(pid).or_default();
                if !bounds.contains(&pact) {
                    bounds.push(pact);
                }
            }
            None => {
                let waiting = self
                    .pending_bounds
                    .iter()
                    .any(|(p, i)| *p == pid && i.lexeme == bound.lexeme);
                if !waiting {
                    self.pending_bounds.push((pid, bound.clone()));
                }
            }
        }
    }

    /// Bounds that named a pact before it existed: resolve them now, or say what the name
    /// is instead. A phase after `hoist_callables`, when every pact is declared.
    fn resolve_pending_bounds(&mut self, _ast: &Ast) -> Result<()> {
        for (pid, bound) in std::mem::take(&mut self.pending_bounds) {
            let ty = self.resolve_name(&bound, bound.location)?;
            let Some(pact) = ty.as_single_pact() else {
                return Err(NotAPact {
                    src: self.src(bound.location),
                    at: bound.location.into(),
                    ty: ty.to_string(),
                }
                .into());
            };
            let bounds = self.param_bounds.entry(pid).or_default();
            if !bounds.contains(&pact) {
                bounds.push(pact);
            }
        }
        Ok(())
    }

    /// A parameter default that named a not-yet-solved `const` at `hoist_callables` is
    /// reduced now that the const fixpoint has run; one that still has no literal is the
    /// `NonConstDefault` it would have been at hoist.
    // r[impl mimas.fn.default-const]
    fn resolve_pending_defaults(&mut self, _ast: &Ast) -> Result<()> {
        for (dec, index, expr) in std::mem::take(&mut self.pending_defaults) {
            let Some(lit) = self.default_literal(&expr)? else {
                return Err(crate::errors::NonConstDefault {
                    src: self.src(expr.location()),
                    at: expr.location().into(),
                }
                .into());
            };
            if let DecKind::Item { defaults, .. } = &mut self.decs[dec].kind {
                defaults[index] = Some(lit);
            }
        }
        Ok(())
    }

    /// A parameter default's literal: `reduce_const_expr`, then the shape check. A
    /// default lives in the callee's header as a bytecode `Constant`, which has no
    /// struct or dict form, so a literal that would allocate is refused here with a
    /// pointer at the body instead of panicking in the IR.
    pub(crate) fn default_literal(&self, expr: &Expr) -> Result<Option<Literal>> {
        fn allocates(lit: &Literal) -> bool {
            match lit {
                Literal::Struct(_) | Literal::Dictionary(_) => true,
                Literal::Array(items) | Literal::Tuple(items) => items.iter().any(|e| {
                    matches!(e.kind(), ExprKind::Literal(l) if allocates(l))
                }),
                _ => false,
            }
        }
        let lit = self.reduce_const_expr(expr)?;
        if lit.as_ref().is_some_and(allocates) {
            return Err(crate::errors::NonScalarDefault {
                src: self.src(expr.location()),
                at: expr.location().into(),
            }
            .into());
        }
        Ok(lit)
    }

    /// The pacts a receiver of `ty` dispatches through: a bound annotation
    /// (`d: Draw + Named`), `Self` in a default body, or a declared param's bounds
    /// (`x: T` under `<T: Draw>`). `None` for a concrete type -- or an unbounded param,
    /// which has nothing to dispatch through.
    pub(crate) fn pacts_of(&self, ty: &Ty) -> Option<Vec<PactId>> {
        match ty {
            Ty::Param(pid) => self
                .param_bounds
                .get(pid)
                .filter(|b| !b.is_empty())
                .cloned(),
            other => other.as_pacts(),
        }
    }

    /// A fresh vid standing in for the bounded param `pid` at the use `obligation_at`
    /// names: whatever it settles to owes the param's pacts.
    fn owe_bounds(&mut self, pid: ParamId, ty: &Ty) {
        let Some(at) = self.obligation_at else {
            return;
        };
        for pact in self.param_bounds.get(&pid).cloned().unwrap_or_default() {
            self.bound_obligations.push((ty.clone(), pact, at));
        }
    }

    // r[impl mimas.generic.bound]
    /// Every bounded param that got instantiated: does the type it settled to implement
    /// the pact? Runs once bodies are solved, so each vid is as pinned as it will get --
    /// one still open is held to nothing.
    fn check_bounds(&mut self) -> Result<()> {
        for (ty, pact, at) in std::mem::take(&mut self.bound_obligations) {
            let ty = ty.normalized(self);
            let ok = match &ty {
                Ty::Vid(_) | Ty::Never => true,
                Ty::Adt(aid, _) | Ty::Identity(aid, _) => self.pact_impls.contains(&(pact, *aid)),
                Ty::Pacts(pids) => pids.contains(&pact),
                Ty::Skolem(p) => *p == pact,
                Ty::Param(p) => self.param_bounds.get(p).is_some_and(|b| b.contains(&pact)),
                other => self
                    .builtin_adt(other)
                    .is_some_and(|aid| self.pact_impls.contains(&(pact, aid))),
            };
            if !ok {
                return Err(BoundNotSatisfied {
                    src: self.src(at),
                    at: at.into(),
                    ty: ty.to_string(),
                    pact: self.pacts[pact].name.clone(),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Re-enter the parameter scope of an already-declared adt -- `impl List` sees `List<T>`'s
    /// `T` with the same `ParamId` the enum's stored member types use.
    pub(crate) fn push_adt_params(&mut self, adt: AdtId) {
        let scope: FxHashMap<String, ParamId> =
            self.adts[adt].type_params.iter().cloned().collect();
        self.type_params.push(scope);
    }

    pub(crate) fn pop_type_params(&mut self) {
        self.type_params.pop();
    }

    /// Is `ident` one of the in-scope declared type params (`T` in `fn map<T>`)? Innermost
    /// scope wins, so a fn param shadows an impl's adt param of the same name.
    pub(crate) fn type_param(&self, ident: &Ident) -> Option<ParamId> {
        self.type_params
            .iter()
            .rev()
            .find_map(|scope| scope.get(&ident.lexeme))
            .copied()
    }

    /// `Ty::Adt(aid, args)` with a fresh vid for every declared param -- what a ctor use
    /// (`L::C`, `Pair { .. }`) starts from before unification pins the args down.
    pub(crate) fn instantiated_adt(&mut self, aid: AdtId) -> Ty {
        let params: Vec<ParamId> = self.adts[aid].type_params.iter().map(|(_, p)| *p).collect();
        let args = params
            .into_iter()
            .map(|pid| {
                let ty = Ty::Vid(self.vid());
                self.owe_bounds(pid, &ty);
                ty
            })
            .collect();
        Ty::Adt(aid, args)
    }

    /// [`Self::instantiated_adt`] for a use written at `at` -- a bounded param's
    /// obligation is recorded against it.
    pub(crate) fn instantiated_adt_at(&mut self, aid: AdtId, at: Location) -> Ty {
        let prev = self.obligation_at.replace(at);
        let ty = self.instantiated_adt(aid);
        self.obligation_at = prev;
        ty
    }

    /// A type stored inside a generic definition, read back at a use site: every `Ty::Param`
    /// swapped for a fresh vid (consistent within this one call -- `T` in two places still maps
    /// to the same var). This is the HM instantiation step: `map<T, U>` used once gives
    /// `fn([?t], (?t) -> ?u) -> [?u]` which call-site unification specializes freely.
    pub(crate) fn instantiate_params(&mut self, t: &Ty) -> Ty {
        if !t.contains_params() {
            return t.clone();
        }
        let mut map = FxHashMap::default();
        self.substitute_params(t, &mut map)
    }

    /// `stored` is a member/variant/impl signature declared inside (possibly generic) adt
    /// `aid`, being read through a `Ty::Adt(aid, args)`/`Identity(aid, args)`. Rebinds each
    /// declared param to the corresponding arg; params with no matching arg get fresh vids,
    /// so `specialize` on a non-generic adt is a no-op.
    pub(crate) fn specialize(&mut self, stored: &Ty, aid: AdtId, args: &[Ty]) -> Ty {
        if !stored.contains_params() {
            return stored.clone();
        }
        let mut map: FxHashMap<ParamId, Ty> = self.adts[aid]
            .type_params
            .iter()
            .map(|(_, pid)| *pid)
            .zip(args.iter().cloned())
            .collect();
        self.substitute_params(stored, &mut map)
    }

    /// Walk `t` swapping each `Ty::Param` for `map[pid]`; missing entries mint a fresh vid
    /// lazily (and memoize, so one param always gets one var per call). Recurses through every
    /// composite, including adt/identity type args.
    pub(crate) fn substitute_params(&mut self, t: &Ty, map: &mut FxHashMap<ParamId, Ty>) -> Ty {
        match t {
            Ty::Param(pid) => match map.get(pid) {
                Some(ty) => ty.clone(),
                None => {
                    let ty = Ty::Vid(self.vid());
                    map.insert(*pid, ty.clone());
                    self.owe_bounds(*pid, &ty);
                    ty
                }
            },
            Ty::Array(v, len) => Ty::Array(Box::new(self.substitute_params(v, map)), *len),
            Ty::Dict(v) => Ty::Dict(Box::new(self.substitute_params(v, map))),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.substitute_params(t, map)).collect()),
            Ty::Option(inner) => Ty::Option(Box::new(self.substitute_params(inner, map))),
            Ty::Result(inner) => Ty::Result(Box::new(self.substitute_params(inner, map))),
            Ty::Adt(aid, args) => Ty::Adt(
                *aid,
                args.iter()
                    .map(|a| self.substitute_params(a, map))
                    .collect(),
            ),
            Ty::Identity(aid, args) => Ty::Identity(
                *aid,
                args.iter()
                    .map(|a| self.substitute_params(a, map))
                    .collect(),
            ),
            Ty::Fn(h) => {
                let mut header = FnHeader::new(
                    h.parameters
                        .iter()
                        .map(|p| {
                            FnParam::new(
                                p.name.clone(),
                                self.substitute_params(&p.ty, map),
                                p.has_default,
                            )
                        })
                        .collect(),
                    self.substitute_params(&h.return_ty, map),
                    h.is_method,
                );
                header.is_ctor = h.is_ctor;
                Ty::Fn(header)
            }
            Ty::Unit
            | Ty::Never
            | Ty::Null
            | Ty::Bool
            | Ty::Int
            | Ty::Float
            | Ty::Str
            | Ty::Any
            | Ty::Vid(_)
            | Ty::Anon(_)
            | Ty::Pacts(_)
            | Ty::Skolem(_) => t.clone(),
        }
    }
}

impl Solver {
    pub(crate) fn solve_pat(&mut self, pat: &Pat, ty: Ty) -> Result<()> {
        // normalize so an unresolved Vid reveals its concrete shape (e.g. tuple destruction where
        // the rhs's type was unified only after we got here).
        let ty = ty.normalized(self);
        match (pat.kind(), ty) {
            (PatKind::Ident(ident), ty) => self.declare(ident, pat.id(), ty).map(|_| ()),
            (PatKind::Tuple(idents), Ty::Tuple(tys)) => match idents.len().cmp(&tys.len()) {
                std::cmp::Ordering::Less => Err(MissingTupleMembers {
                    src: self.src(pat.location()),
                    at: pat.location().into(),
                    ctor: None,
                })?,
                std::cmp::Ordering::Greater => Err(ExtraTupleMembers {
                    src: self.src(pat.location()),
                    at: pat.location().into(),
                    ctor: None,
                })?,
                std::cmp::Ordering::Equal => idents
                    .iter()
                    .zip(tys.iter())
                    .try_for_each(|(pat, ty)| self.solve_pat(pat, ty.clone())),
            },
            (a, b) => Err(InvalidPattern {
                src: self.src(pat.location()),
                at: pat.location().into(),
                pattern: a.to_string(),
                ty: b.to_string(),
            })?,
        }
    }

    /// The set of names a pattern binds. Every alternative of an or-pattern must bind this same
    /// set (same as Rust's E0408). That, plus the shared dec the alternatives unify through, is
    /// what rejects `Foo::Bar(n) | Foo::Buzz(n)` when the payloads differ in type
    fn bound_names(pat: &Pat) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut stack = vec![pat];
        while let Some(pat) = stack.pop() {
            match pat.kind() {
                PatKind::Ident(ident) if ident.lexeme != "_" => {
                    out.insert(ident.lexeme.clone());
                }
                PatKind::Ident(_) => {}
                PatKind::Tuple(pats) | PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
                    stack.extend(pats);
                }
                PatKind::Struct(_, fields) => stack.extend(fields.values()),
                PatKind::NullBind(inner) => stack.push(inner),
                PatKind::Variant(_) | PatKind::Literal(_) => {}
                PatKind::Poison(poison) => poison.escaped(),
            }
        }
        out
    }

    /// Solves a match-arm pattern against the scrutinee type. Binds any names the pattern
    /// introduces into the current scope. Unifies literal/variant patterns with the scrutinee
    /// type.
    ///
    /// `reuse` is set while solving the non-first alternatives of an or-pattern: rather than mint a
    /// fresh dec, an ident rebinds to the one the first alternative already declared (looked up in
    /// this arm's own rib), so all alternatives share a single binding and its type. That sharing
    /// is what makes the payload types of `Foo::Bar(n) | Foo::Fizz(n)` unify.
    pub(crate) fn solve_match_pat(&mut self, pat: &Pat, ty: Ty, reuse: bool) -> Result<()> {
        // `match` on a dynamic payload (`absolve`'s `any` arm): the pattern is
        // a runtime probe, not a static claim — literal/variant arms skip the
        // scrutinee unify but still resolve the variant and bind fields with
        // member types (sound: the arm only runs when the probe passed).
        let dynamic = matches!(ty.clone().normalized(self), Ty::Any);
        let pat_src = self.src(pat.location());
        let bad = |ty: &Ty| InvalidPattern {
            src: pat_src.clone(),
            at: pat.location().into(),
            pattern: pat.to_string(),
            ty: ty.to_string(),
        };

        let unify = |solver: &mut Solver, mut a: Ty, mut b: Ty, loc: Location| -> Result<()> {
            a.fulfill_ty(&mut b, solver)
                .map_err(|e| e.into_type_mismatch(solver, loc))
        };

        match pat.kind() {
            PatKind::Ident(ident) if reuse && ident.lexeme != "_" => {
                // the or-pattern already checked every alternative binds the same names, so the
                // first alternative's dec is guaranteed present in this arm's rib.
                let dec = self
                    .ribs
                    .resolve_current_only(ident)
                    .expect("first or-alternative declared this binding");
                self.node_decs.insert(pat.id(), dec);
                unify(self, Ty::Vid(self.decs[dec].vid), ty, pat.location())
            }
            PatKind::Ident(ident) => self.declare(ident, pat.id(), ty).map(|_| ()),
            PatKind::Poison(poison) => poison.escaped(),
            PatKind::Literal(lit) if dynamic => {
                // a literal on `any` is a runtime eq probe — `42`, `"x"`,
                // `null` all compare fine against whatever was raised
                let _ = lit;
                Ok(())
            }
            PatKind::Literal(lit) => unify(
                self,
                ty,
                match lit {
                    parse::Literal::True | parse::Literal::False => Ty::Bool,
                    parse::Literal::Null => Ty::Null,
                    parse::Literal::Unit => Ty::Unit,
                    parse::Literal::String(_) => Ty::Str,
                    parse::Literal::Int(_) | parse::Literal::Hex(_) => Ty::Int,
                    parse::Literal::Float(_) => Ty::Float,
                    _ => Ty::Vid(crate::components::Vid::UNKNOWN),
                },
                pat.location(),
            ),
            PatKind::Tuple(pats) => {
                let normalized = ty.normalized(self);
                let Ty::Tuple(tys) = normalized else {
                    Err(bad(&normalized))?
                };
                if pats.len() != tys.len() {
                    Err(bad(&Ty::Tuple(tys.clone())))?
                }
                for (p, t) in pats.iter().zip(tys) {
                    self.solve_match_pat(p, t, reuse)?;
                }
                Ok(())
            }
            PatKind::NullBind(inner_pat) => {
                let normalized = ty.normalized(self);
                let (inner, probe) = match &normalized {
                    Ty::Option(inner) | Ty::Result(inner) => {
                        (inner.as_ref().clone(), normalized.clone())
                    }
                    // a non-null probe on `any`: `x?` binds whatever isn't
                    // `null`, still `any` inside the arm
                    Ty::Any => (Ty::Any, Ty::Option(Box::new(Ty::Any))),
                    _ => Err(bad(&normalized))?,
                };
                // the IR reads this node's ty to pick the success test (null vs raised)
                self.shadow_expr_ty(pat.id(), probe, pat.location())?;
                self.solve_match_pat(inner_pat, inner, reuse)
            }
            PatKind::Or(alts) => {
                let Some((first, rest)) = alts.split_first() else {
                    return Ok(());
                };
                // the first alternative declares the canonical bindings (or, if this or-pattern is
                // itself nested in a reuse context, rebinds the enclosing ones); the rest reuse.
                self.solve_match_pat(first, ty.clone(), reuse)?;
                let expected = Self::bound_names(first);
                for alt in rest {
                    if let Some(name) = expected
                        .symmetric_difference(&Self::bound_names(alt))
                        .next()
                    {
                        Err(crate::errors::InconsistentOrBindings {
                            src: self.src(pat.location()),
                            at: pat.location().into(),
                            name: name.clone(),
                        })?
                    }
                    self.solve_match_pat(alt, ty.clone(), true)?;
                }
                Ok(())
            }
            PatKind::Variant(_) | PatKind::TupleVariant(_, _) | PatKind::Struct(_, _) => {
                let path = match pat.kind() {
                    PatKind::Variant(p) | PatKind::TupleVariant(p, _) | PatKind::Struct(p, _) => p,
                    _ => unreachable!(),
                };
                let (adt, variant_name) = match path.kind() {
                    ExprKind::Ident(ident) => {
                        let ty = self.resolve_name(ident, ident.location())?;
                        let Some(adt) = ty.as_adt().copied() else {
                            Err(bad(&ty))?
                        };
                        (adt, None)
                    }
                    ExprKind::Access(Access::DoubleColon { left, right }) => {
                        let adt = match self.resolve_path_head(left)? {
                            Ty::Adt(adt, _) | Ty::Identity(adt, _) => adt,
                            lty => Err(bad(&lty))?,
                        };
                        // module::struct -- treat as the inner struct adt
                        if self.adts[adt].flags.contains(AdtFlags::IS_MODULE) {
                            let field = self.adts[adt]
                                .as_struct()
                                .fields
                                .get(&right.lexeme)
                                .cloned()
                                .ok_or_else(|| FieldNotFound {
                                    src: self.src(right.location),
                                    at: right.location.into(),
                                    field_name: right.lexeme.clone(),
                                })?;
                            let Some(inner) = field.ty.as_adt().copied() else {
                                Err(bad(&field.ty))?
                            };
                            (inner, None)
                        } else {
                            // enum variant
                            if !self.adts[adt].variants.contains_key(&right.lexeme) {
                                Err(crate::errors::VariantNotFound {
                                    src: self.src(right.location),
                                    at: right.location.into(),
                                    name: right.lexeme.clone(),
                                })?
                            }
                            let dec = self.adts[adt].variants[&right.lexeme].dec();
                            self.note(right, Ty::adt(adt), dec);
                            (adt, Some(right.lexeme.clone()))
                        }
                    }
                    _ => Err(InvalidPattern {
                        src: self.src(path.location()),
                        at: path.location().into(),
                        pattern: path.to_string(),
                        ty: "<path>".to_string(),
                    })?,
                };

                // the pattern's own instantiation of the adt: unifying it against the scrutinee
                // binds these fresh arg vids to the scrutinee's actual args (e.g. `int`), and
                // `specialize` then plugs them into the variant's declared member types.
                // a dynamic scrutinee (`any`) skips the unify — the pattern is a
                // runtime probe and the args stay the variant's declared types.
                let ctor_ty = self.instantiated_adt(adt);
                if !dynamic {
                    unify(self, ty, ctor_ty.clone(), pat.location())?;
                }
                let ctor_args: Vec<Ty> = match &ctor_ty {
                    // a dynamic scrutinee pins none of the adt's args — the
                    // probe proves the variant, not its parameters, so bound
                    // members stay `any` rather than lingering as vids
                    Ty::Adt(_, args) if dynamic => vec![Ty::Any; args.len()],
                    Ty::Adt(_, args) => args.clone(),
                    _ => vec![],
                };
                let ctor_ty = ctor_ty.normalized(self);
                let variant_key = variant_name
                    .unwrap_or_else(|| self.adts[adt].variants.keys().next().unwrap().clone());
                let variant = self.adts[adt]
                    .variants
                    .get(&variant_key)
                    .ok_or_else(|| bad(&ctor_ty))?
                    .clone();
                // shadow the path with the layout adt id (per-variant for enums, the parent for
                // non-enum structs) so IR can read it to emit IsInstance / GetField.
                let layout = variant.layout().unwrap_or(adt);
                self.shadow_expr_ty(path.id(), Ty::adt(layout), path.location())?;
                match (pat.kind(), variant) {
                    (PatKind::Variant(_), _) => Ok(()),
                    (PatKind::TupleVariant(_, sub_pats), Variant::Tuple(tv)) => {
                        if tv.members.len() != sub_pats.len() {
                            Err(bad(&ctor_ty))?
                        }
                        for (sp, mty) in sub_pats.iter().zip(tv.members) {
                            let mty = self.specialize(&mty, adt, &ctor_args);
                            self.solve_match_pat(sp, mty, reuse)?;
                        }
                        Ok(())
                    }
                    (PatKind::Struct(_, field_pats), Variant::Struct(sv)) => {
                        for (name, sub) in field_pats {
                            let fty =
                                sv.fields.get(name).map(|f| f.ty.clone()).ok_or_else(|| {
                                    FieldNotFound {
                                        src: self.src(sub.location()),
                                        at: sub.location().into(),
                                        field_name: name.clone(),
                                    }
                                })?;
                            let fty = self.specialize(&fty, adt, &ctor_args);
                            self.solve_match_pat(sub, fty, reuse)?;
                        }
                        Ok(())
                    }
                    _ => Err(bad(&ctor_ty))?,
                }
            }
        }
    }
}

// Internal impls
impl Solver {
    pub(crate) fn visit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        #[cfg(feature = "logging")]
        println!("{}", crate::utils::Printer::stmt(stmt));
        match stmt.kind() {
            StmtKind::Assignment(Assignment { left, right, op }) => {
                fn find_root(left: &Expr) -> std::result::Result<&Ident, Location> {
                    match left.kind() {
                        ExprKind::Ident(ident) => Ok(ident),
                        ExprKind::Access(Access::Dot { left, .. })
                        | ExprKind::Access(Access::Square { left, .. }) => find_root(left),
                        _ => Err(left.location()),
                    }
                }

                let root = find_root(left).map_err(|loc| InvalidAssignTarget {
                    src: self.src(loc),
                    at: loc.into(),
                })?;
                let dec_id = match self.ribs.resolve(root) {
                    Some(d) => d,
                    // `m = 5`: the name resolves -- as a constant, and constants
                    // can't be assigned
                    None if shared::units::parse(&root.lexeme).is_some() => Err(AssignToConst {
                        src: self.src(root.location),
                        at: root.location.into(),
                        name: root.lexeme.clone(),
                    })?,
                    None => Err(NotFound {
                        src: self.src(root.location),
                        at: root.location.into(),
                        name: root.lexeme.clone(),
                    })?,
                };
                // `x = v` doesn't *read* x, but the grades pass needs the target's dec to
                // count the store -- same lookup as every other ident site
                self.node_decs.insert(root.id, dec_id);

                match self.decs[dec_id].kind {
                    DecKind::Constant(_) => Err(AssignToConst {
                        src: self.src(left.location()),
                        at: left.location().into(),
                        name: root.lexeme.clone(),
                    })?,
                    // only `x =` itself matters, if x was something like an adt `x.y =` would be
                    // fine
                    DecKind::LoopVar if matches!(left.kind(), ExprKind::Ident(_)) => {
                        Err(AssignToLoopVar {
                            src: self.src(left.location()),
                            at: left.location().into(),
                            name: root.lexeme.clone(),
                        })?
                    }
                    _ => {}
                }

                // `xs[i] = v` writes through the collection itself; if a `for` loop is
                // iterating that exact place the write can skip/double-visit elements or
                // never terminate.
                if let ExprKind::Access(Access::Square { left: receiver, .. }) = left.kind()
                    && let Some(path) = access_path(receiver)
                {
                    self.check_iter_guard(dec_id, &path, left.location())?;
                }

                if let ExprKind::Access(Access::Square { left: receiver, .. }) = left.kind() {
                    let receiver = receiver.query(self)?.normalized(self);
                    if receiver == Ty::Str
                        || matches!(&receiver, Ty::Option(inner) if **inner == Ty::Str)
                    {
                        Err(AssignToStringIndex {
                            src: self.src(left.location()),
                            at: left.location().into(),
                        })?
                    }
                }

                // `??=` mirrors read-form `??`: the target must be an option and the rhs
                // must fulfill its inner type (Coalescence::solve enforces the same).
                if *op == parse::AssignmentOp::NullCoalescenceEqual {
                    let lhs = left.query(self)?.normalized(self);
                    match lhs {
                        Ty::Option(inner) => right.fulfill_ty(*inner, self)?,
                        other => Err(crate::errors::CoalesceNonOptional {
                            src: self.src(left.location()),
                            at: left.location().into(),
                            ty: other.to_string(),
                        })?,
                    }
                } else {
                    // `x <op>= v` is `x = x <op> v`: an operand adt's operator
                    // method (`x.add(v)`, or `v.radd(x)` when the rhs owns the
                    // impl) beats the plain same-type rule.
                    let place = left.query(self)?;
                    // `xs += x`/`xs -= x` move the length — on a `[T; n]`
                    // place that's the same refusal `push`/`pop`/`extend`
                    // get. Checked outright rather than via the result
                    // type so the message names the rule, not a mismatch.
                    if matches!(
                        *op,
                        parse::AssignmentOp::PlusEqual | parse::AssignmentOp::MinusEqual
                    ) && let Ty::Array(_, Len::Const(n)) = place.clone().normalized(self)
                    {
                        Err(crate::errors::SizedArrayMutation {
                            src: self.src(left.location()),
                            at: left.location().into(),
                            method: op.to_string(),
                            ty: format!("[_; {n}]"),
                        })?;
                    }
                    let dispatched = match crate::traits::op_assign_names(*op) {
                        Some(names) => {
                            let place_n = place.clone().normalized(self);
                            let right_n = right.query(self)?.normalized(self);
                            crate::traits::op_dispatch(
                                self,
                                stmt.id(),
                                names,
                                left,
                                right,
                                &place_n,
                                &right_n,
                                left.location(),
                            )?
                        }
                        None => None,
                    };
                    match dispatched {
                        // the method's result still has to fit the place
                        Some(ret) => left.fulfill_ty(ret, self)?,
                        None => {
                            // `xs += ys`/`xs -= ys` — `add`/`sub` only take
                            // elements, so a list rhs falls through here to
                            // `union`/`minus`, the same call `xs + ys` makes.
                            let op = match *op {
                                parse::AssignmentOp::PlusEqual => Some(parse::EvaluationOp::Plus),
                                parse::AssignmentOp::MinusEqual => Some(parse::EvaluationOp::Minus),
                                _ => None,
                            };
                            let seq = match op {
                                Some(op) => {
                                    let place_n = place.clone().normalized(self);
                                    let right_n = right.query(self)?.normalized(self);
                                    crate::traits::seq_op_dispatch(
                                        self,
                                        stmt.id(),
                                        op,
                                        right,
                                        &place_n,
                                        &right_n,
                                        left.location(),
                                    )?
                                }
                                None => None,
                            };
                            match seq {
                                Some(ret) => left.fulfill_ty(ret, self)?,
                                None => right.fulfill_ty(place, self)?,
                            }
                        }
                    }
                }
            }
            StmtKind::Let(Let {
                left,
                annotation,
                right,
                else_branch,
            }) => {
                let vid = self.node_vid(right.id());
                let mut ty = Ty::Vid(vid);

                if let Some(annotation) = &annotation {
                    self.fulfill_annotation(&mut ty, annotation, right, left.location())?;
                };

                right.fulfill_ty(ty.clone(), self)?;

                if annotation.is_none() && ty.clone().normalized(self) == Ty::Null {
                    Err(BareNullBinding {
                        src: self.src(right.location()),
                        at: right.location().into(),
                    })?;
                }

                // a bare pact name has a Ty (so accesses like `P::CONST` can consume it) but
                // no value form -- binding one would ICE at codegen. pact-ANNOTATED lets with
                // a concrete rhs are fine, so key off the rhs expr being the pact ident itself.
                if matches!(right.kind(), ExprKind::Ident(_))
                    && let Some(&dec) = self.node_decs.get(&right.id())
                    && matches!(self.decs[dec].kind, DecKind::Pact(_))
                {
                    Err(crate::errors::PactIsNotAValue {
                        src: self.src(right.location()),
                        at: right.location().into(),
                    })?;
                }

                match else_branch {
                    // no `else` -> the pattern must be irrefutable (ident / tuple destructure)
                    None => self.solve_pat(left, ty)?,
                    // let-else: solve the `else` BEFORE binding (its names must not be in scope
                    // there) and isolate its flow so the diverging `else` doesn't make the rest
                    // of the block read as unreachable. then bind refutably.
                    Some(else_branch) => {
                        self.control_flow.enter();
                        let else_ty = else_branch.query(self)?;
                        self.control_flow.exit();
                        if else_ty.normalized(self) != Ty::Never {
                            Err(crate::errors::LetElseMustDiverge {
                                src: self.src(else_branch.location()),
                                at: else_branch.location().into(),
                            })?;
                        }
                        self.solve_match_pat(left, ty, false)?;
                    }
                }

                // a `table {}`/`query {}`/`.schema(…)`-typed RHS carries its column schema
                // onto the binding, so `df.pull("x")` resolves through the name
                self.propagate_frame_schemas(left, right);
            }
            StmtKind::Expr(expr) => expr.query(self).map(|_| ())?,
            StmtKind::Module(_) => {
                // module decls are handled in the pre-pass of `solve`.
            }
            StmtKind::Unit(_) => {
                // `unit` decls register into the units table at parse.
            }
            StmtKind::Item(item) => match item.kind() {
                // block-scoped consts get solved here, top-level consts are handled by the
                // hoist_constants/solve_consts fixpoint
                ItemKind::Const(con) => self.solve_const(con, item.id()).map(|_| ())?,
                // top-level uses install into this phase's import rib; a use
                // inside a block binds in its block and resolves here, where
                // every module's members are already hoisted.
                ItemKind::Use(_) => self.process_use(item)?,
                ItemKind::Poison(poison) => poison.escaped(),
                // a `space`'s fields must all be axes -- types implementing `Opp`
                ItemKind::Struct(st) if item.attrs().iter().any(|a| a.name.lexeme == "space") => {
                    self.check_space_axes(st)?
                }
                // all we need to do at this point is process the fn defaults
                ItemKind::Pact(pact) => {
                    let Some(pid) = self
                        .resolve_name(&pact.name, pact.name.location)?
                        .as_single_pact()
                    else {
                        unreachable!("a pact name always resolves to a single pact");
                    };
                    let prev_self = self.pact_self.replace(pid);
                    for item in pact.items.iter() {
                        let PactItem::Fn {
                            default: Some(body),
                            parameters,
                            return_type,
                            ..
                        } = item
                        else {
                            continue;
                        };

                        self.ribs.push_function();
                        for binding in parameters {
                            let param = self.bind_parameter(binding)?;
                            self.declare(
                                binding.left.as_ident().unwrap(),
                                binding.left.id(),
                                param.ty,
                            )?;
                        }
                        let expected = return_type
                            .clone()
                            .map_or(Ok(Ty::Unit), |a| Ty::from_annotation(a, self))?;
                        self.fn_stack.push(FnRun {
                            expected_ty: expected.clone(),
                        });
                        self.control_flow.enter();
                        // r[impl mimas.record.context] a literal the body yields fills the return type
                        self.hint_records(body, &expected);
                        let body_ty = body.query(self)?;
                        self.fn_stack.pop();
                        self.control_flow.exit();
                        if body_ty != Ty::Unit {
                            body.fulfill_ty(expected, self)?;
                        }
                        self.ribs.pop();
                    }
                    self.pact_self = prev_self;
                }
                ItemKind::Tests(tests) => {
                    for case in &tests.cases {
                        if !self.touch_node(case.id) {
                            continue;
                        }
                        self.ribs.push_function();
                        self.fn_stack.push(FnRun {
                            expected_ty: Ty::Bool,
                        });
                        self.control_flow.enter();
                        case.expr.fulfill_ty(Ty::Bool, self)?;
                        self.fn_stack.pop();
                        self.control_flow.exit();
                        self.ribs.pop();
                        let ty = Ty::Fn(FnHeader::new(vec![], Ty::Bool, false));
                        let vid = self.node_vid(case.id);
                        self.register_sub(vid, ty)
                            .map_err(|e| e.into_type_mismatch(self, case.expr.location()))?;
                    }
                }
                // each of these needs its inner Solve impl run during the body pass -- that's
                // where field-annotation unification (Struct/Enum), body type-checking
                // (Function), and method-body solving (Impl) happens. mirrors `Query for Expr`'s
                // touch/solve/register_sub dance for memoization. listed explicitly so new
                // ItemKinds force a compile error rather than silently panicking.
                ItemKind::Function(_)
                | ItemKind::Struct(_)
                | ItemKind::Enum(_)
                | ItemKind::Impl(_) => {
                    if self.touch_node(item.id()) {
                        let ty = match item.kind() {
                            ItemKind::Function(f) => f.solve(item.id(), item.location(), self)?,
                            ItemKind::Struct(s) => s.solve(item.id(), item.location(), self)?,
                            ItemKind::Enum(e) => e.solve(item.id(), item.location(), self)?,
                            ItemKind::Impl(i) => i.solve(item.id(), item.location(), self)?,
                            // narrowed by the outer arm -- inner exhaustiveness can't see it
                            _ => unreachable!("outer match restricts to Function/Struct/Enum/Impl"),
                        };
                        let vid = self.node_vid(item.id());
                        self.register_sub(vid, ty)
                            .map_err(|e| e.into_type_mismatch(self, item.location()))?;
                    }
                }
            },
        }
        Ok(())
    }

    /// Resolves the left of a `::` path (or a struct literal's name) at the type level. Unlike
    /// [Solve for Ident], a bare ident here may name a module or a builtin namespace.
    pub(crate) fn resolve_path_head(&mut self, expr: &parse::Expr) -> Result<Ty> {
        match expr.kind() {
            parse::ExprKind::Ident(ident) => self.resolve_name(ident, expr.location()),
            _ => self.query_non_value(expr),
        }
    }

    /// Queries `expr` somewhere it doesn't have to be a value. See [Self::non_value].
    pub(crate) fn query_non_value(&mut self, expr: &parse::Expr) -> Result<Ty> {
        let outer = self.non_value.replace(expr.id());
        let ty = expr.query(self);
        self.non_value = outer;
        ty
    }

    pub fn resolve_name(&mut self, ident: &Ident, read_location: Location) -> Result<Ty> {
        let (ty, dec) = if ident.is_identity() {
            // inside a pact, both `self` and `Self` are the implementing type -- one fixed but
            // unknown type, so two `Self`s are known to match (unlike two values of the bound)
            let ty = match self.pact_self {
                Some(pid) => Ty::Skolem(pid),
                None => self.impl_target_ty().ok_or_else(|| {
                    Error::from(SelfOutOfContext {
                        src: self.src(read_location),
                        at: read_location.into(),
                    })
                })?,
            };
            // `self` is still the method's declared parameter: link the use to its
            // dec so the grades pass counts the read (`None` here left every `self`
            // param looking unread, so `unused parameter self` fired on every impl
            // method). `Self` and pact signatures bind nothing, so they keep `None`.
            let dec = self.ribs.resolve(ident);
            // r[impl mimas.self.param]
            // a value `self` in a body that never declared it (`fn f(x: int)
            // -> Self { self }`) has nothing to read -- it used to reach
            // codegen unresolved and panic there
            if dec.is_none() && ident.lexeme == "self" {
                return Err(SelfOutOfContext {
                    src: self.src(read_location),
                    at: read_location.into(),
                }
                .into());
            }
            (ty, dec)
        } else {
            match self.ribs.resolve(ident) {
                Some(dec_id) => {
                    self.check_vis(dec_id, read_location)?;
                    // natives carry a fresh `Ty::Fn` per use so each call gets its own type vars;
                    // without this, two calls with different types unify and the second fails.
                    let ty = if let Some(binding) = self.dec_to_native.get(&dec_id) {
                        let sig = binding.sig.clone();
                        self.instantiate_native(&sig, None)?
                    } else {
                        // a generic decl's stored signature holds `Ty::Param`s -- swap them
                        // for fresh vars per use (the HM instantiation step), so each call
                        // site can pick its own concrete types. value bindings are exempt:
                        // a `fn` parameter or `let` is monomorphic, so `r: Rect<T>` must
                        // read the *same* `T` every occurrence -- freshening per read would
                        // mint unrelated vars and `r.x + r.w` would no longer see one type.
                        let stored = Ty::Vid(self.decs[dec_id].vid).normalized(self);
                        match self.decs[dec_id].kind {
                            DecKind::Local | DecKind::Global | DecKind::LoopVar => stored,
                            _ => {
                                // this read is where a bounded param's obligation lands
                                let prev = self.obligation_at.replace(read_location);
                                let ty = self.instantiate_params(&stored);
                                self.obligation_at = prev;
                                ty
                            }
                        }
                    };
                    (ty, Some(dec_id))
                }
                None if self.library.contains_key(&ident.lexeme) => {
                    (Ty::adt(self.library[&ident.lexeme]), None)
                }
                None => {
                    let pact = self.pacts.iter().find(|(_, p)| p.name == ident.lexeme);
                    if let Some((pid, _)) = pact {
                        (Ty::pacts(vec![pid]), None)
                    // a bare unit name is a constant: unbound `kWh` reads as `1kWh`, and
                    // `deg / s` composes through ordinary quantity arithmetic. Everything
                    // declared -- bindings, natives, pacts -- wins before this fires.
                    // r[impl mimas.unit.constant]
                    } else if let Some((dim, scale)) = shared::units::parse(&ident.lexeme) {
                        self.unit_const(ident, dim, scale)
                    } else {
                        // a tagged literal whose tag no native backs: say so in the
                        // literal's own spelling, not the lowered `__lit_` name
                        let name = match ident.lexeme.strip_prefix("__lit_") {
                            Some(tag) => {
                                format!("{tag}\"…\" (no `{tag}` literal tag is registered)")
                            }
                            None => ident.lexeme.clone(),
                        };
                        // `check(x)` touching the word parsed as a call --
                        // the check is a line, its hint is the spacing fix
                        if name == "check" {
                            Err(CheckIsNotCallable {
                                src: self.src(read_location),
                                at: read_location.into(),
                            })?
                        }
                        // `use music` then `pat`: the bare import bound the
                        // module's name, so a member name fails here -- say
                        // which module has it and how to really import it.
                        for (module, adt) in self.bare_module_uses.clone() {
                            if self.module_field(adt, ident).is_ok() {
                                Err(crate::errors::BareImportMember {
                                    src: self.src(read_location),
                                    at: read_location.into(),
                                    module,
                                    name: name.clone(),
                                })?
                            }
                        }
                        Err(NotFound {
                            src: self.src(read_location),
                            at: read_location.into(),
                            name,
                        })?
                    }
                }
            }
        };
        self.note(ident, ty.clone(), dec);
        Ok(ty)
    }

    /// Records what `ident` resolved to. This is only used by tooling.
    /// Notes `ty` as the expected type of every bare-brace literal `expr` is or
    /// directly holds — through `(…)`, array literals, and the value a block, an
    /// `if` branch or a `match` arm yields, so `let ps: [Pt] =
    /// [{ x: 1 }, { x: 2 }]` fills two `Pt`s. The first hint a literal gets wins.
    /// r[impl mimas.record.context]
    pub(crate) fn hint_records(&mut self, expr: &Expr, ty: &Ty) {
        match expr.kind() {
            ExprKind::Literal(Literal::Record(_)) => {
                self.record_hints.entry(expr.id()).or_insert_with(|| ty.clone());
            }
            ExprKind::Literal(Literal::Array(items)) => {
                let inner = match ty.clone().normalized(self) {
                    Ty::Array(inner, _) => *inner,
                    Ty::Option(o) => match *o {
                        Ty::Array(inner, _) => *inner,
                        _ => return,
                    },
                    _ => return,
                };
                for item in items {
                    self.hint_records(item, &inner);
                }
            }
            ExprKind::Grouping(g) => self.hint_records(&g.inner, ty),
            // the value a block, a branch or an arm hands back fills the same slot
            ExprKind::Block(b) => {
                if let Some(y) = &b.yielded_expr {
                    self.hint_records(y, ty);
                }
            }
            ExprKind::If(i) => {
                self.hint_records(&i.main_body, ty);
                if let Some(e) = &i.else_expr {
                    self.hint_records(e, ty);
                }
            }
            ExprKind::Match(m) => {
                for case in &m.cases {
                    self.hint_records(case.body(), ty);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn note(&mut self, ident: &Ident, ty: Ty, dec: Option<DecId>) {
        if let Some(dec) = dec {
            self.node_decs.insert(ident.id, dec);
        }
        let vid = self.node_vid(ident.id);
        // a name resolved twice yields the same type both times (a fresh native instantiation
        // just links to the earlier one), and nothing downstream depends on this vid
        let _ = self.register_sub(vid, ty);
    }

    /// Const-folds an expr using the solver's dec/lit tables to resolve Ident leaves. Returns
    /// the literal iff every leaf either reduces syntactically or is a const-bound ident whose
    /// value is already known. Errors (e.g. divide-by-zero) propagate up as diagnostics --
    /// reduce returns raw location-tagged errors and we materialize them with source here.
    pub(crate) fn reduce_const_expr(&self, expr: &Expr) -> Result<Option<Literal>> {
        let env = ConstEnv::default();
        self.reduce_const_in(expr, &env).map_err(|e| e.into_diag(self))
    }

    /// `reduce` with a param/`let` env: the resolve callback answers idents
    /// from `env` before the const table and folds field-ctor and Mimas-fn
    /// calls. Blocks, ifs and accesses have no stock `reduce` arm — they're
    /// intercepted here (and re-entered from the callback when nested inside
    /// a stock arm's operand).
    fn reduce_const_in(
        &self,
        expr: &Expr,
        env: &ConstEnv,
    ) -> crate::utils::ReduceResult<Option<Literal>> {
        match expr.kind() {
            ExprKind::Block(b) => return self.reduce_block(b, env),
            ExprKind::If(i) => return self.reduce_if(i, env),
            // `v.x`/`xs[i]` need the literal receiver — `reduce` has no arm
            // for them. `T::m`/`E::V` stay leaves: `resolve` answers variants
            // and assoc consts there.
            ExprKind::Access(Access::Dot { .. } | Access::Square { .. }) => {
                return self.reduce_access(expr, env);
            }
            _ => {}
        }
        let out = crate::utils::reduce(expr, &|e| {
            if self.is_unit_variant(e) {
                // `Team::Blue` as a value is the variant's empty struct literal --
                // emitted at each use site like any struct const
                return Some(Literal::Struct(parse::StructLiteral {
                    name: e.clone(),
                    fields: vec![],
                    base: None,
                }));
            }
            if let Some(&dec) = self.node_decs.get(&e.id()) {
                if let Some(lit) = env.lits.get(&dec) {
                    return Some(lit.clone());
                }
                if let Some(lit) = self.decs[dec].kind.const_value() {
                    return Some(lit.clone());
                }
            }
            match e.kind() {
                ExprKind::Call(_) => self
                    .const_ctor_literal(e, env)
                    .or_else(|| self.fold_mimas_call(e, env)),
                // the stock walker has no Block/If/Access arm — they decline to
                // resolve and land here, where the env-aware walk folds them
                ExprKind::Block(_) | ExprKind::If(_) | ExprKind::Access(_) => {
                    match self.reduce_const_in(e, env) {
                        Ok(lit) => lit,
                        Err(err) => {
                            let mut slot = env.err.borrow_mut();
                            if slot.is_none() {
                                *slot = Some(err);
                            }
                            None
                        }
                    }
                }
                _ => None,
            }
        })?;
        if let Some(err) = env.err.borrow_mut().take() {
            return Err(err);
        }
        match out {
            Some(lit) => self.reify_literal(lit, env),
            None => Ok(None),
        }
    }

    /// `reduce` only *checks* a literal's field exprs — it keeps them as
    /// written, so a `Literal` produced inside a fold env can still carry
    /// `self.x * k` where the use site has no `self`. Re-fold every embedded
    /// expr to its literal so the value that lands on the dec is closed.
    fn reify_literal(
        &self,
        lit: Literal,
        env: &ConstEnv,
    ) -> crate::utils::ReduceResult<Option<Literal>> {
        let reify = |e: &Expr| -> crate::utils::ReduceResult<Option<Expr>> {
            Ok(self.reduce_const_in(e, env)?.map(|l| l.into_expr()))
        };
        let reify_elems = |elems: Vec<Expr>| -> crate::utils::ReduceResult<Option<Vec<Expr>>> {
            let mut out = Vec::with_capacity(elems.len());
            for e in &elems {
                let Some(lit) = reify(e)? else {
                    return Ok(None);
                };
                out.push(lit);
            }
            Ok(Some(out))
        };
        match lit {
            Literal::Struct(mut s) => {
                for (_, field) in s.fields.iter_mut() {
                    let Some(lit) = reify(field)? else {
                        return Ok(None);
                    };
                    *field = lit;
                }
                if let Some(base) = s.base.as_mut() {
                    let Some(lit) = reify(base)? else {
                        return Ok(None);
                    };
                    *base = lit;
                }
                Ok(Some(Literal::Struct(s)))
            }
            Literal::Array(elems) => Ok(reify_elems(elems)?.map(Literal::Array)),
            Literal::Tuple(elems) => Ok(reify_elems(elems)?.map(Literal::Tuple)),
            Literal::Dictionary(fields) => {
                let mut out = Vec::with_capacity(fields.len());
                for (key, value) in &fields {
                    let Some(lit) = reify(value)? else {
                        return Ok(None);
                    };
                    out.push((key.clone(), lit));
                }
                Ok(Some(Literal::Dictionary(out)))
            }
            _ => Ok(Some(lit)),
        }
    }

    /// `f(args…)` / `recv.f(args…)` where `f` is a recorded Mimas fn: bind the
    /// params to the reduced args (defaults filling the tail) and fold the
    /// body. Anything non-const, a wrong arity, or a cycle declines to `None`
    /// and the call stays a runtime call outside a `const`.
    // r[impl mimas.const.eval]
    fn fold_mimas_call(&self, expr: &Expr, env: &ConstEnv) -> Option<Literal> {
        let ExprKind::Call(call) = expr.kind() else {
            return None;
        };
        let (dec, item, recv) = self.mimas_callee(call)?;
        if env.visiting.contains(&dec) {
            return None;
        }
        let mut inner = ConstEnv {
            visiting: {
                let mut v = env.visiting.clone();
                v.insert(dec);
                v
            },
            err: env.err.clone(),
            ..ConstEnv::default()
        };
        let offset = match recv {
            Some(recv_expr) => {
                let lit = self.reduce_const_in(recv_expr, env).ok()??;
                inner.lits.insert(item.params[0], lit);
                1
            }
            None => 0,
        };
        let params = &item.params[offset..];
        let defaults = match &self.decs[dec].kind {
            DecKind::Item { defaults, .. } => defaults.as_slice(),
            _ => &[],
        };
        if call.arguments.len() > params.len() {
            return None;
        }
        for (i, &param) in params.iter().enumerate() {
            let lit = match call.arguments.get(i) {
                Some(arg) => {
                    if arg.name.is_some() {
                        return None;
                    }
                    self.reduce_const_in(&arg.value, env).ok()??
                }
                None => defaults.get(offset + i)?.clone()?,
            };
            inner.lits.insert(param, lit);
        }
        self.reduce_const_in(&item.body, &inner).ok()?
    }

    /// The recorded Mimas fn a call invokes, plus its receiver expr for a
    /// dot-call (`recv.f(..)` — the access's left). For a `T::f(recv, ..)`
    /// call `self` is just arg 0; a bare `f(..)` call on a method has no
    /// receiver to bind and is never const.
    fn mimas_callee<'e>(
        &self,
        call: &'e Call,
    ) -> Option<(DecId, &ConstItem, Option<&'e Expr>)> {
        let dec = *self.node_decs.get(&call.left.id())?;
        if self.dec_to_native.contains_key(&dec) {
            return None;
        }
        let item = self.const_items.get(&dec)?;
        if item.params.len() < usize::from(item.method) {
            return None;
        }
        let recv = match call.left.kind() {
            ExprKind::Access(Access::Dot { left, .. }) if item.method => Some(left),
            ExprKind::Ident(_) if item.method => return None,
            _ => None,
        };
        Some((dec, item, recv))
    }

    /// Reduce a block under an env that grows a binding per `let`: the only
    /// statements a const block may hold are `let`s and expression
    /// statements (both must fold), anything else is a side effect.
    fn reduce_block(
        &self,
        block: &parse::Block,
        env: &ConstEnv,
    ) -> crate::utils::ReduceResult<Option<Literal>> {
        let mut scoped = env.clone();
        for stmt in &block.body {
            match stmt.kind() {
                StmtKind::Let(let_stmt) if let_stmt.else_branch.is_none() => {
                    let Some(lit) = self.reduce_const_in(&let_stmt.right, &scoped)? else {
                        return Ok(None);
                    };
                    let Some(ident) = let_stmt.left.as_ident() else {
                        return Ok(None);
                    };
                    let Some(&dec) = self
                        .node_decs
                        .get(&let_stmt.left.id())
                        .or_else(|| self.node_decs.get(&ident.id))
                    else {
                        return Ok(None);
                    };
                    scoped.lits.insert(dec, lit);
                }
                StmtKind::Expr(expr) => {
                    if self.reduce_const_in(expr, &scoped)?.is_none() {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        }
        match &block.yielded_expr {
            Some(tail) => self.reduce_const_in(tail, &scoped),
            None => Ok(Some(Literal::Unit)),
        }
    }

    /// `if cond { a } else { b }` folds when the condition does: reduce it and
    /// take the branch it names. A missing `else` yields `Unit` on `false`.
    fn reduce_if(
        &self,
        if_expr: &parse::If,
        env: &ConstEnv,
    ) -> crate::utils::ReduceResult<Option<Literal>> {
        if if_expr.binding.is_some() {
            return Ok(None);
        }
        match self.reduce_const_in(&if_expr.condition, env)? {
            Some(Literal::True) => self.reduce_const_in(&if_expr.main_body, env),
            Some(Literal::False) => match &if_expr.else_expr {
                Some(else_expr) => self.reduce_const_in(else_expr, env),
                None => Ok(Some(Literal::Unit)),
            },
            _ => Ok(None),
        }
    }

    /// `v.f`, `v[0]` and `self.f` project onto a literal: fold the receiver,
    /// then read the field or element out of the `Literal` it produced.
    fn reduce_access(
        &self,
        expr: &Expr,
        env: &ConstEnv,
    ) -> crate::utils::ReduceResult<Option<Literal>> {
        let ExprKind::Access(access) = expr.kind() else {
            return Ok(None);
        };
        // a callable (a method value) resolves through node_decs -- a bare
        // `v.f` used as a value is not a field projection and stays non-const
        if let Some(&dec) = self.node_decs.get(&expr.id())
            && self
                .sub(self.decs[dec].vid)
                .is_some_and(|ty| matches!(ty, Ty::Fn(_)))
        {
            return Ok(None);
        }
        let (recv, key): (Literal, &Expr) = match access {
            Access::Dot {
                left,
                right,
                kind: AccessKind::Direct,
            } => match self.reduce_const_in(left, env)? {
                Some(recv) => (recv, right),
                None => return Ok(None),
            },
            Access::Square {
                left,
                key,
                kind: AccessKind::Direct,
            } => match self.reduce_const_in(left, env)? {
                Some(recv) => (recv, key),
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        match access {
            Access::Square { .. } => match (recv, self.reduce_const_in(key, env)?) {
                (Literal::Array(elems) | Literal::Tuple(elems), Some(Literal::Int(i))) => {
                    let Some(elem) = elems.get(i as usize) else {
                        return Ok(None);
                    };
                    self.reduce_const_in(elem, env)
                }
                _ => Ok(None),
            },
            _ => {
                let name = match key.kind() {
                    ExprKind::Ident(ident) => FieldName::Ident(&ident.lexeme),
                    ExprKind::Literal(Literal::Int(i)) => FieldName::Index(*i as usize),
                    _ => return Ok(None),
                };
                self.project_field(&recv, &name)
            }
        }
    }

    /// Read field `name` off a `Literal::Struct` (or an element off a
    /// `Literal::Tuple`), folding the stored field expr to its literal. Any
    /// other literal shape declines.
    fn project_field(
        &self,
        recv: &Literal,
        name: &FieldName<'_>,
    ) -> crate::utils::ReduceResult<Option<Literal>> {
        let expr = match (recv, name) {
            (Literal::Struct(s), _) => s
                .fields
                .iter()
                .find(|(key, _)| match (key, name) {
                    (FieldKey::Ident(i), FieldName::Ident(n)) => i.lexeme == **n,
                    (FieldKey::Int(a), FieldName::Index(b)) => a == b,
                    _ => false,
                })
                .map(|(_, expr)| expr),
            (Literal::Tuple(elems), FieldName::Index(i)) => elems.get(*i),
            _ => None,
        };
        let Some(expr) = expr else {
            return Ok(None);
        };
        self.reduce_const_in(expr, &ConstEnv::default())
    }

    /// `xs[i]`/`v.f` count as const when the receiver and key are -- the fold
    /// is what actually proves the projection exists.
    fn is_const_access(&self, expr: &Expr, scope: &mut ConstScope) -> bool {
        let ExprKind::Access(access) = expr.kind() else {
            return false;
        };
        // a callable dec on the access itself (a method/assoc value) is not a
        // field projection
        if let Some(&dec) = self.node_decs.get(&expr.id())
            && self
                .sub(self.decs[dec].vid)
                .is_some_and(|ty| matches!(ty, Ty::Fn(_)))
        {
            return false;
        }
        match access {
            Access::Dot { left, kind: AccessKind::Direct, .. } => {
                self.is_const_expr_in(left, scope)
            }
            Access::Square {
                left,
                key,
                kind: AccessKind::Direct,
            } => self.is_const_expr_in(left, scope) && self.is_const_expr_in(key, scope),
            Access::DoubleColon { .. } => self
                .node_decs
                .get(&expr.id())
                .is_some_and(|&dec| self.decs[dec].kind.is_constant()),
            _ => false,
        }
    }

    /// `f(args…)`/`recv.f(args…)` is const when `f` is a recorded Mimas fn,
    /// the receiver and every arg are const, and the body under its params is
    /// a const expression. A callee already being folded (recursion) is not.
    fn is_const_mimas_call(&self, call: &Call, scope: &mut ConstScope) -> bool {
        let Some((dec, item, recv)) = self.mimas_callee(call) else {
            return false;
        };
        if scope.visiting.contains(&dec) {
            return false;
        }
        let offset = match recv {
            Some(recv_expr) => {
                if !self.is_const_expr_in(recv_expr, scope) {
                    return false;
                }
                1
            }
            None => 0,
        };
        let params = &item.params[offset..];
        let defaults = match &self.decs[dec].kind {
            DecKind::Item { defaults, .. } => {
                defaults.get(offset..).unwrap_or(&[])
            }
            _ => &[],
        };
        let required = params
            .len()
            .saturating_sub(defaults.iter().filter(|d| d.is_some()).count());
        if call.arguments.len() > params.len()
            || call.arguments.len() < required
            || call.arguments.iter().any(|a| a.name.is_some())
            || !call
                .arguments
                .iter()
                .all(|a| self.is_const_expr_in(&a.value, scope))
        {
            return false;
        }
        let mut inner = scope.clone();
        inner.decs.extend(item.params.iter().copied());
        inner.visiting.insert(dec);
        self.is_const_expr_in(&item.body, &mut inner)
    }

    /// Does `expr` call a Mimas fn anywhere? Used by the early `solve_consts`
    /// fixpoint to tell "fails because a callee's body isn't recorded yet"
    /// (deferrable — `solve_bodies` may make it fold) from "fails on its own
    /// terms" (error now).
    fn calls_unsolved_fn(&self, expr: &Expr) -> bool {
        struct Probe<'s> {
            solver: &'s Solver,
            hit: bool,
        }
        impl parse::visit::Visitor for Probe<'_> {
            fn expr(&mut self, expr: &Expr) {
                if let ExprKind::Call(call) = expr.kind()
                    && let Some(&dec) = self.solver.node_decs.get(&call.left.id())
                    && matches!(self.solver.decs[dec].kind, DecKind::Item { .. })
                    && !self.solver.dec_to_native.contains_key(&dec)
                {
                    self.hit = true;
                }
            }
        }
        let mut probe = Probe {
            solver: self,
            hit: false,
        };
        parse::visit::walk_expr(expr, &mut probe);
        probe.hit
    }

    /// `T::f(...)` where `f` is a field-constructor native: each field of `T`'s single
    /// named-field variant comes verbatim from a parameter, so the call *is* a struct
    /// literal in disguise. Returns the name expr (the `T` path -- cloned, so its
    /// `node_tys` still resolves), `(field name, arg index)` pairs in declaration order,
    /// and the arity the call must match. Positional args only: a named arg would need
    /// field-name matching rather than position, and stays a runtime call.
    // r[impl mimas.const.ctor]
    fn const_ctor_fields(&self, expr: &Expr) -> Option<(Expr, Vec<(String, u8)>, usize)> {
        let ExprKind::Call(call) = expr.kind() else {
            return None;
        };
        let ExprKind::Access(parse::Access::DoubleColon { .. }) = call.left.kind() else {
            return None;
        };
        let dec = *self.node_decs.get(&call.left.id())?;
        let binding = self.dec_to_native.get(&dec)?;
        let map = binding.sig.ctor.as_ref()?;
        let Ty::Adt(adt_id, _) = binding.sig.recv.clone()?.normalized(self) else {
            return None;
        };
        let fields = self.adts[adt_id]
            .try_as_struct()?
            .fields
            .keys()
            .cloned()
            .zip(map.iter().copied())
            .collect();
        // the call node is the literal's `name`: the type ident `left` is a
        // path head, so nothing notes a ty or dec on it — the call itself
        // types as the adt, which is what `Literal::Struct` emit needs
        Some((expr.clone(), fields, binding.sig.params.len()))
    }

    /// The `Literal::Struct` a const ctor call reduces to -- each field's arg reduced in
    /// turn. A non-const or erroring arg declines the whole call (the caller's
    /// `NonConstantValue` then reports it like any other unreduced call).
    fn const_ctor_literal(&self, expr: &Expr, env: &ConstEnv) -> Option<Literal> {
        let ExprKind::Call(call) = expr.kind() else {
            return None;
        };
        let (name, field_args, arity) = self.const_ctor_fields(expr)?;
        if call.arguments.len() != arity || call.arguments.iter().any(|a| a.name.is_some())
        {
            return None;
        }
        let mut fields = Vec::with_capacity(field_args.len());
        for (key, index) in field_args {
            let lit = self
                .reduce_const_in(&call.arguments[index as usize].value, env)
                .ok()??;
            fields.push((
                parse::FieldKey::Ident(Ident::synthetic(key)),
                lit.into_expr(),
            ));
        }
        Some(Literal::Struct(parse::StructLiteral {
            name,
            fields,
            base: None,
        }))
    }

    /// `E::V` naming a field-less enum variant (`opp Team { Blue, Red }`'s
    /// `Team::Blue`): a value with nothing to compute, so a `const` may be one.
    pub(crate) fn is_unit_variant(&self, expr: &Expr) -> bool {
        matches!(
            expr.kind(),
            ExprKind::Access(parse::Access::DoubleColon { .. })
        ) && self.node_decs.get(&expr.id()).is_some_and(|&dec| {
            matches!(self.decs[dec].kind, DecKind::Variant { layout, .. }
                    if self.adts[layout].as_singular().is_some_and(|v| v.is_empty()))
        })
    }

    pub(crate) fn is_const_expr(&self, expr: &Expr) -> bool {
        self.is_const_expr_in(expr, &mut ConstScope::default())
    }

    /// `is_const_expr` under a [`ConstScope`]: the same syntactic whitelist plus
    /// param/`let` decs from the calls and blocks being folded.
    fn is_const_expr_in(&self, expr: &Expr, scope: &mut ConstScope) -> bool {
        if self.is_unit_variant(expr) {
            return true;
        }
        match expr.kind() {
            ExprKind::Ident(_) => self.node_decs.get(&expr.id()).is_some_and(|&dec| {
                scope.decs.contains(&dec) || self.decs[dec].kind.is_constant()
            }),
            // elements may be anything const, structs and dicts included: an
            // array holding one allocates at each use site (`emit_const_dec`)
            ExprKind::Literal(parse::Literal::Array(elems) | parse::Literal::Tuple(elems)) => {
                elems.iter().all(|e| self.is_const_expr_in(e, scope))
            }
            ExprKind::Literal(_) => true,
            ExprKind::Grouping(g) => self.is_const_expr_in(&g.inner, scope),
            ExprKind::Unary(u) => self.is_const_expr_in(&u.right, scope),
            ExprKind::Evaluation(e) => {
                self.is_const_expr_in(&e.left, scope) && self.is_const_expr_in(&e.right, scope)
            }
            ExprKind::Equality(e) => {
                self.is_const_expr_in(&e.left, scope) && self.is_const_expr_in(&e.right, scope)
            }
            ExprKind::Logical(l) => {
                self.is_const_expr_in(&l.left, scope) && self.is_const_expr_in(&l.right, scope)
            }
            ExprKind::Call(call) => {
                // `Vec2::new(1.0, 2.0)`: a field-constructor call whose args are
                // all const reduces to the struct literal (r[mimas.const.ctor])
                let ctor = self
                    .const_ctor_fields(expr)
                    .is_some_and(|(_, _, arity)| call.arguments.len() == arity)
                    && call
                        .arguments
                        .iter()
                        .all(|a| a.name.is_none() && self.is_const_expr_in(&a.value, scope));
                // `f(const…)` where `f`'s body is itself a const expression
                ctor || self.is_const_mimas_call(call, scope)
            }
            ExprKind::Block(block) => self.is_const_block(block, scope),
            ExprKind::If(if_expr) => {
                if_expr.binding.is_none()
                    && self.is_const_expr_in(&if_expr.condition, scope)
                    && self.is_const_expr_in(&if_expr.main_body, scope)
                    && if_expr
                        .else_expr
                        .as_ref()
                        .is_none_or(|e| self.is_const_expr_in(e, scope))
            }
            ExprKind::Access(_) => self.is_const_access(expr, scope),
            _ => false,
        }
    }

    /// A block is const when it holds only `let`s over const rhs's and const
    /// expression statements — bound names enter the scope as it walks.
    fn is_const_block(&self, block: &parse::Block, scope: &mut ConstScope) -> bool {
        let mut scoped = scope.clone();
        for stmt in &block.body {
            match stmt.kind() {
                StmtKind::Let(let_stmt) if let_stmt.else_branch.is_none() => {
                    if !self.is_const_expr_in(&let_stmt.right, &mut scoped) {
                        return false;
                    }
                    let Some(ident) = let_stmt.left.as_ident() else {
                        return false;
                    };
                    if let Some(&dec) = self
                        .node_decs
                        .get(&let_stmt.left.id())
                        .or_else(|| self.node_decs.get(&ident.id))
                    {
                        scoped.decs.insert(dec);
                    }
                }
                StmtKind::Expr(expr) => {
                    if !self.is_const_expr_in(expr, &mut scoped) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        match &block.yielded_expr {
            Some(tail) => self.is_const_expr_in(tail, &mut scoped),
            None => true,
        }
    }

    pub(crate) fn impl_target(&self) -> Option<AdtId> {
        self.ribs.impl_target()
    }

    /// `Self` as a type inside an impl body: `Ty::Adt(target)` for a plain impl, or
    /// `Adt(target, [Param A, Param B, ...])` for a generic one -- `impl Pair` implements
    /// `Pair<A, B>` with the adt's own params in scope, not any single instantiation.
    pub(crate) fn impl_target_ty(&self) -> Option<Ty> {
        let aid = self.impl_target()?;
        // `impl Opp for bool`: inside, `Self` is the primitive, not the
        // library's `bool` namespace adt
        if let Some(prim) = self.primitive_of(aid) {
            return Some(prim);
        }
        let args = self.adts[aid]
            .type_params
            .iter()
            .map(|(_, pid)| Ty::Param(*pid))
            .collect();
        Some(Ty::Adt(aid, args))
    }

    /// Resolves a function/closure parameter's type into an `FnParam` for the `FnHeader`: takes
    /// annotation if present, unifies with the default value's type if present, fixes `self` to
    /// the current impl target if any. Allocates one `Vid` per param and unifies into it
    /// directly via `TyExt::fulfill_ty`, and does not create a `Dec` or touch `node_decs`. The
    /// real per-body binding happens in `Function::solve`, which is the sole writer of the param's
    /// dec; this used to double-bind (throwaway Dec here + real Dec there).
    pub(crate) fn bind_parameter(&mut self, binding: &Binding) -> Result<FnParam> {
        let Binding {
            left,
            right,
            annotation,
        } = binding;
        let ident = left.as_ident().unwrap();
        let vid = self.vid();
        let mut ty = Ty::Vid(vid);

        // `self` inside an impl method gets its type fixed to the impl target -- same effect as
        // writing `self: Self` explicitly. outside an impl, the body-pass declare leaves it as a
        // vid and SelfOutOfContext fires at first use.
        if ident.lexeme == "self"
            && let Some(mut target) = self.impl_target_ty()
        {
            ty.fulfill_ty(&mut target, self)
                .map_err(|e| e.into_type_mismatch(self, ident.location))?;
        }

        if let Some(annotation) = annotation.clone() {
            let mut target = Ty::from_annotation(annotation, self)?;
            ty.fulfill_ty(&mut target, self)
                .map_err(|e| e.into_type_mismatch(self, ident.location))?;
        }
        let has_default = if let Some(right) = right {
            let mut rhs_ty = right.query(self)?;
            ty.fulfill_ty(&mut rhs_ty, self)
                .map_err(|e| e.into_type_mismatch(self, ident.location))?;
            true
        } else {
            false
        };

        Ok(FnParam::new(
            Some(ident.lexeme.clone()),
            Ty::Vid(vid).normalized(self),
            has_default,
        ))
    }

    /// Binds a local into the current scope.
    pub(crate) fn declare(&mut self, ident: &Ident, node_id: NodeId, ty: Ty) -> Result<Ty> {
        self.declare_internal(ident, Some(node_id), ty)
    }

    fn declare_internal(&mut self, ident: &Ident, node_id: Option<NodeId>, ty: Ty) -> Result<Ty> {
        #[cfg(feature = "logging")]
        println!("{}", crate::utils::Printer::declare(ident, &ty));

        // a `let` declared in the file's top-level statement scope is a global: it stays
        // visible across function boundaries and its storage lives in the entry frame. every
        // other declaration context is an ordinary local.
        let kind = match self.ribs.current_kind() {
            RibKind::Script => DecKind::Global,
            _ => DecKind::Local,
        };
        let dec_id = self.dec_id(ident, ty.clone(), kind, Vis::Public);
        if let Some(node_id) = node_id {
            self.node_decs.insert(node_id, dec_id);
        }
        self.ribs.current_mut().insert(ident.clone(), dec_id);
        Ok(ty)
    }

    /// Binds a hoisted item (fn, type, or `const`) into the root module scope,
    /// so it resolves from anywhere, including across function boundaries.
    pub(crate) fn declare_item(
        &mut self,
        ident: &Ident,
        ty: Ty,
        node_id: Option<NodeId>,
        constant: bool,
        vis: Vis,
    ) -> Result<Ty> {
        #[cfg(feature = "logging")]
        println!("{}", crate::utils::Printer::declare(ident, &ty));

        if constant && let Some(prev_id) = self.ribs.module_lookup(ident) {
            let prev = &self.decs[prev_id];
            if prev.kind.is_constant() {
                let prev_location = prev.location;
                Err(MultipleConstDeclarations {
                    src: self.src(ident.location),
                    name: ident.lexeme.clone(),
                    at: ident.location.into(),
                    original: vec![crate::errors::ConstDefinedHere {
                        src: self.src(prev_location),
                        at: prev_location.into(),
                        name: ident.lexeme.clone(),
                    }],
                })?;
            }
        }

        let kind = if constant {
            DecKind::Constant(None)
        } else {
            DecKind::Item {
                defaults: vec![],
                wheres: vec![],
            }
        };
        let dec_id = self.dec_id(ident, ty.clone(), kind, vis);
        self.ribs.module_mut().insert(ident.clone(), dec_id);
        self.module_items
            .entry(self.ribs.current_module())
            .or_default()
            .insert(ident.lexeme.clone(), dec_id);

        if let Some(node_id) = node_id {
            self.node_decs.insert(node_id, dec_id);
        }

        Ok(ty)
    }

    pub(crate) fn run_loop_body(&mut self, body: &Expr) -> Result<LoopRun> {
        self.loop_stack.push(LoopRun {
            collect_ty: None,
            break_ty: None,
        });
        body.fulfill_ty(Ty::Unit, self)?;
        Ok(self
            .loop_stack
            .pop()
            .map(|mut l| {
                l.break_ty = l.break_ty.map(|v| v.normalized(self));
                l.collect_ty = l.collect_ty.map(|v| v.normalized(self));
                l
            })
            .unwrap())
    }

    pub(crate) fn regsiter_loop_control_flow_ty(
        &mut self,
        mut ty: Ty,
        target_option: &mut Option<Ty>,
    ) -> std::result::Result<(), UnificationError> {
        if let Some(break_ty) = target_option
            && let Err(e) = ty.fulfill_ty(break_ty, self)
        {
            ty = Ty::coerce_option(ty, break_ty.clone(), self).ok_or(e)?;
        }

        *target_option = Some(ty);
        Ok(())
    }

    /// Runs `f` against the innermost loop frame. The frame is detached for the duration so `f`
    /// can hold `&mut Session` alongside `&mut LoopRun`.
    pub(crate) fn with_loop_mut<R>(
        &mut self,
        not_in_loop: impl FnOnce() -> Error,
        f: impl FnOnce(&mut Solver, &mut LoopRun) -> Result<R>,
    ) -> Result<R> {
        let mut loop_data = self.loop_stack.pop().ok_or_else(not_in_loop)?;
        let r = f(self, &mut loop_data);
        self.loop_stack.push(loop_data);
        r
    }

    pub(crate) fn touch_node(&mut self, node_id: NodeId) -> bool {
        self.node_visits.insert(node_id)
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub(crate) struct LoopRun {
    pub collect_ty: Option<Ty>,
    pub break_ty: Option<Ty>,
}

#[derive(Clone)]
pub(crate) struct FnRun {
    pub expected_ty: Ty,
}

// Per-dec `Option<Ty>`-shape signature for native callables. Lets each Ident resolution mint
// fresh vids per call site instead of sharing one set of inference variables. `recv` carries the
// pattern type for the receiver (e.g. `[Anon(0)]` for `Vec<T>`) -- methods use it at dispatch to
// reject receivers whose element type doesn't fit.
#[derive(Clone)]
pub(crate) struct NativeFnSig {
    pub params: Vec<Option<Ty>>,
    /// Per-slot dimension from unit-carrying `MimasType`s, parallel to `params`. The dims
    /// pass checks call args against it; `None` leaves the slot unchecked.
    pub param_dims: Vec<Option<shared::units::Dim>>,
    pub return_ty: Option<Ty>,
    /// Dimension of the return value, when it measures something.
    pub return_dim: Option<shared::units::Dim>,
    pub recv: Option<Ty>,
    /// The native's declared effect footprint (`#[effects(...)]`), read by the grades
    /// pass. `None` = unannotated: inference treats the call as `Fx::unknown()`.
    pub effects: Option<shared::Fx>,
    /// Parallel to `params` (`#[consumes]`): passing a binding to a true slot
    /// consumes it -- the grades pass kills the `DecId` at the call site.
    pub consumes: Vec<bool>,
    /// `#[consumes(self)]` on a method -- the receiver binding is consumed.
    pub consumes_recv: bool,
    /// `#[must_use]` -- the grades pass warns on a discarded result.
    pub must_use: bool,
    /// A field constructor — `Some(map)` where `map[i]` is the parameter supplying the
    /// receiver's field `i` (declared order), so a `T::f(const…)` call reduces to
    /// `T { f = const, … }` and a `const` may hold it (r[mimas.const.ctor]).
    pub ctor: Option<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) struct NativeBinding {
    pub id: NativeId,
    pub sig: NativeFnSig,
    pub takes_self: bool,
    /// From `ApiMethod::mutates_recv` -- lets `check_iter_guard` know a call mutates its
    /// receiver. Always false for non-method natives.
    pub mutates_recv: bool,
    /// From `ApiMethod::validate`/`ApiFunction::validate` -- when every call argument is a
    /// literal this proves the call can't raise (`frame_call` narrows `T!` to `T`) or reports
    /// it definitely will (a compile error). See `api::LitValidator`.
    pub validate: Option<api::LitValidator>,
    /// Def site of the registered Rust fn (`file!()`/`line!()`), joined at install from
    /// `vm::api::NativeSrc` submissions. Hosts use it to link a built-in's symbol menu to
    /// its source.
    pub src: Option<api::NativeSrc>,
    /// The `///` docstring `#[native]`/`#[mimas]` harvested (`vm::api::NativeMeta`), so the
    /// symbol menu can show a built-in's doc without leaving the page. Empty when none.
    pub doc: String,
}

/// A collection an active `for` loop is iterating, expressed as the binding the iterator's
/// root name resolved to plus the field/index path from there (`s.arr` guards `s` with path
/// `[Field("arr")]`). A `Square` hop matches any index -- `xs[i]` vs `xs[j]` is beyond this
/// check's remit.
///
/// This is a lint for the beginner trap of mutating a collection mid-loop, not an alias
/// analysis: writes reached through a *different* binding (`let ys = xs; ys.push(..)`), a
/// helper function's parameter, or a pact-bound method call are all knowingly missed.
#[derive(Clone)]
pub(crate) struct IterGuard {
    pub dec: DecId,
    pub path: Vec<GuardSeg>,
    /// The `for` iterator expr's location, for the `IteratingHere` sub-diagnostic.
    pub iterated_at: Location,
    /// The iterator expr rendered, for messages.
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GuardSeg {
    Field(String),
    Index,
}

/// Walks `expr`'s access spine to its root ident, returning it plus the field/index hops from
/// there (`s.arr[i]` -> the `s` ident and `[Field("arr"), Index]`). `None` when `expr` isn't a
/// place rooted at a name -- calls, literals, operator results all produce fresh values that
/// can't alias an iterated collection in any way this check needs to know about.
pub(crate) fn root_and_path(expr: &Expr) -> Option<(&Ident, Vec<GuardSeg>)> {
    let mut path = Vec::new();
    let mut cur = expr;
    loop {
        match cur.kind() {
            ExprKind::Ident(ident) => {
                path.reverse();
                return Some((ident, path));
            }
            ExprKind::Access(Access::Dot { left, right, .. }) => {
                path.push(GuardSeg::Field(right.as_ident()?.lexeme.clone()));
                cur = left;
            }
            ExprKind::Access(Access::DoubleColon { left, right }) => {
                path.push(GuardSeg::Field(right.lexeme.clone()));
                cur = left;
            }
            ExprKind::Access(Access::Square { left, .. }) => {
                path.push(GuardSeg::Index);
                cur = left;
            }
            ExprKind::Grouping(g) => cur = &g.inner,
            ExprKind::Unwrap(u) => cur = &u.expr,
            ExprKind::Demote(d) => cur = &d.expr,
            ExprKind::Absolve(a) => cur = &a.left,
            _ => return None,
        }
    }
}

/// The access path of a place expr, dropping the root (already identified by its resolved dec).
pub(crate) fn access_path(expr: &Expr) -> Option<Vec<GuardSeg>> {
    root_and_path(expr).map(|(_, path)| path)
}

/// One module-nested native function, as handed to [`Solver::register_native_module`].
pub struct NativeModuleFn {
    pub name: String,
    pub params: Vec<Option<Ty>>,
    pub return_ty: Option<Ty>,
    pub native_id: NativeId,
}

/// What a `use` item resolved to, kept on its node id so every scope it lands
/// in takes the same decs. `Named` bindings install as-is; `Glob` re-collects
/// the module's visible members at each install.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedUse {
    /// `use m;`, `use m::x;`, `use m::{x, y};` -- each name to one dec.
    Named(Vec<(String, DecId)>),
    /// `use m::*;` -- the module's members visible *here*, re-collected at
    /// each install so members hoisted in a later phase still land.
    Glob(AdtId),
}

/// The name a pact member key spells: `mul#1` is an overload of `mul`.
/// r[impl mimas.pact.overload]
pub fn pact_key_base(key: &str) -> &str {
    key.split('#').next().unwrap_or(key)
}

/// A Mimas `fn` recorded for the const call fold (`Function::solve`). A `const`
/// may call one when the call's args are constant and the body is itself a
/// const expression — the call reduces by folding the body under a
/// param → literal env. `params` is every parameter's dec in declaration order
/// (`self` first for a method); `body`'s `node_decs` are in place.
// r[impl mimas.const.eval]
#[derive(Clone)]
pub(crate) struct ConstItem {
    pub params: Vec<DecId>,
    pub body: Expr,
    /// The first parameter is `self` — a dot-call supplies the receiver.
    pub method: bool,
}

/// Scope state for the `is_const_expr` walk: decs that count as constant (a
/// called fn's params, `let`s bound in a const block) and the callees already
/// being folded — a recursive call has no finite literal, so it isn't const.
/// Cloned into block/call scope.
#[derive(Clone, Default)]
struct ConstScope {
    decs: FxHashSet<DecId>,
    visiting: FxHashSet<DecId>,
}

/// The value-side twin of [`ConstScope`] for `reduce_const_in`: dec → literal
/// bindings for params and `let`s in the bodies being folded and the callees
/// already being folded. `err` is shared across clones — the first nested
/// fold error wins.
#[derive(Clone, Default)]
struct ConstEnv {
    lits: FxHashMap<DecId, Literal>,
    visiting: FxHashSet<DecId>,
    err: Rc<std::cell::RefCell<Option<crate::utils::ReduceError>>>,
}

/// A field projection target: `v.x` by name, `v.0`/`t[i]` by position.
enum FieldName<'e> {
    Ident(&'e str),
    Index(usize),
}
