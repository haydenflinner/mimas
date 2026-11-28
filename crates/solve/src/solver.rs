use crate::{
    Error, Result, Unification, UnificationError,
    components::*,
    errors::{
        AssignToConst, AssignToLoopVar, AssignToStringIndex, BareNullBinding, BoundNotSatisfied,
        CheckIsNotCallable, ExtraTupleMembers, FieldNotFound, InvalidAssignTarget, InvalidPattern,
        InvalidUseTarget, IteratingHere, MissingTupleMembers, MultipleConstDeclarations,
        MultipleFnDeclarations,
        MutateWhileIterating, NonConstantValue, NotAPact, NotFound, PactRequires, SelfOutOfContext,
        ViaRequiresImpl, ViaSignatureMismatch, ExpectedUnitValue,
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
    /// While the body of `impl Show for T`'s `show` solves: `T`, whose
    /// values display built-in there (r[impl mimas.show.self]).
    pub(crate) show_self: Option<AdtId>,
    /// f-string holes inside `impl Show for T`'s `show`, with `T`: emit
    /// renders them with `T`'s `show` held off (r[impl mimas.show.self]).
    pub(crate) built_in_holes: FxHashMap<NodeId, AdtId>,
    pub(crate) node_to_vid: FxIndexMap<NodeId, Vid>,
    pub(crate) node_decs: FxIndexMap<NodeId, DecId>,
    pub(crate) rewrites: FxIndexMap<NodeId, Expr>,
    pub(crate) raw_places: FxHashSet<NodeId>,
    /// `list.remove(key)` calls lowered from `list[key].remove()`. Without
    /// the mark the synthesized call re-enters `subscript_removal`, and a
    /// `list` that is itself `a[i]` trips the zero-argument guard.
    pub(crate) synth_removal: FxHashSet<NodeId>,
    pub(crate) place_sources: FxIndexMap<NodeId, Expr>,
    pub(crate) op_decs: FxIndexMap<NodeId, crate::OpDispatch>,
    pub(crate) propagates: FxIndexMap<NodeId, crate::Propagation>,
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
    /// `pact A: B` bounds, recorded at `hoist_pacts` when `B` may not be
    /// declared yet -- settled by `resolve_pending_impls` (mimas.pact.requires).
    pub(crate) pending_pact_requires: Vec<(PactId, Ident)>,
    /// `impl X for T via Y` decls, recorded at `hoist_callables` -- settled
    /// once every impl is registered, by `resolve_pending_impls` (mimas.impl.via).
    pub(crate) pending_vias: Vec<PendingVia>,
    /// Every pact impl as (adt, pact, decl-site): the required-pact check
    /// runs over these once all impls are registered (mimas.pact.requires).
    pub(crate) pending_required: Vec<(AdtId, PactId, Location)>,
    /// Members a `via` impl forwards: (target adt, impl'd pact, member key).
    /// `Impl::solve` counts them as covered (mimas.impl.via).
    pub(crate) via_covered: FxHashSet<(AdtId, PactId, String)>,
    /// Where the instantiation under way is written, so `substitute_params` can record an
    /// obligation against it. `None` when the caller has no use site to blame.
    pub(crate) obligation_at: Option<Location>,
    pub(crate) closure_captures: FxIndexMap<NodeId, FxIndexSet<DecId>>,
    /// Closures the generator desugar builds (`__generator(|__generator_type| body)`):
    /// one call runs the whole body, so a write to a captured parameter lasts
    /// as long as anything reads it. r[impl mimas.closure.capture-write]
    pub(crate) run_once_closures: FxHashSet<NodeId>,
    /// The pact whose `impl` block is hoisting its members, if any (C3).
    pub(crate) hoisting_pact: Option<PactId>,
    /// `S::default()` calls a field-default rewrite made: the memberwise
    /// default under `..` (`mimas.struct.field-default`).
    pub(crate) plain_default_calls: FxHashSet<NodeId>,
    /// `a..b` nodes that are a `for` head: counted loops, not `Range` values.
    pub(crate) for_ranges: FxHashSet<NodeId>,
    /// Names an associated type answers to in the pact or impl being read:
    /// `Item` → its stand-in (a pact) or its binding (an impl).
    pub(crate) assoc_scope: Vec<FxHashMap<String, Ty>>,
    /// Each pact associated type's stand-in param: (pact, name). The param
    /// is opaque in the pact's default bodies; a call site swaps it for
    /// the receiver's binding.
    pub(crate) assoc_params: FxHashMap<ParamId, (PactId, String)>,
    /// What each impl fixes an associated type to: (pact, type, name) → type.
    pub(crate) assoc_bindings: FxHashMap<(PactId, AdtId, String), Ty>,
    /// A bounded type parameter's associated type, `T::Item`: one opaque
    /// param per (owner param, stand-in param), and the reverse lookup.
    pub(crate) assoc_of_param: FxHashMap<(ParamId, ParamId), ParamId>,
    pub(crate) assoc_owner: FxHashMap<ParamId, (ParamId, ParamId)>,
    /// The shape of the `impl … for [T]`/`{K: V}` being read, its params
    /// in scope; `None` outside one.
    pub(crate) impl_shape: Option<Ty>,
    /// Pact members' own type parameters (`U` in `fn map<U>`).
    pub(crate) member_params: FxHashSet<ParamId>,
    /// Set while grafting a pact default onto an impl: member params stay.
    pub(crate) keep_member_params: bool,
    /// A `T::Item` read where `T` was still open: (owner, stand-in, the
    /// vid that stands for the item), settled after each call.
    pub(crate) assoc_pending: Vec<(Ty, ParamId, Ty)>,
    /// Each built-in-shape impl's shape, `(pact, array) → [T]`, so an
    /// associated type binding reads `T` off the concrete list.
    pub(crate) impl_shapes: FxHashMap<(PactId, AdtId), Ty>,
    /// Each pact's associated type names.
    pub(crate) pact_assoc: FxHashMap<PactId, Vec<String>>,
    /// Each impl member a pact owns: its dec and the pact (C3). A member whose
    /// name another member of the type already holds lives under the key
    /// `name@Pact` in `Adt::impls`.
    pub(crate) member_pact: FxHashMap<DecId, PactId>,
    /// `(type, name)` pairs two pacts each give the type, with no inherent
    /// member of the name: a plain `x.name()` must say which (C3).
    pub(crate) ambiguous_members: FxHashMap<(AdtId, String), Vec<PactId>>,
    /// Param bindings spelled `x: inout T`, by the binding ident's node id --
    /// `Function::solve` marks the dec each declares once it exists.
    pub(crate) inout_params: FxHashSet<NodeId>,
    /// Decs of `inout` parameters: the place each holds is lent for the call,
    /// so a closure capture or a returned `&` of one is an escape.
    /// r[impl mimas.inout.escape]
    pub(crate) inout_decs: FxHashSet<DecId>,
    /// Calls that lend a `&` argument: emit scopes the borrow's pending
    /// cleanup to the call so the loan ends when the callee returns.
    pub(crate) inout_calls: FxHashSet<NodeId>,
    /// `obj.reset()` on a `fn reset(inout self)` method: call node → the
    /// minted `__borrow_place` marker whose descriptor emit lowers as arg-0
    /// in place of the receiver's value.
    /// r[impl mimas.inout.self]
    pub(crate) self_place_mints: FxIndexMap<NodeId, Expr>,
    /// `&` call nodes a lending or keeping context claimed: an `inout`
    /// parameter's argument or a `let`'s value. Any other `&` is rejected
    /// where `borrow_place` runs.
    /// r[impl mimas.place.marker]
    pub(crate) allowed_borrows: FxHashSet<NodeId>,
    /// `__borrow_place` markers an `inout self` call minted itself: errors
    /// name the lent receiver, not a `&` the user never wrote.
    pub(crate) implicit_self_borrows: FxHashSet<NodeId>,
    /// The `let`-claimed subset of `allowed_borrows`: a kept place writes
    /// through shared storage, which a whole-binding `&` can't promise.
    pub(crate) kept_borrows: FxHashSet<NodeId>,
    /// `&` nodes the `for r in &xs` desugar minted for its element places:
    /// they describe a slot of the iterated collection, so the descriptor
    /// must not offer `remove` -- shrinking the list mid-loop is the
    /// length mutation the root freeze forbids.
    /// r[impl mimas.for.inplace]
    pub(crate) element_borrows: FxHashSet<NodeId>,
    /// `&` on a bare binding (`&c`): the descriptor loans a one-slot cell
    /// instead, so the `&` node maps to the binding's dec and the cell
    /// binding's ident until the cell's dec exists.
    pub(crate) binding_borrow_cells: FxIndexMap<NodeId, (DecId, Ident)>,
    /// Resolved `binding_borrow_cells`: `&` node → (root dec, cell dec).
    pub(crate) binding_borrows: FxIndexMap<NodeId, (DecId, DecId)>,
    /// Calls whose `&` args loan whole bindings: call node → `(binding,
    /// cell)` pairs whose write-backs emit after the call returns.
    /// r[impl mimas.inout.param]
    pub(crate) inout_writebacks: FxIndexMap<NodeId, Vec<(DecId, DecId)>>,
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
    /// `data "…"` sources the host can answer -- name -> column schema
    /// (see `load::set_data_resolver` and `frames::data_source_call`).
    /// `None` leaves `data` frames opaque rather than erroring.
    pub(crate) data_resolver: Option<crate::DataResolver>,
    /// Unit dims of `25kW`-style literals, merged from every solved ast (`NodeId`s are
    /// globally unique, so one map is collision-free).
    pub(crate) ast_quantities: FxHashMap<NodeId, shared::units::Dim>,
    /// `#1.10` literals from every file: `dec` wherever they stand.
    pub(crate) exact_literals: rustc_hash::FxHashSet<NodeId>,
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
    /// Display site (`print`/`dbg`/`warn` call, top-level echo) → the shown
    /// value's shape. Surfaces on [`Resolutions::print_shapes`](crate::Resolutions).
    pub(crate) print_shapes: FxIndexMap<shared::Location, shared::units::Shape>,
    /// f-string hole → its value's shape.
    pub(crate) hole_shapes: FxHashMap<NodeId, shared::units::Shape>,
    /// The unit the page wrote for each dimension a display names.
    pub(crate) shown_units: FxIndexMap<shared::units::Dim, shared::units::Shown>,
    /// Values a failing check's message names, with the unit to name them in.
    pub(crate) check_units: FxHashMap<NodeId, shared::units::Shown>,
    /// Numbers with units written in a check, as written.
    pub(crate) check_texts: FxHashMap<NodeId, String>,
    /// `dec` literals and products to scale into the page's units.
    pub(crate) dec_scales: FxHashMap<NodeId, f64>,
    /// Exprs whose value negates at emit — the dims pass marked them as
    /// crossing to the opposite axis convention where they land.
    pub(crate) sign_negs: FxHashSet<NodeId>,
    /// The type each bare-brace literal's slot expects (`{ x: 1 }` passed where a
    /// `Pt` goes), noted by `fulfill_ty` before the literal is solved — a struct
    /// hint makes it that struct, anything else leaves it a dict.
    /// r[impl mimas.record.context]
    pub(crate) record_hints: FxHashMap<NodeId, Ty>,
    /// Atoms (`#EmptyList`): the built-in `atom` enum, each literal's layout, the
    /// arms and matches the inferred-set checks wait on. r[impl mimas.atom.type]
    pub(crate) atoms: crate::atoms::AtomState,
    /// The annotation of each field of a non-generic struct, by the field's dec:
    /// a literal solved before `solve_types` (a `const`'s) reads its field types
    /// from here (r[impl mimas.record.const]).
    pub(crate) field_annotations: FxHashMap<DecId, parse::components::Annotation>,
    /// Int literals (and their `-`/`+`/`(…)` wrappers) that fill a `float`
    /// slot: each types, folds and runs as a float (r[impl mimas.num.literal]).
    pub(crate) int_as_float: FxHashSet<NodeId>,
    /// Numeric literals that are `dec`, exact: in a `dec` slot, or beside a
    /// `dec` in arithmetic or a comparison (r[impl mimas.dec.literal]).
    pub(crate) dec_literals: FxHashSet<NodeId>,
    /// Each `dec` literal as written (`3.50`), so it keeps its scale: an
    /// `f64` would have dropped the trailing zero.
    pub(crate) dec_texts: FxHashMap<NodeId, String>,
    /// `Name(args…)` on a named-field struct: the field each argument fills,
    /// by the call's node (r[impl mimas.struct.positional]).
    pub(crate) ctor_slots: FxHashMap<NodeId, Vec<usize>>,
    /// The same calls' field names, by argument: folding builds the struct
    /// literal from them (r[impl mimas.struct.positional]).
    pub(crate) ctor_names: FxHashMap<NodeId, Vec<String>>,
    /// Each value that fills a struct field without naming it (`V(0, -1)`,
    /// a tuple in a struct's slot) and the field it fills: an editor shows
    /// the long form, `V(x = 0, y = -1)` (r[impl app.field-names]).
    pub(crate) field_hints: Vec<(Location, String)>,
    /// `const`s whose value calls a Mimas fn that folding can't run (a loop, a
    /// list push, a native): computed once, before the program runs, by the
    /// real VM. Each is the const item's node, its dec, and its value
    /// (r[impl mimas.const.computed]).
    pub(crate) computed_consts: Vec<(NodeId, DecId, Expr)>,
    /// Dimensions of api-adt fields (`Input.dt` is `s`): `(adt name, field name)`.
    /// Populated by `install_user_adt` from `ApiVariantFields::Named` dims; the dims pass
    /// consults it when a `.field` read isn't on a source-declared struct.
    pub(crate) field_dims: FxHashMap<(String, String), shared::units::Dim>,
    /// Declared unit of each source-struct member, through `?`/`[..]`:
    /// `(struct name, field name)`. `next: index?` and `kids: [idx]` both
    /// land as `INDEX`. Recorded by the dims pass; surfaces on
    /// [`ResolvedAdt::member_dims`](crate::ResolvedAdt).
    pub(crate) member_units: FxHashMap<(String, String), shared::units::Dim>,
    /// Display shape of each source-struct member whose annotation carries
    /// units, one per tuple slot: `(struct name, field name)`. Surfaces on
    /// [`ResolvedAdt::member_shapes`](crate::ResolvedAdt).
    pub(crate) member_shapes: FxHashMap<(String, String), shared::units::Shape>,
    /// Exprs that fulfill a `[T; n]` position with a length the compiler couldn't
    /// prove -- emit wraps each in a runtime dim check (see `lens.rs`).
    /// Surfaces on [`Resolutions::len_checks`](crate::Resolutions).
    pub(crate) len_checks: FxIndexMap<NodeId, Vec<Option<usize>>>,
    /// `S::default()` calls the solver blessed: call node → the struct's adt.
    /// Emit lowers each to a `NewInstance` with every member set to its
    /// type's default rather than a call.
    pub(crate) default_ctors: FxIndexMap<NodeId, AdtId>,

    /// `multi fn` generics: the generic's dec → its per-position bound spaces
    /// and the `method` arms registered against it. r[impl mimas.multi]
    pub(crate) multis: FxHashMap<DecId, MultiData>,
    /// Each call whose callee resolved to a `multi fn`'s dec: call node → the
    /// generic's dec. Emit lowers these as runtime dispatch, not `CallDirect`.
    pub(crate) multi_calls: FxIndexMap<NodeId, DecId>,

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
    /// Each page's declared `needs {…}` authority: `(page, decl site, set)`.
    /// `page` is the `//# page:` mark name for a spliced include, or the
    /// file name for a standalone file's own text. Empty for a page that
    /// declares none -- ambient authority (`Fx::ANY`) applies there.
    /// `Resolutions::page_needs`.
    pub(crate) page_needs: Vec<(String, shared::Location, shared::Fx)>,
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
    /// The prelude's own adts by name (`Post`, `Posts`, `Stream`): a user
    /// type of the same name is a different adt, so name checks that mean
    /// the std type go through this map. r[impl mimas.group.affine.errors]
    pub(crate) prelude_adts: FxHashMap<String, AdtId>,

    /// Decs minted before the current `solve_all`. A `fn` redeclare only
    /// errors when the dec it would replace was hoisted in this same solve:
    /// a session that re-solves into one solver keeps the old silent replace,
    /// and so does a user `fn` shadowing an import or prelude name.
    pub(crate) declare_floor: usize,

    pub(crate) control_flow: ControlFlow,
    pub(crate) ribs: Ribs,
    pub(crate) loop_stack: Vec<LoopRun>,
    pub(crate) fn_stack: Vec<FnRun>,
    pub(crate) non_value: Option<NodeId>,
    /// Collections a `for` loop is iterating, pushed for the duration of each body's solve.
    pub(crate) iter_guards: Vec<IterGuard>,

    /// Calls solved while `iter_guards` was non-empty: the callee's
    /// derived writes (a grades-pass fixpoint) judge them after the
    /// walk, so each record keeps the call's arg roots and the guards
    /// that were live. r[impl mimas.for.inplace-freeze]
    pub(crate) guarded_calls: Vec<GuardedCall>,
    /// A plain argument sharing a lent root, deferred when the callee
    /// resolves: the verdict is whether the callee's derived writes
    /// reach the lent path -- `f(&w.tour, w)` is legal when `w`'s slot
    /// only reads. r[impl mimas.inout.exclusive]
    pub(crate) deferred_inout: Vec<DeferredInout>,
    /// The grades pass's derived write sets: fn dec -> (param position,
    /// access path) pairs the body may write through, calls included.
    /// `[]` names the whole param's storage. Filled by `grades::check`.
    /// r[impl mimas.fn.writes]
    pub(crate) fn_writes: FxHashMap<DecId, Vec<(u32, Vec<GuardSeg>)>>,
    /// `__fn_writes(f)` calls `sig!` emitted: call node → the arg's dec (if
    /// it resolved) and where the literal lands. Filled after the writes
    /// fixpoint. r[impl mimas.fn.writes]
    pub(crate) fn_writes_asks: FxIndexMap<NodeId, (Option<DecId>, Location)>,

    /// `for`-element bindings whose first-level member writes can never reach a stored
    /// element: the iterator was computed (`f()`, `xs.enumerate()`) or yields fresh
    /// values every pass (dict `(key, value)` pairs). The assignment path warns
    /// (r[mimas.for.elem-write]).
    pub(crate) ephemeral_elems: FxHashMap<DecId, FreshElem>,

    /// Bindings holding a place-producing pull iterator (`let it = s.live`),
    /// mapped to the root dec and path the pulls' places reach into, plus the
    /// source expr's text for guard messages. `for v in it` pushes the same
    /// freeze `for v in s.live` pushes inline.
    /// r[impl mimas.for.place-iter-freeze]
    pub(crate) producer_roots: FxHashMap<DecId, (DecId, Vec<GuardSeg>, String)>,

    /// In-scope declared type params (`fn map<T, U>`, `impl Pair` re-entering `Pair<A, B>`),
    /// innermost last. `Ty::from_annotation` resolves a bare `T` through these before ribs,
    /// units, or builtins. Entries persist only for the duration of one item's solve.
    pub(crate) type_params: Vec<FxHashMap<String, ParamId>>,
}

/// A `multi fn`'s registered state: one bound space per parameter (the runtime
/// type ids the position accepts; `None` = unconstrained) plus each `method`
/// arm in declaration order. `bounds`/`keys` fill in at `check_multis`, once
/// every impl is registered and pact-bound keys can enumerate implementers.
/// r[impl mimas.multi]
#[derive(Clone, Debug, Default)]
pub(crate) struct MultiData {
    /// The generic's declared parameter annotations, verbatim -- resolved to
    /// `bounds` by `check_multis`.
    pub bound_annotations: Vec<Option<parse::components::Annotation>>,
    /// Resolved bound space per position (`None` = any value).
    pub bounds: Vec<Option<FxHashSet<AdtId>>>,
    pub methods: Vec<PendingMethod>,
}

/// One `method` arm of a `multi fn`: its own fn dec (the body lowers like any
/// item) and, once `check_multis` resolves them, the key sets its parameters
/// match against (`None` = wildcard).
#[derive(Clone, Debug)]
pub(crate) struct PendingMethod {
    pub dec: DecId,
    /// The method's declared parameter annotations, verbatim.
    pub key_annotations: Vec<Option<parse::components::Annotation>>,
    /// Resolved key set per position (`None` = wildcard).
    pub keys: Vec<Option<FxHashSet<AdtId>>>,
    pub location: Location,
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
            show_self: None,
            built_in_holes: FxHashMap::default(),
            node_to_vid: FxIndexMap::default(),
            node_decs: FxIndexMap::default(),
            rewrites: FxIndexMap::default(),
            sign_negs: FxHashSet::default(),
            raw_places: FxHashSet::default(),
            synth_removal: FxHashSet::default(),
            place_sources: FxIndexMap::default(),
            op_decs: FxIndexMap::default(),
            propagates: FxIndexMap::default(),
            pact_ops: FxIndexMap::default(),
            closure_captures: FxIndexMap::default(),
            run_once_closures: FxHashSet::default(),
            hoisting_pact: None,
            plain_default_calls: FxHashSet::default(),
            for_ranges: FxHashSet::default(),
            assoc_scope: Vec::new(),
            assoc_params: FxHashMap::default(),
            assoc_bindings: FxHashMap::default(),
            assoc_of_param: FxHashMap::default(),
            assoc_owner: FxHashMap::default(),
            impl_shape: None,
            member_params: FxHashSet::default(),
            keep_member_params: false,
            assoc_pending: vec![],
            impl_shapes: FxHashMap::default(),
            pact_assoc: FxHashMap::default(),
            member_pact: FxHashMap::default(),
            ambiguous_members: FxHashMap::default(),
            inout_params: FxHashSet::default(),
            inout_decs: FxHashSet::default(),
            inout_calls: FxHashSet::default(),
            self_place_mints: FxIndexMap::default(),
            allowed_borrows: FxHashSet::default(),
            implicit_self_borrows: FxHashSet::default(),
            kept_borrows: FxHashSet::default(),
            element_borrows: FxHashSet::default(),
            binding_borrow_cells: FxIndexMap::default(),
            binding_borrows: FxIndexMap::default(),
            inout_writebacks: FxIndexMap::default(),
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
            data_resolver: None,
            ast_quantities: FxHashMap::default(),
            exact_literals: Default::default(),
            ast_quantity_units: FxHashMap::default(),
            unit_consts: FxHashMap::default(),
            const_dims: Vec::new(),
            node_dims: FxIndexMap::default(),
            want_dims: FxIndexMap::default(),
            print_dims: FxIndexMap::default(),
            echo_units: FxIndexMap::default(),
            print_shapes: FxIndexMap::default(),
            hole_shapes: FxHashMap::default(),
            shown_units: FxIndexMap::default(),
            check_units: FxHashMap::default(),
            check_texts: FxHashMap::default(),
            dec_scales: FxHashMap::default(),
            record_hints: FxHashMap::default(),
            atoms: Default::default(),
            field_annotations: FxHashMap::default(),
            int_as_float: FxHashSet::default(),
            dec_literals: FxHashSet::default(),
            dec_texts: FxHashMap::default(),
            ctor_slots: FxHashMap::default(),
            ctor_names: FxHashMap::default(),
            field_hints: Vec::new(),
            computed_consts: Vec::new(),
            field_dims: FxHashMap::default(),
            member_units: FxHashMap::default(),
            member_shapes: FxHashMap::default(),
            len_checks: FxIndexMap::default(),
            default_ctors: FxIndexMap::default(),
            multis: FxHashMap::default(),
            multi_calls: FxIndexMap::default(),
            dec_uses: FxIndexMap::default(),
            fn_effects: FxIndexMap::default(),
            script_effects: FxIndexMap::default(),
            script_segments: Vec::new(),
            page_needs: Vec::new(),
            warnings: Vec::new(),
            bare_module_uses: FxHashMap::default(),
            resolved_uses: FxHashMap::default(),
            in_prelude: false,
            prelude_types: FxHashSet::default(),
            prelude_adts: FxHashMap::default(),
            declare_floor: 0,
            non_value: None,
            iter_guards: vec![],
            guarded_calls: vec![],
            deferred_inout: vec![],
            fn_writes: FxHashMap::default(),
            fn_writes_asks: FxIndexMap::default(),
            ephemeral_elems: FxHashMap::default(),
            producer_roots: FxHashMap::default(),
            type_params: vec![],
            param_bounds: FxHashMap::default(),
            pact_groups: FxHashMap::default(),
            impl_member: FxHashMap::default(),
            pending_bounds: vec![],
            pending_defaults: vec![],
            bound_obligations: vec![],
            pending_pact_requires: vec![],
            pending_vias: vec![],
            pending_required: vec![],
            via_covered: FxHashSet::default(),
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

    /// The `data "…"` resolver for this solve -- the type-provider hook.
    /// `load::Prepared::solve` installs the ambient one
    /// (`load::set_data_resolver`); a host driving a bare `Solver` can set
    /// its own here.
    pub fn set_data_resolver(&mut self, resolver: Option<crate::DataResolver>) {
        self.data_resolver = resolver;
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
        // an in-place freeze covers the whole subtree of its root -- a
        // place generator's root (`for v in s.live` freezes `s`) has the
        // walked collection under it (`s.items`), so a mutating call or
        // slot write anywhere below the root breaks the walk the same
        // way `xs[i] =` does under `for r in &xs`.
        // r[impl mimas.for.place-iter-freeze]
        if let Some(guard) = self.iter_guards.iter().find(|g| {
            g.dec == dec
                && (g.path == *path || (g.inplace && path.starts_with(&g.path)))
        })
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

    /// Errors if `path` off `dec` names a place a `for` iteration freeze would
    /// refuse to lend: for a plain `for` the iterated place exactly, for
    /// `for r in &xs`'s in-place guard any `&` whose loan overlaps the
    /// frozen one -- at or under it (`&s.f` while `&s` lends), or reaching
    /// across it (`&xs` while `&xs[i]` lends). Either direction hands out a
    /// second live place into slots the first loan's elements may replace.
    /// r[impl mimas.for.inplace-freeze]
    pub(crate) fn check_iter_lend(
        &self,
        dec: DecId,
        path: &[GuardSeg],
        at: Location,
    ) -> Result<()> {
        if let Some(guard) = self.iter_guards.iter().find(|g| {
            g.dec == dec
                && (g.path == *path
                    || (g.inplace && (g.path.starts_with(path) || path.starts_with(&g.path))))
        }) {
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

    /// The root dec and path a place-producing pull iterator's places reach
    /// into: `s.live` and `s.live[k]` produce from `s`, `s.t.live` from
    /// `s.t`, and a bound pull name (`let it = s.live`) inherits its
    /// `producer_roots` entry. `None` for calls (`gen()`), whose producer
    /// can't be named and so can't be frozen.
    /// r[impl mimas.for.place-iter-freeze]
    pub(crate) fn place_iter_receiver(&mut self, e: &Expr) -> Option<(DecId, Vec<GuardSeg>)> {
        let mut recv = e;
        while let ExprKind::Grouping(g) = recv.kind() {
            recv = &g.inner;
        }
        while let ExprKind::Access(Access::Square { left, .. }) = recv.kind() {
            recv = left;
        }
        match recv.kind() {
            ExprKind::Access(Access::Dot { left, .. }) => {
                let (root, path) = crate::root_and_path(left)?;
                self.ribs.resolve(root).map(|dec| (dec, path))
            }
            ExprKind::Ident(name) => {
                let dec = self.ribs.resolve(name)?;
                self.producer_roots.get(&dec).map(|(d, p, _)| (*d, p.clone()))
            }
            _ => None,
        }
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
            Ty::Dict(..) => "dict",
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
            self.exact_literals.extend(ast.exact_literals());
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

        // the `//# page:` regions of each file: each is a namespace (C2)
        crate::components::regions::install(
            asts.iter()
                .filter_map(|ast| {
                    let file = ast.own_stmts().first()?.location().file_id();
                    Some((
                        file,
                        ast.page_marks().into_iter().map(|(s, n)| (s, n.to_string())).collect(),
                    ))
                })
                .collect(),
        );
        // a `fn` redeclare errors only against decs this solve mints -- see `declare_floor`
        self.declare_floor = self.decs.len();
        // every atom a literal writes gets its variant before any body solves
        self.prescan_atoms(&asts)?;

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
        // `pact A: B` bounds and `impl X for T via Y` forwards -- every pact is
        // declared, every impl registered
        run_phase(self, Self::resolve_pending_impls, &mut saved_ribs)?;
        lap("resolve_pending_impls");
        self.check_subscript_names()?;

        // member annotations settle before the const fixpoint so a `const`
        // holding a generic literal meets `Ty::Param`s, not the placeholder
        // vids it would otherwise pin for the decl's own annotation pass
        run_phase(self, Self::settle_type_members, &mut saved_ribs)?;
        lap("settle_type_members");

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
        // a bare name two included pages define, used where neither is the
        // use's own: name both spellings. r[impl mimas.region.bare]
        if let Some((name, at, regions)) =
            crate::components::regions::take_ambiguous().into_iter().next()
        {
            let spellings = regions
                .iter()
                .map(|r| {
                    if r.is_empty() {
                        format!("`{name}`")
                    } else {
                        format!("`{}::{name}`", crate::components::regions::spelled(r))
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(crate::errors::AmbiguousName {
                src: self.src(at),
                at: at.into(),
                name,
                spellings,
            })?;
        }

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
        // `multi fn`s: resolve bound/key annotations now that every impl is
        // registered, and check keys stay inside their bound + no arm is
        // unreachable r[impl mimas.multi.coverage]
        self.check_multis()?;
        lap("check_multis");
        // the inferred atom sets need every body: arms and matches over atoms
        crate::atoms::check(self, &asts)?;
        lap("atoms");
        // last, once every node has its type: do the units add up?
        crate::dims::check(self, &asts)?;
        lap("dims");
        // and, with decs resolved everywhere the walk reaches: the usage/effect grades
        crate::grades::check(self, &asts)?;
        lap("grades");
        // a const computed before the program runs must be pure
        self.check_computed_consts()?;
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

    /// Settle every adt's member types from their annotations ahead of
    /// the early const fixpoint. Hoisting leaves each member a bare Vid
    /// that `Struct::solve`/`Enum::solve` fulfills in `solve_types` — a
    /// `const` holding a generic literal solves earlier and pins that
    /// shared Vid (`const Q = Pair { a = 1.0, b = "x" }` froze `T` to
    /// `float`, so the annotation pass then reported the bogus
    /// "expected float but found T" on the struct decl). With the member
    /// already `Ty::Param`, `specialize` instantiates it per use like
    /// any post-`solve_types` site. Best-effort per member: an
    /// annotation that cannot resolve yet (it names a still-pending
    /// const) is left for the authoritative pass, which reports real
    /// annotation errors as before.
    fn settle_type_members(&mut self, ast: &Ast) -> Result<()> {
        Self::for_each_item(ast, |item| {
            let adt = match item.kind() {
                ItemKind::Struct(st) => st.name.query(self)?.as_adt().copied(),
                ItemKind::Enum(e) => e.head.query(self)?.as_adt().copied(),
                _ => None,
            };
            let Some(adt) = adt else { return Ok(()) };

            // the member's stored ty against its parsed annotation —
            // collected first so `from_annotation`'s &mut borrows don't
            // fight the adt reads
            let members: Vec<(Ty, Annotation)> = match item.kind() {
                ItemKind::Struct(st) => {
                    let is_tuple = self.adts[adt].is_tuple_struct();
                    st.fields
                        .iter()
                        .enumerate()
                        .filter_map(|(i, field)| {
                            let ty = if is_tuple {
                                self.adts[adt].as_tuple().members.get(i).cloned()?
                            } else {
                                self.adts[adt]
                                    .as_struct()
                                    .fields
                                    .get(&field.name.to_string())?
                                    .ty
                                    .clone()
                            };
                            Some((ty, field.annotation.clone()))
                        })
                        .collect()
                }
                ItemKind::Enum(e) => e
                    .members
                    .iter()
                    .flat_map(|(member_name, member)| match member {
                        Member::Tuple(annotations) => annotations
                            .iter()
                            .enumerate()
                            .filter_map(|(i, annotation)| {
                                self.adts[adt]
                                    .variants
                                    .get(&member_name.lexeme)?
                                    .as_tuple()
                                    .members
                                    .get(i)
                                    .cloned()
                                    .map(|ty| (ty, annotation.clone()))
                            })
                            .collect::<Vec<_>>(),
                        Member::Struct(fields) => fields
                            .iter()
                            .filter_map(|field| {
                                self.adts[adt]
                                    .variants
                                    .get(&member_name.lexeme)?
                                    .as_struct()
                                    .ty(&field.name.to_string())
                                    .map(|ty| (ty, field.annotation.clone()))
                            })
                            .collect(),
                    })
                    .collect(),
                _ => unreachable!(),
            };

            // same scope `Struct::solve`/`Enum::solve` use: the impl rib
            // makes the adt self-referential, the param scope resolves
            // `T` to its `Ty::Param`
            self.ribs.push_impl(adt);
            let generic = self.adts[adt].is_generic();
            if generic {
                self.push_adt_params(adt);
            }
            for (mut member_ty, annotation) in members {
                let Ok(mut ty) = Ty::from_annotation(annotation, self) else {
                    continue;
                };
                let _ = member_ty.fulfill_ty(&mut ty, self);
            }
            if generic {
                self.pop_type_params();
            }
            self.ribs.pop();

            // write the settled types back the way `Solve`'s
            // `normalize_members` does — the stored member must BE
            // `Ty::Param`, not a Vid bound to one, or `specialize`'s
            // `contains_params` early-out skips the substitution and
            // the use site meets the rigid param head-on
            let mut normalized = self.adts[adt].clone();
            normalized.normalize_members(self);
            self.adts[adt] = normalized;
            Ok(())
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
            if self.defer_call_consts
                && (self.calls_unsolved_fn(right) || has_atom(right) || has_keyed_dict(right))
            {
                return Ok(ty);
            }
            // r[impl mimas.const.computed]
            // a call into Mimas code that folding can't run (a loop, a push,
            // a native, a notation's reader): the const is computed once,
            // before the program runs, and must be pure (`check_computed_consts`)
            if !self.defer_call_consts
                && left.as_ident().is_some()
                // an atom has no folded literal form: a const holding one is
                // computed (r[impl mimas.atom.const])
                && (self.calls_unsolved_fn(right)
                    || self.refers_to_computed(right)
                    || has_atom(right)
                    || has_keyed_dict(right))
                && let Some(dec) = self.node_decs.get(&id).copied()
            {
                if !self.computed_consts.iter().any(|(_, d, _)| *d == dec) {
                    self.decs[dec].kind = DecKind::Global;
                    // the emitter binds the name like a `let`'s pattern
                    self.node_decs.insert(left.id(), dec);
                    if let PatKind::Ident(ident) = left.kind() {
                        self.node_decs.insert(ident.id, dec);
                    }
                    self.computed_consts.push((id, dec, right.clone()));
                }
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
                    help: None,
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
        param_dims: Vec<Option<shared::units::DimTree>>,
        return_ty: Option<Ty>,
        return_dim: Option<shared::units::DimTree>,
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
    /// A pact default's signature as an impl member: `Self` filled in, but a
    /// member's own `<U>` kept as a param, so each call still gets its own
    /// (r[impl mimas.pact.member-generic]).
    pub(crate) fn graft_pact_fn(&mut self, header: &FnHeader, self_ty: &Ty) -> Ty {
        let prev = std::mem::replace(&mut self.keep_member_params, true);
        let ty = self.instantiate_pact_fn(header, self_ty);
        self.keep_member_params = prev;
        ty
    }

    pub(crate) fn instantiate_pact_fn(&mut self, header: &FnHeader, self_ty: &Ty) -> Ty {
        let mut memo: FxHashMap<Vid, Vid> = FxHashMap::default();
        self.fresh_vids(&Ty::Fn(header.clone()), self_ty, &mut memo)
    }

    /// Normalize `t`, then replace any still-unbound vid with a fresh memoized one and any `Self`
    /// with `self_ty`. Bound vids resolve to their concrete types and stay put; only the open
    /// slots get fresh handles.
    fn fresh_vids(&mut self, t: &Ty, self_ty: &Ty, memo: &mut FxHashMap<Vid, Vid>) -> Ty {
        match t.clone().normalized(self) {
            // a pact's associated type: what the receiver's impl fixes it to;
            // a bounded parameter's own opaque `T::Item`; itself inside a
            // default body (r[impl mimas.pact.assoc-type])
            // a pact member's own `<U>`: fresh at each call, one var per
            // param, and its bound owed there (r[impl mimas.pact.member-generic])
            Ty::Param(p) if self.member_params.contains(&p) && !self.keep_member_params => {
                let key = Vid::from(u32::MAX - p.index() as u32);
                let fresh = match memo.get(&key) {
                    Some(v) => *v,
                    None => {
                        let v = self.vid();
                        memo.insert(key, v);
                        self.owe_bounds(p, &Ty::Vid(v));
                        v
                    }
                };
                Ty::Vid(fresh)
            }
            // `O::Item` of a member's own `<O: Iter>`: read through `O`'s
            // fresh var, settled once the call fixes `O`
            Ty::Param(p)
                if !self.keep_member_params
                    && self
                        .assoc_owner
                        .get(&p)
                        .is_some_and(|(owner, _)| self.member_params.contains(owner)) =>
            {
                let (owner, item) = self.assoc_owner[&p];
                let owner_ty = self.fresh_vids(&Ty::Param(owner), self_ty, memo);
                match self.assoc_for(item, &owner_ty) {
                    Some(t) => t,
                    None => {
                        let v = Ty::Vid(self.vid());
                        self.assoc_pending.push((owner_ty, item, v.clone()));
                        v
                    }
                }
            }
            Ty::Param(p) if self.assoc_params.contains_key(&p) => {
                match self.assoc_for(p, self_ty) {
                    Some(t) => t,
                    None => Ty::Vid(self.vid()),
                }
            }
            Ty::Vid(v) => Ty::Vid(*memo.entry(v).or_insert_with(|| self.vid())),
            Ty::Skolem(_) => self_ty.clone(),
            Ty::Array(inner, len) => {
                Ty::Array(Box::new(self.fresh_vids(&inner, self_ty, memo)), len)
            }
            Ty::Dict(k, v) => Ty::dict_of(
                self.fresh_vids(&k, self_ty, memo),
                self.fresh_vids(&v, self_ty, memo),
            ),
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

    /// The associated type stand-in `item` read through `owner`: the
    /// impl's binding for a concrete type, the opaque `T::Item` for a
    /// bounded param, the stand-in itself for a pact's `Self`. `None` when
    /// the owner is still open or fixes nothing.
    // r[impl mimas.pact.assoc-type]
    pub(crate) fn assoc_for(&mut self, item: ParamId, owner: &Ty) -> Option<Ty> {
        let (pid, name) = self.assoc_params.get(&item)?.clone();
        let concrete = match owner.clone().normalized(self) {
            Ty::Adt(aid, args) | Ty::Identity(aid, args) => Some((aid, args)),
            Ty::Param(p) => return Some(Ty::Param(self.param_assoc(p, item, &name))),
            Ty::Skolem(_) => return Some(Ty::Param(item)),
            Ty::Vid(_) => None,
            other => self.builtin_adt(&other).map(|a| (a, vec![])),
        };
        let (aid, args) = concrete?;
        let bound = self.assoc_bindings.get(&(pid, aid, name)).cloned()?;
        // `impl Iter for [T]`: read `T` off the concrete list
        if let Some(shape) = self.impl_shapes.get(&(pid, aid)).cloned() {
            let mut map = FxHashMap::default();
            bind_shape(&shape, &owner.clone().normalized(self), &mut map);
            return Some(self.substitute_params(&bound, &mut map));
        }
        Some(self.specialize(&bound, aid, &args))
    }

    /// Pending `T::Item`s whose `T` has settled: unify each with what
    /// `T`'s impl fixes. Runs after every call and before the bound check.
    // r[impl mimas.pact.assoc-path]
    pub(crate) fn settle_assoc(&mut self) -> std::result::Result<(), UnificationError> {
        if self.assoc_pending.is_empty() {
            return Ok(());
        }
        for (owner, item, mut v) in std::mem::take(&mut self.assoc_pending) {
            let owner = owner.normalized(self);
            match self.assoc_for(item, &owner) {
                Some(mut t) => v.fulfill_ty(&mut t, self)?,
                None if matches!(owner, Ty::Vid(_)) => self.assoc_pending.push((owner, item, v)),
                None => {}
            }
        }
        Ok(())
    }

    /// The stand-in param of the associated type `name` that one of
    /// `owner`'s bounds declares (`Item` for `T: Iter`), if any.
    pub(crate) fn assoc_named(&self, owner: ParamId, name: &str) -> Option<ParamId> {
        let bounds = self.param_bounds.get(&owner)?;
        let pacts: Vec<PactId> = bounds.iter().flat_map(|b| self.pact_closure(*b)).collect();
        self.assoc_params
            .iter()
            .find(|(_, (pid, n))| n == name && pacts.contains(pid))
            .map(|(p, _)| *p)
    }

    /// `T::Item` for a bounded param `T`: one opaque param per (owner,
    /// stand-in), minted once and named for messages.
    fn param_assoc(&mut self, owner: ParamId, item: ParamId, name: &str) -> ParamId {
        if let Some(p) = self.assoc_of_param.get(&(owner, item)) {
            return *p;
        }
        // a fresh node id keeps it clear of every declared param's id
        // (and small: param names live in a vector indexed by id)
        let fresh = ParamId::from(parse::NodeId::new().index() as u32);
        shared::name_param(fresh.index(), &format!("{owner}::{name}", owner = Ty::Param(owner)));
        self.assoc_of_param.insert((owner, item), fresh);
        self.assoc_owner.insert(fresh, (owner, item));
        fresh
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
            Ty::Dict(k, v) => {
                Ty::dict_of(self.substitute_anons(k, memo), self.substitute_anons(v, memo))
            }
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
            enum_of: None,
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
                    let mut adt = Adt::new_variant_layout(qualified.clone(), variant);
                    adt.enum_of = Some(api_adt.adt_id);
                    adt
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
                        f.return_dim.clone(),
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
                        return_dim: f.return_dim.clone(),
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
                        return_dim: m.return_dim.clone(),
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
    /// Why `ty` cannot be a dict key, or `None` when it can (or is not known yet: a type
    /// parameter is checked when the program runs). A key must keep its equality while it
    /// sits in the dict, so the containers that change in place (a list, a dict), a function
    /// (no value equality) and `()` are refused. A tuple, an option or a struct or enum is a
    /// key when its parts are; the dict stores a frozen copy of a struct key.
    /// r[impl mimas.dict.key.refused]
    pub(crate) fn key_problem(&self, ty: &Ty) -> Option<String> {
        fn walk(s: &Solver, ty: &Ty, seen: &mut Vec<AdtId>) -> Option<String> {
            match ty.clone().normalized(s) {
                Ty::Array(..) => Some(
                    "a list can change after it goes in, and the dict could not find it again; \
                     use a tuple"
                        .into(),
                ),
                Ty::Dict(..) => Some("a dict can change after it goes in".into()),
                Ty::Fn(_) => Some("a function has no equality a key could use".into()),
                Ty::Unit => Some("`()` holds no value to key by".into()),
                Ty::Tuple(members) => members.iter().find_map(|m| walk(s, m, seen)),
                Ty::Option(inner) => walk(s, &inner, seen),
                // `Self` in an impl is the adt itself
                Ty::Adt(aid, _) | Ty::Identity(aid, _) => {
                    if seen.contains(&aid) {
                        return None;
                    }
                    seen.push(aid);
                    let adt = &s.adts[aid];
                    let field_tys: Vec<Ty> = adt
                        .variants
                        .values()
                        .flat_map(|v| match v {
                            Variant::Tuple(t) => t.members.clone(),
                            Variant::Struct(st) => {
                                st.fields.values().map(|f| f.ty.clone()).collect()
                            }
                        })
                        .collect();
                    let name = adt.name.clone();
                    field_tys.iter().find_map(|t| walk(s, t, seen)).map(|why| {
                        format!("a `{name}` holds a value that cannot be a key ({why})")
                    })
                }
                _ => None,
            }
        }
        walk(self, ty, &mut Vec::new())
    }
}

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
    pub(crate) fn bind_param(&mut self, pid: ParamId, bound: &Ident) {
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
                Literal::Struct(_) => true,
                Literal::Array(items) | Literal::Tuple(items) => items
                    .iter()
                    .any(|e| matches!(e.kind(), ExprKind::Literal(l) if allocates(l))),
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

    /// `pid` plus every pact it transitively requires (`pact A: B`): the set an
    /// `A` bound grants member access to, and an `impl A for T` owes impls for.
    /// Cycle-safe -- `A: B, B: A` just yields both.
    // r[impl mimas.pact.requires]
    /// Whether `for x in v` walks `v` through `v.iter()`: a type with a
    /// one-receiver `iter` method, or a param bounded by a pact that has
    /// one (`T: Iter`). r[impl mimas.protocol.iter]
    pub(crate) fn walks_by_iter(&self, t: &Ty) -> bool {
        let one_receiver = |h: &FnHeader| h.is_method && h.parameters.len() == 1;
        match t {
            Ty::Adt(aid, _) | Ty::Identity(aid, _) => self.adts[*aid]
                .impls
                .get("iter")
                .is_some_and(|f| matches!(&f.ty, Ty::Fn(h) if one_receiver(h))),
            // a built-in shape with `impl Iter` (`str`, a dict, a list)
            Ty::Str | Ty::Dict(..) | Ty::Array(..) => self.builtin_adt(t).is_some_and(|aid| {
                self.adts[aid]
                    .impls
                    .get("iter")
                    .is_some_and(|f| matches!(&f.ty, Ty::Fn(h) if one_receiver(h)))
            }),
            Ty::Param(p) => self.param_bounds.get(p).is_some_and(|bounds| {
                bounds.iter().any(|b| {
                    self.pact_closure(*b).into_iter().any(|q| {
                        self.pacts[q].functions.get("iter").is_some_and(|(h, _)| one_receiver(h))
                    })
                })
            }),
            _ => false,
        }
    }

    pub(crate) fn pact_closure(&self, pid: PactId) -> Vec<PactId> {
        let mut seen = FxHashSet::default();
        let mut out = vec![];
        let mut stack = vec![pid];
        while let Some(p) = stack.pop() {
            if !seen.insert(p) {
                continue;
            }
            out.push(p);
            stack.extend(self.pacts[p].requires.iter().copied());
        }
        out
    }

    /// `pact A: B` bounds and `impl X for T via Y` forwards, settled once every
    /// pact is declared and every impl is registered -- a phase after
    /// `resolve_pending_bounds`.
    fn resolve_pending_impls(&mut self, _ast: &Ast) -> Result<()> {
        if self.pending_pact_requires.is_empty()
            && self.pending_vias.is_empty()
            && self.pending_required.is_empty()
        {
            return Ok(());
        }
        for (pid, ident) in std::mem::take(&mut self.pending_pact_requires) {
            let ty = self.resolve_name(&ident, ident.location)?;
            let Some(req) = ty.as_single_pact() else {
                return Err(NotAPact {
                    src: self.src(ident.location),
                    at: ident.location.into(),
                    ty: ty.to_string(),
                }
                .into());
            };
            self.pacts[pid].requires.push(req);
        }
        // forwards next: an auto-registered via pact owes its own required
        // pacts, and the impl check below runs once they're all in
        for via in std::mem::take(&mut self.pending_vias) {
            self.settle_via(&via)?;
        }
        for (adt, pid, at) in std::mem::take(&mut self.pending_required) {
            for req in self.pact_closure(pid) {
                if req != pid && !self.pact_impls.contains(&(req, adt)) {
                    return Err(PactRequires {
                        src: self.src(at),
                        at: at.into(),
                        target: self.adts[adt].name.clone(),
                        pact: self.pacts[pid].name.clone(),
                        requires: self.pacts[req].name.clone(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    /// `impl X for T via Y`: register `T`'s `Y` impl (synthesized when the
    /// block didn't write one and every `Y` member carries a default), then
    /// cover each `X` member the block omits with the `Y` member of the same
    /// name -- matching signature, `T`'s member slot filled if empty.
    // r[impl mimas.impl.via]
    fn settle_via(&mut self, via: &PendingVia) -> Result<()> {
        let ty = self.resolve_name(&via.via, via.via.location)?;
        let Some(yid) = ty.as_single_pact() else {
            return Err(NotAPact {
                src: self.src(via.via.location),
                at: via.via.location.into(),
                ty: ty.to_string(),
            }
            .into());
        };
        let adt = via.adt;

        // the impl needs a real `Y` impl to forward through. a missing one is
        // synthesized exactly like `impl Y for T {}` -- legal only while every
        // `Y` member has a default to graft.
        // r[impl mimas.impl.via.synth]
        if !self.pact_impls.contains(&(yid, adt)) {
            let missing: Vec<String> = self.pacts[yid]
                .functions
                .iter()
                .filter(|(_, (_, default))| default.is_none())
                .map(|(k, _)| crate::pact_key_base(k).to_string())
                .chain(self.pacts[yid].constants.keys().cloned())
                .collect();
            if !missing.is_empty() {
                return Err(ViaRequiresImpl {
                    src: self.src(via.via.location),
                    at: via.via.location.into(),
                    target: via.target.lexeme.clone(),
                    via: self.pacts[yid].name.clone(),
                    missing: missing.join(", "),
                }
                .into());
            }
            self.pact_impls.insert((yid, adt));
            self.pending_required.push((adt, yid, via.location));
            self.graft_pact_defaults(adt, yid, via.target.location)?;
        }

        self.ribs.push_impl(adt);
        let generic = self.adts[adt].is_generic();
        if generic {
            self.push_adt_params(adt);
        }
        let result = self.settle_via_members(via, yid);
        if generic {
            self.pop_type_params();
        }
        self.ribs.pop();
        result
    }

    /// The coverage half of `settle_via`, run under the impl rib so
    /// `impl_target_ty` answers.
    fn settle_via_members(&mut self, via: &PendingVia, yid: PactId) -> Result<()> {
        let adt = via.adt;
        let xid = via.pact;
        let target_ty = self
            .impl_target_ty()
            .expect("settle_via pushes the impl rib first");

        // every `X` member the block omits (overload keys excluded -- `via`
        // matches by name, and `mul#1` isn't a name) must name a `Y` member of
        // the same signature; what it can't cover stays for `Impl::solve`'s
        // missing-member report
        let omitted: Vec<String> = self.pacts[xid]
            .functions
            .keys()
            .chain(self.pacts[xid].constants.keys())
            .filter(|k| !k.contains('#') && !via.provided.contains(crate::pact_key_base(k)))
            .cloned()
            .collect();
        for key in omitted {
            // `X`'s declared shape for the member, `Self` := the impl target
            let x_ty = if let Some((header, _)) = self.pacts[xid].functions.get(&key) {
                let header = header.clone();
                self.instantiate_pact_fn(&header, &target_ty)
            } else {
                let ty = self.pacts[xid].constants[&key].clone();
                let mut memo = FxHashMap::default();
                self.fresh_vids(&ty, &target_ty, &mut memo)
            };
            // `Y`'s namesake, or -- for a member `X` itself defaulted -- `X`'s
            // own default, grafted the way an ordinary impl would have.
            // neither leaves it for the missing-member report
            // r[impl mimas.impl.via.provided]
            let (y_ty, member): (Ty, Option<DecId>) =
                if let Some((header, _)) = self.pacts[yid].functions.get(&key) {
                    let header = header.clone();
                    (
                        self.instantiate_pact_fn(&header, &target_ty),
                        self.pact_members.get(&(yid, key.clone())).copied(),
                    )
                } else if let Some(ty) = self.pacts[yid].constants.get(&key) {
                    let ty = ty.clone();
                    let mut memo = FxHashMap::default();
                    (self.fresh_vids(&ty, &target_ty, &mut memo), None)
                } else if self.pacts[xid]
                    .functions
                    .get(&key)
                    .is_some_and(|(_, default)| default.is_some())
                {
                    (
                        x_ty.clone(),
                        self.pact_members.get(&(xid, key.clone())).copied(),
                    )
                } else {
                    continue;
                };
            // r[impl mimas.impl.via.match]
            let mut want = x_ty;
            if y_ty.clone().fulfill_ty(&mut want, self).is_err() {
                return Err(ViaSignatureMismatch {
                    src: self.src(via.via.location),
                    at: via.via.location.into(),
                    member: key.clone(),
                    pact: self.pacts[xid].name.clone(),
                    via: self.pacts[yid].name.clone(),
                }
                .into());
            }
            // constants have no runtime dispatch: cover only when `T`'s surface
            // already holds the `Y` impl's value (a `via`-synthesized impl
            // never has consts). a `fn` slot gets the `Y` member's dec, which
            // resolves to `T`'s impl or the `Y` default
            let is_const = self.pacts[yid].constants.contains_key(&key);
            if is_const {
                if self.adts[adt].impls.contains_key(&key) {
                    self.via_covered.insert((adt, xid, key));
                }
            } else {
                if !self.adts[adt].impls.contains_key(&key)
                    && let Some(dec) = member
                {
                    self.adts[adt]
                        .impls
                        .insert(key.clone(), Field::new_constant(y_ty, via.location, dec));
                }
                self.via_covered.insert((adt, xid, key));
            }
        }
        Ok(())
    }

    /// Fill `adt`'s omitted members from the pact's defaulted members, the way
    /// `Impl::hoist`'s graft does -- for a `via`-synthesized impl, which has no
    /// impl block of its own to hang the graft on.
    // r[impl mimas.impl.via]
    fn graft_pact_defaults(&mut self, adt: AdtId, pid: PactId, at: Location) -> Result<()> {
        self.ribs.push_impl(adt);
        let generic = self.adts[adt].is_generic();
        if generic {
            self.push_adt_params(adt);
        }
        let result = (|| {
            let target_ty = self.impl_target_ty().unwrap();
            for name in self.pacts[pid]
                .functions
                .keys()
                .cloned()
                .collect::<Vec<_>>()
            {
                if self.adts[adt].impls.contains_key(&name) {
                    continue;
                }
                let header = self.pacts[pid].functions[&name].0.clone();
                let ty = self.graft_pact_fn(&header, &target_ty);
                let dec = self.pact_members[&(pid, name.clone())];
                self.adts[adt]
                    .impls
                    .insert(name, Field::new_constant(ty, at, dec));
            }
            Ok(())
        })();
        if generic {
            self.pop_type_params();
        }
        self.ribs.pop();
        result
    }

    /// The pacts a receiver of `ty` dispatches through: a bound annotation
    /// (`d: Draw + Named`), `Self` in a default body, or a declared param's bounds
    /// (`x: T` under `<T: Draw>`). `None` for a concrete type -- or an unbounded param,
    /// which has nothing to dispatch through. Each bound contributes its whole
    /// required-pact closure, so `x: A` sees `B`'s members when `pact A: B`.
    // r[impl mimas.pact.requires.reach]
    pub(crate) fn pacts_of(&self, ty: &Ty) -> Option<Vec<PactId>> {
        let bounds = match ty {
            Ty::Param(pid) => self
                .param_bounds
                .get(pid)
                .filter(|b| !b.is_empty())
                .cloned(),
            other => other.as_pacts(),
        }?;
        let mut out: Vec<PactId> = vec![];
        for pid in bounds {
            for p in self.pact_closure(pid) {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        Some(out)
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
                Ty::Pacts(pids) => pids.iter().any(|p| self.pact_closure(*p).contains(&pact)),
                Ty::Skolem(p) => self.pact_closure(*p).contains(&pact),
                Ty::Param(p) => self.param_bounds.get(p).is_some_and(|b| {
                    b.iter()
                        .any(|bound| self.pact_closure(*bound).contains(&pact))
                }),
                other => self
                    .builtin_adt(other)
                    .is_some_and(|aid| self.pact_impls.contains(&(pact, aid))),
            };
            // a post is the one type std leaves out of `AbelianGroup` on
            // purpose: say why, and what a stream of posts is
            // r[impl mimas.group.affine.posts]
            if !ok && matches!(crate::affine::kind(self, &ty), Some(crate::affine::Kind::Post)) {
                return Err(crate::errors::AffineMisuse {
                    src: self.src(at),
                    at: at.into(),
                    msg: format!("`{ty}` is a post, and posts do not add"),
                    label: format!("`{ty}` is held to `{}` here", self.pacts[pact].name),
                    help: "a post is a place, not a change, so it is no group. A stream of posts is `posts(xs)` on the posts' coordinates; a stream of the panels between them is `Δ ps`"
                        .into(),
                }
                .into());
            }
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

    /// The runtime ids an adt's values test as: a primitive's sentinel
    /// (`int` → `AdtId::PRIM_INT`), an enum's variant layouts, anything
    /// else the adt itself. Shared by `multi fn` bounds and `method` keys.
    /// r[impl mimas.multi]
    pub(crate) fn dispatch_set(&self, aid: AdtId) -> FxHashSet<AdtId> {
        let adt = &self.adts[aid];
        if let Some(p) = AdtId::primitive(&adt.name) {
            return [p].into_iter().collect();
        }
        if adt.flags.contains(AdtFlags::IS_ENUM) {
            return adt.variants.values().filter_map(|v| v.layout()).collect();
        }
        [aid].into_iter().collect()
    }

    /// The enum a variant layout adt belongs to (`V::Num`'s layout → `V`) —
    /// recorded on the layout adt when it's pushed, so this is one lookup,
    /// not a scan of every adt per unify. r[impl mimas.multi]
    pub(crate) fn enum_parent(&self, layout: AdtId) -> Option<AdtId> {
        self.adts[layout].enum_of
    }

    /// A dispatch annotation (`x: Ship`, `x: V::Num`, `x: Draw`, `x: int`) →
    /// the runtime ids it matches. `None` back means the annotation isn't a
    /// dispatchable type at all. r[impl mimas.multi]
    fn multi_set_from_annotation(
        &mut self,
        annotation: &parse::components::Annotation,
        at: Location,
    ) -> Result<Option<FxHashSet<AdtId>>> {
        use parse::components::Annotation as A;
        // a name → its dispatch ids: an adt's own set, a pact's implementers'
        // sets unioned. type params and anything else aren't dispatchable.
        let from_ident = |solver: &mut Solver, ident: &Ident| -> Result<Option<FxHashSet<AdtId>>> {
            let Some(dec) = solver.ribs.resolve(ident) else {
                return Ok(None);
            };
            match solver.decs[dec].kind {
                DecKind::Adt(aid) => Ok(Some(solver.dispatch_set(aid))),
                DecKind::Pact(pid) => {
                    let implementers: Vec<AdtId> = solver
                        .pact_impls
                        .iter()
                        .filter(|(p, _)| *p == pid)
                        .map(|(_, a)| *a)
                        .collect();
                    Ok(Some(
                        implementers
                            .into_iter()
                            .flat_map(|a| solver.dispatch_set(a))
                            .collect(),
                    ))
                }
                _ => Ok(None),
            }
        };
        match annotation {
            A::Kw(kw) => Ok(Some(
                [match kw {
                    parse::lex::TyKw::Int => AdtId::PRIM_INT,
                    parse::lex::TyKw::Float => AdtId::PRIM_FLOAT,
                    parse::lex::TyKw::Str => AdtId::PRIM_STR,
                    parse::lex::TyKw::Bool => AdtId::PRIM_BOOL,
                }]
                .into_iter()
                .collect(),
            )),
            A::Ty(ident) => from_ident(self, ident),
            // `E::V` -- exactly one variant layout of the enum
            A::Path(segments) if segments.len() == 2 => {
                let Some(dec) = self.ribs.resolve(&segments[0]) else {
                    return Ok(None);
                };
                let DecKind::Adt(aid) = self.decs[dec].kind else {
                    return Ok(None);
                };
                let adt = &self.adts[aid];
                if !adt.flags.contains(AdtFlags::IS_ENUM) {
                    return Ok(None);
                }
                let Some(variant) = adt.variants.get(&segments[1].lexeme) else {
                    Err(crate::errors::MultiBadKey {
                        src: self.src(at),
                        at: at.into(),
                        what: format!(
                            "`{}` is not a variant of `{}`",
                            segments[1].lexeme, adt.name
                        ),
                    })?
                };
                Ok(variant.layout().map(|l| [l].into_iter().collect()))
            }
            // `Pair<A, B>` -- dispatch sees the adt, never its type arguments
            A::Applied(path, _) => path.last().map_or(Ok(None), |i| from_ident(self, i)),
            _ => Ok(None),
        }
    }

    /// The `multi fn` pass: resolve every bound and key annotation (impls all
    /// registered by now), keep keys inside their bound, reject a `method`
    /// that can never run, and check each arm's `->` agrees with the
    /// generic's. Coverage itself needs no enumeration -- the `multi fn`'s
    /// body is the mandatory fallback, so every tuple reaches an arm.
    /// r[impl mimas.multi.coverage]
    fn check_multis(&mut self) -> Result<()> {
        if self.multis.is_empty() {
            return Ok(());
        }
        let generics: Vec<DecId> = self.multis.keys().copied().collect();
        for gdec in generics {
            let (bound_annotations, method_sites, gloc) = {
                let data = &self.multis[&gdec];
                (
                    data.bound_annotations.clone(),
                    data.methods
                        .iter()
                        .map(|m| (m.dec, m.key_annotations.clone(), m.location))
                        .collect::<Vec<_>>(),
                    self.decs[gdec].location,
                )
            };
            let mut bounds = Vec::with_capacity(bound_annotations.len());
            for ann in &bound_annotations {
                bounds.push(match ann {
                    None => None,
                    Some(a) => Some(self.multi_set_from_annotation(a, gloc)?.ok_or_else(|| {
                        crate::errors::MultiBadKey {
                            src: self.src(gloc),
                            at: gloc.into(),
                            what: format!("`{a}` is not a dispatchable type"),
                        }
                    })?),
                });
            }
            let gret = self
                .sub(self.decs[gdec].vid)
                .and_then(|ty| match ty {
                    Ty::Fn(h) => Some(h.return_ty.as_ref().clone()),
                    _ => None,
                })
                .unwrap_or(Ty::Unit);

            let mut seen: Vec<(Vec<Option<FxHashSet<AdtId>>>, Location)> = Vec::new();
            for (i, (mdec, key_annotations, mloc)) in method_sites.into_iter().enumerate() {
                if key_annotations.len() != bounds.len() {
                    Err(crate::errors::MultiArityMismatch {
                        src: self.src(mloc),
                        at: mloc.into(),
                        expected: bounds.len(),
                        found: key_annotations.len(),
                    })?
                }
                let mut keys = Vec::with_capacity(key_annotations.len());
                for (pos, ann) in key_annotations.iter().enumerate() {
                    let key = match ann {
                        None => None,
                        Some(a) => {
                            Some(self.multi_set_from_annotation(a, mloc)?.ok_or_else(|| {
                                crate::errors::MultiBadKey {
                                    src: self.src(mloc),
                                    at: mloc.into(),
                                    what: format!("`{a}` is not a dispatchable type"),
                                }
                            })?)
                        }
                    };
                    if let (Some(k), Some(b)) = (&key, &bounds[pos])
                        && !k.is_subset(b)
                    {
                        Err(crate::errors::MultiKeyOutOfBounds {
                            src: self.src(mloc),
                            at: mloc.into(),
                            what: ann.as_ref().unwrap().to_string(),
                        })?
                    }
                    keys.push(key);
                }
                // an arm is dead when an earlier arm's keys cover it at every
                // position (`None` covers everything). first-match order makes
                // the earlier arm win, so this one can never run.
                // r[impl mimas.multi.dominated]
                let dead = seen.iter().any(|(prev, _)| {
                    (0..keys.len()).all(|pos| match (&prev[pos], &keys[pos]) {
                        (None, _) => true,
                        (Some(_), None) => false,
                        (Some(earlier), Some(later)) => later.is_subset(earlier),
                    })
                });
                if dead {
                    Err(crate::errors::MultiMethodUnreachable {
                        src: self.src(mloc),
                        at: mloc.into(),
                        name: self.decs[gdec].name.clone(),
                    })?
                }
                // overlapping but incomparable arms: `(int, P)` and `(P, int)`
                // both match `(int, int)` when `int` implements `P`, yet
                // neither's key space holds the other -- first-match order
                // would silently pick one. That's an error: the author names
                // the overlap with a more specific arm, or disjoins the keys.
                // r[impl mimas.multi.ambiguous]
                if let Some((_, ploc)) = seen.iter().find(|(prev, _)| {
                    (0..keys.len()).all(|pos| match (&prev[pos], &keys[pos]) {
                        // co-applicable: some value exists in both sets
                        (None, _) | (_, None) => true,
                        (Some(earlier), Some(later)) => !earlier.is_disjoint(later),
                    }) && !(0..keys.len()).all(|pos| match (&prev[pos], &keys[pos]) {
                        // earlier ⊆ later pointwise: the earlier arm is simply
                        // the more specific one and wins on purpose
                        (_, None) => true,
                        (None, Some(_)) => false,
                        (Some(earlier), Some(later)) => earlier.is_subset(later),
                    })
                }) {
                    Err(crate::errors::MultiMethodAmbiguous {
                        src: self.src(mloc),
                        at: mloc.into(),
                        earlier_at: (ploc.file_id == mloc.file_id)
                            .then(|| (*ploc).into()),
                        name: self.decs[gdec].name.clone(),
                    })?
                }
                // the arm's `->` must agree with the generic's (an unwritten
                // one on either side stays open and passes)
                let mret = self
                    .sub(self.decs[mdec].vid)
                    .and_then(|ty| match ty {
                        Ty::Fn(h) => Some(h.return_ty.as_ref().clone()),
                        _ => None,
                    })
                    .unwrap_or(Ty::Unit);
                let mut mret = mret.normalized(self);
                let mut gret = gret.clone().normalized(self);
                if Unification::unify(&mut mret, &mut gret, self).is_err() {
                    Err(crate::errors::MultiReturnMismatch {
                        src: self.src(mloc),
                        at: mloc.into(),
                        found: mret.to_string(),
                        expected: gret.to_string(),
                    })?
                }
                seen.push((keys.clone(), mloc));
                let data = self.multis.get_mut(&gdec).unwrap();
                data.methods[i].keys = keys;
            }
            self.multis.get_mut(&gdec).unwrap().bounds = bounds;
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

    /// `impl Iter for [T]`: the shape's own params join the innermost
    /// type-param scope, and `Self` inside reads as the shape (`[T]`), not
    /// the library's `array` namespace. Returns the shape's type.
    // r[impl mimas.impl.builtin-shape]
    pub(crate) fn enter_impl_shape(
        &mut self,
        shape: &Option<(parse::components::Annotation, Vec<parse::components::TypeParam>)>,
    ) -> Result<Option<Ty>> {
        let Some((ann, params)) = shape else {
            self.impl_shape = None;
            return Ok(None);
        };
        let declared = self.declare_type_params(params);
        match self.type_params.last_mut() {
            Some(scope) => scope.extend(declared),
            None => self.type_params.push(declared.into_iter().collect()),
        }
        let ty = Ty::from_annotation(ann.clone(), self)?;
        self.impl_shape = Some(ty.clone());
        Ok(Some(ty))
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
            // `T::Item` read at a use site: what `T`'s own stand-in fixes it
            // to, now or once `T` settles (r[impl mimas.pact.assoc-path])
            Ty::Param(pid) if !map.contains_key(pid) && self.assoc_owner.contains_key(pid) => {
                let (owner, item) = self.assoc_owner[pid];
                let owner_ty = self.substitute_params(&Ty::Param(owner), map);
                let ty = match self.assoc_for(item, &owner_ty) {
                    Some(t) => t,
                    None => {
                        let v = Ty::Vid(self.vid());
                        self.assoc_pending.push((owner_ty, item, v.clone()));
                        v
                    }
                };
                map.insert(*pid, ty.clone());
                ty
            }
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
            Ty::Dict(k, v) => {
                Ty::dict_of(self.substitute_params(k, map), self.substitute_params(v, map))
            }
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
            // a Vid that already settled substitutes what it became: an inferred
            // `->` that unified with a declared param reads back through the
            // caller's `T` (`fn f<T>(x: T) { x }` instantiates `T -> T` per use),
            // and one that settled concrete reports the concrete type. A still
            // free Vid passes through shared -- one monotype across call sites
            // (`mimas.fn.inferred`).
            // r[impl mimas.fn.inferred]
            Ty::Vid(v) => match self.sub(*v).cloned() {
                Some(t) => self.substitute_params(&t, map),
                None => t.clone(),
            },
            Ty::Unit
            | Ty::Never
            | Ty::Null
            | Ty::Bool
            | Ty::Int
            | Ty::Float
            | Ty::Str
            | Ty::Any
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
            // Destructuring a still-unresolved type: the pattern pins the
            // vid's shape — `let (i, r) = x` makes `x` a pair, and `x`'s
            // own inference unifies with it when it lands. A desugared
            // closure pattern param (`|(i, r)| …` → `let (i, r) = __p1`)
            // hits exactly this.
            (PatKind::Tuple(idents), Ty::Vid(vid)) => {
                let members = idents.iter().map(|_| Ty::Vid(self.vid())).collect();
                let mut tuple = Ty::Tuple(members);
                let mut vid_ty = Ty::Vid(vid);
                vid_ty
                    .fulfill_ty(&mut tuple, self)
                    .map_err(|e| e.into_type_mismatch(self, pat.location()))?;
                let Ty::Tuple(tys) = tuple else {
                    unreachable!("just unified with a tuple")
                };
                idents
                    .iter()
                    .zip(tys.iter())
                    .try_for_each(|(pat, ty)| self.solve_pat(pat, ty.clone()))
            }
            // `let P { x, y = b } = p`: each field's value binds its
            // sub-pattern (C10) r[impl mimas.struct.let-pattern]
            (PatKind::Struct(head, fields), Ty::Adt(aid, args) | Ty::Identity(aid, args))
                if self.adts[aid].try_as_struct().is_some()
                    && self
                        .resolve_path_head(head)
                        .ok()
                        .and_then(|t| t.normalized(self).as_adt().copied())
                        == Some(aid) =>
            {
                let variant = self.adts[aid].try_as_struct().cloned().unwrap();
                for (name, sub) in fields {
                    let Some(field) = variant.fields.get(name) else {
                        return Err(FieldNotFound {
                            src: self.src(pat.location()),
                            at: pat.location().into(),
                            field_name: name.clone(),
                        }
                        .into());
                    };
                    let ty = self.specialize(&field.ty, aid, &args);
                    self.solve_pat(sub, ty)?;
                }
                self.shadow_expr_ty(head.id(), Ty::adt(aid), head.location())?;
                Ok(())
            }
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
                    // `#Name`: the scrutinee's enum's variant, or the `atom` one
                    // r[impl mimas.atom.pattern]
                    ExprKind::Atom(atom) => {
                        let (adt, name) = self.atom_arm(&atom.name, &ty)?;
                        (adt, Some(name))
                    }
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
                if self.subscript_assignment(stmt.id(), &Assignment { left: left.clone(), right: right.clone(), op: *op })? { return Ok(()); }
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
                        help: None,
                    })?,
                };
                // `x = v` doesn't *read* x, but the grades pass needs the target's dec to
                // count the store -- same lookup as every other ident site
                self.node_decs.insert(root.id, dec_id);

                // `t[0] = 7` on a tuple: a tuple is not written in place, so
                // say how to build the new one instead of a list mismatch
                // with an unresolved element type
                if let ExprKind::Access(Access::Square { left: target, .. }) = left.kind()
                    && let Ty::Tuple(items) = target.query(self)?.normalized(self)
                {
                    let name = target.to_string();
                    let slots = (0..items.len())
                        .map(|i| format!("{name}.{i}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    Err(crate::errors::TupleSlotWrite {
                        src: self.src(left.location()),
                        at: left.location().into(),
                        name,
                        slots,
                    })?;
                }

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

                // `b -= 1` inside a closure, where `b` is a local of an
                // enclosing body: the closure holds a copy of `b`, so the
                // write changes that call's copy and the next call starts
                // from the captured value again. The write can never be
                // seen, so it is refused. A member or slot write (`s.b = v`,
                // `xs[i] = v`) reaches the shared value and stays legal, and
                // a top-level `let` is not captured at all (closures address
                // the entry frame's slot directly).
                // r[impl mimas.closure.capture-write]
                if matches!(left.kind(), ExprKind::Ident(_))
                    && matches!(self.decs[dec_id].kind, DecKind::Local)
                    && let Some((_, crossed)) = self.ribs.resolve_with_closures(root)
                    && crossed.iter().any(|c| !self.run_once_closures.contains(c))
                {
                    Err(crate::errors::AssignToCapture {
                        src: self.src(left.location()),
                        at: left.location().into(),
                        name: root.lexeme.clone(),
                    })?
                }

                // `xs = v` rebinds a name an in-place `for r in &xs` lent
                // element places out of -- under that freeze the name IS the
                // loan, so reassigning it is the same refusal `xs.push`
                // gets. Any write at-or-above the lent place (`xs[i] = v`
                // under `for r in &xs[i]`, `xs = v` under `for r in &xs[i]`)
                // replaces the slot the loan resolves through and stales it.
                // A place generator's freeze reaches the other way too: under
                // `for v in s.live` the walked store sits under the root, so
                // `s.items = v` or `s.count = v` rebinding a field the
                // generator resolves through stales open pulls -- blocked,
                // unless the write path dips under a slot (`s.items[i].f = v`
                // is the same element member write `v.f = v` performs).
                // Writes strictly under `for r in &xs`'s lent place stay
                // consistent writes to lent storage. A plain `for` snapshots
                // its value and stays allowed. r[impl mimas.for.inplace-freeze]
                // r[impl mimas.for.place-iter-freeze]
                if let Some(guard) = self.iter_guards.iter().find(|g| {
                    g.inplace
                        && g.dec == dec_id
                        && crate::access_path(left).is_some_and(|p| {
                            g.path.starts_with(&p)
                                || (p.starts_with(&g.path)
                                    && !p[g.path.len()..]
                                        .iter()
                                        .any(|seg| matches!(seg, GuardSeg::Index)))
                        })
                }) {
                    Err(MutateWhileIterating {
                        src: self.src(left.location()),
                        at: left.location().into(),
                        name: guard.name.clone(),
                        iterating: vec![IteratingHere {
                            src: self.src(guard.iterated_at),
                            at: guard.iterated_at.into(),
                            name: guard.name.clone(),
                        }],
                    })?
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
                            let op_orig = op;
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
                                None => {
                                    // `p += q` on two posts, `p *= k` on one
                                    // r[impl mimas.group.affine.add-posts]
                                    if let Some(op) = op.or(match *op_orig {
                                        parse::AssignmentOp::StarEqual => Some(parse::EvaluationOp::Multiply),
                                        parse::AssignmentOp::SlashEqual => Some(parse::EvaluationOp::Divide),
                                        _ => None,
                                    }) {
                                        let place_n = place.clone().normalized(self);
                                        let right_n = right.query(self)?.normalized(self);
                                        if let Some(e) = crate::affine::binary(self, op, &place_n, &right_n, left.location()) {
                                            return Err(e);
                                        }
                                    }
                                    right.fulfill_ty(place, self)?
                                }
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

                // `let p = &e` keeps the place: claim the `&` for this binding
                // before it solves, since `&` is an error everywhere else.
                // A `yield &e` lowers to the same claim on its `__yield_value`
                // let. r[impl mimas.place.let] r[impl mimas.generator.place]
                let mut rhs = right;
                while let ExprKind::Grouping(g) = rhs.kind() {
                    rhs = &g.inner;
                }
                if let ExprKind::Call(c) = rhs.kind()
                    && c.left.as_ident().is_some_and(|i| {
                        i.lexeme == "__borrow_place" || i.lexeme == "__yield_place"
                    })
                {
                    self.allowed_borrows.insert(rhs.id());
                    self.kept_borrows.insert(rhs.id());
                }

                // `let ps: [__Place<T>] = [&xs[0], &xs[1]]` keeps a
                // descriptor per element when the annotation asks for
                // them. r[impl mimas.place.field]
                if let ExprKind::Literal(Literal::Array(items)) = rhs.kind()
                    && let Ty::Array(inner, _) = ty.clone().normalized(self)
                    && self.is_place_type(&inner).is_some()
                {
                    for item in items {
                        if let ExprKind::Call(c) = item.kind()
                            && c.left.as_ident().is_some_and(|i| {
                                i.lexeme == "__borrow_place" || i.lexeme == "__yield_place"
                            })
                        {
                            self.allowed_borrows.insert(item.id());
                            self.kept_borrows.insert(item.id());
                        }
                    }
                }

                right.fulfill_ty(ty.clone(), self)?;

                // `let it = s.live` remembers which root the pull iterator
                // produces places from, so `for v in it` can freeze `s`'s
                // store like `for v in s.live` does inline. Recorded after
                // `solve_pat` binds the name below.
                // r[impl mimas.for.place-iter-freeze]
                let produced = if let Ty::Fn(header) = ty.clone().normalized(self)
                    && header.parameters.is_empty()
                    && let Ty::Option(element) = header.return_ty.as_ref()
                    && self
                        .is_place_type(&element.clone().normalized(self))
                        .is_some()
                    && let Some((dec, path)) = self.place_iter_receiver(rhs)
                {
                    Some((dec, path, right.to_string()))
                } else {
                    None
                };

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

                if let Some(produced) = produced
                    && let Some(&bound) = self.node_decs.get(&left.id())
                {
                    self.producer_roots.insert(bound, produced);
                }

                // `let p = &x` where `x` is an inout param re-lends the same
                // place: the fresh binding keeps the loan's escape rules.
                // r[impl mimas.inout.escape]
                if let Some(source) = self.place_sources.get(&right.id())
                    && let Some((root, _)) = crate::root_and_path(source)
                    && self
                        .ribs
                        .resolve(root)
                        .is_some_and(|d| self.inout_decs.contains(&d))
                {
                    let mut stack = vec![left];
                    while let Some(pat) = stack.pop() {
                        match pat.kind() {
                            PatKind::Ident(_) => {
                                if let Some(&dec) = self.node_decs.get(&pat.id()) {
                                    self.inout_decs.insert(dec);
                                }
                            }
                            PatKind::Tuple(pats)
                            | PatKind::TupleVariant(_, pats)
                            | PatKind::Or(pats) => stack.extend(pats),
                            PatKind::Struct(_, fields) => stack.extend(fields.values()),
                            PatKind::NullBind(inner) => stack.push(inner),
                            _ => {}
                        }
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
            StmtKind::Needs(_) => {
                // `needs` decls are the grades pass's business — see `grades::check`.
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
                    // default bodies name the pact's associated types as
                    // their opaque stand-ins (r[impl mimas.pact.assoc-type])
                    let assoc: FxHashMap<String, Ty> = pact
                        .items
                        .iter()
                        .filter_map(|item| match item {
                            PactItem::Type { name, .. } => {
                                Some((name.lexeme.clone(), Ty::Param(Self::param_id_of(name))))
                            }
                            _ => None,
                        })
                        .collect();
                    self.assoc_scope.push(assoc);
                    for item in pact.items.iter() {
                        let PactItem::Fn {
                            default: Some(body),
                            generics,
                            parameters,
                            return_type,
                            ..
                        } = item
                        else {
                            continue;
                        };

                        // the member's own `<U>` is rigid in its default body
                        // (r[impl mimas.pact.member-generic])
                        self.push_type_params(generics);
                        self.ribs.push_function();
                        for binding in parameters {
                            let param = self.bind_parameter(binding)?;
                            self.declare(
                                binding.left.as_ident().unwrap(),
                                binding.left.id(),
                                param.ty,
                            )?;
                            if self.inout_params.contains(&binding.left.id())
                                && let Some(&dec) = self.node_decs.get(&binding.left.id())
                            {
                                self.inout_decs.insert(dec);
                            }
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
                        self.pop_type_params();
                    }
                    self.assoc_scope.pop();
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
            // `a::b` never reads a value: `page::cards()` reaches the `page`
            // module beside a `let page`. r[impl mimas.path.not-value]
            parse::ExprKind::Ident(ident)
                if self.ribs.resolve(ident).is_some_and(|d| {
                    matches!(
                        self.decs[d].kind,
                        DecKind::Local | DecKind::Global | DecKind::LoopVar
                    )
                }) =>
            {
                let is_value = |d: DecId| {
                    matches!(self.decs[d].kind, DecKind::Local | DecKind::Global | DecKind::LoopVar)
                };
                if let Some(d) = self.ribs.resolve_skipping(ident, is_value) {
                    self.check_vis(d, expr.location())?;
                    let stored = Ty::Vid(self.decs[d].vid).normalized(self);
                    return Ok(stored);
                }
                if let Some(adt) = self.library.get(&ident.lexeme).copied() {
                    return Ok(Ty::adt(adt));
                }
                Err(NotFound {
                    src: self.src(expr.location()),
                    at: expr.location().into(),
                    name: ident.lexeme.clone(),
                    help: Some(format!("`{}` is a value here; a `::` path starts at a module or a type", ident.lexeme)),
                }
                .into())
            }
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
            // `fn m(inout self)` lends the receiver's own slot: `self` reads
            // as the `__Place<Self>` descriptor it was declared with, not the
            // impl target's plain value type. r[impl mimas.inout.self]
            let ty = if ident.lexeme == "self"
                && let Some(d) = dec
                && self.inout_decs.contains(&d)
            {
                Ty::Vid(self.decs[d].vid).normalized(self)
            } else {
                ty
            };
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
                            help: direction_help(&name),
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
    /// A struct field's type, read from its annotation when the literal or
    /// constructor call that needs it comes before `solve_types` (a `const`'s
    /// value): a no-op once the field is solved, or for a generic struct.
    // r[impl mimas.record.const]
    pub(crate) fn settle_field_ty(
        &mut self,
        field_ty: &Ty,
        dec: DecId,
    ) -> std::result::Result<(), crate::UnificationError> {
        if let Ty::Vid(_) = field_ty.clone().normalized(self)
            && let Some(ann) = self.field_annotations.get(&dec).cloned()
            && let Ok(mut declared) = Ty::from_annotation(ann, self)
        {
            field_ty.clone().fulfill_ty(&mut declared, self)?;
        }
        Ok(())
    }

    /// `3`, `-1` or `(2)` filling a `float` slot is that float: mark the
    /// literal and its wrappers. An `int` variable is never widened.
    // r[impl mimas.num.literal]
    pub(crate) fn float_literal(&mut self, expr: &Expr) {
        match expr.kind() {
            ExprKind::Literal(Literal::Int(_)) => {
                self.int_as_float.insert(expr.id());
            }
            ExprKind::Unary(u)
                if matches!(u.op, parse::UnaryOp::Negative | parse::UnaryOp::Positive)
                    && is_int_literal(&u.right) =>
            {
                self.int_as_float.insert(expr.id());
                self.float_literal(&u.right);
            }
            ExprKind::Grouping(g) if is_int_literal(&g.inner) => {
                self.int_as_float.insert(expr.id());
                self.float_literal(&g.inner);
            }
            _ => {}
        }
    }

    /// Whether `ty` is the `dec` type.
    pub(crate) fn is_dec(&self, ty: &Ty) -> bool {
        matches!(ty.clone().normalized(self), Ty::Adt(id, _) if self.adts[id].name == "dec")
    }

    /// The `dec` type, once the library has registered it.
    pub(crate) fn dec_ty(&self) -> Option<Ty> {
        self.adts
            .iter()
            .find(|(_, a)| a.name == "dec")
            .map(|(id, _)| Ty::adt(id))
    }

    /// Mark a numeric literal (through `-` and parentheses) as a `dec`.
    // r[impl mimas.dec.literal]
    pub(crate) fn dec_literal(&mut self, expr: &Expr) {
        match expr.kind() {
            ExprKind::Literal(Literal::Int(_) | Literal::Float(_)) => {
                self.dec_literals.insert(expr.id());
                let loc = expr.location();
                if let Some(src) = self.sources.get(&loc.file_id) {
                    let (a, n) = (loc.span.start(), loc.span.len());
                    if let Some(text) = src.inner().get(a..a + n) {
                        self.dec_texts.insert(expr.id(), text.to_string());
                    }
                }
            }
            ExprKind::Unary(u)
                if matches!(u.op, parse::UnaryOp::Negative | parse::UnaryOp::Positive) =>
            {
                self.dec_literal(&u.right)
            }
            ExprKind::Grouping(g) => self.dec_literal(&g.inner),
            // `2.99 + 4.50 + 0.75` in a `dec` slot: every number in it is a `dec`
            ExprKind::Evaluation(ev) if is_num_tree(expr) => {
                self.dec_literal(&ev.left);
                self.dec_literal(&ev.right);
            }
            _ => {}
        }
    }

    /// `price + 0.25`, `price == 0.30`: a numeric literal beside a `dec`
    /// is a `dec` too, so the arithmetic stays exact.
    // r[impl mimas.dec.literal]
    pub(crate) fn dec_literal_beside(&mut self, left: &Expr, right: &Expr) -> crate::Result<()> {
        for (lit, other) in [(right, left), (left, right)] {
            // `#0.1 + 0.2`: an exact literal counts as a `dec` beside it
            let exact = |e: &Expr| match e.kind() {
                ExprKind::Grouping(g) => self.exact_literals.contains(&g.inner.id()),
                _ => self.exact_literals.contains(&e.id()),
            };
            if is_num_literal(lit) && !exact(lit) && (!is_num_literal(other) || exact(other)) {
                let ty = other.query(self)?;
                if self.is_dec(&ty) {
                    self.dec_literal(lit);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn hint_records(&mut self, expr: &Expr, ty: &Ty) {
        // a numeric literal in a `dec` slot is that decimal, exactly
        // r[impl mimas.dec.literal]
        if is_num_tree(expr) {
            let dec_slot = match ty.clone().normalized(self) {
                Ty::Option(inner) => self.is_dec(&inner),
                t => self.is_dec(&t),
            };
            if dec_slot {
                self.dec_literal(expr);
                return;
            }
        }
        // a whole-number literal in a `float` slot is that float
        // r[impl mimas.num.literal]
        if is_int_literal(expr) {
            let float_slot = match ty.clone().normalized(self) {
                Ty::Float => true,
                Ty::Option(inner) => matches!(*inner, Ty::Float),
                _ => false,
            };
            if float_slot {
                self.float_literal(expr);
            }
            return;
        }
        match expr.kind() {
            // an atom fits the enum its slot expects (r[impl mimas.atom.enum])
            ExprKind::Literal(Literal::Record(_) | Literal::Tuple(_)) | ExprKind::Atom(_) => {
                self.record_hints
                    .entry(expr.id())
                    .or_insert_with(|| ty.clone());
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
            // a closure in a slot of a known fn type learns its parameter
            // types from it, as a call argument does: a struct field
            // (`Btn { on: |me| me.label }`), a `let f: (Btn) -> str`, a
            // return value (r[impl mimas.closure.expected])
            ExprKind::Closure(c) if !self.node_visits.contains(&expr.id()) => {
                let want = match ty.clone().normalized(self) {
                    Ty::Option(inner) => inner.normalized(self),
                    t => t,
                };
                if let Ty::Fn(header) = want {
                    self.expected_closures
                        .entry(expr.id())
                        .or_insert(header.parameters);
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
        self.reduce_const_in(expr, &env)
            .map_err(|e| e.into_diag(self))
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
                    .or_else(|| self.const_struct_call(e, env))
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
            // r[impl mimas.num.literal]
            Some(Literal::Int(n)) if self.int_as_float.contains(&expr.id()) => {
                Ok(Some(Literal::Float(n as f64)))
            }
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
            Ok(self.reduce_const_in(e, env)?.map(|l| match l {
                // a record is typed by its slot: keep the node the checker
                // typed, or the emitter can't tell the struct from a dict
                // (r[impl mimas.record.const])
                l @ (Literal::Record(_) | Literal::Tuple(_)) => {
                    Expr::with_id(l.into(), e.id(), e.location())
                }
                l => l.into_expr(),
            }))
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
            Literal::Record(fields) => {
                let mut out = Vec::with_capacity(fields.len());
                for (key, value) in &fields {
                    let Some(lit) = reify(value)? else {
                        return Ok(None);
                    };
                    out.push((key.clone(), lit));
                }
                Ok(Some(Literal::Record(out)))
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
    fn mimas_callee<'e>(&self, call: &'e Call) -> Option<(DecId, &ConstItem, Option<&'e Expr>)> {
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
            Access::Dot {
                left,
                kind: AccessKind::Direct,
                ..
            } => self.is_const_expr_in(left, scope),
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
            DecKind::Item { defaults, .. } => defaults.get(offset..).unwrap_or(&[]),
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
    /// Whether `expr` reads a computed const (`mimas.const.computed`): a const
    /// built from one is computed too, after it.
    fn refers_to_computed(&self, expr: &Expr) -> bool {
        struct Probe<'s> {
            solver: &'s Solver,
            hit: bool,
        }
        impl parse::visit::Visitor for Probe<'_> {
            fn expr(&mut self, expr: &Expr) {
                if let ExprKind::Ident(i) = expr.kind()
                    && let Some(dec) = self
                        .solver
                        .node_decs
                        .get(&i.id)
                        .or(self.solver.node_decs.get(&expr.id()))
                    && self.solver.computed_consts.iter().any(|(_, d, _)| d == dec)
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

    /// A computed const runs before the page does, so it may not touch the
    /// world. Its value's calls are followed into every Mimas fn they reach:
    /// a native contributes its declared effects; an unannotated method on a
    /// built-in value (`str`, a list, a dict, a number, a `bool`) is pure,
    /// since the std methods with effects declare them (`chance`, `shuffle`);
    /// any other unannotated native, or a call through a value, fails closed.
    // r[impl mimas.const.computed]
    pub(crate) fn check_computed_consts(&self) -> Result<()> {
        for (_, _, value) in &self.computed_consts {
            let mut seen = FxHashSet::default();
            let (fx, reasons) = self.computed_effects(value, &mut seen);
            if !(fx & shared::Fx::ANY).is_empty() {
                let effects = if reasons.is_empty() {
                    format!("{:?}", fx & shared::Fx::ANY)
                } else {
                    // name the impl(s) that keep this out of `const` -- the
                    // author should know which one to reach for `let` over
                    reasons.join(", ")
                };
                Err(crate::errors::ImpureConst {
                    src: self.src(value.location()),
                    at: value.location().into(),
                    effects,
                    help: "compute it in a `let` instead: a `let` runs with the page and may use effects".into(),
                })?
            }
        }
        Ok(())
    }

    fn computed_effects(
        &self,
        expr: &Expr,
        seen: &mut FxHashSet<DecId>,
    ) -> (shared::Fx, Vec<String>) {
        struct Walk<'s, 'h> {
            solver: &'s Solver,
            seen: &'h mut FxHashSet<DecId>,
            fx: shared::Fx,
            reasons: Vec<String>,
        }
        impl parse::visit::Visitor for Walk<'_, '_> {
            fn expr(&mut self, call_expr: &Expr) {
                let ExprKind::Call(call) = call_expr.kind() else {
                    return;
                };
                let solver = self.solver;
                if solver.ctor_names.contains_key(&call_expr.id()) {
                    return; // a struct built from values does no work
                }
                let callee = match call.left.kind() {
                    ExprKind::Access(parse::Access::Dot { right, .. }) => {
                        right.as_ident().map(|i| i.id)
                    }
                    ExprKind::Access(parse::Access::DoubleColon { right, .. }) => Some(right.id),
                    _ => Some(call.left.id()),
                };
                let Some(dec) = callee.and_then(|id| solver.node_decs.get(&id).copied()) else {
                    self.fx |= shared::Fx::unknown();
                    return;
                };
                if let Some(binding) = solver.dec_to_native.get(&dec) {
                    let value_method = binding.sig.recv.as_ref().is_some_and(|r| {
                        matches!(
                            r.clone().normalized(solver),
                            Ty::Str
                                | Ty::Int
                                | Ty::Float
                                | Ty::Bool
                                | Ty::Array(..)
                                | Ty::Dict(..)
                                | Ty::Option(_)
                        )
                    });
                    self.fx |= binding.sig.effects.unwrap_or(if value_method {
                        shared::Fx::pure()
                    } else {
                        shared::Fx::unknown()
                    });
                    return;
                }
                match solver.decs[dec].kind {
                    DecKind::Adt(_) | DecKind::Variant { .. } | DecKind::Constant(_) => {}
                    // a Mimas fn: follow it once; a fn with no body here is a
                    // host fn the checker can't see into, so it fails closed --
                    // unless it's a pact member: the page is a closed world, so
                    // every implementer is known and the call's effects are the
                    // union of theirs. r[impl mimas.const.pact-fx]
                    DecKind::Item { .. } => match solver.const_items.get(&dec) {
                        Some(item) => {
                            if self.seen.insert(dec) {
                                let (fx, reasons) =
                                    solver.computed_effects(&item.body, self.seen);
                                self.fx |= fx;
                                self.reasons.extend(reasons);
                            }
                        }
                        None if solver.pact_members.values().any(|&m| m == dec) => {
                            for (mdec, aname) in solver.member_impl_decs(dec) {
                                match solver.fn_effects.get(&mdec) {
                                    Some(fx) if (*fx & shared::Fx::ANY).is_empty() => {}
                                    Some(fx) => {
                                        self.fx |= *fx;
                                        self.reasons.push(format!(
                                            "`{name}` on `{aname}`",
                                            name = solver.decs[dec].name,
                                        ));
                                    }
                                    None => self.fx |= shared::Fx::unknown(),
                                }
                            }
                        }
                        None => self.fx |= shared::Fx::unknown(),
                    },
                    _ => self.fx |= shared::Fx::unknown(),
                }
            }
        }
        let mut walk = Walk {
            solver: self,
            seen,
            fx: shared::Fx::pure(),
            reasons: Vec::new(),
        };
        parse::visit::walk_expr(expr, &mut walk);
        (walk.fx, walk.reasons)
    }

    /// Every method dec an implementer grafted for `member` (a pact member's
    /// dec): `(method dec, implementer name)` per impl. Empty when the pact
    /// has no impls -- the call then has no effects to union.
    // r[impl mimas.const.pact-fx]
    pub(crate) fn member_impl_decs(&self, member: DecId) -> Vec<(DecId, String)> {
        let Some((&pid, key)) = self
            .pact_members
            .iter()
            .find(|(_, d)| **d == member)
            .map(|((p, k), _)| (p, k))
        else {
            return vec![];
        };
        self.pact_impls
            .iter()
            .filter(|(p, _)| *p == pid)
            .flat_map(|(_, aid)| {
                let adt = &self.adts[*aid];
                let aname = adt.name.clone();
                adt.impls
                    .iter()
                    .map(|(name, f)| (name, f.dec))
                    .chain(
                        adt.overloads
                            .iter()
                            .flat_map(|(n, fs)| fs.iter().map(move |f| (n, f.dec))),
                    )
                    .filter(move |(name, dec)| {
                        // overloads graft under their member key (`mul#1`), so
                        // `impl_member` identifies them; a plain member matches
                        // by name
                        self.impl_member
                            .get(dec)
                            .is_some_and(|ms| ms.contains(&member))
                            || **name == *key
                    })
                    .map(move |(_, d)| (d, aname.clone()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

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
        if call.arguments.len() != arity || call.arguments.iter().any(|a| a.name.is_some()) {
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

    /// `Name(args…)` on a named-field struct, every argument const: the struct
    /// literal it builds. The call node types as the struct, so it stands as
    /// the literal's name, as a field constructor's does.
    // r[impl mimas.struct.positional]
    fn const_struct_call(&self, expr: &Expr, env: &ConstEnv) -> Option<Literal> {
        let ExprKind::Call(call) = expr.kind() else {
            return None;
        };
        let names = self.ctor_names.get(&expr.id())?;
        let mut fields = Vec::with_capacity(names.len());
        for (arg, name) in call.arguments.iter().zip(names) {
            let lit = self.reduce_const_in(&arg.value, env).ok()??;
            let value = match lit {
                // typed by its slot: keep the node the checker typed
                l @ (Literal::Record(_) | Literal::Tuple(_)) => {
                    Expr::with_id(l.into(), arg.value.id(), arg.value.location())
                }
                l => l.into_expr(),
            };
            fields.push((parse::FieldKey::Ident(Ident::synthetic(name)), value));
        }
        Some(Literal::Struct(parse::StructLiteral {
            name: expr.clone(),
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
            ExprKind::Ident(_) => self
                .node_decs
                .get(&expr.id())
                .is_some_and(|&dec| scope.decs.contains(&dec) || self.decs[dec].kind.is_constant()),
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
                // `Vec2(0, -1)` on a named-field struct (r[mimas.struct.positional])
                let struct_call = self.ctor_names.contains_key(&expr.id())
                    && call
                        .arguments
                        .iter()
                        .all(|a| self.is_const_expr_in(&a.value, scope));
                // `f(const…)` where `f`'s body is itself a const expression
                ctor || struct_call || self.is_const_mimas_call(call, scope)
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
        // `impl Iter for [T]`: `Self` is the shape the impl names
        if let Some(shape) = &self.impl_shape
            && self.builtin_adt(shape) == Some(aid)
        {
            return Some(shape.clone());
        }
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
        // vid and SelfOutOfContext fires at first use. `_self` counts too: it is the receiver
        // under a deliberately-unread name.
        // `fn f(inout self)` lends the receiver's slot: the annotation arm
        // below fixes `self` to `__Place<Self>`, so the `Self` fix would fight
        // it. `Self` resolves to the impl target inside the annotation either
        // way. r[impl mimas.inout.self]
        let inout_self =
            ident.is_receiver() && matches!(annotation, Some(Annotation::Inout(..)));
        if ident.is_receiver()
            && !inout_self
            && let Some(mut target) = self.impl_target_ty()
        {
            ty.fulfill_ty(&mut target, self)
                .map_err(|e| e.into_type_mismatch(self, ident.location))?;
        }

        // `x: inout T` declares the convention: the callee holds the `__Place<T>`
        // descriptor the caller lent with `&`, so the param's own type is that
        // descriptor -- place_read / place_assignment then route the body's
        // reads and writes through it like any other place binding.
        // r[impl mimas.inout.param]
        let annotation = annotation.clone().map(|a| match a {
            Annotation::Inout(kw, inner) => {
                self.inout_params.insert(left.id());
                let place = Ident::new("__Place".to_string(), left.location());
                // `inout self` synthesized `Self` as the inner type, but `Self`
                // is not a resolvable name in an annotation -- the impl
                // target's own name is, and inside its impl the bare name reads
                // as `Self` already.
                let inner = if ident.is_receiver()
                    && let Some(aid) = self.impl_target()
                {
                    Annotation::Ty(Ident::new(
                        self.adts[aid].name.clone(),
                        kw.location,
                    ))
                } else {
                    *inner
                };
                Annotation::Applied(vec![place], vec![inner])
            }
            a => a,
        });
        if let Some(annotation) = annotation {
            let mut target = Ty::from_annotation(annotation, self)?;
            ty.fulfill_ty(&mut target, self)
                .map_err(|e| e.into_type_mismatch(self, ident.location))?;
        }
        let has_default = if let Some(right) = right {
            if matches!(binding.annotation, Some(Annotation::Inout(..))) {
                // a loan has no value of its own to fall back on
                return Err(crate::errors::InoutDefault {
                    src: self.src(right.location()),
                    at: right.location().into(),
                }
                .into());
            }
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

        let region = crate::components::regions::of(ident.location);
        // across `//# page:` regions one name may be declared once per region
        // (C2): only a redeclare in the same file and region is the error
        // r[impl mimas.region.bare]
        let prev_here = if crate::components::regions::any() {
            self.ribs
                .module()
                .in_region(&ident.lexeme, &region)
                .map(|(_, d)| d)
                .filter(|d| self.decs[*d].location.file_id() == ident.location.file_id())
        } else {
            self.ribs.module_lookup(ident)
        };
        if let Some(prev_id) = prev_here {
            let prev = &self.decs[prev_id];
            let prev_location = prev.location;
            if constant && prev.kind.is_constant() {
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
            // a second `fn` in one scope reads like an overload but is a
            // silent replace -- name it. a user `fn map` still shadows the
            // shipped `map`: the replaced dec carries the prelude's file id,
            // or was minted by an earlier `solve` into this same solver, so
            // only a same-solve user-vs-user redeclare trips. (papercuts:
            // "fn g is already defined at line N")
            if !constant
                && matches!(prev.kind, DecKind::Item { .. })
                && !self.in_prelude
                && prev.location.file_id() != shared::PRELUDE_FILE
                && prev_id.index() >= self.declare_floor
            {
                let line = {
                    let src = self.src(prev_location);
                    let start: usize = prev_location.span().start();
                    src.inner().as_bytes()[..start.min(src.inner().len())]
                        .iter()
                        .filter(|b| **b == b'\n')
                        .count()
                        + 1
                };
                // a `notation n` lowers to `fn __lit_n`: say it in the author's word
                // r[impl mimas.region.macro]
                if let Some(tag) = ident.lexeme.strip_prefix("__lit_") {
                    Err(crate::errors::MultipleNotationDeclarations {
                        src: self.src(ident.location),
                        name: tag.to_string(),
                        at: ident.location.into(),
                        line,
                    })?;
                }
                Err(MultipleFnDeclarations {
                    src: self.src(ident.location),
                    name: ident.lexeme.clone(),
                    at: ident.location.into(),
                    line,
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
        self.export_item(&ident.lexeme, &region, dec_id);

        if let Some(node_id) = node_id {
            self.node_decs.insert(node_id, dec_id);
        }

        Ok(ty)
    }

    /// Whether the adt implements the pact named `pact` (`Show`, `Ord`).
    pub(crate) fn implements_named(&self, aid: AdtId, pact: &str) -> bool {
        self.pacts
            .iter()
            .any(|(pid, p)| p.name == pact && self.pact_impls.contains(&(pid, aid)))
    }

    /// Whether a value of `ty` writes itself through `Show` (a user adt that
    /// implements it; the built-in scalars keep their own display).
    pub(crate) fn shows(&self, ty: &Ty) -> bool {
        // inside `T`'s own `show`, a `T` writes its built-in form
        // (r[impl mimas.show.self])
        matches!(ty, Ty::Adt(aid, _) | Ty::Identity(aid, _)
            if self.implements_named(*aid, "Show") && self.show_self != Some(*aid))
    }

    /// Lists a hoisted item among its module's exports. An item of a named
    /// `//# page:` region also exports as `region::name` (`button-kit::click`),
    /// and never takes the bare name from the page's own item.
    pub(crate) fn export_item(&mut self, name: &str, region: &str, dec_id: DecId) {
        let module = self.ribs.current_module();
        let page_holds = |s: &Self| {
            s.module_items
                .get(&module)
                .and_then(|m| m.get(name))
                .is_some_and(|d| crate::components::regions::of(s.decs[*d].location).is_empty())
        };
        if !region.is_empty() && region != crate::components::regions::OUTSIDE {
            self.module_items
                .entry(module)
                .or_default()
                .insert(format!("{region}::{name}"), dec_id);
            if page_holds(self) {
                return;
            }
        }
        self.module_items
            .entry(module)
            .or_default()
            .insert(name.to_string(), dec_id);
    }

    /// The "ends in a value, not `()`" error, aimed at the block's last
    /// line — that is where the missing `;` goes.
    // r[impl mimas.block.semicolon]
    pub(crate) fn unit_block_error(&self, body: &Expr, ty: Ty) -> Error {
        let tail = match body.kind() {
            ExprKind::Block(blk) => blk.yielded_expr.as_ref().unwrap_or(body),
            _ => body,
        };
        ExpectedUnitValue {
            src: self.src(tail.location()),
            at: tail.location().into(),
            ty: ty.to_string(),
        }
        .into()
    }

    /// A `()`-expected body whose tail yields a settled value is almost
    /// always a missing `;` — name it, and point at that line. `()`,
    /// `!` and still-open types fulfill `()` the usual way.
    pub(crate) fn fulfill_unit_block(&mut self, body: &Expr) -> Result<()> {
        let ty = body.query(self)?.normalized(self);
        if matches!(ty, Ty::Unit | Ty::Never | Ty::Vid(_)) {
            body.fulfill_ty(Ty::Unit, self)
        } else {
            Err(self.unit_block_error(body, ty))
        }
    }

    pub(crate) fn run_loop_body(&mut self, body: &Expr) -> Result<LoopRun> {
        self.loop_stack.push(LoopRun {
            collect_ty: None,
            break_ty: None,
        });
        self.fulfill_unit_block(body)?;
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
    /// Per-slot dimension claim from unit-carrying `MimasType`s, parallel to `params` —
    /// a `DimTree` may nest inside container params (`[px]`, `(px↔, px↕)`). The dims
    /// pass checks call args against it; `None` leaves the slot unchecked.
    pub param_dims: Vec<Option<shared::units::DimTree>>,
    pub return_ty: Option<Ty>,
    /// Dimension claim of the return value, when it measures something — a
    /// `DimTree` so a `(PxX, PxY)`-typed native can promise `(px↔, px↕)`.
    pub return_dim: Option<shared::units::DimTree>,
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
    /// `for r in &xs` holds live places into the collection, so its root
    /// freeze covers rebinding the name too (`xs = other`): a plain `for`
    /// snapshots the value and doesn't care. The mutating-call and index
    /// checks apply to both kinds regardless.
    /// r[impl mimas.for.inplace-freeze]
    pub inplace: bool,
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

/// A call that solved while `for` freezes were live: everything the
/// grades pass needs to ask "could this call write a frozen place".
/// `arg_roots[i]` is the (root dec, access path) for param slot `i` --
/// the receiver counts as slot 0 on `x.m(..)`, like `call_args`.
/// r[impl mimas.for.inplace-freeze]
#[derive(Clone)]
pub(crate) struct GuardedCall {
    pub at: Location,
    /// The callee's dec when the call's left resolved to one -- a user
    /// fn, a `let`-bound closure, a native. `None` for a callee that's a
    /// plain value (a fn-typed param, an access result).
    pub callee: Option<DecId>,
    /// The (root dec, access path) of each argument, aligned to the
    /// callee's parameter slots -- `None` for args that name no place.
    pub arg_roots: Vec<Option<(DecId, Vec<GuardSeg>)>>,
    pub guards: Vec<IterGuard>,
}

/// A plain `inout`-sibling argument the pairwise check couldn't settle:
/// same root as a lend, so the grades pass's derived writes on its
/// param slot decide whether the pair actually aliases.
/// r[impl mimas.inout.exclusive]
#[derive(Clone)]
pub(crate) struct DeferredInout {
    /// The plain argument's site -- where the error points when it fires.
    pub at: Location,
    /// The callee's fn dec -- `deferred_inout` only records calls that
    /// resolved to one; unresolved callees errored at solve.
    pub callee: Option<DecId>,
    /// The plain argument's parameter position.
    pub slot: usize,
    /// The lent sibling's access path off the shared root.
    pub lent_path: Vec<GuardSeg>,
    /// The shared root's name, for the error.
    pub lent_name: String,
}

/// Why a `for` loop's element binding holds a fresh value each pass -- the wording a
/// member-write warning uses to say *which* production made it fresh.
#[derive(Debug, Clone)]
pub(crate) enum FreshElem {
    /// A non-place iterator (`f()`, `xs.enumerate()`) produced the element.
    Iterated(String),
    /// A dict yields a fresh `(key, value)` pair even when the dict itself is a place.
    DictPair(String),
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
            ExprKind::Propagate(d) => cur = &d.expr,
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

/// An `impl X for T via Y` decl recorded at hoist: the target and pact it
/// implements, the via pact it forwards from, the member names its block
/// writes itself (all other `X` members must name a `Y` member of the same
/// signature, mimas.impl.via), and the decl site for errors.
#[derive(Debug, Clone)]
pub(crate) struct PendingVia {
    pub adt: AdtId,
    pub pact: PactId,
    pub via: Ident,
    pub provided: FxHashSet<String>,
    pub target: Ident,
    pub location: Location,
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

/// An int literal, possibly signed or parenthesized: `3`, `-1`, `(2)`.
/// Arithmetic made only of numbers written out: `2.99 + 4.50 * 2`.
fn is_num_tree(expr: &Expr) -> bool {
    match expr.kind() {
        ExprKind::Evaluation(ev) => {
            matches!(
                ev.op,
                parse::EvaluationOp::Plus
                    | parse::EvaluationOp::Minus
                    | parse::EvaluationOp::Multiply
                    | parse::EvaluationOp::Divide
            ) && is_num_tree(&ev.left)
                && is_num_tree(&ev.right)
        }
        ExprKind::Grouping(g) => is_num_tree(&g.inner),
        ExprKind::Unary(u) => {
            matches!(u.op, parse::UnaryOp::Negative | parse::UnaryOp::Positive)
                && is_num_tree(&u.right)
        }
        _ => is_num_literal(expr),
    }
}

/// A number written out: `3`, `0.25`, `-1.5`, `(2)`.
fn is_num_literal(expr: &Expr) -> bool {
    match expr.kind() {
        ExprKind::Literal(Literal::Int(_) | Literal::Float(_)) => true,
        ExprKind::Unary(u) => {
            matches!(u.op, parse::UnaryOp::Negative | parse::UnaryOp::Positive)
                && is_num_literal(&u.right)
        }
        ExprKind::Grouping(g) => is_num_literal(&g.inner),
        _ => false,
    }
}

fn is_int_literal(expr: &Expr) -> bool {
    match expr.kind() {
        ExprKind::Literal(Literal::Int(_)) => true,
        ExprKind::Unary(u) => {
            matches!(u.op, parse::UnaryOp::Negative | parse::UnaryOp::Positive)
                && is_int_literal(&u.right)
        }
        ExprKind::Grouping(g) => is_int_literal(&g.inner),
        _ => false,
    }
}

/// `up 50px` used to spell a direction in front; `50px↑` spells it
/// after (`50px@word` for a direction with no glyph). An unresolved
/// name that is a direction word gets that note as its help.
fn direction_help(name: &str) -> Option<String> {
    shared::units::direction(name).map(|_| {
        match shared::units::builtin_direction_glyph(name) {
            Some(g) => format!("directions follow their operand — write `x{g}`, not `{name} x`"),
            None => format!("directions follow their operand — write `x@{name}`, not `{name} x`"),
        }
    })
}

/// Whether an expression writes an atom literal anywhere (`#Fixed`, `[#A, #B]`).
/// Whether `expr` holds a dict literal with value keys (`{1: "a"}`), which
/// the parser lowers to a block that fills `{:}` (`__dict_N`). A const
/// holding one is computed once, like a call (r[impl mimas.dict.key.literal]).
fn has_keyed_dict(expr: &Expr) -> bool {
    struct Find(bool);
    impl parse::Visitor for Find {
        fn expr(&mut self, e: &Expr) {
            if let ExprKind::Block(b) = e.kind()
                && let Some(first) = b.body.first()
                && let StmtKind::Let(l) = first.kind()
                && let PatKind::Ident(i) = l.left.kind()
                && i.lexeme.starts_with("__dict_")
            {
                self.0 = true;
            }
        }
    }
    let mut find = Find(false);
    parse::walk_expr(expr, &mut find);
    find.0
}

fn has_atom(expr: &Expr) -> bool {
    struct Find(bool);
    impl parse::Visitor for Find {
        fn expr(&mut self, e: &Expr) {
            self.0 |= matches!(e.kind(), ExprKind::Atom(_));
        }
    }
    let mut find = Find(false);
    parse::walk_expr(expr, &mut find);
    find.0
}

/// Match a built-in shape (`[T]`, `{K: V}`) against a concrete type,
/// recording what each of its params stands for.
fn bind_shape(shape: &Ty, ty: &Ty, map: &mut FxHashMap<ParamId, Ty>) {
    match (shape, ty) {
        (Ty::Param(p), t) => {
            map.insert(*p, t.clone());
        }
        (Ty::Array(a, _), Ty::Array(b, _)) => bind_shape(a, b, map),
        (Ty::Dict(ak, av), Ty::Dict(bk, bv)) => {
            bind_shape(ak, bk, map);
            bind_shape(av, bv, map);
        }
        _ => {}
    }
}
