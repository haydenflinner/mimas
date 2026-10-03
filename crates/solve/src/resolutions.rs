#![allow(unused)] // temp

use std::{collections::HashMap, sync::Arc};

use api::NativeId;
use indexmap::IndexMap;
use miette::NamedSource;
use parse::{Literal, NodeId};
use shared::{FileId, IdVec, Location, PactId, ParamId, TyNames};

use crate::{
    Solver,
    components::{Adt, AdtFlags, AdtId, DecId, DecKind, Ty, TyExt, Variant, Vis},
    grades::UseInfo,
};

pub struct Resolutions {
    pub node_tys: IndexMap<NodeId, Ty>,
    /// The unit dimension the dims pass computed for each expr —
    /// `b.x + 2.0px` is `Float` in `node_tys` but `px` here.
    pub node_dims: IndexMap<NodeId, shared::units::Dim>,
    /// The unit dimension the expr's context demanded — `b.x + BALL` passed to a
    /// `px` param is `Any` in `node_dims` but `px` here.
    pub want_dims: IndexMap<NodeId, shared::units::Dim>,
    pub node_decs: IndexMap<NodeId, DecId>,
    /// Read/write site counts per declaration -- the `{0,1,ω}` usage grades the
    /// lint suite is built on (`Use::Never` is what "unused" warnings report on).
    pub dec_uses: IndexMap<DecId, UseInfo>,
    /// Inferred effect set per `fn` dec: the union of what its body can do, seeded by
    /// `#[effects]` declarations on natives. `Fx::unknown()`-flagged sets carry
    /// `Fx::UNAUDITED` -- something in the call chain couldn't be graded.
    pub fn_effects: IndexMap<DecId, shared::Fx>,
    /// Effect set of each file's top-level statements, keyed by the ast's name. Hosts
    /// running peer pages gate on this: `fx.fits(granted)` or the code doesn't eval.
    pub script_effects: IndexMap<String, shared::Fx>,
    /// Per top-level statement: `(file name, site, inferred fx)` in source order.
    /// Spliced `use`-includes share one file name -- the site's byte span is what
    /// attributes each statement back to the include that wrote it.
    pub script_segments: Vec<(String, shared::Location, shared::Fx)>,
    /// Non-fatal diagnostics from the grades pass's lint suite. The solve succeeded;
    /// these are severity-warning reports for the host to render.
    pub warnings: Vec<miette::Report>,
    pub decs: IdVec<DecId, ResolvedDecl>,
    pub adts: IdVec<AdtId, ResolvedAdt>,
    pub pact_names: IdVec<PactId, String>,
    pub module_paths: HashMap<AdtId, Vec<String>>,
    pub closure_captures: IndexMap<NodeId, Vec<DecId>>,
    /// Exprs whose array lengths the solver couldn't prove against a `[T; n]`
    /// contract -- emit wraps each in a runtime dim check (see `lens.rs`).
    pub len_checks: IndexMap<NodeId, Vec<Option<usize>>>,
    /// `S::default()` calls the solver blessed: call node → the struct's adt.
    /// Emit lowers each to a `NewInstance` whose members are the types'
    /// defaults instead of a call.
    pub default_ctors: IndexMap<NodeId, AdtId>,
    /// The `__contract_fail` native's id, for contract-failure lowering at emit.
    /// `None` in embeddings that never installed the stdlib.
    pub contract_native: Option<NativeId>,
    /// The `panic` native's id -- emit's contract-failure fallback when the
    /// prettier `__contract_fail` channel isn't installed.
    pub panic_native: Option<NativeId>,
    pub root: ResolvedModule,
    pub sources: HashMap<FileId, NamedSource<Arc<str>>>,
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
    pub items: IndexMap<String, DecId>,
    pub modules: IndexMap<String, ResolvedModule>,
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
    /// The adt's declared type params (`struct P<T>` → `[T]`), positionally
    /// parallel to the `args` on a `Ty::Adt`/`Ty::Identity` instantiation.
    pub type_params: Vec<ParamId>,
    pub implements: Vec<PactId>,
    pub methods: IndexMap<String, DecId>,
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
        let type_params = adt.type_params.iter().map(|(_, pid)| *pid).collect();

        let mut implements: Vec<PactId> = solver
            .pact_impls
            .iter()
            .filter_map(|(p, a)| (*a == aid).then_some(*p))
            .collect();
        implements.sort_by_key(|p| p.index());

        let methods = adt
            .impls
            .iter()
            .filter(|(_, field)| matches!(solver.decs[field.dec].kind, DecKind::Item { .. }))
            .map(|(name, field)| (name.clone(), field.dec))
            .collect();

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
            out: &mut HashMap<AdtId, Vec<String>>,
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
        let mut paths = HashMap::new();
        for (name, &adt) in &solver.root_modules {
            paths.insert(adt, vec![name.clone()]);
            module_paths(&solver, adt, std::slice::from_ref(name), &mut paths);
        }

        let mut owners: HashMap<DecId, AdtId> = HashMap::new();
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
        let mut implements: HashMap<DecId, DecId> = HashMap::new();
        for &(pid, aid) in &solver.pact_impls {
            for (name, field) in &solver.adts[aid].impls {
                if let Some(&member) = solver.pact_members.get(&(pid, name.clone()))
                    && member != field.dec
                {
                    implements.insert(field.dec, member);
                }
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

        Self {
            node_tys,
            node_dims: solver.node_dims,
            want_dims: solver.want_dims,
            node_decs: solver.node_decs,
            dec_uses: solver.dec_uses,
            fn_effects: solver.fn_effects,
            script_effects: solver.script_effects,
            script_segments: solver.script_segments,
            warnings: solver.warnings,
            decs: resolved_decs,
            adts: resolved_adts,
            pact_names,
            module_paths: paths,
            closure_captures,
            len_checks: solver.len_checks,
            default_ctors: solver.default_ctors,
            contract_native,
            panic_native,
            root,
            sources: solver.sources,
        }
    }
}

/// What the grades pass learned about a load, handed to hosts deciding whether
/// to run the code at all -- mobile-code proof-carrying, half one: the page's
/// `Fx` is a checked property before a single bytecode executes.
pub struct GradeAudit {
    /// Each file's inferred top-level effect set, keyed by the file name the
    /// caller gave `load_files`. `UNAUDITED`-flagged sets contain calls the
    /// inference couldn't grade (unannotated natives, pact dispatch, closure
    /// callees) -- treat them as "could do anything" when gating.
    pub script_effects: IndexMap<String, shared::Fx>,
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
