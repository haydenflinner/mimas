//! Graded modes: one walk over each ast produces usage grades and effect grades.
//!
//! Usage grades power the lint suite. Every `let`/`for`/param/pattern binding is tracked and
//! every ident site the solver resolved (`node_decs` -- which now includes assignment targets)
//! counts as a read or a write of its declaration. Writes feed a small flow analysis:
//! `x = 1` lands in a *pending* set, the next read of `x` discharges it, and a pending write
//! killed by an overwrite or by scope exit is a dead store. Branches fork the pending set and
//! merge it by union (a write is only dead if dead on every path), `return`/`raise` retire the
//! flow into a list the enclosing scope still checks, and `break`/`continue` pool into the
//! loop's exit merge. Closure bodies share the enclosing flow but may run zero or many times,
//! so pending bookkeeping for decs declared outside a closure is suspended inside it.
//!
//! The result -- `dec_uses: FxIndexMap<DecId, UseInfo>` -- is honest per-site counts
//! (`Never`/`Once`/`Many`, saturating): `{0,1,ω}` grades for whoever reads them. The first
//! lints built on it are unused binding, unused parameter, and dead store -- warnings on
//! `Solver::warnings`, surfaced through `Resolutions`/`Loaded`, never errors. Script-rib
//! globals are skipped: reachable from every fn, their true count is ω by definition.
//!
//! Effect grades are the static half of running other people's pages. `shared::Fx` is the
//! powerset over `{doc, net, rng, yield, io, time}`; natives declare theirs with
//! `#[effects(...)]` (an `inventory` submission joined at install into `NativeBinding::sig`),
//! and a call's effect lands on the *caller*: a native contributes its declared set (`None` =
//! `Fx::unknown()`, the union of all six plus `UNAUDITED` -- missing annotation fails closed),
//! a user fn contributes a call edge, a pact-dispatched member or an unresolvable callee
//! (a fn-typed local, an operator, a closure value) contributes `unknown`. Edges then close
//! transitively by fixpoint: `fn_effects[f] = own[f] | union fn_effects[callee]`, monotone in
//! ⊆, so recursion settles on its own. Script top-level code isn't a fn dec, so its set is
//! kept separately per ast (`script_effects`, keyed by file name) -- page-eval effects
//! surface there for the host to check against its caps ledger. Closure bodies attribute
//! their effects to the enclosing scope (a `let f = || net_get(); ...` may call it anywhere,
//! or never -- v1 treats the write as "could do anything it captured", not as a call).
//!
//! Stage 1 (`consuming`/`must_use`) rides the same walk. A native may mark a param
//! `#[consumes]` (or `#[consumes(self)]` on a method): handing a place to that slot kills
//! the named region of its dec -- `f(r.id)` dies `r.id` (and `r` read whole afterwards,
//! not `r.other`), `f(r)` dies `r` outright, and touching a dead path is a compile
//! error, the one finding this pass fails the solve for. Consumed marks union across
//! branches (a maybe-dead region is gone), a store to a dead field reborns its subtree,
//! and consuming is refused for index-terminated paths (the element can't be named --
//! bind `let x = a[i]` first), globals, captures, and decs bound outside the enclosing
//! loop. `#[must_use]` on a native warns when a bare call statement discards its result.
//!
//! What the pass deliberately does not do beyond that: it does not grade types (modes live
//! on bindings and signatures -- `int` is ungraded), and Stage 3's certificate checks read
//! the maps this pass fills without adding their own.

use crate::{FxHashMap, FxHashSet};

use miette::Report;
use parse::{
    Access, Ast, Expr, ExprKind, FStringPart, Function, Ident, Item, ItemKind, Literal, PactItem,
    Stmt, StmtKind, Visitor,
    components::{Pat, PatKind},
    stmt::AssignmentOp,
};
use shared::{Fx, Located, Location};

use crate::{
    GuardSeg, Solver,
    components::{DecId, DecKind, TyExt},
    errors::{
        ConsumeForbidden, DeadStore, LoopCall, MustUseResult, ParamWriteDies,
        ShadowedBuiltin, UnusedBinding, UnusedParam, UseAfterConsume,
    },
};

/// One `#[consumes]` kill: the field path under the dec that died (`[]` =
/// the whole binding), where it happened, and the callee that took it.
/// The vec is a small set under `paths_overlap` -- `r.a` dead and `r.b`
/// dead are separate marks, while `r` dead alone covers both.
type ConsumeMark = (Vec<String>, Location, String);

/// The statically known field prefix of an access path: `a.b[i]` ->
/// `["b"]`, `a[i]` -> `[]`. Anything past the first `Index` isn't a
/// place this pass can name.
fn field_prefix(path: &[GuardSeg]) -> Vec<String> {
    path.iter()
        .map_while(|s| match s {
            GuardSeg::Field(f) => Some(f.clone()),
            GuardSeg::Index => None,
        })
        .collect()
}

/// A consumed mark and a later touch collide when one path is a prefix
/// of the other: `r.id` dead kills `r` read whole and `r.id.x`, while
/// `r.other` stays live. `[]` overlaps everything (whole-binding dead).
/// `a` and `b` overlap when either reaches the other — `xs` loans
/// `xs[i]` and vice versa, `w.tour` never meets `w.other`. Two `Index`
/// hops can't be told apart, so `xs[i]` vs `xs[j]` counts as overlap.
fn guard_paths_overlap(a: &[GuardSeg], b: &[GuardSeg]) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

fn paths_overlap(a: &[String], b: &[String]) -> bool {
    a.iter().zip(b).all(|(x, y)| x == y)
}

/// How many sites read or write a declaration. Saturates at `Many` -- the lint suite and any
/// future `consuming` checks only need `{0, 1, ω}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Use {
    #[default]
    Never,
    Once,
    Many,
}

impl Use {
    fn tick(&mut self) {
        *self = match self {
            Use::Never => Use::Once,
            _ => Use::Many,
        };
    }
}

/// Per-declaration use counts. Sites, not paths: `if c { x } else { x }` records `Once` for
/// `x` twice -- `Many` reads. (Path-accurate counting is a merge-join away if a consumer ever
/// wants it; the lints only need `Never` vs not.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UseInfo {
    pub reads: Use,
    pub writes: Use,
}

/// Where a lint-tracked binding came from, so the warning names it right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Site {
    /// `let`, `for`, `match`/`if let`/`while let` bindings
    Binding,
    /// `fn` and closure parameters
    Param,
}

/// Who a call's effects attribute to: a named fn's dec, or one top-level
/// statement's slot in `script_sites` (there is no script dec, and splitting
/// per statement is what lets a host attribute a spliced page's effects back
/// to the include that wrote them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Caller {
    Script(usize),
    Fn(DecId),
}

/// One path's piece of the dead-store analysis. `pending` is the last-writer set: writes
/// whose value a read might still see. Writes die three ways and each is recorded, because
/// deadness is a whole-program property a local kill can't always decide: a span killed
/// inside an `if` arm is only dead if *every* arm killed it, and a span read on any path is
/// never dead at all.
#[derive(Clone, Default)]
struct Flow {
    /// dec -> the write sites whose value no read has consumed yet (last writers).
    pending: FxHashMap<DecId, FxHashSet<Location>>,
    /// dec -> write sites this path overwrote but did not itself produce. Arbitrated at
    /// merges: a span in here died on this path, and only stays a candidate if it died on
    /// every sibling path too.
    killed: FxHashMap<DecId, FxHashSet<Location>>,
    /// dec -> write sites a read on this path consumed. Any read makes the store live.
    read_spans: FxHashMap<DecId, FxHashSet<Location>>,
    /// Write sites created inside this path segment. Killing one is unconditional -- it
    /// can't exist on a sibling path -- so it reports straight away.
    born: FxHashSet<Location>,
    /// dec -> the `(path, site, callee)` marks `#[consumes]` params took on this
    /// path. Unioned at merges: a region *maybe* consumed on any surviving path is
    /// gone from the join on -- matching Rust's move-out-of-one-branch behavior.
    consumed: FxHashMap<DecId, Vec<ConsumeMark>>,
    diverged: bool,
}

impl Flow {
    /// The flow an alternative path starts from: same pending/killed/read sets, but no
    /// writes born yet -- an inherited span killed inside an arm has to wait for the merge
    /// to decide its fate.
    fn fork(&self) -> Flow {
        let mut f = self.clone();
        f.born.clear();
        f
    }

    /// The state a join point can be in: pending and read sites union, killed sites
    /// intersect -- a write only stays dead when every arriving path killed it and no
    /// arriving path kept or read it. A path that can't reach the join (`return`, `break`
    /// already steered elsewhere) contributes nothing; if none arrive the join itself is
    /// unreachable.
    fn merge(flows: impl IntoIterator<Item = Flow>) -> Flow {
        let arms: Vec<Flow> = flows.into_iter().filter(|f| !f.diverged).collect();
        let mut pending: FxHashMap<DecId, FxHashSet<Location>> = FxHashMap::default();
        let mut read_spans: FxHashMap<DecId, FxHashSet<Location>> = FxHashMap::default();
        let mut killed: FxHashMap<DecId, FxHashSet<Location>> = FxHashMap::default();
        for f in &arms {
            for (dec, spans) in &f.pending {
                pending
                    .entry(*dec)
                    .or_default()
                    .extend(spans.iter().copied());
            }
            for (dec, spans) in &f.read_spans {
                read_spans
                    .entry(*dec)
                    .or_default()
                    .extend(spans.iter().copied());
            }
            // a dec this arm never killed leaves the arbitration
            killed.retain(|dec, _| f.killed.contains_key(dec));
            for (dec, spans) in &f.killed {
                killed
                    .entry(*dec)
                    .or_insert_with(|| spans.clone())
                    .retain(|s| spans.contains(s));
            }
        }
        // a span live on any path resolves the arbitration for good
        for (dec, spans) in &mut killed {
            spans.retain(|s| {
                !pending.get(dec).is_some_and(|p| p.contains(s))
                    && !read_spans.get(dec).is_some_and(|r| r.contains(s))
            });
        }
        killed.retain(|_, spans| !spans.is_empty());
        // born restarts empty: a pending span inherited through a merge was written on a
        // path a sibling may still read from, so it isn't segment-local -- only writes
        // that happen after the merge can be eager-killed
        let born: FxHashSet<Location> = FxHashSet::default();
        let mut consumed: FxHashMap<DecId, Vec<ConsumeMark>> = FxHashMap::default();
        for f in &arms {
            for (dec, marks) in &f.consumed {
                consumed
                    .entry(*dec)
                    .or_default()
                    .extend(marks.iter().cloned());
            }
        }
        Flow {
            pending,
            killed,
            read_spans,
            born,
            consumed,
            diverged: arms.is_empty(),
        }
    }
}

struct Pass<'a> {
    solver: &'a Solver,

    /// dec -> read/write site counts; merged into `Solver::dec_uses` at the end.
    uses: FxHashMap<DecId, UseInfo>,
    /// Bindings eligible for unused-warnings: (dec, where it came from).
    lint_targets: Vec<(DecId, Site)>,
    /// Walking the methods of an `impl Pact for T`: their signatures are
    /// the pact's, so their parameters aren't linted.
    in_pact_impl: bool,
    /// The scope each dec was bound in (scope_decs.len() at bind time), for the
    /// closure-isolation rule.
    dec_scope: FxHashMap<DecId, usize>,
    /// Dead-store candidates: pending writes killed by overwrite or scope exit. Emitted only
    /// if the dec was read at all -- otherwise `unused binding` already owns the complaint.
    dead_candidates: Vec<(DecId, Location)>,
    warnings: Vec<Report>,

    /// Declarations living in each open scope (blocks, fn bodies, arms, loop bodies) --
    /// popped scopes kill their pending writes.
    scope_decs: Vec<Vec<DecId>>,
    /// scope_decs.len() at each enclosing closure boundary. Inside a closure, pending
    /// bookkeeping (set/clear) is skipped for decs bound at or outside the floor: the
    /// closure may run zero or many times, so its reads/writes can't decide whether an
    /// outer write is dead.
    closure_floors: Vec<usize>,
    /// scope_decs.len() at each enclosing loop's entry. Consuming a dec bound at or
    /// outside the floor inside the loop body would poison the next iteration -- refused.
    loop_floors: Vec<usize>,
    /// Decs read anywhere inside each enclosing loop (header and body alike), one set per
    /// nesting level. The pass walks a body once, so a store at the bottom can never be
    /// observed reaching a read -- but the back-edge carries it to next iteration's reads.
    /// A dec in this set keeps its body-end writes live (conservative: it can't see
    /// whether an intervening write beats the read, only that one exists).
    loop_reads: Vec<FxHashSet<DecId>>,
    /// Decs whose use-after-consume already reported (one error per dead
    /// path, not per read of it).
    consume_reported: FxHashSet<(DecId, Vec<String>)>,
    /// The spine root's bare-name consume check already ran at the
    /// access's deeper path -- `r.id` doesn't read all of `r`. Set by the
    /// access arms before walking `left`, cleared by the next read.
    skip_root_check: bool,
    /// Inside an assignment's left spine -- those reads locate the slot,
    /// they don't observe the value (the target's own check+revive ran
    /// first). `x.f = v` is a store, `x.f += v` still reads.
    in_lvalue: bool,
    /// Fatal findings (`consuming` violations). Unlike the lints these stop the solve --
    /// collected so sibling files still fill their effects and warnings first.
    errors: Vec<Report>,

    flow: Flow,
    /// Flows that exited the enclosing fn early (`return`, `raise`, `let-else`). Their
    /// pending writes still die when their decs' scopes pop.
    retired: Vec<Flow>,
    /// Loop exit flows, innermost last (`break`/`continue` push into the top).
    breaks: Vec<Vec<Flow>>,

    /// Caller stack: `[Script(stmt_idx)]` at file top level, `Fn(dec)` pushed
    /// inside `fn` items. The script slot is re-assigned per top-level
    /// statement (see `run`), so each statement's effect set lands separately.
    callers: Vec<Caller>,
    /// Top-level statement sites, in source order -- `callers[0]`'s index into
    /// this is the current script segment.
    script_sites: Vec<Location>,
    /// Effects directly observed in each caller's own text.
    own_fx: FxHashMap<Caller, Fx>,
    /// caller -> user-fn callee decs (the fixpoint's edges).
    edges: FxHashMap<Caller, FxHashSet<DecId>>,
    /// `fn` name site -> dec, so a `Function` node can name its caller.
    dec_by_site: FxHashMap<(usize, usize), DecId>,
    /// pact member decs -- a call resolved to one dispatches to an impl this file can't see,
    /// so it's `unknown`, not an edge.
    pact_members: FxHashSet<DecId>,

    /// `needs {…}` decls walked: `(site, declared set)`. After the fixpoint
    /// each page's inferred set must fit the union its `needs` wrote.
    needs: Vec<(Location, Fx)>,
    /// `use "…" granting {…}` sites walked: `(site, spelled name, granted
    /// set)`. Each must fit the granting page's own authority (its `needs`,
    /// or ambient `Fx::ANY` when it declares none) -- and the spliced-in
    /// page's inferred set must fit the grant.
    grants: Vec<(Location, String, Fx)>,

    /// Call sites walked under an enclosing loop: `(site, caller, callee
    /// Fns marked `#[scans]` — their collection walks are deliberate:
    /// calls to them, and looped calls inside them, don't lint.
    scan_marked: FxHashSet<DecId>,
    /// dec, per-arg ambient bits)`. `loop_reads` non-empty is the
    /// "inside a loop" signal — it is pushed before a `while` header
    /// walks and survives into closures, the two places a per-iteration
    /// call hides. Pact calls record each impl dec, since the pact dec
    /// itself owns no edges. Arg positions line up with the callee's
    /// parameters, the receiver counted as 0 on `x.m(..)` calls.
    loop_calls: Vec<(Location, Caller, DecId, Vec<bool>)>,
    /// Callers whose own body holds a `for`/`while`/`loop` — the seeds
    /// the `iterates` fixpoint propagates outward through `edges`.
    own_loops: FxHashSet<Caller>,

    /// dec -> the position of the fn parameter that bound it (`self`
    /// is position 0 for methods). Flat: a dec names one binding ever,
    /// so an entry left by another fn's walk can never resolve wrongly.
    param_pos: FxHashMap<DecId, u32>,
    /// by-value param decs that were re-stored (`n = 0`) -- the store
    /// dies at the body's end and never reaches the caller.
    param_stores: FxHashSet<DecId>,
    /// local dec -> the param positions its value derives from. `let
    /// bs = w.balls` stands `bs` for `w`, a `for` binding stands for
    /// its iterator's roots, and a fresh `f(w)` or `[]` roots at none.
    provenance: FxHashMap<DecId, FxHashSet<u32>>,
    /// caller -> the param positions its own `for` iterators and
    /// `while` headers root at: `for t in w.things` scans `w`, `for i
    /// in 0 .. n` scans nothing. Seeds the `scans` fixpoint.
    own_scans: FxHashMap<Caller, FxHashSet<u32>>,
    /// Every resolved call: `(caller, callee, per-arg param
    /// positions)`, aligned like `loop_calls`'s args. The scans
    /// fixpoint walks it backward — a callee scanning position p marks
    /// the caller scanning whatever roots its arg at p.
    call_args: Vec<(Caller, DecId, Vec<FxHashSet<u32>>)>,

    /// Caller -> the param positions its own text writes through, with
    /// the path suffix each write lands on: `w.tour.f = ` seeds `w`'s
    /// position with `[tour, f]`; an `inout` param's `x = v` seeds `x`'s
    /// with `[]` (the caller's slot); `xs.push(..)` on a param seeds
    /// `[]` (its whole storage). A store to a plain param's binding
    /// (`n = 0`, `p = P{..}`) is local-only and never lands here.
    /// r[impl mimas.fn.writes]
    own_writes: FxHashMap<Caller, Vec<(u32, Vec<GuardSeg>)>>,
    /// Every resolved call, for the writes fixpoint: `(caller, callee,
    /// per-arg (param position, access path))`. `f(w.tour)` records
    /// `w`'s position under `[tour]`, so `f` writing `w.tour.step`
    /// marks the caller writing `w.tour.step`. Arg positions line up
    /// like `call_args`'s.
    call_write_args: Vec<(Caller, DecId, Vec<Vec<(u32, Vec<GuardSeg>)>>)>,
    /// Writes a `let`-bound closure makes to decs bound above its
    /// floor, keyed by the binding's dec -- `poke()` inside a frozen
    /// loop resolves `poke` to its dec and reads this set.
    closure_writes: FxHashMap<DecId, Vec<(DecId, Vec<GuardSeg>)>>,
    /// `let`-bound closure decs for each open closure body, innermost
    /// last (`None` for a closure not directly bound by a `let`).
    closure_stack: Vec<Option<DecId>>,
    /// The dec the closure body about to be walked gets bound to --
    /// set by the `let` arm when its right is a `Closure` expr.
    pending_closure_dec: Option<DecId>,
}

/// The expensive-precision lints — `loop-call` today — run unless the
/// host opts out with `MIMAS_DEEP_LINT=0` (or an empty value). The
/// corpus compiles clean under them, so on is the default: a quadratic
/// shape is exactly the thing a call site cannot show a reader.
fn deep_lints() -> bool {
    !matches!(std::env::var("MIMAS_DEEP_LINT"), Ok(v) if v == "0" || v.is_empty())
}

/// Fill `solver.dec_uses`, `solver.fn_effects`, `solver.script_effects`, `solver.warnings`.
/// Runs after `dims::check` at the tail of `solve_all`. Lints only warn, but a
/// `#[consumes]` violation is a real refusal: `Err` fails the solve like any other pass.
pub(crate) fn check(solver: &mut Solver, asts: &[&Ast]) -> crate::errors::Result<()> {
    let mut errors: Vec<Report> = Vec::new();
    // call edges and per-caller effects gather across all asts before the fixpoint -- a fn in
    // file A can call a fn in file B
    let mut edges: FxHashMap<DecId, FxHashSet<DecId>> = FxHashMap::default();
    let mut fn_fx: FxHashMap<DecId, Fx> = FxHashMap::default();
    // per file: (name, [(stmt site, own fx, callee edges)]) -- folded into
    // `script_effects`/`script_segments` after the fixpoint
    let mut script_fx: Vec<(String, Vec<(Location, Fx, FxHashSet<DecId>)>)> = Vec::new();
    // per file, in `asts` order: its `needs` decls and `granting` sites --
    // checked per page once the fixpoint has every caller's set
    let mut decl_fx: Vec<(Vec<(Location, Fx)>, Vec<(Location, String, Fx)>)> = Vec::new();
    // the quadratic lint's raw material: call sites under an enclosing
    // loop, callers holding their own loops (the `iterates` seeds), the
    // param positions each caller's loops scan, and every call's arg
    // roots for propagating those scans up the graph
    let mut loop_calls: Vec<(Location, Caller, DecId, Vec<bool>)> = Vec::new();
    let mut own_loops: FxHashSet<Caller> = FxHashSet::default();
    let mut own_scans: FxHashMap<Caller, FxHashSet<u32>> = FxHashMap::default();
    let mut own_writes: FxHashMap<Caller, Vec<(u32, Vec<GuardSeg>)>> = FxHashMap::default();
    let mut call_write_args: Vec<(Caller, DecId, Vec<Vec<(u32, Vec<GuardSeg>)>>)> = Vec::new();
    let mut closure_writes: FxHashMap<DecId, Vec<(DecId, Vec<GuardSeg>)>> = FxHashMap::default();
    let mut call_args: Vec<(Caller, DecId, Vec<FxHashSet<u32>>)> = Vec::new();
    let mut scan_marked: FxHashSet<DecId> = FxHashSet::default();

    for ast in asts {
        let mut dec_by_site = FxHashMap::default();
        for (id, dec) in solver.decs.iter() {
            if matches!(dec.kind, DecKind::Item { .. }) {
                dec_by_site
                    .entry((dec.location.file_id, dec.location.span.start))
                    .or_insert(id);
            }
        }
        let mut pass = Pass {
            solver,
            uses: FxHashMap::default(),
            lint_targets: Vec::new(),
            in_pact_impl: false,
            dec_scope: FxHashMap::default(),
            dead_candidates: Vec::new(),
            warnings: Vec::new(),
            scope_decs: Vec::new(),
            closure_floors: Vec::new(),
            loop_floors: Vec::new(),
            loop_reads: Vec::new(),
            consume_reported: FxHashSet::default(),
            skip_root_check: false,
            in_lvalue: false,
            errors: Vec::new(),
            flow: Flow::default(),
            retired: Vec::new(),
            breaks: Vec::new(),
            callers: vec![Caller::Script(0)],
            script_sites: Vec::new(),
            own_fx: FxHashMap::default(),
            edges: FxHashMap::default(),
            dec_by_site,
            pact_members: solver.pact_members.values().copied().collect(),
            needs: Vec::new(),
            grants: Vec::new(),
            loop_calls: Vec::new(),
            own_loops: FxHashSet::default(),
            param_pos: FxHashMap::default(),
            param_stores: FxHashSet::default(),
            provenance: FxHashMap::default(),
            own_scans: FxHashMap::default(),
            call_args: Vec::new(),
            scan_marked: FxHashSet::default(),
            own_writes: FxHashMap::default(),
            call_write_args: Vec::new(),
            closure_writes: FxHashMap::default(),
            closure_stack: Vec::new(),
            pending_closure_dec: None,
        };
        pass.push_scope();
        pass.run_top(ast.stmts());
        pass.pop_scope();
        pass.finish_lints();

        // one `(location, own, calls)` triple per top-level statement -- the
        // segments a host maps back onto `use`-include byte ranges
        let mut segments = Vec::with_capacity(pass.script_sites.len());
        for (i, &site) in pass.script_sites.iter().enumerate() {
            let own = pass
                .own_fx
                .get(&Caller::Script(i))
                .copied()
                .unwrap_or(Fx::empty());
            let calls = pass.edges.remove(&Caller::Script(i)).unwrap_or_default();
            segments.push((site, own, calls));
        }
        script_fx.push((ast.name().to_string(), segments));
        decl_fx.push((
            std::mem::take(&mut pass.needs),
            std::mem::take(&mut pass.grants),
        ));
        for (caller, fx) in pass.own_fx {
            if let Caller::Fn(d) = caller {
                *fn_fx.entry(d).or_insert(Fx::empty()) |= fx;
            }
        }
        for (caller, callees) in pass.edges {
            if let Caller::Fn(d) = caller {
                edges.entry(d).or_default().extend(callees);
            }
        }
        loop_calls.extend(pass.loop_calls.iter().cloned());
        own_loops.extend(pass.own_loops.iter().copied());
        call_args.extend(pass.call_args.iter().cloned());
        call_write_args.extend(pass.call_write_args.iter().cloned());
        closure_writes.extend(pass.closure_writes.iter().map(|(d, w)| (*d, w.clone())));
        for (caller, ws) in pass.own_writes.iter() {
            own_writes
                .entry(*caller)
                .or_insert_with(Vec::new)
                .extend(ws.iter().cloned());
        }
        for (caller, set) in pass.own_scans.iter() {
            own_scans
                .entry(*caller)
                .or_insert_with(FxHashSet::default)
                .extend(set.iter().copied());
        }
        scan_marked.extend(pass.scan_marked.iter().copied());
        // `pass` borrows `&*solver`; pull its results out before writing solver fields
        let Pass {
            uses,
            warnings,
            errors: errs,
            ..
        } = pass;
        errors.extend(errs);
        for (dec, info) in uses {
            let entry = solver.dec_uses.entry(dec).or_default();
            for _ in 0..info.reads as usize {
                entry.reads.tick();
            }
            for _ in 0..info.writes as usize {
                entry.writes.tick();
            }
        }
        solver
            .warnings
            .extend(warnings.into_iter().map(std::rc::Rc::new));
    }

    // fixpoint: fn_effects[f] |= union fn_fx[callee] until stable. Each round only adds bits,
    // so this terminates in at most (edge count) rounds; recursive cycles settle on their own.
    // r[impl mimas.fx.infer]
    loop {
        let mut grown = Vec::new();
        for (caller, callees) in &edges {
            let mut acc = fn_fx.get(caller).copied().unwrap_or(Fx::empty());
            for callee in callees {
                acc |= fn_fx.get(callee).copied().unwrap_or(Fx::empty());
            }
            if acc != fn_fx.get(caller).copied().unwrap_or(Fx::empty()) {
                grown.push((*caller, acc));
            }
        }
        if grown.is_empty() {
            break;
        }
        for (caller, acc) in grown {
            *fn_fx.entry(caller).or_insert(Fx::empty()) |= acc;
        }
    }
    // r[impl mimas.lint.loop-call] -- a call under an enclosing loop to a
    // callee that iterates too is the Shlemiel shape: every outer pass
    // pays the inner collection's walk again. But the verdict has to hold
    // without the caller in the graph: `visit` calling `visit` on each
    // child is a descent, not a rescan, and a callee that only "iterates"
    // by reaching back into this caller is the same cycle painted twice.
    // `iterates` is therefore recomputed per caller over the graph minus
    // that caller — seeds at callers holding a loop, grown along `edges`
    // to stability like the fx fixpoint's.
    //
    // A deep lint, on by default (`MIMAS_DEEP_LINT=0` opts out): the
    // corpus's deliberate scanners carry `#[scans]`, so what remains is a
    // real quadratic shape the call site cannot show a reader.
    if !deep_lints() {
        loop_calls.clear();
    }
    // `scans[f]` — the param positions `f`'s loops read, directly or
    // through calls. Seeds are `own_scans` (`for t in w.things` marks
    // `w`); the fixpoint walks `call_args` backward the way `fn_fx`'s
    // walks `edges`, so `handle(w) → feed(w)` makes `handle` scan `w`
    // too. Recursive cycles settle on their own.
    let mut scans: FxHashMap<DecId, FxHashSet<u32>> = FxHashMap::default();
    for (caller, set) in &own_scans {
        if let Caller::Fn(d) = caller {
            scans.entry(*d).or_default().extend(set.iter().copied());
        }
    }
    loop {
        let mut grown: Vec<(DecId, FxHashSet<u32>)> = Vec::new();
        for (caller, callee, args) in &call_args {
            let Caller::Fn(f) = caller else { continue };
            let Some(cs) = scans.get(callee) else { continue };
            let mut add = FxHashSet::default();
            for p in cs.iter() {
                if let Some(roots) = args.get(*p as usize) {
                    add.extend(roots.iter().copied());
                }
            }
            let have = scans.get(f).cloned().unwrap_or_default();
            if !add.is_empty() && !add.is_subset(&have) {
                grown.push((*f, add));
            }
        }
        if grown.is_empty() {
            break;
        }
        for (f, add) in grown {
            scans.entry(f).or_default().extend(add);
        }
    }
    // `writes[f]` — the (param position, path) pairs `f` may write
    // through, directly or through calls. `[]` means "somewhere in the
    // param's storage" — a write reached through a derived binding, a
    // mutating native on the root, or an `inout` store all widen to it.
    // The fixpoint walks `call_write_args` the way `scans`' walks
    // `call_args`: `handle(w) → feed(w.balls)` makes `handle` write
    // `w.balls`, recursion settles like `fn_fx`'s cycles.
    // r[impl mimas.fn.writes]
    let mut writes: FxHashMap<DecId, Vec<(u32, Vec<GuardSeg>)>> = FxHashMap::default();
    for (caller, set) in &own_writes {
        if let Caller::Fn(d) = caller {
            writes.entry(*d).or_default().extend(set.iter().cloned());
        }
    }
    loop {
        let mut grown: Vec<(DecId, Vec<(u32, Vec<GuardSeg>)>)> = Vec::new();
        for (caller, callee, args) in &call_write_args {
            let Caller::Fn(f) = caller else { continue };
            let Some(cw) = writes.get(callee) else { continue };
            let mut add = Vec::new();
            for (cp, cpath) in cw {
                if let Some(argpaths) = args.get(*cp as usize) {
                    for (ap, apath) in argpaths {
                        let mut path = apath.clone();
                        path.extend(cpath.iter().cloned());
                        add.push((*ap, path));
                    }
                }
            }
            let have = writes.get(f).cloned().unwrap_or_default();
            add.retain(|w| !have.contains(w));
            if !add.is_empty() {
                grown.push((*f, add));
            }
        }
        if grown.is_empty() {
            break;
        }
        for (f, add) in grown {
            writes.entry(f).or_default().extend(add);
        }
    }
    solver.fn_writes = writes;

    // `__fn_writes(f)` placeholders `sig!` emitted become the literal
    // write set: `sig!(f).writes` reads `["w.tour.step"]`, param names
    // from the fn's own header; a whole-binding write is the bare name.
    // r[impl mimas.fn.writes]
    let asks = std::mem::take(&mut solver.fn_writes_asks);
    for (call_id, (dec, at)) in asks {
        let mut items = Vec::new();
        if let Some(dec) = dec {
            let names: Vec<Option<String>> =
                match crate::components::Ty::Vid(solver.decs[dec].vid).normalized(solver) {
                    crate::components::Ty::Fn(h) => {
                        h.parameters.iter().map(|p| p.name.clone()).collect()
                    }
                    _ => Vec::new(),
                };
            for (p, segs) in solver.fn_writes.get(&dec).into_iter().flatten() {
                let mut s = names
                    .get(*p as usize)
                    .and_then(|n| n.clone())
                    .unwrap_or_else(|| format!("arg{p}"));
                for seg in segs {
                    match seg {
                        GuardSeg::Field(f) => {
                            s.push('.');
                            s.push_str(f);
                        }
                        GuardSeg::Index => s.push_str("[]"),
                    }
                }
                items.push(Expr::new(ExprKind::Literal(Literal::String(s)), at));
            }
        }
        solver.rewrites.insert(
            call_id,
            Expr::new(ExprKind::Literal(Literal::Array(items)), at),
        );
    }

    // A call under a `for` freeze the body's own text never saw:
    // `poke()`'s closure or `f(xs)`'s body mutates the iterated
    // collection. Deferred from solve to here, where the callee's write
    // set is finally known. r[impl mimas.for.inplace-freeze]
    for gc in &solver.guarded_calls {
        let call_writes: Vec<(DecId, Vec<GuardSeg>)> = match gc.callee {
            // `None`: the callee is a value -- a param, a returned
            // closure, a dispatch. Can't prove its writes; any argument
            // reaching a frozen path is refused.
            None => gc
                .arg_roots
                .iter()
                .flatten()
                .filter(|(dec, path)| {
                    gc.guards.iter().any(|g| {
                        g.dec == *dec
                            && (g.path == *path || g.inplace && path.starts_with(&g.path))
                    })
                })
                .map(|(dec, path)| (*dec, path.clone()))
                .collect(),
            Some(d) => {
                if let Some(writes) = closure_writes.get(&d) {
                    writes.clone()
                } else if let Some(cw) = solver.fn_writes.get(&d) {
                    cw.iter()
                        .filter_map(|(cp, cpath)| {
                            gc.arg_roots
                                .get(*cp as usize)
                                .and_then(|o| o.as_ref())
                                .map(|(dec, apath)| {
                                    let mut path = apath.clone();
                                    path.extend(cpath.iter().cloned());
                                    (*dec, path)
                                })
                        })
                        .collect()
                } else {
                    // a native, pact member, or unresolvable dec: mutating
                    // receivers and `&` lends were already refused at solve;
                    // everything else can't reach the frozen place.
                    Vec::new()
                }
            }
        };
        for (dec, wpath) in call_writes {
            if let Some(guard) = gc.guards.iter().find(|g| {
                g.dec == dec && (g.path == wpath || g.inplace && wpath.starts_with(&g.path))
            }) {
                errors.push(
                    crate::errors::MutateWhileIterating {
                        src: solver.src(gc.at),
                        at: gc.at.into(),
                        name: guard.name.clone(),
                        iterating: vec![crate::errors::IteratingHere {
                            src: solver.src(guard.iterated_at),
                            at: guard.iterated_at.into(),
                            name: guard.name.clone(),
                        }],
                    }
                    .into(),
                );
                break;
            }
        }
    }
    // A plain argument sharing a lent root: refused at solve only when
    // the callee couldn't be resolved -- a resolvable callee's verdict
    // waits on whether it writes that slot's overlap with the lend.
    // r[impl mimas.inout.exclusive]
    for d in &solver.deferred_inout {
        let Some(callee) = d.callee else { continue };
        let Some(cw) = solver.fn_writes.get(&callee) else {
            continue;
        };
        if cw.iter().any(|(cp, cpath)| {
            *cp as usize == d.slot && guard_paths_overlap(cpath, &d.lent_path)
        }) {
            errors.push(
                crate::errors::InoutAlias {
                    src: solver.src(d.at),
                    at: d.at.into(),
                    root: d.lent_name.clone(),
                }
                .into(),
            );
        }
    }
    let seeds: FxHashSet<DecId> = own_loops
        .iter()
        .filter_map(|c| match c {
            Caller::Fn(d) => Some(*d),
            Caller::Script(_) => None,
        })
        .collect();
    let mut iterates_of: FxHashMap<Option<DecId>, FxHashSet<DecId>> = FxHashMap::default();
    let mut reported = FxHashSet::default();
    for (at, caller, callee, ambients) in loop_calls {
        let (blocked, outer) = match caller {
            Caller::Fn(d) => (Some(d), solver.decs[d].name.clone()),
            Caller::Script(_) => (None, "the top level".to_string()),
        };
        let name = &solver.decs[callee].name;
        if name.starts_with("__")
            || outer.starts_with("__")
            || !reported.insert((at.file_id, at.span.start, callee))
        {
            continue;
        }
        let iterates = iterates_of.entry(blocked).or_insert_with(|| {
            let mut set: FxHashSet<DecId> =
                seeds.iter().copied().filter(|d| Some(*d) != blocked).collect();
            loop {
                let mut grown = Vec::new();
                for (caller, callees) in &edges {
                    if Some(*caller) == blocked || set.contains(caller) {
                        continue;
                    }
                    if callees
                        .iter()
                        .any(|g| Some(*g) != blocked && set.contains(g))
                    {
                        grown.push(*caller);
                    }
                }
                if grown.is_empty() {
                    break;
                }
                set.extend(grown);
            }
            set
        });
        // iterating is not enough — the callee must scan a shared
        // place, not the element the caller just handed it. `num(t)`
        // per token `t` costs O(|t|) a call; `ball_index(w, ..)` per
        // ball costs O(world) a call, and that is the quadratic shape.
        let scans_arg = scans.get(&callee).is_some_and(|cs| {
            cs.iter()
                .any(|p| ambients.get(*p as usize).copied().unwrap_or(false))
        });
        if !iterates.contains(&callee)
            || Some(callee) == blocked
            || !scans_arg
            // `#[scans]` on the callee says its walks are deliberate;
            // on the caller, that its looped calls are. Either way the
            // pair is acknowledged, not flagged.
            || scan_marked.contains(&callee)
            || blocked.is_some_and(|d| scan_marked.contains(&d))
        {
            continue;
        }
        let name = name.clone();
        solver.warnings.push(std::rc::Rc::new(
            LoopCall {
                src: solver.src(at),
                at: at.into(),
                name,
                outer,
            }
            .into(),
        ));
    }
    for (caller, fx) in fn_fx {
        solver.fn_effects.insert(caller, fx);
    }
    for (i, (name, segments)) in script_fx.into_iter().enumerate() {
        let mut file_fx = Fx::empty();
        let mut page_fx: FxHashMap<&str, Fx> = FxHashMap::default();
        let marks = asts[i].page_marks();
        // the page a byte offset splices from: the last `//# page:` mark at
        // or before it (`""` is the file's own text, marked or implicit)
        let page_of = |byte: usize| -> &str {
            marks
                .iter()
                .rev()
                .find(|(start, _)| *start <= byte)
                .map_or("", |(_, name)| *name)
        };
        for (site, own, calls) in segments {
            let fx = own
                | calls.iter().fold(Fx::empty(), |acc, d| {
                    acc | solver.fn_effects.get(d).copied().unwrap_or(Fx::empty())
                });
            file_fx |= fx;
            *page_fx
                .entry(page_of(site.span.start))
                .or_insert(Fx::empty()) |= fx;
            solver.script_segments.push((name.clone(), site, fx));
        }
        solver.script_effects.insert(name.clone(), file_fx);

        // Effect-authority checks (`mimas.fx.needs`,
        // `mimas.use.page.grant-narrow`, `mimas.use.page.grant-check`). A
        // `needs` decl bounds its page's inferred set and is the page's
        // authority for `granting` clauses; absent one the ambient
        // authority is `Fx::ANY` (the host decides what a page may ask
        // for) -- still narrower than `unknown()`, which also carries
        // UNAUDITED.
        let (needs, grants) = &decl_fx[i];
        let mut authority: FxHashMap<&str, (Location, Fx)> = FxHashMap::default();
        for (site, declared) in needs {
            let page = page_of(site.span.start);
            let entry = authority.entry(page).or_insert((*site, Fx::empty()));
            entry.1 |= *declared;
        }
        for (page, (site, declared)) in &authority {
            let inferred = page_fx.get(page).copied().unwrap_or(Fx::empty());
            solver
                .page_needs
                .push((page_name(name.as_str(), page), *site, *declared));
            // r[impl mimas.fx.needs]
            if !inferred.fits(*declared) {
                errors.push(
                    crate::errors::NeedsExceeded {
                        src: solver.src(*site),
                        at: (*site).into(),
                        declared: declared.to_string(),
                        missing: inferred.difference(*declared).to_string(),
                    }
                    .into(),
                );
            }
        }
        for (site, spelled, granted) in grants {
            let page = page_of(site.span.start);
            let held = authority.get(page).map_or(Fx::ANY, |(_, fx)| *fx);
            // r[impl mimas.use.page.grant-narrow]
            if !granted.fits(held) {
                errors.push(
                    crate::errors::GrantExceeded {
                        src: solver.src(*site),
                        at: (*site).into(),
                        name: spelled.clone(),
                        granted: granted.to_string(),
                        authority: held.to_string(),
                    }
                    .into(),
                );
            }
            // r[impl mimas.use.page.grant-check] -- the spliced-in page's own
            // top-level set must fit what its `use` granted. Lookup by the
            // spelled name first (`kit@latest` marks land verbatim), then the
            // version tail's page part; an unspliced name is the host's
            // `include.missing`, not the grant check's, business.
            let page_part = match parse::use_target(spelled) {
                parse::UseTarget::Live(page) | parse::UseTarget::Follow(page) => page,
                parse::UseTarget::Pinned { page, .. } => page,
            };
            let required = [spelled.as_str(), page_part].into_iter().find_map(|want| {
                page_fx
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(want))
                    .map(|(_, fx)| *fx)
            });
            if let Some(required) = required
                && !required.fits(*granted)
            {
                errors.push(
                    crate::errors::IncludeGrantDenied {
                        src: solver.src(*site),
                        at: (*site).into(),
                        name: spelled.clone(),
                        missing: required.difference(*granted).to_string(),
                        granted: granted.to_string(),
                    }
                    .into(),
                );
            }
        }
    }
    // consume violations ride every other phase's convention: first error out
    if let Some(err) = errors.into_iter().next() {
        return Err(err);
    }
    Ok(())
}

/// What a `needs` decl's page is called on `page_needs`: the mark's own
/// name, or the file's when the decl sits in unmarked (own) text.
fn page_name(file: &str, page: &str) -> String {
    if page.is_empty() { file } else { page }.to_string()
}

impl<'a> Pass<'a> {
    // ---- bookkeeping

    fn dec_of(&self, ident: &Ident) -> Option<DecId> {
        self.solver.node_decs.get(&ident.id).copied()
    }

    fn cur(&self) -> Caller {
        *self.callers.last().unwrap_or(&Caller::Script(0))
    }

    /// Every dec a pattern binds — `bind`'s match arms without the side
    /// effects, for provenance and param-position bookkeeping.
    fn pat_decs(&self, pat: &Pat) -> Vec<DecId> {
        let mut out = Vec::new();
        fn go(p: &Pat, out: &mut Vec<DecId>, pass: &Pass) {
            match p.kind() {
                PatKind::Ident(ident) => {
                    if let Some(dec) = pass.dec_of(ident) {
                        out.push(dec);
                    }
                }
                PatKind::Tuple(pats) | PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
                    for p in pats {
                        go(p, out, pass);
                    }
                }
                PatKind::Struct(_, fields) => {
                    for p in fields.values() {
                        go(p, out, pass);
                    }
                }
                PatKind::NullBind(p) => go(p, out, pass),
                _ => {}
            }
        }
        go(pat, &mut out, self);
        out
    }

    /// The param positions `e`'s value derives from. `w.balls` roots at
    /// `w`; `bs` bound `let bs = w.balls` stands for `w` through
    /// `provenance`; a fn dec or a literal roots at none — iterating
    /// `f(w)`'s result is still recorded as `w`-scaled work, a safe
    /// over-read for the scans fixpoint.
    fn param_roots(&self, e: &Expr) -> FxHashSet<u32> {
        struct Roots<'p, 'a> {
            pass: &'p Pass<'a>,
            out: FxHashSet<u32>,
        }
        impl Visitor for Roots<'_, '_> {
            fn ident(&mut self, ident: &Ident) {
                let Some(dec) = self.pass.dec_of(ident) else {
                    return;
                };
                if let Some(&p) = self.pass.param_pos.get(&dec) {
                    self.out.insert(p);
                }
                if let Some(ps) = self.pass.provenance.get(&dec) {
                    self.out.extend(ps.iter().copied());
                }
            }
        }
        let mut roots = Roots {
            pass: self,
            out: FxHashSet::default(),
        };
        parse::walk_expr(e, &mut roots);
        roots.out
    }

    /// Param positions a `while` header paces a collection on: the roots
    /// of places it indexes or calls into — `w.balls.len()`, `m[k]`,
    /// `q.next()` pace on the caller's collection, so the trip count
    /// scales with it. `while d > PI` paces on a scalar and isn't a
    /// scan; `while f(w)` paces on `f`, not `w`.
    fn paced_roots(&self, e: &Expr) -> FxHashSet<u32> {
        struct Paced<'p, 'a> {
            pass: &'p Pass<'a>,
            out: FxHashSet<u32>,
        }
        impl Visitor for Paced<'_, '_> {
            fn expr(&mut self, expr: &Expr) {
                let left = match expr.kind() {
                    ExprKind::Call(c) => Some(&c.left),
                    ExprKind::Access(Access::Square { left, .. }) => Some(left),
                    _ => None,
                };
                if let Some(left) = left {
                    self.out.extend(self.pass.param_roots(left));
                }
            }
        }
        let mut paced = Paced {
            pass: self,
            out: FxHashSet::default(),
        };
        parse::walk_expr(e, &mut paced);
        paced.out
    }

    /// `true` when `e` names a place bound outside the innermost loop —
    /// a shared `w` or `w.balls` re-read every pass, not a value the
    /// loop made (`b.id`, `f(w)`, a literal). A place indexed by a
    /// loop-bound key reads as per-element: `w.balls[i]` is `b` again.
    fn place_ambient(&self, e: &Expr) -> bool {
        let floor = self.loop_floors.last().copied().unwrap_or(usize::MAX);
        let inside = |pass: &Pass, dec: DecId| {
            pass.dec_scope.get(&dec).copied().unwrap_or(0) > floor
        };
        let mut spine = e;
        loop {
            match spine.kind() {
                ExprKind::Ident(i) => {
                    let Some(dec) = self.dec_of(i) else {
                        return false;
                    };
                    return !inside(self, dec);
                }
                ExprKind::Access(Access::Dot { left, .. }) => spine = left,
                ExprKind::Access(Access::Square { left, key, .. }) => {
                    struct Keys<'p, 'a> {
                        pass: &'p Pass<'a>,
                        decs: Vec<DecId>,
                    }
                    impl Visitor for Keys<'_, '_> {
                        fn ident(&mut self, ident: &Ident) {
                            if let Some(dec) = self.pass.dec_of(ident) {
                                self.decs.push(dec);
                            }
                        }
                    }
                    let mut keys = Keys {
                        pass: self,
                        decs: Vec::new(),
                    };
                    parse::walk_expr(key, &mut keys);
                    if keys.decs.iter().any(|&d| inside(self, d)) {
                        return false;
                    }
                    spine = left;
                }
                _ => return false,
            }
        }
    }

    /// `true` when `dec` was bound at or outside the innermost closure boundary -- pending
    /// bookkeeping can't decide its deadness from a body that may run zero or many times.
    fn outside_closure(&self, dec: DecId) -> bool {
        match self.closure_floors.last() {
            Some(&floor) => self.dec_scope.get(&dec).copied().unwrap_or(0) <= floor,
            None => false,
        }
    }

    /// Reading/storing `path` under `dec` after a `#[consumes]` call took an
    /// overlapping region: `r` read whole after `r.id` died counts, `r.other`
    /// doesn't. Reports once per dead path, not per touch.
    fn consume_check(&mut self, dec: DecId, path: &[String], at: Location) {
        if self.flow.diverged {
            return;
        }
        let Some(marks) = self.flow.consumed.get(&dec) else {
            return;
        };
        let Some((segs, consumed_at, by)) = marks.iter().find(|(s, ..)| paths_overlap(s, path))
        else {
            return;
        };
        if !self.consume_reported.insert((dec, segs.clone())) {
            return;
        }
        let mut name = self.solver.decs[dec].name.clone();
        for s in segs {
            name.push('.');
            name.push_str(s);
        }
        self.errors.push(
            UseAfterConsume {
                src: self.solver.src(at),
                at: at.into(),
                consumed_at: (*consumed_at).into(),
                name,
                by: by.clone(),
            }
            .into(),
        );
    }

    /// `dec` was read (by name, or through a resolved access leaf) at `at`. Its pending
    /// writes move to `read_spans` -- proof of life that vetoes a dead verdict at later
    /// merges. Reading a dec a `#[consumes]` call already took is a solve error.
    fn read(&mut self, dec: DecId, at: Location) {
        self.uses.entry(dec).or_default().reads.tick();
        // a read inside any enclosing loop's text can observe that loop's
        // body-end stores on a later pass -- the back-edge the single
        // walk can't see
        for reads in &mut self.loop_reads {
            reads.insert(dec);
        }
        // the spine root's own check already ran at the access's deeper
        // path; an lvalue spine locates the slot without reading it
        if !self.in_lvalue && !std::mem::take(&mut self.skip_root_check) {
            self.consume_check(dec, &[], at);
        }
        if self.outside_closure(dec) || self.flow.diverged {
            return;
        }
        if let Some(spans) = self.flow.pending.remove(&dec) {
            self.flow.read_spans.entry(dec).or_default().extend(spans);
        }
    }

    /// `dec` was stored to. `pend` for a real re-store (`let x = `, `x = `); field/index and
    /// compound writes tick the count without a pending slot (they observe the binding, and
    /// only `Local` decs are ever tracked -- globals/items/constants are ω or frozen anyway).
    fn write(&mut self, dec: DecId, at: Location, pend: bool) {
        self.uses.entry(dec).or_default().writes.tick();
        if pend {
            // a real re-store (`let x =`, `x =`) reborns the binding -- the consumed
            // state it carried describes the old value, not the new one
            self.flow.consumed.remove(&dec);
        }
        let local = matches!(self.solver.decs[dec].kind, DecKind::Local);
        if !pend || !local || self.flow.diverged || self.outside_closure(dec) {
            return;
        }
        if let Some(spans) = self.flow.pending.remove(&dec) {
            for loc in spans {
                if self.flow.born.remove(&loc) {
                    // this path's own write, killed unconditionally -- dead for sure
                    self.dead_candidates.push((dec, loc));
                } else {
                    // inherited span: dead on this path, but a sibling may still hold
                    // it pending or have read it -- the merge arbitrates
                    self.flow.killed.entry(dec).or_default().insert(loc);
                }
            }
        }
        self.flow.pending.entry(dec).or_default().insert(at);
        self.flow.born.insert(at);
    }

    /// A write landed inside `dec`'s storage at `path` -- `w.tour.f = `,
    /// `xs[i] = `, `xs.push(..)`. Param decs seed `own_writes` with their
    /// position; a derived binding (`let xs2 = w.list`) marks each param
    /// it stands for at `[]`, the path under `w` lost to provenance.
    /// r[impl mimas.fn.writes]
    fn member_write(&mut self, dec: DecId, path: Vec<GuardSeg>) {
        if let Some(&p) = self.param_pos.get(&dec) {
            self.own_writes
                .entry(self.cur())
                .or_default()
                .push((p, path.clone()));
        } else if let Some(ps) = self.provenance.get(&dec).cloned() {
            for p in ps {
                self.own_writes
                    .entry(self.cur())
                    .or_default()
                    .push((p, Vec::new()));
            }
        }
        // a dec bound outside the innermost `let` closure is that
        // closure's caller-visible write set -- `poke()` inside a
        // frozen loop resolves `poke`'s dec to this list.
        if self.outside_closure(dec) {
            for frame in self.closure_stack.iter().flatten() {
                self.closure_writes
                    .entry(*frame)
                    .or_default()
                    .push((dec, path.clone()));
            }
        }
    }

    /// `x = v` / `x op= v` re-stored `dec`'s binding. Only an `inout`
    /// param's store is caller-visible (it writes the lent slot); a
    /// plain param's dies at the body's end -- the dead-store lint's
    /// domain, not the write set's.
    fn binding_write(&mut self, dec: DecId) {
        if self.param_pos.contains_key(&dec) {
            if self.solver.inout_decs.contains(&dec) {
                self.own_writes
                    .entry(self.cur())
                    .or_default()
                    .push((self.param_pos[&dec], Vec::new()));
            } else {
                // `n = 0` on a by-value param rebinds the local copy -- the
                // caller's argument never sees it.
                self.param_stores.insert(dec);
            }
        }
        if self.outside_closure(dec) {
            for frame in self.closure_stack.iter().flatten() {
                self.closure_writes
                    .entry(*frame)
                    .or_default()
                    .push((dec, Vec::new()));
            }
        }
    }

    /// `(param position, access path)` an argument contributes to a
    /// call's write map: `w.tour` gives `w`'s position under `[tour]`; a
    /// value with no nameable root contributes nothing; a binding that
    /// only stands for a param through provenance gives `[]` -- the
    /// precise field path is lost, so its writes widen to the param.
    fn write_arg(&self, e: &Expr) -> Vec<(u32, Vec<GuardSeg>)> {
        let roots = self.param_roots(e);
        // a `&w.tour` lend records its pre-`&` expr in `place_sources` --
        // the argument's own node is the minted `__borrow_place` call.
        let e = self.solver.place_sources.get(&e.id()).unwrap_or(e);
        let (root_dec, path) = crate::root_and_path(e)
            .and_then(|(root, p)| self.dec_of(root).map(|d| (Some(d), p)))
            .unwrap_or((None, Vec::new()));
        roots
            .iter()
            .map(|&p| {
                (
                    p,
                    if root_dec.is_some_and(|d| self.param_pos.get(&d) == Some(&p)) {
                        path.clone()
                    } else {
                        Vec::new()
                    },
                )
            })
            .collect()
    }

    fn push_scope(&mut self) {
        self.scope_decs.push(Vec::new());
    }

    /// The scope's own decs die here. A write span is a dead store when no surviving
    /// path (this flow, any that retired early via `return`/`raise`/`let-else`, and the
    /// innermost loop's `break`/`continue` exits -- the only exit flows that can carry
    /// writes for decs this young) still carries it pending or read it -- writes only
    /// ever live on paths that stored them, so pending and killed spans both count as
    /// deaths.
    fn pop_scope(&mut self) {
        let Some(decs) = self.scope_decs.pop() else {
            return;
        };
        for dec in decs {
            let mut spans: FxHashSet<Location> = FxHashSet::default();
            let mut live: FxHashSet<Location> = FxHashSet::default();
            let exits = self.breaks.last().into_iter().flatten();
            for flow in std::iter::once(&self.flow)
                .chain(self.retired.iter())
                .chain(exits)
            {
                if let Some(s) = flow.pending.get(&dec) {
                    spans.extend(s.iter().copied());
                }
                if let Some(s) = flow.killed.get(&dec) {
                    spans.extend(s.iter().copied());
                }
                if let Some(s) = flow.read_spans.get(&dec) {
                    live.extend(s.iter().copied());
                }
            }
            for &loc in spans.difference(&live) {
                self.dead_candidates.push((dec, loc));
            }
            self.flow.pending.remove(&dec);
            self.flow.killed.remove(&dec);
            self.flow.read_spans.remove(&dec);
            self.flow.consumed.remove(&dec);
            for flow in self
                .retired
                .iter_mut()
                .chain(self.breaks.last_mut().into_iter().flatten())
            {
                flow.pending.remove(&dec);
                flow.killed.remove(&dec);
                flow.read_spans.remove(&dec);
                flow.consumed.remove(&dec);
            }
        }
    }

    /// `pend` for a real store (`let x = v`) -- the write lands in pending until a read
    /// consumes it. Params, `for` bindings and pattern binds count as written but never pend:
    /// nothing can sit between their declaration and a read.
    fn bind(&mut self, pat: &Pat, site: Site, pend: bool) {
        match pat.kind() {
            PatKind::Ident(ident) => {
                let Some(dec) = self.dec_of(ident) else {
                    return;
                };
                self.lint_targets.push((dec, site));
                self.dec_scope.insert(dec, self.scope_decs.len());
                if let Some(scope) = self.scope_decs.last_mut() {
                    scope.push(dec);
                }
                self.write(dec, ident.location, pend);
            }
            PatKind::Tuple(pats) | PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
                for p in pats {
                    self.bind(p, site, pend);
                }
            }
            PatKind::Struct(_, fields) => {
                for p in fields.values() {
                    self.bind(p, site, pend);
                }
            }
            PatKind::NullBind(p) => self.bind(p, site, pend),
            _ => {}
        }
    }

    // ---- lints (run once the whole ast has been walked and counts are final)

    fn finish_lints(&mut self) {
        for &(dec, site) in &self.lint_targets {
            let d = &self.solver.decs[dec];
            if !matches!(d.kind, DecKind::Local | DecKind::LoopVar) {
                continue;
            }
            if d.name.starts_with('_') {
                continue;
            }
            let reads = self.uses.get(&dec).map_or(Use::Never, |u| u.reads);
            if reads != Use::Never {
                continue;
            }
            let at = d.location;
            let src = self.solver.src(at);
            let report: Report = match site {
                // `fn bump(n: int) { n = 0 }` -- `n` was stored, not just
                // ignored: say the store stays in the function rather than
                // the weaker "never read".
                // r[impl mimas.lint.param-write]
                Site::Param if self.param_stores.contains(&dec) => ParamWriteDies {
                    src,
                    at: at.into(),
                    name: d.name.clone(),
                }
                .into(),
                Site::Param => UnusedParam {
                    src,
                    at: at.into(),
                    name: d.name.clone(),
                }
                .into(),
                Site::Binding => UnusedBinding {
                    src,
                    at: at.into(),
                    name: d.name.clone(),
                }
                .into(),
            };
            self.warnings.push(report);
        }
        let mut seen_deads = FxHashSet::default();
        for &(dec, at) in &self.dead_candidates {
            let reads = self.uses.get(&dec).map_or(Use::Never, |u| u.reads);
            if reads == Use::Never {
                // `let x = v` overwritten without a read is the unused-binding warning's job
                continue;
            }
            // both arms of an `if` can kill the same earlier write -- report it once
            if !seen_deads.insert((dec, at)) {
                continue;
            }
            let d = &self.solver.decs[dec];
            // a param's binding is a local copy -- `n = 0` unread at the
            // body's end never reached the caller, which is a different
            // story than a dead local and worth saying differently.
            // r[impl mimas.lint.param-write]
            if self.param_pos.contains_key(&dec)
                && !self.solver.inout_decs.contains(&dec)
            {
                self.warnings.push(
                    ParamWriteDies {
                        src: self.solver.src(at),
                        at: at.into(),
                        name: d.name.clone(),
                    }
                    .into(),
                );
                continue;
            }
            self.warnings.push(
                DeadStore {
                    src: self.solver.src(at),
                    at: at.into(),
                    name: d.name.clone(),
                }
                .into(),
            );
        }
    }

    /// A `#[consumes]` slot took `arg`. Kills the named region of the binding's
    /// dec: `f(r.id)` dies `r.id` (and `r` read whole afterwards), `f(r)` dies
    /// `r` outright. Globals are shared with the entry frame, captures and
    /// loop-outer decs can't die somewhere they'll be read again, and an index
    /// can't name the element it hit.
    fn consume_arg(&mut self, arg: &'a Expr, by: &str) {
        let Some((root, path)) = crate::root_and_path(arg) else {
            // a computed value has no binding to kill -- consumed for free
            return;
        };
        let Some(dec) = self.dec_of(root) else {
            return;
        };
        let name = self.solver.decs[dec].name.clone();
        let fields = field_prefix(&path);
        let fail = |pass: &mut Self, why: &str| {
            if pass.consume_reported.insert((dec, fields.clone())) {
                pass.errors.push(
                    ConsumeForbidden {
                        src: pass.solver.src(arg.location()),
                        at: arg.location().into(),
                        name: name.clone(),
                        why: why.to_string(),
                    }
                    .into(),
                );
            }
        };
        if fields.len() != path.len() {
            fail(
                self,
                "an element's index isn't fixed -- bind it (`let x = a[i]`) first",
            );
            return;
        }
        match self.solver.decs[dec].kind {
            DecKind::Local | DecKind::LoopVar => {}
            DecKind::Global => {
                fail(
                    self,
                    "globals live in the shared entry frame -- they can't be consumed",
                );
                return;
            }
            _ => {
                fail(self, "only a local binding can be consumed");
                return;
            }
        }
        if self.outside_closure(dec) {
            fail(
                self,
                "it's bound outside this closure -- the closure may run more than once",
            );
            return;
        }
        if let Some(&floor) = self.loop_floors.last()
            && self.dec_scope.get(&dec).copied().unwrap_or(0) <= floor
        {
            fail(
                self,
                "it's bound outside this loop -- consuming it would poison later iterations",
            );
            return;
        }
        self.uses.entry(dec).or_default().reads.tick();
        // a whole-binding mark subsumes every mark under it; a field mark
        // is subsumed by an earlier whole one
        let marks = self.flow.consumed.entry(dec).or_default();
        if fields.is_empty() {
            marks.clear();
            marks.push((fields, arg.location(), by.to_string()));
        } else if !marks.iter().any(|(s, ..)| s.is_empty()) {
            marks.push((fields, arg.location(), by.to_string()));
        }
    }

    // ---- calls and effects

    /// The dec a call dispatches to, when the solver resolved one: bare `f()`, `x.m()`,
    /// `Type::assoc()`, `self.m()`.
    fn callee_dec(&self, callee: &Expr) -> Option<DecId> {
        match callee.kind() {
            ExprKind::Ident(i) => self.dec_of(i),
            ExprKind::Access(Access::Dot { right, .. }) => right
                .as_ident()
                .and_then(|i| self.solver.node_decs.get(&i.id).copied()),
            ExprKind::Access(Access::DoubleColon { right, .. })
            | ExprKind::Access(Access::Identity { right }) => {
                self.solver.node_decs.get(&right.id).copied()
            }
            _ => None,
        }
    }

    fn call_effect(&mut self, call_expr: &Expr) {
        let caller = self.cur();
        let fx = match self.callee_dec(call_expr) {
            None => Fx::unknown(),
            Some(dec) => {
                if let Some(binding) = self.solver.dec_to_native.get(&dec) {
                    // `None` = the native never declared `#[effects]` -- fail closed
                    binding.sig.effects.unwrap_or_else(Fx::unknown)
                } else if self.pact_members.contains(&dec) {
                    // the page is a closed world: every implementer is known,
                    // so a pact-dispatched call edges to each method dec and
                    // the fixpoint unions their effects transitively (nested
                    // pact calls fold in the same way, cycles settle like
                    // ordinary recursion). r[impl mimas.fx.pact-call]
                    for (mdec, _) in self.solver.member_impl_decs(dec) {
                        self.edges.entry(caller).or_default().insert(mdec);
                    }
                    Fx::empty()
                } else {
                    match self.solver.decs[dec].kind {
                        DecKind::Item { .. } => {
                            self.edges.entry(caller).or_default().insert(dec);
                            Fx::empty()
                        }
                        // type/variant construction and constants do no work
                        DecKind::Adt(_) | DecKind::Variant { .. } | DecKind::Constant(_) => {
                            Fx::empty()
                        }
                        // a fn-typed local, global, or closure value -- unknowable
                        _ => Fx::unknown(),
                    }
                }
            }
        };
        if !fx.is_empty() {
            *self.own_fx.entry(caller).or_insert(Fx::empty()) |= fx;
        }
    }

    // ---- the walk

    fn run(&mut self, stmts: &'a [Stmt]) {
        for stmt in stmts {
            self.stmt(stmt);
        }
    }

    /// The file's top-level statements: each gets its own script-caller slot so
    /// its effect set lands as a segment a host can attribute back to a `use`
    /// include (nested blocks just contribute to their statement's slot).
    fn run_top(&mut self, stmts: &'a [Stmt]) {
        for stmt in stmts {
            let idx = self.script_sites.len();
            self.script_sites.push(stmt.location());
            self.callers[0] = Caller::Script(idx);
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, stmt: &'a Stmt) {
        if let Some(rewritten) = self.solver.rewrites.get(&stmt.id()) {
            self.expr(rewritten);
            return;
        }
        match stmt.kind() {
            StmtKind::Let(l) => {
                let roots = self.param_roots(&l.right);
                // a `let f = || …` gives the closure body's captured
                // writes a name to resolve at `f(..)` call sites
                self.pending_closure_dec = matches!(l.right.kind(), ExprKind::Closure(_))
                    .then(|| self.pat_decs(&l.left).first().copied())
                    .flatten();
                self.expr(&l.right);
                self.pending_closure_dec = None;
                if let Some(else_branch) = &l.else_branch {
                    // `else` always diverges, but its stores still die with this scope
                    let pre = self.flow.clone();
                    self.flow = pre.fork();
                    self.expr(else_branch);
                    let end = std::mem::replace(&mut self.flow, pre);
                    self.retired.push(end);
                }
                self.bind(&l.left, Site::Binding, true);
                if !roots.is_empty() {
                    for d in self.pat_decs(&l.left) {
                        self.provenance.insert(d, roots.clone());
                    }
                }
            }
            StmtKind::Assignment(a) => {
                match a.left.kind() {
                    ExprKind::Ident(ident) => {
                        // `x = v`: the ident is the target, not a read. `x += v` and friends
                        // read the old value first. A pure write target
                        // (`n = 0` on an `inout`) never lands in `node_decs`
                        // -- it is resolved, not read.
                        let dec = self
                            .dec_of(ident)
                            .or_else(|| self.solver.ribs.resolve(ident));
                        if !matches!(a.op, AssignmentOp::Identity)
                            && let Some(dec) = dec
                        {
                            self.read(dec, ident.location);
                        }
                        let roots = self.param_roots(&a.right);
                        self.expr(&a.right);
                        if let Some(dec) = dec {
                            self.write(dec, ident.location, true);
                            self.binding_write(dec);
                            // `x = v` re-roots x to v's params (`x = []`
                            // clears them); `x += v` only adds.
                            match a.op {
                                AssignmentOp::Identity if roots.is_empty() => {
                                    self.provenance.remove(&dec);
                                }
                                AssignmentOp::Identity => {
                                    self.provenance.insert(dec, roots);
                                }
                                _ if !roots.is_empty() => {
                                    self.provenance
                                        .entry(dec)
                                        .or_default()
                                        .extend(roots);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {
                        // `x.f = v` / `x[i] = v`: a pure-field target revives its
                        // subtree (`s.f` dead is a store, not a read -- only a
                        // *dead ancestor*, or a wholly dead `s`, still errors).
                        // `x.f += v` reads the slot too, and an index-terminated
                        // path can't name its element -- both take the ordinary
                        // overlap check at the field prefix. The spine is walked
                        // with checks off: it locates the slot, it doesn't read it.
                        if let Some((root, path)) = crate::root_and_path(&a.left)
                            && let Some(dec) = self.dec_of(root)
                            && !self.flow.diverged
                            && let Some(marks) = self.flow.consumed.get(&dec)
                        {
                            let fields = field_prefix(&path);
                            let pure_field = fields.len() == path.len() && !fields.is_empty();
                            let reads_slot = !matches!(a.op, AssignmentOp::Identity) || !pure_field;
                            if let Some((segs, consumed_at, by)) = marks.iter().find(|(s, ..)| {
                                paths_overlap(s, &fields) && (reads_slot || s.len() < fields.len())
                            }) && self.consume_reported.insert((dec, segs.clone()))
                            {
                                let mut name = self.solver.decs[dec].name.clone();
                                for s in segs {
                                    name.push('.');
                                    name.push_str(s);
                                }
                                self.errors.push(
                                    UseAfterConsume {
                                        src: self.solver.src(a.left.location()),
                                        at: a.left.location().into(),
                                        consumed_at: (*consumed_at).into(),
                                        name,
                                        by: by.clone(),
                                    }
                                    .into(),
                                );
                            }
                            // the store reborns its subtree: marks at or under
                            // the target die (a whole-binding mark never does)
                            if pure_field && let Some(ms) = self.flow.consumed.get_mut(&dec) {
                                ms.retain(|(s, ..)| {
                                    s.is_empty() || !s.starts_with(fields.as_slice())
                                });
                            }
                        }
                        let lv = std::mem::replace(&mut self.in_lvalue, true);
                        self.expr(&a.left);
                        self.in_lvalue = lv;
                        self.expr(&a.right);
                        if let Some((root, path)) = crate::root_and_path(&a.left)
                            && let Some(dec) = self.dec_of(root)
                        {
                            self.write(dec, root.location, false);
                            self.member_write(dec, path);
                        }
                    }
                }
            }
            StmtKind::Needs(needs) => {
                // the page's declared authority -- checked against its inferred
                // set after the fixpoint (`mimas.fx.needs`)
                self.needs.push((stmt.location(), needs.needs_fx()));
            }
            StmtKind::Expr(e) => {
                // a `#[must_use]` native's result dropped on the floor warns
                // (`let _x = f()` still counts as a use -- Rust agrees)
                if let ExprKind::Call(c) = e.kind()
                    && let Some(dec) = self.callee_dec(&c.left)
                    && let Some(binding) = self.solver.dec_to_native.get(&dec)
                    && binding.sig.must_use
                {
                    self.warnings.push(
                        MustUseResult {
                            src: self.solver.src(e.location()),
                            at: e.location().into(),
                            name: self.solver.decs[dec].name.clone(),
                        }
                        .into(),
                    );
                }
                self.expr(e);
            }
            StmtKind::Item(item) => self.item(item),
            StmtKind::Module(_) | StmtKind::Unit(_) => {}
        }
    }

    fn item(&mut self, item: &'a Item) {
        match item.kind() {
            ItemKind::Use(parse::Use::Host(host)) => {
                // a `granting {…}` clause: the page hands the set down to the
                // include -- checked after the fixpoint (`mimas.use.page.grant-check`)
                if let Some(fx) = host.grants_fx() {
                    self.grants.push((item.location(), host.name.clone(), fx));
                }
            }
            ItemKind::Function(f) => {
                if item.attrs().iter().any(|a| a.name.lexeme == "scans")
                    && let Some(&d) = self
                        .dec_by_site
                        .get(&(f.name.location.file_id, f.name.location.span.start))
                {
                    self.scan_marked.insert(d);
                }
                self.function(f);
            }
            ItemKind::Const(c) => self.expr(&c.right),
            ItemKind::Impl(imp) => {
                let was = std::mem::replace(&mut self.in_pact_impl, imp.pact.is_some());
                for inner in &imp.items {
                    self.item(inner);
                }
                self.in_pact_impl = was;
            }
            ItemKind::Pact(pact) => {
                for member in &pact.items {
                    if let PactItem::Fn {
                        name,
                        parameters,
                        default: Some(body),
                        ..
                    } = member
                    {
                        let dec = self
                            .dec_by_site
                            .get(&(name.location.file_id, name.location.span.start))
                            .copied();
                        let caller = dec.map_or_else(|| self.cur(), Caller::Fn);
                        self.callers.push(caller);
                        self.own_fx.entry(caller).or_insert(Fx::empty());
                        self.in_fn(|s| {
                            // the pact fixes a default method's parameters: every
                            // impl takes them, so one the default ignores (`self`
                            // included) is the contract, not a mistake to lint
                            // r[impl mimas.lint.pact-default]
                            let linted = s.lint_targets.len();
                            for (i, p) in parameters.iter().enumerate() {
                                s.bind(&p.left, Site::Param, false);
                                for d in s.pat_decs(&p.left) {
                                    s.param_pos.insert(d, i as u32);
                                }
                            }
                            s.lint_targets.truncate(linted);
                            s.expr(body);
                        });
                        self.callers.pop();
                    }
                }
            }
            ItemKind::Tests(tests) => {
                for case in &tests.cases {
                    self.expr(&case.expr);
                }
            }
            _ => {}
        }
    }

    /// A body walked with a fresh flow and scopes, then thrown away: fn bodies are
    /// control-flow islands -- their pending writes and divergences never leak to the
    /// enclosing text.
    fn in_fn(&mut self, f: impl FnOnce(&mut Self)) {
        let saved = (
            std::mem::take(&mut self.flow),
            std::mem::take(&mut self.scope_decs),
            std::mem::take(&mut self.retired),
            std::mem::take(&mut self.breaks),
            std::mem::take(&mut self.closure_floors),
            std::mem::take(&mut self.loop_reads),
        );
        self.push_scope();
        f(self);
        self.pop_scope();
        (
            self.flow,
            self.scope_decs,
            self.retired,
            self.breaks,
            self.closure_floors,
            self.loop_reads,
        ) = saved;
    }

    fn function(&mut self, f: &'a Function) {
        // `and`/`or`/`not` are operator words unless the source declares a
        // fn by that name (the lexer's pre-scan) -- allowed, but said
        if let Some(symbol) = parse::lex::shadowable_word_symbol(&f.name.lexeme) {
            self.warnings.push(
                ShadowedBuiltin {
                    src: self.solver.src(f.name.location),
                    at: f.name.location.into(),
                    name: f.name.lexeme.to_string(),
                    symbol,
                }
                .into(),
            );
        }
        let dec = self
            .dec_by_site
            .get(&(f.name.location.file_id, f.name.location.span.start))
            .copied();
        let caller = dec.map_or_else(|| self.cur(), Caller::Fn);
        self.callers.push(caller);
        // an entry even when the body ends up pure -- `fn_effects` should answer for
        // every walked fn, not only effectful ones
        self.own_fx.entry(caller).or_insert(Fx::empty());
        let pact_sig = std::mem::take(&mut self.in_pact_impl);
        self.in_fn(|s| {
            // r[impl mimas.lint.pact-default]
            let linted = s.lint_targets.len();
            for (i, p) in f.parameters.iter().enumerate() {
                s.bind(&p.left, Site::Param, false);
                for d in s.pat_decs(&p.left) {
                    s.param_pos.insert(d, i as u32);
                }
            }
            if pact_sig {
                s.lint_targets.truncate(linted);
            }
            s.expr(&f.body);
        });
        self.in_pact_impl = pact_sig;
        self.callers.pop();
    }

    fn expr(&mut self, e: &'a Expr) {
        if let Some(rewritten) = self.solver.rewrites.get(&e.id()) {
            self.expr(rewritten);
            return;
        }
        match e.kind() {
            ExprKind::Ident(ident) => {
                if let Some(dec) = self.dec_of(ident) {
                    self.read(dec, ident.location);
                }
            }
            ExprKind::Literal(l) => self.literal(l),
            ExprKind::Grouping(g) => self.expr(&g.inner),
            ExprKind::Unary(u) => self.expr(&u.right),
            ExprKind::Evaluation(ev) => {
                self.expr(&ev.left);
                self.expr(&ev.right);
            }
            ExprKind::Equality(eq) => {
                self.expr(&eq.left);
                self.expr(&eq.right);
            }
            ExprKind::Logical(lg) => {
                self.expr(&lg.left);
                self.expr(&lg.right);
            }
            ExprKind::Coalescence(c) => {
                self.expr(&c.left);
                self.expr(&c.right);
            }
            ExprKind::Unwrap(u) => self.expr(&u.expr),
            ExprKind::Propagate(d) => self.expr(&d.expr),
            ExprKind::Absolve(a) => {
                self.expr(&a.left);
                let pre = self.flow.clone();
                self.flow = pre.fork();
                self.expr(&a.handler);
                let h_end = std::mem::replace(&mut self.flow, pre.clone());
                self.flow = Flow::merge([h_end, pre]);
            }
            ExprKind::Block(b) => {
                self.push_scope();
                self.run(&b.body);
                if let Some(y) = &b.yielded_expr {
                    self.expr(y);
                }
                self.pop_scope();
            }
            ExprKind::If(i) => {
                self.expr(&i.condition);
                let pre = self.flow.clone();
                self.flow = pre.fork();
                self.push_scope();
                if let Some(b) = &i.binding {
                    self.bind(b, Site::Binding, false);
                }
                self.expr(&i.main_body);
                self.pop_scope();
                let then_end = std::mem::replace(&mut self.flow, pre.fork());
                let else_end = match &i.else_expr {
                    Some(e) => {
                        self.expr(e);
                        self.flow.clone()
                    }
                    None => pre,
                };
                self.flow = Flow::merge([then_end, else_end]);
            }
            ExprKind::Match(m) => {
                self.expr(&m.identity);
                let pre = self.flow.clone();
                let mut arms = Vec::with_capacity(m.cases.len());
                for case in &m.cases {
                    self.flow = pre.fork();
                    self.push_scope();
                    self.bind(case.pat(), Site::Binding, false);
                    if let Some(g) = case.guard() {
                        self.expr(g);
                    }
                    self.expr(case.body());
                    self.pop_scope();
                    arms.push(self.flow.clone());
                }
                self.flow = Flow::merge(arms);
            }
            ExprKind::Call(c) => {
                self.expr(&c.left);
                // `x.m(…)` evaluates the whole receiver -- the left walk already
                // checked `x.m`, but a partial death under `x` still blocks it
                if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind()
                    && right
                        .as_ident()
                        .is_some_and(|i| self.solver.node_decs.contains_key(&i.id))
                    && let Some((rroot, rpath)) = crate::root_and_path(left)
                    && let Some(rdec) = self.dec_of(rroot)
                {
                    self.consume_check(rdec, &field_prefix(&rpath), left.location());
                }
                let callee = self.callee_dec(&c.left);
                let binding = callee.and_then(|d| self.solver.dec_to_native.get(&d));
                let callee_name = callee
                    .map(|d| self.solver.decs[d].name.clone())
                    .unwrap_or_default();
                // arg eval is left-to-right: a consuming slot kills the binding before
                // any later arg's walk can read it (`f(close_me(x), x)` errors)
                for (i, arg) in c.arguments.iter().enumerate() {
                    self.expr(&arg.value);
                    if binding.is_some_and(|b| b.sig.consumes.get(i).copied().unwrap_or(false)) {
                        self.consume_arg(&arg.value, &callee_name);
                    }
                }
                if binding.is_some_and(|b| b.sig.consumes_recv)
                    && let ExprKind::Access(Access::Dot { left, .. }) = c.left.kind()
                {
                    self.consume_arg(left, &callee_name);
                }
                // `xs.push(..)` mutates its receiver -- the binding was both read and written
                if let ExprKind::Access(Access::Dot { left, .. }) = c.left.kind()
                    && binding.is_some_and(|b| b.mutates_recv)
                    && let Some((root, rpath)) = crate::root_and_path(left)
                    && let Some(root_dec) = self.dec_of(root)
                {
                    self.write(root_dec, root.location, false);
                    self.member_write(root_dec, rpath);
                }
                // `n = 0` on an `inout` param solved into `n.write(0)`:
                // a whole-slot store through the lent place, so the write
                // set records the param's `[]` path. `p.remove()` counts
                // the same for removable places. r[impl mimas.fn.writes]
                if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind()
                    && right
                        .as_ident()
                        .is_some_and(|i| matches!(i.lexeme.as_str(), "write" | "remove"))
                    && let Some((root, _)) = crate::root_and_path(left)
                    && let Some(root_dec) = self.dec_of(root).or_else(|| self.solver.ribs.resolve(root))
                    && self.solver.inout_decs.contains(&root_dec)
                    && let Some(&p) = self.param_pos.get(&root_dec)
                {
                    self.own_writes
                        .entry(self.cur())
                        .or_default()
                        .push((p, Vec::new()));
                }
                if callee_name == "__cleanup_register"
                    && let Some(action) = c.arguments.first()
                    && !matches!(action.value.kind(), ExprKind::Closure(_))
                {
                    self.call_effect(&action.value);
                }
                self.call_effect(&c.left);
                // A call under an enclosing loop is quadratic iff the
                // callee iterates a shared place — a verdict that needs
                // the whole call graph, so every site records its arg
                // roots here and judges after the fixpoint. The
                // receiver counts as position 0 on `x.m(..)`, matching
                // `self` in the method's parameters; a pact call names
                // every impl dec, since the pact dec owns no edges.
                // r[impl mimas.lint.loop-call]
                if let Some(d) = callee {
                    let mut positions = Vec::with_capacity(c.arguments.len() + 1);
                    if let ExprKind::Access(Access::Dot { left, .. }) = c.left.kind() {
                        positions.push(self.param_roots(left));
                    }
                    for arg in &c.arguments {
                        positions.push(self.param_roots(&arg.value));
                    }
                    let mut ambients = Vec::new();
                    if !self.loop_reads.is_empty() {
                        ambients.reserve(c.arguments.len() + 1);
                        if let ExprKind::Access(Access::Dot { left, .. }) = c.left.kind() {
                            ambients.push(self.place_ambient(left));
                        }
                        for arg in &c.arguments {
                            ambients.push(self.place_ambient(&arg.value));
                        }
                    }
                    let mut wpositions = Vec::with_capacity(c.arguments.len() + 1);
                    if let ExprKind::Access(Access::Dot { left, .. }) = c.left.kind() {
                        wpositions.push(self.write_arg(left));
                    }
                    for arg in &c.arguments {
                        wpositions.push(self.write_arg(&arg.value));
                    }
                    if self.pact_members.contains(&d) {
                        for (m, _) in self.solver.member_impl_decs(d) {
                            self.call_args.push((self.cur(), m, positions.clone()));
                            self.call_write_args
                                .push((self.cur(), m, wpositions.clone()));
                            if !self.loop_reads.is_empty() {
                                self.loop_calls.push((
                                    e.location(),
                                    self.cur(),
                                    m,
                                    ambients.clone(),
                                ));
                            }
                        }
                    } else {
                        self.call_args.push((self.cur(), d, positions));
                        self.call_write_args.push((self.cur(), d, wpositions));
                        if !self.loop_reads.is_empty() {
                            self.loop_calls
                                .push((e.location(), self.cur(), d, ambients));
                        }
                    }
                }
            }
            ExprKind::Access(Access::Dot { left, right, .. }) => {
                // `r.id` reads `r.id`, not all of `r` -- check the place at its
                // own field path, then let the spine's root ident skip its bare
                // check (its read already happened, deeper, right here)
                if !self.in_lvalue
                    && let Some((root, path)) = crate::root_and_path(e)
                    && let Some(dec) = self.dec_of(root)
                {
                    self.consume_check(dec, &field_prefix(&path), e.location());
                }
                self.skip_root_check = true;
                self.expr(left);
                self.skip_root_check = false;
                self.expr(right);
            }
            ExprKind::Access(Access::Square { left, key, .. }) => {
                // `a.b[i]` overlaps a consumed mark up to the index -- the
                // element itself can't be named
                if !self.in_lvalue
                    && let Some((root, path)) = crate::root_and_path(e)
                    && let Some(dec) = self.dec_of(root)
                {
                    self.consume_check(dec, &field_prefix(&path), e.location());
                }
                self.skip_root_check = true;
                self.expr(left);
                self.skip_root_check = false;
                // index keys evaluate normally even inside an lvalue spine
                let lv = std::mem::replace(&mut self.in_lvalue, false);
                self.expr(key);
                self.in_lvalue = lv;
            }
            ExprKind::Access(Access::DoubleColon { left, right }) => {
                self.expr(left);
                if let Some(dec) = self.solver.node_decs.get(&right.id).copied() {
                    self.uses.entry(dec).or_default().reads.tick();
                }
            }
            ExprKind::Access(Access::Identity { right }) => {
                if let Some(dec) = self.solver.node_decs.get(&right.id).copied() {
                    self.uses.entry(dec).or_default().reads.tick();
                }
            }
            ExprKind::Closure(c) => {
                // captures are reads of unknown multiplicity -- ω them
                // r[impl mimas.lint.capture-reads]
                // and a capture reads the store it captures: the pending
                // write is live (reads inside the body skip outer decs)
                if let Some(caps) = self.solver.closure_captures.get(&e.id()) {
                    for &dec in caps {
                        self.uses.entry(dec).or_default().reads = Use::Many;
                        for reads in &mut self.loop_reads {
                            reads.insert(dec);
                        }
                        if let Some(spans) = self.flow.pending.remove(&dec) {
                            self.flow.read_spans.entry(dec).or_default().extend(spans);
                        }
                    }
                }
                let saved_retired = std::mem::take(&mut self.retired);
                let saved_breaks = std::mem::take(&mut self.breaks);
                let diverged = self.flow.diverged;
                self.closure_floors.push(self.scope_decs.len());
                self.closure_stack.push(self.pending_closure_dec.take());
                self.push_scope();
                for p in &c.parameters {
                    self.bind(&p.left, Site::Param, false);
                }
                self.expr(&c.body);
                self.pop_scope();
                self.closure_stack.pop();
                self.closure_floors.pop();
                self.flow.diverged = diverged;
                self.retired = saved_retired;
                self.breaks = saved_breaks;
            }
            ExprKind::For(f) => {
                self.own_loops.insert(self.cur());
                // the iterated collection's param roots are this
                // caller's own scans; the bound element inherits them
                // as provenance (`for b in w.balls` makes `b` a `w`).
                // A literal collection runs a bounded count of passes
                // however big the world gets — `for t in [w.machines,
                // w.gates]` is the kinds' width, not a `w` scan — while
                // the element still reads as a `w` value, so a loop
                // over IT scales.
                let roots = self.param_roots(&f.iterator);
                if !roots.is_empty() && !matches!(f.iterator.kind(), ExprKind::Literal(_)) {
                    self.own_scans
                        .entry(self.cur())
                        .or_default()
                        .extend(roots.iter().copied());
                }
                self.expr(&f.iterator);
                let pre = self.flow.clone();
                // the body may run zero times: its kills need the merge's arbitration
                self.flow.born.clear();
                self.breaks.push(Vec::new());
                self.loop_floors.push(self.scope_decs.len());
                self.loop_reads.push(FxHashSet::default());
                self.push_scope();
                self.bind(&f.binding, Site::Binding, false);
                if !roots.is_empty() {
                    for d in self.pat_decs(&f.binding) {
                        self.provenance.insert(d, roots.clone());
                    }
                }
                self.expr(&f.body);
                self.pop_scope();
                self.loop_floors.pop();
                let body_end = std::mem::take(&mut self.flow);
                // a `for` may run zero times: the pre-loop state exits too
                self.end_loop(Some(pre), body_end);
            }
            ExprKind::While(w) => {
                self.own_loops.insert(self.cur());
                // `while i < w.balls.len()` paces itself on `w` — that
                // is a `w` scan, the same as `for b in w.balls`
                let roots = self.paced_roots(&w.header);
                if !roots.is_empty() {
                    self.own_scans
                        .entry(self.cur())
                        .or_default()
                        .extend(roots.iter().copied());
                }
                // the header re-evaluates per pass, so its reads count
                // toward the back-edge set too
                self.loop_reads.push(FxHashSet::default());
                self.expr(&w.header);
                let pre = self.flow.clone();
                self.flow.born.clear();
                self.breaks.push(Vec::new());
                self.loop_floors.push(self.scope_decs.len());
                self.push_scope();
                if let Some(b) = &w.binding {
                    self.bind(b, Site::Binding, false);
                    if !roots.is_empty() {
                        for d in self.pat_decs(b) {
                            self.provenance.insert(d, roots.clone());
                        }
                    }
                }
                self.expr(&w.body);
                self.pop_scope();
                self.loop_floors.pop();
                let body_end = std::mem::take(&mut self.flow);
                self.end_loop(Some(pre), body_end);
            }
            ExprKind::Loop(l) => {
                self.own_loops.insert(self.cur());
                self.breaks.push(Vec::new());
                self.loop_floors.push(self.scope_decs.len());
                self.loop_reads.push(FxHashSet::default());
                self.expr(&l.body);
                self.loop_floors.pop();
                let mut body_end = self.flow.clone();
                let mut exits = self.breaks.pop().unwrap_or_default();
                for flow in exits.iter_mut().chain(std::iter::once(&mut body_end)) {
                    self.absorb_back_edge(flow);
                }
                self.loop_reads.pop();
                // `loop` with no `break` never falls through
                if exits.is_empty() {
                    self.flow.diverged = true;
                } else {
                    let mut arms = exits;
                    arms.push(body_end);
                    self.flow = Flow::merge(arms);
                }
            }
            ExprKind::Collect(c) => self.expr(&c.value),
            ExprKind::Return(r) => {
                if let Some(v) = &r.value {
                    self.expr(v);
                }
                self.retired.push(self.flow.clone());
                self.flow.diverged = true;
            }
            ExprKind::Raise(r) => {
                self.expr(&r.value);
                self.retired.push(self.flow.clone());
                self.flow.diverged = true;
            }
            ExprKind::Break(b) => {
                if let Some(v) = &b.value {
                    self.expr(v);
                }
                if let Some(exits) = self.breaks.last_mut() {
                    exits.push(self.flow.clone());
                }
                self.flow.diverged = true;
            }
            ExprKind::Continue(_) => {
                if let Some(exits) = self.breaks.last_mut() {
                    exits.push(self.flow.clone());
                }
                self.flow.diverged = true;
            }
            ExprKind::Range(r) => {
                self.expr(&r.start);
                self.expr(&r.end);
            }
            ExprKind::In(i) => {
                self.expr(&i.left);
                self.expr(&i.right);
            }
            ExprKind::FString(f) => {
                for part in &f.parts {
                    if let FStringPart::Expr(x) = part {
                        self.expr(x);
                    }
                }
            }
            ExprKind::Poison(_) => {}
        }
    }

    /// Post-loop state: the union of the pre-body state (when the loop can run zero times),
    /// the body's own fallthrough (a pending write at body end can be read on the next pass),
    /// and every `break`/`continue` exit.
    fn end_loop(&mut self, pre: Option<Flow>, mut body_end: Flow) {
        let mut arms = self.breaks.pop().unwrap_or_default();
        for flow in arms.iter_mut().chain(std::iter::once(&mut body_end)) {
            self.absorb_back_edge(flow);
        }
        self.loop_reads.pop();
        if let Some(pre) = pre {
            arms.push(pre);
        }
        arms.push(body_end);
        self.flow = Flow::merge(arms);
    }

    /// The back-edge the single body walk can't see: a store pending (or killed) where an
    /// iteration ends can reach a read on the next pass, so a dec the loop's text read
    /// anywhere keeps those spans live. Runs on body-end and `break`/`continue` exit
    /// flows alike -- the two aren't told apart here; a `break` flow's spans merging into
    /// post-loop stay pending for post-loop reads regardless, so over-marking only ever
    /// misses the rare write-then-break of a loop-read dec.
    fn absorb_back_edge(&mut self, flow: &mut Flow) {
        let Some(reads) = self.loop_reads.last() else {
            return;
        };
        for dec in reads {
            let pending = flow.pending.remove(dec).unwrap_or_default();
            let killed = flow.killed.remove(dec).unwrap_or_default();
            if !pending.is_empty() || !killed.is_empty() {
                flow.read_spans
                    .entry(*dec)
                    .or_default()
                    .extend(pending.into_iter().chain(killed));
            }
        }
    }

    fn literal(&mut self, l: &'a Literal) {
        match l {
            Literal::Array(items) => {
                for e in items {
                    self.expr(e);
                }
            }
            Literal::Record(fields) => {
                for (_, e) in fields {
                    self.expr(e);
                }
            }
            Literal::Struct(s) => {
                self.expr(&s.name);
                for (_, e) in &s.fields {
                    self.expr(e);
                }
                if let Some(base) = &s.base {
                    self.expr(base);
                }
            }
            Literal::Tuple(items) => {
                for e in items {
                    self.expr(e);
                }
            }
            _ => {}
        }
    }
}
