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
//! The result -- `dec_uses: IndexMap<DecId, UseInfo>` -- is honest per-site counts
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

use std::collections::{HashMap, HashSet};

use miette::Report;
use parse::{
    Access, Ast, Expr, ExprKind, FStringPart, Function, Ident, Item, ItemKind, Literal, PactItem,
    Stmt, StmtKind,
    components::{Pat, PatKind},
    stmt::AssignmentOp,
};
use shared::{Fx, Located, Location};

use crate::{
    GuardSeg, Solver,
    components::{DecId, DecKind},
    errors::{
        ConsumeForbidden, DeadStore, MustUseResult, UnusedBinding, UnusedParam,
        UseAfterConsume,
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
    pending: HashMap<DecId, HashSet<Location>>,
    /// dec -> write sites this path overwrote but did not itself produce. Arbitrated at
    /// merges: a span in here died on this path, and only stays a candidate if it died on
    /// every sibling path too.
    killed: HashMap<DecId, HashSet<Location>>,
    /// dec -> write sites a read on this path consumed. Any read makes the store live.
    read_spans: HashMap<DecId, HashSet<Location>>,
    /// Write sites created inside this path segment. Killing one is unconditional -- it
    /// can't exist on a sibling path -- so it reports straight away.
    born: HashSet<Location>,
    /// dec -> the `(path, site, callee)` marks `#[consumes]` params took on this
    /// path. Unioned at merges: a region *maybe* consumed on any surviving path is
    /// gone from the join on -- matching Rust's move-out-of-one-branch behavior.
    consumed: HashMap<DecId, Vec<ConsumeMark>>,
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
        let mut pending: HashMap<DecId, HashSet<Location>> = HashMap::new();
        let mut read_spans: HashMap<DecId, HashSet<Location>> = HashMap::new();
        let mut killed: HashMap<DecId, HashSet<Location>> = HashMap::new();
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
        // post-merge the flow is again a single path, so every surviving pending span is
        // fair game for an eager dead verdict
        let born: HashSet<Location> = pending.values().flatten().copied().collect();
        let mut consumed: HashMap<DecId, Vec<ConsumeMark>> = HashMap::new();
        for f in &arms {
            for (dec, marks) in &f.consumed {
                consumed.entry(*dec).or_default().extend(marks.iter().cloned());
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
    uses: HashMap<DecId, UseInfo>,
    /// Bindings eligible for unused-warnings: (dec, where it came from).
    lint_targets: Vec<(DecId, Site)>,
    /// The scope each dec was bound in (scope_decs.len() at bind time), for the
    /// closure-isolation rule.
    dec_scope: HashMap<DecId, usize>,
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
    /// Decs whose use-after-consume already reported (one error per dead
    /// path, not per read of it).
    consume_reported: HashSet<(DecId, Vec<String>)>,
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
    own_fx: HashMap<Caller, Fx>,
    /// caller -> user-fn callee decs (the fixpoint's edges).
    edges: HashMap<Caller, HashSet<DecId>>,
    /// `fn` name site -> dec, so a `Function` node can name its caller.
    dec_by_site: HashMap<(usize, usize), DecId>,
    /// pact member decs -- a call resolved to one dispatches to an impl this file can't see,
    /// so it's `unknown`, not an edge.
    pact_members: HashSet<DecId>,
}

/// Fill `solver.dec_uses`, `solver.fn_effects`, `solver.script_effects`, `solver.warnings`.
/// Runs after `dims::check` at the tail of `solve_all`. Lints only warn, but a
/// `#[consumes]` violation is a real refusal: `Err` fails the solve like any other pass.
pub(crate) fn check(solver: &mut Solver, asts: &[&Ast]) -> crate::errors::Result<()> {
    let mut errors: Vec<Report> = Vec::new();
    // call edges and per-caller effects gather across all asts before the fixpoint -- a fn in
    // file A can call a fn in file B
    let mut edges: HashMap<DecId, HashSet<DecId>> = HashMap::new();
    let mut fn_fx: HashMap<DecId, Fx> = HashMap::new();
    // per file: (name, [(stmt site, own fx, callee edges)]) -- folded into
    // `script_effects`/`script_segments` after the fixpoint
    let mut script_fx: Vec<(String, Vec<(Location, Fx, HashSet<DecId>)>)> = Vec::new();

    for ast in asts {
        let mut dec_by_site = HashMap::new();
        for (id, dec) in solver.decs.iter() {
            if matches!(dec.kind, DecKind::Item { .. }) {
                dec_by_site
                    .entry((dec.location.file_id, dec.location.span.start))
                    .or_insert(id);
            }
        }
        let mut pass = Pass {
            solver,
            uses: HashMap::new(),
            lint_targets: Vec::new(),
            dec_scope: HashMap::new(),
            dead_candidates: Vec::new(),
            warnings: Vec::new(),
            scope_decs: Vec::new(),
            closure_floors: Vec::new(),
            loop_floors: Vec::new(),
            consume_reported: HashSet::new(),
            skip_root_check: false,
            in_lvalue: false,
            errors: Vec::new(),
            flow: Flow::default(),
            retired: Vec::new(),
            breaks: Vec::new(),
            callers: vec![Caller::Script(0)],
            script_sites: Vec::new(),
            own_fx: HashMap::new(),
            edges: HashMap::new(),
            dec_by_site,
            pact_members: solver.pact_members.values().copied().collect(),
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
        solver.warnings.extend(warnings);
    }

    // fixpoint: fn_effects[f] |= union fn_fx[callee] until stable. Each round only adds bits,
    // so this terminates in at most (edge count) rounds; recursive cycles settle on their own.
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
    for (caller, fx) in fn_fx {
        solver.fn_effects.insert(caller, fx);
    }
    for (name, segments) in script_fx {
        let mut file_fx = Fx::empty();
        for (site, own, calls) in segments {
            let fx = own
                | calls.iter().fold(Fx::empty(), |acc, d| {
                    acc | solver.fn_effects.get(d).copied().unwrap_or(Fx::empty())
                });
            file_fx |= fx;
            solver.script_segments.push((name.clone(), site, fx));
        }
        solver.script_effects.insert(name, file_fx);
    }
    // consume violations ride every other phase's convention: first error out
    if let Some(err) = errors.into_iter().next() {
        return Err(err);
    }
    Ok(())
}

impl<'a> Pass<'a> {
    // ---- bookkeeping

    fn dec_of(&self, ident: &Ident) -> Option<DecId> {
        self.solver.node_decs.get(&ident.id).copied()
    }

    fn cur(&self) -> Caller {
        *self.callers.last().unwrap_or(&Caller::Script(0))
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
        let Some((segs, consumed_at, by)) =
            marks.iter().find(|(s, ..)| paths_overlap(s, path))
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

    fn push_scope(&mut self) {
        self.scope_decs.push(Vec::new());
    }

    /// The scope's own decs die here. A write span is a dead store when no surviving
    /// path (this flow, plus any that retired early via `return`/`raise`/`let-else`)
    /// still carries it pending or read it -- writes only ever live on paths that
    /// stored them, so pending and killed spans both count as deaths.
    fn pop_scope(&mut self) {
        let Some(decs) = self.scope_decs.pop() else {
            return;
        };
        for dec in decs {
            let mut spans: HashSet<Location> = HashSet::new();
            let mut live: HashSet<Location> = HashSet::new();
            for flow in std::iter::once(&self.flow).chain(self.retired.iter()) {
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
            for flow in &mut self.retired {
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
        let mut seen_deads = HashSet::new();
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
                    Fx::unknown()
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
        match stmt.kind() {
            StmtKind::Let(l) => {
                self.expr(&l.right);
                if let Some(else_branch) = &l.else_branch {
                    // `else` always diverges, but its stores still die with this scope
                    let pre = self.flow.clone();
                    self.flow = pre.fork();
                    self.expr(else_branch);
                    let end = std::mem::replace(&mut self.flow, pre);
                    self.retired.push(end);
                }
                self.bind(&l.left, Site::Binding, true);
            }
            StmtKind::Assignment(a) => {
                match a.left.kind() {
                    ExprKind::Ident(ident) => {
                        // `x = v`: the ident is the target, not a read. `x += v` and friends
                        // read the old value first.
                        let dec = self.dec_of(ident);
                        if !matches!(a.op, AssignmentOp::Identity)
                            && let Some(dec) = dec
                        {
                            self.read(dec, ident.location);
                        }
                        self.expr(&a.right);
                        if let Some(dec) = dec {
                            self.write(dec, ident.location, true);
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
                            let pure_field =
                                fields.len() == path.len() && !fields.is_empty();
                            let reads_slot =
                                !matches!(a.op, AssignmentOp::Identity) || !pure_field;
                            if let Some((segs, consumed_at, by)) = marks.iter().find(
                                |(s, ..)| {
                                    paths_overlap(s, &fields)
                                        && (reads_slot || s.len() < fields.len())
                                },
                            ) && self.consume_reported.insert((dec, segs.clone()))
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
                            if pure_field
                                && let Some(ms) = self.flow.consumed.get_mut(&dec)
                            {
                                ms.retain(|(s, ..)| {
                                    s.is_empty() || !s.starts_with(fields.as_slice())
                                });
                            }
                        }
                        let lv = std::mem::replace(&mut self.in_lvalue, true);
                        self.expr(&a.left);
                        self.in_lvalue = lv;
                        self.expr(&a.right);
                        if let Some((root, _)) = crate::root_and_path(&a.left)
                            && let Some(dec) = self.dec_of(root)
                        {
                            self.write(dec, root.location, false);
                        }
                    }
                }
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
            StmtKind::Module(_) => {}
        }
    }

    fn item(&mut self, item: &'a Item) {
        match item.kind() {
            ItemKind::Function(f) => self.function(f),
            ItemKind::Const(c) => self.expr(&c.right),
            ItemKind::Impl(imp) => {
                for inner in &imp.items {
                    self.item(inner);
                }
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
                            for p in parameters {
                                s.bind(&p.left, Site::Param, false);
                            }
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
        ) = saved;
    }

    fn function(&mut self, f: &'a Function) {
        let dec = self
            .dec_by_site
            .get(&(f.name.location.file_id, f.name.location.span.start))
            .copied();
        let caller = dec.map_or_else(|| self.cur(), Caller::Fn);
        self.callers.push(caller);
        // an entry even when the body ends up pure -- `fn_effects` should answer for
        // every walked fn, not only effectful ones
        self.own_fx.entry(caller).or_insert(Fx::empty());
        self.in_fn(|s| {
            for p in &f.parameters {
                s.bind(&p.left, Site::Param, false);
            }
            s.expr(&f.body);
        });
        self.callers.pop();
    }

    fn expr(&mut self, e: &'a Expr) {
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
            ExprKind::Demote(d) => self.expr(&d.expr),
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
                    && let Some((root, _)) = crate::root_and_path(left)
                    && let Some(root_dec) = self.dec_of(root)
                {
                    self.write(root_dec, root.location, false);
                }
                self.call_effect(&c.left);
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
                if let Some(caps) = self.solver.closure_captures.get(&e.id()) {
                    for &dec in caps {
                        self.uses.entry(dec).or_default().reads = Use::Many;
                    }
                }
                let saved_retired = std::mem::take(&mut self.retired);
                let saved_breaks = std::mem::take(&mut self.breaks);
                let diverged = self.flow.diverged;
                self.closure_floors.push(self.scope_decs.len());
                self.push_scope();
                for p in &c.parameters {
                    self.bind(&p.left, Site::Param, false);
                }
                self.expr(&c.body);
                self.pop_scope();
                self.closure_floors.pop();
                self.flow.diverged = diverged;
                self.retired = saved_retired;
                self.breaks = saved_breaks;
            }
            ExprKind::For(f) => {
                self.expr(&f.iterator);
                let pre = self.flow.clone();
                // the body may run zero times: its kills need the merge's arbitration
                self.flow.born.clear();
                self.breaks.push(Vec::new());
                self.loop_floors.push(self.scope_decs.len());
                self.push_scope();
                self.bind(&f.binding, Site::Binding, false);
                self.expr(&f.body);
                self.pop_scope();
                self.loop_floors.pop();
                let body_end = std::mem::take(&mut self.flow);
                // a `for` may run zero times: the pre-loop state exits too
                self.end_loop(Some(pre), body_end);
            }
            ExprKind::While(w) => {
                self.expr(&w.header);
                let pre = self.flow.clone();
                self.flow.born.clear();
                self.breaks.push(Vec::new());
                self.loop_floors.push(self.scope_decs.len());
                self.push_scope();
                if let Some(b) = &w.binding {
                    self.bind(b, Site::Binding, false);
                }
                self.expr(&w.body);
                self.pop_scope();
                self.loop_floors.pop();
                let body_end = std::mem::take(&mut self.flow);
                self.end_loop(Some(pre), body_end);
            }
            ExprKind::Loop(l) => {
                self.breaks.push(Vec::new());
                self.loop_floors.push(self.scope_decs.len());
                self.expr(&l.body);
                self.loop_floors.pop();
                let body_end = self.flow.clone();
                let exits = self.breaks.pop().unwrap_or_default();
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
    fn end_loop(&mut self, pre: Option<Flow>, body_end: Flow) {
        let mut arms = self.breaks.pop().unwrap_or_default();
        if let Some(pre) = pre {
            arms.push(pre);
        }
        arms.push(body_end);
        self.flow = Flow::merge(arms);
    }

    fn literal(&mut self, l: &'a Literal) {
        match l {
            Literal::Array(items) => {
                for e in items {
                    self.expr(e);
                }
            }
            Literal::Dictionary(fields) => {
                for (_, e) in fields {
                    self.expr(e);
                }
            }
            Literal::Struct(s) => {
                self.expr(&s.name);
                for (_, e) in &s.fields {
                    self.expr(e);
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
