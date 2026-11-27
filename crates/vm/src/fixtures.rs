//! Fixtures are our take on the per-interpreter "singleton" storage in [Catherine West's
//! fabricator](https://github.com/kyren/fabricator) (`Registry::singleton` in its vm crate) --
//! which is itself the old type-keyed-storage idiom you may know as `anymap`, `http::Extensions`,
//! or Bevy's `Resources`. We use our own name partly because we've reshaped it (departures at the
//! bottom), and partly because "singleton" oversells it: there's one per *type per Vm*, not one
//! per program.
//!
//! Similar to the story in freeze.rs, Fixtures are inspired by a series of other libraries:
//! Catherine West's `Singleton` in [fabricator](https://github.com/kyren/fabricator), Bevy's
//! [Resource](https://docs.rs/bevy_ecs/latest/bevy_ecs/system/trait.Resource.html), `http::Extensions`, and the broader type-keyed storage idiom of `anymap`.
//!
//! [freeze](crate::freeze) explained itself with a coat check at a party, so let's stay in the
//! venue. We made a big assumption back there: "how do we know coat check exists?"
//!
//! Oh yeah, if you thought the last metaphor fell apart, buckle up for this one.
//!
//! A native fn runs deep inside the party with nothing in its pockets but a [Ctx](crate::Ctx).
//! The host, meanwhile, is standing outside the party entirely. If the two want to share anything
//! that isn't a gc value -- a [FreezeCell](crate::freeze::FreezeCell), an RNG, the script's args
//! -- it has to live somewhere *both* can find. That somewhere is the venue's lobby, and the
//! things in it are its fixtures. The one we care about? Why, the coat check of course!
//!
//! But just like with Freeze, there's some promises and assumptions we have to make for this to
//! work.
//!
//! 1. There is exactly _one_ coat check. Imagine if there was more than one, it'd be chaos! If we
//!    can agree there's only one we can all know what we mean when we say "the coat check".
//! 2. Fixtures assemble themselves the first time anyone asks. The coat check attendants, as hard
//!    working as they are, do not live in the coat check; they show up and open it when guests
//!    actually arrive to hand over coats. That's the `Default` bound plus `entry().or_insert_with`
//!    in [Fixtures::get]: there's no install step to forget and no ordering bug where a native asks
//!    before the attendants are ready. Either side can be first.
//! 3. Fixtures are bolted to the building. Once one exists it never moves and is never torn down
//!    until the whole venue is -- entries are never removed, and each one is `Box`ed so it keeps
//!    its own patch of floor even when the lobby's registry (the HashMap) reshuffles itself. That's
//!    why it's safe!
//!
//! There, isn't that better? That was definitely worth it, again.

use std::{
    any::{Any, TypeId},
    cell::{Cell, RefCell},
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use shared::{BodyId, FileId, IdVec, Location};
use web_time::Instant;

/// Marker for types that can live as a per-Vm fixture. Must be `Default`-constructible and
/// `'static` -- see the module docs for why both bounds are load-bearing.
pub trait Fixture: Default + Any + 'static {}
impl<T: Default + Any + 'static> Fixture for T {}

/// Debug info for the loaded program: per-chunk loc tables (indexed by `BodyId`) and each
/// file's source text. `Vm::load_program`/`Vm::set_sources` refresh it so `std::dbg` natives
/// can map a live frame's `(chunk, ip)` back to a source line -- natives see only a `Ctx`,
/// never the `Vm`, so this is where the mapping has to live.
///
/// `stack` mirrors the live `(chunk, ip)` of every frame. The dispatch loop holds `thread`
/// mutably borrowed for the whole run, so a native can never read the real stack; the
/// `CallNative` arm re-syncs this copy instead (parents' ips are their frozen return-site
/// saves, the top entry gets the ip just past the call op).
#[derive(Default)]
pub struct DebugInfo {
    pub locs: RefCell<IdVec<BodyId, Vec<(u32, Location)>>>,
    /// Each chunk's byte offset into the shared program bytes -- `Frame::ip` is absolute, the
    /// loc tables are chunk-relative, so callers subtract this before probing them.
    pub offsets: RefCell<IdVec<BodyId, u32>>,
    pub sources: RefCell<HashMap<FileId, Arc<str>>>,
    pub stack: RefCell<Vec<(BodyId, u32)>>,
}

impl DebugInfo {
    /// The innermost frame's source location. Inside a native that's the
    /// call site — the stack mirror's top ip sits just *past* the call
    /// op, so step back one byte to land inside the call op's own span
    /// (otherwise a statement whose trailing op carries the next
    /// statement's loc attributes one call site late).
    pub fn top_loc(&self) -> Option<Location> {
        let stack = self.stack.borrow();
        let &(body, ip) = stack.last()?;
        let locs = self.locs.borrow();
        let offsets = self.offsets.borrow();
        let rel = ip.checked_sub(*offsets.get(body)?)?.saturating_sub(1);
        let entries = locs.get(body)?;
        let i = entries.partition_point(|(off, _)| *off <= rel);
        (i > 0).then(|| entries[i - 1].1)
    }
}

/// Execution trace counters — the sonification/profiling substrate.
/// `run_dispatch` bumps `hits[ip]` once per dispatched op (`ip` is a byte
/// offset into the program's shared code `bytes` — operand bytes simply
/// never rise) and `calls` on every `Flow::Call` edge the driver takes.
/// Hosts drain per frame/step via `Vm::trace_take`; counts are deltas
/// since the last drain, not cumulative, so each read is "what ran in
/// this slice of time".
///
/// Always on: one indexed increment per op is lost in dispatch noise, and
/// keeping it unconditional means `take` works on any Vm — no recompile
/// to start listening.
#[derive(Default)]
pub struct Trace {
    /// `hits[i]` = ops dispatched at byte offset `i` since the last
    /// `take_hits`. Sized to the program's code length on `load_program`.
    pub hits: RefCell<Vec<u32>>,
    /// `(caller_body << 32) | callee_body` → count since the last
    /// `take_calls`. Only the `Flow::Call` driver arm records — calls
    /// inlined inside bcgen bodies bypass it, the same observability
    /// trade the debugger makes.
    pub calls: RefCell<HashMap<u64, u32>>,
    /// Ops dispatched so far: the monotone clock `edge_stack` start
    /// values are taken from. `ops_left` resets at every entry, so it
    /// cannot measure how long a frame lived; this only ever grows.
    pub ops_total: Cell<u64>,
    /// The live frame stack as `(caller_body, callee_body, ops_total at
    /// push)`, kept in step with `ThreadState::frames` by the
    /// dispatch's push/pop hooks. A frame pushed by a path the hooks
    /// do not see (bcgen bodies' in-body calls) never lands here, and
    /// `pop_frame` notices the chunk mismatch and leaves the stack
    /// alone — an unrecorded call folds into its parent's ops instead
    /// of desyncing the stack.
    pub edge_stack: RefCell<Vec<(u32, u32, u64)>>,
    /// `(caller_body << 32) | callee_body` → ops that ran while the
    /// callee frame was on the stack under that caller, cumulative
    /// over the program. `u32::MAX` is the caller of the entry frame.
    /// Read by `Vm::op_tree`; reset with the program.
    pub edge_ops: RefCell<HashMap<u64, u64>>,
    /// The ordered path recorder is running. Off by default: unlike
    /// `hits` it grows with the run, so only a host that drains it
    /// (`Vm::trace_path_take`) turns it on.
    pub path_on: Cell<bool>,
    /// The ordered path since the last take. See [`Path`].
    pub path: RefCell<Path>,
    /// The frame-span recorder is running: one chrome-trace `X` event
    /// per popped call edge, wall-timed. Off by default — a clock read
    /// per call edge is real noise, so only a host that drains
    /// (`Vm::trace_spans_take`) turns it on.
    pub spans_on: Cell<bool>,
    /// The `Instant` span `t0`s are measured from, captured by
    /// [`spans_enable`](Self::spans_enable). On wasm `web_time::Instant`
    /// is `performance.now`, so `t0`s land on the same clock a web
    /// host's trace ring already uses.
    span_epoch: RefCell<Option<Instant>>,
    /// Open spans `(callee, t0_us)`, mirroring `edge_stack` while
    /// `spans_on` — the same match-or-fold discipline applies.
    span_stack: RefCell<Vec<(u32, f64)>>,
    /// Finished spans `(callee, t0_us, dur_us)` since the last take.
    /// Past [`SPANS_LIMIT`](Self::SPANS_LIMIT) new spans only count
    /// into `spans_dropped`.
    pub spans: RefCell<Vec<(u32, f64, f64)>>,
    /// Spans dropped at the cap since the last take.
    pub spans_dropped: Cell<u32>,
}

/// The run as an ordered path, stored as taken jumps only. Execution is
/// sequential until an op transfers control, so the list of `(from, to)`
/// discontinuities rebuilds every op in order: start at `start`, follow
/// `next` until `from`, continue at `to`, stop at `end`. Calls, returns,
/// branches and loop back-edges are all the same record.
///
/// `out` is a flat stream of two-u32 records:
/// - `from, to`: one taken jump.
/// - `MARK + p, count`: the `p` records before this one ran `count`
///   more times. Those records may hold markers of their own, so a
///   loop nest folds level by level and a loop is one marker however
///   long it runs.
/// - `MARK, dropped`: the stream reached [`Path::LIMIT`] and `dropped`
///   jumps are missing. Never silent.
#[derive(Default)]
pub struct Path {
    /// `next[ip]` = byte offset of the op after the op at `ip`; what a
    /// tick compares against to see a jump. Built on `load_program`.
    pub next: Vec<u32>,
    prev: u32,
    start: u32,
    out: Vec<[u32; 2]>,
    dropped: u32,
}

impl Path {
    const NONE: u32 = u32::MAX;
    /// Longest repeating pattern folded, in records.
    pub const MAX_PERIOD: usize = 32;
    /// `from` values at or above this are markers; no ip gets this high.
    pub const MARK: u32 = u32::MAX - 255;
    /// Records `out` may hold (128 MB) before jumps count as dropped.
    const LIMIT: usize = 1 << 24;

    pub fn reset(&mut self, next: Vec<u32>) {
        *self = Path { next, prev: Self::NONE, ..Path::default() };
    }

    #[inline]
    fn step(&mut self, ip: u32) {
        if self.prev == Self::NONE {
            self.start = ip;
        } else if self.next.get(self.prev as usize) != Some(&ip) {
            self.jump(self.prev, ip);
        }
        self.prev = ip;
    }

    /// A pattern may repeat only if its markers point inside it.
    fn closed(pat: &[[u32; 2]]) -> bool {
        pat.iter()
            .enumerate()
            .all(|(j, r)| r[0] < Self::MARK || (r[0] - Self::MARK) as usize <= j)
    }

    fn jump(&mut self, from: u32, to: u32) {
        if self.out.len() >= Self::LIMIT {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        self.out.push([from, to]);
        // Fold the tail until nothing folds: one fold can complete the
        // pattern of the loop around it.
        'fold: loop {
            let len = self.out.len();
            for p in 1..=Self::MAX_PERIOD {
                // `pattern, marker(p), pattern` → count + 1
                if len >= 2 * p + 1
                    && self.out[len - p - 1][0] == Self::MARK + p as u32
                    && self.out[len - p - 1][1] < u32::MAX
                    && self.out[len - p..] == self.out[len - 2 * p - 1..len - p - 1]
                {
                    self.out.truncate(len - p);
                    self.out[len - p - 1][1] += 1;
                    continue 'fold;
                }
                // `pattern, pattern` → `pattern, marker(p) × 1`
                if len >= 2 * p
                    && self.out[len - p..] == self.out[len - 2 * p..len - p]
                    && Self::closed(&self.out[len - p..])
                {
                    self.out.truncate(len - p);
                    self.out.push([Self::MARK + p as u32, 1]);
                    continue 'fold;
                }
            }
            return;
        }
    }

    /// `[start, end, records…]` since the last take, empty when nothing
    /// ran. The next tick starts a new path.
    pub fn take(&mut self) -> Vec<u32> {
        if self.prev == Self::NONE {
            return Vec::new();
        }
        let mut v = vec![self.start, self.prev];
        v.extend(self.out.drain(..).flatten());
        if self.dropped > 0 {
            v.extend([Self::MARK, self.dropped]);
        }
        self.prev = Self::NONE;
        self.dropped = 0;
        v
    }

    /// A taken path back to every op's ip in run order: the inverse of
    /// the recording, for tests and offline tools. `f` gets each ip.
    pub fn expand(&self, path: &[u32], mut f: impl FnMut(u32)) {
        let [start, end, records @ ..] = path else { return };
        // markers unrolled to plain jumps first; `at[i]` is where
        // record `i`'s jumps begin
        let mut jumps: Vec<[u32; 2]> = Vec::new();
        let mut at: Vec<usize> = Vec::new();
        for r in records.chunks_exact(2) {
            let here = jumps.len();
            if r[0] > Self::MARK {
                let p = (r[0] - Self::MARK) as usize;
                let from = at[at.len() - p];
                for _ in 0..r[1] {
                    jumps.extend_from_within(from..here);
                }
            } else if r[0] < Self::MARK {
                jumps.push([r[0], r[1]]);
            }
            at.push(here);
        }
        let mut ip = *start;
        for [from, to] in jumps {
            while ip != from {
                f(ip);
                ip = self.next[ip as usize];
            }
            f(ip);
            ip = to;
        }
        while ip != *end {
            f(ip);
            ip = self.next[ip as usize];
        }
        f(ip);
    }
}

impl Trace {
    /// One dispatched op at byte offset `ip`. The borrow is held only for
    /// the increment — a native that re-enters `run_dispatch` (injected
    /// calls) must never hit a live `RefMut` and panic.
    #[inline]
    pub fn tick(&self, ip: usize) {
        self.ops_total.set(self.ops_total.get() + 1);
        if let Some(h) = self.hits.borrow_mut().get_mut(ip) {
            *h = h.wrapping_add(1);
        }
        if self.path_on.get() {
            self.path.borrow_mut().step(ip as u32);
        }
    }

    /// A bulk op debit — natives charge `ops_debt` for work that never
    /// passes `tick`, so the clock takes it in one step.
    pub fn charge(&self, ops: u64) {
        self.ops_total.set(self.ops_total.get() + ops);
    }

    /// A frame `callee` pushed under `caller` (`u32::MAX` for the entry
    /// frame): remember when it started so `pop_frame` can charge the
    /// edge.
    pub fn push_frame(&self, caller: u32, callee: u32) {
        self.edge_stack
            .borrow_mut()
            .push((caller, callee, self.ops_total.get()));
        if self.spans_on.get() && self.span_epoch.borrow().is_some() {
            self.span_stack
                .borrow_mut()
                .push((callee, self.span_now()));
        }
    }

    /// A frame `callee` popped: charge `ops_total - start` to its edge.
    /// If the top entry names another chunk, this frame was pushed by
    /// a path the hooks miss — leave the stack aligned and fold its
    /// ops into the parent's edge.
    pub fn pop_frame(&self, callee: u32) {
        let mut stack = self.edge_stack.borrow_mut();
        if let Some(&(caller, top, start)) = stack.last() {
            if top == callee {
                stack.pop();
                let spent = self.ops_total.get().saturating_sub(start);
                *self
                    .edge_ops
                    .borrow_mut()
                    .entry(((caller as u64) << 32) | callee as u64)
                    .or_default() += spent;
                if self.spans_on.get() && self.span_epoch.borrow().is_some() {
                    drop(stack);
                    self.close_span(callee);
                }
            }
        }
    }

    /// `n` frames dropped at once (an unwind truncate or a thread's
    /// clear): each pops in order.
    pub fn pop_frames(&self, callee: &[u32]) {
        for &c in callee.iter().rev() {
            self.pop_frame(c);
        }
    }

    /// Program load or reset: the clock and the frame edges start
    /// fresh; `edge_ops` drains like `take_calls` so hosts can read it.
    pub fn reset_frames(&self) {
        self.edge_stack.borrow_mut().clear();
        self.span_stack.borrow_mut().clear();
        self.ops_total.set(0);
    }

    /// Finished spans `spans` may hold before new completions count
    /// into `spans_dropped` instead.
    pub const SPANS_LIMIT: usize = 1 << 16;

    /// Start or stop the frame-span recorder. Enabling sets the wall
    /// epoch; disabling clears the open and finished buffers.
    // r[impl app.perf.trace.eval]
    pub fn spans_enable(&self, on: bool) {
        self.spans_on.set(on);
        *self.span_epoch.borrow_mut() = on.then(Instant::now);
        self.span_stack.borrow_mut().clear();
        if !on {
            self.spans.borrow_mut().clear();
            self.spans_dropped.set(0);
        }
    }

    /// Drain finished spans: `[dropped, callee, t0_us, dur_us, …]`
    /// flat so one wire type carries it — body promoted to f64.
    pub fn take_spans(&self) -> Vec<f64> {
        let mut spans = self.spans.borrow_mut();
        let mut out = Vec::with_capacity(spans.len() * 3 + 1);
        out.push(self.spans_dropped.replace(0) as f64);
        for (body, t0, dur) in spans.drain(..) {
            out.extend([body as f64, t0, dur]);
        }
        out
    }

    /// µs since `span_epoch`; the span clock.
    fn span_now(&self) -> f64 {
        Instant::now()
            .saturating_duration_since(self.span_epoch.borrow().unwrap())
            .as_micros() as f64
    }

    /// The matched edge popped while `spans_on`: record the callee's
    /// wall span. `span_stack` mirrors `edge_stack`, so a `pop_frame`
    /// that matched leaves exactly its entry here — a caller that
    /// enabled mid-run may hold more open spans than edges.
    fn close_span(&self, callee: u32) {
        let Some((_, t0)) = self.span_stack.borrow_mut().pop() else {
            return;
        };
        let dur = self.span_now() - t0;
        let mut spans = self.spans.borrow_mut();
        if spans.len() < Self::SPANS_LIMIT {
            spans.push((callee, t0, dur));
        } else {
            self.spans_dropped.set(self.spans_dropped.get() + 1);
        }
    }

    /// One call edge taken: `caller` (the frame on top at dispatch) →
    /// `callee`. Packed into a u64 so the map key is a plain int.
    pub fn call(&self, caller: u32, callee: u32) {
        *self
            .calls
            .borrow_mut()
            .entry(((caller as u64) << 32) | callee as u64)
            .or_default() += 1;
    }

    /// Sparse `(ip, count)` pairs since the last take, zeroing as it
    /// reads. O(code size), called at frame cadence at most.
    pub fn take_hits(&self) -> Vec<(u32, u32)> {
        let mut hits = self.hits.borrow_mut();
        let mut out = Vec::new();
        for (i, h) in hits.iter_mut().enumerate() {
            if *h != 0 {
                out.push((i as u32, *h));
                *h = 0;
            }
        }
        out
    }

    /// The call-edge deltas since the last take: `(packed_edge, count)`.
    pub fn take_calls(&self) -> Vec<(u64, u32)> {
        std::mem::take(&mut *self.calls.borrow_mut())
            .into_iter()
            .collect()
    }
}

pub type OutSink = Box<dyn FnMut(&str)>;

/// Where `print`/`dbg` and friends send their lines. The default sink is
/// the process's stdout; an embedder (a repl, the wasm eval worker) swaps
/// in a capture with [`Out::set`] and drains it after the run. Each call
/// gets one line's text — the trailing newline is the sink's business.
pub struct Out(pub RefCell<OutSink>);

impl Default for Out {
    fn default() -> Self {
        Self(RefCell::new(Box::new(|line| println!("{line}"))))
    }
}

impl Out {
    /// Emit one line to the current sink.
    pub fn write(&self, line: &str) {
        (self.0.borrow_mut())(line);
    }

    /// Swap the sink. The old one is dropped — this is a one-way trip per
    /// Vm, which is all capture-style embedders need (a Vm is one session).
    pub fn set(&self, sink: impl FnMut(&str) + 'static) {
        *self.0.borrow_mut() = Box::new(sink);
    }
}

/// What a line recorded at a call site is: `Print` is program output
/// (`print`/`dbg`/`warn`); `Warn` is a non-fatal diagnostic — something was
/// skipped or defaulted, the run kept going, and the host gets a span to
/// point at. A warning also reaches `Out`; a print never does the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintKind {
    Print,
    Warn,
}

/// Every `print`/`dbg`/`warn` line, keyed by call site — a rich host (the lit
/// page) hovers a `print(x)` and shows what it printed, or underlines a call
/// that warned; outside a host nothing reads this and `Out` is all a print is.
/// Capped so a print in a per-frame game loop can't grow without bound —
/// overflow lands in `dropped` (and still reaches `Out`). Warns are also
/// deduped per `(site, text)`: the same problem reported every frame is one
/// warning, not sixty.
#[derive(Default)]
pub struct Prints {
    /// `(call site, kind, rendered line)` in emit order.
    pub lines: RefCell<Vec<(Location, PrintKind, String)>>,
    /// Warn sites already reported — survives `take`, so a warn inside a
    /// game loop reports once per session, not once per frame. `pub(crate)`
    /// for `snapshot`: the dedupe set rides a checkpoint restore so a warn
    /// that fired before the snapshot doesn't re-fire after it.
    pub(crate) warned: RefCell<std::collections::HashSet<(Location, String)>>,
    /// Lines past [`Prints::CAP`] — counted so a host can say "…N more".
    pub dropped: Cell<usize>,
}

impl Prints {
    pub const CAP: usize = 256;

    pub fn push(&self, loc: Location, text: String) {
        let mut lines = self.lines.borrow_mut();
        if lines.len() < Self::CAP {
            lines.push((loc, PrintKind::Print, text));
        } else {
            self.dropped.set(self.dropped.get() + 1);
        }
    }

    /// Record a non-fatal diagnostic at `loc`. Repeats of the same
    /// `(site, text)` are suppressed — the host still holds the first.
    /// Returns whether the warning was new (the caller may still want to
    /// echo it to `Out` only then).
    pub fn push_warn(&self, loc: Location, text: String) -> bool {
        if !self.warned.borrow_mut().insert((loc, text.clone())) {
            return false;
        }
        let mut lines = self.lines.borrow_mut();
        if lines.len() < Self::CAP {
            lines.push((loc, PrintKind::Warn, text));
        } else {
            self.dropped.set(self.dropped.get() + 1);
        }
        true
    }

    /// Drain the recorded lines (leaves `dropped` and the warn dedup —
    /// they're per-Vm history).
    pub fn take(&self) -> Vec<(Location, PrintKind, String)> {
        std::mem::take(&mut *self.lines.borrow_mut())
    }
}

/// `print`/`dbg`/`warn` call site → the dimension the dims pass computed for the
/// reported value — `print(25kW)` writes `25000 W`, not a bare `25000`. The
/// table ships on the `Program` and `load_program` refreshes it, so a Vm that
/// swaps programs never attributes a unit to a call site that didn't earn one.
/// Keyed by the same call `Location` `DebugInfo::top_loc` reports, which is
/// also what `Prints` records.
#[derive(Default)]
pub struct PrintDims {
    pub dims: RefCell<HashMap<Location, shared::units::Dim>>,
    /// Top-level echo site → the unit the echo shows its quantity in.
    pub shown: RefCell<HashMap<Location, shared::units::Shown>>,
}

impl PrintDims {
    /// The unit a `print`/`dbg`/`warn` call at `loc` should append (`m/px`, `W`),
    /// `None` for a plain number or a site the checker couldn't follow.
    pub fn at(&self, loc: Location) -> Option<shared::units::Dim> {
        self.dims.borrow().get(&loc).copied()
    }

    /// How the echo at `loc` shows its quantity, when the dims pass knew it.
    pub fn shown(&self, loc: Location) -> Option<shared::units::Shown> {
        self.shown.borrow().get(&loc).cloned()
    }
}

/// Per-Vm random stream behind `float::random`/`int::random`/`bool::random`/
/// `arr.shuffle`/`arr.choose` (and anything else that draws): the one place
/// nondeterminism enters a program, so a host that seeds it gets a replayable
/// run. `None` (the default) means each draw takes fresh entropy, same as
/// `rand::rng()` did before; `seed` installs a xorshift64* stream — the same
/// algorithm the headless `mrt` runtime uses, so an interpreter seeded `s`
/// and a transpiled run seeded `s` produce identical sequences.
///
/// `script` is the other half of the story — the hookable read. A sim
/// harness queues exact unit values for upcoming draws ("the third
/// `float::random` returns exactly 0.25"), taking manual control of a
/// specific nondet point instead of the whole stream. Scripted draws
/// *replace* the stream value at that position — the stream still
/// advances — so unscripted draws replay at identical indices either
/// way and neighbors are untouched.
#[derive(Default)]
pub struct Rng {
    /// `Some` = seeded xorshift64* state; `None` = entropy per draw.
    state: Cell<Option<u64>>,
    /// Exact `[0,1)` values queued for the next draws.
    script: RefCell<VecDeque<f64>>,
    /// `with rng` scopes, LIFO: each entry saved the state+script in force
    /// before its push installed new ones, keyed by the frame depth it was
    /// pushed at (`DebugInfo::stack`'s len — the pushing frame included). A
    /// scope's `pop` runs at the same depth its `push` did; an entry that
    /// outlives the live stack was orphaned by an unwind — a hard fault
    /// skipping the desugar's pop — and [`Rng::reconcile`] restores it.
    with: RefCell<Vec<WithScope>>,
}

/// One installed `with rng` scope (see [`Rng::with`]).
struct WithScope {
    depth: usize,
    state: Option<u64>,
    script: VecDeque<f64>,
}

impl Rng {
    /// Install the deterministic stream — a sim harness calls this before
    /// the program's first random op. `0` is legal but degenerate
    /// (xorshift64* pins at 0 → every draw 0.0); callers wanting
    /// "0 means unseeded" semantics guard before calling.
    pub fn seed(&self, seed: u64) {
        self.state.set(Some(seed));
    }

    /// The live stream state — `None` while unseeded. A host snapshots
    /// this into checkpoints the way `mrt::random_seed` does, so a
    /// resumed run continues the same sequence rather than restarting.
    pub fn state(&self) -> Option<u64> {
        self.state.get()
    }

    /// Queue exact unit values for the next `values.len()` draws — the
    /// specific-read hook: override *these* draws, leave the rest of the
    /// stream untouched. Out-of-`[0,1)` values are clamped into it.
    pub fn script(&self, values: impl IntoIterator<Item = f64>) {
        self.script.borrow_mut().extend(values);
    }

    /// How many scripted draws are left — a test asserts the program
    /// consumed exactly what was queued.
    pub fn scripted(&self) -> usize {
        self.script.borrow().len()
    }

    /// One `[0, 1)` draw — the xorshift64* step (bit-identical to
    /// `mrt::random`'s stream), or `None` when unseeded and the caller
    /// falls back to its entropy source; a queued script value replaces
    /// whatever the stream produced at this position. Every random op
    /// draws through here, so `shuffle`, `choose`, and the `random`s
    /// share one composable stream.
    pub fn next_unit(&self) -> Option<f64> {
        let stream = self.state.get().map(|mut x| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state.set(Some(x));
            (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
        });
        if let Some(u) = self.script.borrow_mut().pop_front() {
            return Some(u.clamp(0.0, 1.0 - f64::EPSILON));
        }
        stream
    }

    /// `with rng`'s install half (`__fx_rng_push`): save the current
    /// state+script on the scope stack, keyed by `depth`, and install the
    /// scope's own. `depth` is the live frame count the pushing native saw.
    // r[impl mimas.fx.with]
    pub fn push_with(&self, depth: usize, state: Option<u64>, script: VecDeque<f64>) {
        let saved = WithScope {
            depth,
            state: self.state.replace(state),
            script: std::mem::replace(&mut *self.script.borrow_mut(), script),
        };
        self.with.borrow_mut().push(saved);
    }

    /// `with rng`'s restore half (`__fx_rng_pop`): pop the top scope and put
    /// its saved state+script back. An empty stack can't come from the
    /// desugar (push and pop pair lexically), so this is a no-op then.
    pub fn pop_with(&self) {
        if let Some(scope) = self.with.borrow_mut().pop() {
            self.state.set(scope.state);
            *self.script.borrow_mut() = scope.script;
        }
    }

    /// Drop every scope orphaned by an unwind. `depth` is the live frame
    /// count *now*; an entry recorded deeper was pushed by a frame that's
    /// gone (a fault truncated it — the structured exits `return`/`raise`/
    /// `break`/`continue`/`?` can't skip a `with` because the parser refuses
    /// them inside one). Popping deepest-first leaves the *shallowest*
    /// orphan's saved state in `saved` — the state the oldest dead scope was
    /// keeping for its outside, which is what the survivors should see.
    // r[impl mimas.fx.with.restore]
    pub fn reconcile(&self, depth: usize) {
        let saved = {
            let mut with = self.with.borrow_mut();
            let mut saved = None;
            while with.last().is_some_and(|s| s.depth > depth) {
                saved = with.pop();
            }
            saved
        };
        if let Some(scope) = saved {
            self.state.set(scope.state);
            *self.script.borrow_mut() = scope.script;
        }
    }

    /// Live `with` scopes — a test asserts a program left none behind.
    pub fn with_scopes(&self) -> usize {
        self.with.borrow().len()
    }
}

/// Per-Vm type-keyed storage for host-shared state. Lazily creates entries on first access via
/// `T::default()`. Designed to hold things like [FreezeCell](crate::freeze::FreezeCell)s that
/// host code installs `&mut T` borrows into during mimas execution.
#[derive(Default)]
pub struct Fixtures {
    cells: RefCell<HashMap<TypeId, Box<dyn Any>>>,
}

// SAFETY: Fixtures holds only `'static` heap-boxed values, none containing Gc handles.
unsafe impl<'gc> gc_arena::Collect<'gc> for Fixtures {
    const NEEDS_TRACE: bool = false;
}

impl Fixtures {
    pub fn get<T: Fixture>(&self) -> &T {
        let tid = TypeId::of::<T>();
        let ptr: *const T = {
            let mut cells = self.cells.borrow_mut();
            let entry = cells.entry(tid).or_insert_with(|| Box::new(T::default()));
            entry
                .downcast_ref::<T>()
                .expect("fixture type mismatch (TypeId collision?)") as *const T
        };
        // SAFETY: entries are never removed; Box's heap address is stable for the
        // lifetime of self. The returned reference is tied to &self via inference.
        unsafe { &*ptr }
    }
}

/// The values of bare top-level expressions, by their line: what a
/// notebook cell shows under itself. `echo` (the stdlib native every
/// top-level expression statement is compiled to call) records here only
/// while `on` — a host sets it for the pass it reads, so a plain run
/// formats nothing. Test-body statements call `echo` the same way at any
/// depth, which is how a failed check's rerun tapes itself (check.tape).
/// Each entry is the call site, the value's display text (quoted strings,
/// `Name { .. }`), and its [`Inspect`] tree for a richer view. Capped like
/// [`Prints`].
#[derive(Default)]
pub struct Echoes {
    pub on: Cell<bool>,
    pub lines: RefCell<Vec<(Location, String, crate::Inspect)>>,
    pub dropped: Cell<usize>,
}

impl Echoes {
    pub const CAP: usize = 4096;

    pub fn push(&self, loc: Location, text: String, value: crate::Inspect) {
        let mut lines = self.lines.borrow_mut();
        if lines.len() < Self::CAP {
            lines.push((loc, text, value));
        } else {
            self.dropped.set(self.dropped.get() + 1);
        }
    }

    pub fn take(&self) -> Vec<(Location, String, crate::Inspect)> {
        std::mem::take(&mut *self.lines.borrow_mut())
    }
}

/// What one failing `check`/`assert!` compared: the operand values its
/// failure message names, kept so a host shows the example, not just its
/// text. The `__check_operand` stdlib native (which the assert desugar
/// emits inside the failure branch, one call per operand, just before
/// `panic` unwinds) builds one of these; a check runner drains it after
/// each case. A pass never reaches the calls, so a record only ever
/// exists on the failing path.
pub struct CheckWitness {
    /// The first `__check_operand` call's source site, when debug info
    /// had one — all of them share the check's `start`.
    pub loc: Option<Location>,
    /// Each operand's [`Inspect`](crate::Inspect) tree, in the desugar's
    /// order: `left, right` for `==`/`!=`/`<`/`in`/`≈`, the call's
    /// arguments for a predicate check, the checked value alone for a
    /// bare `check x`.
    pub operands: Vec<crate::Inspect>,
    /// The same operands as live handles, for a host that wants more than
    /// the trees — drawing one through its own `draw` method
    /// ([`Vm::call_method_on_stashed`](crate::Vm::call_method_on_stashed))
    /// needs the real `Val`, not the snapshot. They root their values
    /// until this record drops.
    pub vals: Vec<crate::Stashed>,
}

/// The [`CheckWitness`] being recorded. One slot, not a history: a
/// recorded failure unwinds its check at once, and the runner takes the
/// record after each case, so nothing accrues and a stale one can't ride
/// onto a later check's report.
#[derive(Default)]
pub struct Witnesses {
    last: RefCell<Option<CheckWitness>>,
}

impl Witnesses {
    /// The next operand of the failure being recorded. The desugar emits
    /// one `__check_operand` call per operand — separate calls keep each
    /// operand's own static type; a single `[…]` argument list would
    /// force heterogeneous operands (a predicate's mixed arguments) to
    /// share one element type and not compile.
    pub fn push(
        &self,
        loc: Option<Location>,
        operand: crate::Inspect,
        val: crate::Stashed,
    ) {
        let mut last = self.last.borrow_mut();
        let w = last.get_or_insert_with(|| CheckWitness {
            loc,
            operands: Vec::new(),
            vals: Vec::new(),
        });
        w.operands.push(operand);
        w.vals.push(val);
    }

    /// The recorded failure's operands, taken — callers drain after each
    /// check case so attribution stays exact.
    pub fn take(&self) -> Option<CheckWitness> {
        self.last.borrow_mut().take()
    }
}

/// What the `check`s inside blocks saw (r[impl check.block]). A `check`
/// in a `fn` body, a loop or an `if` runs where it stands, with that
/// block's locals in scope, and a failure does not stop the run: the
/// desugar calls `__check_miss` instead of `panic`. The check's own entry
/// in the run's check list (`__check_verdict`, registered at the top
/// level under the check's source text) reads the first failure back.
/// `fresh` holds the first failure since the last `take_fresh`, so a check
/// runner fails the `#[test]` fn or top-level check whose call made it.
#[derive(Default)]
pub struct BlockChecks {
    failed: RefCell<HashMap<String, String>>,
    fresh: RefCell<Option<String>>,
}

impl BlockChecks {
    /// A failure of the block check `name`. The first message for a name
    /// stays; a loop that fails ten times reports its first.
    pub fn miss(&self, name: &str, message: String) {
        self.fresh.borrow_mut().get_or_insert_with(|| message.clone());
        self.failed
            .borrow_mut()
            .entry(name.to_string())
            .or_insert(message);
    }

    /// The first failure recorded for `name`, or `None` when every run of
    /// it passed or it never ran.
    pub fn verdict(&self, name: &str) -> Option<String> {
        self.failed.borrow().get(name).cloned()
    }

    /// The first failure since the last call, taken.
    pub fn take_fresh(&self) -> Option<String> {
        self.fresh.borrow_mut().take()
    }

    /// Forget every record: a new run of the top level starts clean.
    pub fn clear(&self) {
        self.failed.borrow_mut().clear();
        self.fresh.borrow_mut().take();
    }
}

/// What the one renderer needs beyond the value (r[impl mimas.display.one]):
/// each display site's static shape (`print`, `dbg`, `warn`, a top-level
/// echo), the shapes f-string holes name by index (`__shown`), each struct's
/// field shapes (declared units), and the unit the page wrote for each
/// dimension. `load_program` refreshes it.
#[derive(Default)]
pub struct DisplayShapes {
    pub sites: RefCell<HashMap<Location, shared::units::Shape>>,
    pub holes: RefCell<Vec<shared::units::Shape>>,
    pub fields: RefCell<Vec<Vec<shared::units::Shape>>>,
    pub units: RefCell<HashMap<shared::units::Dim, shared::units::Shown>>,
}

impl DisplayShapes {
    /// The shape of the value shown at `loc`, `Plain` when nothing is known.
    pub fn at(&self, loc: Location) -> shared::units::Shape {
        self.sites.borrow().get(&loc).cloned().unwrap_or_default()
    }

    /// The `i`th f-string hole shape.
    pub fn hole(&self, i: usize) -> shared::units::Shape {
        self.holes.borrow().get(i).cloned().unwrap_or_default()
    }

    /// The field shapes of struct `sid`, in slot order.
    pub fn fields_of(&self, sid: usize) -> Vec<shared::units::Shape> {
        self.fields.borrow().get(sid).cloned().unwrap_or_default()
    }

    /// How a quantity of `dim` shows: the page's unit, else the coherent one.
    pub fn unit(&self, dim: shared::units::Dim) -> shared::units::Shown {
        self.units.borrow().get(&dim).cloned().unwrap_or_else(|| shared::units::Shown {
            name: dim.name(),
            scale: 1.0,
            list: false,
        })
    }
}
