//! `JitSession` — the observe → specialize → reinstall tiering API.
//!
//! No VM surgery: [`JitSession::observe`] installs one profiling [`BodyFn`]
//! per body through the ordinary `Vm::install_bc` table. The shim runs the
//! interpreter's own [`step`] per dispatch (one op at a time — the driver
//! redispatches on `Flow::Next`), recording facts on *fresh frame entries*
//! only: a re-dispatched body mid-run has `code.ip != chunk.offset` and is
//! skipped, so the window contents it sees are genuinely the entry values.
//!
//! Gathered facts:
//! - `entry[r]` — the merged `Val` tag (+ shared GC payload) reg `r` held at
//!   every entry — drives forced scalar shadows and frozen lookups.
//! - `calls` — for each observed `Op::Call` site (keyed by the caller
//!   frame's resume offset, `frames[n-2].ip`), the callee `BodyId`s seen —
//!   a single target installs a monomorphic inline cache.
//!
//! [`step`]: vm::bc::jit::step

use std::cell::RefCell;
use std::collections::HashMap;

use compile::{Decode, Op, Program};
use vm::Vm;
use vm::bc::jit::{frame_base, frame_nregs, regs_ptr, step_at};
use vm::bc::{
    BodyFn, BodyId, Chunk, Ctx, Decoder, Flow, Frame, Function, Gc, IdVec, RtResult, StrInterner,
    ThreadState, Val,
};

use crate::{BodyFacts, Error, Facts, Jit, Obs, ObsTag};

/// One register's merge state: `tag` merges `differing → Mixed`, `ptr`
/// merges `differing → 0`. `seen` distinguishes "never entered" from an
/// observed `Unknown`.
#[derive(Clone, Copy, Default)]
struct RegObs {
    seen: bool,
    obs: Obs,
}

fn classify(v: Val<'_>) -> Obs {
    let (tag, ptr) = match v {
        Val::Null => (ObsTag::Null, 0),
        Val::Bool(_) => (ObsTag::Bool, 0),
        Val::Int(_) => (ObsTag::Int, 0),
        Val::Float(_) => (ObsTag::Float, 0),
        Val::Fn(_) => (ObsTag::Fn, 0),
        Val::Str(s) => (ObsTag::Str, gc_addr(s.0)),
        Val::Array(a) => (ObsTag::Array, gc_addr(a.0)),
        Val::IntArray(a) => (ObsTag::IntArray, gc_addr(a.0)),
        Val::FloatArray(a) => (ObsTag::FloatArray, gc_addr(a.0)),
        Val::Dict(d) => (ObsTag::Dict, gc_addr(d.0)),
        Val::Instance(i) => (ObsTag::Instance, gc_addr(i.0)),
        Val::Closure(c) => (ObsTag::Closure, gc_addr(c.0)),
        _ => (ObsTag::Other, 0),
    };
    Obs { tag, ptr }
}

fn gc_addr<T>(g: Gc<'_, T>) -> usize {
    Gc::as_ptr(g) as usize
}

/// Fold one observation into the merge: first sighting takes it, later
/// disagreements decay `tag → Mixed` / `ptr → 0`.
fn merge(mo: &mut RegObs, o: Obs) {
    if !mo.seen {
        *mo = RegObs { seen: true, obs: o };
        return;
    }
    if mo.obs.tag != o.tag {
        mo.obs.tag = ObsTag::Mixed;
    }
    if mo.obs.ptr != o.ptr {
        mo.obs.ptr = 0;
    }
}

/// Observation accumulator behind the profiling shims.
struct Profile {
    /// `entry[b][r]` — merged register observation for body `b`.
    entry: Vec<HashMap<u32, RegObs>>,
    /// `(caller body, caller resume ip)` → `callee body` → hit count.
    calls: HashMap<(u32, usize), HashMap<u32, u32>>,
    /// `(body, op ip)` → merged observation of a `GetIndex`/`GetField`'s
    /// receiver operand — mid-body values `entry` never sees (a global
    /// `LoadEntry`'d into a reg still lands here).
    sites: HashMap<(u32, usize), RegObs>,
    /// Fresh-entry count per body.
    counts: Vec<u64>,
}

impl Profile {
    fn new(nbodies: usize) -> Self {
        Profile {
            entry: vec![HashMap::new(); nbodies],
            calls: HashMap::new(),
            sites: HashMap::new(),
            counts: vec![0; nbodies],
        }
    }

    /// Record one fresh frame entry: per-reg tags/ptrs of the window, the
    /// caller's call site (for the IC table), and the hit count.
    ///
    /// SAFETY: `thread`/`chunks` are the driver's live pointers for this
    /// dispatch — read-only here.
    unsafe fn record<'gc>(
        &mut self,
        t: *const ThreadState<'gc>,
        chunks: *const IdVec<BodyId, Chunk>,
    ) {
        unsafe {
            let top = (*t).frames.last().unwrap();
            let b = top.chunk.index();
            if b >= self.counts.len() {
                return;
            }
            self.counts[b] += 1;
            let regs = regs_ptr(t as *mut ThreadState<'gc>);
            let base = top.base;
            let chs: &IdVec<BodyId, Chunk> = &*chunks;
            let n = chs[top.chunk].regs as usize;
            let m = self.entry.get_mut(b).unwrap();
            for r in 0..n {
                let o = classify(*regs.add(base + r));
                merge(m.entry(r as u32).or_default(), o);
            }
            // the caller's saved ip names the `Op::Call` site that got us
            // here — `(caller.chunk, caller.ip)` is the compile-time `next`
            // of that call op.
            if (*t).frames.len() >= 2 {
                let frames: &Vec<Frame> = &(*t).frames;
                let caller = &frames[frames.len() - 2];
                self.calls
                    .entry((caller.chunk.index() as u32, caller.ip))
                    .or_default()
                    .entry(b as u32)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }
    }

    /// Record the receiver operand of the op about to run — `GetIndex`/
    /// `GetField` sites feed `BodyFacts::sites`. Runs *before* `step_at` so
    /// the window still holds the operand; the op is decoded out of a
    /// scratch `Decoder` sharing `code.bytes` (moved out and back — decode
    /// itself never touches the live `ip`).
    ///
    /// SAFETY: same contract as [`record`](Self::record).
    unsafe fn record_site<'gc>(
        &mut self,
        t: *const ThreadState<'gc>,
        code: *mut Decoder,
    ) {
        unsafe {
            let ip = (*code).ip;
            let mut dec = Decoder {
                bytes: std::mem::take(&mut (*code).bytes),
                ip,
            };
            let op = Op::decode(&mut dec);
            (*code).bytes = dec.bytes;
            let r = match op {
                Op::GetIndex { set, .. } => set.index(),
                Op::GetField { src, .. } => src.index(),
                _ => return,
            };
            let top = (*t).frames.last().unwrap();
            let regs = regs_ptr(t as *mut ThreadState<'gc>);
            let o = classify(*regs.add(top.base + r));
            merge(self.sites.entry((top.chunk.index() as u32, ip)).or_default(), o);
        }
    }
}

thread_local! {
    /// The one live observation — body shims borrow it per dispatch.
    /// Reentrant borrows are impossible by construction: `record` drops its
    /// borrow before `step_at` runs a nested call's own shim.
    static PROFILING: RefCell<Option<Profile>> = const { RefCell::new(None) };
}

/// The profiling [`BodyFn`] installed by [`JitSession::observe`]. Runs one
/// interpreter step per dispatch — with the driver's own fuel/op-budget
/// charge — and records the register window on fresh entries.
///
/// SAFETY: the `BodyFn` contract — all pointers borrow the driver's live
/// state; `out` is fully written (by `step_at`) before returning.
unsafe extern "C" fn prof_body<'gc>(
    thread: *mut ThreadState<'gc>,
    code: *mut Decoder,
    ctx: Ctx<'gc>,
    strs: *const StrInterner,
    chunks: *const IdVec<BodyId, Chunk>,
    _signatures: *const IdVec<BodyId, Option<Function>>,
    fuel: *mut usize,
    _op_ip: *mut usize,
    out: *mut RtResult<Flow<'gc>>,
) {
    unsafe {
        // fresh frame entry iff ip sits at the chunk's first op — a body
        // re-dispatched after `Flow::Next` resumes mid-stream and must not
        // be mistaken for an entry (its regs are mid-body values).
        let top = (*thread).frames.last().unwrap();
        let chs: &IdVec<BodyId, Chunk> = &*chunks;
        let fresh = (*code).ip == chs[top.chunk].offset;
        PROFILING.with(|p| {
            if let Some(prof) = &mut *p.borrow_mut() {
                if fresh {
                    prof.record(thread, chunks);
                }
                prof.record_site(thread, code);
            }
        });
        // the driver's own accounting for a `step` dispatch: one op each
        // from `fuel` and `ops_left`.
        *fuel -= 1;
        (*thread).ops_left -= 1;
        let regs = regs_ptr(thread);
        let base = frame_base(thread);
        let n = frame_nregs(thread, chunks);
        step_at(thread, regs.add(base), n, code, ctx, strs, out);
    }
}

/// An observation session over one [`Vm`]'s bodies — see the module docs.
/// Obtain with [`JitSession::observe`], run the workload, then
/// [`specialize`](Self::specialize) to swap in facts-specialized bodies.
pub struct JitSession {
    /// The specialized install — kept alive here; installed bodies are
    /// borrowed code memory.
    jit: Option<Jit>,
}

impl JitSession {
    /// Install profiling shims on `vm` and start recording.
    ///
    /// `program` is the program whose bodies the vm runs (chunk count sizes
    /// the table). The profiling shim runs the interpreter's `step` per op —
    /// observation throughput is interpreter-ish, so observe for a bounded
    /// number of frames/iterations, not the whole run.
    pub fn observe(vm: &mut Vm, program: &Program) -> Result<JitSession, Error> {
        let nbodies = program.chunks.len();
        PROFILING.with(|p| {
            if p.borrow().is_some() {
                return Err(Error(
                    "a JitSession is already observing on this thread".into(),
                ));
            }
            *p.borrow_mut() = Some(Profile::new(nbodies));
            Ok(())
        })?;
        vm.install_bc(vec![Some(prof_body as BodyFn); nbodies]);
        Ok(JitSession { jit: None })
    }

    /// How many fresh entries body `b` has seen — for "recompile after N
    /// calls" tiering policies.
    pub fn calls(&self, body: usize) -> u64 {
        PROFILING.with(|p| {
            p.borrow()
                .as_ref()
                .and_then(|prof| prof.counts.get(body).copied())
                .unwrap_or(0)
        })
    }

    /// A snapshot of the facts gathered so far — bodies not yet entered
    /// carry no observations and specialize as generic.
    pub fn facts(&self) -> Facts {
        PROFILING.with(|p| {
            p.borrow()
                .as_ref()
                .map(|prof| {
                    // merge live state without consuming it
                    let mut f = Facts {
                        bodies: Vec::new(),
                        frozen: HashMap::new(),
                    };
                    for (b, regs) in prof.entry.iter().enumerate() {
                        let n = regs
                            .keys()
                            .copied()
                            .max()
                            .map(|r| r as usize + 1)
                            .unwrap_or(0);
                        let mut entry = vec![Obs::default(); n];
                        for (&r, ro) in regs {
                            entry[r as usize] = if ro.seen { ro.obs } else { Obs::default() };
                        }
                        f.bodies.push(BodyFacts {
                            entry,
                            calls: HashMap::new(),
                            sites: HashMap::new(),
                        });
                        let _ = b;
                    }
                    for (&(cb, ip), callees) in &prof.calls {
                        if callees.len() == 1 {
                            if let Some((&callee, _)) = callees.iter().next() {
                                if let Some(bf) = f.bodies.get_mut(cb as usize) {
                                    bf.calls.insert(ip, callee);
                                }
                            }
                        }
                    }
                    for (&(b, ip), ro) in &prof.sites {
                        if ro.seen {
                            if let Some(bf) = f.bodies.get_mut(b as usize) {
                                bf.sites.insert(ip, ro.obs);
                            }
                        }
                    }
                    f
                })
                .unwrap_or_default()
        })
    }

    /// Declare `ptr` (a `Gc::as_ptr` payload address — typically read out of
    /// [`BodyFacts::entry`]) immutable and add it to the facts. See
    /// [`crate::Frozen`] for the contract — wrong answers, not crashes, if
    /// it's broken.
    pub fn freeze(&self, facts: &mut Facts, ptr: usize, kind: crate::FrozenKind) {
        if ptr != 0 {
            facts
                .frozen
                .insert(ptr, crate::Frozen { kind, unguarded: false });
        }
    }

    /// Compile `program` with `facts` and install the specialized bodies on
    /// `vm`, replacing the profiling shims (observation ends). The returned
    /// [`Jit`] is owned by the session — keep the session alive while the
    /// bodies are installed.
    ///
    /// Safe to call mid-run between `run_frame`/`run` boundaries: installed
    /// bodies are looked up per dispatch, so in-flight frames continue in
    /// the new code on their next op.
    pub fn specialize(&mut self, vm: &mut Vm, program: &Program, facts: &Facts) -> Result<(), Error> {
        let jit = crate::compile_with(program, facts)?;
        vm.install_bc(jit.bodies());
        PROFILING.with(|p| *p.borrow_mut() = None);
        self.jit = Some(jit);
        Ok(())
    }

    /// [`specialize`](Self::specialize) with the facts gathered so far.
    pub fn specialize_now(&mut self, vm: &mut Vm, program: &Program) -> Result<(), Error> {
        let facts = self.facts();
        self.specialize(vm, program, &facts)
    }
}

impl Drop for JitSession {
    fn drop(&mut self) {
        // stop recording even if `specialize` never ran — the profiling
        // shims remain installed but become no-ops that just run `step`.
        PROFILING.with(|p| *p.borrow_mut() = None);
    }
}
