//! A [`Snapshot`] is the whole paused state of a [`Vm`] cloned out of the GC arena into plain
//! data: the thread's registers and call frames, the decoder's `ip`, and every heap object still
//! reachable. [`Vm::restore`] writes one back — allocating fresh `Gc` objects for the snapshot's
//! nodes and rewiring the thread — so the program resumes exactly where the snapshot caught it.
//!
//! Why clone instead of serialize: snapshots are a host-memory structure (the replay panel's
//! checkpoint ring keeps a few dozen). They never cross a wire, so they keep `Arc<str>` and
//! `BodyId` handles rather than bytes — no encode/decode step, no format.
//!
//! Two boundary rules:
//!
//! * A snapshot belongs to the Vm **and program** it was taken on — node `BodyId`s index that
//!   program's chunks. `Vm::load_program` makes every outstanding snapshot meaningless; hosts
//!   must drop them when they swap programs (the wasm bridge keys them per slot for this).
//! * Only `regs`-reachable state survives — that's the same root set the collector keeps.
//!   `Stashed` handles are host-side and invalid after restore (`restore` resets the root set,
//!   so a stale handle fails loudly via [`Ctx::holds`] rather than silently reading an orphaned
//!   object). Values that can't be cloned out — a `DataFrame`, `PlExpr`, `GroupBy`, or
//!   `DarklyImage` held mid-game — make the snapshot itself fail with
//!   [`SnapError::Unshareable`]; the host's fallback is replaying inputs from frame 0.
//!
//! Sharing and cycles are preserved: every `Gc` object is a node in `nodes`, keyed by pointer
//! identity on snapshot and re-created exactly once on restore, so two aliases of one array stay
//! aliases (and `==`'s `Gc::ptr_eq` fast path still answers honestly).

use std::collections::{HashSet, VecDeque};

use gc_arena::Gc;
use rustc_hash::FxHashMap;
use shared::{BodyId, Location};

use crate::{
    Fields, Val,
    fixtures::{DebugInfo, Prints},
    heap::Frame,
    val::{DictMap, SharedStr},
    vm::Vm,
};

/// A [`Val`] as plain data. `Node(i)` indexes [`Snapshot::nodes`]; the leaf kinds are inline.
#[derive(Debug, Clone)]
pub enum SnapVal {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    /// `Val::Fn` — a program-level body, stable for the snapshot's lifetime.
    Fn(u32),
    /// Strings travel by content and re-intern on restore — `Str` equality is content-wise,
    /// so identity needn't survive (and can't: the interner dedupes).
    Str(SharedStr),
    Raised(SharedStr),
    Node(u32),
}

/// One heap object, children already snapped. `Closure` is immutable once created (no
/// `RefLock`), so on restore it's built whole with resolved captures — see the two-phase
/// comment in [`Vm::restore`].
#[derive(Debug, Clone)]
pub enum SnapNode {
    Array(Vec<SnapVal>),
    /// A live `IntArray`/`FloatArray` holding a primitive store — the raw
    /// elements ride the snapshot unboxed, so a typed array restores as a
    /// typed array. A typed array whose store demoted to `Vals` snapshots as
    /// `SnapNode::Array` of the *inner* handle (see `Snapper::val`) and
    /// restores as `Val::Array` — same contents, same aliasing.
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
    Dict(Vec<(SharedStr, SnapVal)>),
    Instance {
        struct_id: u32,
        fields: Vec<SnapVal>,
    },
    Closure {
        function: u32,
        captures: Vec<SnapVal>,
    },
}

/// A call [`Frame`] as plain data — `chunk`/`ip` are only meaningful to the snapshot's program.
#[derive(Debug, Clone)]
pub struct SnapFrame {
    pub chunk: u32,
    pub ip: usize,
    pub return_reg: u32,
    pub base: usize,
}

/// Everything [`Vm::restore`] needs to resume the thread mid-flight.
#[derive(Debug)]
pub struct Snapshot {
    /// `Decoder::ip` — the live instruction pointer (the top frame's `ip` is only a save slot).
    pub ip: usize,
    /// `ThreadState::ops_left` at snapshot time — hosts re-arm budgets per entry anyway.
    pub ops_left: u64,
    /// The flat register file — every live frame's window sits inside it.
    pub regs: Vec<SnapVal>,
    pub frames: Vec<SnapFrame>,
    /// Heap objects in discovery order — `SnapVal::Node` indexes here.
    pub nodes: Vec<SnapNode>,
    /// `Prints`' warn-dedupe set — carries across a restore so a warn that already fired
    /// inside a per-frame loop doesn't re-fire once into the host's popovers.
    pub warned: HashSet<(Location, String)>,
}

/// Why a snapshot or restore couldn't happen.
#[derive(Debug)]
pub enum SnapError {
    /// A `regs`-reachable value can't leave the arena (a polars `DataFrame`/`Expr`/`GroupBy` or
    /// a `DarklyImage` — all host-foreign types). The host's fallback is input replay.
    Unshareable(&'static str),
    /// Restore hit a closure-capture cycle — unreachable in practice (a closure can only
    /// capture what existed when it was made), defended so the walk can't recurse forever.
    ClosureCycle,
}

impl std::fmt::Display for SnapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapError::Unshareable(kind) => write!(f, "a {kind} in game state can't snapshot"),
            SnapError::ClosureCycle => write!(f, "closure capture cycle in snapshot restore"),
        }
    }
}

impl std::error::Error for SnapError {}

/// The walk's bookkeeping: `Gc` pointer → node index, plus the queue of nodes whose contents
/// still need snapping (breadth-first — a node's own children may reference the node itself).
#[derive(Default)]
struct Snapper<'gc> {
    nodes: Vec<SnapNode>,
    ids: FxHashMap<*const (), u32>,
    pending: VecDeque<(u32, Val<'gc>)>,
}

impl<'gc> Snapper<'gc> {
    /// Register `ptr`/`val` as a node (or reuse its existing index) and queue it for filling.
    fn node(&mut self, ptr: *const (), val: Val<'gc>) -> SnapVal {
        if let Some(&i) = self.ids.get(&ptr) {
            return SnapVal::Node(i);
        }
        let i = self.nodes.len() as u32;
        self.nodes.push(SnapNode::Array(Vec::new())); // placeholder, filled from `pending`
        self.ids.insert(ptr, i);
        self.pending.push_back((i, val));
        SnapVal::Node(i)
    }

    fn val(&mut self, v: Val<'gc>) -> Result<SnapVal, SnapError> {
        Ok(match v {
            Val::Null => SnapVal::Null,
            Val::Bool(b) => SnapVal::Bool(b),
            Val::Int(i) => SnapVal::Int(i),
            Val::Float(f) => SnapVal::Float(f),
            Val::Fn(b) => SnapVal::Fn(b.into()),
            Val::Str(s) => SnapVal::Str(Gc::as_ref(s.0).clone()),
            Val::Raised(s) => SnapVal::Raised(Gc::as_ref(s.0).clone()),
            Val::Array(a) => self.node(Gc::as_ptr(a.0) as *const (), v),
            // A demoted typed array snaps as the *inner* `Array`'s node —
            // keyed on the inner `Gc` — so a `Val::Array` handle a native
            // pulled out of it earlier (via `as_untyped_array`) snaps to the
            // same node and the alias survives restore.
            Val::IntArray(a) | Val::FloatArray(a)
                if matches!(&*a.0.borrow(), crate::val::ArrayStore::Vals(_)) =>
            {
                let crate::val::ArrayStore::Vals(inner) = &*a.0.borrow() else {
                    unreachable!()
                };
                self.node(Gc::as_ptr(inner.0) as *const (), Val::Array(*inner))
            }
            Val::IntArray(a) => self.node(Gc::as_ptr(a.0) as *const (), v),
            Val::FloatArray(a) => self.node(Gc::as_ptr(a.0) as *const (), v),
            Val::Dict(d) => self.node(Gc::as_ptr(d.0) as *const (), v),
            Val::Instance(i) => self.node(Gc::as_ptr(i.0) as *const (), v),
            Val::Closure(c) => self.node(Gc::as_ptr(c.0) as *const (), v),
            #[cfg(feature = "dataframe")]
            Val::DataFrame(_) => return Err(SnapError::Unshareable("dataframe")),
            #[cfg(feature = "dataframe")]
            Val::PlExpr(_) => return Err(SnapError::Unshareable("expr")),
            #[cfg(feature = "dataframe")]
            Val::GroupBy(_) => return Err(SnapError::Unshareable("group_by")),
            #[cfg(feature = "darkly")]
            Val::DarklyImage(_) => return Err(SnapError::Unshareable("image")),
        })
    }

    /// Snap a queued node's contents into `nodes[i]` — children are copied out under a shared
    /// borrow first so the recursive `val` calls can't re-borrow the same `RefLock`.
    fn fill(&mut self, i: u32, v: Val<'gc>) -> Result<(), SnapError> {
        self.nodes[i as usize] = match v {
            Val::Array(a) => {
                let items: Vec<Val> = a.0.borrow().clone();
                let items = items
                    .iter()
                    .map(|&v| self.val(v))
                    .collect::<Result<_, _>>()?;
                SnapNode::Array(items)
            }
            // primitives ride unboxed; the node's tag is chosen by the store's
            // content kind (an `IntArray` holding `Floats` after a mixed-write
            // restore comes back a `FloatArray` — tags are birth hints only).
            // `Vals` never reaches here: `val` forwards demoted stores to the
            // inner `Array`'s node.
            Val::IntArray(a) | Val::FloatArray(a) => match &*a.0.borrow() {
                crate::val::ArrayStore::Empty => SnapNode::IntArray(Vec::new()),
                crate::val::ArrayStore::Ints(v) => SnapNode::IntArray(v.clone()),
                crate::val::ArrayStore::Floats(v) => SnapNode::FloatArray(v.clone()),
                crate::val::ArrayStore::Vals(_) => {
                    unreachable!("demoted stores snap as the inner array's node")
                }
            },
            Val::Dict(d) => {
                let items: Vec<(SharedStr, Val)> =
                    d.0.borrow()
                        .iter()
                        .map(|(k, &v)| (Gc::as_ref(k.0).clone(), v))
                        .collect();
                let items = items
                    .into_iter()
                    .map(|(k, v)| Ok((k, self.val(v)?)))
                    .collect::<Result<_, SnapError>>()?;
                SnapNode::Dict(items)
            }
            Val::Instance(inst) => {
                let (struct_id, fields): (u32, Vec<Val>) = {
                    let b = inst.0.borrow();
                    (b.struct_id, b.fields.as_slice().to_vec())
                };
                let fields = fields
                    .iter()
                    .map(|&v| self.val(v))
                    .collect::<Result<_, _>>()?;
                SnapNode::Instance { struct_id, fields }
            }
            Val::Closure(c) => {
                let (function, captures): (u32, Vec<Val>) = {
                    let data = Gc::as_ref(c.0);
                    (data.function.into(), data.captures.clone())
                };
                let captures = captures
                    .iter()
                    .map(|&v| self.val(v))
                    .collect::<Result<_, _>>()?;
                SnapNode::Closure { function, captures }
            }
            _ => unreachable!("only container vals queue as nodes"),
        };
        Ok(())
    }
}

/// The restore's value resolver — `resolved[i]` memoizes each node's fresh `Gc`. Containers
/// (which hold their children through a `RefLock`) are pre-allocated empty before resolution
/// starts, so a `SnapVal::Node` pointing at one returns its handle immediately and heap cycles
/// resolve without recursion. Closures are built on demand with their captures already thawed —
/// they can't be pre-allocated (immutable) and can't cycle (a capture names an older value).
struct Thaw<'gc, 'a> {
    nodes: &'a [SnapNode],
    resolved: Vec<Option<Val<'gc>>>,
    visiting: HashSet<u32>,
    ctx: crate::Ctx<'gc>,
}

impl<'gc> Thaw<'gc, '_> {
    fn thaw(&mut self, sv: &SnapVal) -> Result<Val<'gc>, SnapError> {
        Ok(match sv {
            SnapVal::Null => Val::Null,
            SnapVal::Bool(b) => Val::Bool(*b),
            SnapVal::Int(i) => Val::Int(*i),
            SnapVal::Float(f) => Val::Float(*f),
            SnapVal::Fn(b) => Val::Fn(BodyId::from(*b)),
            SnapVal::Str(s) => Val::Str(self.ctx.intern(s)),
            SnapVal::Raised(s) => Val::Raised(self.ctx.intern(s)),
            SnapVal::Node(i) => match self.resolved[*i as usize] {
                Some(v) => v,
                None => self.thaw_closure(*i)?,
            },
        })
    }

    fn thaw_closure(&mut self, i: u32) -> Result<Val<'gc>, SnapError> {
        let SnapNode::Closure { function, captures } = &self.nodes[i as usize] else {
            unreachable!("pre-allocated nodes are never thawed");
        };
        if !self.visiting.insert(i) {
            return Err(SnapError::ClosureCycle);
        }
        let captures = captures
            .iter()
            .map(|sv| self.thaw(sv))
            .collect::<Result<Vec<_>, _>>()?;
        self.visiting.remove(&i);
        let c = self.ctx.new_closure(BodyId::from(*function), captures);
        let v = Val::Closure(c);
        self.resolved[i as usize] = Some(v);
        Ok(v)
    }
}

impl Vm {
    /// Clone the paused thread's full reachable state out of the arena. Meaningful only between
    /// dispatch runs (a host's frame boundary) — mid-op state isn't a resumable point.
    pub fn snapshot(&mut self) -> Result<Snapshot, SnapError> {
        let Vm { code, arena, .. } = self;
        let ip = code.ip;
        arena.mutate(|_, state| {
            let mut snapper = Snapper::default();
            let (regs, frames, ops_left) = {
                let t = state.thread.borrow();
                let regs = t
                    .regs
                    .iter()
                    .map(|&v| snapper.val(v))
                    .collect::<Result<Vec<_>, _>>()?;
                let frames = t
                    .frames
                    .iter()
                    .map(|f| SnapFrame {
                        chunk: f.chunk.into(),
                        ip: f.ip,
                        return_reg: f.return_reg,
                        base: f.base,
                    })
                    .collect();
                (regs, frames, t.ops_left)
            };
            while let Some((i, v)) = snapper.pending.pop_front() {
                snapper.fill(i, v)?;
            }
            let warned = state.fixtures.get::<Prints>().warned.borrow().clone();
            Ok(Snapshot {
                ip,
                ops_left,
                regs,
                frames,
                nodes: snapper.nodes,
                warned,
            })
        })
    }

    /// Write `snap` back: fresh heap objects for its nodes, the thread's registers/frames
    /// restored verbatim, `code.ip` rewound. Objects the restore replaces stay allocated until
    /// the next collection — the arena is never cleared, the root set simply moves on.
    ///
    /// The snapshot must come from this same Vm and program — `BodyId`s index its chunks.
    pub fn restore(&mut self, snap: &Snapshot) -> Result<(), SnapError> {
        let Vm { code, arena, .. } = self;
        arena.mutate(|mc, state| {
            let ctx = state.ctx(mc);
            let mut thaw = Thaw {
                nodes: &snap.nodes,
                resolved: vec![None; snap.nodes.len()],
                visiting: HashSet::new(),
                ctx,
            };
            // Pre-allocate every mutable container so cycles and closures resolve against real
            // handles; fill them once resolution exists.
            for (i, node) in snap.nodes.iter().enumerate() {
                thaw.resolved[i] = Some(match node {
                    SnapNode::Array(_) => Val::Array(ctx.new_array(Vec::new())),
                    SnapNode::IntArray(_) => Val::IntArray(ctx.new_int_array(Vec::new())),
                    SnapNode::FloatArray(_) => Val::FloatArray(ctx.new_float_array(Vec::new())),
                    SnapNode::Dict(_) => Val::Dict(ctx.new_dict(DictMap::new())),
                    SnapNode::Instance { struct_id, .. } => {
                        Val::Instance(ctx.new_instance(*struct_id, Fields::new(Vec::new())))
                    }
                    SnapNode::Closure { .. } => continue,
                });
            }
            let regs = snap
                .regs
                .iter()
                .map(|sv| thaw.thaw(sv))
                .collect::<Result<Vec<_>, _>>()?;
            for (i, node) in snap.nodes.iter().enumerate() {
                match (node, thaw.resolved[i]) {
                    (SnapNode::Array(items), Some(Val::Array(a))) => {
                        let vals = items
                            .iter()
                            .map(|sv| thaw.thaw(sv))
                            .collect::<Result<Vec<_>, _>>()?;
                        *a.0.borrow_mut(mc) = vals;
                    }
                    (SnapNode::IntArray(items), Some(Val::IntArray(a))) => {
                        *a.0.borrow_mut(mc) = crate::val::ArrayStore::Ints(items.clone());
                    }
                    (SnapNode::FloatArray(items), Some(Val::FloatArray(a))) => {
                        *a.0.borrow_mut(mc) = crate::val::ArrayStore::Floats(items.clone());
                    }
                    (SnapNode::Dict(items), Some(Val::Dict(d))) => {
                        let mut map = DictMap::new();
                        for (k, sv) in items {
                            map.insert(ctx.intern(k), thaw.thaw(sv)?);
                        }
                        *d.0.borrow_mut(mc) = map;
                    }
                    (SnapNode::Instance { fields, .. }, Some(Val::Instance(inst))) => {
                        let vals = fields
                            .iter()
                            .map(|sv| thaw.thaw(sv))
                            .collect::<Result<Vec<_>, _>>()?;
                        inst.0.borrow_mut(mc).fields = Fields::new(vals);
                    }
                    (SnapNode::Closure { .. }, _) => {} // built whole in `thaw_closure`
                    _ => unreachable!("thawed node kind matches its snapshot kind"),
                }
            }
            {
                let mut t = state.thread.borrow_mut(mc);
                t.regs = regs;
                t.frames = snap
                    .frames
                    .iter()
                    .map(|f| Frame {
                        chunk: BodyId::from(f.chunk),
                        ip: f.ip,
                        return_reg: f.return_reg,
                        base: f.base,
                    })
                    .collect();
                t.ops_left = snap.ops_left;
            }
            code.ip = snap.ip;
            // Stashed host handles name the heap the snapshot replaced — invalidate them
            // rather than let a fetch return an orphaned object.
            ctx.reset_roots();
            *state.fixtures.get::<Prints>().warned.borrow_mut() = snap.warned.clone();
            // Rebuild the (chunk, ip) mirror `DebugInfo` keeps for warn call sites: parents
            // hold their frozen return-site ips, the top frame the live `code.ip`.
            let mut stack: Vec<(BodyId, u32)> = snap
                .frames
                .iter()
                .map(|f| (BodyId::from(f.chunk), f.ip as u32))
                .collect();
            if let Some(top) = stack.last_mut() {
                top.1 = snap.ip as u32;
            }
            *state.fixtures.get::<DebugInfo>().stack.borrow_mut() = stack;
            Ok(())
        })
    }
}
