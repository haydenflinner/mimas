use api::AdtBinding;
use compile::{BodyId, Function};
use gc_arena::{
    Collect, DynamicRoot, DynamicRootSet, Gc, GcWeak, Mutation, Rootable,
    lock::{Lock, RefLock},
};
use rustc_hash::FxHashMap;
use shared::{FnHeader, IdVec, units::Shape};
use std::{
    any::TypeId,
    cell::{Cell, Ref, RefCell},
    fmt::Write as _,
    ops,
    rc::Rc,
    sync::Arc,
};

use crate::{
    Fields, RtErr, RtResult,
    val::{
        Array, ArrayStore, Closure, ClosureData, Dict, DictMap, FloatArray, Instance, InstanceData,
        IntArray, Seq, SharedStr, Str, Val,
    },
};

const MAX_DISPLAY_DEPTH: usize = 1000;

/// The state of one display walk: `path` is the stack of containers around the value
/// being printed (a cycle revisits one of them), `seen` every struct instance printed so
/// far (an alias revisits one of those).
#[derive(Default)]
struct Walk {
    path: Vec<*const ()>,
    seen: Vec<*const ()>,
}

#[derive(Default, Debug)]
pub struct MimasBindings(pub FxHashMap<TypeId, AdtBinding>);

// SAFETY: TypeId + AdtBinding are 'static, no Gc handles inside.
unsafe impl<'gc> Collect<'gc> for MimasBindings {
    const NEEDS_TRACE: bool = false;
}

#[derive(Collect, Clone, Debug)]
#[collect(require_static)]
pub struct Frame {
    pub chunk: BodyId,
    pub ip: usize,
    pub return_reg: u32,
    pub base: usize,
}

#[derive(Collect, Clone, Debug)]
#[collect(no_drop)]
pub struct Cleanup<'gc> {
    pub depth: usize,
    pub action: Val<'gc>,
}

pub type Cleanups<'gc> = Gc<'gc, RefLock<Vec<Cleanup<'gc>>>>;

#[derive(Debug)]
pub struct ThreadState<'gc> {
    pub regs: Vec<Val<'gc>>,
    pub frames: Vec<Frame>,
    pub cleanups: Cleanups<'gc>,
    /// Ops this thread may still run before `RtErr::OutOfFuel`; the host re-arms it per entry
    /// (`Vm::set_op_budget`). `u64::MAX` (the default) is effectively unlimited.
    pub ops_left: u64,
    /// Frames the thread may hold before a call fails with `RtErr::RecursionTooDeep`
    /// (`Vm::set_max_depth`).
    pub max_frames: usize,
}

// Manual so a full mark can tally the register stack's bytes for the memory budget (see
// `crate::mem`); otherwise the same trace the derive wrote.
unsafe impl<'gc> Collect<'gc> for ThreadState<'gc> {
    fn trace<T: gc_arena::collect::Trace<'gc>>(&self, cc: &mut T) {
        crate::mem::note_traced(
            self.cleanups.borrow().capacity() * std::mem::size_of::<Cleanup>()
                + self.regs.capacity() * std::mem::size_of::<Val>()
                + self.frames.capacity() * std::mem::size_of::<Frame>(),
        );
        cc.trace(&self.regs);
        cc.trace(&self.cleanups);
    }
}

pub type Thread<'gc> = Gc<'gc, RefLock<ThreadState<'gc>>>;

#[derive(Copy, Clone, Collect)]
#[collect(no_drop)]
pub struct InternedStrings<'gc>(pub Gc<'gc, RefLock<FxHashMap<SharedStr, GcWeak<'gc, SharedStr>>>>);

impl<'gc> InternedStrings<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        InternedStrings(Gc::new(mc, RefLock::new(FxHashMap::default())))
    }

    /// The interned `Str` for `s`, and whether this call allocated it.
    /// Strings that fit `Str::INLINE_MAX` pack into the value word itself —
    /// they never reach the table at all.
    pub fn intern(self, mc: &Mutation<'gc>, s: &str) -> (Str<'gc>, bool) {
        if let Some(short) = Str::inline(s) {
            return (short, false);
        }
        self.intern_heap_pair(mc, s)
    }

    /// Always-heap interning — used where an inline `Str` can't satisfy
    /// the caller (a `'gc` borrow into arena memory). Shares the table
    /// with `intern`, so a materialized short string still dedups.
    fn intern_heap_pair(self, mc: &Mutation<'gc>, s: &str) -> (Str<'gc>, bool) {
        if let Some(weak) = self.0.borrow().get(s)
            && let Some(gc) = weak.upgrade(mc)
        {
            return (Str::from_gc(gc), false);
        }
        let shared: SharedStr = Arc::from(s);
        let gc = Gc::new(mc, shared.clone());
        self.0.borrow_mut(mc).insert(shared, Gc::downgrade(gc));
        (Str::from_gc(gc), true)
    }

    /// Drop the table entries of strings the finished mark found dead, and return the byte total
    /// of the strings still live. Without this the table's own `Arc` keeps every string the
    /// program ever made in memory.
    pub(crate) fn prune_dead(self, fc: &gc_arena::Finalization<'gc>) -> usize {
        let mut live = 0usize;
        self.0.borrow_mut(fc).retain(|k, weak| {
            if weak.is_dead(fc) {
                false
            } else {
                live += k.len() + std::mem::size_of::<SharedStr>() * 2;
                true
            }
        });
        live
    }
}

#[derive(Clone, Collect)]
#[collect(no_drop)]
pub struct State<'gc> {
    pub strings: InternedStrings<'gc>,
    pub thread: Thread<'gc>,
    pub active_thread: Gc<'gc, Lock<Thread<'gc>>>,
    pub active_cleanups: Gc<'gc, Lock<Cleanups<'gc>>>,
    // Transient dispatch pointer. The root thread traces these values.
    #[collect(require_static)]
    pub entry_regs: Cell<*mut ()>,
    #[collect(require_static)]
    pub generator_parents: RefCell<Vec<Frame>>,
    pub generator_value: Gc<'gc, Lock<Option<Val<'gc>>>>,
    pub natives: Gc<'gc, RefLock<Vec<Option<crate::native::NativeRef<'gc>>>>>,
    pub mimas_bindings: Gc<'gc, RefLock<MimasBindings>>,
    pub fixtures: Gc<'gc, crate::fixtures::Fixtures>,
    /// Struct/variant names indexed by `struct_id` (see [`InstanceData::struct_id`]), for
    /// rendering instances as `Name { .. }` instead of `@id { .. }`. Filled in by
    /// `Vm::load_program`.
    pub struct_names: Gc<'gc, RefLock<Vec<String>>>,
    /// Declared field names per struct, indexed by `struct_id` like `struct_names` -- a duplicate
    /// of `Vm::field_names` (which natives can't reach; they only ever see a `Ctx`, never the
    /// outer `Vm`), kept in the arena so a native fn can do name-aware reflection over an
    /// `Instance` (e.g. `to_dataframe`, converting an array of structs into DataFrame columns
    /// named after their declared fields). Filled in by `Vm::load_program`.
    pub field_names: Gc<'gc, RefLock<Vec<Vec<String>>>>,
    /// For a `struct_id` that is an enum variant's layout, the variant's index
    /// in declaration order (`Some`); `None` for structs. Indexed like
    /// `struct_names` -- the third table [`Val::inspect`] needs, so an
    /// `Instance` can carry its variant ordinal into [`Inspect::Instance`].
    /// Filled in by `Vm::load_program`.
    pub variant_ordinals: Gc<'gc, RefLock<Vec<Option<u32>>>>,
    /// Swapped for a fresh set every time a program is loaded, which is what invalidates the
    /// handles the host took out against the previous one -- see `Ctx::reset_roots`.
    pub roots: Gc<'gc, Lock<DynamicRootSet<'gc>>>,
    /// Cooperative pause, set by `yield_frame`-style natives. Lives on `State` (not
    /// `ThreadState`) because the dispatch loop holds `thread.borrow_mut` for the whole
    /// run -- a native writing thread state mid-op would double-borrow and panic. The
    /// dispatch loop reads it per op like a second fuel line.
    pub paused: Cell<bool>,
    /// Pause-and-retry, set by a native that needs the host to run an errand before the
    /// call can answer (a `ws` op whose receipt arrives next frame): the dispatch loop
    /// rewinds to the call op and exits paused, so the resumed native runs again and
    /// returns the real value. Read alongside `paused` the way the generator `resume`
    /// retry is, not instead of it.
    #[collect(require_static)]
    pub retry_call: Cell<bool>,
    #[collect(require_static)]
    pub unwinding_cleanup: Cell<bool>,
    /// On while the dispatch loop can replay the current op after running
    /// the `show`s a display left pending (r[impl mimas.show.runtime]). A
    /// field, not a fixture: the loop writes it around every op.
    #[collect(require_static)]
    pub show_replay: Cell<bool>,
    /// A display recorded `show`s to run (see `fixtures::ShowCalls`).
    #[collect(require_static)]
    pub show_pending: Cell<bool>,
    /// Whether any type of the loaded program implements `Show`: off, the
    /// renderer never looks for a `show`.
    #[collect(require_static)]
    pub any_show: Cell<bool>,
    /// The memory budget and the nested-call count — see [`crate::mem`].
    #[collect(require_static)]
    pub mem: crate::mem::MemBudget,
}

impl<'gc> State<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let thread = Gc::new(
            mc,
            RefLock::new(ThreadState {
                regs: Vec::new(),
                frames: Vec::new(),
                cleanups: Gc::new(mc, RefLock::new(vec![])),
                ops_left: u64::MAX,
                max_frames: crate::mem::DEFAULT_MAX_FRAMES,
            }),
        );
        let natives = Gc::new(mc, RefLock::new(Vec::new()));
        let mimas_bindings = Gc::new(mc, RefLock::new(MimasBindings::default()));
        let fixtures = Gc::new(mc, crate::fixtures::Fixtures::default());
        let struct_names = Gc::new(mc, RefLock::new(Vec::new()));
        let field_names = Gc::new(mc, RefLock::new(Vec::new()));
        let variant_ordinals = Gc::new(mc, RefLock::new(Vec::new()));
        State {
            strings: InternedStrings::new(mc),
            thread,
            active_thread: Gc::new(mc, Lock::new(thread)),
            active_cleanups: Gc::new(mc, Lock::new(thread.borrow().cleanups)),
            entry_regs: Cell::new(std::ptr::null_mut()),
            generator_parents: RefCell::new(vec![]),
            generator_value: Gc::new(mc, Lock::new(None)),
            natives,
            mimas_bindings,
            fixtures,
            struct_names,
            field_names,
            variant_ordinals,
            roots: Gc::new(mc, Lock::new(DynamicRootSet::new(mc))),
            paused: Cell::new(false),
            retry_call: Cell::new(false),
            unwinding_cleanup: Cell::new(false),
            show_replay: Cell::new(false),
            show_pending: Cell::new(false),
            any_show: Cell::new(false),
            mem: crate::mem::MemBudget::default(),
        }
    }

    pub fn ctx(&'gc self, mutation: &'gc Mutation<'gc>) -> Ctx<'gc> {
        Ctx {
            mutation,
            state: self,
        }
    }
}

// `repr(C)` so `Ctx` can ride the `extern "C"` `bc::BodyFn` ABI by value:
// two pointers, defined order — JIT bodies take it as two machine words.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Ctx<'gc> {
    mutation: &'gc Mutation<'gc>,
    state: &'gc State<'gc>,
}

impl<'gc> ops::Deref for Ctx<'gc> {
    type Target = Mutation<'gc>;
    fn deref(&self) -> &Self::Target {
        self.mutation
    }
}

impl<'gc> Ctx<'gc> {
    /// Rebuild a `Ctx` from its two machine words. `bc::jit` shims take the
    /// pair split into two params: AAPCS64 never splits a composite across the
    /// register/stack boundary, so a by-value `Ctx` at argument position ≥ 7
    /// lands *entirely* on the stack while the emitter's flat signature sends
    /// the first word in a register — a corrupting ABI mismatch.
    #[doc(hidden)]
    pub unsafe fn from_parts(mutation: *const Mutation<'gc>, state: *const State<'gc>) -> Ctx<'gc> {
        // SAFETY: callers pass the two words of a live `Ctx` verbatim.
        unsafe {
            Ctx {
                mutation: &*mutation,
                state: &*state,
            }
        }
    }

    pub fn state(self) -> &'gc State<'gc> {
        self.state
    }

    pub fn mutation(self) -> &'gc Mutation<'gc> {
        self.mutation
    }

    pub fn intern(self, s: &str) -> Str<'gc> {
        let (str, fresh) = self.state.strings.intern(self.mutation, s);
        if fresh {
            self.charge(s.len() + crate::mem::BOX_BYTES);
        }
        str
    }

    /// Intern always as a heap `Gc` — never inline — for callers that need
    /// `Str::as_gc_str` (arena-memory `&'gc str`) rather than `as_str`.
    pub fn intern_heap(self, s: &str) -> Str<'gc> {
        let (str, fresh) = self.state.strings.intern_heap_pair(self.mutation, s);
        if fresh {
            self.charge(s.len() + crate::mem::BOX_BYTES);
        }
        str
    }

    /// Count `bytes` of new heap against the memory budget (see [`crate::mem`]). The Vm's own
    /// constructors charge for themselves; a native that grows a container in place charges for
    /// the growth. Never fails here: the dispatch loop stops the program at its next op.
    #[inline]
    pub fn charge(self, bytes: usize) {
        self.state.mem.charge(bytes);
    }

    /// Count `ops` against the running entry's op budget — for a native that runs work the
    /// dispatch loop can't see (a nested Vm). Taken off at the next op; an entry whose budget
    /// it exhausts fails with `RtErr::OutOfFuel` there.
    #[inline]
    pub fn charge_ops(self, ops: u64) {
        self.state.mem.charge_ops(ops);
    }

    /// Fail before allocating one string or array of `bytes` that is bigger than the budget allows
    /// for a single value. Use it where a size comes from a script integer.
    pub fn check_alloc(self, bytes: usize) -> RtResult<()> {
        let limit = crate::mem::max_object_bytes(self.state.mem.limit.get());
        if bytes > limit {
            Err(RtErr::TooLarge { limit })
        } else {
            Ok(())
        }
    }

    /// [`check_alloc`](Self::check_alloc) for `count` items of `size` bytes each.
    pub fn check_alloc_n(self, count: usize, size: usize) -> RtResult<()> {
        self.check_alloc(count.checked_mul(size).unwrap_or(usize::MAX))
    }

    /// Fetch the per-Vm fixture of type `T`, creating it on first access -- see the
    /// [fixtures](crate::fixtures) module docs for the full story.
    pub fn fixture<T: crate::fixtures::Fixture>(self) -> &'gc T {
        self.state.fixtures.get::<T>()
    }

    /// A value's [`Inspect`] tree with the program's struct and field names —
    /// what the env pane reads for a `let`, available to a native too.
    pub fn inspect(self, value: Val<'gc>) -> crate::Inspect {
        let struct_names = self.state.struct_names.borrow();
        let field_names = self.state.field_names.borrow();
        let variant_ordinals = self.state.variant_ordinals.borrow();
        let mut seen = std::collections::HashSet::new();
        value.inspect(&struct_names, &field_names, &variant_ordinals, &mut seen)
    }

    /// Emit a non-fatal diagnostic at the current call site: the host gets a
    /// `(call site, text)` pair through [`Prints`](crate::fixtures::Prints)
    /// (same pipeline as `print` popovers, tagged `Warn`), and `warning: …`
    /// reaches the `Out` sink too. Repeated reports of the same message from
    /// the same site are suppressed, so a warn inside a per-frame loop is safe.
    pub fn warn(self, msg: &str) {
        use crate::fixtures::{DebugInfo, Out, Prints};
        let fresh = match self.fixture::<DebugInfo>().top_loc() {
            Some(loc) => self.fixture::<Prints>().push_warn(loc, msg.to_string()),
            // no call-site loc (native driven from outside a run) — can't
            // dedupe, so let it through
            None => true,
        };
        if fresh {
            self.fixture::<Out>().write(&format!("warning: {msg}"));
        }
    }

    /// Register cleanup without borrowing the thread held by dispatch.
    pub fn defer(self, action: Val<'gc>) {
        let depth = self
            .fixture::<crate::fixtures::DebugInfo>()
            .stack
            .borrow()
            .len()
            .saturating_sub(self.state.generator_parents.borrow().len())
            .max(1);
        self.charge(std::mem::size_of::<Cleanup>());
        self.state
            .active_cleanups
            .get()
            .borrow_mut(&self)
            .push(Cleanup { depth, action });
    }

    pub fn promote_cleanups(self) {
        let depth = self
            .fixture::<crate::fixtures::DebugInfo>()
            .stack
            .borrow()
            .len()
            .saturating_sub(self.state.generator_parents.borrow().len());
        for cleanup in self
            .state
            .active_cleanups
            .get()
            .borrow_mut(&self)
            .iter_mut()
        {
            if cleanup.depth == depth {
                cleanup.depth = depth.saturating_sub(1).max(1);
            }
        }
    }

    pub fn cleanup_mark(self) -> usize {
        self.state.active_cleanups.get().borrow().len()
    }

    // r[impl mimas.sub.cleanup-order]
    pub fn take_cleanup(self, mark: usize) -> Option<Val<'gc>> {
        let depth = self
            .fixture::<crate::fixtures::DebugInfo>()
            .stack
            .borrow()
            .len()
            .saturating_sub(self.state.generator_parents.borrow().len());
        let stack = self.state.active_cleanups.get();
        let mut stack = stack.borrow_mut(&self);
        if stack.len() > mark && stack.last().is_some_and(|c| c.depth >= depth) {
            stack.pop().map(|c| c.action)
        } else {
            None
        }
    }

    pub fn thread(self) -> Thread<'gc> {
        self.state.active_thread.get()
    }

    /// Hand a value to the host. The returned handle has no `'gc` of its own, so it can be held
    /// across collections and later read back with [`Ctx::fetch`] or called with
    /// [`Vm::call_value`](crate::Vm::call_value). It keeps the value -- and, for a closure,
    /// everything it captured -- alive for as long as it lives.
    pub fn stash(self, value: Val<'gc>) -> Stashed {
        let rooted = Gc::new(self.mutation, value);
        Stashed(
            self.state
                .roots
                .get()
                .stash::<Rootable![Val<'_>]>(self.mutation, rooted),
        )
    }

    /// Read a [`Stashed`] value back. Panics if the handle isn't one this Vm can still honor --
    /// either it was stashed into a different Vm, or the program it was taken against has since
    /// been replaced by [`Vm::load_program`](crate::Vm::load_program). Check
    /// [`holds`](Self::holds) first when the handle's provenance isn't certain.
    pub fn fetch(self, stashed: &Stashed) -> Val<'gc> {
        match self.state.roots.get().try_fetch(&stashed.0) {
            Ok(value) => *value,
            // gc-arena's own panic just says "mismatched root set", which doesn't hint at the
            // reload -- by far the likelier of the two ways to get here
            Err(_) => panic!(
                "stashed handle doesn't belong to this Vm's loaded program: it was stashed into \
                 another Vm, or taken against a program that has since been replaced"
            ),
        }
    }

    /// Whether [`fetch`](Self::fetch) would still honor `stashed`. False for a handle from another
    /// Vm, and for one taken against a program that has since been replaced.
    pub fn holds(self, stashed: &Stashed) -> bool {
        self.state.roots.get().contains(&stashed.0)
    }

    /// Drop every root, invalidating all outstanding [`Stashed`] handles. A handle names a body in
    /// the program it was taken against, which means something else entirely in the next one, so
    /// loading a program resets the set rather than letting stale handles index into it.
    pub(crate) fn reset_roots(self) {
        self.state
            .roots
            .set(self.mutation, DynamicRootSet::new(self.mutation));
    }

    /// The signature of a script fn or closure `body`. `None` for a body with no fn type of its
    /// own. Given a function value rather than a `BodyId`, reach for [`Val::signature`] instead.
    pub fn signature(self, body: BodyId) -> Option<Ref<'gc, FnHeader>> {
        let table = self.fixture::<Signatures>().0.borrow();
        Ref::filter_map(table, |table| table.get(body)?.as_ref().map(|f| &f.header)).ok()
    }

    /// The binding for a Rust type registered with `add_adt`, by its `TypeId`. It's borrowed, so
    /// don't hold onto it.
    pub fn binding(self, type_id: TypeId) -> Option<Ref<'gc, AdtBinding>> {
        Ref::filter_map(self.state.mimas_bindings.borrow(), |bindings| {
            bindings.0.get(&type_id)
        })
        .ok()
    }

    pub fn new_array(self, items: Vec<Val<'gc>>) -> Array<'gc> {
        self.charge(crate::mem::BOX_BYTES + items.capacity() * std::mem::size_of::<Val>());
        Array(Gc::new(self.mutation, RefLock::new(items.into())))
    }

    /// The `[]` op's result: an empty sequence in `ArrayStore::Empty`, which
    /// picks `Ints`/`Floats`/`Vals` from its first pushed element. Semantically
    /// indistinguishable from an empty `Val::Array` (`[] == []`, prints `[]`),
    /// but the hot `arr = []; for .. arr.push(i)` loop never allocates per
    /// element's 16-byte tag.
    pub fn new_seq(self) -> Val<'gc> {
        self.charge(crate::mem::BOX_BYTES);
        Val::IntArray(Seq(Gc::new(self.mutation, RefLock::new(ArrayStore::Empty))))
    }

    /// Wrap `items` in the cheapest sequence shape that holds them: all-`Int`
    /// becomes `IntArray(Ints)`, all-`Float` becomes `FloatArray(Floats)`, and
    /// anything else stays an ordinary `Val::Array`. (Promotion only ever
    /// happens at construction — see [`ArrayStore`].)
    pub fn array_val(self, items: Vec<Val<'gc>>) -> Val<'gc> {
        if items.iter().all(|v| matches!(v, Val::Int(_))) {
            let ints = items
                .into_iter()
                .map(|v| match v {
                    Val::Int(i) => i,
                    _ => unreachable!(),
                })
                .collect();
            Val::IntArray(self.new_int_array(ints))
        } else if items.iter().all(|v| matches!(v, Val::Float(_))) {
            let floats = items
                .into_iter()
                .map(|v| match v {
                    Val::Float(f) => f,
                    _ => unreachable!(),
                })
                .collect();
            Val::FloatArray(self.new_float_array(floats))
        } else {
            Val::Array(self.new_array(items))
        }
    }

    pub fn new_int_array(self, items: Vec<i64>) -> IntArray<'gc> {
        self.charge(crate::mem::BOX_BYTES + items.capacity() * 8);
        Seq(Gc::new(
            self.mutation,
            RefLock::new(ArrayStore::Ints(items)),
        ))
    }

    pub fn new_float_array(self, items: Vec<f64>) -> FloatArray<'gc> {
        self.charge(crate::mem::BOX_BYTES + items.capacity() * 8);
        Seq(Gc::new(
            self.mutation,
            RefLock::new(ArrayStore::Floats(items)),
        ))
    }

    pub fn new_dict(self, items: DictMap<'gc>) -> Dict<'gc> {
        self.charge(crate::mem::BOX_BYTES + items.heap_bytes());
        Dict(Gc::new(self.mutation, RefLock::new(items)))
    }

    pub fn new_instance(self, struct_id: u32, fields: Fields<'gc>) -> Instance<'gc> {
        self.charge(crate::mem::BOX_BYTES + fields.heap_bytes());
        Instance(Gc::new(
            self.mutation,
            RefLock::new(InstanceData { struct_id, fields }),
        ))
    }

    pub fn new_generator(self, closure: Closure<'gc>) -> Closure<'gc> {
        let data = Gc::as_ref(closure.0);
        self.charge(2 * crate::mem::BOX_BYTES + data.captures.len() * std::mem::size_of::<Val>());
        let thread = Gc::new(
            self.mutation,
            RefLock::new(ThreadState {
                regs: vec![],
                frames: vec![],
                cleanups: Gc::new(self.mutation, RefLock::new(vec![])),
                ops_left: 0,
                max_frames: 0,
            }),
        );
        Closure(Gc::new(
            self.mutation,
            ClosureData {
                function: data.function,
                captures: data.captures.clone(),
                generator: Some(thread),
            },
        ))
    }

    pub fn new_closure(self, function: BodyId, captures: Vec<Val<'gc>>) -> Closure<'gc> {
        self.charge(crate::mem::BOX_BYTES + captures.capacity() * std::mem::size_of::<Val>());
        Closure(Gc::new(
            self.mutation,
            ClosureData {
                function,
                captures,
                generator: None,
            },
        ))
    }

    #[cfg(feature = "dataframe")]
    pub fn new_dataframe(self, df: polars::frame::DataFrame) -> crate::val::DataFrame<'gc> {
        crate::val::DataFrame(Gc::new(self.mutation, RefLock::new(gc_arena::Static(df))))
    }

    #[cfg(feature = "dataframe")]
    pub fn new_plexpr(self, expr: polars::prelude::Expr) -> crate::val::PlExpr<'gc> {
        crate::val::PlExpr(Gc::new(self.mutation, gc_arena::Static(expr)))
    }

    #[cfg(feature = "dataframe")]
    pub fn new_group_by(self, gb: polars::prelude::LazyGroupBy) -> crate::val::GroupBy<'gc> {
        crate::val::GroupBy(Gc::new(self.mutation, gc_arena::Static(gb)))
    }

    #[cfg(feature = "darkly")]
    pub fn new_darkly_image(self, image: crate::val::RawImage) -> crate::val::DarklyImage<'gc> {
        crate::val::DarklyImage(Gc::new(self.mutation, gc_arena::Static(image)))
    }

    #[cfg(feature = "tensor")]
    pub fn new_tensor(self, t: crate::tensor::Prim) -> crate::val::Tensor<'gc> {
        crate::val::Tensor(Gc::new(self.mutation, gc_arena::Static(t)))
    }

    /// `key` as a dict can store it, or why it cannot be a key. A string, int, bool, dec,
    /// null, or float other than NaN is a key as it is. A tuple or struct is a key when all
    /// of its parts are: the dict stores a frozen copy, so the key is the value at insert
    /// time, and a later change to the value the caller holds does not move it. A key read
    /// back out of the dict is that frozen copy. A dict, a function or a NaN is refused.
    /// Lists pass here too (a tuple is a list at run time); the checker refuses a list key
    /// type before a program runs.
    /// r[impl mimas.dict.key.runtime]
    /// r[impl mimas.dict.key.copy]
    pub fn dict_key(self, key: Val<'gc>) -> RtResult<Val<'gc>> {
        match key {
            Val::Str(_) | Val::Int(_) | Val::Bool(_) | Val::Dec(_) | Val::Null => Ok(key),
            Val::Array(_) | Val::IntArray(_) | Val::FloatArray(_) | Val::Instance(_) => {
                Self::key_parts(key)?;
                if self.is_frozen(key) {
                    return Ok(key);
                }
                let copy = self.deep_clone(key);
                self.freeze(copy);
                Ok(copy)
            }
            other => Self::key_parts(other).map(|_| other),
        }
    }

    /// Whether every part of `key` can be part of a dict key.
    fn key_parts(key: Val<'gc>) -> RtResult<()> {
        let refuse = |what: &str| {
            Err(RtErr::Custom(format!(
                "{what} cannot be a dict key: a key is a string, number, bool, or a tuple or \
                 struct made of those"
            )))
        };
        match key {
            Val::Str(_) | Val::Int(_) | Val::Bool(_) | Val::Dec(_) | Val::Null => Ok(()),
            Val::Float(f) if f.is_nan() => Err(RtErr::Custom(
                "NaN cannot be a dict key: NaN is not equal to itself, so it could never be \
                 found again"
                    .into(),
            )),
            Val::Float(_) => Ok(()),
            Val::Array(a) => a.0.borrow().iter().try_for_each(|&v| Self::key_parts(v)),
            Val::IntArray(a) | Val::FloatArray(a) => {
                let a = a.0.borrow();
                (0..a.len()).try_for_each(|i| Self::key_parts(a.at(i)))
            }
            Val::Instance(i) => i.0.borrow().fields.iter().try_for_each(|&v| Self::key_parts(v)),
            Val::Dict(_) => refuse("a dict"),
            Val::Fn(_) | Val::Closure(_) => refuse("a function"),
            Val::Raised(_) => refuse("a raised error"),
            #[allow(unreachable_patterns)]
            _ => refuse("this value"),
        }
    }

    // fresh allocation per container so the result shares no mutable state with `value`. scalars,
    // interned strs, fns and closures are copied by handle (immutable / callable). recurses like
    // `display`, so it shares display's no-cycles assumption.
    pub fn deep_clone(self, value: Val<'gc>) -> Val<'gc> {
        match value {
            Val::Array(a) => {
                let items: Vec<Val<'gc>> =
                    a.0.borrow().iter().map(|&v| self.deep_clone(v)).collect();
                Val::Array(self.new_array(items))
            }
            // clone keeps the backing kind: a typed array deep-clones into a
            // fresh typed array (demoted `Vals` stores re-box through
            // `array_val`, re-promoting if the contents happen to be
            // homogeneous again — invisible either way).
            Val::IntArray(a) | Val::FloatArray(a) => match &*a.0.borrow() {
                ArrayStore::Empty => self.new_seq(),
                ArrayStore::Ints(v) => Val::IntArray(self.new_int_array(v.clone())),
                ArrayStore::Floats(v) => Val::FloatArray(self.new_float_array(v.clone())),
                ArrayStore::Vals(inner) => {
                    let items: Vec<Val<'gc>> = inner
                        .0
                        .borrow()
                        .iter()
                        .map(|&v| self.deep_clone(v))
                        .collect();
                    self.array_val(items)
                }
            },
            Val::Dict(d) => {
                let items: DictMap<'gc> =
                    d.0.borrow()
                        .iter()
                        .map(|(&k, &v)| (k, self.deep_clone(v)))
                        .collect();
                Val::Dict(self.new_dict(items))
            }
            Val::Instance(i) => {
                let (struct_id, fields) = {
                    let inst = i.0.borrow();
                    let fields: Vec<Val<'gc>> =
                        inst.fields.iter().map(|&v| self.deep_clone(v)).collect();
                    (inst.struct_id, fields)
                };
                Val::Instance(self.new_instance(struct_id, Fields::new(fields)))
            }
            other => other,
        }
    }

    /// Make a value read-only, all the way down: the value of a `const`. A list, a
    /// dict or a struct is frozen by a shared borrow that is never given back, so
    /// every later `borrow_mut` on it fails. Reads go on as before. This costs
    /// nothing on a write to any other value: the borrow check is the one a write
    /// already makes. The interpreter's writes turn the failure into
    /// [`RtErr::const_write`]; a write path that does not check panics, and so
    /// still cannot change the value.
    // r[impl mimas.const.frozen]
    pub fn freeze(self, value: Val<'gc>) {
        match value {
            Val::Array(a) => {
                if a.0.try_borrow_mut(self.mutation).is_err() {
                    return; // frozen already (or shared with a value that is)
                }
                let items: Vec<Val<'gc>> = a.0.borrow().iter().copied().collect();
                std::mem::forget(a.0.borrow());
                for v in items {
                    self.freeze(v);
                }
            }
            Val::IntArray(a) | Val::FloatArray(a) => {
                if a.0.try_borrow_mut(self.mutation).is_err() {
                    return;
                }
                let inner = match &*a.0.borrow() {
                    ArrayStore::Vals(inner) => Some(*inner),
                    _ => None,
                };
                std::mem::forget(a.0.borrow());
                if let Some(inner) = inner {
                    self.freeze(Val::Array(inner));
                }
            }
            Val::Dict(d) => {
                if d.0.try_borrow_mut(self.mutation).is_err() {
                    return;
                }
                let items: Vec<Val<'gc>> = d.0.borrow().iter().map(|(_, &v)| v).collect();
                std::mem::forget(d.0.borrow());
                for v in items {
                    self.freeze(v);
                }
            }
            Val::Instance(i) => {
                if i.0.try_borrow_mut(self.mutation).is_err() {
                    return;
                }
                let items: Vec<Val<'gc>> = i.0.borrow().fields.iter().copied().collect();
                std::mem::forget(i.0.borrow());
                for v in items {
                    self.freeze(v);
                }
            }
            _ => {}
        }
    }

    /// Whether a list, a dict or a struct is frozen (see [`freeze`](Self::freeze)).
    pub fn is_frozen(self, value: Val<'gc>) -> bool {
        match value {
            Val::Array(a) => a.0.try_borrow_mut(self.mutation).is_err(),
            Val::IntArray(a) | Val::FloatArray(a) => a.0.try_borrow_mut(self.mutation).is_err(),
            Val::Dict(d) => d.0.try_borrow_mut(self.mutation).is_err(),
            Val::Instance(i) => i.0.try_borrow_mut(self.mutation).is_err(),
            _ => false,
        }
    }

    /// Render a value being *shown* -- strings are quoted (`"x"`). Use this for anything
    /// inspection-shaped: dbg output, test harnesses, error reports.
    ///
    /// Fails with [`RtErr::DisplayTooDeep`] if `value` nests past
    /// [`MAX_DISPLAY_DEPTH`] -- see that constant's docs. A cycle is not an error:
    /// a revisited container prints a `Node 1 ↺` back-reference (see [`Self::render_into`]).
    pub fn display(self, value: Val<'gc>) -> RtResult<String> {
        let mut out = String::new();
        self.render_into(&mut out, value, true, 0, &mut Walk::default(), &Shape::Plain)?;
        self.shows_ready()?;
        Ok(out)
    }

    /// `Err(ShowPending)` when the walk just done met `Show` values whose
    /// `show` has not run: the dispatch loop runs them and replays the op
    /// (r[impl mimas.show.runtime]).
    #[inline(always)]
    pub fn shows_ready(self) -> RtResult<()> {
        if self.state.show_pending.get() && self.state.show_replay.get() {
            return Err(RtErr::ShowPending);
        }
        Ok(())
    }

    /// Writes a `Show` value as its `show` text, when its type implements
    /// `Show` and the text is known; `false` sends the caller to the built-in
    /// form. A text not yet known is recorded as pending (when the op can be
    /// replayed) and the value is skipped: the op runs again. A `show` whose run faulted writes the built-in form and a
    /// note. A value whose `show` is running writes the built-in form.
    #[inline(never)]
    fn show_into(self, out: &mut String, value: Val<'gc>, sid: usize) -> bool {
        let shapes = self.fixture::<crate::fixtures::DisplayShapes>();
        let Some(body) = shapes.show_bodies.borrow().get(sid).copied().flatten() else {
            return false;
        };
        let Val::Instance(i) = value else { return false };
        let key = Gc::as_ptr(i.0) as *const ();
        let calls = self.fixture::<crate::fixtures::ShowCalls>();
        // r[impl mimas.show.self] inside `T`'s own `show`, a `T` is built-in
        if calls.running.borrow().contains(&body) {
            return false;
        }
        let fresh = calls.op.get() == self.state.fixtures.get::<crate::fixtures::Trace>().ops_total.get();
        if fresh {
            match calls.texts.borrow().get(&key) {
                Some(crate::fixtures::ShowText::Text(t)) => {
                    out.push_str(t);
                    return true;
                }
                Some(crate::fixtures::ShowText::Failed(_)) => return false,
                None => {}
            }
        }
        if self.state.show_replay.get() {
            let mut pending = calls.pending.borrow_mut();
            if !pending.iter().any(|(k, _, _)| *k == key) {
                pending.push((key, self.stash(value), body));
            }
            self.state.show_pending.set(true);
            // the op replays once this `show` ran, so nothing written now
            // is kept: skip the value, and do not collect what it holds
            return true;
        }
        false
    }

    /// The note a failed `show` leaves after the built-in form.
    fn show_note(self, value: Val<'gc>) -> Option<String> {
        let Val::Instance(i) = value else { return None };
        let key = Gc::as_ptr(i.0) as *const ();
        let calls = self.fixture::<crate::fixtures::ShowCalls>();
        if calls.op.get() != self.state.fixtures.get::<crate::fixtures::Trace>().ops_total.get() {
            return None;
        }
        match calls.texts.borrow().get(&key) {
            Some(crate::fixtures::ShowText::Failed(note)) => Some(note.clone()),
            _ => None,
        }
    }

    /// [Ctx::display] or [Ctx::to_string] (`quote` picks), following the
    /// value's static `shape`: tuple parentheses and units the runtime value
    /// cannot tell (r[impl mimas.display.one]).
    pub fn render_shaped(self, value: Val<'gc>, shape: &Shape, quote: bool) -> RtResult<String> {
        let mut out = String::new();
        self.render_into(&mut out, value, quote, 0, &mut Walk::default(), shape)?;
        self.shows_ready()?;
        Ok(out)
    }

    /// Render a value as *text* -- a string yields its own contents, unquoted. This is what
    /// f-strings and `print` use; every non-string renders the same as [Ctx::display].
    pub fn to_string(self, value: Val<'gc>) -> RtResult<String> {
        let mut out = String::new();
        self.render_into(&mut out, value, false, 0, &mut Walk::default(), &Shape::Plain)?;
        self.shows_ready()?;
        Ok(out)
    }

    /// [Ctx::to_string_into] without the `show` check: the f-string op asks
    /// once, after its last hole ([Ctx::shows_ready]).
    pub fn render_text_into(self, out: &mut String, value: Val<'gc>) -> RtResult<()> {
        self.render_into(out, value, false, 0, &mut Walk::default(), &Shape::Plain)
    }

    /// [Ctx::to_string], appending into an existing buffer (the f-string builder).
    pub fn to_string_into(self, out: &mut String, value: Val<'gc>) -> RtResult<()> {
        self.render_into(out, value, false, 0, &mut Walk::default(), &Shape::Plain)?;
        self.shows_ready()
    }

    /// A list's length, however it is backed (0 for a non-list).
    fn array_len(self, value: Val<'gc>) -> usize {
        match value {
            Val::Array(a) => a.0.borrow().len(),
            Val::IntArray(a) | Val::FloatArray(a) => a.0.borrow().len(),
            _ => 0,
        }
    }

    /// A list's `i`th item, however it is backed.
    fn array_at(self, value: Val<'gc>, i: usize) -> Val<'gc> {
        match value {
            Val::Array(a) => a.0.borrow()[i],
            Val::IntArray(a) | Val::FloatArray(a) => a.0.borrow().at(i),
            _ => Val::Null,
        }
    }

    /// A quantity's text: the number in the unit the page wrote for `dim`
    /// (`25kW` shows `25 kW`), or in the coherent unit when it wrote none.
    fn quantity_into(self, out: &mut String, value: Val<'gc>, dim: shared::units::Dim) -> bool {
        if !matches!(value, Val::Float(_) | Val::Int(_) | Val::Dec(_)) {
            return false;
        }
        self.fixture::<crate::fixtures::DisplayShapes>().with_unit(dim, |name, scale| {
            match value {
                // a `dec` quantity is already held in the page's unit
                Val::Dec(d) => {
                    let _ = write!(out, "{d}");
                }
                // an `int` quantity (`q.to_int()`) shows its number too
                // (r[impl mimas.unit.int-shown])
                Val::Int(i) => shared::units::shown_number_into(out, i as f64 / scale),
                _ => {
                    let x = value.as_float().unwrap_or_default();
                    shared::units::shown_number_into(out, x / scale);
                }
            }
            out.push(' ');
            out.push_str(name);
        });
        true
    }

    /// `path.path` holds the containers being rendered around `value`; `path.seen` holds
    /// every node-shaped instance (one with a field holding another instance) already
    /// printed in this walk. A struct instance met again --
    /// an ancestor (a cycle: a doubly linked list's `prev`) or one printed earlier in a
    /// sibling field (an alias: a list's `tail`) -- prints a short reference named by its
    /// content, `Node 1 ↺` (the struct name and its first scalar field), instead of
    /// descending, failing, or printing the same chain twice. A shared instance with no
    /// instance fields (`Point { 1, 2 }`, a unit variant) is not a node and prints in full
    /// each time it is met outside a cycle. A list or dict met again
    /// prints a reference only on a cycle (`list ↺`); shared lists and dicts that do not
    /// contain themselves still print in full, so `[xs, xs]` reads as two lists.
    fn render_into(
        self,
        out: &mut String,
        value: Val<'gc>,
        quote: bool,
        depth: usize,
        path: &mut Walk,
        shape: &Shape,
    ) -> RtResult<()> {
        let ptr = match value {
            Val::Array(a) => Some(Gc::as_ptr(a.0) as *const ()),
            Val::IntArray(a) | Val::FloatArray(a) => Some(Gc::as_ptr(a.0) as *const ()),
            Val::Dict(d) => Some(Gc::as_ptr(d.0) as *const ()),
            Val::Instance(i) => Some(Gc::as_ptr(i.0) as *const ()),
            Val::Raised(s) => Some(Gc::as_ptr(s) as *const ()),
            _ => None,
        };
        let Some(ptr) = ptr else {
            return self.render_node(out, value, quote, depth, path, shape);
        };
        // r[impl mimas.display.cycle]
        // r[impl mimas.display.cycle.full]
        // only node-shaped instances (a field that holds another instance) count as
        // aliases: a shared `Point { 1, 2 }` or a unit variant still prints in full
        let is_node = match value {
            Val::Instance(i) => i.0.borrow().fields.iter().any(|v| matches!(v, Val::Instance(_))),
            _ => false,
        };
        let again = path.path.contains(&ptr) || (is_node && path.seen.contains(&ptr));
        if again {
            match value {
                Val::Instance(i) => {
                    let inst = i.0.borrow();
                    let name = self
                        .state
                        .struct_names
                        .borrow()
                        .get(inst.struct_id as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("@{}", inst.struct_id));
                    out.push_str(&name);
                    // named by content: the first scalar field (`Node 1`), so the reader
                    // can find the node it means without counting positions
                    let label = inst.fields.iter().find(|v| {
                        matches!(v, Val::Int(_) | Val::Float(_) | Val::Dec(_) | Val::Str(_) | Val::Bool(_))
                    });
                    if let Some(&v) = label {
                        out.push(' ');
                        self.render_node(out, v, true, depth + 1, path, &Shape::Plain)?;
                    }
                    out.push_str(" ↺");
                }
                Val::Dict(_) => out.push_str("dict ↺"),
                Val::Raised(_) => out.push_str("raised ↺"),
                _ => out.push_str("list ↺"),
            }
            return Ok(());
        }
        path.path.push(ptr);
        if is_node {
            path.seen.push(ptr);
        }
        let r = self.render_node(out, value, quote, depth, path, shape);
        path.path.pop();
        r
    }

    /// The one renderer behind `print`, f-strings, echo and check messages
    /// (r[impl mimas.display.one]). `shape` adds what the static type knows:
    /// a tuple's parentheses and a quantity's unit. A struct shows its field
    /// names and each field's declared unit.
    ///
    /// A `Show` pact (lang lane, L4) hooks in here, at the top of the
    /// `Val::Instance` arm: a type that implements it renders as its `show`.
    fn render_node(
        self,
        out: &mut String,
        value: Val<'gc>,
        quote: bool,
        depth: usize,
        path: &mut Walk,
        shape: &Shape,
    ) -> RtResult<()> {
        if depth > MAX_DISPLAY_DEPTH {
            return Err(RtErr::DisplayTooDeep);
        }
        if let Shape::Q(dim) = shape
            && self.quantity_into(out, value, *dim)
        {
            return Ok(());
        }
        // a list of quantities shows its unit once, after the list
        if let Shape::List(inner) = shape
            && let Shape::Q(dim) = **inner
            && matches!(value, Val::Array(_) | Val::IntArray(_) | Val::FloatArray(_))
        {
            let shapes = self.fixture::<crate::fixtures::DisplayShapes>();
            let scale = shapes.with_unit(dim, |_, scale| scale);
            let n = self.array_len(value);
            out.push('[');
            for i in 0..n {
                if i > 0 {
                    out.push_str(", ");
                }
                let v = self.array_at(value, i);
                let x = match v {
                    Val::Int(i) => Some(i as f64),
                    _ => v.as_float(),
                };
                match x {
                    Some(x) if !matches!(v, Val::Dec(_)) => {
                        shared::units::shown_number_into(out, x / scale)
                    }
                    _ => self.render_into(out, v, true, depth + 1, path, &Shape::Plain)?,
                }
            }
            out.push_str("] ");
            shapes.with_unit(dim, |name, _| out.push_str(name));
            return Ok(());
        }
        // a tuple is a list at runtime: its static shape gives the parentheses
        if let Shape::Tuple(items) = shape
            && matches!(value, Val::Array(_) | Val::IntArray(_) | Val::FloatArray(_))
        {
            let n = self.array_len(value);
            out.push('(');
            for i in 0..n {
                if i > 0 {
                    out.push_str(", ");
                }
                let v = self.array_at(value, i);
                let s = items.get(i).unwrap_or(&Shape::Plain);
                self.render_into(out, v, true, depth + 1, path, s)?;
            }
            if n == 1 {
                out.push(',');
            }
            out.push(')');
            return Ok(());
        }
        let item_shape = match shape {
            Shape::List(s) | Shape::Dict(s) => &**s,
            _ => &Shape::Plain,
        };
        match value {
            Val::Null => out.push_str("null"),
            Val::Bool(b) => out.push_str(if b { "true" } else { "false" }),
            // -- alloc
            Val::Int(i) => out.push_str(itoa::Buffer::new().format(i)),
            Val::Float(f) => shared::write_float(out, f),
            Val::Dec(d) => {
                let _ = write!(out, "{d}");
            }
            Val::Str(s) => {
                if quote {
                    let _ = write!(out, "{:?}", s.as_str());
                } else {
                    out.push_str(s.as_str());
                }
            }
            // nested strings always quote -- `quote` only governs the top-level value, so
            // `print("x")` is bare while `print(["x"])` renders ["x"]
            Val::Array(a) => {
                out.push('[');
                for (i, &v) in a.0.borrow().iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.render_into(out, v, true, depth + 1, path, item_shape)?;
                }
                out.push(']');
            }
            // identical rendering to `Val::Array` — `[1, 2]` is `[1, 2]`
            // however it's backed. Elements box out one at a time so the
            // borrow drops before the recursive render.
            Val::IntArray(a) | Val::FloatArray(a) => {
                let n = a.0.borrow().len();
                out.push('[');
                for i in 0..n {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let v = a.0.borrow().at(i);
                    self.render_into(out, v, true, depth + 1, path, item_shape)?;
                }
                out.push(']');
            }
            Val::Dict(d) => {
                // a dict of quantities shows its unit once, after the braces
                let unit = match item_shape {
                    Shape::Q(dim) => Some(self.fixture::<crate::fixtures::DisplayShapes>().unit(*dim)),
                    _ => None,
                };
                out.push_str("{");
                for (i, (k, v)) in d.0.borrow().iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    // a string key prints bare, as a record field; any other key prints
                    // as the value it is
                    match k {
                        Val::Str(s) => {
                            let _ = write!(out, "{}: ", s.as_str());
                        }
                        k => {
                            self.render_into(out, *k, true, depth + 1, path, &Shape::Plain)?;
                            out.push_str(": ");
                        }
                    }
                    match (&unit, v.as_float()) {
                        (Some(shown), Some(x)) if !matches!(v, Val::Dec(_)) => {
                            out.push_str(&shared::units::shown_number(x / shown.scale))
                        }
                        _ if unit.is_some() => {
                            self.render_into(out, *v, true, depth + 1, path, &Shape::Plain)?
                        }
                        _ => self.render_into(out, *v, true, depth + 1, path, item_shape)?,
                    }
                }
                out.push('}');
                if let Some(shown) = unit {
                    out.push(' ');
                    out.push_str(&shown.name);
                }
            }
            Val::Fn(b) => {
                let _ = write!(out, "<fn @{}>", b.index());
            }
            Val::Closure(c) => {
                let _ = write!(
                    out,
                    "<closure @{} captures={}>",
                    c.0.function.index(),
                    c.0.captures.len()
                );
            }
            Val::Raised(s) => {
                out.push_str("<raised ");
                self.render_into(out, *s, true, depth + 1, path, &Shape::Plain)?;
                out.push('>');
            }
            Val::Instance(i) => {
                // r[impl mimas.show.runtime] a type that implements `Show`
                // writes itself through its `show`
                if self.state.any_show.get()
                    && self.show_into(out, value, i.0.borrow().struct_id as usize)
                {
                    return Ok(());
                }
                let inst = i.0.borrow();
                let sid = inst.struct_id as usize;
                let names = self.state.struct_names.borrow();
                let atom = match names.get(sid) {
                    Some(name) => {
                        out.push_str(name);
                        // an atom's payload reads as it is written, `#ParseError(line: 3)`
                        name.starts_with('#')
                    }
                    None => {
                        let _ = write!(out, "@{}", inst.struct_id);
                        false
                    }
                };
                drop(names);
                // a unit variant (`Sky::Moon`) prints as its name alone; a
                // struct names each field and shows its declared unit
                // (`M { w: 3 px, n: "a" }`), a positional one (`Square(4)`)
                // lists its values in parentheses. The borrows stay held
                // across the walk, as the instance's own does: nothing a
                // render does writes them.
                if !inst.fields.is_empty() {
                    let field_names = self.state.field_names.borrow();
                    let labels: &[String] = field_names.get(sid).map_or(&[], Vec::as_slice);
                    let fields_table = self.fixture::<crate::fixtures::DisplayShapes>().fields.borrow();
                    let shapes: &[Shape] = fields_table.get(sid).map_or(&[], Vec::as_slice);
                    let positional = labels.is_empty()
                        || labels.iter().all(|l| l.bytes().all(|c| c.is_ascii_digit()));
                    let (open, close) = if positional || atom { ("(", ")") } else { (" { ", " }") };
                    out.push_str(open);
                    for (i, &v) in inst.fields.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        if !positional && let Some(label) = labels.get(i) {
                            out.push_str(label);
                            out.push_str(": ");
                        }
                        let s = shapes.get(i).unwrap_or(&Shape::Plain);
                        self.render_into(out, v, true, depth + 1, path, s)?;
                    }
                    out.push_str(close);
                }
                drop(inst);
                // a `show` that faulted: the built-in form, then why
                if self.state.any_show.get()
                    && let Some(note) = self.show_note(value)
                {
                    out.push_str(" (show failed: ");
                    out.push_str(&note);
                    out.push(')');
                }
            }
            #[cfg(feature = "dataframe")]
            Val::DataFrame(d) => {
                // polars' own Display (the "fmt" feature) -- a real formatted table, not just a
                // shape summary.
                // a Z-set whose weights are all 1 prints without `w`
                let _ = write!(out, "{}", crate::val::shown_frame(&d.0.borrow().0));
            }
            #[cfg(feature = "dataframe")]
            Val::PlExpr(e) => {
                let _ = write!(out, "{}", e.0.0);
            }
            // LazyGroupBy has no Display of its own (it's an uncollected plan, nothing to show
            // rows/columns for) -- same placeholder style as Closure's `<closure @..>`.
            #[cfg(feature = "dataframe")]
            Val::GroupBy(_) => out.push_str("<group_by>"),
            // raw pixel bytes have no meaningful text rendering -- dims are the useful summary.
            #[cfg(feature = "darkly")]
            Val::DarklyImage(img) => {
                let _ = write!(out, "<image {}x{}>", img.0.0.width, img.0.0.height);
            }
            #[cfg(feature = "tensor")]
            Val::Tensor(t) => {
                let _ = write!(out, "{}", crate::tensor::render(&t.0.0));
            }
        }
        Ok(())
    }
}

/// A [`Val`] the host is holding onto, from [`Ctx::stash`]. Dropping it lets the value be
/// collected again.
///
/// A handle belongs to the program it was taken against. Loading another one invalidates every
/// outstanding handle, and fetching or calling a stale one fails rather than reaching into
/// whatever the new program happens to have at that body.
pub struct Stashed(DynamicRoot<Rootable![Val<'_>]>);

/// The loaded program's per-body signatures, shared between the Vm and any native that gets handed
/// a function value. A per-Vm fixture, so a native can reach it from a [`Ctx`] alone.
///
/// The Vm holds the same `Rc`, and both are written only by `Vm::install_signatures` -- reading a
/// table here that the Vm has moved on from would mean natives reporting the wrong signatures.
#[derive(Default)]
pub(crate) struct Signatures(pub RefCell<Rc<IdVec<BodyId, Option<Function>>>>);
