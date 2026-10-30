//! The per-Vm memory budget and the call-depth limit.
//!
//! A host that runs untrusted programs next to its own data (a browser tab, say) needs a program
//! that grows without end to stop with an error, not to take the whole process down. The op budget
//! does not do this: `s = s + s` thirty times is few ops and a gigabyte of memory.
//!
//! # How the budget counts
//!
//! - **Charge.** Every heap allocation the Vm makes (strings, arrays, dicts, instances, closures)
//!   and every growth of an array or dict adds its bytes to `since` ([`Ctx::charge`]).
//!   `live + since` is an upper bound on what the heap holds: a charge never subtracts, because
//!   the Vm cannot see a value die between collections.
//! - **Settle.** At an entry boundary (between dispatch batches, before a host call) the Vm can
//!   collect. When the bound passes half the limit, or passes the limit, [`Vm::mem_settle`] runs a
//!   full collection and measures what is really live: each container's `Collect::trace` adds its
//!   buffer bytes to a thread-local tally while gc-arena marks ([`note_traced`]), live interned
//!   strings add their lengths, and each live `Gc` adds [`BOX_BYTES`]. `live` takes the measured
//!   value and `since` goes back to zero.
//! - **Fail.** When a charge puts `live + since` over the limit, the dispatch loop stops at the
//!   next op. A batched entry (`run`, `run_frame`) settles first and fails only if the measured
//!   heap is still over. An entry that runs inside one arena mutation (a host call, a check) can't
//!   collect before it returns, so everything it allocated is really still in memory: it fails at
//!   once with [`RtErr::OutOfMemory`].
//!
//! One allocation can't be bigger than [`max_object_bytes`] ([`Ctx::check_alloc`]): a size that
//! comes from a script integer (`"x".repeat(n)`) fails before the allocator sees it.
//!
//! [`Ctx::charge`]: crate::Ctx::charge
//! [`Ctx::check_alloc`]: crate::Ctx::check_alloc
//! [`Vm::mem_settle`]: crate::Vm::mem_settle
//! [`RtErr::OutOfMemory`]: crate::RtErr::OutOfMemory

use std::cell::Cell;

/// The default budget: no limit. A host that runs untrusted code sets one with
/// [`Vm::set_mem_budget`](crate::Vm::set_mem_budget).
pub const UNLIMITED: usize = usize::MAX;

/// Bytes one `Gc` allocation costs before its buffers: gc-arena's header plus a small payload.
/// An estimate, used for charges and for the measured live size.
pub const BOX_BYTES: usize = 48;

/// Calls in progress (Vm frames) before a call fails with [`RtErr::RecursionTooDeep`].
///
/// [`RtErr::RecursionTooDeep`]: crate::RtErr::RecursionTooDeep
pub const DEFAULT_MAX_FRAMES: usize = 10_000;

/// Host-to-script calls nested on the Rust stack (a native such as `map` calls a closure, the
/// closure calls `map` again, ...). Each level costs Rust stack, which a wasm host can't grow, so
/// this limit is much lower than [`DEFAULT_MAX_FRAMES`].
pub const MAX_NESTED_CALLS: u32 = 200;

/// Bytes a single string or array may take, for a Vm whose budget is `limit`.
pub fn max_object_bytes(limit: usize) -> usize {
    if limit == UNLIMITED {
        // no budget: only stop sizes that can't be allocated at all
        isize::MAX as usize / 2
    } else {
        (limit / 4).max(1 << 20)
    }
}

/// `bytes` as whole MiB, rounded up, for error text.
pub fn mib(bytes: usize) -> usize {
    bytes.div_ceil(1 << 20)
}

/// `live + since` passed the limit: settle at the next boundary, or fail if the entry can't.
pub(crate) const SIG_OVER: u8 = 1;
/// A settle measured more than the limit: fail at the next op.
pub(crate) const SIG_FATAL: u8 = 2;
/// A native spent ops outside the dispatch loop (a nested Vm it ran): take `ops_debt` off the
/// thread's op budget at the next op. Rides the same per-op check as the memory bits.
pub(crate) const SIG_OPS: u8 = 4;

/// Budget state, on [`State`](crate::State) so a native's [`Ctx`](crate::Ctx) can charge it.
#[derive(Clone)]
pub struct MemBudget {
    /// Bytes the heap may hold. [`UNLIMITED`] turns the budget off.
    pub(crate) limit: Cell<usize>,
    /// Bytes measured live at the last settle.
    pub(crate) live: Cell<usize>,
    /// Bytes charged since the last settle.
    pub(crate) since: Cell<usize>,
    /// `SIG_*` bits, read by the dispatch loop once per op.
    pub(crate) signal: Cell<u8>,
    /// Host-to-script calls in progress on the Rust stack (see [`MAX_NESTED_CALLS`]).
    pub(crate) nested: Cell<u32>,
    /// Ops natives charged with [`Ctx::charge_ops`](crate::Ctx::charge_ops) that the dispatch
    /// loop has not yet taken off the thread's budget.
    pub(crate) ops_debt: Cell<u64>,
}

impl Default for MemBudget {
    fn default() -> Self {
        Self {
            limit: Cell::new(UNLIMITED),
            live: Cell::new(0),
            since: Cell::new(0),
            signal: Cell::new(0),
            nested: Cell::new(0),
            ops_debt: Cell::new(0),
        }
    }
}

impl MemBudget {
    /// The upper bound on heap bytes: measured live plus charged since.
    pub fn estimate(&self) -> usize {
        self.live.get().saturating_add(self.since.get())
    }

    pub fn limit(&self) -> usize {
        self.limit.get()
    }

    #[inline]
    pub(crate) fn charge(&self, bytes: usize) {
        let since = self.since.get().saturating_add(bytes);
        self.since.set(since);
        if self.live.get().saturating_add(since) > self.limit.get() {
            self.signal.set(self.signal.get() | SIG_OVER);
        }
    }

    /// Whether an entry boundary should settle now: over the limit, past half of it, or (with no
    /// limit) enough new allocation that the interned-string table wants pruning.
    pub(crate) fn wants_settle(&self) -> bool {
        let limit = self.limit.get();
        let since = self.since.get();
        // settle once the unmeasured part passes half the headroom left: an entry that can't
        // collect then still has at least that much to allocate before it fails
        self.signal.get() & SIG_OVER != 0
            || (limit != UNLIMITED
                && since > (limit.saturating_sub(self.live.get()) / 2).max(limit / 32))
            || since > PRUNE_EVERY
    }

    #[inline]
    pub(crate) fn charge_ops(&self, ops: u64) {
        if ops > 0 {
            self.ops_debt.set(self.ops_debt.get().saturating_add(ops));
            self.signal.set(self.signal.get() | SIG_OPS);
        }
    }
}

/// With no limit, settle (and so prune dead interned strings) after this many charged bytes.
const PRUNE_EVERY: usize = 256 << 20;

thread_local! {
    static TRACED: Cell<usize> = const { Cell::new(0) };
}

/// Called from a container's `Collect::trace`: add its buffer bytes to this thread's tally. gc-arena
/// traces each reachable object once per full mark, so the tally after an uninterrupted mark is
/// the live buffer total.
#[inline]
pub(crate) fn note_traced(bytes: usize) {
    TRACED.with(|t| t.set(t.get().wrapping_add(bytes)));
}

pub(crate) fn reset_traced() {
    TRACED.with(|t| t.set(0));
}

pub(crate) fn take_traced() -> usize {
    TRACED.with(|t| t.replace(0))
}
