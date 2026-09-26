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
    collections::HashMap,
    sync::Arc,
};

use shared::{BodyId, FileId, IdVec, Location};

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

/// Where `print`/`dbg` and friends send their lines. The default sink is
/// the process's stdout; an embedder (a repl, the wasm eval worker) swaps
/// in a capture with [`Out::set`] and drains it after the run. Each call
/// gets one line's text — the trailing newline is the sink's business.
pub struct Out(pub RefCell<Box<dyn FnMut(&str)>>);

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
/// (`print`/`dbg`); `Warn` is a non-fatal diagnostic — something was
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
    /// game loop reports once per session, not once per frame.
    warned: RefCell<std::collections::HashSet<(Location, String)>>,
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
