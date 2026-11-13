//! One draw point for every random native — `float::random`,
//! `int::random`, `bool::random`, `arr.shuffle`, `arr.choose` all take a
//! `[0,1)` unit through here, so a host that seeds the Vm's `Rng` fixture
//! (or scripts specific draws) controls the whole surface at once.
//! Unseeded Vms draw fresh entropy per call, exactly as before.
//!
//! `random::seed(n)` is the program-facing half: the run pins its own
//! stream from inside the source. A fuzz harness or a "same deal every
//! time" page reseeds on demand; the draw streams on the VM and on
//! rustgen's `mrt` are the same xorshift64*, so one seed replays
//! identically on both engines.
//!
//! `with rng = <spec> { … }` is the scoped half: the parser desugars the
//! region to `__fx_rng_push(spec); …body…; __fx_rng_pop()`, and the
//! fixture keeps the enclosing state+script on a frame-depth-keyed stack —
//! the pop restores it, and a hard fault that skips the pop is healed by
//! [`Rng::reconcile`] at the unwind boundary (or by the next draw, since
//! every access below reconciles first). Effect scoping like this is the
//! point other effects hook later: a `net` fixture could take a
//! `{url: response}` map for its `spec` the way `rng` takes a seed.

use std::collections::VecDeque;

use macros::native;
use rand::RngExt;
use vm::{
    Ctx, RtErr, Val,
    anon,
    api::Api,
    fixtures::{DebugInfo, Rng},
};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    let mut m = api.module("random");
    m.add(seed);
    api.add(__fx_rng_push);
    api.add(__fx_rng_pop);
}

/// The live frame count, as the `DebugInfo` mirror was just resynced for
/// this call — the depth a `with` scope records.
fn depth(ctx: Ctx<'_>) -> usize {
    ctx.fixture::<DebugInfo>().stack.borrow().len()
}

/// `[0,1)` — scripted value, then seeded xorshift64* step (matching the
/// headless `mrt` stream draw-for-draw), then entropy.
pub(crate) fn unit<'gc>(ctx: Ctx<'gc>) -> f64 {
    let rng = ctx.fixture::<Rng>();
    // a `with` scope an unwind orphaned restores before this draw
    rng.reconcile(depth(ctx));
    rng.next_unit().unwrap_or_else(|| rand::rng().random())
}

#[native]
#[effects(rng)]
/// Pins the random stream to `seed` — every `random`/`shuffle`/`choose`
/// after this draws the same sequence a fresh session started on `seed`
/// would. Seed `0` is legal but degenerate (xorshift pins at 0).
///
/// ```mimas
/// random::seed(42);
/// let deal = int::random(52); // the same card every run
/// ```
fn seed<'gc>(ctx: Ctx<'gc>, seed: i64) {
    let rng = ctx.fixture::<Rng>();
    rng.reconcile(depth(ctx));
    rng.seed(seed as u64);
}

#[native]
#[effects(rng)]
/// `with rng = <spec>`'s install half — the parser desugars the region to
/// `__fx_rng_push(spec); …; __fx_rng_pop()`. `spec` is an `int` seed (a
/// fresh xorshift64* stream, `random::seed`'s scoped twin), a `[float]` of
/// scripted unit draws over the live stream, or `()` — keep the live state
/// but restore it at the region's end (a checkpoint). Anything else
/// faults. `__` marks it internal — source writes `with`, not this.
// r[impl mimas.fx.with]
fn __fx_rng_push<'gc>(ctx: Ctx<'gc>, spec: anon::T<'gc>) -> Result<(), RtErr> {
    let depth = depth(ctx);
    let rng = ctx.fixture::<Rng>();
    rng.reconcile(depth);
    let (state, script) = match spec.0 {
        Val::Int(n) => (Some(n as u64), VecDeque::new()),
        Val::Null => (rng.state(), VecDeque::new()),
        seq if seq.is_seq() => {
            let len = seq.seq_len().unwrap_or(0);
            let mut script = VecDeque::with_capacity(len);
            for i in 0..len {
                let drawn = seq.seq_get(i).and_then(|v| {
                    v.as_float().or_else(|| v.as_int().map(|n| n as f64))
                });
                let Some(unit) = drawn else {
                    return Err(RtErr::InvalidArgument(
                        "a `with rng` script's draws are floats".into(),
                    ));
                };
                script.push_back(unit);
            }
            (rng.state(), script)
        }
        _ => {
            return Err(RtErr::InvalidArgument(
                "`with rng` wants an int seed, a [float] of scripted draws, or ()".into(),
            ));
        }
    };
    rng.push_with(depth, state, script);
    Ok(())
}

#[native]
#[effects(rng)]
/// `with rng`'s restore half — pops the scope the region's push left on
/// the fixture. Runs at the region's tail; an unwind that skips it is
/// `reconcile`d away at the frame-truncation boundary instead.
// r[impl mimas.fx.with]
fn __fx_rng_pop<'gc>(ctx: Ctx<'gc>) {
    let depth = depth(ctx);
    let rng = ctx.fixture::<Rng>();
    rng.reconcile(depth);
    rng.pop_with();
}
