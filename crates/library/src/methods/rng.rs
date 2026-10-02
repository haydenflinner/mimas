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

use macros::native;
use rand::RngExt;
use vm::{Ctx, api::Api, fixtures::Rng};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    let mut m = api.module("random");
    m.add(seed);
}

/// `[0,1)` — scripted value, then seeded xorshift64* step (matching the
/// headless `mrt` stream draw-for-draw), then entropy.
pub(crate) fn unit<'gc>(ctx: Ctx<'gc>) -> f64 {
    ctx.fixture::<Rng>()
        .next_unit()
        .unwrap_or_else(|| rand::rng().random())
}

#[native]
/// Pins the random stream to `seed` — every `random`/`shuffle`/`choose`
/// after this draws the same sequence a fresh session started on `seed`
/// would. Seed `0` is legal but degenerate (xorshift pins at 0).
///
/// ```mimas
/// random::seed(42);
/// let deal = int::random(52); // the same card every run
/// ```
fn seed<'gc>(ctx: Ctx<'gc>, seed: i64) {
    ctx.fixture::<Rng>().seed(seed as u64);
}
