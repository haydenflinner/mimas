//! One draw point for every random native — `float::random`,
//! `int::random`, `bool::random`, `arr.shuffle`, `arr.choose` all take a
//! `[0,1)` unit through here, so a host that seeds the Vm's `Rng` fixture
//! (or scripts specific draws) controls the whole surface at once.
//! Unseeded Vms draw fresh entropy per call, exactly as before.

use rand::RngExt;
use vm::{Ctx, fixtures::Rng};

/// `[0,1)` — scripted value, then seeded xorshift64* step (matching the
/// headless `mrt` stream draw-for-draw), then entropy.
pub(crate) fn unit<'gc>(ctx: Ctx<'gc>) -> f64 {
    ctx.fixture::<Rng>()
        .next_unit()
        .unwrap_or_else(|| rand::rng().random())
}
