use macros::native;
use shared::Ty;
use vm::{Ctx, api::Api};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_assoc(Ty::Bool, random);
}

/// Returns `true` or `false`, each with equal chance.
///
/// ```mimas
/// let side = if bool::random() "heads" else "tails";
/// ```
#[native]
#[effects(rng)]
fn random<'gc>(ctx: Ctx<'gc>) -> bool {
    super::rng::unit(ctx) < 0.5
}
