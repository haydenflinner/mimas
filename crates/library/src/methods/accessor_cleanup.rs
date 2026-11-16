use macros::native;
use vm::{Ctx, RtErr, anon, api::Api};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add(__cleanup_register);
    api.add(__cleanup_mark);
    api.add(__cleanup_take);
    api.add(__place_live);
    api.add(__cleanup_promote);
}

// r[impl mimas.sub.end]
#[native]
#[effects()]
fn __cleanup_register<'gc>(ctx: Ctx<'gc>, action: anon::Fn0<'gc, ()>) {
    ctx.defer(action.0);
}

#[native]
#[effects()]
fn __cleanup_mark(ctx: Ctx<'_>) -> i64 {
    ctx.cleanup_mark() as i64
}

#[native]
#[effects()]
fn __cleanup_take<'gc>(ctx: Ctx<'gc>, mark: i64) -> Option<anon::Fn0<'gc, ()>> {
    ctx.take_cleanup(mark.max(0) as usize)
        .map(|v| anon::Fn0(v, std::marker::PhantomData))
}

#[native]
#[effects()]
fn __place_live(lifetime: &[bool]) -> Result<(), RtErr> {
    if lifetime.first() == Some(&true) {
        Ok(())
    } else {
        Err(RtErr::Custom(
            "the borrowed accessor place has left its scope".into(),
        ))
    }
}

#[native]
#[effects()]
fn __cleanup_promote(ctx: Ctx<'_>) {
    ctx.promote_cleanups();
}
