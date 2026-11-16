use macros::native;
use vm::{Ctx, RtErr, Val, anon, api::Api};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add(__generator);
    api.add(__generator_yield);
}

#[native]
fn __generator<'gc>(
    ctx: Ctx<'gc>,
    body: anon::Fn1<'gc, anon::T<'gc>, ()>,
) -> Result<anon::Fn0<'gc, Option<anon::T<'gc>>>, RtErr> {
    let closure = body
        .0
        .as_closure()
        .ok_or_else(|| RtErr::Custom("a generator body must be a closure".into()))?;
    Ok(anon::Fn0(
        Val::Closure(ctx.new_generator(closure)),
        std::marker::PhantomData,
    ))
}

#[native]
fn __generator_yield<'gc>(ctx: Ctx<'gc>, value: anon::T<'gc>) -> Result<(), RtErr> {
    if std::ptr::eq(&*ctx.thread(), &*ctx.state().thread) {
        return Err(RtErr::Custom("yield requires an active generator".into()));
    }
    if matches!(value.0, Val::Null) {
        return Err(RtErr::Custom(
            "a generator cannot yield null; wrap the payload in a struct".into(),
        ));
    }
    ctx.state().generator_value.set(&ctx, Some(value.0));
    ctx.state().paused.set(true);
    Ok(())
}

