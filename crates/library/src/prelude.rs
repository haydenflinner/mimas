use macros::native;
use vm::{
    Ctx, RtErr, Val,
    anon::{self},
    api::Api,
    conversion::NeverReturn,
    fixtures::{DebugInfo, Out, Prints},
};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add(print);
    api.add(panic);
    api.add(todo);
    api.add(dbg);
}

#[native]
fn print<'gc>(ctx: Ctx<'gc>, msg: Val<'gc>) -> Result<(), RtErr> {
    let line = ctx.to_string(msg)?;
    ctx.fixture::<Out>().write(&line);
    // `display` (quoted strings, structured values) is the richer read
    // a host tooltip wants; the plain line above is all `print` is
    // without one.
    if let Some(loc) = ctx.fixture::<DebugInfo>().top_loc() {
        let text = ctx.display(msg).unwrap_or(line);
        ctx.fixture::<Prints>().push(loc, text);
    }
    Ok(())
}

#[native]
fn panic<'gc>(ctx: Ctx<'gc>, msg: Option<anon::T<'gc>>) -> Result<NeverReturn, RtErr> {
    let text = match msg {
        Some(msg) => format!("panic: {}", ctx.to_string(msg.0)?),
        None => "explicit panic".to_string(),
    };
    Err(RtErr::Custom(text))
}

#[native]
fn todo<'gc>(ctx: Ctx<'gc>, msg: Option<anon::T<'gc>>) -> Result<NeverReturn, RtErr> {
    Err(match msg {
        Some(m) => RtErr::Custom(format!("todo: {}", ctx.to_string(m.0)?)),
        None => RtErr::Custom("todo".into()),
    })
}

#[native]
fn dbg<'gc>(ctx: Ctx<'gc>, val: anon::T<'gc>) -> Result<anon::T<'gc>, RtErr> {
    let text = format!("dbg value: {}", ctx.display(val.0)?);
    ctx.fixture::<Out>().write(&text);
    if let Some(loc) = ctx.fixture::<DebugInfo>().top_loc() {
        ctx.fixture::<Prints>().push(loc, text);
    }
    Ok(val)
}
