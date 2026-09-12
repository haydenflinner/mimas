use macros::native;
use vm::{
    Ctx, RtErr, Val,
    anon::{self},
    api::Api,
    conversion::NeverReturn,
};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add(print);
    api.add(panic);
    api.add(todo);
    api.add(dbg);
}

#[native]
fn print<'gc>(ctx: Ctx<'gc>, msg: Val<'gc>) -> Result<(), RtErr> {
    println!("{}", ctx.to_string(msg)?);
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
    println!("dbg value: {}", ctx.display(val.0)?);
    Ok(val)
}
