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
    api.add(warn);
    api.add(panic);
    api.add(todo);
    api.add(dbg);
}

#[native]
#[effects(io)]
/// Writes `value` to standard output, followed by a newline. A string prints as its plain text,
/// and any other value prints the same way it would inside an f-string.
///
/// ```mimas
/// print("hello");           // hello
/// print([1, 2, 3]);         // [1, 2, 3]
/// print(f"{1 + 2} apples"); // 3 apples
/// ```
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

/// A non-fatal diagnostic: the host marks this call site with the message,
/// `warning: …` reaches the out sink, and the run continues — the reporting
/// side of "skip the bit that's in err". Repeats from the same site are
/// suppressed, so `warn` inside a loop doesn't flood.
#[native]
#[effects(io)]
fn warn<'gc>(ctx: Ctx<'gc>, msg: &str) {
    ctx.warn(msg);
}

#[native]
/// Stops the script with a runtime error. The error reads `panic: ` followed by `msg`, or
/// `explicit panic` when there's no message.
///
/// `panic` never returns, and its [never type](../reference/special-types.md) lets it stand in for
/// a value of any type:
///
/// ```mimas
/// let config = ~{ port = 8080 };
/// let port = config["port"] ?? panic("no port configured");
/// ```
///
/// For a failure the caller should be able to recover from, return a
/// [result](../reference/error-handling.md#results) and `raise` instead.
#[effects()]
fn panic<'gc>(ctx: Ctx<'gc>, msg: Option<anon::T<'gc>>) -> Result<NeverReturn, RtErr> {
    let text = match msg {
        Some(msg) => format!("panic: {}", ctx.to_string(msg.0)?),
        None => "explicit panic".to_string(),
    };
    Err(RtErr::Custom(text))
}

#[native]
/// Stops the script with a runtime error that marks unfinished code. The error reads `todo: `
/// followed by `msg`, or just `todo` when there's no message.
///
/// Like [`panic`](#panic), it never returns, and it can stand in for a body you haven't written
/// yet:
///
/// ```mimas
/// fn load_save(path: str) -> ~{int} {
///     todo("read the save format")
/// }
/// ```
#[effects()]
fn todo<'gc>(ctx: Ctx<'gc>, msg: Option<anon::T<'gc>>) -> Result<NeverReturn, RtErr> {
    Err(match msg {
        Some(m) => RtErr::Custom(format!("todo: {}", ctx.to_string(m.0)?)),
        None => RtErr::Custom("todo".into()),
    })
}

#[native]
#[effects(io)]
/// Prints `value` to standard output in its debug form, then returns it. The line starts with
/// `dbg value:`, and strings keep their quotes.
///
/// Since it returns its argument, `dbg` can wrap any expression without changing the result:
///
/// ```mimas
/// let total = dbg(2 + 3) * 10; // prints `dbg value: 5`, and total is 50
/// let name = dbg("ada");       // prints `dbg value: "ada"`
/// ```
fn dbg<'gc>(ctx: Ctx<'gc>, val: anon::T<'gc>) -> Result<anon::T<'gc>, RtErr> {
    let text = format!("dbg value: {}", ctx.display(val.0)?);
    ctx.fixture::<Out>().write(&text);
    if let Some(loc) = ctx.fixture::<DebugInfo>().top_loc() {
        ctx.fixture::<Prints>().push(loc, text);
    }
    Ok(val)
}
