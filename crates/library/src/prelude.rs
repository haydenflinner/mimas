use macros::native;
use vm::conversion::Raisable;
use vm::{
    Ctx, RtErr, Val,
    anon::{self},
    api::Api,
    conversion::NeverReturn,
    fixtures::{DebugInfo, Echoes, Out, PrintDims, Prints, Witnesses},
};

/// The ` {unit}` a `print`/`dbg`/`warn` appends when the dims pass proved the value's
/// dimension (`print(25kW)` writes `25000 W`). Empty for plain numbers, for
/// sites the checker couldn't follow, and when there's no call-site loc.
fn unit_suffix(ctx: Ctx<'_>, loc: Option<shared::Location>) -> String {
    loc.and_then(|loc| ctx.fixture::<PrintDims>().at(loc))
        .filter(|d| !d.is_none())
        .map(|d| format!(" {}", d.name()))
        .unwrap_or_default()
}

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add(print);
    api.add(warn);
    api.add(panic);
    api.add(todo);
    api.add(dbg);
    api.add(echo);
    api.add(__in_unit);
    api.add(__contract_fail);
    api.add(__check_operand);
    api.add(sum);
    api.add(avg);
    api.add(count);
    api.add(sum_by);
    api.add(avg_by);
    api.add(count_by);
}

#[native]
#[effects(io)]
/// Writes `value` to standard output, followed by a newline. A string prints as its plain text,
/// and any other value prints the same way it would inside an f-string.
///
/// A quantity prints with its unit — `print(25kW)` writes `25000 W`.
///
/// ```mimas
/// print("hello");           // hello
/// print([1, 2, 3]);         // [1, 2, 3]
/// print(f"{1 + 2} apples"); // 3 apples
/// ```
fn print<'gc>(ctx: Ctx<'gc>, msg: Val<'gc>) -> Result<(), RtErr> {
    let loc = ctx.fixture::<DebugInfo>().top_loc();
    // a `dec` quantity is held in the page's unit, so it prints with that
    // unit's name, not the coherent one (r[impl mimas.dec.units])
    let suffix = match (msg, loc.and_then(|l| ctx.fixture::<PrintDims>().shown(l))) {
        (Val::Dec(_), Some(shown)) => format!(" {}", shown.name),
        _ => unit_suffix(ctx, loc),
    };
    let line = ctx.to_string(msg)?;
    ctx.fixture::<Out>().write(&format!("{line}{suffix}"));
    // `display` (quoted strings, structured values) is the richer read
    // a host tooltip wants; the plain line above is all `print` is
    // without one.
    if let Some(loc) = loc {
        let text = ctx.display(msg).unwrap_or(line);
        ctx.fixture::<Prints>().push(loc, format!("{text}{suffix}"));
    }
    Ok(())
}

/// A non-fatal diagnostic: the host marks this call site with the message,
/// `warning: …` reaches the out sink, and the run continues — the reporting
/// side of "skip the bit that's in err". Repeats from the same site are
/// suppressed, so `warn` inside a loop doesn't flood.
///
/// A quantity warns with its unit — `warn(25kW)` writes `warning: 25000 W`.
#[native]
#[effects(io)]
fn warn<'gc>(ctx: Ctx<'gc>, msg: Val<'gc>) -> Result<(), RtErr> {
    let loc = ctx.fixture::<DebugInfo>().top_loc();
    let suffix = unit_suffix(ctx, loc);
    let line = ctx.to_string(msg)?;
    ctx.warn(&format!("{line}{suffix}"));
    Ok(())
}

#[native]
/// Stops the script with a runtime error. The error reads `panic: ` followed by `msg`, or
/// `explicit panic` when there's no message.
///
/// `panic` never returns, and its [never type](../reference/special-types.md) lets it stand in for
/// a value of any type:
///
/// ```mimas
/// let config = { port : 8080 };
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
/// fn load_save(path: str) -> {int} {
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

/// Compiler-emitted contract check (`[T; n]` dims, `where` predicates). Its message is the
/// diagnostic title as authored at emit -- no `panic:`-style prefix -- carried on
/// `RtErr::Contract`. Not for user code; the `__` marks it internal.
#[native]
#[effects()]
/// The value of a bare expression at a file's top level: the compiler hands
/// every top-level expression statement's value here, and while a host has
/// turned `Echoes` on this records the value's display text and structure at
/// the statement's line — what a notebook cell shows under itself. Off (the
/// default) it does nothing, so a plain run formats nothing. Unit is skipped,
/// so a call made for its effect stays silent, and nothing reaches the out
/// sink.
/// r[impl mimas.echo]
fn echo<'gc>(ctx: Ctx<'gc>, val: anon::T<'gc>) -> Result<(), RtErr> {
    let echoes = ctx.fixture::<Echoes>();
    if !echoes.on.get() || matches!(val.0, Val::Null) {
        return Ok(());
    }
    if let Some(loc) = ctx.fixture::<DebugInfo>().top_loc() {
        // r[impl mimas.echo.unit]
        let text = match ctx.fixture::<PrintDims>().shown(loc) {
            Some(shown) => shown_text(ctx, val.0, &shown)?,
            None => format!("{}{}", ctx.display(val.0)?, unit_suffix(ctx, Some(loc))),
        };
        echoes.push(loc, text, ctx.inspect(val.0));
    }
    Ok(())
}

/// An echo of a quantity in the unit the program wrote for it: `12 lb`, a list
/// as `[12, 8, 3] lb`, a grouped result (`sum_by`) as `{grain : 15, veg : 3} lb`.
/// Anything else falls back to the plain display with the unit appended.
fn shown_text<'gc>(ctx: Ctx<'gc>, v: Val<'gc>, shown: &shared::units::Shown) -> Result<String, RtErr> {
    use shared::units::shown_number;
    let num = |x: f64| shown_number(x / shown.scale);
    let unit = &shown.name;
    if !shown.list {
        if let Some(x) = v.as_float() {
            return Ok(format!("{} {unit}", num(x)));
        }
        // a `dec` quantity is already in the page's unit (r[impl mimas.dec.units])
        if let Val::Dec(d) = v {
            return Ok(format!("{d} {unit}"));
        }
    } else if let Some(d) = v.as_dict() {
        let pairs: Vec<(String, Val<'gc>)> =
            d.0.borrow().iter().map(|(k, x)| (k.as_str().to_string(), *x)).collect();
        let mut parts = Vec::with_capacity(pairs.len());
        for (k, x) in pairs {
            match x.as_float() {
                Some(f) => parts.push(format!("{k}: {}", num(f))),
                None => parts.push(format!("{k}: {}", ctx.display(x)?)),
            }
        }
        return Ok(format!("{{{}}} {unit}", parts.join(", ")));
    } else if let Ok(xs) = <Vec<f64> as vm::conversion::MimasType>::from_value(ctx, v) {
        let parts: Vec<String> = xs.into_iter().map(num).collect();
        return Ok(format!("[{}] {unit}", parts.join(", ")));
    }
    Ok(format!("{} {unit}", ctx.display(v)?))
}

/// The elements of a list as numbers, for the aggregates below; `what` names
/// the caller in the error a non-number raises.
fn numbers<'gc>(xs: &[anon::T<'gc>], what: &str) -> Result<(Vec<f64>, bool), RtErr> {
    let mut out = Vec::with_capacity(xs.len());
    let mut ints = true;
    for x in xs {
        match x.0 {
            Val::Int(i) => out.push(i as f64),
            Val::Float(f) => {
                ints = false;
                out.push(f)
            }
            _ => {
                return Err(RtErr::InvalidArgument(format!(
                    "`{what}` works on numbers, and this list holds something else"
                )));
            }
        }
    }
    Ok((out, ints))
}

#[native]
#[effects()]
/// Adds up a list of numbers. A list of quantities sums in their unit, so
/// `sum([12lb, 8lb])` is `20lb`; an empty list sums to `0`.
///
/// ```mimas
/// let qty = [12lb, 8lb, 3lb];
/// let total = sum(qty); // 23 lb
/// ```
/// r[impl mimas.agg.free]
fn sum<'gc>(xs: Vec<anon::T<'gc>>) -> Result<anon::T<'gc>, RtErr> {
    Ok(anon::Anon(sum_vals(&xs, "sum")?))
}

fn sum_vals<'gc>(xs: &[anon::T<'gc>], what: &str) -> Result<Val<'gc>, RtErr> {
    let (ns, ints) = numbers(xs, what)?;
    Ok(if ints && !ns.is_empty() {
        Val::Int(ns.iter().map(|n| *n as i64).sum())
    } else {
        Val::Float(crate::fsum(ns.iter().copied()))
    })
}

#[native]
#[effects()]
/// The mean of a list of numbers, in the list's unit. An empty list has no
/// mean and raises.
///
/// ```mimas
/// let price = [1.10usd, 1.40usd, 0.90usd];
/// let typical = avg(price)!; // 1.1333 usd
/// ```
/// r[impl mimas.agg.free]
fn avg<'gc>(xs: Vec<anon::T<'gc>>) -> Result<Raisable<f64>, RtErr> {
    let (ns, _) = numbers(&xs, "avg")?;
    if ns.is_empty() {
        return Ok(Raisable::Raised("avg of no values".into()));
    }
    Ok(Raisable::Ok(crate::fsum(ns.iter().copied()) / ns.len() as f64))
}

#[native]
#[effects()]
/// How many: the number of `true`s in a list of `bool`s, otherwise the
/// list's length.
///
/// ```mimas
/// let low = [true, false, true];
/// let n = count(low);        // 2
/// let rows = count(["a", "b"]); // 2
/// ```
/// r[impl mimas.agg.free]
fn count<'gc>(xs: Vec<anon::T<'gc>>) -> i64 {
    count_vals(&xs)
}

fn count_vals<'gc>(xs: &[anon::T<'gc>]) -> i64 {
    if !xs.is_empty() && xs.iter().all(|x| matches!(x.0, Val::Bool(_))) {
        xs.iter().filter(|x| matches!(x.0, Val::Bool(true))).count() as i64
    } else {
        xs.len() as i64
    }
}

/// `vals` bucketed by the text of the matching `keys` entry, in the order each
/// key first appears.
fn groups<'gc>(
    ctx: Ctx<'gc>,
    vals: &[anon::T<'gc>],
    keys: &[anon::U<'gc>],
    what: &str,
) -> Result<Vec<(String, Vec<anon::T<'gc>>)>, RtErr> {
    if vals.len() != keys.len() {
        return Err(RtErr::InvalidArgument(format!(
            "`{what}` needs a key per value: {} values, {} keys",
            vals.len(),
            keys.len()
        )));
    }
    let mut out: Vec<(String, Vec<anon::T<'gc>>)> = Vec::new();
    for (v, k) in vals.iter().zip(keys) {
        let k = ctx.to_string(k.0)?;
        match out.iter_mut().find(|(g, _)| *g == k) {
            Some((_, xs)) => xs.push(*v),
            None => out.push((k, vec![*v])),
        }
    }
    Ok(out)
}

fn group_map<'gc>(ctx: Ctx<'gc>, rows: Vec<(String, Val<'gc>)>) -> vm::Dict<'gc> {
    let mut map = vm::DictMap::new();
    for (k, v) in rows {
        map.insert(ctx.intern(&k), v);
    }
    ctx.new_dict(map)
}

fn group_dict<'gc>(ctx: Ctx<'gc>, rows: Vec<(String, Val<'gc>)>) -> anon::DictOf<'gc, 0> {
    anon::DictOf(group_map(ctx, rows))
}

/// A grouped result whose values are all of one primitive type (`count_by` is
/// `{int}`, `avg_by` is `{float}`), kept in key order — a `HashMap` would lose it.
struct Grouped<'gc, const FLOAT: bool>(vm::Dict<'gc>);

impl<'gc, const FLOAT: bool> vm::conversion::MimasType<'gc> for Grouped<'gc, FLOAT> {
    fn mimas_ty(_: &vm::Registry) -> Option<shared::Ty> {
        Some(shared::Ty::Dict(Box::new(if FLOAT { shared::Ty::Float } else { shared::Ty::Int })))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, vm::conversion::TypeError> {
        v.as_dict().map(Grouped).ok_or_else(|| vm::conversion::TypeError {
            expected: "dict".into(),
            got: format!("{v:?}"),
        })
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Dict(self.0)
    }
}

#[native]
#[effects()]
/// `sum` per group: `vals` added up for each distinct entry of `keys` (its
/// text), in the order the keys first appear. Quantities keep their unit.
///
/// ```mimas
/// let qty = [12lb, 8lb, 3lb];
/// let kind = ["grain", "bean", "grain"];
/// let by_kind = sum_by(qty, kind); // {grain : 15, bean : 8} lb
/// ```
/// r[impl mimas.agg.by]
fn sum_by<'gc>(
    ctx: Ctx<'gc>,
    vals: Vec<anon::T<'gc>>,
    keys: Vec<anon::U<'gc>>,
) -> Result<anon::DictOf<'gc, 0>, RtErr> {
    let mut rows = Vec::new();
    for (k, xs) in groups(ctx, &vals, &keys, "sum_by")? {
        rows.push((k, sum_vals(&xs, "sum_by")?));
    }
    Ok(group_dict(ctx, rows))
}

#[native]
#[effects()]
/// `avg` per group, like [`sum_by`](#sum_by). Every group has at least one
/// value, so no group raises.
/// r[impl mimas.agg.by]
fn avg_by<'gc>(
    ctx: Ctx<'gc>,
    vals: Vec<anon::T<'gc>>,
    keys: Vec<anon::U<'gc>>,
) -> Result<Grouped<'gc, true>, RtErr> {
    let mut rows = Vec::new();
    for (k, xs) in groups(ctx, &vals, &keys, "avg_by")? {
        let (ns, _) = numbers(&xs, "avg_by")?;
        rows.push((k, Val::Float(crate::fsum(ns.iter().copied()) / ns.len() as f64)));
    }
    Ok(Grouped(group_map(ctx, rows)))
}

#[native]
#[effects()]
/// `count` per group, like [`sum_by`](#sum_by): the `true`s among each
/// group's `bool`s, otherwise how many values the group has.
/// r[impl mimas.agg.by]
fn count_by<'gc>(
    ctx: Ctx<'gc>,
    vals: Vec<anon::T<'gc>>,
    keys: Vec<anon::U<'gc>>,
) -> Result<Grouped<'gc, false>, RtErr> {
    let rows = groups(ctx, &vals, &keys, "count_by")?
        .into_iter()
        .map(|(k, xs)| (k, Val::Int(count_vals(&xs))))
        .collect();
    Ok(Grouped(group_map(ctx, rows)))
}

#[native]
fn __contract_fail<'gc>(
    ctx: Ctx<'gc>,
    msg: Option<anon::T<'gc>>,
    got: Option<anon::T<'gc>>,
) -> Result<NeverReturn, RtErr> {
    let mut text = match msg {
        Some(msg) => ctx.to_string(msg.0)?,
        None => "contract failed".to_string(),
    };
    // the value that broke the check, when the check has one -- the length that
    // arrived against the one that was pinned, say
    if let Some(got) = got {
        text = format!("{text}, got {}", ctx.to_string(got.0)?);
    }
    Err(RtErr::Contract(text))
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
    let loc = ctx.fixture::<DebugInfo>().top_loc();
    let suffix = unit_suffix(ctx, loc);
    let text = format!("dbg value: {}{}", ctx.display(val.0)?, suffix);
    ctx.fixture::<Out>().write(&text);
    if let Some(loc) = loc {
        ctx.fixture::<Prints>().push(loc, text);
    }
    Ok(val)
}

#[native]
#[effects()]
/// One operand of a failing `check`/`assert!`, recorded for the host:
/// the assert desugar emits this call inside the failure branch — once
/// per operand, just before `panic` unwinds — so a check runner shows
/// the values the message names, an inspectable example object rather
/// than text alone. One call per operand, not one `[…]` argument list:
/// each call keeps the operand's own type where a list would force them
/// to share one. The value lands as its `Inspect` tree plus a live
/// handle a host can draw through (see `Witnesses`). Not for user code;
/// the `__` marks it internal.
fn __check_operand<'gc>(ctx: Ctx<'gc>, val: anon::T<'gc>) -> Result<(), RtErr> {
    let loc = ctx.fixture::<DebugInfo>().top_loc();
    ctx.fixture::<Witnesses>()
        .push(loc, ctx.inspect(val.0), ctx.stash(val.0));
    Ok(())
}

#[native]
#[effects()]
/// A quantity as a failing check names it: `value` (in base units) shown
/// in a unit `scale` base units large and called `name`, written as a
/// literal (`20px`), so both sides of `20px == 30px` read the same way
/// as the code. Not for user code; the compiler calls it
/// (r[impl check.report.units]).
fn __in_unit<'gc>(ctx: Ctx<'gc>, val: anon::T<'gc>, scale: f64, name: &str) -> Result<String, RtErr> {
    if let Some(x) = val.0.as_float() {
        return Ok(format!("{}{name}", shared::units::shown_number(x / scale)));
    }
    if let Val::Dec(d) = val.0 {
        return Ok(format!("{d}{name}"));
    }
    let shown = shared::units::Shown { name: name.to_string(), scale, list: false };
    shown_text(ctx, val.0, &shown)
}
