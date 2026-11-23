//! `__stream_each`: the step-by-step stream operations that the
//! `AbelianGroup` pact does not cover (`group.mim`). A stream holds any
//! group, so `∪ ∩ ⋈ ⁺ ⁻` (which only tables with weights have) and a real
//! factor (which only real values take) cannot be written generically in
//! mimas. This native applies them to each step and says which step had no
//! such operation.

use macros::native;
use vm::{Ctx, RtErr, Val, anon, api::Api};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add(__stream_each);
}

/// The type name of a step for an error.
fn kind(v: Val<'_>) -> &'static str {
    match v {
        Val::Int(_) => "an int",
        Val::Float(_) => "a float",
        Val::Instance(_) => "a struct",
        #[cfg(feature = "dataframe")]
        Val::DataFrame(_) => "a table",
        _ => "a value",
    }
}

/// `op` on each step: `"*"` scales `xs[i]` by `k`; `"⁺"`/`"⁻"` take the
/// parts of each table; `"∪"`, `"∩"`, `"⋈"` pair `xs[i]` with `ys[i]` over
/// the steps both have.
/// r[impl mimas.group.stream]
#[native]
fn __stream_each<'gc>(
    ctx: Ctx<'gc>,
    op: &str,
    xs: &[anon::T<'gc>],
    ys: &[anon::T<'gc>],
    k: f64,
) -> Result<Vec<anon::T<'gc>>, RtErr> {
    let mut out = Vec::with_capacity(xs.len());
    match op {
        "*" => {
            for x in xs {
                let v = vm::bin(x.0, ctx, Val::Float(k), vm::BinOp::Mult).map_err(|_| {
                    RtErr::Custom(format!(
                        "`k * s` with a real `k` scales each step, and a step here is {}, \
                         which takes no real factor: scale by an int, or map the steps",
                        kind(x.0)
                    ))
                })?;
                out.push(anon::Anon(v));
            }
        }
        _ => {
            let pair = matches!(op, "∪" | "∩" | "⋈");
            let n = if pair { xs.len().min(ys.len()) } else { xs.len() };
            for i in 0..n {
                out.push(anon::Anon(step(ctx, op, xs[i].0, ys.get(i).map(|y| y.0))?));
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "dataframe")]
fn step<'gc>(ctx: Ctx<'gc>, op: &str, x: Val<'gc>, y: Option<Val<'gc>>) -> Result<Val<'gc>, RtErr> {
    let err = |e: polars::prelude::PolarsError| RtErr::Custom(format!("{op} on a stream: {e}"));
    let frame = |v: Val<'gc>| match v {
        Val::DataFrame(d) => Ok(d.0.borrow().0.clone()),
        other => Err(not_table(op, other)),
    };
    let a = frame(x)?;
    let out = match (op, y) {
        ("⁺", _) => crate::zset::added(&a),
        ("⁻", _) => crate::zset::removed(&a),
        ("∪", Some(y)) => crate::zset::union(&a, &frame(y)?),
        ("∩", Some(y)) => crate::zset::intersect(&a, &frame(y)?),
        ("⋈", Some(y)) => crate::zset::natural_join(&a, &frame(y)?),
        _ => return Err(RtErr::Custom(format!("no stream operation `{op}`"))),
    };
    Ok(Val::DataFrame(ctx.new_dataframe(out.map_err(err)?)))
}

#[cfg(not(feature = "dataframe"))]
fn step<'gc>(_ctx: Ctx<'gc>, op: &str, x: Val<'gc>, _y: Option<Val<'gc>>) -> Result<Val<'gc>, RtErr> {
    Err(not_table(op, x))
}

fn not_table(op: &str, v: Val<'_>) -> RtErr {
    RtErr::Custom(format!(
        "`{op}` on a stream works step by step on tables with weights, and a step here is {}: \
         map the steps with an operation of its own type",
        kind(v)
    ))
}
