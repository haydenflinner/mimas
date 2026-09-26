use api::Intrinsic;
use macros::native;
use rand::RngExt;
use shared::Ty;
use vm::{RtErr, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_method(abs);
    api.add_method(min);
    api.add_method(max);
    api.add_method(clamp);
    api.add_method(to_str);
    api.add_method(to);
    let id = api.add_method(to_float);
    api.mark_intrinsic(id, Intrinsic::ToFloat);
    api.add_assoc(Ty::Int, random);
}

#[native]
fn abs<'gc>(n: i64) -> i64 {
    n.abs()
}

#[native]
fn min<'gc>(a: i64, b: i64) -> i64 {
    a.min(b)
}

#[native]
fn max<'gc>(a: i64, b: i64) -> i64 {
    a.max(b)
}

#[native]
fn clamp<'gc>(n: i64, lo: i64, hi: i64) -> Result<i64, RtErr> {
    if lo > hi {
        return Err(RtErr::InvalidArgument("clamp requires lo <= hi".into()));
    }
    Ok(n.clamp(lo, hi))
}

#[native]
fn to_str<'gc>(n: i64) -> String {
    n.to_string()
}

#[native]
fn random<'gc>(len: i64) -> Result<i64, RtErr> {
    // random_range panics on an empty range
    if len <= 0 {
        return Err(RtErr::InvalidArgument("random requires len > 0".into()));
    }
    Ok(rand::rng().random_range(0..len))
}

#[native]
fn to_float<'gc>(v: i64) -> f64 {
    v as f64
}

/// `q.to("g")` on an int -- an int-valued quantity (`5.5kg.to_int()`) reads back in any
/// unit of the same kind, exactly like float's `to`: the compiler checks the unit
/// measures the same thing; at runtime only the name has to be a unit. A computed unit
/// string can fail, so the sig is honest `float!`; literal units are proven at solve
/// time and keep a plain `float` return (see `frame_method_call`).
#[native]
fn to(n: i64, unit: &str) -> Raisable<f64> {
    match shared::units::parse(unit) {
        Some((_, scale)) => Raisable::Ok(n as f64 / scale),
        None => Raisable::Raised(format!("`{unit}` isn't a unit")),
    }
}
