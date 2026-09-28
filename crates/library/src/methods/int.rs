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
/// Returns the absolute value.
///
/// ```mimas
/// let a = (-7).abs(); // 7
/// ```
fn abs<'gc>(n: i64) -> i64 {
    n.abs()
}

#[native]
/// Returns the smaller of the number and `other`.
///
/// ```mimas
/// let a = 3.min(8); // 3
/// ```
fn min<'gc>(a: i64, b: i64) -> i64 {
    a.min(b)
}

#[native]
/// Returns the larger of the number and `other`.
///
/// ```mimas
/// let a = 3.max(8);        // 8
/// let hp = (5 - 9).max(0); // 0
/// ```
fn max<'gc>(a: i64, b: i64) -> i64 {
    a.max(b)
}

#[native]
/// Returns the number moved into the range from `low` to `high`, inclusive. A number already in
/// the range comes back unchanged.
///
/// ```mimas
/// let a = 15.clamp(0, 10);   // 10
/// let b = (-3).clamp(0, 10); // 0
/// let c = 4.clamp(0, 10);    // 4
/// ```
fn clamp<'gc>(n: i64, lo: i64, hi: i64) -> Result<i64, RtErr> {
    if lo > hi {
        return Err(RtErr::InvalidArgument("clamp requires lo <= hi".into()));
    }
    Ok(n.clamp(lo, hi))
}

#[native]
/// Returns the number written out in base 10, with a leading `-` if it's negative. An f-string
/// does the same inside a larger string.
///
/// ```mimas
/// let a = (-42).to_str(); // "-42"
/// let b = f"{7} lives";   // "7 lives"
/// ```
fn to_str<'gc>(n: i64) -> String {
    n.to_string()
}

#[native]
/// Returns a random `int` that is at least `0` and less than `len`. That makes
/// `int::random(xs.len())` a random index into `xs`.
///
/// `len` must be greater than `0`.
///
/// ```mimas
/// let roll = int::random(6) + 1; // 1 to 6
/// ```
fn random<'gc>(len: i64) -> Result<i64, RtErr> {
    // random_range panics on an empty range
    if len <= 0 {
        return Err(RtErr::InvalidArgument("random requires len > 0".into()));
    }
    Ok(rand::rng().random_range(0..len))
}

#[native]
/// Returns the number as a `float`. Arithmetic that mixes `int` and `float` converts on its own,
/// but passing an `int` where a `float` is expected doesn't, and needs this first:
///
/// ```mimas
/// fn half(x: float) -> float {
///     x / 2.0
/// }
///
/// let n = 3;
/// let a = half(n.to_float());  // 1.5
/// let b = n.to_float().sqrt(); // 1.7320508075688772
/// ```
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

/// Same literal-unit proof as `float::to`'s -- the join keys on Rust path, so each `to`
/// registers its own validator (see `float.rs` for the contract).
fn valid_unit(args: &[shared::Literal]) -> Result<(), String> {
    match args.first() {
        Some(shared::Literal::Str(u)) if shared::units::parse(u).is_none() => {
            Err(format!("`{u}` isn't a unit"))
        }
        _ => Ok(()),
    }
}

vm::inventory::submit! {
    vm::api::NativeValidator {
        path: concat!(module_path!(), "::to"),
        validate: valid_unit,
    }
}
