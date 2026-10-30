use api::Intrinsic;
use macros::native;
use shared::Ty;
use vm::{Ctx, RtErr, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_method(floor);
    api.add_method(to_str);
    api.add_method(ceil);
    api.add_method(round);
    api.add_method(to_int);
    api.add_method(abs);
    api.add_method(max);
    api.add_method(min);
    let id = api.add_method(sqrt);
    api.add_method(hypot);
    api.mark_intrinsic(id, Intrinsic::Sqrt);
    api.add_method(format);
    api.add_method(to);
    api.add_method(clamp);
    api.add_method(signum);
    api.add_method(sin);
    api.add_method(cos);
    api.add_method(tan);
    api.add_method(asin);
    api.add_method(acos);
    api.add_method(atan);
    api.add_method(atan2);
    api.add_method(pow);
    api.add_method(exp);
    api.add_method(ln);
    api.add_assoc(Ty::Float, random);
}

#[native]
/// Returns the number moved into the range from `low` to `high`, inclusive. A number already in
/// the range comes back unchanged.
///
/// ```mimas
/// let a = 1.5.clamp(0.0, 1.0);    // 1.0
/// let b = (-0.2).clamp(0.0, 1.0); // 0.0
/// ```
fn clamp(n: f64, low: f64, high: f64) -> Result<f64, RtErr> {
    // a NaN bound must error too — f64::clamp panics on both cases
    if low.is_nan() || high.is_nan() || low > high {
        return Err(RtErr::InvalidArgument("clamp requires lo <= hi".into()));
    }
    Ok(n.clamp(low, high))
}

#[native]
/// Returns the sign of the number as `1.0` or `-1.0`. Zero counts as positive (`0.0.signum()` is
/// `1.0`), unless it's `-0.0`.
///
/// ```mimas
/// let a = (-3.2).signum(); // -1.0
/// let b = 5.0.signum();    // 1.0
/// ```
fn signum(n: f64) -> f64 {
    n.signum()
}

#[native]
/// Returns the sine of an angle in radians. To work in degrees, multiply by `PI / 180.0` first.
///
/// ```mimas
/// use std::math::PI;
///
/// let a = (PI / 2.0).sin();          // 1.0
/// let b = (90.0 * PI / 180.0).sin(); // 1.0
/// ```
fn sin(radians: f64) -> f64 {
    radians.sin()
}

#[native]
/// Returns the cosine of an angle in radians.
///
/// ```mimas
/// use std::math::PI;
///
/// let a = PI.cos(); // -1.0
/// ```
fn cos(radians: f64) -> f64 {
    radians.cos()
}

#[native]
/// Returns the tangent of an angle in radians.
///
/// ```mimas
/// let a = 0.0.tan(); // 0.0
/// ```
fn tan(radians: f64) -> f64 {
    radians.tan()
}

#[native]
/// Returns the arcsine in radians, between `-PI / 2.0` and `PI / 2.0`. A number outside `-1.0` to
/// `1.0` gives `NaN`.
///
/// ```mimas
/// let a = 1.0.asin(); // 1.5707963267948966 (PI / 2.0)
/// ```
fn asin(n: f64) -> f64 {
    n.asin()
}

#[native]
/// Returns the arccosine in radians, between `0.0` and `PI`. A number outside `-1.0` to `1.0`
/// gives `NaN`.
///
/// ```mimas
/// let a = 1.0.acos(); // 0.0
/// ```
fn acos(n: f64) -> f64 {
    n.acos()
}

#[native]
/// Returns the arctangent in radians, between `-PI / 2.0` and `PI / 2.0`. To get an angle from a
/// point or direction, [`atan2`](#atan2) is usually the better choice.
///
/// ```mimas
/// let a = 1.0.atan(); // 0.7853981633974483 (PI / 4.0)
/// ```
fn atan(n: f64) -> f64 {
    n.atan()
}

#[native]
/// Returns the angle in radians from the positive x axis to the point (`x`, `y`), where `y` is the
/// number this is called on. The result is between `-PI` and `PI`.
///
/// Unlike `(y / x).atan()`, it uses the signs of both coordinates to find the right quadrant, and
/// an `x` of `0.0` is fine.
///
/// ```mimas
/// let a = 1.0.atan2(1.0);  // 0.7853981633974483 (PI / 4.0)
/// let b = 1.0.atan2(-1.0); // 2.356194490192345 (3.0 * PI / 4.0)
/// ```
fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

#[native]
/// Returns the number raised to the power `exponent`.
///
/// ```mimas
/// let a = 2.0.pow(10.0); // 1024.0
/// let b = 9.0.pow(0.5);  // 3.0
/// ```
fn pow(n: f64, exponent: f64) -> f64 {
    n.powf(exponent)
}

#[native]
/// Returns `E` raised to the power of the number.
///
/// ```mimas
/// let a = 1.0.exp(); // 2.718281828459045
/// ```
fn exp(n: f64) -> f64 {
    n.exp()
}

#[native]
/// Returns the natural logarithm (base `E`). `0.0` gives negative infinity, and a negative number
/// gives `NaN`.
///
/// ```mimas
/// use std::math::E;
///
/// let a = E.ln();   // 1.0
/// let b = 1.0.ln(); // 0.0
/// ```
fn ln(n: f64) -> f64 {
    n.ln()
}

#[native]
/// Returns the largest whole number less than or equal to the number, as a `float`.
///
/// ```mimas
/// let a = 2.7.floor();    // 2.0
/// let b = (-2.7).floor(); // -3.0
/// ```
fn floor(n: f64) -> f64 {
    n.floor()
}

#[native]
/// Returns the nearest whole number, as a `float`. A number exactly halfway between two whole
/// numbers rounds away from zero.
///
/// ```mimas
/// let a = 2.4.round();    // 2.0
/// let b = 2.5.round();    // 3.0
/// let c = (-2.5).round(); // -3.0
/// ```
fn round(n: f64) -> f64 {
    n.round()
}

#[native]
/// Returns the smallest whole number greater than or equal to the number, as a `float`.
///
/// ```mimas
/// let a = 2.2.ceil();    // 3.0
/// let b = (-2.2).ceil(); // -2.0
/// ```
fn ceil(n: f64) -> f64 {
    n.ceil()
}

#[native]
/// Returns the number as an `int`, dropping anything after the decimal point. Round first with
/// [`round`](#round) to get the nearest `int` instead.
///
/// ```mimas
/// let a = 2.9.to_int();         // 2
/// let b = (-2.9).to_int();      // -2
/// let c = 2.9.round().to_int(); // 3
/// ```
fn to_int(n: f64) -> i64 {
    n as i64
}

/// `q.to("kWh")` -- a quantity as a plain number in `unit`. A quantity is stored in base units
/// (`25kW` is `25000.0`), so this divides by the unit's scale. The compiler checks that `q` and
/// `unit` measure the same thing; at runtime only the name has to be a unit. A computed unit
/// string can fail, so the sig is honest `float!`; literal units are proven at solve time and
/// keep a plain `float` return (see `frame_method_call`).
#[native]
fn to(n: f64, unit: &str) -> Raisable<f64> {
    match shared::units::parse(unit) {
        Some((_, scale)) => Raisable::Ok(n / scale),
        None => Raisable::Raised(format!("`{unit}` isn't a unit")),
    }
}

/// A literal unit string is checked at solve time: a known unit proves `to` can't raise
/// (`float!` narrows to `float`), an unknown one is a compile error. `int::to` has its own
/// copy -- the validators are joined by Rust path, so each `to` needs its own submission.
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

#[native]
/// Returns the absolute value.
///
/// ```mimas
/// let a = (-1.5).abs(); // 1.5
/// ```
fn abs(n: f64) -> f64 {
    n.abs()
}

#[native]
/// Returns the larger of the number and `other`.
///
/// ```mimas
/// let a = 1.5.max(2.5); // 2.5
/// ```
fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

#[native]
/// Returns the smaller of the number and `other`.
///
/// ```mimas
/// let a = 1.5.min(2.5); // 1.5
/// ```
fn min(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[native]
/// Returns the square root. A negative number gives `NaN`.
///
/// ```mimas
/// let a = 9.0.sqrt(); // 3.0
/// ```
fn sqrt(n: f64) -> f64 {
    n.sqrt()
}

#[native]
/// Returns the length of the hypotenuse of a right triangle with legs `a` and `b`,
/// `sqrt(a * a + b * b)`, computed without overflowing when the legs are huge.
///
/// ```mimas
/// let diagonal = 3.0.hypot(4.0); // 5.0
/// ```
fn hypot(a: f64, b: f64) -> f64 {
    a.hypot(b)
}

#[native]
/// Returns the number as a string with exactly `places` digits after the decimal point. The last
/// digit is rounded, with a tie going to the even digit (`2.5.format(0)` is `"2"`, but
/// `3.5.format(0)` is `"4"`).
///
/// ```mimas
/// let a = 3.14159.format(2); // "3.14"
/// let b = 2.0.format(3);     // "2.000"
/// ```
fn format(n: f64, places: usize) -> Result<String, RtErr> {
    // 1074 fractional digits covers the longest possible f64 expansion; anything past
    // that is zero padding, and a huge count would allocate it
    if places > 1074 {
        return Err(RtErr::InvalidArgument(
            "format places cannot exceed 1074".into(),
        ));
    }
    Ok(format!("{n:.places$}"))
}

#[native]
/// Returns the number as a string, using the fewest digits that still read back as the same
/// `float`. A whole number has no `.0`. For a fixed number of decimal places, use
/// [`format`](#format).
///
/// ```mimas
/// let a = 0.1.to_str();         // "0.1"
/// let b = 2.0.to_str();         // "2"
/// let c = (0.1 + 0.2).to_str(); // "0.30000000000000004"
/// ```
fn to_str(n: f64) -> String {
    n.to_string()
}

#[native]
/// Returns a random `float` that is at least `0.0` and less than `len`.
///
/// `len` must be greater than `0.0`.
///
/// ```mimas
/// if float::random(1.0) < 0.25 {
///     print("a one in four chance");
/// }
/// ```
#[effects(rng)]
fn random<'gc>(ctx: Ctx<'gc>, len: f64) -> Result<f64, RtErr> {
    if len.is_nan() || len <= 0.0 {
        return Err(RtErr::InvalidArgument("random requires len > 0".into()));
    }
    Ok(super::rng::unit(ctx) * len)
}
