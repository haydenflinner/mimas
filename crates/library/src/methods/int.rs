use api::Intrinsic;
use macros::native;
use shared::{Literal, Ty};
use vm::{Ctx, RtErr, Val, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_method(abs);
    api.add_method(min);
    api.add_method(max);
    api.add_method(clamp);
    api.add_method(pow);
    api.add_method(to_str);
    api.add_method(to);
    let id = api.add_method(to_float);
    api.mark_intrinsic(id, Intrinsic::ToFloat);
    api.add_assoc(Ty::Int, random);
    api.assoc_constant(
        Ty::Int,
        "MAX",
        Ty::Int,
        Literal::Int(i64::MAX),
        "The largest int, `9223372036854775807` -- a running minimum over ints can start here.",
    );
    api.assoc_constant(
        Ty::Int,
        "MIN",
        Ty::Int,
        Literal::Int(i64::MIN),
        "The smallest int, `-9223372036854775808`.",
    );
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
/// Returns the number raised to the power `exponent`, an int. The exponent must not be
/// negative (`2.pow(-1)` is a half: write `2.0.pow(-1.0)`), and a power past `int::MAX`
/// is an error rather than a wrapped number.
///
/// ```mimas
/// let a = 2.pow(10); // 1024
/// let b = 7.pow(0);  // 1
/// let c = (-3).pow(3); // -27
/// ```
fn pow<'gc>(n: i64, exponent: i64) -> Result<i64, RtErr> {
    let e = u32::try_from(exponent)
        .map_err(|_| RtErr::InvalidArgument(format!("pow needs an exponent of 0 or more, got {exponent}; a negative power is a float (`{n}.0.pow({exponent}.0)`)")))?;
    n.checked_pow(e)
        .ok_or_else(|| RtErr::InvalidArgument(format!("{n}.pow({exponent}) overflows int")))
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
#[effects(rng)]
/// Returns a random `int` that is at least `0` and less than `len`. That makes
/// `int::random(xs.len())` a random index into `xs`.
///
/// `len` must be greater than `0`.
///
/// ```mimas
/// let roll = int::random(6) + 1; // 1 to 6
/// ```
fn random<'gc>(ctx: Ctx<'gc>, len: i64) -> Result<i64, RtErr> {
    if len <= 0 {
        return Err(RtErr::InvalidArgument("random requires len > 0".into()));
    }
    // one unit draw scaled — the same shape `mrt::random_int` produces,
    // so seeded interpreter and native streams stay in lockstep
    Ok(((super::rng::unit(ctx) * len as f64) as i64).clamp(0, len - 1))
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
/// unit of the same kind, exactly like float's `to`, and `q.to(1g)` divides by the
/// quantity's scale without the string. The compiler checks the unit measures the same
/// thing; at runtime only the name has to be a unit. A computed unit string can fail,
/// so the sig is honest `float!`; literal units are proven at solve time and keep a
/// plain `float` return (see `frame_method_call`).
#[native]
fn to<'gc>(n: i64, unit: Val<'gc>) -> Raisable<f64> {
    match unit {
        Val::Float(u) => Raisable::Ok(n as f64 / u),
        Val::Int(u) => Raisable::Ok(n as f64 / u as f64),
        Val::Str(u) => match shared::units::parse(u.as_str()) {
            Some((_, scale)) => Raisable::Ok(n as f64 / scale),
            None => Raisable::Raised(format!("`{}` isn't a unit", u.as_str())),
        },
        _ => Raisable::Raised("`to` needs a unit name or a quantity".into()),
    }
}

/// Same literal-unit proof as `float::to`'s -- the join keys on Rust path, so each `to`
/// registers its own validator (see `float.rs` for the contract).
fn valid_unit(args: &[shared::Literal]) -> Result<(), vm::api::LitFault> {
    match args.first() {
        Some(shared::Literal::Str(u)) if shared::units::parse(u).is_none() => {
            Err(vm::api::LitFault::from(format!("`{u}` isn't a unit")).arg(0))
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
