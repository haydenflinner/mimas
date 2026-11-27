//! `dec`'s methods, and the `to_dec` conversions into it (docs/decimal.md).
//! r[impl mimas.dec]
use macros::native;
use shared::dec::{Dec, DecError};
use vm::{RtErr, api::Api, conversion::{DecTy, Raisable}};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    // before the methods: their `dec` parameters resolve through the registry
    api.add_adt::<DecTy>();
    api.add_method_named("round", dec_round);
    api.add_method_named("to_float", dec_to_float);
    api.add_method_named("to_str", dec_to_str);
    api.add_method_named("abs", dec_abs);
    api.add_method_named("to_dec", int_to_dec);
    api.add_method_named("to_dec", float_to_dec);
    api.add_method_named("to_dec", str_to_dec);
}

fn fault(what: &str, e: DecError) -> RtErr {
    RtErr::Custom(format!("dec: {what}: {e}"))
}

#[native]
/// This number rounded to `places` digits after the point, half away from
/// zero, as a spreadsheet's `ROUND` does: `2.675` to 2 places is `2.68`.
fn dec_round(d: Dec, places: i64) -> Result<Dec, RtErr> {
    d.round(places as i32).map_err(|e| fault(&format!("{d}.round({places})"), e))
}

#[native]
/// The nearest `float`, for arithmetic that needs one (`sqrt`, `sin`).
fn dec_to_float(d: Dec) -> f64 {
    d.to_f64()
}

#[native]
/// The digits as stored: `3.50` stays `"3.50"`.
fn dec_to_str(d: Dec) -> String {
    d.to_string()
}

#[native]
/// The size of this number, without its sign.
fn dec_abs(d: Dec) -> Result<Dec, RtErr> {
    d.abs().map_err(|e| fault(&format!("{d}.abs()"), e))
}

#[native]
/// This whole number as an exact decimal.
fn int_to_dec(n: i64) -> Result<Dec, RtErr> {
    Dec::from_int(n).map_err(|e| fault(&n.to_string(), e))
}

#[native]
/// The decimal with this float's shortest digits: `0.1.to_dec()` is
/// exactly `0.1`.
fn float_to_dec(f: f64) -> Result<Dec, RtErr> {
    Dec::from_f64(f).map_err(|e| fault(&f.to_string(), e))
}

#[native]
/// The decimal this text writes (`"3.50"`, `"-12"`, `"1.5e3"`), or `null`
/// when it isn't one, as `str.to_int` answers. r[impl mimas.str.parse-option]
fn str_to_dec(s: String) -> Option<Dec> {
    s.trim().parse::<Dec>().ok()
}
