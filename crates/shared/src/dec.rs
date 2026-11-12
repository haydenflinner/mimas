//! `dec`: an exact decimal number in 64 bits (docs/decimal.md).
//!
//! The layout follows Douglas Crockford's DEC64: the high 56 bits are a
//! signed coefficient, the low 8 bits a signed exponent, and the value is
//! `coefficient × 10^exponent`. That gives about 16 significant digits, and
//! a `dec` fits unboxed in the VM's `Val`, beside an `i64` or an `f64`.
//!
//! Add, subtract and multiply are exact: they fail with [`DecError`] when
//! the exact result needs more than the 56-bit coefficient holds, rather
//! than dropping digits. Division rounds to 16 significant digits, half to
//! even. Comparison and equality are numeric, so `1.0 == 1.00`. A `dec`
//! keeps the scale it was written with (`3.50` shows as `3.50`), and a sum
//! keeps the larger scale, as a ledger does.
//! r[impl mimas.dec]

use std::cmp::Ordering;
use std::fmt;

/// The largest coefficient: 2^55 - 1.
const COEF_MAX: i64 = (1 << 55) - 1;
const COEF_MIN: i64 = -(1 << 55);
/// Significant digits a quotient keeps.
const DIV_DIGITS: u32 = 16;

/// An exact decimal: a 56-bit coefficient and an 8-bit exponent in one `i64`.
#[derive(Clone, Copy, Debug)]
pub struct Dec(i64);

/// Why an operation on `dec` has no exact answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecError {
    /// The exact result needs more digits than 56 bits hold.
    Overflow,
    /// Division by zero.
    DivByZero,
    /// Text that isn't a decimal number.
    Parse,
}

impl fmt::Display for DecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DecError::Overflow => "the number is too large for a dec (past 10^127)",
            DecError::DivByZero => "division by zero",
            DecError::Parse => "not a decimal number",
        })
    }
}

impl std::error::Error for DecError {}

fn pow10(n: u32) -> Option<i128> {
    10i128.checked_pow(n)
}

impl Dec {
    pub const ZERO: Dec = Dec(0);

    /// `coefficient × 10^exponent`, if the coefficient fits in 56 bits.
    pub fn new(coefficient: i64, exponent: i8) -> Result<Dec, DecError> {
        if !(COEF_MIN..=COEF_MAX).contains(&coefficient) {
            return Err(DecError::Overflow);
        }
        Ok(Dec((coefficient << 8) | (exponent as u8 as i64)))
    }

    /// The coefficient, the high 56 bits.
    pub fn coefficient(self) -> i64 {
        self.0 >> 8
    }

    /// The exponent, the low 8 bits.
    pub fn exponent(self) -> i8 {
        self.0 as u8 as i8
    }

    /// The 64 bits as stored, for snapshots and hashing of the layout.
    pub fn to_bits(self) -> i64 {
        self.0
    }

    pub fn from_bits(bits: i64) -> Dec {
        Dec(bits)
    }

    pub fn from_int(n: i64) -> Result<Dec, DecError> {
        // a whole number too wide for 56 bits keeps 16 digits and a positive exponent
        let mut c = n as i128;
        let mut e = 0i32;
        while c > COEF_MAX as i128 || c < COEF_MIN as i128 {
            if c % 10 != 0 {
                return Err(DecError::Overflow);
            }
            c /= 10;
            e += 1;
        }
        Dec::new(c as i64, e as i8)
    }

    /// From a coefficient and exponent in wide integers, removing trailing
    /// zeros only when the coefficient doesn't fit, and failing when the
    /// value can't be held exactly.
    fn from_wide_exact(mut c: i128, mut e: i32) -> Result<Dec, DecError> {
        while c > COEF_MAX as i128 || c < COEF_MIN as i128 {
            if c % 10 != 0 {
                return Err(DecError::Overflow);
            }
            c /= 10;
            e += 1;
        }
        if e > i8::MAX as i32 {
            return Err(DecError::Overflow);
        }
        if e < i8::MIN as i32 {
            // too small to write: only zero survives
            return if c == 0 { Ok(Dec::ZERO) } else { Err(DecError::Overflow) };
        }
        Dec::new(c as i64, e as i8)
    }

    /// From a wide coefficient and exponent, exact when it fits, else
    /// rounded to 16 significant digits, half to even, the way division
    /// rounds and decimal64 rounds every operation. Money keeps few digits
    /// and stays exact; a damping loop (`v * 0.9` each frame) or a small
    /// step added to a large position rounds where it used to fail.
    /// Only a magnitude past about 10^127 is still an error.
    // r[impl mimas.dec.rounding]
    fn from_wide_rounded(c: i128, e: i32) -> Result<Dec, DecError> {
        if let Ok(d) = Dec::from_wide_exact(c, e) {
            return Ok(d);
        }
        let digits = |x: i128| x.unsigned_abs().to_string().len() as i32;
        let negative = c < 0;
        let mut q = c.abs();
        let mut e = e;
        // drop digits down to DIV_DIGITS + 1, remembering whether anything
        // nonzero went (the sticky bit), then round the guard digit away
        let mut sticky = false;
        let keep = DIV_DIGITS as i32 + 1;
        while digits(q) > keep || e < i8::MIN as i32 {
            sticky |= q % 10 != 0;
            q /= 10;
            e += 1;
            if q == 0 {
                break;
            }
        }
        if digits(q) == keep {
            let last = q % 10;
            q /= 10;
            e += 1;
            if last > 5 || (last == 5 && (sticky || q % 2 != 0)) {
                q += 1;
            }
            if digits(q) > DIV_DIGITS as i32 {
                // 9999999999999999.5 rounded up to 10000000000000000
                q /= 10;
                e += 1;
            }
        }
        // the trailing zeros rounding leaves say nothing about the scale
        while q != 0 && q % 10 == 0 && e < 0 {
            q /= 10;
            e += 1;
        }
        Dec::from_wide_exact(if negative { -q } else { q }, e)
    }

    /// Both values with one exponent, the smaller, as wide coefficients.
    fn aligned(self, o: Dec) -> Option<(i128, i128, i32)> {
        let (ea, eb) = (self.exponent() as i32, o.exponent() as i32);
        let e = ea.min(eb);
        let a = (self.coefficient() as i128).checked_mul(pow10((ea - e) as u32)?)?;
        let b = (o.coefficient() as i128).checked_mul(pow10((eb - e) as u32)?)?;
        Some((a, b, e))
    }

    pub fn add(self, o: Dec) -> Result<Dec, DecError> {
        if self.coefficient() == 0 {
            return Ok(o.with_min_scale(self));
        }
        if o.coefficient() == 0 {
            return Ok(self.with_min_scale(o));
        }
        match self.aligned(o) {
            Some((a, b, e)) => Dec::from_wide_rounded(a + b, e),
            // scales so far apart that the smaller one is far below the
            // 16th digit of the larger: the sum, rounded, is the larger
            None => Ok(if (self.exponent() as i32) > (o.exponent() as i32) { self } else { o }),
        }
    }

    pub fn sub(self, o: Dec) -> Result<Dec, DecError> {
        self.add(o.neg()?)
    }

    pub fn neg(self) -> Result<Dec, DecError> {
        Dec::new(-self.coefficient(), self.exponent())
    }

    /// `self` written with at least the scale of `other`: `0 + 1.50` is `1.50`.
    fn with_min_scale(self, other: Dec) -> Dec {
        let (e, eo) = (self.exponent() as i32, other.exponent() as i32);
        if eo >= e {
            return self;
        }
        match pow10((e - eo) as u32).and_then(|p| (self.coefficient() as i128).checked_mul(p)) {
            Some(c) => Dec::from_wide_exact(c, eo).unwrap_or(self),
            None => self,
        }
    }

    pub fn mul(self, o: Dec) -> Result<Dec, DecError> {
        let c = (self.coefficient() as i128) * (o.coefficient() as i128);
        let e = self.exponent() as i32 + o.exponent() as i32;
        Dec::from_wide_rounded(c, e)
    }

    /// The quotient to 16 significant digits, half to even, without the
    /// trailing zeros that rounding leaves (`1 / 4` is `0.25`).
    pub fn div(self, o: Dec) -> Result<Dec, DecError> {
        if o.coefficient() == 0 {
            return Err(DecError::DivByZero);
        }
        if self.coefficient() == 0 {
            return Ok(Dec::ZERO);
        }
        let (a, b) = (self.coefficient() as i128, o.coefficient() as i128);
        // scale the dividend so the quotient has DIV_DIGITS + 1 digits, then round one away
        let digits = |x: i128| x.unsigned_abs().to_string().len() as i32;
        // the quotient of a d_a-digit and a d_b-digit number has d_a - d_b or
        // d_a - d_b + 1 digits; aim for DIV_DIGITS + 1 (one guard digit), and
        // divide again with one less shift when it came out a digit long
        let mut shift = ((DIV_DIGITS as i32 + 1) - (digits(a) - digits(b))).max(0);
        let (q0, r) = loop {
            let num = a.checked_mul(pow10(shift as u32).ok_or(DecError::Overflow)?).ok_or(DecError::Overflow)?;
            let (q0, r) = (num / b, num % b);
            if digits(q0) > DIV_DIGITS as i32 + 1 && shift > 0 {
                shift -= 1;
                continue;
            }
            break (q0, r);
        };
        let negative = (a < 0) != (b < 0);
        // drop the guard digit, half to even; a nonzero remainder means "more than half"
        let last = (q0 % 10).abs();
        let mut q = q0 / 10;
        let mut e = self.exponent() as i32 - o.exponent() as i32 - shift + 1;
        if last > 5 || (last == 5 && (r != 0 || q % 2 != 0)) {
            q += if negative { -1 } else { 1 };
        }
        while q != 0 && q % 10 == 0 && e < 0 {
            q /= 10;
            e += 1;
        }
        Dec::from_wide_exact(q, e)
    }

    /// Rounded to `places` digits after the point, half away from zero, as a
    /// spreadsheet's `ROUND` does: `2.675` to 2 places is `2.68`.
    pub fn round(self, places: i32) -> Result<Dec, DecError> {
        let e = self.exponent() as i32;
        if e >= -places {
            return Ok(self);
        }
        let drop = (-places - e) as u32;
        let p = pow10(drop).ok_or(DecError::Overflow)?;
        let c = self.coefficient() as i128;
        let (q, r) = (c / p, c % p);
        let q = if r.abs() * 2 >= p { q + c.signum() } else { q };
        Dec::from_wide_exact(q, -places)
    }

    pub fn to_f64(self) -> f64 {
        // through the text, so the nearest double to the written digits
        self.to_string().parse().unwrap_or(f64::NAN)
    }

    /// The `dec` nearest a double, by its shortest round-trip digits:
    /// `0.1f64` is `0.1`.
    pub fn from_f64(f: f64) -> Result<Dec, DecError> {
        if !f.is_finite() {
            return Err(DecError::Parse);
        }
        format!("{f:?}").parse()
    }

    pub fn is_zero(self) -> bool {
        self.coefficient() == 0
    }

    pub fn signum(self) -> i64 {
        self.coefficient().signum()
    }

    pub fn abs(self) -> Result<Dec, DecError> {
        if self.coefficient() < 0 { self.neg() } else { Ok(self) }
    }

    /// The same value with no trailing zeros, the form equal values share.
    pub fn normalized(self) -> Dec {
        let (mut c, mut e) = (self.coefficient(), self.exponent() as i32);
        if c == 0 {
            return Dec::ZERO;
        }
        while c % 10 == 0 && e < i8::MAX as i32 {
            c /= 10;
            e += 1;
        }
        Dec::new(c, e as i8).unwrap_or(self)
    }
}

impl PartialEq for Dec {
    fn eq(&self, o: &Dec) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Dec {}

impl std::hash::Hash for Dec {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.normalized().0.hash(h)
    }
}

impl PartialOrd for Dec {
    fn partial_cmp(&self, o: &Dec) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Dec {
    fn cmp(&self, o: &Dec) -> Ordering {
        match self.aligned(*o) {
            Some((a, b, _)) => a.cmp(&b),
            // exponents too far apart to align: the signs, then the sizes, decide
            None => {
                let (sa, sb) = (self.signum(), o.signum());
                if sa != sb {
                    return sa.cmp(&sb);
                }
                let mag = |d: Dec| {
                    d.coefficient().unsigned_abs().to_string().len() as i32 + d.exponent() as i32
                };
                let by_mag = mag(*self).cmp(&mag(*o));
                if sa < 0 { by_mag.reverse() } else { by_mag }
            }
        }
    }
}

impl fmt::Display for Dec {
    /// The digits as stored: `3.50`, `-0.001`, `1200`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (c, e) = (self.coefficient(), self.exponent() as i32);
        let neg = c < 0;
        let digits = c.unsigned_abs().to_string();
        let body = if e >= 0 {
            if e > 20 {
                format!("{digits}e{e}")
            } else {
                format!("{digits}{}", "0".repeat(e as usize))
            }
        } else {
            let point = -e as usize;
            if point > 30 {
                format!("{digits}e{e}")
            } else if digits.len() > point {
                let (i, frac) = digits.split_at(digits.len() - point);
                format!("{i}.{frac}")
            } else {
                format!("0.{}{digits}", "0".repeat(point - digits.len()))
            }
        };
        if neg { write!(f, "-{body}") } else { f.write_str(&body) }
    }
}

impl std::str::FromStr for Dec {
    type Err = DecError;

    /// `3.50`, `-12`, `1.5e3`, `2E-4`, `1_000.25`.
    fn from_str(s: &str) -> Result<Dec, DecError> {
        let s = s.trim().replace('_', "");
        let (mant, exp) = match s.find(['e', 'E']) {
            Some(i) => (&s[..i], s[i + 1..].parse::<i32>().map_err(|_| DecError::Parse)?),
            None => (s.as_str(), 0),
        };
        let (neg, mant) = match mant.strip_prefix('-') {
            Some(m) => (true, m),
            None => (false, mant.strip_prefix('+').unwrap_or(mant)),
        };
        let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
        if int.is_empty() && frac.is_empty() {
            return Err(DecError::Parse);
        }
        if !int.chars().chain(frac.chars()).all(|c| c.is_ascii_digit()) {
            return Err(DecError::Parse);
        }
        let digits = format!("{int}{frac}");
        let trimmed = digits.trim_start_matches('0');
        if trimmed.len() > 18 {
            return Err(DecError::Overflow);
        }
        let mut c: i128 = if trimmed.is_empty() { 0 } else { trimmed.parse().map_err(|_| DecError::Parse)? };
        if neg {
            c = -c;
        }
        Dec::from_wide_exact(c, exp - frac.len() as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Dec {
        s.parse().unwrap()
    }

    // r[verify mimas.dec]
    #[test]
    fn sums_are_exact_and_keep_the_written_scale() {
        assert_eq!(d("0.1").add(d("0.2")).unwrap(), d("0.3"));
        assert_eq!(d("1.10").add(d("2.20")).unwrap().to_string(), "3.30");
        let mut s = Dec::ZERO;
        for _ in 0..10 {
            s = s.add(d("0.1")).unwrap();
        }
        assert_eq!(s.to_string(), "1.0");
        assert_eq!(d("5").sub(d("7.25")).unwrap().to_string(), "-2.25");
    }

    #[test]
    fn products_are_exact() {
        // the pound is defined as exactly 0.45359237 kg
        assert_eq!(d("48").mul(d("0.45359237")).unwrap().to_string(), "21.77243376");
        assert_eq!(d("1.10").mul(d("3")).unwrap().to_string(), "3.30");
        assert_eq!(d("-0.5").mul(d("0.5")).unwrap().to_string(), "-0.25");
    }

    #[test]
    fn quotients_round_to_sixteen_digits_half_even() {
        assert_eq!(d("1").div(d("4")).unwrap().to_string(), "0.25");
        assert_eq!(d("1").div(d("3")).unwrap().to_string(), "0.3333333333333333");
        assert_eq!(d("2").div(d("3")).unwrap().to_string(), "0.6666666666666667");
        assert_eq!(d("-2").div(d("3")).unwrap().to_string(), "-0.6666666666666667");
        assert_eq!(d("10").div(d("2")).unwrap(), d("5"));
        assert_eq!(d("1").div(d("0")), Err(DecError::DivByZero));
        assert_eq!(d("0.6").div(d("3")).unwrap(), d("0.2"));
    }

    #[test]
    fn rounding_is_half_away_from_zero() {
        assert_eq!(d("2.675").round(2).unwrap().to_string(), "2.68");
        assert_eq!(d("-2.675").round(2).unwrap().to_string(), "-2.68");
        assert_eq!(d("2.674").round(2).unwrap().to_string(), "2.67");
        assert_eq!(d("2.5").round(0).unwrap().to_string(), "3");
        assert_eq!(d("1.2").round(3).unwrap().to_string(), "1.2");
    }

    #[test]
    fn equal_values_compare_and_hash_alike() {
        use std::collections::HashSet;
        assert_eq!(d("1.0"), d("1.00"));
        assert!(d("0.3") < d("0.31"));
        assert!(d("-1") < d("0.001"));
        assert!(d("1e100") > d("1e-100"));
        let set: HashSet<Dec> = [d("0.30"), d("0.3"), d("0.1").add(d("0.2")).unwrap()].into_iter().collect();
        assert_eq!(set.len(), 1);
    }

    // r[verify mimas.dec.rounding]
    #[test]
    fn a_result_past_sixteen_digits_rounds_half_even() {
        let big = d("36028797018963967");
        assert_eq!(big.add(d("1")).unwrap(), d("3602879701896397e1"));
        assert_eq!(big.mul(d("10")).unwrap().to_string(), "360287970189639670");
        assert_eq!(d("0.1234567890123456").mul(d("0.1234567890123456")).unwrap(), d("0.01524157875323882"));
        // a damping loop keeps going instead of failing at the 16th step
        let mut v = d("1.5");
        for _ in 0..200 {
            v = v.mul(d("0.9")).unwrap();
        }
        assert!(v > Dec::ZERO && v < d("0.000000002"));
        // a small step on a large position, and scales too far apart to align
        assert_eq!(d("1000000000000000").add(d("0.001")).unwrap(), d("1000000000000000"));
        assert_eq!(d("1e100").add(d("1e-100")).unwrap(), d("1e100"));
        // half to even on an exact tie
        assert_eq!(d("1000000000000000.5").add(d("0")).unwrap(), d("1000000000000000.5"));
        assert_eq!(d("10000000000000005").mul(d("10")).unwrap(), d("100000000000000050"));
        assert_eq!(d("1234567890123456.5").add(d("1")).unwrap(), d("1234567890123457.5")); // fits: exact
        assert_eq!(d("3602879701896396.5").add(d("1")).unwrap(), d("3602879701896398")); // a tie past 16 digits: to even
        // money stays exact
        assert_eq!(d("1.10").mul(d("12")).unwrap().to_string(), "13.20");
    }

    #[test]
    fn text_and_doubles_round_trip() {
        for s in ["0", "3.50", "-0.001", "1200", "12.345678901234"] {
            assert_eq!(d(s).to_string(), s);
        }
        assert_eq!(d("1.5e3").to_string(), "1500");
        assert_eq!(Dec::from_f64(0.1).unwrap(), d("0.1"));
        assert_eq!(d("0.1").to_f64(), 0.1);
        assert_eq!(Dec::from_int(-42).unwrap().to_string(), "-42");
        assert!("1.2.3".parse::<Dec>().is_err());
    }

    /// Random operands against `rust_decimal` (28 digits), rounded to our
    /// 16 significant digits half to even for division.
    #[test]
    fn agrees_with_rust_decimal() {
        use rust_decimal::{Decimal, RoundingStrategy};
        let mut seed: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..20_000 {
            let a = Dec::new((next() % 2_000_000_001) as i64 - 1_000_000_000, -((next() % 7) as i8)).unwrap();
            let b = Dec::new((next() % 2_000_001) as i64 - 1_000_000, -((next() % 5) as i8)).unwrap();
            let (ra, rb): (Decimal, Decimal) = (a.to_string().parse().unwrap(), b.to_string().parse().unwrap());
            assert_eq!(a.add(b).unwrap().to_string().parse::<Decimal>().unwrap(), ra + rb, "{a} + {b}");
            assert_eq!(a.mul(b).unwrap().to_string().parse::<Decimal>().unwrap(), ra * rb, "{a} * {b}");
            assert_eq!(a.cmp(&b), ra.cmp(&rb), "{a} <=> {b}");
            if !b.is_zero() {
                let want = (ra / rb).round_sf_with_strategy(16, RoundingStrategy::MidpointNearestEven).unwrap().normalize();
                let got: Decimal = a.div(b).unwrap().to_string().parse().unwrap();
                assert_eq!(got.normalize(), want, "{a} / {b}");
            }
        }
    }
}
