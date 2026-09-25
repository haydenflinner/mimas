//! `Interval` — a number with its uncertainty attached.
//!
//! An estimate isn't a point: the grant might be $24k or $36k, the sun
//! might deliver 3.8 or 4.6 peak hours. `Interval { lo, mid, hi }` carries
//! the plausible range and a best guess through ordinary arithmetic:
//! `+ - * /` (with other intervals or plain numbers) propagate the bounds
//! by interval arithmetic and the best guess by the plain operation, so a
//! whole model written over `Interval`s comes out with honest error bars
//! and no extra code. `.sample(u)` draws from the triangular distribution
//! the three numbers describe — the Monte Carlo hook.
//!
//! Ordering (`<` `<=` `>` `>=`) against another interval or a plain number is
//! *certain*: `x < y` is true only when it holds for every value in both
//! ranges (`x.hi < y.lo`). So `!(x < y)` does not mean `x >= y` — the ranges
//! may overlap; ask `x.lo < y.hi` (possibly) or compare the `mid`s (best guess).
//!
//! Interval arithmetic is conservative: it assumes every input sits at its
//! worst *independently*, and it can't see that the same estimate used
//! twice (`x - x`) cancels. For a tighter picture, sample.

use macros::{MimasStruct, native};
use vm::conversion::MimasType;
use vm::{BinOp, Ctx, RtErr, RtResult, UnaryOp, Val, api::Api};

#[derive(Debug, Clone, Copy, MimasStruct)]
pub struct Interval {
    pub lo: f64,
    pub mid: f64,
    pub hi: f64,
}

impl Interval {
    fn point(x: f64) -> Self {
        Interval { lo: x, mid: x, hi: x }
    }

    fn hull(mid: f64, candidates: [f64; 4]) -> Self {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for c in candidates {
            if c.is_nan() {
                continue; // 0 * inf — no information
            }
            lo = lo.min(c);
            hi = hi.max(c);
        }
        if lo > hi {
            return Interval { lo: f64::NEG_INFINITY, mid, hi: f64::INFINITY };
        }
        Interval { lo, mid, hi }
    }

    pub fn add(self, o: Self) -> Self {
        Interval { lo: self.lo + o.lo, mid: self.mid + o.mid, hi: self.hi + o.hi }
    }
    pub fn sub(self, o: Self) -> Self {
        Interval { lo: self.lo - o.hi, mid: self.mid - o.mid, hi: self.hi - o.lo }
    }
    pub fn mul(self, o: Self) -> Self {
        Self::hull(
            self.mid * o.mid,
            [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi],
        )
    }
    pub fn div(self, o: Self) -> Self {
        if o.lo <= 0.0 && o.hi >= 0.0 {
            // the divisor might be zero: nothing can be said about the bounds
            return Interval { lo: f64::NEG_INFINITY, mid: self.mid / o.mid, hi: f64::INFINITY };
        }
        Self::hull(
            self.mid / o.mid,
            [self.lo / o.lo, self.lo / o.hi, self.hi / o.lo, self.hi / o.hi],
        )
    }
    /// `self ^ n` for a non-negative base — monotone, so the ends map
    /// straight through (a decreasing power for negative `n` swaps them).
    pub fn pow(self, n: f64) -> Self {
        let (a, b) = (self.lo.max(0.0).powf(n), self.hi.max(0.0).powf(n));
        Interval { lo: a.min(b), mid: self.mid.max(0.0).powf(n), hi: a.max(b) }
    }
    /// The triangular distribution over `[lo, hi]` with mode `mid`, at
    /// quantile `u` in 0..1.
    pub fn sample(self, u: f64) -> f64 {
        let (a, c, b) = (self.lo, self.mid.clamp(self.lo, self.hi), self.hi);
        if b <= a {
            return c;
        }
        let split = (c - a) / (b - a);
        if u < split {
            a + (u * (b - a) * (c - a)).sqrt()
        } else {
            b - ((1.0 - u) * (b - a) * (b - c)).sqrt()
        }
    }
}

mod iv {
    use super::*;

    /// `Interval::pm(mid, rel)` — `mid` give or take a fraction:
    /// `Interval::pm(30000.0, 0.1)` is 27000 … 33000.
    #[native]
    pub fn pm<'gc>(_ctx: Ctx<'gc>, mid: f64, rel: f64) -> Interval {
        let d = (mid * rel).abs();
        Interval { lo: mid - d, mid, hi: mid + d }
    }

    /// `Interval::within(mid, delta)` — `mid` give or take an absolute amount.
    #[native]
    pub fn within<'gc>(_ctx: Ctx<'gc>, mid: f64, delta: f64) -> Interval {
        let d = delta.abs();
        Interval { lo: mid - d, mid, hi: mid + d }
    }

    /// `Interval::span(lo, hi)` — anywhere between; the best guess is the middle.
    #[native]
    pub fn span<'gc>(_ctx: Ctx<'gc>, lo: f64, hi: f64) -> Interval {
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        Interval { lo, mid: (lo + hi) / 2.0, hi }
    }

    /// `Interval::of(lo, mid, hi)` — an asymmetric estimate.
    #[native]
    pub fn of<'gc>(_ctx: Ctx<'gc>, lo: f64, mid: f64, hi: f64) -> Interval {
        Interval { lo: lo.min(mid), mid, hi: hi.max(mid) }
    }

    /// `Interval::exact(x)` — no uncertainty (what a plain number becomes
    /// when it meets an interval).
    #[native]
    pub fn exact<'gc>(_ctx: Ctx<'gc>, x: f64) -> Interval {
        Interval::point(x)
    }

    /// `x.width()` — `hi - lo`.
    #[native]
    pub fn width<'gc>(_ctx: Ctx<'gc>, x: Interval) -> f64 {
        x.hi - x.lo
    }

    /// `x.contains(v)` — is `v` inside the range?
    #[native]
    pub fn contains<'gc>(_ctx: Ctx<'gc>, x: Interval, v: f64) -> bool {
        x.lo <= v && v <= x.hi
    }

    /// `x.sample(u)` — a draw from the triangular distribution (`u` in 0..1).
    #[native]
    pub fn sample<'gc>(_ctx: Ctx<'gc>, x: Interval, u: f64) -> f64 {
        x.sample(u)
    }

    /// `x.pow(n)` — `x` to the power `n`, for a non-negative `x`.
    #[native]
    pub fn pow<'gc>(_ctx: Ctx<'gc>, x: Interval, n: f64) -> Interval {
        x.pow(n)
    }

    /// `x.overlaps(y)` — could the two ranges share a value? (the "possibly"
    /// that `<`/`>` don't say)
    #[native]
    pub fn overlaps<'gc>(_ctx: Ctx<'gc>, x: Interval, y: Interval) -> bool {
        x.lo <= y.hi && y.lo <= x.hi
    }

    fn operand<'gc>(ctx: Ctx<'gc>, v: Val<'gc>) -> Option<Interval> {
        if let Ok(i) = Interval::from_value(ctx, v) {
            return Some(i);
        }
        match v {
            Val::Float(f) => Some(Interval::point(f)),
            Val::Int(i) => Some(Interval::point(i as f64)),
            _ => None,
        }
    }

    /// Infix `a <op> b` where either side is an interval and the other an
    /// interval or a plain number.
    pub fn bin<'gc>(ctx: Ctx<'gc>, a: Val<'gc>, b: Val<'gc>, op: BinOp) -> RtResult<Val<'gc>> {
        let (Some(x), Some(y)) = (operand(ctx, a), operand(ctx, b)) else {
            return Err(RtErr::invalid_bin(a, op, b));
        };
        let out = match op {
            BinOp::Add => x.add(y),
            BinOp::Sub => x.sub(y),
            BinOp::Mult => x.mul(y),
            BinOp::Div => x.div(y),
            // certain comparisons — see the module docs
            BinOp::LessThan => return Ok(Val::Bool(x.hi < y.lo)),
            BinOp::LessEqual => return Ok(Val::Bool(x.hi <= y.lo)),
            BinOp::GreaterThan => return Ok(Val::Bool(x.lo > y.hi)),
            BinOp::GreaterEqual => return Ok(Val::Bool(x.lo >= y.hi)),
            _ => return Err(RtErr::invalid_bin(a, op, b)),
        };
        Ok(out.into_value(ctx))
    }

    /// `-x` / `+x`.
    pub fn un<'gc>(ctx: Ctx<'gc>, a: Val<'gc>, op: UnaryOp) -> RtResult<Val<'gc>> {
        let Ok(x) = Interval::from_value(ctx, a) else {
            return Err(RtErr::InvalidUnaryOperand);
        };
        let out = match op {
            UnaryOp::Negative => Interval { lo: -x.hi, mid: -x.mid, hi: -x.lo },
            UnaryOp::Positive => x,
            _ => return Err(RtErr::InvalidUnaryOperand),
        };
        Ok(out.into_value(ctx))
    }
}

/// Register `Interval`, its constructors, methods and operators.
pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_adt::<Interval>();
    api.add_assoc_of::<Interval, _, _>("pm", iv::pm);
    api.add_assoc_of::<Interval, _, _>("within", iv::within);
    api.add_assoc_of::<Interval, _, _>("span", iv::span);
    api.add_assoc_of::<Interval, _, _>("of", iv::of);
    api.add_assoc_of::<Interval, _, _>("exact", iv::exact);
    api.add_method(iv::width);
    api.add_method(iv::contains);
    api.add_method(iv::sample);
    api.add_method(iv::pow);
    api.add_method(iv::overlaps);
    api.add_bin_op::<Interval, _>(iv::bin);
    api.add_unary_op::<Interval, _>(iv::un);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_propagates_bounds() {
        let a = Interval { lo: 9.0, mid: 10.0, hi: 11.0 };
        let b = Interval { lo: 1.0, mid: 2.0, hi: 3.0 };
        let s = a.add(b);
        assert_eq!((s.lo, s.mid, s.hi), (10.0, 12.0, 14.0));
        let d = a.sub(b);
        assert_eq!((d.lo, d.mid, d.hi), (6.0, 8.0, 10.0));
        let p = a.mul(b);
        assert_eq!((p.lo, p.mid, p.hi), (9.0, 20.0, 33.0));
        let q = a.div(b);
        assert_eq!((q.lo, q.hi), (3.0, 11.0));
        assert_eq!(q.mid, 5.0);
    }

    #[test]
    fn signs_and_zero_divisors() {
        let a = Interval { lo: -2.0, mid: 1.0, hi: 3.0 };
        let p = a.mul(a);
        assert_eq!((p.lo, p.hi), (-6.0, 9.0));
        let z = Interval { lo: -1.0, mid: 1.0, hi: 1.0 };
        let q = Interval::point(1.0).div(z);
        assert_eq!((q.lo, q.hi), (f64::NEG_INFINITY, f64::INFINITY));
    }

    #[test]
    fn sampling_stays_in_range_and_centres_on_the_mode() {
        let a = Interval { lo: 0.0, mid: 2.0, hi: 10.0 };
        assert_eq!(a.sample(0.0), 0.0);
        assert_eq!(a.sample(1.0), 10.0);
        assert!((a.sample(0.2) - 2.0).abs() < 1e-9);
        for i in 0..=20 {
            let v = a.sample(i as f64 / 20.0);
            assert!((0.0..=10.0).contains(&v));
        }
    }
}
