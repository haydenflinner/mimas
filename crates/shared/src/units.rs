//! Units of measure: the table, and the dimension algebra behind compile-time dimensional analysis.
//!
//! A quantity is an ordinary `float` whose value is kept in coherent base units (`25kW` is the
//! float `25000.0`, watts), so nothing changes at runtime. What changes is the checker: it tracks
//! each quantity's [`Dim`] -- an exponent for every base dimension -- and refuses arithmetic that
//! doesn't add up (`5kW + 3s`), while letting dimensionless numbers scale anything.
//!
//! Base dimensions are the seven SI ones plus a currency (`usd`) and two for graphics (`px`,
//! `frame`), so `px/frame` (a speed in pixels per frame) is its own thing rather than a length
//! over a time. Every unit name in [`lookup`] carries a dimension and a scale to the base
//! unit, which is how `25kW`, `25000W` and `25000` all mean the same watts.

/// Number of base dimensions.
pub const N: usize = 10;

/// The base dimensions, in [`Dim`] exponent order.
pub const BASES: [&str; N] = ["m", "kg", "s", "A", "K", "mol", "cd", "usd", "px", "frame"];

/// An exponent per base dimension. All zeros is dimensionless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dim(pub [i8; N]);

impl Dim {
    pub const NONE: Dim = Dim([0; N]);

    pub fn is_none(&self) -> bool {
        *self == Self::NONE
    }

    pub fn mul(self, other: Dim) -> Dim {
        let mut out = self.0;
        for (o, b) in out.iter_mut().zip(other.0) {
            *o += b;
        }
        Dim(out)
    }

    pub fn div(self, other: Dim) -> Dim {
        let mut out = self.0;
        for (o, b) in out.iter_mut().zip(other.0) {
            *o -= b;
        }
        Dim(out)
    }

    /// `self ^ n`.
    pub fn powi(self, n: i8) -> Dim {
        Dim(self.0.map(|e| e * n))
    }

    /// The square root, when every exponent is even.
    pub fn sqrt(self) -> Option<Dim> {
        self.0
            .iter()
            .all(|e| e % 2 == 0)
            .then(|| Dim(self.0.map(|e| e / 2)))
    }

    /// A reader-friendly name: a familiar unit when one fits (`kW`-ish things say `power (W)`),
    /// else the base composition (`kg*m^2/s^3`).
    pub fn describe(&self) -> String {
        if self.is_none() {
            return "a plain number".into();
        }
        for (name, dim, scale) in NAMED {
            if dim == *self && scale == 1.0 {
                return format!("{} ({})", family(*self), name);
            }
        }
        // a ratio or product of two familiar units reads better than raw base exponents:
        // money per energy is `usd/J`, not `s^2*usd/m^2*kg`
        for (an, ad, _) in NAMED.iter() {
            for (bn, bd, _) in NAMED.iter() {
                if ad.div(*bd) == *self && !bd.is_none() {
                    return format!("{an}/{bn}");
                }
            }
        }
        for (an, ad, _) in NAMED.iter() {
            for (bn, bd, _) in NAMED.iter() {
                if ad.mul(*bd) == *self && an <= bn {
                    return format!("{an}*{bn}");
                }
            }
        }
        let part = |pos: bool| {
            let mut s = Vec::new();
            for (i, e) in self.0.iter().enumerate() {
                let e = if pos { *e } else { -*e };
                if e > 0 {
                    s.push(if e == 1 {
                        BASES[i].to_string()
                    } else {
                        format!("{}^{e}", BASES[i])
                    });
                }
            }
            s.join("*")
        };
        match (part(true), part(false)) {
            (n, d) if d.is_empty() => n,
            (n, d) if n.is_empty() => format!("1/{d}"),
            (n, d) => format!("{n}/{d}"),
        }
    }
}

/// The word for a dimension, when it has one.
fn family(dim: Dim) -> &'static str {
    match SI.iter().find(|(_, d)| *d == dim) {
        Some((f, _)) => f,
        None => "quantity",
    }
}

const fn d(e: [i8; N]) -> Dim {
    Dim(e)
}

// exponents:        m  kg  s  A  K mol cd usd px frame
const LENGTH: Dim = d([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
const MASS: Dim = d([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
const TIME: Dim = d([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
const CURRENT: Dim = d([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
const TEMP: Dim = d([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
const AMOUNT: Dim = d([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
const LUMINOUS: Dim = d([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
const MONEY: Dim = d([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]);
const PIXELS: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
const FRAMES: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
const FREQ: Dim = d([0, 0, -1, 0, 0, 0, 0, 0, 0, 0]);
const FORCE: Dim = d([1, 1, -2, 0, 0, 0, 0, 0, 0, 0]);
const PRESSURE: Dim = d([-1, 1, -2, 0, 0, 0, 0, 0, 0, 0]);
const ENERGY: Dim = d([2, 1, -2, 0, 0, 0, 0, 0, 0, 0]);
const POWER: Dim = d([2, 1, -3, 0, 0, 0, 0, 0, 0, 0]);
const CHARGE: Dim = d([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
const VOLTAGE: Dim = d([2, 1, -3, -1, 0, 0, 0, 0, 0, 0]);

/// Dimension families, for error messages.
const SI: [(&str, Dim); 17] = [
    ("length", LENGTH),
    ("mass", MASS),
    ("time", TIME),
    ("current", CURRENT),
    ("temperature", TEMP),
    ("amount", AMOUNT),
    ("luminous intensity", LUMINOUS),
    ("money", MONEY),
    ("pixels", PIXELS),
    ("frames", FRAMES),
    ("frequency", FREQ),
    ("force", FORCE),
    ("pressure", PRESSURE),
    ("energy", ENERGY),
    ("power", POWER),
    ("charge", CHARGE),
    ("voltage", VOLTAGE),
];

/// The coherent unit each family is kept in.
const NAMED: [(&str, Dim, f64); 17] = [
    ("m", LENGTH, 1.0),
    ("kg", MASS, 1.0),
    ("s", TIME, 1.0),
    ("A", CURRENT, 1.0),
    ("K", TEMP, 1.0),
    ("mol", AMOUNT, 1.0),
    ("cd", LUMINOUS, 1.0),
    ("usd", MONEY, 1.0),
    ("px", PIXELS, 1.0),
    ("frame", FRAMES, 1.0),
    ("Hz", FREQ, 1.0),
    ("N", FORCE, 1.0),
    ("Pa", PRESSURE, 1.0),
    ("J", ENERGY, 1.0),
    ("W", POWER, 1.0),
    ("C", CHARGE, 1.0),
    ("V", VOLTAGE, 1.0),
];

/// Every unit name: (dimension, scale to the coherent base unit).
pub fn lookup(name: &str) -> Option<(Dim, f64)> {
    const NONE: Dim = Dim::NONE;
    Some(match name {
        // length
        "m" => (LENGTH, 1.0),
        "km" => (LENGTH, 1e3),
        "cm" => (LENGTH, 1e-2),
        "mm" => (LENGTH, 1e-3),
        "inch" => (LENGTH, 0.0254),
        "ft" => (LENGTH, 0.3048),
        "mi" => (LENGTH, 1609.344),
        // mass
        "kg" => (MASS, 1.0),
        "g" => (MASS, 1e-3),
        "mg" => (MASS, 1e-6),
        "lb" => (MASS, 0.453_592_37),
        "tonne" => (MASS, 1e3),
        // time
        "s" => (TIME, 1.0),
        "ms" => (TIME, 1e-3),
        "us" => (TIME, 1e-6),
        "min" => (TIME, 60.0),
        "h" => (TIME, 3600.0),
        "d" => (TIME, 86_400.0),
        "wk" => (TIME, 604_800.0),
        "yr" => (TIME, 31_557_600.0),
        // electricity, heat, amount, light
        "A" => (CURRENT, 1.0),
        "mA" => (CURRENT, 1e-3),
        "K" => (TEMP, 1.0),
        "mol" => (AMOUNT, 1.0),
        "cd" => (LUMINOUS, 1.0),
        "V" => (VOLTAGE, 1.0),
        "kV" => (VOLTAGE, 1e3),
        "C" => (CHARGE, 1.0),
        // mechanics and energy
        "Hz" => (FREQ, 1.0),
        "kHz" => (FREQ, 1e3),
        "N" => (FORCE, 1.0),
        "kN" => (FORCE, 1e3),
        "Pa" => (PRESSURE, 1.0),
        "kPa" => (PRESSURE, 1e3),
        "J" => (ENERGY, 1.0),
        "kJ" => (ENERGY, 1e3),
        "MJ" => (ENERGY, 1e6),
        "Wh" => (ENERGY, 3600.0),
        "kWh" => (ENERGY, 3.6e6),
        "MWh" => (ENERGY, 3.6e9),
        "W" => (POWER, 1.0),
        "mW" => (POWER, 1e-3),
        "kW" => (POWER, 1e3),
        "MW" => (POWER, 1e6),
        // money and graphics: their own base dimensions
        "usd" => (MONEY, 1.0),
        "px" => (PIXELS, 1.0),
        "frame" => (FRAMES, 1.0),
        // dimensionless scales: `90deg`, `6pct` are plain floats
        "rad" => (NONE, 1.0),
        "deg" => (NONE, std::f64::consts::PI / 180.0),
        "pct" => (NONE, 0.01),
        _ => return None,
    })
}

/// A unit expression like `usd/kWh`, `m/s^2` or `W*h`: names joined by `*` and `/`, each with an
/// optional `^n`. Multiplication and division associate left to right.
pub fn parse(expr: &str) -> Option<(Dim, f64)> {
    let mut dim = Dim::NONE;
    let mut scale = 1.0;
    let mut rest = expr;
    let mut op = '*';
    loop {
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let (name, tail) = rest.split_at(end);
        let (d, s) = lookup(name)?;
        let (exp, tail) = match tail.strip_prefix('^') {
            Some(t) => {
                let neg = t.starts_with('-');
                let t = t.strip_prefix('-').unwrap_or(t);
                let digits = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
                let n: i8 = t[..digits].parse().ok()?;
                (if neg { -n } else { n }, &t[digits..])
            }
            None => (1, tail),
        };
        let (d, s) = (d.powi(exp), s.powi(exp as i32));
        if op == '*' {
            dim = dim.mul(d);
            scale *= s;
        } else {
            dim = dim.div(d);
            scale /= s;
        }
        match tail.chars().next() {
            None => return Some((dim, scale)),
            Some(c @ ('*' | '/')) => {
                op = c;
                rest = &tail[1..];
            }
            Some(_) => return None,
        }
    }
}

/// The length of the unit suffix at the start of `rest` (what follows a number), if it opens
/// with a unit: a unit name, then any `/name` continuations and `^n` exponents. A name only
/// counts when it is *exactly* a unit, so `5kWhx` and `3x` stay the plain errors they were.
/// Only `/` continues a literal's unit (a `*` would swallow `3m*s` where `s` is a variable).
pub fn scan_suffix(rest: &str) -> Option<usize> {
    fn ident(s: &str) -> usize {
        s.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(s.len())
    }
    fn exponent(s: &str) -> usize {
        let Some(t) = s.strip_prefix('^') else {
            return 0;
        };
        let neg = usize::from(t.starts_with('-'));
        let digits = t[neg..].chars().take_while(char::is_ascii_digit).count();
        if digits == 0 { 0 } else { 1 + neg + digits }
    }
    if !rest.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut end = ident(rest);
    lookup(&rest[..end])?;
    end += exponent(&rest[end..]);
    while let Some(next) = rest[end..].strip_prefix('/') {
        let n = ident(next);
        if !next.starts_with(|c: char| c.is_ascii_alphabetic()) || lookup(&next[..n]).is_none() {
            break;
        }
        end += 1 + n;
        end += exponent(&rest[end..]);
    }
    Some(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_carry_dimension_and_scale() {
        let (d, s) = lookup("kW").unwrap();
        assert_eq!(d, POWER);
        assert_eq!(s, 1e3);
        assert_eq!(lookup("kWh").unwrap().0, POWER.mul(TIME));
        assert!(lookup("bogus").is_none());
    }

    #[test]
    fn expressions_combine() {
        // a tariff: dollars per kilowatt-hour, in dollars per joule
        let (d, s) = parse("usd/kWh").unwrap();
        assert_eq!(d, MONEY.div(ENERGY));
        assert!((s - 1.0 / 3.6e6).abs() < 1e-18);
        let (d, s) = parse("m/s^2").unwrap();
        assert_eq!(d, LENGTH.div(TIME.powi(2)));
        assert_eq!(s, 1.0);
        // W*h is energy: watt-hours
        assert_eq!(parse("W*h").unwrap(), (ENERGY, 3600.0));
        assert!(parse("m/bogus").is_none());
    }

    #[test]
    fn suffixes_stop_where_units_stop() {
        assert_eq!(scan_suffix("kW"), Some(2));
        assert_eq!(scan_suffix("kW + 1"), Some(2));
        assert_eq!(scan_suffix("usd/kWh)"), Some(7));
        assert_eq!(scan_suffix("m/s^2 "), Some(5));
        // `/x` isn't a unit, so the division stays a division
        assert_eq!(scan_suffix("W/x"), Some(1));
        assert_eq!(scan_suffix("W/2"), Some(1));
        assert_eq!(scan_suffix("kWhx"), None);
        assert_eq!(scan_suffix("x"), None);
        assert_eq!(scan_suffix("_m"), None);
    }

    #[test]
    fn describing_dimensions() {
        assert_eq!(POWER.describe(), "power (W)");
        assert_eq!(Dim::NONE.describe(), "a plain number");
        assert_eq!(POWER.mul(TIME).describe(), "energy (J)");
        // a tariff reads as a ratio of familiar things
        assert_eq!(MONEY.div(ENERGY).describe(), "usd/J");
        assert_eq!(LENGTH.div(TIME).describe(), "m/s");
    }

    #[test]
    fn square_roots_need_even_exponents() {
        assert_eq!(LENGTH.powi(2).sqrt(), Some(LENGTH));
        assert_eq!(LENGTH.sqrt(), None);
    }
}
