//! Units of measure: the table, and the dimension algebra behind compile-time dimensional analysis.
//!
//! A quantity is an ordinary `float` whose value is kept in coherent base units (`25kW` is the
//! float `25000.0`, watts), so nothing changes at runtime. What changes is the checker: it tracks
//! each quantity's [`Dim`] -- an exponent for every base dimension -- and refuses arithmetic that
//! doesn't add up (`5kW + 3s`), while letting dimensionless numbers scale anything.
//!
//! Base dimensions are the seven SI ones plus a currency (`usd`), two for graphics (`px`,
//! `frame`), and three for the studio: `beat` (musical time, convertible to `s` only via a
//! tempo), `st` (semitones — pitch), and `b` (bits — data). `px/frame` (a speed in pixels per
//! frame) is its own thing rather than a length over a time. Every unit name in [`lookup`]
//! carries a dimension and a scale to the base unit, which is how `25kW`, `25000W` and `25000`
//! all mean the same watts.

/// Number of slots in a [`Dim`]: the base dimensions (builtins, then `unit`
/// decls), then the axis parity slots.
pub const N: usize = AXIS0 + MAX_AXES;

/// Where the axis slots start. An axis slot is a parity, not an exponent: a
/// quantity *along x* flips sign when x is mirrored, and a product of two
/// such quantities doesn't -- so `*`/`/` xor the axis slots where they add
/// the exponent slots. See [`Dim::axes`].
pub const AXIS0: usize = FIRST_CUSTOM + MAX_CUSTOM;

/// Every slot that carries a unit exponent — builtin bases and `unit`
/// decls alike, i.e. everything before the axis slots. Loops over a
/// dim's "bare" part want this bound: `0..FIRST_CUSTOM` silently skips
/// declared units.
pub const UNIT_SLOTS: usize = AXIS0;

/// Axis slots: the four builtin axes (`x` `y` `z` `t`), four for `axis`
/// decls, and two norm slots (`snorm`/`unorm`) at the end. Norm slots are
/// parity-only namespaces — no direction words, not declarable — so a
/// normalized channel gets the axis algebra (an untagged number adopts,
/// two norms won't add) without spending a decl.
pub const MAX_AXES: usize = 10;
const BUILTIN_AXES: usize = 4;

/// Where the norm slots start: every slot below is an axis a program may
/// declare or a builtin axis; slots at and above are fixed namespaces.
const NORM0: usize = 8;
/// The norm slots, by name and parity index: `snorm` a signed `[-1, 1]`
/// channel, `unorm` an unsigned `[0, 1]` one — the GPU format spellings.
const NORMS: [(&str, usize); 2] = [("snorm", NORM0), ("unorm", NORM0 + 1)];

/// Where a `unit tile;` declaration's base dimension sits in the [`Dim`]
/// array -- the builtins occupy the slots before it.
const FIRST_CUSTOM: usize = 17;

/// How many new base dimensions (`unit tile;`) one program may declare.
/// Derived decls (`unit league = 3mi;`) don't spend a slot -- they share
/// the dimension of the unit they're scaled from.
pub const MAX_CUSTOM: usize = 8;

/// The base dimensions, in [`Dim`] exponent order. The custom slots name
/// themselves through [`declare`]; the empty strings are the fallback for
/// a `Dim` that outlives its compilation.
pub const BASES: [&str; N] = [
    "m", "kg", "s", "A", "K", "mol", "cd", "usd", "px", "frame", "beat", "st", "b", "index", "rad",
    "spx", "wpx",
    "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "",
    "", "",
];

/// An exponent per base dimension. All zeros is dimensionless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim(pub [i8; N]);

// `[i8; 33]` outgrew the array `Default` impls — `NONE` is the default.
impl Default for Dim {
    fn default() -> Self {
        Self::NONE
    }
}

impl Dim {
    pub const NONE: Dim = Dim([0; N]);

    pub fn is_none(&self) -> bool {
        *self == Self::NONE
    }

    pub const fn mul(self, other: Dim) -> Dim {
        let mut out = self.0;
        let mut i = 0;
        while i < AXIS0 {
            out[i] += other.0[i];
            i += 1;
        }
        // axis slots are parities: along x times along x is no longer along x,
        // whatever the convention — `x→·x←` cancels like `x↔·x↔` does (the
        // value's sign correction is the checker's, not the dim's)
        while i < N {
            let (a, b) = (self.0[i], other.0[i]);
            out[i] = if a == 0 || b == 0 {
                a + b
            } else {
                0
            };
            i += 1;
        }
        Dim(out)
    }

    pub const fn div(self, other: Dim) -> Dim {
        let mut out = self.0;
        let mut i = 0;
        while i < AXIS0 {
            out[i] -= other.0[i];
            i += 1;
        }
        while i < N {
            let (a, b) = (self.0[i], other.0[i]);
            out[i] = if a == 0 || b == 0 {
                a + b
            } else {
                0
            };
            i += 1;
        }
        Dim(out)
    }

    /// `self ^ n`.
    pub const fn powi(self, n: i8) -> Dim {
        let mut out = self.0;
        let mut i = 0;
        while i < AXIS0 {
            out[i] *= n;
            i += 1;
        }
        while i < N {
            out[i] = if n % 2 == 0 { 0 } else { out[i] };
            i += 1;
        }
        Dim(out)
    }

    /// The square root, when every exponent is even (and nothing is along an axis).
    pub const fn sqrt(self) -> Option<Dim> {
        let mut out = self.0;
        let mut i = 0;
        while i < AXIS0 {
            if self.0[i] % 2 != 0 {
                return None;
            }
            out[i] = self.0[i] / 2;
            i += 1;
        }
        // an axis tag isn't an exponent — it can't halve; only an absent one roots
        while i < N {
            if self.0[i] == 1 {
                return None;
            }
            i += 1;
        }
        Some(Dim(out))
    }

    /// The axis slots as a bitmask: which axes this quantity is along (odd parity).
    pub const fn axes(&self) -> u16 {
        let mut bits = 0u16;
        let mut i = 0;
        while i < MAX_AXES {
            if self.0[AXIS0 + i] != 0 {
                bits |= 1 << i;
            }
            i += 1;
        }
        bits
    }

    /// This dimension with no axis: the exponents alone.
    pub const fn bare(&self) -> Dim {
        let mut out = self.0;
        let mut i = AXIS0;
        while i < N {
            out[i] = 0;
            i += 1;
        }
        Dim(out)
    }

    /// A dimensionless quantity along one axis slot -- what a direction word tags.
    /// Slot values: 0 absent, 1 an unsigned axis tag (`↔`/`↕`), +2 the positive
    /// end (`→`/`↑`), −2 the negative end (`←`/`↓`).
    pub const fn axis(slot: usize) -> Dim {
        let mut e = [0i8; N];
        e[AXIS0 + slot] = 1;
        Dim(e)
    }

    /// The signed end of an axis slot: `→`/`↑`/`⊗`/`⏩` are +2, `←`/`↓`/`⊙`/`⏪`
    /// are −2. A direction *claim* on a type — `wpx↓` is world y measured
    /// down-positive — where `axis()`'s unsigned tag accepts either
    /// convention. Opposite-signed scalars convert by negating the value;
    /// unsigned and absent sides still adopt.
    // r[impl mimas.unit.direction.signed]
    pub const fn axis_dir(slot: usize, negative: bool) -> Dim {
        let mut e = [0i8; N];
        e[AXIS0 + slot] = if negative { -2 } else { 2 };
        Dim(e)
    }

    /// Two dims that agree except for opposite axis signs — `wpx↑` vs `wpx↓`
    /// are conventions of the same quantity, convertible by negating a
    /// scalar value. Composite containers and places can't convert
    /// element-wise; the caller decides which it holds.
    pub fn sign_convertible(want: Dim, got: Dim) -> bool {
        let axes = want.axes() == 0
            || got.axes() == 0
            || (want.axes() == got.axes()
                && (0..MAX_AXES).all(|i| {
                    let (w, g) = (want.0[AXIS0 + i], got.0[AXIS0 + i]);
                    w == 1 || g == 1 || w.abs() == g.abs()
                }));
        axes && (want.bare() == got.bare() || pixel_agree(want.bare(), got.bare()))
    }

    /// `a` holds the negative end of a slot `b` holds positive — `a` is
    /// the side a conversion negates. The pair's opposition is
    /// symmetric, but the flip belongs to `a`: `sign_opposed(a, b) ||
    /// sign_opposed(b, a)` answers "do these differ by convention".
    pub fn sign_opposed(a: Dim, b: Dim) -> bool {
        (0..MAX_AXES).any(|i| {
            let (a, b) = (a.0[AXIS0 + i], b.0[AXIS0 + i]);
            a == -2 && b == 2
        })
    }

    /// `got`'s value must negate to land in `want`'s convention: a
    /// value is stored reading its positive end wherever the slot
    /// doesn't claim the negative one, so the sign flips iff exactly
    /// one side claims it — `40↓` under `wpx`, `wpx↕` or `wpx↑` is −40;
    /// an unsigned value into a `wpx↓` slot is too.
    pub fn sign_flip(want: Dim, got: Dim) -> bool {
        (0..MAX_AXES).any(|i| {
            let (w, g) = (want.0[AXIS0 + i], got.0[AXIS0 + i]);
            (w == -2) != (g == -2)
        })
    }

    /// `.to("px↓")` re-expresses the value in the named convention, so
    /// the number negates when the source reads the other end. A bare
    /// or unsigned target (`px`, `px↕`) names no end — `.to` returns
    /// the scalar in the value's own convention as stored.
    pub fn read_flip(want: Dim, got: Dim) -> bool {
        (0..MAX_AXES).any(|i| {
            let (w, g) = (want.0[AXIS0 + i], got.0[AXIS0 + i]);
            (w == 2 || w == -2) && ((w == -2) != (g == -2))
        })
    }

    /// Whether any slot carries the negative end — the factor a product
    /// or quotient normalizes before composing.
    pub fn has_neg_end(self) -> bool {
        (0..MAX_AXES).any(|i| self.0[AXIS0 + i] == -2)
    }

    /// This dim with every signed axis read at its positive end — the
    /// convention mixed `x↓ + y↑` arithmetic reports its result in.
    pub fn pos_axes(self) -> Dim {
        let mut out = self;
        for e in &mut out.0[AXIS0..] {
            if *e == -2 {
                *e = 2;
            }
        }
        out
    }

    /// This dim's axis claims written over `base`'s — the direction
    /// suffix's merge: `x↓` claims the y convention whatever the operand
    /// already carried there; an axis the suffix doesn't name is kept.
    pub fn with_axes(self, base: Dim) -> Dim {
        let mut out = base;
        for i in 0..MAX_AXES {
            if self.0[AXIS0 + i] != 0 {
                out.0[AXIS0 + i] = self.0[AXIS0 + i];
            }
        }
        out
    }

    /// Whether two dimensions can meet in `+`, `-`, a comparison or a slot:
    /// the exponents must match — except inside the pixel family, where a
    /// bare `px` is *unspecified* and stands for any specific pixel space —
    /// and the axes must match unless one side has none: an untagged
    /// quantity adopts the other's axis (`b.x + BALL`), the way a plain
    /// number meets an `index`.
    // r[impl mimas.unit.axis]
    // r[impl mimas.unit.pixels]
    pub fn agree(a: Dim, b: Dim) -> bool {
        (a.bare() == b.bare() || pixel_agree(a.bare(), b.bare()))
            && (a.axes() == 0 || b.axes() == 0 || (a.axes() == b.axes() && Self::signs_ok(a, b)))
    }

    /// Per-slot convention match under equal axis sets: same tag, or an
    /// unsigned `↔`/`↕` adopting a signed `→`/`←`. Opposite signs disagree —
    /// a scalar converts instead (`sign_convertible`), a composite clashes.
    fn signs_ok(a: Dim, b: Dim) -> bool {
        (0..MAX_AXES).all(|i| {
            let (a, b) = (a.0[AXIS0 + i], b.0[AXIS0 + i]);
            a == b || a == 1 || b == 1
        })
    }

    /// The dimension two agreeing values share: the tagged one's — and the
    /// specific pixel space's when the other side's pixels are unspecified.
    /// An unsigned axis adopts the signed side's convention.
    pub fn absorb(a: Dim, b: Dim) -> Dim {
        let bare = if a.bare() == b.bare() || specific_px_mass(a.bare()) != 0 {
            a.bare()
        } else {
            b.bare()
        };
        let mut out = bare;
        // agreeing axes are equal or one-sided, so the union is the
        // adopted set — an unsigned slot takes the signed side's end
        let mut i = 0;
        while i < MAX_AXES {
            let (a, b) = (a.0[AXIS0 + i], b.0[AXIS0 + i]);
            out.0[AXIS0 + i] = if b != 0 && (a == 0 || a == 1) { b } else { a };
            i += 1;
        }
        out
    }

    /// `x`, `y`, `x·y` -- the axes this is along, by name. A signed slot
    /// names its end: `y↑` reads "up-positive y".
    pub fn along(&self) -> String {
        (0..MAX_AXES)
            .filter(|i| self.0[AXIS0 + i] != 0)
            .map(|i| axis_dir_name(i, self.0[AXIS0 + i]))
            .collect::<Vec<_>>()
            .join("·")
    }

    /// The suffix a type annotation writes for the axes: `↔`, `↕`, `→`,
    /// `↓`, or `@name`/`@dir`.
    fn axis_suffix(&self) -> String {
        (0..MAX_AXES)
            .filter(|i| self.0[AXIS0 + i] != 0)
            .map(|i| axis_dir_suffix(i, self.0[AXIS0 + i]))
            .collect()
    }

    /// The base dimension a `unit name;` declaration claims -- slot
    /// `FIRST_CUSTOM + slot` alone, exponent 1.
    pub const fn custom(slot: usize) -> Dim {
        let mut e = [0i8; N];
        e[FIRST_CUSTOM + slot] = 1;
        Dim(e)
    }

    /// For a pure custom base dim (`Dim::custom(3)`), the declared slot;
    /// `None` for builtin dims and composites.
    fn custom_slot(&self) -> Option<usize> {
        let mut slot = None;
        for (i, e) in self.0.iter().enumerate().take(AXIS0) {
            match (*e, i >= FIRST_CUSTOM) {
                (0, _) => {}
                (1, true) if slot.is_none() => slot = Some(i - FIRST_CUSTOM),
                _ => return None,
            }
        }
        slot
    }

    /// The declared name of a pure custom base dim (`tile` for
    /// `Dim::custom(0)` after `unit tile;`), else `None`.
    fn custom_name(&self) -> Option<String> {
        self.custom_slot().and_then(custom_name)
    }

    /// The unit name for a type annotation — `px`, `px/s`, `kg*m^2/s^3` —
    /// unlike `describe`, which answers in prose (`pixels (px)`).
    pub fn name(&self) -> String {
        if self.axes() != 0 {
            return format!("{}{}", self.bare().name(), self.axis_suffix());
        }
        if self.is_none() {
            return "float".into();
        }
        for (name, dim, scale) in NAMED {
            if dim == *self && scale == 1.0 {
                return name.to_string();
            }
        }
        let part = |pos: bool| {
            let mut s = Vec::new();
            for (i, e) in self.0.iter().enumerate() {
                let e = if pos { *e } else { -*e };
                if e > 0 {
                    s.push(if e == 1 {
                        base_name(i)
                    } else {
                        format!("{}^{e}", base_name(i))
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

    /// A reader-friendly name: a familiar unit when one fits (`kW`-ish things say `power (W)`),
    /// else the base composition (`kg*m^2/s^3`).
    pub fn describe(&self) -> String {
        if self.axes() != 0 {
            return format!("{} along {}", self.bare().describe(), self.along());
        }
        if self.is_none() {
            return "a plain number".into();
        }
        // `unit tile;` -- the declared name is the whole story
        if let Some(name) = self.custom_name() {
            return name;
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
                    // a unit times itself is its square: `px²`, not `px*px`
                    return if an == bn { format!("{an}²") } else { format!("{an}*{bn}") };
                }
            }
        }
        let part = |pos: bool| {
            let mut s = Vec::new();
            for (i, e) in self.0.iter().enumerate() {
                let e = if pos { *e } else { -*e };
                if e > 0 {
                    s.push(if e == 1 {
                        base_name(i)
                    } else {
                        format!("{}^{e}", base_name(i))
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

    /// Like [`Self::describe`], but names the unit the way it was written (`length (ft)`
    /// for `5ft`) instead of the dimension's coherent unit (`length (m)`).
    pub fn describe_unit(&self, unit: &str) -> String {
        if self.axes() != 0 {
            return format!("{} along {}", self.bare().describe_unit(unit), self.along());
        }
        if self.is_none() {
            return "a plain number".into();
        }
        // `18tile` -- a custom base has no family name to add
        if self.custom_name().is_some() {
            return unit.to_string();
        }
        format!("{} ({unit})", family(*self))
    }
}

/// A dimension claim that may nest inside a container parameter: `Vec<Px>`
/// checks as `[px]`, `Vec<(PxX, PxY)>` as `(px↔, px↕)` element-wise. Native
/// signatures carry these in `param_dims`; the solver's `D` mirrors the same
/// shape for values it infers itself.
// r[impl mimas.unit.nested]
#[derive(Debug, Clone, PartialEq)]
pub enum DimTree {
    /// A measured quantity in this dimension (`Dim::NONE` = a plain number).
    Q(Dim),
    /// A list whose elements each satisfy this tree.
    Arr(Box<DimTree>),
    /// A tuple whose elements each satisfy the tree at their index.
    Tup(Vec<DimTree>),
}

impl DimTree {
    /// The unit spelling for a signature dump — `px`, `[px]`, `(px↔, px↕)`.
    pub fn name(&self) -> String {
        match self {
            DimTree::Q(d) => d.name(),
            DimTree::Arr(inner) => format!("[{}]", inner.name()),
            DimTree::Tup(parts) => format!(
                "({})",
                parts.iter().map(DimTree::name).collect::<Vec<_>>().join(", ")
            ),
        }
    }

    /// The scalar `Dim` this tree asserts, when it is one.
    pub fn scalar(&self) -> Option<Dim> {
        match self {
            DimTree::Q(d) => Some(*d),
            _ => None,
        }
    }
}

/// How an echo shows a quantity: in the unit the program wrote for its
/// dimension (`12lb` echoes `12 lb`, not `5.44 kg`), divided by `scale` (the
/// unit's size in base units). `list` marks a list of such quantities, which
/// echoes as `[12, 8, 3] lb`. r[impl mimas.echo.unit]
#[derive(Debug, Clone, PartialEq)]
pub struct Shown {
    pub name: String,
    pub scale: f64,
    pub list: bool,
}

/// The static shape of a displayed value, where the runtime value cannot
/// tell it: a tuple is a list at runtime, and a float does not carry its
/// unit. The dims pass computes one per display site (`print`, an echo, an
/// f-string hole) from the value's type and dimension; the renderer
/// (`Ctx::render_shaped`) follows it. `Plain` is what the value alone says.
/// r[impl mimas.display.one]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Shape {
    #[default]
    Plain,
    /// A quantity in this dimension: `3 px`.
    Q(Dim),
    /// A list whose items have this shape. A list of quantities shows its
    /// unit once, after the list: `[12, 8] lb`.
    List(Box<Shape>),
    /// A tuple, one shape per item: `("a", 1)`, `(2 px, 3 px)`.
    Tuple(Vec<Shape>),
    /// A dict whose values have this shape.
    Dict(Box<Shape>),
}

impl Shape {
    /// Every dimension the shape names, appended to `out`.
    pub fn dims_into(&self, out: &mut Vec<Dim>) {
        match self {
            Shape::Plain => {}
            Shape::Q(d) => out.push(*d),
            Shape::List(s) | Shape::Dict(s) => s.dims_into(out),
            Shape::Tuple(ss) => ss.iter().for_each(|s| s.dims_into(out)),
        }
    }

    /// The shape of a struct field declared as `ty` whose numbers carry `dim`
    /// (a field's declared unit reaches through `?` and `[..]`).
    pub fn of_field(ty: &crate::Ty, dim: Option<Dim>) -> Shape {
        use crate::Ty;
        match ty {
            Ty::Option(inner) => Shape::of_field(inner, dim),
            Ty::Array(inner, _) => match Shape::of_field(inner, dim) {
                Shape::Plain => Shape::Plain,
                s => Shape::List(Box::new(s)),
            },
            Ty::Dict(_, v) => match Shape::of_field(v, dim) {
                Shape::Plain => Shape::Plain,
                s => Shape::Dict(Box::new(s)),
            },
            Ty::Tuple(ts) => Shape::Tuple(ts.iter().map(|t| Shape::of_field(t, None)).collect()),
            Ty::Float | Ty::Int => match dim {
                Some(d) if !d.is_none() && d != INDEX => Shape::Q(d),
                _ => Shape::Plain,
            },
            _ => Shape::Plain,
        }
    }

    /// Whether this shape says anything the value alone does not.
    pub fn is_plain(&self) -> bool {
        match self {
            Shape::Plain => true,
            Shape::Q(_) | Shape::Tuple(_) => false,
            Shape::List(s) | Shape::Dict(s) => s.is_plain(),
        }
    }
}

/// A float as an echo writes it beside a unit: at most 12 significant
/// digits, trailing zeros dropped, so a value converted out of base units
/// (`12lb` round-trips as `12.000000000000002`) reads `12`.
pub fn shown_number(v: f64) -> String {
    if !v.is_finite() || v == 0.0 {
        return if v == 0.0 { "0".into() } else { format!("{v}") };
    }
    let mag = v.abs().log10().floor() as i32;
    let prec = (11 - mag).clamp(0, 20) as usize;
    let s = format!("{v:.prec$}");
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    if s == "-0" { "0".into() } else { s }
}

/// The word for a dimension, when it has one.
fn family(dim: Dim) -> &'static str {
    match SI.iter().find(|(_, d)| *d == dim) {
        Some((f, _)) => f,
        None => "quantity",
    }
}

/// The dim tables write just the builtin exponents — custom slots pad to 0.
const fn d(e: [i8; FIRST_CUSTOM]) -> Dim {
    let mut out = [0i8; N];
    let mut i = 0;
    while i < FIRST_CUSTOM {
        out[i] = e[i];
        i += 1;
    }
    Dim(out)
}

// exponents:        m  kg  s  A  K mol cd usd px frame beat st bit index rad spx wpx
pub const LENGTH: Dim = d([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const MASS: Dim = d([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const TIME: Dim = d([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const CURRENT: Dim = d([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const TEMP: Dim = d([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const AMOUNT: Dim = d([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const LUMINOUS: Dim = d([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const MONEY: Dim = d([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const PIXELS: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const FRAMES: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
pub const BEAT: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
pub const PITCH: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
pub const DATA: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
/// A position into a collection: `i: index` is an `int` that can't be mixed
/// with any quantity (`i + 3px` is an error). Plain numbers convert freely
/// (`i += 1`, `i < xs.len()`) -- the dimension exists to keep *measurements*
/// out of index variables, not to make counting verbose.
pub const INDEX: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
/// An angle, in radians: `deg`, `turn`, `grad`, `arcmin`, `arcsec` and `mrad`
/// are scaled spellings of it. Unlike `prob`/`pct`, an angle *does* compose —
/// `angle / s` is angular velocity and `rad^2` is a solid angle.
pub const ANGLE: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]);
pub const FREQ: Dim = d([0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const FORCE: Dim = d([1, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const PRESSURE: Dim = d([-1, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const ENERGY: Dim = d([2, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const POWER: Dim = d([2, 1, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const CHARGE: Dim = d([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
pub const VOLTAGE: Dim = d([2, 1, -3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
/// A screen-space pixel: a coordinate the host measured in its own space
/// (a browser point, a viewport extent, a sheet-plane position) and handed
/// the page. `spx` is a *specific* pixel space — it never meets `wpx` in
/// arithmetic or a typed slot; the page crosses deliberately through
/// `game::to_world`/`game::to_screen`. Bare `px` is the pixel family's
/// unspecified member: it meets any specific space and absorbs into it.
pub const SCREEN_PX: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
/// A world-space pixel: a coordinate in the page's own world — the field,
/// board or chart plane the camera looks at. `wpx` never meets `spx`;
/// `game::to_world`/`game::to_screen` own the seam. A render or
/// backbuffer pixel (`rpx`) would join the family as a fourth member if
/// a native ever hands raster coords across the boundary.
pub const WORLD_PX: Dim = d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);

/// The pixel units: `PIXELS` (`px`) first — the family's *unspecified*
/// member — then every specific pixel space. A render/backbuffer px
/// (`rpx`) joins here if a native ever hands raster coords across.
const PIXEL_UNITS: [Dim; 3] = [PIXELS, SCREEN_PX, WORLD_PX];

/// The base slot a singleton dim occupies (`PIXELS` → the `px` slot).
fn base_slot(u: Dim) -> usize {
    u.0[..FIRST_CUSTOM]
        .iter()
        .position(|&e| e != 0)
        .expect("a base unit")
}

/// The dim's exponent mass on the *specific* pixel spaces (`spx`, `wpx`),
/// ignoring `px` itself: `wpx/s` → 1, `px/s` → 0.
fn specific_px_mass(d: Dim) -> i8 {
    PIXEL_UNITS[1..]
        .iter()
        .map(|u| d.0[base_slot(*u)])
        .sum()
}

/// Pixel-family agreement for `Dim::agree`: non-pixel exponents must match
/// exactly; on the pixel slots, each side's *specific* mass must either
/// match the other's or be supplied entirely by the other's `px` mass.
/// So `px` meets `wpx` and `px/s` meets `wpx/s` — the unspecified unit
/// stands in for the space — while `wpx` and `spx` never meet, and a
/// `px` mass too small to cover the other's specifics (or any leftovers)
/// keeps the dims apart.
// r[impl mimas.unit.pixels]
fn pixel_agree(a: Dim, b: Dim) -> bool {
    let px_i = base_slot(PIXELS);
    let (mut pa, mut pb) = (0i8, 0i8);
    let mut spec_eq = true;
    let (mut sa, mut sb) = (0i8, 0i8);
    // every unit slot compares, custom units included — `tile` and
    // `elixir` differ above FIRST_CUSTOM and must not "agree" here
    for i in 0..UNIT_SLOTS {
        let (x, y) = (a.0[i], b.0[i]);
        if i == px_i {
            (pa, pb) = (x, y);
        } else if PIXEL_UNITS[1..].iter().any(|u| base_slot(*u) == i) {
            spec_eq &= x == y;
            sa += x;
            sb += y;
        } else if x != y {
            return false;
        }
    }
    if spec_eq {
        pa == pb
    } else {
        (sa == 0 && pa == pb + sb) || (sb == 0 && pb == pa + sa)
    }
}

/// Dimension families, for error messages.
const SI: [(&str, Dim); 25] = [
    ("length", LENGTH),
    ("volume", LENGTH.powi(3)),
    ("mass", MASS),
    ("time", TIME),
    ("current", CURRENT),
    ("temperature", TEMP),
    ("amount", AMOUNT),
    ("luminous intensity", LUMINOUS),
    ("money", MONEY),
    ("pixels", PIXELS),
    ("frames", FRAMES),
    ("beats", BEAT),
    ("pitch", PITCH),
    ("data", DATA),
    ("frequency", FREQ),
    ("force", FORCE),
    ("pressure", PRESSURE),
    ("energy", ENERGY),
    ("power", POWER),
    ("charge", CHARGE),
    ("voltage", VOLTAGE),
    ("index", INDEX),
    ("angle", ANGLE),
    ("screen pixels", SCREEN_PX),
    ("world pixels", WORLD_PX),
];

/// The coherent unit each family is kept in.
const NAMED: [(&str, Dim, f64); 24] = [
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
    ("b", DATA, 1.0),
    ("beat", BEAT, 1.0),
    ("st", PITCH, 1.0),
    ("Hz", FREQ, 1.0),
    ("N", FORCE, 1.0),
    ("Pa", PRESSURE, 1.0),
    ("J", ENERGY, 1.0),
    ("W", POWER, 1.0),
    ("C", CHARGE, 1.0),
    ("V", VOLTAGE, 1.0),
    ("idx", INDEX, 1.0),
    ("rad", ANGLE, 1.0),
    ("spx", SCREEN_PX, 1.0),
    ("wpx", WORLD_PX, 1.0),
];

/// The word for base-dimension slot `i` — builtins by name, custom slots
/// by their declaration (or a placeholder for a `Dim` read outside its
/// compilation).
fn base_name(i: usize) -> String {
    if i < FIRST_CUSTOM {
        BASES[i].to_string()
    } else {
        custom_name(i - FIRST_CUSTOM).unwrap_or_else(|| format!("unit{}", i - FIRST_CUSTOM))
    }
}

/// A program's `unit` declarations: `unit tile;` claims a fresh base
/// dimension; `unit league = 3mi;` names a scale of known units.
///
/// The table is thread-local and cleared when a new source starts
/// compiling ([`clear`] is called from `Parser::new`; the files of one
/// program parse under `Parser::new_extending` so they share it). A `Vm`
/// [`snapshot`]s it after solving and [`install`]s it before running, so
/// `.to("tile")` at runtime resolves even when another program compiled
/// in between. Nothing here is ambient across compilations.
#[derive(Clone, Debug, Default)]
pub struct CustomUnits {
    /// Every declared name -> (dimension, scale to coherent base).
    names: std::collections::HashMap<String, (Dim, f64)>,
    /// Declared base dims, in declaration order — entry `i` names the
    /// `Dim` slot `FIRST_CUSTOM + i`.
    bases: Vec<String>,
    /// `axis flow (out, in);` decls: the axis name and its direction
    /// words, negative first. Slot `BUILTIN_AXES + i`.
    axes: Vec<(String, String, String)>,
    /// A builtin axis redeclared with other words (`axis y (south, north);`):
    /// slot -> (negative, positive). The old words stop being words.
    renamed: std::collections::HashMap<usize, (String, String)>,
}

thread_local! {
    static CUSTOM: std::cell::RefCell<std::rc::Rc<CustomUnits>> =
        std::cell::RefCell::new(std::rc::Rc::new(CustomUnits::new()));
}

impl CustomUnits {
    fn new() -> Self {
        Self {
            names: std::collections::HashMap::new(),
            bases: Vec::new(),
            axes: Vec::new(),
            renamed: std::collections::HashMap::new(),
        }
    }
}

/// Forget every `unit` declaration — the table is per-program, so a new
/// parse starts empty rather than leaking the last file's names.
pub fn clear() {
    CUSTOM.with(|c| *c.borrow_mut() = std::rc::Rc::new(CustomUnits::new()));
}

/// The live table — a `Vm` keeps its program's so `run` can [`install`] it.
pub fn snapshot() -> std::rc::Rc<CustomUnits> {
    CUSTOM.with(|c| c.borrow().clone())
}

/// Make `units` the live table — see [`snapshot`]. Cheap when the table
/// already installed is `units`, so run-loop batches can call it freely.
pub fn install(units: &std::rc::Rc<CustomUnits>) {
    CUSTOM.with(|c| {
        let mut c = c.borrow_mut();
        if !std::rc::Rc::ptr_eq(&c, units) {
            *c = units.clone();
        }
    });
}

/// Register a `unit name` declaration: `spec` `None` claims the next base
/// dimension slot; `Some((dim, scale))` derives from known units.
/// Redeclaring a name *the same way* is a no-op; a different spec, a
/// builtin name, or more than [`MAX_CUSTOM`] base dims is an error.
pub fn declare(name: &str, spec: Option<(Dim, f64)>) -> Result<(Dim, f64), String> {
    if lookup_builtin(name).is_some() {
        return Err(format!("`{name}` is a builtin unit already"));
    }
    CUSTOM.with(|c| {
        let mut binding = c.borrow_mut();
        let c = std::rc::Rc::make_mut(&mut binding);
        if let Some(&existing) = c.names.get(name) {
            let same = match spec {
                Some(spec) => spec == existing,
                // `unit tile;` again is the same decl iff tile's base slot
                // still says `tile`
                None => existing.0.custom_slot().is_some_and(|i| c.bases[i] == name),
            };
            return if same {
                Ok(existing)
            } else {
                Err(format!("`{name}` is already declared as a different unit"))
            };
        }
        let spec = match spec {
            Some(spec) => spec,
            None => {
                if c.bases.len() >= MAX_CUSTOM {
                    return Err(format!(
                        "a program gets at most {MAX_CUSTOM} `unit` base dimensions"
                    ));
                }
                let spec = (Dim::custom(c.bases.len()), 1.0);
                c.bases.push(name.to_string());
                spec
            }
        };
        c.names.insert(name.to_string(), spec);
        Ok(spec)
    })
}

/// The builtin axes: name, negative direction word, positive direction word,
/// and the glyphs for the axis and its two directions. `y` is up-positive —
/// every coordinate a Mimas program sees (game prims, mouse, squish) is —
/// `z` is into the screen; `t` is time.
const BUILTIN_AXIS: [(&str, &str, &str, Option<&str>, &str, &str); BUILTIN_AXES] = [
    ("x", "left", "right", Some("↔"), "←", "→"),
    ("y", "down", "up", Some("↕"), "↓", "↑"),
    ("z", "outward", "inward", None, "⊙", "⊗"),
    ("t", "earlier", "later", None, "⏪", "⏩"),
];

/// `axis NAME (NEG, POS);` -- a new axis with its two direction words, or a
/// builtin axis renamed (`axis y (south, north);`). Redeclaring the same way
/// is a no-op; a name that is a unit, or a fifth custom axis, is an error.
pub fn declare_axis(name: &str, neg: &str, pos: &str) -> Result<Dim, String> {
    if lookup_builtin(name).is_some() {
        return Err(format!("`{name}` is a unit, not an axis"));
    }
    if let Some(slot) = BUILTIN_AXIS.iter().position(|a| a.0 == name) {
        CUSTOM.with(|c| {
            let mut binding = c.borrow_mut();
            let c = std::rc::Rc::make_mut(&mut binding);
            c.renamed.insert(slot, (neg.to_string(), pos.to_string()));
        });
        return Ok(Dim::axis(slot));
    }
    CUSTOM.with(|c| {
        let mut binding = c.borrow_mut();
        let c = std::rc::Rc::make_mut(&mut binding);
        if let Some(i) = c.axes.iter().position(|a| a.0 == name) {
            return if c.axes[i].1 == neg && c.axes[i].2 == pos {
                Ok(Dim::axis(BUILTIN_AXES + i))
            } else {
                Err(format!("`{name}` is already an axis with other directions"))
            };
        }
        if c.axes.len() >= NORM0 - BUILTIN_AXES {
            return Err(format!(
                "a program gets at most {} `axis` declarations",
                NORM0 - BUILTIN_AXES
            ));
        }
        c.axes.push((name.to_string(), neg.to_string(), pos.to_string()));
        Ok(Dim::axis(BUILTIN_AXES + c.axes.len() - 1))
    })
}

/// The axis a name denotes (`x`, or a declared one), as a dimension.
pub fn axis_lookup(name: &str) -> Option<Dim> {
    if let Some(slot) = BUILTIN_AXIS.iter().position(|a| a.0 == name) {
        return Some(Dim::axis(slot));
    }
    if let Some((_, slot)) = NORMS.iter().find(|(n, _)| *n == name) {
        return Some(Dim::axis(*slot));
    }
    CUSTOM.with(|c| {
        c.borrow()
            .axes
            .iter()
            .position(|a| a.0 == name)
            .map(|i| Dim::axis(BUILTIN_AXES + i))
    })
}

/// The axis an axis glyph denotes: `↔` is x, `↕` is y.
pub fn axis_of_glyph(c: char) -> Option<Dim> {
    BUILTIN_AXIS
        .iter()
        .position(|a| a.3 == Some(c.encode_utf8(&mut [0; 4]) as &str))
        .map(Dim::axis)
}

/// A direction word or glyph: the axis it lies on and whether it is the
/// negative end (`down` and `←` are; `up` and `→` are not). A glyph is a
/// spelling of its builtin word — `↑` resolves wherever `up` does, so an
/// `axis y (up, down)` rename that makes up the negative end flips `↑`
/// with it. When a rename drops the word entirely (`axis y (south,
/// north)`) the glyph keeps its builtin meaning.
pub fn direction(word: &str) -> Option<(Dim, bool)> {
    // glyphs spell the builtin direction words
    let glyph = matches!(word, "←" | "→" | "↑" | "↓" | "⊙" | "⊗" | "⏪" | "⏩");
    let word = match word {
        "←" => "left",
        "→" => "right",
        "↑" => "up",
        "↓" => "down",
        "⊙" => "outward",
        "⊗" => "inward",
        "⏪" => "earlier",
        "⏩" => "later",
        w => w,
    };
    CUSTOM
        .with(|c| {
            let c = c.borrow();
            for (i, (_, neg, pos)) in c.axes.iter().enumerate() {
                if word == neg {
                    return Some((Dim::axis_dir(BUILTIN_AXES + i, true), true));
                }
                if word == pos {
                    return Some((Dim::axis_dir(BUILTIN_AXES + i, false), false));
                }
            }
            for (slot, a) in BUILTIN_AXIS.iter().enumerate() {
                let (neg, pos) = match c.renamed.get(&slot) {
                    Some((n, p)) => (n.as_str(), p.as_str()),
                    None => (a.1, a.2),
                };
                if word == neg {
                    return Some((Dim::axis_dir(slot, true), true));
                }
                if word == pos {
                    return Some((Dim::axis_dir(slot, false), false));
                }
            }
            None
        })
        .or_else(|| {
            glyph.then(|| {
                BUILTIN_AXIS.iter().enumerate().find_map(|(slot, a)| {
                    if word == a.1 {
                        Some((Dim::axis_dir(slot, true), true))
                    } else {
                        (word == a.2).then(|| (Dim::axis_dir(slot, false), false))
                    }
                })
            })?
        })
}

/// The glyph a builtin direction word spells. `⊙`/`⊗` are the physics
/// pair — out of the page, into it — now that tensor spells `⊠`.
pub fn builtin_direction_glyph(word: &str) -> Option<&'static str> {
    Some(match word {
        "left" => "←",
        "right" => "→",
        "up" => "↑",
        "down" => "↓",
        "outward" => "⊙",
        "inward" => "⊗",
        "earlier" => "⏪",
        "later" => "⏩",
        _ => return None,
    })
}

/// The closed range a norm parity promises, and its name — `snorm`
/// is `[-1, 1]`, `unorm` is `[0, 1]`. Other axes and bare dims have none.
pub fn norm_bounds(d: Dim) -> Option<(&'static str, f64, f64)> {
    let a = d.axes();
    if a & (1 << NORM0) != 0 {
        Some(("snorm", -1.0, 1.0))
    } else if a & (1 << (NORM0 + 1)) != 0 {
        Some(("unorm", 0.0, 1.0))
    } else {
        None
    }
}

/// The name of an axis slot: `x`, or the declared name.
pub fn axis_name(slot: usize) -> String {
    if slot < BUILTIN_AXES {
        return BUILTIN_AXIS[slot].0.to_string();
    }
    if slot >= NORM0 {
        return NORMS[slot - NORM0].0.to_string();
    }
    CUSTOM.with(|c| {
        c.borrow()
            .axes
            .get(slot - BUILTIN_AXES)
            .map_or_else(|| "?".to_string(), |a| a.0.clone())
    })
}

/// The name of an axis end for prose: `x`, `x→`, `flow`, `flow(inward)`.
fn axis_dir_name(slot: usize, signed: i8) -> String {
    if signed.abs() != 2 {
        return axis_name(slot);
    }
    if slot < BUILTIN_AXES {
        let a = &BUILTIN_AXIS[slot];
        return format!("{}{}", a.0, if signed == 2 { a.5 } else { a.4 });
    }
    let dir = CUSTOM.with(|c| {
        c.borrow()
            .axes
            .get(slot - BUILTIN_AXES)
            .map(|a| if signed == 2 { a.2.clone() } else { a.1.clone() })
    });
    match dir {
        Some(d) => format!("{}({d})", axis_name(slot)),
        None => axis_name(slot),
    }
}

/// The annotation spelling of an axis slot's claim: `↔`/`↕` unsigned,
/// `→`/`↓` signed, `@name`/`@dir` where a glyph doesn't exist.
fn axis_dir_suffix(slot: usize, signed: i8) -> String {
    if slot < BUILTIN_AXES {
        let a = &BUILTIN_AXIS[slot];
        let g = match signed {
            2 => Some(a.5),
            -2 => Some(a.4),
            _ => a.3,
        };
        if let Some(g) = g {
            return g.to_string();
        }
    }
    let word = CUSTOM.with(|c| {
        c.borrow()
            .axes
            .get(slot.wrapping_sub(BUILTIN_AXES))
            .map(|a| match signed {
                2 => a.2.clone(),
                -2 => a.1.clone(),
                _ => a.0.clone(),
            })
    });
    format!("@{}", word.unwrap_or_else(|| axis_name(slot)))
}

/// A direction *glyph* as a signed axis claim: `wpx↓` is world y
/// measured down-positive, `px→` right-positive x — the signed end the
/// unsigned `axis_of_glyph` arrows (`↔`/`↕`) leave open.
// r[impl mimas.unit.direction.signed]
pub fn dir_of_glyph(c: char) -> Option<Dim> {
    BUILTIN_AXIS.iter().enumerate().find_map(|(slot, a)| {
        if a.4.chars().next() == Some(c) {
            Some(Dim::axis_dir(slot, true))
        } else if a.5.chars().next() == Some(c) {
            Some(Dim::axis_dir(slot, false))
        } else {
            None
        }
    })
}

/// The declared name behind a custom base-dimension slot.
fn custom_name(slot: usize) -> Option<String> {
    CUSTOM.with(|c| c.borrow().bases.get(slot).cloned())
}

/// Every unit name: (dimension, scale to the coherent base unit).
/// Declared units answer alongside the builtins.
// r[impl mimas.unit.literal]
// r[impl mimas.unit.editor-list]
pub fn lookup(name: &str) -> Option<(Dim, f64)> {
    // an axis in unit position: the glyph (`px↕`, `wpx↓`) or `@name`
    // (`usd@flow`, `px@down`)
    if let Some(rest) = name.strip_prefix('@') {
        let axis = axis_lookup(rest).or_else(|| direction(rest).map(|(d, _)| d));
        return axis.map(|d| (d, 1.0));
    }
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next())
        && let Some(d) = axis_of_glyph(c).or_else(|| dir_of_glyph(c))
    {
        return Some((d, 1.0));
    }
    lookup_builtin(name).or_else(|| CUSTOM.with(|c| c.borrow().names.get(name).copied()))
}

/// Every builtin unit name: (dimension, scale to the coherent base unit).
fn lookup_builtin(name: &str) -> Option<(Dim, f64)> {
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
        "mile" => (LENGTH, 1609.344),
        "miles" => (LENGTH, 1609.344),
        // volume is length³ — no base dim of its own; US liquid gallon,
        // 231 in³ = 3.785_411_784e-3 m³
        "gal" => (LENGTH.powi(3), 3.785_411_784e-3),
        "gallon" => (LENGTH.powi(3), 3.785_411_784e-3),
        "gallons" => (LENGTH.powi(3), 3.785_411_784e-3),
        // litre: 1e-3 m³ — `L` and `l` are both the SI symbol
        "L" => (LENGTH.powi(3), 1e-3),
        "l" => (LENGTH.powi(3), 1e-3),
        "liter" => (LENGTH.powi(3), 1e-3),
        "liters" => (LENGTH.powi(3), 1e-3),
        "litre" => (LENGTH.powi(3), 1e-3),
        "litres" => (LENGTH.powi(3), 1e-3),
        "mL" => (LENGTH.powi(3), 1e-6),
        "ml" => (LENGTH.powi(3), 1e-6),
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
        // money and graphics: their own base dimensions. `spx` is a
        // *host-space* pixel — browser/sheet coords a fixture or native
        // hands the page — and `wpx` a *world-space* one. They never meet
        // each other; bare `px` is the unspecified member that absorbs
        // into whichever specific space it touches.
        "usd" => (MONEY, 1.0),
        "px" => (PIXELS, 1.0),
        "spx" => (SCREEN_PX, 1.0),
        "wpx" => (WORLD_PX, 1.0),
        "frame" => (FRAMES, 1.0),
        // index: an `int` position into a collection. its own dimension so a
        // quantity can never pass for one (`xs[i] + 3px` fails), but plain
        // ints still interconvert (`i += 1`, `i < xs.len()`) -- see
        // `index_plain` in the dims pass. `_i` is the math notation
        // spelling: `3_i` is `3index`, `i: _i` the annotated index.
        "idx" => (INDEX, 1.0),
        "index" => (INDEX, 1.0),
        "_i" => (INDEX, 1.0),
        // data: `b` is the bit — the base unit. Byte spellings follow IEC
        // (`B`=8b, `kB`=10³B, `KiB`=2¹⁰B). The music stdlib also abuses
        // `b` for beats in its mini-notation — although it is notation
        // abuse, it is believed to reduce the abuse liability, in the
        // "FDA"'s words.
        "b" => (DATA, 1.0),
        "bit" => (DATA, 1.0),
        "B" => (DATA, 8.0),
        "kB" => (DATA, 8e3),
        "KiB" => (DATA, 8.0 * 1024.0),
        "MB" => (DATA, 8e6),
        "MiB" => (DATA, 8.0 * 1024.0 * 1024.0),
        "GB" => (DATA, 8e9),
        "GiB" => (DATA, 8.0 * 1024.0 * 1024.0 * 1024.0),
        "TB" => (DATA, 8e12),
        // music: `beat` is musical time, its own dimension — a tempo
        // (`bpm`, or `cpm` counting measures) is beats per second.
        // `meas` is a 4/4 measure, `phrase` four measures; `st`/`oct`
        // measure pitch.
        "beat" => (BEAT, 1.0),
        "meas" => (BEAT, 4.0),
        "phrase" => (BEAT, 16.0),
        "bpm" => (BEAT.div(TIME), 1.0 / 60.0),
        "cpm" => (BEAT.div(TIME), 4.0 / 60.0),
        "st" => (PITCH, 1.0),
        "oct" => (PITCH, 12.0),
        // angles: a real dimension — `deg`/`rad` arithmetic checks, and
        // `a.to("deg")` reads an angle back in degrees. `rad` is the base
        // (1 turn = 2π); `grad`/`gon` the surveyor's 400-per-turn;
        // `arcmin`/`arcsec` the astronomer's sixtieths of a degree; `mrad`
        // the milliradian (not the NATO mil — 1/6400 of a turn, which
        // nothing here calls by name).
        // r[impl mimas.unit.angle]
        "rad" => (ANGLE, 1.0),
        "deg" => (ANGLE, std::f64::consts::PI / 180.0),
        "turn" => (ANGLE, std::f64::consts::TAU),
        "rev" => (ANGLE, std::f64::consts::TAU),
        "grad" => (ANGLE, std::f64::consts::PI / 200.0),
        "gon" => (ANGLE, std::f64::consts::PI / 200.0),
        "arcmin" => (ANGLE, std::f64::consts::PI / 10_800.0),
        "arcsec" => (ANGLE, std::f64::consts::PI / 648_000.0),
        "mrad" => (ANGLE, 1e-3),
        // dimensionless scales: `6pct` is a plain float. `prob`/`logit`
        // scale 1 — they're intent annotations ("this number is a
        // probability"), not new dimensions: probability doesn't compose
        // under the dim algebra (`p * p` stays a probability, so a `prob`
        // base dim would wrongly read `prob^2`, and `p / p` is odds, not
        // dimensionless). `bp` is basis points — `25bp` is `0.0025`;
        // spelled `bp`, not `bps`, so `b/s` keeps meaning bits per second.
        "pct" => (NONE, 0.01),
        "bp" => (NONE, 1e-4),
        "prob" => (NONE, 1.0),
        "logit" => (NONE, 1.0),
        // `snorm`/`unorm`, the GPU spellings: a plain number along a
        // norm parity — `pad.lx + lt` is a clash, `pad.lx + 0.5`
        // adopts, `lx * lx` is dimensionless again. Still only intent:
        // nothing keeps `2.0snorm` out but the contract.
        // r[impl mimas.unit.norm]
        "snorm" => (Dim::axis(NORM0), 1.0),
        "unorm" => (Dim::axis(NORM0 + 1), 1.0),
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
            // a trailing axis: `px↕`, `px/s↔`, `wpx↓`, `usd@flow` -- ends the expression
            Some(c) if axis_of_glyph(c).is_some() || dir_of_glyph(c).is_some() => {
                let after = &tail[c.len_utf8()..];
                let axis = axis_of_glyph(c).or_else(|| dir_of_glyph(c)).unwrap();
                return after.is_empty().then(|| (dim.mul(axis), scale));
            }
            Some('@') => {
                let word = &tail[1..];
                // an axis name tags unsigned; a direction word signs its end
                let axis = axis_lookup(word).or_else(|| direction(word).map(|(d, _)| d));
                return axis.map(|a| (dim.mul(a), scale));
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
    // a trailing axis rides the suffix: `40.0px↕`, `8.0wpx↓`, `3col@flow`
    if let Some(c) = rest[end..].chars().next()
        && (axis_of_glyph(c).is_some() || dir_of_glyph(c).is_some())
    {
        end += c.len_utf8();
    } else if let Some(after) = rest[end..].strip_prefix('@') {
        let n = ident(after);
        // an axis name, or a direction word with no glyph (`@in`,
        // `@north`) — a word that has a glyph stays for `at_suffix`,
        // which refuses `@up` and points at `↑`
        let word = &after[..n];
        if n > 0
            && (axis_lookup(word).is_some()
                || (direction(word).is_some() && builtin_direction_glyph(word).is_none()))
        {
            end += 1 + n;
        }
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
        // `pct`/`bp`/`prob`/`logit` are dimensionless — intent + scale only
        assert_eq!(lookup("prob"), Some((Dim::NONE, 1.0)));
        assert_eq!(lookup("logit"), Some((Dim::NONE, 1.0)));
        assert_eq!(lookup("bp"), Some((Dim::NONE, 1e-4)));
        assert!(lookup("bogus").is_none());
    }

    #[test]
    fn a_unit_times_itself_reads_as_its_square() {
        assert_eq!(PIXELS.mul(PIXELS).describe(), "px²");
    }

    // r[verify mimas.unit.angle]
    #[test]
    fn angle_units_carry_the_angle_dimension() {
        use std::f64::consts::{PI, TAU};
        assert_eq!(lookup("rad"), Some((ANGLE, 1.0)));
        for (name, scale) in [
            ("deg", PI / 180.0),
            ("turn", TAU),
            ("rev", TAU),
            ("grad", PI / 200.0),
            ("gon", PI / 200.0),
            ("arcmin", PI / 10_800.0),
            ("arcsec", PI / 648_000.0),
            ("mrad", 1e-3),
        ] {
            assert_eq!(lookup(name), Some((ANGLE, scale)), "{name}");
        }
        // angles compose like any dimension
        assert_eq!(parse("rad/s").unwrap(), (ANGLE.div(TIME), 1.0));
        assert_eq!(parse("deg^2").unwrap().0, ANGLE.powi(2));
        // but they are not dimensionless — pct-style entries stay NONE
        assert_ne!(ANGLE, Dim::NONE);
        assert_eq!(lookup("deg").unwrap().0.describe(), "angle (rad)");
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

    // r[verify mimas.unit.pixels]
    #[test]
    fn px_is_the_unspecified_pixel() {
        let ax = Dim::axis(0);
        let (px, px_x) = (PIXELS, PIXELS.mul(ax));
        let (spx_x, wpx_x) = (SCREEN_PX.mul(ax), WORLD_PX.mul(ax));

        // the specified spaces never meet each other, on either axis
        for (a, b) in [
            (SCREEN_PX, WORLD_PX),
            (spx_x, wpx_x),
            (SCREEN_PX.div(TIME), WORLD_PX.div(TIME)),
        ] {
            assert!(!Dim::agree(a, b), "{a:?} met {b:?}");
        }

        // `px` meets either space and absorbs into it — `4px + ball_x`
        // keeps the ball's space; an untagged `px` adopts the axis too
        for specific in [SCREEN_PX, WORLD_PX] {
            assert!(Dim::agree(px, specific));
            assert!(Dim::agree(specific, px));
            assert_eq!(Dim::absorb(px, specific), specific);
            assert_eq!(Dim::absorb(specific, px), specific);
            assert!(Dim::agree(px_x, specific.mul(ax)));
            assert!(Dim::agree(px, specific.mul(ax)));
            assert_eq!(Dim::absorb(px_x, specific), specific.mul(ax));
        }

        // compounds keep the rule: `px/s` is an unspecified velocity and
        // meets `wpx/s`, but `wpx/s` never meets `spx/s`
        let (ps, ws, ss) = (PIXELS.div(TIME), WORLD_PX.div(TIME), SCREEN_PX.div(TIME));
        assert!(Dim::agree(ps, ws));
        assert!(Dim::agree(ps, ss));
        assert!(!Dim::agree(ws, ss));
        assert_eq!(Dim::absorb(ps, ws), ws);
        // areas too: `px²` meets `wpx²`, never `px²` meets `wpx`
        assert!(Dim::agree(PIXELS.powi(2), WORLD_PX.powi(2)));
        assert!(!Dim::agree(PIXELS.powi(2), WORLD_PX));
        assert!(!Dim::agree(PIXELS, WORLD_PX.powi(2)));

        // a wrong axis still fails inside the family
        let wpx_y = WORLD_PX.mul(Dim::axis(1));
        assert!(!Dim::agree(px_x, wpx_y));
        assert!(!Dim::agree(wpx_x, wpx_y));
        // and a specific space can never shed its tag into `px`
        assert_eq!(Dim::absorb(spx_x, px_x), spx_x);
        assert_eq!(Dim::absorb(wpx_x, px_x), wpx_x);
    }
}

#[cfg(test)]
mod dir_tests {
    use super::*;

    #[test]
    fn directions_resolve() {
        // a direction is a signed claim on its axis's end
        assert_eq!(direction("up"), Some((Dim::axis_dir(1, false), false)));
        assert_eq!(direction("down"), Some((Dim::axis_dir(1, true), true)));
        assert_eq!(direction("↑"), Some((Dim::axis_dir(1, false), false)));
        assert_eq!(direction("↓"), Some((Dim::axis_dir(1, true), true)));
        assert_eq!(direction("→"), Some((Dim::axis_dir(0, false), false)));
        assert_eq!(direction("bogus"), None);
    }

    // r[verify mimas.unit.direction.signed]
    #[test]
    fn signed_axes_agree_convert_and_print() {
        let (up, dn, any) = (
            WORLD_PX.mul(Dim::axis_dir(1, false)),
            WORLD_PX.mul(Dim::axis_dir(1, true)),
            WORLD_PX.mul(Dim::axis(1)),
        );

        // same end and unsigned-vs-signed meet; opposite ends don't —
        // a scalar converts instead (`x↓ + y↑` negates the `↓` side)
        assert!(Dim::agree(up, up));
        assert!(Dim::agree(up, any));
        assert!(Dim::agree(any, dn));
        assert!(!Dim::agree(up, dn));
        assert!(Dim::sign_convertible(up, dn));
        assert!(Dim::sign_opposed(dn, up));
        assert!(!Dim::sign_opposed(up, dn));

        // opposite ends are still *different* conventions, not just
        // different tags: `wpx↓` can't meet `spx↓` across the seam
        let spx_dn = SCREEN_PX.mul(Dim::axis_dir(1, true));
        assert!(!Dim::sign_convertible(dn, spx_dn));
        // and the ends disagree cross-axis: `wpx↑` never meets `wpx→`
        let right = WORLD_PX.mul(Dim::axis_dir(0, false));
        assert!(!Dim::sign_convertible(up, right));

        // unsigned absorbs into the signed end either way
        assert_eq!(Dim::absorb(any, dn), dn);
        assert_eq!(Dim::absorb(up, any), up);
        // an axis an end doesn't claim is kept (`x↕` + `y↓` claim ≠ clobber)
        assert_eq!(Dim::axis_dir(1, true).with_axes(any), dn);

        // annotations and prose spell the end, not the axis
        assert_eq!(dn.name(), "wpx↓");
        assert_eq!(up.name(), "wpx↑");
        assert_eq!(parse("wpx↓"), Some((dn, 1.0)));
        assert_eq!(parse("px@down"), Some((PIXELS.mul(Dim::axis_dir(1, true)), 1.0)));
        assert_eq!(parse("spx↕"), Some((SCREEN_PX.mul(Dim::axis(1)), 1.0)));
    }
}
