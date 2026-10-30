//! `Col2`/`Col3`/`Col4` and `Row2`/`Row3`/`Row4` — oriented vectors.
//!
//! A `Col` is a coordinate list read top-to-bottom; a `Row` is the same
//! list read left-to-right. They're distinct nominal types on purpose:
//! a function that asks for a `Col2` rejects a `Row2` at compile time,
//! and neither stands in for a scalar — the only door between them is
//! the explicit `v.t()` transpose. Think tuples with an orientation:
//! `Col2 { x = 1.0, y = 2.0 }` spells like `Vec2`, but the checker
//! knows which side of a product it belongs on.
//!
//! Arithmetic keeps orientation honest: `+ - * /` between two of the
//! *same* kind is component-wise (and scalars mix on either side), and
//! infix between a `Col` and a `Row` is an error outright — the solver
//! types `Adt * _` as the LHS type, so a product that returns a scalar
//! or a matrix could never infix honestly. The real products have
//! names instead: `row.inner(col) -> float` and
//! `col.outer(row) -> [[float]]`.

use macros::{MimasStruct, native};
use vm::conversion::MimasType;
use vm::{BinOp, Ctx, RtErr, RtResult, UnaryOp, Val, api::Api};

/// The shared surface every oriented vector gets — emitted per type
/// since the bodies are field-shaped. `new`, `t`, `arr`, `dot`, `len`,
/// `norm`, `dist`, `lerp`, `neg` become mimas methods/assoc fns;
/// `bin`/`un` back the infix `+ - * /` and `-v` dispatch registered via
/// `Api::add_bin_op`/`Api::add_unary_op`. `$ORIENT` is `col` or `row` —
/// it picks what `self * mirror` means.
macro_rules! vecfam {
    ($m:ident, $ORIENT:ident, $V:ident, $M:ident, ($($f:ident),+)) => {
        mod $m {
            use super::*;

            /// `Type::new(..)` — same as the `Type { x = .. }` literal.
            #[native]
            #[effects()]
            pub fn new<'gc>(_ctx: Ctx<'gc> $(, $f: f64)+) -> $V {
                $V { $($f),+ }
            }

            /// `v.t()` — the transpose: `Col` ↔ `Row`. The only door to
            /// the other orientation.
            #[native]
            #[effects()]
            pub fn t<'gc>(_ctx: Ctx<'gc>, v: $V) -> $M {
                $M { $($f: v.$f),+ }
            }

            /// `v.arr()` — the components as a plain `[float]` list.
            #[native]
            #[effects()]
            pub fn arr<'gc>(_ctx: Ctx<'gc>, v: $V) -> Vec<f64> {
                vec![$(v.$f),+]
            }

            /// Dot product — `a.dot(b)` (infix `a * b` is component-wise).
            #[native]
            #[effects()]
            pub fn dot<'gc>(_ctx: Ctx<'gc>, a: $V, b: $V) -> f64 {
                0.0 $(+ a.$f * b.$f)+
            }

            /// Euclidean length — `a.len()`.
            #[native]
            #[effects()]
            pub fn len<'gc>(_ctx: Ctx<'gc>, a: $V) -> f64 {
                (0.0 $(+ a.$f * a.$f)+).sqrt()
            }

            /// Unit vector — `a.norm()`; the zero vector stays zero.
            #[native]
            #[effects()]
            pub fn norm<'gc>(_ctx: Ctx<'gc>, a: $V) -> $V {
                let l = (0.0 $(+ a.$f * a.$f)+).sqrt();
                if l == 0.0 { a } else { $V { $($f: a.$f / l),+ } }
            }

            /// Distance to `b` — `a.dist(b)`.
            #[native]
            #[effects()]
            pub fn dist<'gc>(_ctx: Ctx<'gc>, a: $V, b: $V) -> f64 {
                (0.0 $(+ (a.$f - b.$f) * (a.$f - b.$f))+).sqrt()
            }

            /// Linear blend toward `b` by `t` — `a.lerp(b, 0.5)`.
            #[native]
            #[effects()]
            pub fn lerp<'gc>(_ctx: Ctx<'gc>, a: $V, b: $V, t: f64) -> $V {
                $V { $($f: a.$f + (b.$f - a.$f) * t),+ }
            }

            /// Negated — `a.neg()` (same as `-a`).
            #[native]
            #[effects()]
            pub fn neg<'gc>(_ctx: Ctx<'gc>, a: $V) -> $V {
                $V { $($f: -a.$f),+ }
            }

            /// This type's components when `v` is one, else none.
            fn vals<'gc>(ctx: Ctx<'gc>, v: Val<'gc>) -> Option<Vec<f64>> {
                let v = <$V>::from_value(ctx, v).ok()?;
                Some(vec![$(v.$f),+])
            }

            fn scalar(v: Val<'_>) -> Option<f64> {
                match v {
                    Val::Float(f) => Some(f),
                    Val::Int(i) => Some(i as f64),
                    _ => None,
                }
            }

            fn make<'gc>(ctx: Ctx<'gc>, xs: Vec<f64>) -> Val<'gc> {
                let mut it = xs.into_iter();
                $V { $($f: it.next().unwrap_or(0.0)),+ }.into_value(ctx)
            }

            /// `v <op> v` component-wise and `v <op> s`/`s <op> v`
            /// element-wise — nothing else: a `Col` and a `Row` never
            /// meet infix (the cross-orientation products are the named
            /// `inner`/`outer` methods, which carry real signatures).
            pub fn bin<'gc>(ctx: Ctx<'gc>, a: Val<'gc>, b: Val<'gc>, op: BinOp) -> RtResult<Val<'gc>> {
                if let (Some(x), Some(y)) = (vals(ctx, a), vals(ctx, b)) {
                    let mut out = Vec::with_capacity(x.len());
                    for i in 0..x.len() {
                        out.push(match op {
                            BinOp::Add => x[i] + y[i],
                            BinOp::Sub => x[i] - y[i],
                            BinOp::Mult => x[i] * y[i],
                            BinOp::Div => x[i] / y[i],
                            _ => return Err(RtErr::invalid_bin(a, op, b)),
                        });
                    }
                    return Ok(make(ctx, out));
                }
                if let (Some(x), Some(s)) = (vals(ctx, a), scalar(b)) {
                    let mut out = Vec::with_capacity(x.len());
                    for xi in &x {
                        out.push(match op {
                            BinOp::Add => xi + s,
                            BinOp::Sub => xi - s,
                            BinOp::Mult => xi * s,
                            BinOp::Div => xi / s,
                            _ => return Err(RtErr::invalid_bin(a, op, b)),
                        });
                    }
                    return Ok(make(ctx, out));
                }
                if let (Some(s), Some(y)) = (scalar(a), vals(ctx, b)) {
                    let mut out = Vec::with_capacity(y.len());
                    for yi in &y {
                        out.push(match op {
                            BinOp::Add => s + yi,
                            BinOp::Sub => s - yi,
                            BinOp::Mult => s * yi,
                            BinOp::Div => s / yi,
                            _ => return Err(RtErr::invalid_bin(a, op, b)),
                        });
                    }
                    return Ok(make(ctx, out));
                }
                Err(RtErr::invalid_bin(a, op, b))
            }

            /// `-v` / `+v`.
            pub fn un<'gc>(ctx: Ctx<'gc>, a: Val<'gc>, op: UnaryOp) -> RtResult<Val<'gc>> {
                let Ok(v) = <$V>::from_value(ctx, a) else {
                    return Err(RtErr::InvalidUnaryOperand);
                };
                let out = match op {
                    UnaryOp::Negative => $V { $($f: -v.$f),+ },
                    UnaryOp::Positive => v,
                    _ => return Err(RtErr::InvalidUnaryOperand),
                };
                Ok(out.into_value(ctx))
            }

            orient_extra!($ORIENT, $V, $M, ($($f),+), ($($f),+));
        }
    };
}

/// The orientation product — a named method, not an infix op, since its
/// return type isn't the receiver's (`row.inner(col) -> float`,
/// `col.outer(row) -> [[float]]`). The second field list pairs with the
/// first inside `outer`'s nested repetition.
macro_rules! orient_extra {
    (col, $V:ident, $M:ident, ($($fi:ident),+), ($($fj:ident),+)) => {
        /// `c.outer(r)` — the outer product: a `[[float]]` whose rows
        /// scale the row by each column component.
        #[native]
        #[effects()]
        pub fn outer<'gc>(_ctx: Ctx<'gc>, a: $V, b: $M) -> Vec<Vec<f64>> {
            let xs = [$(a.$fi),+];
            let ys = [$(b.$fj),+];
            xs.iter().map(|xi| ys.iter().map(|yj| xi * yj).collect()).collect()
        }
    };
    (row, $V:ident, $M:ident, ($($fi:ident),+), ($($fj:ident),+)) => {
        /// `r.inner(c)` — the inner product: a scalar.
        #[native]
        #[effects()]
        pub fn inner<'gc>(_ctx: Ctx<'gc>, a: $V, b: $M) -> f64 {
            0.0 $(+ a.$fi * b.$fj)+
        }
    };
}

/// Column vector — read top-to-bottom. `Col2 { x = 1.0, y = 2.0 }`.
#[derive(Debug, Clone, Copy, Default, PartialEq, MimasStruct)]
pub struct Col2 {
    pub x: f64,
    pub y: f64,
}
/// 3-component column vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, MimasStruct)]
pub struct Col3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
/// 4-component column vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, MimasStruct)]
pub struct Col4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}
/// Row vector — read left-to-right: a `Col`'s transpose.
#[derive(Debug, Clone, Copy, Default, PartialEq, MimasStruct)]
pub struct Row2 {
    pub x: f64,
    pub y: f64,
}
/// 3-component row vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, MimasStruct)]
pub struct Row3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
/// 4-component row vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, MimasStruct)]
pub struct Row4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

vecfam!(c2, col, Col2, Row2, (x, y));
vecfam!(c3, col, Col3, Row3, (x, y, z));
vecfam!(c4, col, Col4, Row4, (x, y, z, w));
vecfam!(r2, row, Row2, Col2, (x, y));
vecfam!(r3, row, Row3, Col3, (x, y, z));
vecfam!(r4, row, Row4, Col4, (x, y, z, w));

/// Register the six types, their constructors, methods and operators —
/// top-level, like `Interval`: orientation is a vocabulary, not a
/// module you opt into.
pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    // ADTs first — `.t()` signatures reference the mirror type, so every
    // layout must exist before any method registers.
    api.add_adt::<Col2>();
    api.add_adt::<Col3>();
    api.add_adt::<Col4>();
    api.add_adt::<Row2>();
    api.add_adt::<Row3>();
    api.add_adt::<Row4>();
    macro_rules! reg {
        ($V:ident, $m:ident, $prod:ident) => {
            api.add_assoc_of::<$V, _, _>("new", $m::new);
            api.add_method($m::t);
            api.add_method($m::$prod);
            api.add_method($m::arr);
            api.add_method($m::dot);
            api.add_method($m::len);
            api.add_method($m::norm);
            api.add_method($m::dist);
            api.add_method($m::lerp);
            api.add_method($m::neg);
            api.add_bin_op::<$V, _>($m::bin);
            api.add_unary_op::<$V, _>($m::un);
        };
    }
    reg!(Col2, c2, outer);
    reg!(Col3, c3, outer);
    reg!(Col4, c4, outer);
    reg!(Row2, r2, inner);
    reg!(Row3, r3, inner);
    reg!(Row4, r4, inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_preserves_field_order() {
        let v = Col4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let r = Row4 { x: v.x, y: v.y, z: v.z, w: v.w };
        assert_eq!((r.x, r.y, r.z, r.w), (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn len_and_norm() {
        let v = Col3 { x: 3.0, y: 4.0, z: 0.0 };
        assert_eq!((v.x * v.x + v.y * v.y + v.z * v.z).sqrt(), 5.0);
    }
}
