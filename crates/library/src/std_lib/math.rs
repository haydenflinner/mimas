use macros::native;
use shared::{Literal, Ty};
use vm::{
    BinOp, Ctx, Rad, RtErr, RtResult, UnaryOp, Val,
    api::Api,
    conversion::MimasType,
    glam::{DQuat, DVec2, DVec3},
};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    let mut m = api.module("std::math");
    // r[impl mimas.unit.trig]
    m.constant_dim(
        "PI",
        Ty::Float,
        Literal::Float(std::f64::consts::PI),
        Some(shared::units::ANGLE),
        "The ratio of a circle's circumference to its diameter, about `3.14159`. An angle of `PI` \
         radians is half a turn.",
    );
    m.constant_dim(
        "TAU",
        Ty::Float,
        Literal::Float(std::f64::consts::TAU),
        Some(shared::units::ANGLE),
        "A full turn in radians, equal to `2.0 * PI` (about `6.28319`).",
    );
    m.constant(
        "E",
        Ty::Float,
        Literal::Float(std::f64::consts::E),
        "Euler's number, the base of the natural logarithm, about `2.71828`.",
    );
    m.add_adt::<DVec2>();
    m.add_adt::<DVec3>();
    m.add_adt::<DQuat>();
    api.add_assoc_of::<DVec2, _, _>("new", vec2_new);
    api.add_assoc_of::<DVec2, _, _>("from_angle", vec2_from_angle);
    api.add_method_named("length", vec2_length);
    api.add_method_named("normalize", vec2_normalize);
    // the game kit's short names, the same functions, so one `Vec2` serves
    // both (r[impl mimas.math.vec2-names])
    api.add_method_named("len", vec2_length);
    api.add_method_named("norm", vec2_normalize);
    api.add_method_named("dist", vec2_distance);
    api.add_method_named("scale", vec2_scale);
    api.add_method_named("neg", vec2_neg);
    // r[impl mimas.math.vec-ops] `+ - * /` are component-wise between two
    // vectors and broadcast a scalar on either side, `-v` negates: the game
    // kit's, GLSL's and glam's rules. `dot` stays a method and `cmul` the
    // complex multiply.
    api.add_bin_op::<DVec2, _>(vec2_bin);
    api.add_unary_op::<DVec2, _>(vec2_un);
    api.add_method_named("dot", vec2_dot);
    api.add_method_named("distance", vec2_distance);
    api.add_method_named("angle", vec2_angle);
    api.add_method_named("lerp", vec2_lerp);
    // the complex plane on a Vec2 — `cmul` is rotation-and-scaling in
    // one multiply, `conj`/`arg`/`exp`/`from_polar` the rest of the
    // pair's complex reading
    api.add_method_named("cmul", vec2_cmul);
    api.add_method_named("conj", vec2_conj);
    api.add_method_named("arg", vec2_angle);
    api.add_method_named("exp", vec2_exp);
    api.add_assoc_of::<DVec2, _, _>("from_polar", vec2_from_polar);
    api.add_assoc_of::<DVec3, _, _>("new", vec3_new);
    api.add_method_named("length", vec3_length);
    api.add_method_named("normalize", vec3_normalize);
    api.add_method_named("len", vec3_length);
    api.add_method_named("norm", vec3_normalize);
    api.add_method_named("dist", vec3_distance);
    api.add_method_named("scale", vec3_scale);
    api.add_method_named("neg", vec3_neg);
    api.add_bin_op::<DVec3, _>(vec3_bin);
    api.add_unary_op::<DVec3, _>(vec3_un);
    api.add_method_named("dot", vec3_dot);
    api.add_method_named("cross", vec3_cross);
    api.add_method_named("distance", vec3_distance);
    api.add_method_named("angle", vec3_angle);
    api.add_method_named("lerp", vec3_lerp);
    api.add_assoc_of::<DQuat, _, _>("from_rotation_x", quat_from_rotation_x);
    api.add_assoc_of::<DQuat, _, _>("from_rotation_y", quat_from_rotation_y);
    api.add_assoc_of::<DQuat, _, _>("from_rotation_z", quat_from_rotation_z);
    api.add_assoc_of::<DQuat, _, _>("from_axis_angle", quat_from_axis_angle);
    api.add_method_named("rotate_z", quat_rotate_z);
    api.add_method_named("mul", quat_mul);
    api.add_method_named("conj", quat_conj);
    api.add_method_named("inverse", quat_inverse);
    api.add_method_named("slerp", quat_slerp);
    api.add_method_named("angle", quat_angle);
    api.add_method_named("axis", quat_axis);
    api.add_method_named("rotate", quat_rotate_vec3);
}

/// A scalar operand of a vector operator: `int` or `float`.
fn scalar(v: Val<'_>) -> Option<f64> {
    match v {
        Val::Float(f) => Some(f),
        Val::Int(i) => Some(i as f64),
        _ => None,
    }
}

/// One component of `a <op> b`.
fn apply(op: BinOp, a: f64, b: f64) -> Option<f64> {
    Some(match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mult => a * b,
        BinOp::Div => a / b,
        _ => return None,
    })
}

macro_rules! vec_ops {
    ($bin:ident, $un:ident, $V:ident, ($($f:ident),+)) => {
        /// Infix `v <op> v` (component-wise), `v <op> s` and `s <op> v`
        /// (the scalar broadcast) for `+ - * /` (r[impl mimas.math.vec-ops]).
        fn $bin<'gc>(ctx: Ctx<'gc>, a: Val<'gc>, b: Val<'gc>, op: BinOp) -> RtResult<Val<'gc>> {
            let va = <$V as MimasType>::from_value(ctx, a).ok();
            let vb = <$V as MimasType>::from_value(ctx, b).ok();
            let bad = || RtErr::invalid_bin(a, op, b);
            let out = match (va, vb, scalar(a), scalar(b)) {
                (Some(x), Some(y), _, _) => $V::new($(apply(op, x.$f, y.$f).ok_or_else(bad)?),+),
                (Some(x), _, _, Some(s)) => $V::new($(apply(op, x.$f, s).ok_or_else(bad)?),+),
                (_, Some(y), Some(s), _) => $V::new($(apply(op, s, y.$f).ok_or_else(bad)?),+),
                _ => return Err(bad()),
            };
            Ok(out.into_value(ctx))
        }

        /// `-v` and `+v`.
        fn $un<'gc>(ctx: Ctx<'gc>, a: Val<'gc>, op: UnaryOp) -> RtResult<Val<'gc>> {
            let Ok(v) = <$V as MimasType>::from_value(ctx, a) else {
                return Err(RtErr::InvalidUnaryOperand);
            };
            let out = match op {
                UnaryOp::Negative => -v,
                UnaryOp::Positive => v,
                _ => return Err(RtErr::InvalidUnaryOperand),
            };
            Ok(out.into_value(ctx))
        }
    };
}

vec_ops!(vec2_bin, vec2_un, DVec2, (x, y));
vec_ops!(vec3_bin, vec3_un, DVec3, (x, y, z));

/// Creates a vector from its `x` and `y` components. A struct literal such as
/// `Vec2 { x = 3.0, y = 4.0 }` does the same.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let v = Vec2::new(3.0, 4.0);
/// let y = v.y; // 4.0
/// ```
#[native]
fn vec2_new(x: f64, y: f64) -> DVec2 {
    DVec2 { x, y }
}

/// Creates a vector of length `1.0` pointing at `angle`, measured from the positive
/// x axis toward the positive y axis. Its components are the angle's cosine and sine.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let right = Vec2::from_angle(0.0rad);
/// let x = right.x; // 1.0
/// ```
#[native]
fn vec2_from_angle(angle: Rad) -> DVec2 {
    DVec2::from_angle(angle.0)
}

/// Returns the vector's straight-line length, `sqrt(x * x + y * y)`.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let d = Vec2::new(3.0, 4.0).length(); // 5.0
/// ```
#[native]
fn vec2_length(v: DVec2) -> f64 {
    v.length()
}

/// Returns the vector scaled to length `1.0`, keeping its direction. A zero vector stays zero.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let u = Vec2::new(3.0, 4.0).normalize();
/// let d = u.length(); // 1.0
/// ```
#[native]
fn vec2_normalize(v: DVec2) -> DVec2 {
    v.normalize_or_zero()
}

/// Returns the vector with both components multiplied by `by`.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let half = Vec2::new(3.0, 4.0).scale(0.5);
/// let x = half.x; // 1.5
/// ```
#[native]
fn vec2_scale(v: DVec2, by: f64) -> DVec2 {
    v * by
}

/// `-v` — the unary `-` operator's method.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let backwards = -Vec2::new(1.0, 0.0); // Vec2(-1, 0)
/// ```
#[native]
fn vec2_neg(v: DVec2) -> DVec2 {
    -v
}

/// `Vec3`'s `*` — `v * s` and `s * v` (`rmul`) both land here.
///
/// ```mimas
/// use std::math::Vec3;
///
/// let dbl = Vec3 { x = 1.0, y = 0.0, z = 2.0 } * 2.0;
/// ```
#[native]
fn vec3_scale(v: DVec3, by: f64) -> DVec3 {
    v * by
}

/// `Vec3`'s unary `-`.
///
/// ```mimas
/// use std::math::Vec3;
///
/// let down = -Vec3 { x = 0.0, y = 0.0, z = 1.0 };
/// ```
#[native]
fn vec3_neg(v: DVec3) -> DVec3 {
    -v
}

/// Returns the dot product, `x * other.x + y * other.y`. For two vectors of length `1.0` it's the
/// cosine of the angle between them, and a result of `0.0` means they're perpendicular.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let a = Vec2::new(3.0, 4.0).dot(Vec2::new(1.0, 0.0)); // 3.0
/// ```
#[native]
fn vec2_dot(v: DVec2, other: DVec2) -> f64 {
    v.dot(other)
}

/// Returns the straight-line distance between the two points.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let d = Vec2::new(0.0, 0.0).distance(Vec2::new(3.0, 4.0)); // 5.0
/// ```
#[native]
fn vec2_distance(v: DVec2, other: DVec2) -> f64 {
    v.distance(other)
}

/// Returns the angle of the vector, measured from the positive x axis toward the
/// positive y axis — an angle quantity between `-0.5turn` and `0.5turn`, so
/// `v.angle().to("deg")` reads it in degrees. `arg` is the same method under its
/// complex name; `Vec2::from_angle` goes the other way.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let a = Vec2::new(1.0, 0.0).angle(); // 0.0rad
/// ```
#[native]
fn vec2_angle(v: DVec2) -> Rad {
    Rad(v.to_angle() as f64)
}

/// Returns the point that is `t` of the way from this vector to `other`, so `0.5` gives the
/// midpoint. `t` isn't limited to `0.0` through `1.0`: `1.5` goes past `other` by half the
/// distance again.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let start = Vec2::new(0.0, 0.0);
/// let end = Vec2::new(10.0, 20.0);
/// let quarter = start.lerp(end, 0.25);
/// let y = quarter.y; // 5.0
/// ```
#[native]
fn vec2_lerp(v: DVec2, other: DVec2, t: f64) -> DVec2 {
    v.lerp(other, t)
}

/// Returns the vector's straight-line length, `sqrt(x * x + y * y + z * z)`.
///
/// ```mimas
/// use std::math::Vec3;
///
/// let d = Vec3 { x = 2.0, y = 3.0, z = 6.0 }.length(); // 7.0
/// ```
#[native]
fn vec3_length(v: DVec3) -> f64 {
    v.length()
}

/// Returns the vector scaled to length `1.0`, keeping its direction. A zero vector stays zero.
///
/// ```mimas
/// use std::math::Vec3;
///
/// let u = Vec3 { x = 0.0, y = 0.0, z = 5.0 }.normalize();
/// let z = u.z; // 1.0
/// ```
#[native]
fn vec3_normalize(v: DVec3) -> DVec3 {
    v.normalize_or_zero()
}

/// Creates a rotation of `angle` around the x axis.
///
/// ```mimas
/// use std::math::Quat;
///
/// let pitch = Quat::from_rotation_x(15deg);
/// ```
#[native]
fn quat_from_rotation_x(angle: Rad) -> DQuat {
    DQuat::from_rotation_x(angle.0)
}

/// Creates a rotation of `angle` around the y axis.
///
/// ```mimas
/// use std::math::Quat;
///
/// let yaw = Quat::from_rotation_y(0.25turn);
/// ```
#[native]
fn quat_from_rotation_y(angle: Rad) -> DQuat {
    DQuat::from_rotation_y(angle.0)
}

/// Creates a rotation of `angle` around the z axis. For 2D work, this is a turn within the x-y
/// plane.
///
/// ```mimas
/// use std::math::Quat;
///
/// let quarter_turn = Quat::from_rotation_z(90deg);
/// ```
#[native]
fn quat_from_rotation_z(angle: Rad) -> DQuat {
    DQuat::from_rotation_z(angle.0)
}

/// Creates a rotation of `angle` around `axis` (which needn't be unit — it's normalized).
///
/// ```mimas
/// use std::math::{Quat, Vec3};
///
/// let flip = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5turn);
/// ```
#[native]
fn quat_from_axis_angle(axis: DVec3, angle: Rad) -> DQuat {
    DQuat::from_axis_angle(axis, angle.0)
}

/// Returns the rotation turned a further `angle` around its own z axis.
///
/// ```mimas
/// use std::math::Quat;
///
/// let facing = Quat::from_rotation_z(0.0rad);
/// let turned = facing.rotate_z(90deg); // a quarter turn
/// ```
#[native]
fn quat_rotate_z(q: DQuat, angle: Rad) -> DQuat {
    q * DQuat::from_rotation_z(angle.0)
}

/// `q * other` — the `*` operator: rotations compose, `q * r` being `r`'s turn then `q`'s.
///
/// ```mimas
/// use std::math::Quat;
///
/// let both = Quat::from_rotation_z(30deg) * Quat::from_rotation_z(60deg);
/// check (both.angle().to("deg") - 90.0).abs() < 0.001;
/// ```
#[native]
fn quat_mul(q: DQuat, other: DQuat) -> DQuat {
    q * other
}

/// Returns the quaternion with its imaginary part negated — for a unit quaternion, the
/// rotation that undoes it.
///
/// ```mimas
/// use std::math::Quat;
///
/// let back = Quat::from_rotation_z(30deg).conj();
/// ```
#[native]
fn quat_conj(q: DQuat) -> DQuat {
    q.conjugate()
}

/// Returns the multiplicative inverse — for a unit quaternion, the same as `conj`.
///
/// ```mimas
/// use std::math::Quat;
///
/// let eye = Quat::from_rotation_z(30deg) * Quat::from_rotation_z(30deg).inverse();
/// ```
#[native]
fn quat_inverse(q: DQuat) -> DQuat {
    q.inverse()
}

/// Returns the rotation `t` of the way from `q` to `other` along the shortest arc —
/// the smooth way to animate a turn.
///
/// ```mimas
/// use std::math::Quat;
///
/// let mid = Quat::from_rotation_z(0.0rad).slerp(Quat::from_rotation_z(90deg), 0.5);
/// check (mid.angle().to("deg") - 45.0).abs() < 0.001;
/// ```
#[native]
fn quat_slerp(q: DQuat, other: DQuat, t: f64) -> DQuat {
    q.slerp(other, t)
}

/// Returns the rotation's angle — the `to("deg")`-able quantity the turn amounts to.
///
/// ```mimas
/// use std::math::Quat;
///
/// check (Quat::from_rotation_z(90deg).angle().to("deg") - 90.0).abs() < 0.001;
/// ```
#[native]
fn quat_angle(q: DQuat) -> Rad {
    Rad(q.to_axis_angle().1 as f64)
}

/// Returns the axis the rotation turns around; the identity rotation's axis is arbitrary.
///
/// ```mimas
/// use std::math::{Quat, Vec3};
///
/// check (Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 30deg).axis().z - 1.0).abs() < 0.001;
/// ```
#[native]
fn quat_axis(q: DQuat) -> DVec3 {
    q.to_axis_angle().0
}

/// Returns `v` rotated by this quaternion — the way a facing turns a point.
///
/// ```mimas
/// use std::math::{Quat, Vec3};
///
/// let east = Quat::from_rotation_z(90deg).rotate(Vec3::new(1.0, 0.0, 0.0));
/// check (east.y - 1.0).abs() < 0.001;
/// ```
#[native]
fn quat_rotate_vec3(q: DQuat, v: DVec3) -> DVec3 {
    q * v
}

// ── Vec3 operators ───────────────────────────────────────────────────

/// Creates a vector from its `x`, `y` and `z` components. A struct literal such as
/// `Vec3 { x = 3.0, y = 4.0, z = 0.0 }` does the same.
///
/// ```mimas
/// use std::math::Vec3;
///
/// let v = Vec3::new(1.0, 2.0, 3.0);
/// ```
#[native]
fn vec3_new(x: f64, y: f64, z: f64) -> DVec3 {
    DVec3 { x, y, z }
}

/// Returns the dot product. For two unit vectors it's the cosine of the angle between them.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check Vec3::new(1.0, 0.0, 0.0).dot(Vec3::new(0.0, 1.0, 0.0)) == 0.0;
/// ```
#[native]
fn vec3_dot(v: DVec3, other: DVec3) -> f64 {
    v.dot(other)
}

/// Returns the cross product — a vector perpendicular to both, with the area of the
/// parallelogram they span as its length.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0)).z == 1.0;
/// ```
#[native]
fn vec3_cross(v: DVec3, other: DVec3) -> DVec3 {
    v.cross(other)
}

/// Returns the straight-line distance between the two points.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check Vec3::new(0.0, 0.0, 0.0).distance(Vec3::new(2.0, 3.0, 6.0)) == 7.0;
/// ```
#[native]
fn vec3_distance(v: DVec3, other: DVec3) -> f64 {
    v.distance(other)
}

/// Returns the angle between the two vectors, between `0.0rad` and `0.5turn`.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check (Vec3::new(1.0, 0.0, 0.0).angle(Vec3::new(0.0, 1.0, 0.0)).to("deg") - 90.0).abs() < 0.001;
/// ```
#[native]
fn vec3_angle(v: DVec3, other: DVec3) -> Rad {
    Rad(v.angle_between(other) as f64)
}

/// Returns the point `t` of the way from `v` to `other`; `t` isn't limited to `0.0`..`1.0`.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check Vec3::new(0.0, 0.0, 0.0).lerp(Vec3::new(4.0, 8.0, 0.0), 0.25).y == 2.0;
/// ```
#[native]
fn vec3_lerp(v: DVec3, other: DVec3, t: f64) -> DVec3 {
    v.lerp(other, t)
}

// ── the complex plane on a Vec2 ─────────────────────────────────────

/// Complex multiplication — `a.cmul(b)` treats both vectors as complex numbers:
/// the product has `a`'s angle added to `b`'s and `a`'s length times `b`'s. Read
/// geometrically it is `b` rotated by `a`'s angle and scaled by `a`'s length.
///
/// ```mimas
/// use std::math::Vec2;
///
/// // i squared is -1: (0,1) * (0,1) is (-1,0).
/// check Vec2::new(0.0, 1.0).cmul(Vec2::new(0.0, 1.0)).x == -1.0;
/// ```
#[native]
fn vec2_cmul(v: DVec2, other: DVec2) -> DVec2 {
    v.rotate(other)
}

/// The complex conjugate — `(x, -y)`, the vector mirrored across the x axis.
///
/// ```mimas
/// use std::math::Vec2;
///
/// check Vec2::new(3.0, 4.0).conj().y == -4.0;
/// ```
#[native]
fn vec2_conj(v: DVec2) -> DVec2 {
    DVec2::new(v.x, -v.y)
}

/// Creates the complex number `r` at `angle` — `Vec2::from_angle(angle) * r`.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let v = Vec2::from_polar(2.0, 90deg);
/// check v.y == 2.0;
/// ```
#[native]
fn vec2_from_polar(r: f64, angle: Rad) -> DVec2 {
    DVec2::from_angle(angle.0) * r
}

/// The complex exponential: `(x, y).exp()` is `e^x` at angle `y` — where the
/// `x` component scales the length and `y` is the angle in radians.
///
/// ```mimas
/// use std::math::{PI, Vec2};
///
/// // e to the i pi is -1, give or take float noise:
/// let z = Vec2::new(0.0, PI).exp();
/// check (z.x + 1.0).abs() < 1e-6;
/// ```
#[native]
fn vec2_exp(v: DVec2) -> DVec2 {
    DVec2::from_angle(v.y) * v.x.exp()
}
