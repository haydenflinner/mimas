use macros::native;
use shared::{Literal, Ty};
use vm::{
    Rad,
    api::Api,
    glam::{Quat, Vec2, Vec3},
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
    m.add_adt::<Vec2>();
    m.add_adt::<Vec3>();
    m.add_adt::<Quat>();
    api.add_assoc_of::<Vec2, _, _>("new", vec2_new);
    api.add_assoc_of::<Vec2, _, _>("from_angle", vec2_from_angle);
    api.add_method_named("length", vec2_length);
    api.add_method_named("normalize", vec2_normalize);
    api.add_method_named("add", vec2_add);
    api.add_method_named("sub", vec2_sub);
    api.add_method_named("scale", vec2_scale);
    // the `*`, `/` and unary `-` operators — `mul`/`rmul` cover `v * s` and
    // `s * v`, `div` is `v / s`, `neg` is `-v` (see `op_method_names` in solve)
    api.add_method_named("mul", vec2_scale);
    api.add_method_named("rmul", vec2_scale);
    api.add_method_named("div", vec2_div);
    api.add_method_named("neg", vec2_neg);
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
    api.add_assoc_of::<Vec2, _, _>("from_polar", vec2_from_polar);
    api.add_assoc_of::<Vec3, _, _>("new", vec3_new);
    api.add_method_named("length", vec3_length);
    api.add_method_named("normalize", vec3_normalize);
    api.add_method_named("add", vec3_add);
    api.add_method_named("sub", vec3_sub);
    api.add_method_named("mul", vec3_scale);
    api.add_method_named("rmul", vec3_scale);
    api.add_method_named("div", vec3_div);
    api.add_method_named("neg", vec3_neg);
    api.add_method_named("dot", vec3_dot);
    api.add_method_named("cross", vec3_cross);
    api.add_method_named("distance", vec3_distance);
    api.add_method_named("angle", vec3_angle);
    api.add_method_named("lerp", vec3_lerp);
    api.add_assoc_of::<Quat, _, _>("from_rotation_x", quat_from_rotation_x);
    api.add_assoc_of::<Quat, _, _>("from_rotation_y", quat_from_rotation_y);
    api.add_assoc_of::<Quat, _, _>("from_rotation_z", quat_from_rotation_z);
    api.add_assoc_of::<Quat, _, _>("from_axis_angle", quat_from_axis_angle);
    api.add_method_named("rotate_z", quat_rotate_z);
    api.add_method_named("mul", quat_mul);
    api.add_method_named("conj", quat_conj);
    api.add_method_named("inverse", quat_inverse);
    api.add_method_named("slerp", quat_slerp);
    api.add_method_named("angle", quat_angle);
    api.add_method_named("axis", quat_axis);
    api.add_method_named("rotate", quat_rotate_vec3);
}

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
#[ctor]
fn vec2_new(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y)
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
fn vec2_from_angle(angle: Rad) -> Vec2 {
    Vec2::from_angle(angle.0 as f32)
}

/// Returns the vector's straight-line length, `sqrt(x * x + y * y)`.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let d = Vec2::new(3.0, 4.0).length(); // 5.0
/// ```
#[native]
fn vec2_length(v: Vec2) -> f32 {
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
fn vec2_normalize(v: Vec2) -> Vec2 {
    v.normalize_or_zero()
}

/// Returns the sum of the two vectors, adding them component by component.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let v = Vec2::new(3.0, 4.0).add(Vec2::new(1.0, 1.0));
/// let x = v.x; // 4.0
/// ```
#[native]
fn vec2_add(v: Vec2, other: Vec2) -> Vec2 {
    v + other
}

/// Returns this vector minus `other`, component by component. `b.sub(a)` is the vector that
/// points from `a` to `b`.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let player = Vec2::new(1.0, 1.0);
/// let enemy = Vec2::new(4.0, 5.0);
/// let offset = enemy.sub(player);
/// let x = offset.x; // 3.0
/// ```
#[native]
fn vec2_sub(v: Vec2, other: Vec2) -> Vec2 {
    v - other
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
fn vec2_scale(v: Vec2, by: f32) -> Vec2 {
    v * by
}

/// `v / by` — componentwise division. `v.div(2.0)` is also the `/` operator's method,
/// so `v / 2.0` spells the same thing.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let half = Vec2::new(3.0, 4.0) / 2.0;
/// let x = half.x; // 1.5
/// ```
#[native]
fn vec2_div(v: Vec2, by: f32) -> Vec2 {
    v / by
}

/// `-v` — the unary `-` operator's method.
///
/// ```mimas
/// use std::math::Vec2;
///
/// let backwards = -Vec2::new(1.0, 0.0); // Vec2(-1, 0)
/// ```
#[native]
fn vec2_neg(v: Vec2) -> Vec2 {
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
fn vec3_scale(v: Vec3, by: f32) -> Vec3 {
    v * by
}

/// `Vec3`'s `/` — `v / s`.
///
/// ```mimas
/// use std::math::Vec3;
///
/// let half = Vec3 { x = 1.0, y = 0.0, z = 2.0 } / 2.0;
/// ```
#[native]
fn vec3_div(v: Vec3, by: f32) -> Vec3 {
    v / by
}

/// `Vec3`'s unary `-`.
///
/// ```mimas
/// use std::math::Vec3;
///
/// let down = -Vec3 { x = 0.0, y = 0.0, z = 1.0 };
/// ```
#[native]
fn vec3_neg(v: Vec3) -> Vec3 {
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
fn vec2_dot(v: Vec2, other: Vec2) -> f32 {
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
fn vec2_distance(v: Vec2, other: Vec2) -> f32 {
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
fn vec2_angle(v: Vec2) -> Rad {
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
fn vec2_lerp(v: Vec2, other: Vec2, t: f32) -> Vec2 {
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
fn vec3_length(v: Vec3) -> f32 {
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
fn vec3_normalize(v: Vec3) -> Vec3 {
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
fn quat_from_rotation_x(angle: Rad) -> Quat {
    Quat::from_rotation_x(angle.0 as f32)
}

/// Creates a rotation of `angle` around the y axis.
///
/// ```mimas
/// use std::math::Quat;
///
/// let yaw = Quat::from_rotation_y(0.25turn);
/// ```
#[native]
fn quat_from_rotation_y(angle: Rad) -> Quat {
    Quat::from_rotation_y(angle.0 as f32)
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
fn quat_from_rotation_z(angle: Rad) -> Quat {
    Quat::from_rotation_z(angle.0 as f32)
}

/// Creates a rotation of `angle` around `axis` (which needn't be unit — it's normalized).
///
/// ```mimas
/// use std::math::{Quat, Vec3};
///
/// let flip = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5turn);
/// ```
#[native]
fn quat_from_axis_angle(axis: Vec3, angle: Rad) -> Quat {
    Quat::from_axis_angle(axis, angle.0 as f32)
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
fn quat_rotate_z(q: Quat, angle: Rad) -> Quat {
    q * Quat::from_rotation_z(angle.0 as f32)
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
fn quat_mul(q: Quat, other: Quat) -> Quat {
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
fn quat_conj(q: Quat) -> Quat {
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
fn quat_inverse(q: Quat) -> Quat {
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
fn quat_slerp(q: Quat, other: Quat, t: f32) -> Quat {
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
fn quat_angle(q: Quat) -> Rad {
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
fn quat_axis(q: Quat) -> Vec3 {
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
fn quat_rotate_vec3(q: Quat, v: Vec3) -> Vec3 {
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
#[ctor]
fn vec3_new(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// Returns the sum of the two vectors, component by component.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check (Vec3::new(1.0, 0.0, 0.0) + Vec3::new(0.0, 2.0, 0.0)).y == 2.0;
/// ```
#[native]
fn vec3_add(v: Vec3, other: Vec3) -> Vec3 {
    v + other
}

/// `v - other`, component by component.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check (Vec3::new(3.0, 4.0, 0.0) - Vec3::new(1.0, 1.0, 0.0)).x == 2.0;
/// ```
#[native]
fn vec3_sub(v: Vec3, other: Vec3) -> Vec3 {
    v - other
}

/// Returns the dot product. For two unit vectors it's the cosine of the angle between them.
///
/// ```mimas
/// use std::math::Vec3;
///
/// check Vec3::new(1.0, 0.0, 0.0).dot(Vec3::new(0.0, 1.0, 0.0)) == 0.0;
/// ```
#[native]
fn vec3_dot(v: Vec3, other: Vec3) -> f32 {
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
fn vec3_cross(v: Vec3, other: Vec3) -> Vec3 {
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
fn vec3_distance(v: Vec3, other: Vec3) -> f32 {
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
fn vec3_angle(v: Vec3, other: Vec3) -> Rad {
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
fn vec3_lerp(v: Vec3, other: Vec3, t: f32) -> Vec3 {
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
fn vec2_cmul(v: Vec2, other: Vec2) -> Vec2 {
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
fn vec2_conj(v: Vec2) -> Vec2 {
    Vec2::new(v.x, -v.y)
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
fn vec2_from_polar(r: f32, angle: Rad) -> Vec2 {
    Vec2::from_angle(angle.0 as f32) * r
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
fn vec2_exp(v: Vec2) -> Vec2 {
    Vec2::from_angle(v.y) * v.x.exp()
}
