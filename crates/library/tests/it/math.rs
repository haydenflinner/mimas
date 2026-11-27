test_run!(
    vec2_math,
    "use std::math::Vec2;",
    "Vec2::new(3.0, 4.0).length()" => "5.0",
    "(Vec2 { x = 1.0, y = 2.0 } + Vec2::new(2.0, 3.0)).y" => "5.0",
    "Vec2::new(0.0, 2.0).normalize().y" => "1.0",
    "Vec2::from_angle(0.0rad).x" => "1.0",
    "(Vec2::from_angle(90deg).y - 1.0).abs() < 1.0e-4" => "true",
);

test_run!(
    vec3_and_quat_math,
    "use std::math::{Vec3, Quat};",
    "Vec3::new(2.0, 3.0, 6.0).length()" => "7.0",
    "Quat::from_rotation_z(0.0rad).rotate_z(0.0rad).w" => "1.0",
);

// r[verify mimas.unit.angle]
// r[verify mimas.unit.trig]
test_run!(
    vec2_complex_plane,
    "use std::math::Vec2;",
    // i * i = -1
    "Vec2::new(0.0, 1.0).cmul(Vec2::new(0.0, 1.0)).x" => "-1.0",
    // (1+2i)(3-1i) = 5+5i
    "Vec2::new(1.0, 2.0).cmul(Vec2::new(3.0, -1.0)).x" => "5.0",
    "Vec2::new(1.0, 2.0).cmul(Vec2::new(3.0, -1.0)).y" => "5.0",
    "Vec2::new(1.0, -2.0).conj().y" => "2.0",
    // z * conj(z) = |z|^2 on the real axis
    "Vec2::new(3.0, 4.0).cmul(Vec2::new(3.0, 4.0).conj()).x" => "25.0",
    "Vec2::new(3.0, 4.0).cmul(Vec2::new(3.0, 4.0).conj()).y" => "0.0",
    "(Vec2::new(0.0, 1.0).arg().to(\"deg\") - 90.0).abs() < 1.0e-4" => "true",
    "(Vec2::from_polar(2.0, 90deg).y - 2.0).abs() < 1.0e-4" => "true",
    // exp(i*pi) = -1
    "(Vec2::new(0.0, std::math::PI).exp().x + 1.0).abs() < 1.0e-6" => "true",
);

// r[verify mimas.unit.angle]
test_run!(
    vec3_ops,
    "use std::math::Vec3; let a = Vec3::new(1.0, 0.0, 0.0); let b = Vec3::new(0.0, 1.0, 0.0);",
    "(a + b).z" => "0.0",
    "(a - b).y" => "-1.0",
    "a.dot(b)" => "0.0",
    "a.cross(b).z" => "1.0",
    "a.distance(Vec3::new(4.0, 0.0, 0.0))" => "3.0",
    "(a.angle(b).to(\"deg\") - 90.0).abs() < 1.0e-4" => "true",
    "a.lerp(Vec3::new(3.0, 0.0, 0.0), 0.5).x" => "2.0",
    "(a * 2.0).x" => "2.0",
    "(2.0 * a).x" => "2.0",
    "(a / 0.5).x" => "2.0",
    "(-a).x" => "-1.0",
);

// r[verify mimas.unit.angle]
test_run!(
    quat_ops,
    "use std::math::{Quat, Vec3};",
    // half-turn about z maps +x to -x
    "(Quat::from_rotation_z(180deg).rotate(Vec3::new(1.0, 0.0, 0.0)).x + 1.0).abs() < 1.0e-4" => "true",
    // quat * inverse = identity (w = 1); conj⁻¹ = q for unit quats
    "(Quat::from_rotation_y(30deg).mul(Quat::from_rotation_y(30deg).inverse()).w - 1.0).abs() < 1.0e-4" => "true",
    "(Quat::from_rotation_y(30deg).conj().inverse().w - Quat::from_rotation_y(30deg).w).abs() < 1.0e-6" => "true",
    // composition: 90deg about z then 90deg about z = 180deg about z
    "((Quat::from_rotation_z(90deg) * Quat::from_rotation_z(90deg)).angle().to(\"deg\") - 180.0).abs() < 1.0e-3" => "true",
    "(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 90deg).axis().z - 1.0).abs() < 1.0e-4" => "true",
    "(Quat::from_rotation_z(90deg).slerp(Quat::from_rotation_z(180deg), 0.5).angle().to(\"deg\") - 135.0).abs() < 1.0e-3" => "true",
);

// `float::INF` and friends: the IEEE corners as associated constants on
// the builtin, like `Player::MAX_HEALTH` on an adt, and the predicates
// that are the only honest way to test for them.
test_run!(
    float_constants,
    "float::INF > float::MAX" => "true",
    "float::NEG_INF < float::MIN" => "true",
    "-float::INF == float::NEG_INF" => "true",
    "1.0 / 0.0 == float::INF" => "true",
    "float::INF.is_infinite()" => "true",
    "float::INF.is_finite()" => "false",
    "float::NAN == float::NAN" => "false",
    "float::NAN.is_nan()" => "true",
    "(0.0 / 0.0).is_nan()" => "true",
    "float::NAN.is_infinite()" => "false",
    "1.0.is_finite() && 1.0e300.is_finite()" => "true",
    "float::EPSILON > 0.0 && float::EPSILON < 1.0e-15" => "true",
    "1.0 + float::EPSILON > 1.0" => "true",
    "float::MIN_POSITIVE > 0.0 && float::MIN_POSITIVE < 1.0e-300" => "true",
    "float::MAX > 1.0e308" => "true",
    "float::MIN == -float::MAX" => "true",
    // the running-minimum idiom the constants exist for
    "{ let d = float::INF; for x in [3.0, 1.5, 2.0] { d = d.min(x); } d }" => "1.5",
    "float::INF.min(4.0)" => "4.0",
);

test_run!(
    int_constants,
    "int::MAX" => "9223372036854775807",
    "int::MIN" => "-9223372036854775808",
    "int::MAX > 0 && int::MIN < 0" => "true",
);

test_run!(
    // r[verify mimas.math.vec2-names]
    // L9: the game kit's short names are the same functions
    vec2_short_names,
    "use std::math::Vec2;\nuse std::math::Vec3;",
    "Vec2::new(3.0, 4.0).len()" => "5.0",
    "Vec2::new(0.0, 2.0).norm().y" => "1.0",
    "Vec2::new(0.0, 0.0).dist(Vec2::new(3.0, 4.0))" => "5.0",
    "Vec3::new(0.0, 3.0, 4.0).len()" => "5.0",
);

// One `Vec2`, in f64, with the game kit's operators: `+ - * /` are
// component-wise between two vectors and broadcast a scalar on either
// side, `-v` negates; `dot` stays a method, `cmul` the complex multiply,
// and the short and long names both work.
// r[verify mimas.math.vec-ops]
test_run!(
    vec2_operator_forms,
    "use std::math::Vec2; let a = Vec2::new(1.0, 2.0); let b = Vec2::new(3.0, 4.0);",
    "a + b" => "{ 4.0, 6.0 }",
    "a - b" => "{ -2.0, -2.0 }",
    "a * b" => "{ 3.0, 8.0 }",
    "(a / b).y" => "0.5",
    "a * 2.0" => "{ 2.0, 4.0 }",
    "2.0 * a" => "{ 2.0, 4.0 }",
    "a * 2" => "{ 2.0, 4.0 }",
    "a + 1.0" => "{ 2.0, 3.0 }",
    "1.0 - a" => "{ 0.0, -1.0 }",
    "a / 2.0" => "{ 0.5, 1.0 }",
    "2.0 / a" => "{ 2.0, 1.0 }",
    "-a" => "{ -1.0, -2.0 }",
    "a.dot(b)" => "11.0",
    "a.cmul(b)" => "{ -5.0, 10.0 }",
    "a.lerp(b, 0.5)" => "{ 2.0, 3.0 }",
    "a.len() == a.length()" => "true",
    "a.norm() == a.normalize()" => "true",
    "a.dist(b) == a.distance(b)" => "true",
    "Vec2::new(0.0, 0.0).norm()" => "{ 0.0, 0.0 }",
);

// The components are f64: a value round-trips exactly.
// r[verify mimas.math.vec2-f64]
test_run!(
    vec2_holds_f64,
    "use std::math::{Vec2, Vec3}; let p = Vec2::new(0.1, 0.2);",
    "p.x + p.y" => "0.30000000000000004",
    "Vec2::new(1.0e300, 1.0).x" => "1e300",
    "Vec3::new(1.0, 2.0, 3.0) * Vec3::new(1.0, 2.0, 3.0)" => "{ 1.0, 4.0, 9.0 }",
);
