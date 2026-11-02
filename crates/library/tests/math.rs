#[macro_use]
mod test_runner;

test_run!(
    vec2_math,
    "use std::math::Vec2;",
    "Vec2::new(3.0, 4.0).length()" => "5",
    "Vec2 { x = 1.0, y = 2.0 }.add(Vec2::new(2.0, 3.0)).y" => "5",
    "Vec2::new(0.0, 2.0).normalize().y" => "1",
    "Vec2::from_angle(0.0rad).x" => "1",
    "(Vec2::from_angle(90deg).y - 1.0).abs() < 1.0e-4" => "true",
);

test_run!(
    vec3_and_quat_math,
    "use std::math::{Vec3, Quat};",
    "Vec3::new(2.0, 3.0, 6.0).length()" => "7",
    "Quat::from_rotation_z(0.0rad).rotate_z(0.0rad).w" => "1",
);

// r[verify mimas.unit.angle]
// r[verify mimas.unit.trig]
test_run!(
    vec2_complex_plane,
    "use std::math::Vec2;",
    // i * i = -1
    "Vec2::new(0.0, 1.0).cmul(Vec2::new(0.0, 1.0)).x" => "-1",
    // (1+2i)(3-1i) = 5+5i
    "Vec2::new(1.0, 2.0).cmul(Vec2::new(3.0, -1.0)).x" => "5",
    "Vec2::new(1.0, 2.0).cmul(Vec2::new(3.0, -1.0)).y" => "5",
    "Vec2::new(1.0, -2.0).conj().y" => "2",
    // z * conj(z) = |z|^2 on the real axis
    "Vec2::new(3.0, 4.0).cmul(Vec2::new(3.0, 4.0).conj()).x" => "25",
    "Vec2::new(3.0, 4.0).cmul(Vec2::new(3.0, 4.0).conj()).y" => "0",
    "(Vec2::new(0.0, 1.0).arg().to(\"deg\") - 90.0).abs() < 1.0e-4" => "true",
    "(Vec2::from_polar(2.0, 90deg).y - 2.0).abs() < 1.0e-4" => "true",
    // exp(i*pi) = -1
    "(Vec2::new(0.0, std::math::PI).exp().x + 1.0).abs() < 1.0e-6" => "true",
);

// r[verify mimas.unit.angle]
test_run!(
    vec3_ops,
    "use std::math::Vec3; let a = Vec3::new(1.0, 0.0, 0.0); let b = Vec3::new(0.0, 1.0, 0.0);",
    "(a + b).z" => "0",
    "(a - b).y" => "-1",
    "a.dot(b)" => "0",
    "a.cross(b).z" => "1",
    "a.distance(Vec3::new(4.0, 0.0, 0.0))" => "3",
    "(a.angle(b).to(\"deg\") - 90.0).abs() < 1.0e-4" => "true",
    "a.lerp(Vec3::new(3.0, 0.0, 0.0), 0.5).x" => "2",
    "(a * 2.0).x" => "2",
    "(2.0 * a).x" => "2",
    "(a / 0.5).x" => "2",
    "(-a).x" => "-1",
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
