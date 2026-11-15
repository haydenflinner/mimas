//! Axes and directions: `up 50.0px` is a `px` along y with the sign folded in; `px↕` in a
//! type says which axis; `+ - < ==` need agreeing axes (an untagged side adopts); `* /`
//! xor them. See the units book page and `plan.md` in the host repo.

#[macro_use]
mod test_runner;

// r[verify mimas.unit.axis]
test_run!(
    direction_words_fold_the_sign,
    "(up 50.0px).to(\"px\")" => "-50.0",
    "(down 50.0px).to(\"px\")" => "50.0",
    "(left 2.0px).to(\"px\")" => "-2.0",
    "(right 2.0px).to(\"px\")" => "2.0",
    "(← 2.0px).to(\"px\")" => "-2.0",
    "(↓ 3.0px).to(\"px\")" => "3.0",
    "(earlier 3s).to(\"s\")" => "-3.0",
    "(⏩ 3s).to(\"s\")" => "3.0",
    "(up 50.0px + down 20.0px).to(\"px\")" => "-30.0",
);

test_run!(
    untagged_adopts_and_products_xor,
    "const BALL = 18.0px
     let x: px↔ = 40.0px↔;",
    "(x + BALL).to(\"px\")" => "58.0",
    "((up 2.0px) / (down 4.0px))" => "-0.5",
    "((up 2.0px) * (left 3.0px)).to(\"px*px\")" => "6.0",   // two negatives: the axes xor away, the signs multiply
    "(up 460.0px/s * 0.5s).to(\"px\")" => "-230.0",
);

test_run!(
    axes_through_functions_and_annotations,
    "fn drop(y: px↕, by: px↕) -> px↕ { y + by }
     fn paddle(top: px↕) -> px↕ { top + down 110.0px }
     axis wealth (poorer, richer);
     fn pay(bal: usd@wealth, amt: usd@wealth) -> usd@wealth { bal + amt }",
    "drop(300.0px↕, up 20.0px).to(\"px\")" => "280.0",
    "paddle(10.0px).to(\"px\")" => "120.0",
    "pay(richer 10usd, poorer 4usd).to(\"usd\")" => "6.0",
);

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(
        text.contains(needle),
        "`{src}`: expected `{needle}` in:\n{text}"
    );
}

// r[verify mimas.unit.axis]
#[test]
fn cross_axis_arithmetic_is_rejected() {
    rejected("let a = up 5.0px + left 3.0px;", "cannot add");
    rejected("let a = up 5.0px + left 3.0px;", "along y");
    rejected("let a = up 5.0px + left 3.0px;", "along x");
    rejected("let a: px↕ = right 3.0px;", "along x");
    rejected("let a = (up 2.0px) * (left 3.0px); let b: px↕ = a + 1.0px;", "along");
    rejected("fn f(y: px↕) -> px↕ { y }\n let a = f(right 3.0px);", "along");
    rejected("axis wealth (poorer, richer);\n let a = richer 5usd + up 5usd;", "cannot add");
    rejected("let a = (up 5.0px < left 3.0px);", "cannot order");
    // `atan2(dy, dx)` reads an (x, y) vector: the legs' axes differ on
    // purpose, only the bare dims must agree. `angle_to` is the same
    // call under a name that says what it returns.
    assert!(test_runner::try_execute("let dx: px↔ = 3.0px; let dy: px↕ = 4.0px; let a = dy.atan2(dx);").is_ok());
    assert!(test_runner::try_execute("let dx: px↔ = 3.0px; let dy: px↕ = 4.0px; let a = dy.angle_to(dx);").is_ok());
    rejected("let dy: px↕ = 4.0px; let a = dy.atan2(2.0s);", "atan2");
    rejected("let dy: px↕ = 4.0px; let a = dy.angle_to(2.0s);", "angle_to");
}

#[test]
fn a_direction_word_is_only_a_word_when_declared() {
    // `up` as a name: no declaration makes it a direction in front of an operator
    assert!(test_runner::try_execute("let up = 3.0; let a = up + 1.0;").is_ok());
    // but in front of a quantity it is the builtin direction
    assert!(test_runner::try_execute("let a = up 3.0px;").is_ok());
}

// r[verify mimas.unit.tuple]
test_run!(
    tuples_keep_each_slots_axis,
    "let p: (px↔, px↕) = (3.0px, up 4.0px)
     let (a, b) = p
     let pts: [(px↔, px↕)] = [(1.0px, 2.0px), (3.0px, 4.0px)]",
    "p.0.to(\"px\")" => "3.0",
    "p.1.to(\"px\")" => "-4.0",
    "a.to(\"px\")" => "3.0",
    "b.to(\"px\")" => "-4.0",
    "pts[0].1.to(\"px\")" => "2.0",
    // an untagged pair still adopts, slot by slot
    "(3.0px, 4.0px).1.to(\"px\")" => "4.0",
);

// r[verify mimas.unit.norm]
test_run!(
    norms_are_axis_parities,
    "let lx: snorm = 0.5
     let lt: unorm = 0.8",
    // untagged numbers adopt the norm — a literal never needs the name
    "lx + 0.1" => "0.6",
    "lx * 0.5" => "0.25",
    "lx.to(\"snorm\")" => "0.5",
    // `*` keeps the parity against a plain scalar, cancels against itself
    "(lx * lx).to(\"snorm\")" => "0.25",
);

// r[verify mimas.unit.norm]
#[test]
fn norms_reject_cross_channel_and_quantities() {
    rejected("let lx: snorm = 0.5; let lt: unorm = 0.8; let bad = lx + lt;", "along snorm");
    rejected("let lx: snorm = 0.5; let lt: unorm = 0.8; let bad = lx + lt;", "along unorm");
    rejected("let worse: snorm = 3.0px;", "pixels (px)");
    // a norm is a unit of its own: `snorm` is the whole spelling
    assert!(test_runner::try_execute("let v: snorm = 0.5; let w: snorm = v;").is_ok());
}

// r[verify mimas.unit.norm]
#[test]
fn norm_literals_out_of_range_reject() {
    rejected("let v: snorm = 1.5;", "snorm");
    rejected("let v: snorm = 1.5;", "[-1, 1]");
    rejected("let v: unorm = -0.5;", "unorm");
    rejected("let v: unorm = -0.5;", "[0, 1]");
    rejected("let v: snorm = 2.0snorm;", "out of range");
    // fields, args and returns check at the same boundary
    rejected("struct P { v: snorm }\nlet p = P { v = 2.0 };", "out of range");
    rejected("fn stick(v: snorm) -> snorm { v }\nlet s = stick(1.5);", "out of range");
    rejected("fn f() -> snorm { 2.0 }\nlet s = f();", "out of range");
    // computed values pass — the check is literals-only for now
    assert!(test_runner::try_execute("let a = 3.0; let v: snorm = a * 0.5;").is_ok());
    // in-range literals still fine, sign included
    assert!(test_runner::try_execute("let v: snorm = -1.0; let w: unorm = 0.0; let u: unorm = 1.0;").is_ok());
}

// r[verify mimas.tuple.axis]
test_run!(
    axis_names_spell_tuple_slots,
    "let p: (px↔, px↕) = (3.0px, up 4.0px)
     let t = (1.0, 2.0, 3.0, 4.0)",
    "p.x.to(\"px\")" => "3.0",
    "p.y.to(\"px\")" => "-4.0",
    "t.z" => "3.0",
    "t.t" => "4.0",
    // the slot's own tag survives the read: `p.x` stays along x
    "(p.x + 1.0px).to(\"px\")" => "4.0",
    "(p.x - right 1.0px).to(\"px\")" => "2.0",
);

// r[verify mimas.tuple.axis]
#[test]
fn tuple_axis_names_reject_other_idents() {
    rejected("let p = (1.0, 2.0);\nlet a = p.w;", "dot-indexed");
    rejected("let p = (1.0, 2.0);\nlet a = p.xx;", "dot-indexed");
    // and the tags they read still clash across axes
    rejected("let p: (px↔, px↕) = (1.0px, 2.0px);\nlet a = p.x + p.y;", "cannot add");
}

// r[verify mimas.unit.to]
test_run!(
    to_reads_back_across_an_axis,
    "let span: px↔ = 40.0px",
    // the quantity and the string spellings agree: `to` reads a plain
    // number back, the axis is direction metadata, not a different unit
    "span.to(1px)" => "40.0",
    "span.to(1px↔)" => "40.0",
    "(up 5.0px).to(1px)" => "-5.0",
    "(right 3.0ft).to(1mi)" => "0.0005681818181818183",
    "(right 3.0ft).to(\"ft\")" => "3.0",
);

// r[verify mimas.unit.to]
#[test]
fn to_still_refuses_a_real_mismatch() {
    rejected("let span: px↔ = 40.0px;\nlet a = span.to(1px↕);", "cannot express");
    rejected("let span: px↔ = 40.0px;\nlet a = span.to(1s);", "cannot express");
    rejected("let span: px↔ = 40.0px;\nlet a = span.to(\"s\");", "cannot express");
}

// r[verify mimas.unit.tuple]
#[test]
fn tuple_slots_reject_a_swapped_axis() {
    rejected("let p: (px↔, px↕) = (up 1.0px, 2.0px);", "along y");
    rejected("let p: (px↔, px↕) = (4.0px, right 3.0px);", "along x");
    rejected("let p: (px↔, px↕) = (4.0px, right 3.0px);", "along y");
    rejected(
        "struct Pt { x: px↔, y: px↕ }\n let p = Pt { x = 1.0px, y = right 2.0px };",
        "along x",
    );
    // a list of tuples keeps the check per element
    rejected("let pts: [(px↔, px↕)] = [(1.0px, 2.0px), (3.0px, left 4.0px)];", "along x");
    // a destructure hands each name its slot's axis: `c` is along x here
    rejected("let (c, d) = (right 1.0px, 2.0px);\n let e: px↕ = c;", "along x");
}
