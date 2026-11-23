//! Axes and directions: `50.0px↑` is a `px` claimed at the `↑` convention —
//! the value reads "50 in up-positive terms". A direction is a *convention
//! claim*, not a negation: `50.0px↓` is +50 measured down-positive. Values
//! mix conventions freely — `x↓ + y↑` converts the `↓` side (the checker
//! negates it, the result reports the positive end), and a `px↓` argument
//! into a `wpx↑` parameter converts at the boundary. `px↕` in a type is the
//! unsigned axis claim — it adopts either convention. `x@y` tags an axis
//! unsigned; `x@word` spells a direction with no glyph (`5usd@in`).
//! `+ - < ==` need agreeing axes (an untagged side adopts); `* /` xor them
//! and read every signed factor at its positive end.
//! See the units book page and `plan.md` in the host repo.

use crate::test_runner;

// r[verify mimas.unit.direction.signed]
test_run!(
    direction_words_claim_a_convention,
    // `50↓` is 50 in the down convention — +50 stored, and `.to` reads
    // it back convention-free
    "(50.0px↑).to(\"px\")" => "50.0",
    "(50.0px↓).to(\"px\")" => "50.0",
    "(2.0px←).to(\"px\")" => "2.0",
    "(2.0px→).to(\"px\")" => "2.0",
    "(3s⏪).to(\"s\")" => "3.0",
    "(3s⏩).to(\"s\")" => "3.0",
    // `.to` can name the convention: "50 up" written in down-terms is −50
    "(50.0px↑).to(\"px↓\")" => "-50.0",
    "(50.0px↓).to(\"px↑\")" => "-50.0",
    // mixed conventions add fine — the `↓` side negates to the positive end
    "(50.0px↑ + 20.0px↓).to(\"px\")" => "30.0",
    "(50.0px↓ + 20.0px↑).to(\"px↓\")" => "30.0",
);

// r[verify mimas.unit.direction.signed]
test_run!(
    direction_glyphs_claim_the_same_convention,
    "let v: px↕ = 8.0px",
    "(50.0px↑).to(\"px\")" => "50.0",
    "(50.0px↓).to(\"px\")" => "50.0",
    // a glyph follows any operand, not just literals — `v↓` claims the
    // unsigned value at the down convention
    "(v↑).to(\"px\")" => "8.0",
    "(v↓).to(\"px\")" => "8.0",
    // the physics pair reads as z: `⊙` out of the page (negative end),
    // `⊗` into it (positive)
    "(5.0px⊙).to(\"px\")" => "5.0",
    "(5.0px⊗).to(\"px\")" => "5.0",
    // `x@axis` tags without claiming an end
    "(v@y).to(\"px\")" => "8.0",
);

// r[verify mimas.unit.axis]
#[test]
fn a_rename_flips_the_glyph_with_the_word() {
    // `axis y (up, down)` makes `up` the negative end — `↑` spells the
    // word, so `50↑` claims that end's convention instead
    assert_eq!(
        test_runner::render("axis y (up, down);", "(50.0px↑).to(\"px↓\")"),
        "-50.0"
    );
    assert_eq!(
        test_runner::render("axis y (up, down);", "(50.0px↓).to(\"px\")"),
        "50.0"
    );
    // a renamed word with no glyph spells `@word`
    assert_eq!(
        test_runner::render("axis y (south, north);", "(50.0px@north).to(\"px\")"),
        "50.0"
    );
}

test_run!(
    untagged_adopts_and_products_xor,
    "const BALL = 18.0px
     let x: px↔ = 40.0px↔;",
    "(x + BALL).to(\"px\")" => "58.0",
    "((2.0px↑) / (4.0px↓))" => "-0.5",
    "((2.0px↑) * (3.0px←)).to(\"px*px\")" => "-6.0",   // axes xor away, factors read positive-end
    "(460.0px/s↑ * 0.5s).to(\"px\")" => "230.0",
);

test_run!(
    axes_through_functions_and_annotations,
    "fn drop(y: px↕, by: px↕) -> px↕ { y + by }
     fn paddle(top: px↕) -> px↕ { top + 110.0px↓ }
     axis flow (out, in);
     fn pay(bal: usd@flow, amt: usd@flow) -> usd@flow { bal + amt }",
    "drop(300.0px↕, 20.0px↑).to(\"px\")" => "320.0",
    "paddle(10.0px).to(\"px↓\")" => "120.0",
    "pay(10usd@in, 4usd@out).to(\"usd\")" => "6.0",
);

// r[verify mimas.unit.direction.signed]
test_run!(
    conventions_convert_at_the_boundary,
    // a `wpx↑` parameter declares the up convention — a `↓`-claimed
    // value negates as it crosses
    "fn rise(dy: wpx↑) -> wpx↑ { dy }
     fn fall(dy: wpx↓) -> wpx↓ { dy }
     fn slide(dy: wpx↕) -> wpx↕ { dy }
     let base: wpx↑ = 100.0wpx
     let u: wpx↕ = 8.0wpx",
    "rise(40.0wpx↓).to(\"wpx\")" => "-40.0",
    "fall(40.0wpx↑).to(\"wpx\")" => "-40.0",
    // an unsigned slot takes any convention, reading it positive-end:
    // "40 down" arrives as −40
    "slide(40.0wpx↓).to(\"wpx\")" => "-40.0",
    // and a `↕`-written value claims an end at the boundary
    "rise(u).to(\"wpx\")" => "8.0",
    "rise(base + 40.0wpx↓).to(\"wpx\")" => "60.0",
);

// r[verify mimas.unit.direction.signed]
#[test]
fn opposite_conventions_in_composites_still_reject() {
    // a scalar converts; a tuple's element can't be negated wholesale
    rejected("fn f(p: (wpx↑, wpx↑)) -> wpx↑ { p.0 }\nlet a = f((1.0wpx↓, 2.0wpx↑));", "wpx↓");
    rejected("fn f(p: (wpx↑, wpx↑)) -> wpx↑ { p.0 }\nlet a = f((1.0wpx↓, 2.0wpx↑));", "expected");
    // and the seams still hold: `wpx↓` never converts to `spx↓`
    rejected("fn f(y: spx↓) -> spx↓ { y }\nlet a = f(1.0wpx↓);", "expected");
}

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
    rejected("let a = 5.0px↑ + 3.0px←;", "cannot add");
    rejected("let a = 5.0px↑ + 3.0px←;", "along y");
    rejected("let a = 5.0px↑ + 3.0px←;", "along x");
    rejected("let a: px↕ = 3.0px→;", "along x");
    rejected("let a = (2.0px↑) * (3.0px←); let b: px↕ = a + 1.0px;", "along");
    rejected("fn f(y: px↕) -> px↕ { y }\n let a = f(3.0px→);", "along");
    rejected("axis flow (out, in);\n let a = 5usd@in + 5usd↑;", "cannot add");
    rejected("let a = (5.0px↑ < 3.0px←);", "cannot order");
    // `atan2(dy, dx)` reads an (x, y) vector: the legs' axes differ on
    // purpose, only the bare dims must agree. `angle_to` is the same
    // call under a name that says what it returns.
    assert!(test_runner::try_execute("let dx: px↔ = 3.0px; let dy: px↕ = 4.0px; let a = dy.atan2(dx);").is_ok());
    assert!(test_runner::try_execute("let dx: px↔ = 3.0px; let dy: px↕ = 4.0px; let a = dy.angle_to(dx);").is_ok());
    rejected("let dy: px↕ = 4.0px; let a = dy.atan2(2.0s);", "atan2");
    rejected("let dy: px↕ = 4.0px; let a = dy.angle_to(2.0s);", "angle_to");
}

#[test]
fn the_old_prefix_spelling_is_a_named_mistake() {
    rejected("let a = ↑50.0px;", "a direction follows its operand");
    rejected("let a = ↓ x;", "a direction follows its operand");
    // the word form in front is a call to a name that doesn't exist —
    // and the error says where the direction went
    rejected("let a = up 50.0px;", "directions follow their operand");
    rejected(
        "axis flow (out, in);\nlet a = out 5usd;",
        "directions follow their operand",
    );
    // `up` is still a usable name everywhere else
    assert!(test_runner::try_execute("let up = 3.0; let a = up + 1.0;").is_ok());
    // and a bogus `@word` names its own mistake
    rejected("let a = 5.0px@bogus;", "neither an axis nor a direction");
}

// r[verify mimas.unit.direction.postfix]
#[test]
fn in_is_a_direction_word_and_still_a_keyword() {
    // `in` is the loop/membership keyword — as a direction word it only
    // ever follows `@` or sits inside `axis (…)`, so both reads coexist
    assert!(test_runner::try_execute("axis flow (out, in);\nlet a = 5usd@in;").is_ok());
    assert_eq!(
        test_runner::render("axis flow (out, in);", "(5usd@in + 4usd@out).to(\"usd\")"),
        "1.0"
    );
    // `x@in` on a name takes the same suffix as the literal
    assert_eq!(
        test_runner::render("axis flow (out, in);\nlet v = 3.0usd;", "(v@in).to(\"usd\")"),
        "3.0"
    );
    // and `in`/`∈` keep membership
    assert_eq!(test_runner::render("", "2 in [1, 2, 3]"), "true");
    assert_eq!(test_runner::render("", "2 ∈ [1, 2, 3]"), "true");
}

// r[verify mimas.unit.direction.postfix]
#[test]
fn a_direction_with_a_glyph_spells_the_glyph() {
    // `x@up` is a second spelling for `x↑` — the error points at the glyph
    rejected("let a = 5.0px@up;", "has a glyph");
    rejected("let a = 5.0px@up;", "write `↑` after the value");
    rejected("let a = 3.0s@earlier;", "has a glyph");
    // `inward` spells `⊗` — the physics pair `⊙`/`⊗` (out of, into the
    // page) reads as the z axis now that tensor is `⊠`
    rejected("let a = 5.0px@inward;", "has a glyph");
    rejected("let a = 5.0px@inward;", "write `⊗` after the value");
    // a declared direction with no glyph keeps `@word`, like `in`
    assert!(test_runner::try_execute(
        "axis flow (out, in);\nlet a = 5usd@in;"
    )
    .is_ok());
}

// r[verify mimas.unit.tuple]
test_run!(
    tuples_keep_each_slots_axis,
    "let p: (px↔, px↕) = (3.0px, 4.0px↑)
     let (a, b) = p
     let pts: [(px↔, px↕)] = [(1.0px, 2.0px), (3.0px, 4.0px)]",
    "p.0.to(\"px\")" => "3.0",
    "p.1.to(\"px\")" => "4.0",
    "a.to(\"px\")" => "3.0",
    "b.to(\"px\")" => "4.0",
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
    "let p: (px↔, px↕) = (3.0px, 4.0px↑)
     let t = (1.0, 2.0, 3.0, 4.0)",
    "p.x.to(\"px\")" => "3.0",
    "p.y.to(\"px\")" => "4.0",
    "t.z" => "3.0",
    "t.t" => "4.0",
    // the slot's own tag survives the read: `p.x` stays along x
    "(p.x + 1.0px).to(\"px\")" => "4.0",
    "(p.x - 1.0px→).to(\"px\")" => "2.0",
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
    "(5.0px↑).to(1px)" => "5.0",
    "(3.0ft→).to(1mi)" => "0.0005681818181818183",
    "(3.0ft→).to(\"ft\")" => "3.0",
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
    rejected("let p: (px↔, px↕) = (1.0px↑, 2.0px);", "along y");
    rejected("let p: (px↔, px↕) = (4.0px, 3.0px→);", "along x");
    rejected("let p: (px↔, px↕) = (4.0px, 3.0px→);", "along y");
    rejected(
        "struct Pt { x: px↔, y: px↕ }\n let p = Pt { x = 1.0px, y = 2.0px→ };",
        "along x",
    );
    // a list of tuples keeps the check per element
    rejected("let pts: [(px↔, px↕)] = [(1.0px, 2.0px), (3.0px, 4.0px←)];", "along x");
    // a destructure hands each name its slot's axis: `c` is along x here
    rejected("let (c, d) = (1.0px→, 2.0px);\n let e: px↕ = c;", "along x");
}
