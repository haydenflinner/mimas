//! Axes and directions: `up 50.0px` is a `px` along y with the sign folded in; `px↕` in a
//! type says which axis; `+ - < ==` need agreeing axes (an untagged side adopts); `* /`
//! xor them. See the units book page and `plan.md` in the host repo.

#[macro_use]
mod test_runner;

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
}

#[test]
fn a_direction_word_is_only_a_word_when_declared() {
    // `up` as a name: no declaration makes it a direction in front of an operator
    assert!(test_runner::try_execute("let up = 3.0; let a = up + 1.0;").is_ok());
    // but in front of a quantity it is the builtin direction
    assert!(test_runner::try_execute("let a = up 3.0px;").is_ok());
}
