//! Atoms in the checker: the inferred sets, the did-you-mean, the closed-set
//! exhaustiveness, the fit into declared enums, and the open set that makes
//! no false errors.

/// The checker's message for `src`, which must be rejected.
fn rejects(src: &str) -> String {
    use super::solve_test_utils::*;
    let _t = TestResetter;
    match TEST_SESSION.with(|s| s.borrow_mut().run(src, "test")) {
        Ok(()) => panic!("expected solve to reject:\n{src}"),
        Err(e) => format!("{e:?}"),
    }
}

const HEAD: &str = "fn head(n: int) -> int! {
    if n == 0 { raise #EmptyList }
    n
}
";

// r[verify mimas.atom.literal]
// r[verify mimas.atom.type]
test_ty!(
    atom_is_an_atom,
    "let a = #EmptyList; let b: atom = #ParseError(line: 3); let c = [#A, #B];",
);

// r[verify mimas.atom.set]
// r[verify mimas.atom.match.exhaustive]
test_ty!(
    closed_raise_set_is_exhaustive,
    "fn head(n: int) -> int! {
        if n == 0 { raise #EmptyList }
        n
    }
    fn first_or(n: int, d: int) -> int {
        head(n) absolve |e| match e { #EmptyList => d }
    }",
);

// r[verify mimas.atom.set]
test_ty!(
    closed_value_set_from_branches,
    "let k = if 1 > 0 { #Many } else { #Few };
    let s = match k { #Many => 1, #Few => 2 };",
);

// r[verify mimas.atom.match.unknown]
#[test]
fn typo_in_handler_arm_says_did_you_mean() {
    let msg = rejects(&format!(
        "{HEAD}let x = head(0) absolve |e| match e {{ #EmptyLst => 0, _ => 1 }};"
    ));
    assert!(msg.contains("no code makes `#EmptyLst`"), "{msg}");
    assert!(msg.contains("did you mean `#EmptyList`?"), "{msg}");
}

// r[verify mimas.atom.match.unknown]
#[test]
fn arm_outside_the_closed_set_is_an_error() {
    let msg = rejects(&format!(
        "{HEAD}let t = #Timeout;
        let x = head(0) absolve |e| match e {{ #Timeout => 0, #EmptyList => 1 }};"
    ));
    assert!(msg.contains("`#Timeout` can't reach this match"), "{msg}");
}

// r[verify mimas.atom.match.exhaustive]
#[test]
fn missing_arm_of_a_closed_set_is_an_error() {
    let msg = rejects(
        "fn g(c: bool) -> int! {
            if c { raise #EmptyList }
            if !c { raise #Timeout }
            1
        }
        let x = g(true) absolve |e| match e { #EmptyList => 0 };",
    );
    assert!(msg.contains("missing atom `#Timeout`"), "{msg}");
}

// r[verify mimas.atom.set]
// an open set (a parameter typed only `atom`) names no unknown or missing
// atom: it asks for the `_` arm and nothing more
test_ty!(
    open_set_makes_no_false_errors,
    "fn k(a: atom) -> int { match a { #A => 1, _ => 2 } }
    let x = k(#A) + k(#B);",
);

// r[verify mimas.atom.set]
test_fail!(open_set_needs_a_wildcard, "fn k(a: atom) -> int { match a { #A => 1 } } let x = k(#A);");

// r[verify mimas.atom.set]
// a raised string opens the set: no missing-arm error, `_` stays required
test_ty!(
    raised_string_opens_the_set,
    "fn g(c: bool) -> int! {
        if c { raise #EmptyList }
        raise \"other\"
    }
    let x = g(true) absolve |e| match e { #EmptyList => 0, _ => 1 };",
);

// r[verify mimas.atom.enum]
test_ty!(
    atom_fits_a_declared_enum,
    "enum Day { Mon, Tue, Wed }
    fn f(d: Day) -> bool { d == #Wed }
    let a = f(#Wed);
    let b: Day = #Tue;
    let n = match b { #Mon => 1, #Tue => 2, #Wed => 3 };",
);

// r[verify mimas.atom.enum]
test_ty!(
    payload_atom_fits_a_variant,
    "enum Shape { Circle { r: float }, Sq(float) }
    fn area(s: Shape) -> float { match s { #Circle { r } => r * r, #Sq(a) => a * a } }
    let x = area(#Circle(r: 2.0)) + area(#Sq(3.0));",
);

// r[verify mimas.atom.enum.missing]
#[test]
fn atom_outside_the_enum_names_its_variants() {
    let msg = rejects("enum Day { Mon, Tue, Wed } let d: Day = #Wedd;");
    assert!(msg.contains("`#Wedd` is not a variant of `Day`"), "{msg}");
    assert!(msg.contains("`#Mon`, `#Tue`, `#Wed`"), "{msg}");
    assert!(msg.contains("did you mean `#Wed`?"), "{msg}");
}

// r[verify mimas.atom.enum]
test_fail!(
    payload_must_fit_the_variant,
    "enum Shape { Sq(float) } fn f(s: Shape) -> int { 1 } let x = f(#Sq(\"x\"));",
);

// r[verify mimas.atom.payload]
#[test]
fn one_atom_has_one_shape() {
    let msg = rejects("let a = #P(1); let b = #P(x: 1);");
    assert!(msg.contains("one atom has one payload shape"), "{msg}");
}

// r[verify mimas.atom.payload]
test_fail!(payload_field_types_agree, "let a = #P(1); let b = #P(\"x\");");
