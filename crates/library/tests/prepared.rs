// `solve::Prepared`: the library and the shipped sources taken in once, a
// program solved on a clone of that state — sans I/O, so these are plain
// assertions about state, not fixtures.
use vm::Vm;

fn std_library() -> api::Library<()> {
    Vm::new().install_library(library::std)
}

/// Solving on a prepared solver gives the same answer as the one-shot
/// loader, and a program's own unit declarations and types never reach
/// the next program solved on the same preparation.
#[test]
fn prepared_solves_programs_apart() {
    let lib = std_library();
    let prepared = solve::Prepared::new(&lib).expect("the shipped sources solve");

    // a unit declared by one program …
    let a = prepared.solve([("a", "unit tile\nlet x = 4tile;\nlet y = x + 2tile;")]);
    assert!(a.errors.is_empty(), "{:?}", a.errors);
    // … is unknown to the next: the table went back to the shipped state
    let b = prepared.solve([("b", "let x = 4tile;")]);
    assert!(
        !b.errors.is_empty(),
        "`tile` leaked from the previous program"
    );
    // … and a third program with its own declaration is unaffected
    let c = prepared.solve([("c", "unit tile\nlet x = 4tile;")]);
    assert!(c.errors.is_empty(), "{:?}", c.errors);

    // a program's own enum shadows a built-in one for that program only
    let d = prepared.solve([("d", "enum PosNeg { Up, Down }\nlet p = PosNeg::Up;")]);
    assert!(d.errors.is_empty(), "{:?}", d.errors);
    let e = prepared.solve([("e", "let p = PosNeg::Pos;")]);
    assert!(
        e.errors.is_empty(),
        "the built-in `PosNeg` came back: {:?}",
        e.errors
    );

    // what the one-shot loader says, the prepared one says
    let src = "fn f(n: int) -> int { n * 2 }\nlet q = f(3);\n";
    let one = solve::load_files([("page", src)], &lib);
    let two = prepared.solve([("page", src)]);
    assert!(one.errors.is_empty() && two.errors.is_empty());
    assert_eq!(
        one.asts.len(),
        two.asts.len(),
        "the shipped asts follow the program's"
    );
    assert_eq!(one.asts[0].name(), "page");
    assert_eq!(one.warnings.len(), two.warnings.len());
}

/// Parse errors and solve errors come back per program; a program that
/// fails leaves the preparation good for the next.
#[test]
fn prepared_reports_each_program_alone() {
    let lib = std_library();
    let prepared = solve::Prepared::new(&lib).unwrap();
    let bad_parse = prepared.solve([("p", "let x = ;")]);
    assert!(!bad_parse.errors.is_empty());
    let bad_solve = prepared.solve([("p", "let x: int = \"s\";")]);
    assert!(!bad_solve.errors.is_empty());
    let good = prepared.solve([("p", "let x: int = 1;")]);
    assert!(good.errors.is_empty(), "{:?}", good.errors);
}

/// `load_files` memoizes the preparation per library: the same library
/// hands back the same preparation, a different one its own.
#[test]
fn prepared_is_memoized_per_library() {
    let lib = std_library();
    let p1 = solve::prepared(&lib).unwrap();
    let p2 = solve::prepared(&lib).unwrap();
    assert!(std::rc::Rc::ptr_eq(&p1, &p2));
    let mut other = std_library();
    other.add_source("extra", "fn extra_fn() -> int { 7 }");
    let p3 = solve::prepared(&other).unwrap();
    assert!(!std::rc::Rc::ptr_eq(&p1, &p3));
    assert_ne!(p1.fingerprint(), p3.fingerprint());
    // and the extra source is really there for programs on the other
    let got = p3.solve([("p", "let n = extra_fn();")]);
    assert!(got.errors.is_empty(), "{:?}", got.errors);
    let not = p1.solve([("p", "let n = extra_fn();")]);
    assert!(!not.errors.is_empty());
}
