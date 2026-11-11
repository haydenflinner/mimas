//! Laws as checks (`mimas.pact.law`). Each `check |x| …` line of a pact gives
//! every impl one check, which runs the law over the samples of the type.
use vm::Vm;

fn results(src: &str) -> Vec<(String, Option<String>)> {
    let mut vm = Vm::compile(src, |api| library::std(api)).unwrap_or_else(|e| panic!("{e:?}"));
    vm.run().unwrap();
    vm.run_tests().into_iter().map(|t| (t.name, t.error)).collect()
}

const TURNABLE: &str = "
struct Move { a: float, b: float, c: float, d: float }
const ↻ = Move { a: 0.0, b: -1.0, c: 1.0, d: 0.0 };
const ⇄ = Move { a: -1.0, b: 0.0, c: 0.0, d: 1.0 };
pact Turnable {
    fn turned(self, m: Move) -> Self;
    check |x| x.turned(↻).turned(↻).turned(↻).turned(↻) == x
    check |x| x.turned(⇄).turned(⇄) == x
}";

// r[verify mimas.pact.law.impl] r[verify mimas.pact.samples]
#[test]
fn every_impl_checks_every_law() {
    let src = format!(
        "{TURNABLE}
        // an enum's samples are its variants
        enum Facing {{ N, E, S, W }}
        impl Turnable for Facing {{
            fn turned(self, m: Move) -> Facing {{
                if m == ↻ {{ match self {{ Facing::N => Facing::E, Facing::E => Facing::S, Facing::S => Facing::W, Facing::W => Facing::N }} }}
                else {{ match self {{ Facing::E => Facing::W, Facing::W => Facing::E, _ => self }} }}
            }}
        }}
        // a struct lists its own
        struct Cell {{ x: int, y: int }}
        impl Cell {{ fn samples() -> [Cell] {{ [Cell {{ x: 1, y: 0 }}, Cell {{ x: 2, y: 3 }}] }} }}
        impl Turnable for Cell {{
            fn turned(self, m: Move) -> Cell {{
                if m == ↻ {{ Cell {{ x: -self.y, y: self.x }} }} else {{ Cell {{ x: -self.x, y: self.y }} }}
            }}
        }}"
    );
    let r = results(&src);
    let laws: Vec<_> = r.iter().filter(|(n, _)| n.starts_with("law ")).collect();
    assert_eq!(laws.len(), 4, "{r:?}");
    assert!(laws.iter().all(|(_, e)| e.is_none()), "{r:?}");
    assert!(laws.iter().any(|(n, _)| n == "law Turnable for Cell: |x| x.turned(⇄).turned(⇄) == x"), "{r:?}");
}

// r[verify mimas.pact.law.impl]
#[test]
fn a_broken_impl_fails_naming_the_sample() {
    let src = format!(
        "{TURNABLE}
        struct Bad {{ v: int }}
        impl Bad {{ fn samples() -> [Bad] {{ [Bad {{ v: 1 }}] }} }}
        impl Turnable for Bad {{
            fn turned(self, m: Move) -> Bad {{ if m == ↻ {{ Bad {{ v: -self.v }} }} else {{ Bad {{ v: self.v + 1 }} }} }}
        }}"
    );
    let r = results(&src);
    let (_, rot) = r.iter().find(|(n, _)| n.contains("turned(↻)")).unwrap();
    assert!(rot.is_none(), "{r:?}");
    let (_, flip) = r.iter().find(|(n, _)| n.contains("turned(⇄)")).unwrap();
    let flip = flip.as_ref().expect("the flip law fails for Bad");
    assert!(flip.contains("fails for x = Bad"), "{flip}");
}

// r[verify mimas.pact.law.impl]
#[test]
fn a_law_over_two_values_takes_every_pair() {
    let r = results(
        "pact Merge {
            fn merge(self, o: Self) -> Self;
            check |a, b| a.merge(b) == b.merge(a)
        }
        impl Merge for int { fn merge(self, o: int) -> int { self + o } }",
    );
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(r[0].1.is_none(), "{r:?}");
    assert_eq!(r[0].0, "law Merge for int: |a, b| a.merge(b) == b.merge(a)");
}

// r[verify mimas.pact.samples]
#[test]
fn a_type_without_samples_is_told_what_to_add() {
    let err = Vm::compile(
        &format!(
            "{TURNABLE}
            struct Lone {{ v: int }}
            impl Turnable for Lone {{ fn turned(self, m: Move) -> Lone {{ self }} }}"
        ),
        |api| library::std(api),
    )
    .err()
    .expect("a type with no samples doesn't compile");
    let msg = format!("{err:?}");
    assert!(msg.contains("whose laws need sample values"), "{msg}");
}

// r[verify mimas.pact.law]
#[test]
fn a_law_is_a_closure() {
    let err = Vm::compile("pact P { fn f(self) -> int; check 1 == 1 }", |api| library::std(api))
        .err()
        .expect("a law that isn't a closure is an error");
    assert!(format!("{err:?}").contains("law over the pact's values"), "{err:?}");
}

// r[verify mimas.pact.law.impl]
#[test]
fn law_checks_keep_the_programs_own_units() {
    // the generated checks parse after the program; a page's `unit` decls
    // must survive that parse, or every annotation using them goes undefined
    let r = results(
        "unit elixir;
        fn cost(c: elixir) -> elixir { c }
        pact Twice {
            fn flip(self) -> Self;
            check |x| x.flip().flip() == x
        }
        enum Side { L, R }
        impl Twice for Side { fn flip(self) -> Side { match self { Side::L => Side::R, Side::R => Side::L } } }",
    );
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(r[0].1.is_none(), "{r:?}");
}
