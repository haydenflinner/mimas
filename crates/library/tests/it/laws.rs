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
    // a field the derived generator can't draw (a function) keeps the
    // no-samples error — it names `samples()` and `Arbitrary`
    let err = Vm::compile(
        &format!(
            "{TURNABLE}
            struct Lone {{ f: (int) -> int }}
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

// r[verify mimas.pact.samples] r[verify mimas.pact.samples.arb]
#[test]
fn a_structs_fields_generate_its_samples() {
    // `Cell` has no `samples()`, but its `int` fields are generatable —
    // the law checks run off derived draws instead of failing to compile
    let src = format!(
        "{TURNABLE}
        struct Cell2 {{ x: int, y: int }}
        impl Turnable for Cell2 {{
            fn turned(self, m: Move) -> Cell2 {{
                if m == ↻ {{ Cell2 {{ x = -self.y, y = self.x }} }} else {{ Cell2 {{ x = -self.x, y = self.y }} }}
            }}
        }}"
    );
    let r = results(&src);
    let laws: Vec<_> = r.iter().filter(|(n, _)| n.starts_with("law ")).collect();
    assert_eq!(laws.len(), 2, "{r:?}");
    assert!(laws.iter().all(|(_, e)| e.is_none()), "{r:?}");
}

// r[verify mimas.pact.samples] r[verify mimas.pact.samples.arb]
#[test]
fn an_enums_payloads_generate_too() {
    let src = format!(
        "{TURNABLE}
        enum D {{ N, V(int) }}
        impl Turnable for D {{
            fn turned(self, m: Move) -> D {{ self }}
        }}"
    );
    let r = results(&src);
    assert_eq!(r.len(), 2, "{r:?}");
    assert!(r.iter().all(|(_, e)| e.is_none()), "{r:?}");
}

// r[verify mimas.pact.samples] r[verify mimas.pact.samples.arb]
#[test]
fn a_failing_law_names_the_minimal_witness() {
    // `v > 3` breaks the law; the int field's shrink walks any witness
    // down to `v = 4`, the smallest value that still fails
    let src = "pact Small {
        fn v(self) -> int;
        check |x| x.v() <= 3
    }
    struct S { v: int }
    impl Small for S { fn v(self) -> int { self.v } }";
    let r = results(src);
    assert_eq!(r.len(), 1, "{r:?}");
    let err = r[0].1.as_ref().expect("the law fails for v > 3");
    assert!(err.contains("fails for x = S { v: 4 }"), "{err}");
}

// r[verify mimas.pact.samples] r[verify mimas.pact.samples.arb]
#[test]
fn generated_samples_replay_on_the_same_seed() {
    let src = "pact Small {
        fn v(self) -> int;
        check |x| x.v() <= 3
    }
    struct S { v: int }
    impl Small for S { fn v(self) -> int { self.v } }";
    assert_eq!(results(src), results(src));
}

// r[verify mimas.pact.samples] r[verify mimas.pact.samples.arb]
#[test]
fn an_arbitrary_impl_drives_draws_and_shrink() {
    // a declared `impl Arbitrary` overrides the derivation: gen always
    // yields v = 9 (law fails), shrink hands back v = 4
    let src = "pact Small {
        fn v(self) -> int;
        check |x| x.v() <= 3
    }
    struct N { v: int }
    impl Arbitrary for N {
        fn gen(size: int) -> N { N { v = 9 } }
        fn shrink(self) -> [N] { [N { v = 4 }] }
    }
    impl Small for N { fn v(self) -> int { self.v } }";
    let r = results(src);
    let err = r[0].1.as_ref().expect("the law fails for v = 9");
    assert!(err.contains("fails for x = N { v: 4 }"), "{err}");
}

// r[verify mimas.pact.law.impl]
#[test]
fn a_broken_law_on_a_primitive_shrinks_to_zero() {
    // even over the fixed primitive samples, a failing law reports the
    // minimal pair — subtraction here is never commutative
    let r = results(
        "pact Merge {
            fn merge(self, o: Self) -> Self;
            check |a, b| a.merge(b) == b.merge(a)
        }
        impl Merge for int { fn merge(self, o: int) -> int { self - o } }",
    );
    let err = r[0].1.as_ref().expect("the law fails");
    assert!(err.contains("a = 0, b = 1"), "{err}");
}

// r[verify mimas.pact.samples] r[verify mimas.pact.samples.arb]
#[test]
fn a_recursive_type_generates_through_its_option() {
    // `next: Node?` is a carried self-edge — bounded by `size`, it
    // terminates instead of recursing forever
    let src = "pact Id { fn id(self) -> Self; check |x| x.id() == x }
    struct Node { next: Node? }
    impl Id for Node { fn id(self) -> Node { self } }";
    let r = results(src);
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(r[0].1.is_none(), "{r:?}");
}

