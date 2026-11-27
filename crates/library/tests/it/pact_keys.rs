//! Impl members key by (pact, name) (C3): two pacts may give one type a
//! method of one name, an inherent method and a pact method may share a
//! name, and a user pact may reuse a name std gives a built-in.

use crate::test_runner::try_execute;

const P15: &str = "pact A { fn label(self) -> str; }
pact B { fn label(self) -> str; }
struct P {}
impl A for P { fn label(self) -> str { \"a\" } }
impl B for P { fn label(self) -> str { \"b\" } }
fn via_a<T: A>(t: T) -> str { t.label() }
fn via_b(t: B) -> str { t.label() }";

test_run!(
    // r[verify mimas.pact.member-key]
    two_pacts_give_one_type_one_name,
    P15,
    "A::label(P {})" => "\"a\"",
    "B::label(P {})" => "\"b\"",
    "via_a(P {})" => "\"a\"",
    "via_b(P {})" => "\"b\"",
);

#[test]
// r[verify mimas.pact.member-key]
fn a_plain_call_of_a_name_two_pacts_give_says_which() {
    let err = format!(
        "{:?}",
        try_execute(&format!("{P15}\nlet TEST_VALUE = P {{}}.label();")).unwrap_err()
    );
    assert!(err.contains("more than one pact") && err.contains("`A::label(x)`"), "{err}");
}

test_run!(
    // r[verify mimas.pact.member-key]
    // p16: the inherent method answers a plain call, the pact's a pact call
    an_inherent_and_a_pact_method_share_a_name,
    "pact A { fn label(self) -> str; }
struct P {}
impl P { fn label(self) -> str { \"inherent\" } }
impl A for P { fn label(self) -> str { \"pact\" } }
fn via<T: A>(t: T) -> str { t.label() }",
    "P {}.label()" => "\"inherent\"",
    "via(P {})" => "\"pact\"",
    "A::label(P {})" => "\"pact\"",
);

test_run!(
    // r[verify mimas.pact.member-key]
    // the pact block first, then the inherent one: the same answers
    a_later_inherent_method_takes_the_plain_name,
    "pact A { fn label(self) -> str; }
struct P {}
impl A for P { fn label(self) -> str { \"pact\" } }
impl P { fn label(self) -> str { \"inherent\" } }
fn via<T: A>(t: T) -> str { t.label() }",
    "P {}.label()" => "\"inherent\"",
    "via(P {})" => "\"pact\"",
);

test_run!(
    // r[verify mimas.pact.member-key]
    // p17: a user pact reuses `len`; the native still answers a plain call
    a_user_pact_reuses_a_native_name,
    "pact Sz { fn len(self) -> int; }
impl Sz for str { fn len(self) -> int { 99 } }
fn size<T: Sz>(t: T) -> int { t.len() }",
    "\"abc\".len()" => "3",
    "size(\"abc\")" => "99",
);

test_run!(
    // r[verify mimas.pact.member-key]
    // p12: a user pact reuses `zero`, which std's `AbelianGroup` gives `int`
    a_user_pact_reuses_a_std_pact_name,
    "pact Nil { fn zero() -> Self; fn tag(self) -> str; }
impl Nil for int { fn zero() -> int { 0 } fn tag(self) -> str { \"nil\" } }
fn t<T: Nil>(x: T) -> str { x.tag() }",
    "t(3)" => "\"nil\"",
);
