//! Homogeneous typed arrays: `Val::IntArray`/`Val::FloatArray` back `[1, 2, 3]`
//! literals, `for .. collect` results, and `[]`-born sequences with raw
//! `Vec<i64>`/`Vec<f64>` storage instead of `Vec<Val>`.
//!
//! These tests pin both halves of the contract: the representation is
//! *invisible* (every semantic check below answers the same on a plain
//! `Vec<Val>` array), and promotion/demotion really happens — `seq_kind`
//! peeks at the tag/store from the native side.
//!
//! Note on scope: the bare `Vm::execute` these tests use installs no std
//! library, so `push`/`len`/etc. method calls don't exist here — `Op::Push`
//! is exercised through `for .. collect` (which emits `NewArray`+`Push`),
//! and demotion through natives that take the untyped `Array` handle.
//! The typechecker also refuses heterogeneous literals and `arr * scalar`
//! infix, so the broadcast arms are reached through tuples.
#[macro_use]
mod vm_test_utils;

use vm::Captured::*;
use vm::conversion::MimasType;
use vm::{ArrayStore, Ctx, Val};

/// Representation probe: reports the store kind a sequence arrived under
/// (`typed:` covers both `IntArray`/`FloatArray` tags — the tag is only a
/// birth hint; the `ArrayStore` variant is the truth).
fn seq_kind<'gc>(_ctx: Ctx<'gc>, v: Val<'gc>) -> String {
    match v {
        Val::Array(_) => "array".into(),
        Val::IntArray(a) | Val::FloatArray(a) => match &*a.0.borrow() {
            ArrayStore::Empty => "typed:empty".into(),
            ArrayStore::Ints(_) => "typed:ints".into(),
            ArrayStore::Floats(_) => "typed:floats".into(),
            ArrayStore::Vals(_) => "typed:vals".into(),
        },
        _ => "other".into(),
    }
}

/// Cross-representation equality probe — `Val::eq` directly, so a typed
/// array can be compared against a `Val::Array` handle without the
/// typechecker forcing the shapes to agree.
fn seq_eq<'gc>(_ctx: Ctx<'gc>, a: Val<'gc>, b: Val<'gc>) -> bool {
    a == b
}

/// A plain `Vec<Val>`-backed `Val::Array` holding ints — the pre-typed
/// representation, for cross-representation comparisons.
fn plain_ints<'gc>(ctx: Ctx<'gc>, a: i64, b: i64) -> Val<'gc> {
    Val::Array(ctx.new_array(vec![Val::Int(a), Val::Int(b)]))
}

/// Pushes `v` onto the untyped `Array` handle — `as_untyped_array` demotes a
/// typed input and hands back a handle onto the *same* contents, so this
/// writes through to the `IntArray`/`FloatArray` the script still holds.
fn push_val<'gc>(ctx: Ctx<'gc>, a: vm::Array<'gc>, v: Val<'gc>) {
    a.0.borrow_mut(&ctx).push(v);
}

/// Same write-through, at an index.
fn set_val<'gc>(ctx: Ctx<'gc>, a: vm::Array<'gc>, i: i64, v: Val<'gc>) {
    a.0.borrow_mut(&ctx)[i as usize] = v;
}

/// `Vec<i64>` in/out: `from_value` demotes a typed input to read it;
/// `into_value` routes the result through `array_val` — homogeneous `Vec`
/// comes back typed.
fn sum_ints(_ctx: Ctx<'_>, v: Vec<i64>) -> i64 {
    v.iter().sum()
}

fn make_ints(ctx: Ctx<'_>) -> Val<'_> {
    vec![1i64, 2, 3].into_value(ctx)
}

fn install(api: &mut vm::api::Api<'_, '_>) {
    api.add(seq_kind);
    api.add(seq_eq);
    api.add(plain_ints);
    api.add(push_val);
    api.add(set_val);
    api.add(sum_ints);
    api.add(make_ints);
}

#[track_caller]
fn run(source: &str) -> vm::Captured {
    let mut vm = vm::Vm::execute(source, install).expect("test source compiled and ran");
    vm.resolve_name("TEST_VALUE")
        .expect("TEST_VALUE was bound by the test source")
}

// ---- representation: promotion happens ---------------------------------

#[test]
fn int_literal_is_typed() {
    assert_eq!(
        run("let TEST_VALUE = seq_kind([1, 2, 3]);"),
        Str("typed:ints".into())
    );
}

#[test]
fn float_literal_is_typed() {
    // literals build with `new_seq` + `Push`; the first pushed float picks
    // the `Floats` store (the tag stays `IntArray` — birth hint only)
    assert_eq!(
        run("let TEST_VALUE = seq_kind([1.5, 2.5]);"),
        Str("typed:floats".into())
    );
}

#[test]
fn collect_result_is_typed() {
    assert_eq!(
        run("let TEST_VALUE = seq_kind(for x in [1, 2, 3] collect x * 2);"),
        Str("typed:ints".into())
    );
}

#[test]
fn empty_stays_pending_typed() {
    assert_eq!(
        run("let TEST_VALUE = seq_kind([]);"),
        Str("typed:empty".into())
    );
}

#[test]
fn native_vec_result_is_typed() {
    assert_eq!(
        run("let TEST_VALUE = seq_kind(make_ints());"),
        Str("typed:ints".into())
    );
}

#[test]
fn demotion_via_native_push() {
    assert_eq!(
        run(
            r#"let a = [1, 2];
push_val(a, "x");
let TEST_VALUE = seq_kind(a);"#
        ),
        // same `Val::IntArray` handle, now fronting a `Vals` store
        Str("typed:vals".into())
    );
}

#[test]
fn demoted_contents_survive() {
    assert_eq!(
        run(
            r#"let a = [1, 2];
push_val(a, "x");
let TEST_VALUE = a;"#
        ),
        array!(Int(1), Int(2), Str("x".into()))
    );
}

#[test]
fn same_kind_native_push_keeps_backing() {
    assert_eq!(
        run(
            r#"let a = [1, 2];
push_val(a, 7);
let TEST_VALUE = seq_kind(a);"#
        ),
        // wait — `push_val` takes `vm::Array`, which demotes to `Vals` first
        Str("typed:vals".into())
    );
}

// ---- semantics: invisible to the language ------------------------------

test_vm!(
    typed_index,
    "let a = [10, 20, 30];",
    "a[0]" => Int(10),
    "a[2]" => Int(30),
);

test_vm!(
    typed_set_index_same_kind,
    "let a = [10, 20, 30]; a[1] = 99;",
    "a" => array!(Int(10), Int(99), Int(30)),
);

test_vm!(
    typed_float_reads_box,
    "let a = [1.5, 2.5];",
    "a[0]" => Float(1.5),
    "a[0] + a[1]" => Float(4.0),
);

test_vm!(
    typed_for_in_accumulate,
    "let a = [1, 2, 3]; let total = 0; for x in a { total += x; }",
    "total" => Int(6)
);

test_vm!(
    typed_collect_maps,
    "let a = [1, 2, 3];",
    "for x in a collect x * 2" => array!(Int(2), Int(4), Int(6)),
);

test_vm!(
    typed_contains,
    "let a = [1, 2, 3];",
    "2 in a" => Bool(true),
    "9 in a" => Bool(false),
);

test_vm!(
    typed_equality,
    "let a = [1, 2]; let b = [1, 2];",
    "a == b" => Bool(true),
    "a != b" => Bool(false),
    "a == [1, 3]" => Bool(false),
);

test_vm!(
    typed_display,
    "let a = [1, 2, 3]; let b = [1.5]; let c = [];",
    "f\"{a}\"" => str!("[1, 2, 3]"),
    "f\"{b}\"" => str!("[1.5]"),
    "f\"{c}\"" => str!("[]"),
);

test_vm!(
    typed_nested,
    "let a = [[1, 2], [3, 4]];",
    "a[1][0]" => Int(3),
    "a" => array!(array!(Int(1), Int(2)), array!(Int(3), Int(4))),
);

#[test]
fn tuple_literal_is_typed_and_broadcasts() {
    // tuple constants go through `array_val`, so an all-int tuple is
    // `Ints`-backed — and tuple broadcasting then exercises the typed
    // elementwise path in `bin`/`unary`
    assert_eq!(
        run("let TEST_VALUE = seq_kind((1, 2, 3));"),
        Str("typed:ints".into())
    );
    assert_eq!(run("let TEST_VALUE = (1, 2, 3) * 2;"), array!(Int(2), Int(4), Int(6)));
    assert_eq!(
        run("let TEST_VALUE = (1, 2, 3) + (10, 20, 30);"),
        array!(Int(11), Int(22), Int(33))
    );
    assert_eq!(run("let TEST_VALUE = 10 - (1, 2, 3);"), array!(Int(9), Int(8), Int(7)));
    assert_eq!(
        run("let TEST_VALUE = -(1, 2, 3);"),
        array!(Int(-1), Int(-2), Int(-3))
    );
}

// ---- cross-representation equality -------------------------------------

#[test]
fn typed_eq_plain_array() {
    assert_eq!(
        run("let TEST_VALUE = seq_eq([1, 2], plain_ints(1, 2));"),
        Bool(true)
    );
    assert_eq!(
        run("let TEST_VALUE = seq_eq([1, 2], plain_ints(1, 3));"),
        Bool(false)
    );
}

#[test]
fn demoted_eq_plain_array() {
    assert_eq!(
        run(
            r#"let a = [1, 2];
push_val(a, "x");
let TEST_VALUE = seq_eq(a, plain_ints(1, 2));"#
        ),
        Bool(false) // [1, 2, "x"] != [1, 2]
    );
}

// ---- native conversion --------------------------------------------------

#[test]
fn native_vec_param_reads_typed() {
    assert_eq!(run("let TEST_VALUE = sum_ints([4, 5, 6]);"), Int(15));
}

// ---- snapshot roundtrip --------------------------------------------------

#[test]
fn typed_survives_snapshot_restore() {
    let source = r#"
let ints = [1, 2, 3];
let floats = [4.5, 5.5];
let mixed = [1, 2];
push_val(mixed, "x");
let TEST_VALUE = 0;
"#;
    let mut vm = vm::Vm::execute(source, install).expect("compiled and ran");
    let snap = vm.snapshot().expect("snapshot");
    vm.restore(&snap).expect("restore");
    assert_eq!(
        vm.resolve_name("ints"),
        Some(array!(Int(1), Int(2), Int(3)))
    );
    assert_eq!(
        vm.resolve_name("floats"),
        Some(array!(Float(4.5), Float(5.5)))
    );
    assert_eq!(
        vm.resolve_name("mixed"),
        Some(array!(Int(1), Int(2), Str("x".into())))
    );
}

#[test]
fn primitive_stores_stay_primitive_in_snapshots() {
    // the snapshot node table itself carries raw i64/f64 vectors — no
    // boxing through `SnapVal::Node` per element
    let source = r#"
let ints = [1, 2, 3];
let floats = [4.5, 5.5];
let mixed = [1, 2];
push_val(mixed, "x");
let TEST_VALUE = 0;
"#;
    let mut vm = vm::Vm::execute(source, install).expect("compiled and ran");
    let snap = vm.snapshot().expect("snapshot");
    let has = |f: fn(&vm::SnapNode) -> bool| snap.nodes.iter().any(f);
    assert!(has(|n| matches!(n, vm::SnapNode::IntArray(v) if v == &[1, 2, 3])));
    assert!(has(|n| matches!(n, vm::SnapNode::FloatArray(v) if v == &[4.5, 5.5])));
    // `mixed` demoted to `Vals` before the snapshot, so its node is the
    // inner array — a `SnapNode::Array`, not a primitive store
    assert!(has(|n| matches!(n, vm::SnapNode::Array(_))));
}
