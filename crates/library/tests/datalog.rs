//! `datalog { … }` blocks: relations, facts and rules evaluated to a fixpoint, lowered by the
//! parser to plain Mimas (see `parse::parser::datalog`).
#[macro_use]
mod test_runner;

const CLOSURE: &str = r#"
fn closure(pairs: [(int, int)]) -> [(int, int)] {
    let (_edge_, path_) = datalog {
        rel edge(a: int, b: int) from pairs
        rel path(a: int, b: int)
        path(a, b) if edge(a, b)
        path(a, c) if edge(a, b), path(b, c)
    };
    path_
}
// both atoms recursive: the delta has to reach either side
fn closure2(pairs: [(int, int)]) -> [(int, int)] {
    let (_edge_, path_) = datalog {
        rel edge(a: int, b: int) from pairs
        rel path(a: int, b: int)
        path(a, b) if edge(a, b)
        path(a, c) if path(a, b), path(b, c)
    };
    path_
}
fn chain(n: int) -> [(int, int)] { for i in 0 .. n - 1 collect (i, i + 1) }
"#;

test_run!(
    // r[verify mimas.datalog.block]
    datalog_transitive_closure,
    CLOSURE,
    "closure(chain(4)).len()" => "6",
    "closure(chain(8)).len()" => "28",
    "closure2(chain(8)).len()" => "28",
    "closure2(chain(17)).len()" => "136",
    "closure(chain(4)).contains((0, 3))" => "true",
    "closure(chain(4)).contains((3, 0))" => "false",
    // a cycle reaches itself; the fixpoint still stops
    "closure([(1, 2), (2, 1)]).len()" => "4",
    "closure([(1, 2), (2, 1)]).contains((1, 1))" => "true",
    "closure([]).len()" => "0",
);

test_run!(
    // r[verify mimas.datalog.block]
    datalog_seed_and_dedup,
    "let pairs = [(1, 2), (2, 3), (1, 2)];",
    // a single relation comes back as its array, not a tuple of one; the seed's duplicate
    // collapses
    "datalog { rel edge(a: int, b: int) from pairs }.len()" => "2",
    // two rules deriving the same tuple store it once
    r#"datalog {
        rel edge(a: int, b: int) from pairs
        rel out(a: int)
        out(a) if edge(a, _)
        out(a) if edge(a, b), b > 0
        out(1)
    }.1.len()"# => "2",
);

test_run!(
    // r[verify mimas.datalog.block]
    datalog_single_field_relations,
    "let pairs = [(1, 2), (2, 3), (1, 5)];",
    // a one-field relation is an array of bare values
    r#"datalog {
        rel edge(a: int, b: int) from pairs
        rel src(a: int)
        src(a) if edge(a, _)
    }.1"# => "[1, 2]",
    r#"datalog {
        rel node(n: int) from [3, 4, 3]
        rel twice(n: int)
        twice(n * 2) if node(n)
    }.1.len()"# => "2",
);

test_run!(
    // r[verify mimas.datalog.body]
    datalog_filters,
    "let pairs = [(1, 2), (2, 2), (1, 3), (3, 1)]; let x = 3;",
    // a repeated variable filters by equality
    r#"datalog { rel e(a: int, b: int) from pairs
        rel loop_(a: int)
        loop_(a) if e(a, a) }.1"# => "[2]",
    // a literal filters
    r#"datalog { rel e(a: int, b: int) from pairs
        rel from1(b: int)
        from1(b) if e(1, b) }.1.len()"# => "2",
    // a parenthesized name is a value from outside the block, not a rule variable
    r#"datalog { rel e(a: int, b: int) from pairs
        rel fromx(b: int)
        fromx(b) if e((x), b) }.1"# => "[1]",
    // a variable bound by an earlier atom filters a later one (a join)
    r#"datalog { rel e(a: int, b: int) from pairs
        rel two(a: int, c: int)
        two(a, c) if e(a, b), e(b, c), a != c }.1 == [(1, 2), (3, 2)]"# => "true",
    // conditions see the atoms to their left
    r#"datalog { rel e(a: int, b: int) from pairs
        rel up(a: int, b: int)
        up(a, b) if e(a, b), a < b }.1.len()"# => "2",
);

test_run!(
    // r[verify mimas.datalog.body]
    datalog_facts_see_outer_scope,
    "let bl = 7; let k = 2;",
    r#"datalog {
        rel board(bl: int)
        rel pair(a: int, b: int)
        board(bl)
        board(bl + 1)
        pair(b, (k)) if board(b)
    }.1 == [(7, 2), (8, 2)]"# => "true",
);

// learned.mimas, with its table stubbed
const LEARNED: &str = r#"
fn attr_rows() -> [(str, bool, bool, str, int)]! {
    [("f0", false, false, "clean", 0), ("f2", true, false, "dead-write", 1),
     ("f3", false, true, "x", -1), ("f4", true, false, "dead-write", -1),
     ("f8", false, false, "clean", 3)]
}
let (attr_, hint_) = datalog {
    rel attr(f: str, strict: bool, onr: bool, nw: str, ret: int) from attr_rows()!
    rel hint(f: str, kind: str)
    hint(f, "clean") if attr(f, false, false, "clean", _)
    hint(f, "dead-store passthrough") if attr(f, true, false, "dead-write", r), r >= 0
    hint(f, "opaque writer") if attr(f, _, true, _, _)
};
"#;

test_run!(
    datalog_learned_lesson,
    LEARNED,
    "attr_.len()" => "5",
    "hint_.len()" => "4",
    r#"hint_.contains(("f0", "clean")) && hint_.contains(("f8", "clean"))"# => "true",
    r#"hint_.contains(("f2", "dead-store passthrough"))"# => "true",
    r#"hint_.contains(("f4", "dead-store passthrough"))"# => "false",
    r#"hint_.contains(("f3", "opaque writer"))"# => "true",
);

// carp-kit.mimas's cut-pattern enumeration: unit-typed fields, a fn and an outer local in the
// rules, a rule continued onto a second line
const CARP_KIT: &str = r#"
const KERF = 0.125inch;
fn kerf() -> inch { KERF }
fn patterns(pieces: [inch], bl: inch) -> [(int, inch, inch)] {
    let rows = for i in 0 .. pieces.len() collect (i, pieces[i]);
    let (_piece_, _board_, pattern_) = datalog {
        rel piece(id: int, len: inch) from rows
        rel board(bl: inch)
        rel pattern(mask: int, used: inch, b: inch)
        board(bl)
        pattern(1 << i, l + kerf(), b) if piece(i, l), board(b), l + kerf() <= b + kerf()
        pattern(m | (1 << i), t + l + kerf(), b)
            if pattern(m, t, b), piece(i, l), (m & (1 << i)) == 0, t + l + kerf() <= b + kerf()
    };
    pattern_
}
"#;

test_run!(
    datalog_carp_kit_lesson,
    CARP_KIT,
    // every non-empty subset of three 5' pieces fits a 16' board
    "patterns([60.0inch, 60.0inch, 60.0inch], 192.0inch).len()" => "7",
    // two 8' pieces don't share a 16' board (the kerf between them)
    "patterns([96.0inch, 96.0inch], 192.0inch).len()" => "2",
    "patterns([96.0inch, 96.0inch], 192.0inch)[0].0" => "1",
    // 12 small pieces: every one of the 4095 subsets fits
    "patterns([6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch, 6.0inch], 192.0inch).len()" => "4095",
);

// the lowering's own bindings (`__dl_*`) never trip the lint suite -- the shipped pages are
// linted warning-free
#[test]
fn datalog_lowering_is_lint_clean() {
    for src in [CLOSURE, LEARNED, CARP_KIT] {
        let vm = vm::Vm::execute(src, library::std).expect("compiles");
        let warnings: Vec<String> = vm.warnings().iter().map(|w| format!("{w:?}")).collect();
        assert!(warnings.is_empty(), "warnings:\n{}", warnings.join("\n"));
    }
}

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(
        text.contains(needle),
        "`{src}`: expected `{needle}` in:\n{text}"
    );
}

// r[verify mimas.datalog.strata]
#[test]
fn datalog_errors() {
    let block = |rules: &str| {
        format!(
            "let r = datalog {{\n rel edge(a: int, b: int) from [(1, 2)]\n rel path(a: int, b: int)\n {rules}\n}};"
        )
    };
    // unknown relations
    rejected(&block("paht(a, b) if edge(a, b)"), "`paht` isn't a relation");
    rejected(&block("path(a, b) if edgee(a, b)"), "`edgee` isn't a relation");
    rejected(&block("path(a, b) if edgee(a, _)"), "`edgee` isn't a relation");
    // arity
    rejected(&block("path(a, b) if edge(a)"), "relation `edge` has 2 fields, but this names 1");
    rejected(&block("path(a) if edge(a, _)"), "relation `path` has 2 fields, but this names 1");
    rejected(&block("edge(1, 2, 3)"), "relation `edge` has 2 fields, but this names 3");
    // the head needs the body's variables
    rejected(&block("path(a, c) if edge(a, b)"), "head variable `c` isn't bound by the rule's body");
    rejected(&block("path(a, _) if edge(a, b)"), "`_` can't stand in a head");
    // declarations
    rejected(
        "let r = datalog { rel e(a: int) rel e(b: int) };",
        "relation `e` is already declared",
    );
    rejected("let r = datalog { rel e() };", "relation `e` has no fields");
    // a seed of the wrong shape is a type error at the seed
    rejected("let r = datalog { rel e(a: int, b: int) from [1, 2] };", "mismatched types");
    // stratification: a relation can't depend on its own negation or aggregate
    rejected(
        &block("path(a, b) if edge(a, b), !path(b, a)"),
        "depends on a negation or aggregate of `path`",
    );
    rejected(
        &block("path(a, n) if edge(a, _), agg n = count() in path(a, _)"),
        "depends on a negation or aggregate of `path`",
    );
    // a negated atom can't bind
    rejected(&block("path(a, b) if edge(a, b), !edge(a, c)"), "`c` in a negated atom isn't bound");
    // aggregate shapes
    rejected(&block("path(a, n) if edge(a, _), agg n = count(a) in edge(a, _)"), "`count` isn't an aggregate of this shape");
    rejected(&block("path(a, n) if edge(a, _), agg n = median(b) in edge(a, b)"), "`median` isn't an aggregate of this shape");
}

// Ascent's language, run in mimas: `relation` with positional types, `<--`, `if`/`let`/`for`
// in a body, stratified negation (`!rel(..)`), and aggregates (`agg x = f(e) in rel(..)`).

test_run!(
    // r[verify mimas.datalog.block]
    // r[verify mimas.datalog.body]
    datalog_ascent_spelling,
    "let pairs = [(1, 2), (2, 3), (3, 4)];",
    // the same closure, spelled the way Ascent spells it
    r#"datalog {
        relation edge(int, int) from pairs
        relation path(int, int)
        path(x, y) <-- edge(x, y)
        path(x, z) <-- edge(x, y), path(y, z)
    }.1.len()"# => "6",
    // `if`, `let` and `for` in a body
    r#"datalog {
        relation edge(int, int) from pairs
        relation sums(int, int)
        sums(x, s) <-- edge(x, y), if x > 1, let s = x + y
    }.1 == [(2, 5), (3, 7)]"# => "true",
    r#"datalog {
        relation seed(int) from [1, 2]
        relation num(int)
        num(n) <-- seed(m), for n in [m, m * 10]
    }.1"# => "[1, 10, 2, 20]",
);

test_run!(
    // r[verify mimas.datalog.body]
    // r[verify mimas.datalog.strata]
    datalog_stratified_negation,
    "let pairs = [(1, 2), (2, 3), (5, 6)];
     let nodes = [1, 2, 3, 4, 5, 6];",
    // reach is recursive; unreached negates it once it's complete
    r#"datalog {
        rel edge(a: int, b: int) from pairs
        rel node(n: int) from nodes
        rel reach(n: int)
        rel unreached(n: int)
        reach(1)
        reach(b) if reach(a), edge(a, b)
        unreached(n) if node(n), !reach(n)
    }.3"# => "[4, 5, 6]",
    // `not` spells it too, and a negation can sit in a stratum above another negation
    r#"datalog {
        rel node(n: int) from nodes
        rel even(n: int)
        rel odd(n: int)
        rel neither(n: int)
        even(n) if node(n), n mod 2 == 0
        odd(n) if node(n), not even(n)
        neither(n) if node(n), !even(n), !odd(n)
    }.2"# => "[1, 3, 5]",
);

test_run!(
    // r[verify mimas.datalog.body]
    // r[verify mimas.datalog.strata]
    datalog_aggregates,
    "let w = [(1, 5), (1, 7), (2, 3)];
     let ids = [1, 2, 3];",
    r#"datalog {
        rel weight(id: int, w: int) from w
        rel id(n: int) from ids
        rel stats(id: int, n: int)
        stats(i, n) if id(i), agg n = count() in weight(i, _)
    }.2 == [(1, 2), (2, 1), (3, 0)]"# => "true",
    // an aggregate over no tuples derives nothing (there's no min of nothing)
    r#"datalog {
        rel weight(id: int, w: int) from w
        rel id(n: int) from ids
        rel lo(id: int, w: int)
        lo(i, m) if id(i), agg m = min(x) in weight(i, x)
    }.2 == [(1, 5), (2, 3)]"# => "true",
    r#"datalog {
        rel weight(id: int, w: int) from w
        rel id(n: int) from ids
        rel hi(id: int, w: int)
        hi(i, m) if id(i), agg m = max(x) in weight(i, x)
    }.2 == [(1, 7), (2, 3)]"# => "true",
    // a rule with no positive atom: one total over the whole relation
    r#"datalog {
        rel weight(id: int, w: int) from w
        rel total(s: int)
        total(s) if agg s = sum(x) in weight(_, x)
    }.1"# => "[15]",
    r#"datalog {
        rel weight(id: int, w: int) from w
        rel avg(m: float)
        avg(m) if agg m = mean(x) in weight(_, x)
    }.1"# => "[5]",
);
