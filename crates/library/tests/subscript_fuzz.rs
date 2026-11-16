//! Semantic fuzz of the subscript lowering. `panic_fuzz` over the host
//! corpus catches crashes; this catches wrong *answers*. The accessor
//! lowering rewrites the AST (synth method calls, `__Place` handles,
//! arm dispatch by operation), so a miscompile shows up only as a
//! wrong value, never as a panic.
//!
//! Each case generates a program: a `subscript` accessor with a random
//! arm subset (each arm records its operation in `log`), plus a random
//! op script over the accessor, a built-in list, a nested list, and
//! borrowed slots. A Rust shadow model applies the same ops and the
//! expected arm-dispatch order, and the rendered state is diffed.
//! Cases where the arm subset lacks an operation instead assert the
//! compile error names the missing arm.
//!
//! Deterministic: `MIMAS_FUZZ_CASES` (default 200) and
//! `MIMAS_FUZZ_SEED` replay any failure by case index.

use std::fmt::Write;
use vm::Vm;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        (splitmix64(&mut self.0) % n as u64) as usize
    }
    fn chance(&mut self, pct: usize) -> bool {
        self.below(100) < pct
    }
    fn pick<T>(&mut self, xs: &[T]) -> T
    where
        T: Copy,
    {
        xs[self.below(xs.len())]
    }
}

/// Which arms the generated `subscript` declares.
#[derive(Clone, Copy)]
struct Arms {
    get: bool,
    set: bool,
    inout: bool,
    remove: bool,
}

impl Arms {
    fn can_read(&self) -> bool {
        self.get || self.inout
    }
    fn can_write(&self) -> bool {
        self.set || self.inout
    }
    fn can_compound(&self) -> bool {
        self.inout || (self.get && self.set)
    }
}

/// Rust shadow of the generated program's observable state.
struct Model {
    data: Vec<i64>,
    gone: Vec<bool>,
    log: Vec<i64>,
    xs: Vec<i64>,
    old: Vec<i64>,
    /// `oc.data` once an `AliasRebindC` has detached it; `None` while
    /// `oc` still shares `c`'s list.
    oc_data: Option<Vec<i64>>,
    m: Vec<Vec<i64>>,
    noisy: Vec<i64>,
    ok: bool,
}

impl Model {
    fn render(&self) -> String {
        format!(
            "{}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            self.ok,
            self.data,
            self.gone,
            self.log,
            self.xs,
            self.old,
            self.oc_data.as_ref().unwrap_or(&self.data),
            self.m,
            self.noisy
        )
    }
    /// `get` dispatch: the explicit arm wins; `inout` synthesizes one
    /// that still runs its cleanup tail.
    fn log_get(&mut self, arms: Arms) {
        if arms.get {
            self.log.push(1);
        } else {
            self.log.extend([6, 7]);
        }
    }
    fn log_set(&mut self, arms: Arms) {
        if arms.set {
            self.log.push(2);
        } else {
            self.log.extend([6, 7]);
        }
    }
}

/// An index expression: usually a literal, sometimes `noisy(i)` so a
/// double evaluation would push twice.
fn index(rng: &mut Rng, src: &mut String, m: &mut Model, i: usize) {
    if rng.chance(30) {
        write!(src, "noisy({i})").unwrap();
        m.noisy.push(i as i64);
    } else {
        write!(src, "{i}").unwrap();
    }
}

/// Emits the key list for one flat `data` offset: `i` for one-key
/// accessors, `i, j` (row-major over `W`) for two-key ones.
fn c_index(
    rng: &mut Rng,
    src: &mut String,
    m: &mut Model,
    keys: usize,
    width: usize,
    flat: usize,
) {
    index(rng, src, m, flat / width);
    if keys == 2 {
        src.push_str(", ");
        index(rng, src, m, flat % width);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Read,
    Write,
    WriteRead,
    Compound,
    Remove,
    Borrow,
    BorrowRemove,
    XsRead,
    XsWrite,
    XsCompound,
    XsRemove,
    XsBorrow,
    MRead,
    MWrite,
    MCompound,
    MRemove,
    AliasRebind,
    AliasRebindC,
}

fn gen_case(rng: &mut Rng) -> (String, Option<String>, String) {
    let arms = loop {
        let a = Arms {
            get: rng.chance(80),
            set: rng.chance(60),
            inout: rng.chance(50),
            remove: rng.chance(60),
        };
        if a.get || a.set || a.inout || a.remove {
            break a;
        }
    };

    let keys = if rng.chance(40) { 2 } else { 1 };
    let width = if keys == 2 { 2 + rng.below(2) } else { 1 };
    let mut m = Model {
        data: (0..(3 + rng.below(4)) as i64 * width as i64).collect(),
        gone: vec![],
        log: vec![],
        xs: (0..2 + rng.below(4) as i64).map(|i| i * 10).collect(),
        old: vec![],
        oc_data: None,
        m: vec![
            (0..1 + rng.below(3) as i64).collect(),
            (0..1 + rng.below(3) as i64).map(|i| i + 10).collect(),
        ],
        noisy: vec![],
        ok: true,
    };
    m.gone = vec![false; m.data.len()];

    let (params, flat) = if keys == 2 {
        ("i: int, j: int", format!("i * {width} + j"))
    } else {
        ("i: int", "i".to_string())
    };
    let mut src = String::new();
    src.push_str("let NOISY: [int] = [];\n");
    src.push_str("fn noisy(i: int) -> int { NOISY.push(i); i }\n");
    src.push_str("struct C { data: [int], gone: [bool], log: [int] }\n");
    writeln!(src, "impl C {{\n    subscript({params}) -> int {{").unwrap();
    if arms.get {
        writeln!(src, "        get {{ self.log.push(1); self.data[{flat}] }}").unwrap();
    }
    if arms.set {
        writeln!(
            src,
            "        set {{ self.log.push(2); self.data[{flat}] = new_value; }}"
        )
        .unwrap();
    }
    if arms.inout {
        writeln!(
            src,
            "        inout {{ self.log.push(6); yield self.data[{flat}]; self.log.push(7); }}"
        )
        .unwrap();
    }
    if arms.remove {
        writeln!(src, "        remove {{ self.log.push(4); self.gone[{flat}] = true; }}")
            .unwrap();
    }
    src.push_str("    }\n}\n");
    writeln!(
        src,
        "let c = C {{ data = {:?}, gone = {:?}, log = [] }};",
        m.data, m.gone
    )
    .unwrap();
    src.push_str("let oc = c;\n");
    writeln!(src, "let xs = {:?};", m.xs).unwrap();
    src.push_str("let old: [int] = [];\n");
    writeln!(src, "let m = {:?};", m.m).unwrap();
    src.push_str("let ok = true;\n");

    let mut ops = vec![
        Op::XsRead,
        Op::XsWrite,
        Op::XsCompound,
        Op::XsRemove,
        Op::XsBorrow,
        Op::MRead,
        Op::MWrite,
        Op::MCompound,
        Op::MRemove,
        Op::AliasRebind,
    ];
    if arms.can_read() {
        ops.extend([Op::Read, Op::Read]);
    }
    if arms.can_write() {
        ops.extend([Op::Write, Op::Write]);
    }
    if arms.can_read() && arms.can_write() {
        ops.push(Op::WriteRead);
    }
    if arms.can_compound() {
        ops.push(Op::Compound);
    }
    if arms.remove {
        ops.push(Op::Remove);
    }
    if arms.inout {
        ops.push(Op::Borrow);
        ops.push(Op::AliasRebindC);
    }
    if arms.inout && arms.remove {
        ops.push(Op::BorrowRemove);
    }

    let steps = 6 + rng.below(16);
    let mut alias = 0usize;
    for _ in 0..steps {
        let usable: Vec<Op> = ops
            .iter()
            .copied()
            .filter(|op| {
                match op {
                    Op::XsRead | Op::XsWrite | Op::XsCompound | Op::XsBorrow => {
                        !m.xs.is_empty()
                    }
                    Op::XsRemove | Op::AliasRebind => m.xs.len() > 1,
                    Op::MRead | Op::MWrite | Op::MCompound | Op::MRemove => {
                        m.m.iter().any(|row| !row.is_empty())
                    }
                    _ => true,
                }
            })
            .collect();
        match rng.pick(&usable) {
            Op::Read => {
                let i = rng.below(m.data.len());
                src.push_str("ok = ok && c[");
                c_index(rng, &mut src, &mut m, keys, width, i);
                m.log_get(arms);
                writeln!(src, "] == {};", m.data[i]).unwrap();
            }
            Op::Write => {
                let i = rng.below(m.data.len());
                let v = rng.below(100) as i64;
                src.push_str("c[");
                c_index(rng, &mut src, &mut m, keys, width, i);
                writeln!(src, "] = {v};").unwrap();
                m.log_set(arms);
                m.data[i] = v;
            }
            Op::WriteRead => {
                let i = rng.below(m.data.len());
                let j = rng.below(m.data.len());
                src.push_str("c[");
                c_index(rng, &mut src, &mut m, keys, width, i);
                src.push_str("] = c[");
                c_index(rng, &mut src, &mut m, keys, width, j);
                src.push_str("];\n");
                m.log_get(arms);
                m.data[i] = m.data[j];
                m.log_set(arms);
            }
            Op::Compound => {
                let i = rng.below(m.data.len());
                let v = rng.below(10) as i64;
                src.push_str("c[");
                c_index(rng, &mut src, &mut m, keys, width, i);
                writeln!(src, "] += {v};").unwrap();
                if arms.inout {
                    m.log.extend([6, 7]);
                } else {
                    m.log.extend([1, 2]);
                }
                m.data[i] += v;
            }
            Op::Remove => {
                let i = rng.below(m.data.len());
                src.push_str("c[");
                c_index(rng, &mut src, &mut m, keys, width, i);
                src.push_str("].remove();\n");
                m.log.push(4);
                m.gone[i] = true;
            }
            op @ (Op::Borrow | Op::BorrowRemove) => {
                // A block scopes the place: the inout tail runs at the
                // closing brace, so the 6/7 pair brackets the op.
                let i = rng.below(m.data.len());
                let v = rng.below(100) as i64;
                src.push_str("{ let p = &c[");
                c_index(rng, &mut src, &mut m, keys, width, i);
                m.log.push(6);
                if op == Op::Borrow {
                    writeln!(src, "]; p = {v}; }};").unwrap();
                    m.data[i] = v;
                    m.log.push(7);
                } else {
                    src.push_str("]; p.remove(); };\n");
                    m.log.push(4);
                    m.gone[i] = true;
                    m.log.push(7);
                }
            }
            Op::XsRead => {
                let i = rng.below(m.xs.len());
                src.push_str("ok = ok && xs[");
                index(rng, &mut src, &mut m, i);
                writeln!(src, "] == {};", m.xs[i]).unwrap();
            }
            Op::XsWrite => {
                let i = rng.below(m.xs.len());
                let v = rng.below(100) as i64;
                src.push_str("xs[");
                index(rng, &mut src, &mut m, i);
                writeln!(src, "] = {v};").unwrap();
                m.xs[i] = v;
            }
            Op::XsCompound => {
                let i = rng.below(m.xs.len());
                let v = rng.below(10) as i64;
                src.push_str("xs[");
                index(rng, &mut src, &mut m, i);
                writeln!(src, "] += {v};").unwrap();
                m.xs[i] += v;
            }
            Op::XsRemove => {
                let i = rng.below(m.xs.len());
                src.push_str("ok = ok && xs[");
                index(rng, &mut src, &mut m, i);
                src.push_str("].remove() == ");
                writeln!(src, "{};", m.xs[i]).unwrap();
                m.xs.remove(i);
            }
            Op::XsBorrow => {
                let i = rng.below(m.xs.len());
                let v = rng.below(100) as i64;
                src.push_str("{ let p = &xs[");
                index(rng, &mut src, &mut m, i);
                writeln!(src, "]; p = {v}; }};").unwrap();
                m.xs[i] = v;
            }
            op @ (Op::MRead | Op::MWrite | Op::MCompound | Op::MRemove) => {
                let rows = m
                    .m
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| !row.is_empty())
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>();
                let i = rng.pick(&rows);
                let j = rng.below(m.m[i].len());
                match op {
                    Op::MRead => {
                        src.push_str("ok = ok && m[");
                        index(rng, &mut src, &mut m, i);
                        src.push_str("][");
                        index(rng, &mut src, &mut m, j);
                        writeln!(src, "] == {};", m.m[i][j]).unwrap();
                    }
                    Op::MWrite => {
                        let v = rng.below(100) as i64;
                        src.push_str("m[");
                        index(rng, &mut src, &mut m, i);
                        src.push_str("][");
                        index(rng, &mut src, &mut m, j);
                        writeln!(src, "] = {v};").unwrap();
                        m.m[i][j] = v;
                    }
                    Op::MCompound => {
                        let v = rng.below(10) as i64;
                        src.push_str("m[");
                        index(rng, &mut src, &mut m, i);
                        src.push_str("][");
                        index(rng, &mut src, &mut m, j);
                        writeln!(src, "] += {v};").unwrap();
                        m.m[i][j] += v;
                    }
                    _ => {
                        src.push_str("ok = ok && m[");
                        index(rng, &mut src, &mut m, i);
                        src.push_str("][");
                        index(rng, &mut src, &mut m, j);
                        src.push_str("].remove() == ");
                        writeln!(src, "{};", m.m[i][j]).unwrap();
                        m.m[i].remove(j);
                    }
                }
            }
            Op::AliasRebind => {
                let i = rng.below(m.xs.len());
                let v = rng.below(100) as i64;
                let fresh: Vec<i64> = (0..m.xs.len() as i64).map(|k| k + 50).collect();
                writeln!(src, "old = xs;").unwrap();
                writeln!(src, "let p{alias} = &xs[").unwrap();
                index(rng, &mut src, &mut m, i);
                src.push_str("];\n");
                writeln!(src, "xs = {fresh:?};").unwrap();
                writeln!(src, "p{alias} = {v};").unwrap();
                alias += 1;
                m.old = m.xs.clone();
                m.xs = fresh;
                m.old[i] = v;
            }
            Op::AliasRebindC => {
                // Borrow a slot, rebind `c` to a fresh struct that
                // keeps the shared `log`, then write through the
                // place: the write must land on the old list, which
                // `oc` still sees. The block scopes `p` so the inout
                // tail is deterministic.
                let i = rng.below(m.data.len());
                let v = rng.below(100) as i64;
                let fresh: Vec<i64> = (0..m.data.len() as i64).map(|k| k + 200).collect();
                src.push_str("{ oc = c; let p = &c[");
                c_index(rng, &mut src, &mut m, keys, width, i);
                writeln!(
                    src,
                    "]; c = C {{ data = {fresh:?}, gone = {:?}, log = c.log }}; p = {v}; }};",
                    vec![false; m.data.len()]
                )
                .unwrap();
                m.oc_data = Some(m.data.clone());
                m.data = fresh;
                m.gone = vec![false; m.data.len()];
                m.log.push(6);
                m.oc_data.as_mut().unwrap()[i] = v;
                m.log.push(7);
            }
        }
    }

    // Negative variant: emit one op the arm subset cannot serve and
    // require the compile error to name the missing arm.
    let key0 = if keys == 2 { "0, 0" } else { "0" };
    let mut missing = vec![];
    if !arms.can_read() {
        missing.push((format!("let q = c[{key0}];"), "no `get` arm"));
    }
    if !arms.can_write() {
        missing.push((format!("c[{key0}] = 1;"), "no `set` arm"));
    }
    if !arms.can_compound() {
        missing.push((format!("c[{key0}] += 1;"), "no `inout or get/set` arm"));
    }
    if !arms.remove {
        missing.push((format!("c[{key0}].remove();"), "no `remove` arm"));
    }
    if !missing.is_empty() && rng.chance(25) {
        let (poison, fragment) = missing.remove(rng.below(missing.len()));
        src.push_str(&poison);
        src.push('\n');
        return (src, Some(fragment.to_string()), String::new());
    }

    src.push_str(
        "let TEST_VALUE = f\"{ok}|{c.data}|{c.gone}|{c.log}|{xs}|{old}|{oc.data}|{m}|{NOISY}\";\n",
    );
    (src, None, m.render())
}

fn run(source: &str) -> Result<String, String> {
    let mut vm = Vm::execute(source, library::std).map_err(|e| e.to_string())?;
    vm.resolve_name_to_string("TEST_VALUE")
        .ok_or_else(|| "TEST_VALUE was not bound".to_string())?
        .map_err(|e| e.to_string())
}

#[test]
fn subscript_semantic_fuzz() {
    let cases: u64 = std::env::var("MIMAS_FUZZ_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let base: u64 = std::env::var("MIMAS_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5EED_5EED);
    let mut misses = Vec::new();
    for case in 0..cases {
        let mut rng = Rng(base.wrapping_add(case).wrapping_mul(0x9E3779B97F4A7C15));
        let (src, expect_err, want) = gen_case(&mut rng);
        if let Some(fragment) = expect_err {
            match run(&src) {
                Err(e) if e.contains(&fragment) => {}
                Err(e) => misses.push(format!(
                    "case {case} (replay: MIMAS_FUZZ_CASES={} MIMAS_FUZZ_SEED={base})\n{src}\n  wanted error containing `{fragment}`, got {e}",
                    case + 1
                )),
                Ok(got) => misses.push(format!(
                    "case {case} (replay: MIMAS_FUZZ_CASES={} MIMAS_FUZZ_SEED={base})\n{src}\n  wanted error containing `{fragment}`, program ran to {got}",
                    case + 1
                )),
            }
            if misses.len() >= 5 {
                break;
            }
            continue;
        }
        match run(&src) {
            Ok(got) if got == want => {}
            Ok(got) => misses.push(format!(
                "case {case} (replay: MIMAS_FUZZ_CASES={} MIMAS_FUZZ_SEED={base})\n{src}\n  got  {got}\n  want {want}",
                case + 1
            )),
            Err(e) => misses.push(format!(
                "case {case} (replay: MIMAS_FUZZ_CASES={} MIMAS_FUZZ_SEED={base})\n{src}\n  error {e}",
                case + 1
            )),
        }
        if misses.len() >= 5 {
            break;
        }
    }
    assert!(
        misses.is_empty(),
        "{} fuzz case(s) came out wrong:\n{}",
        misses.len(),
        misses.join("\n---\n")
    );
}
