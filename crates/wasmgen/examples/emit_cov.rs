// Compile a .mim that calls cov::* natives into resumable wasm, printing a
// JSON manifest that also maps native ids -> names, so the Node driver can
// bind env.n<id>_<sig> imports to the cov sink.
//
// The cov fns are compile-time declarations only — their Rust bodies never
// run; the wasm imports call the host-side cov sink.
//
// usage: emit_cov <file.mim> [out.wasm]
use mimas::native;
use vm::anon::T;
use vm::Ctx;

#[native]
fn hit<'gc>(_ctx: Ctx<'gc>, _p: i64) {}

#[native]
fn pass<'gc>(_ctx: Ctx<'gc>, _p: i64, v: T<'gc>) -> T<'gc> {
    v
}

#[native]
fn begin<'gc>(_ctx: Ctx<'gc>, _d: i64) -> bool {
    true
}

#[native]
fn lhs<'gc>(_ctx: Ctx<'gc>, v: T<'gc>) -> T<'gc> {
    v
}

#[native]
fn rhs<'gc>(_ctx: Ctx<'gc>, v: T<'gc>) -> T<'gc> {
    v
}

#[native]
fn cmp<'gc>(_ctx: Ctx<'gc>, _d: i64, _k: i64, _op: i64, v: bool) -> bool {
    v
}

#[native]
fn cond<'gc>(_ctx: Ctx<'gc>, _d: i64, _k: i64, v: bool) -> bool {
    v
}

#[native]
fn dec<'gc>(_ctx: Ctx<'gc>, _d: i64, v: bool) -> bool {
    v
}

// DataFrame-style host objects: i64 handles, scalars back
#[native]
fn new<'gc>(_ctx: Ctx<'gc>, n: i64) -> i64 {
    n
}

#[native]
fn filter<'gc>(_ctx: Ctx<'gc>, h: i64, lo: i64) -> i64 {
    let _ = lo;
    h
}

#[native]
fn len<'gc>(_ctx: Ctx<'gc>, h: i64) -> i64 {
    h
}

#[native]
fn sum<'gc>(_ctx: Ctx<'gc>, h: i64) -> f64 {
    h as f64
}

fn main() {
    let mut pos = std::env::args().skip(1);
    let path = pos.next().expect("usage: emit_cov <file.mim> [out.wasm]");
    let out = pos.next().unwrap_or_else(|| "out.wasm".to_string());
    let source = std::fs::read_to_string(&path).expect("read");
    let mut ids = Vec::new();
    let (program, _s) = vm::Vm::compile_parts(&[("main", source.as_str())], |api| {
        mimas::library::std(api);
        let mut cov = api.module("cov");
        ids.push((cov.add(hit).index() as u32, "cov::hit"));
        ids.push((cov.add(pass).index() as u32, "cov::pass"));
        ids.push((cov.add(begin).index() as u32, "cov::begin"));
        ids.push((cov.add(lhs).index() as u32, "cov::lhs"));
        ids.push((cov.add(rhs).index() as u32, "cov::rhs"));
        ids.push((cov.add(cmp).index() as u32, "cov::cmp"));
        ids.push((cov.add(cond).index() as u32, "cov::cond"));
        ids.push((cov.add(dec).index() as u32, "cov::dec"));
        let mut df = api.module("df");
        ids.push((df.add(new).index() as u32, "df::new"));
        ids.push((df.add(filter).index() as u32, "df::filter"));
        ids.push((df.add(len).index() as u32, "df::len"));
        ids.push((df.add(sum).index() as u32, "df::sum"));
    })
    .expect("compile");
    let w = mimas_wasmgen::resume::emit_resumable(&program, &Default::default())
        .expect("emit resumable");
    std::fs::write(&out, &w.bytes).expect("write wasm");
    let natives = ids
        .iter()
        .map(|(i, n)| format!("\"{i}\":\"{n}\""))
        .collect::<Vec<_>>()
        .join(",");
    let bodies = w
        .bodies
        .iter()
        .map(|b| format!("{{\"body\":{},\"func\":{},\"name\":\"{}\"}}", b.body, b.func, b.name))
        .collect::<Vec<_>>()
        .join(",");
    let skipped = w
        .skipped
        .iter()
        .map(|s| {
            format!(
                "{{\"body\":{},\"reason\":\"{}\"}}",
                s.body,
                s.reason.replace('"', "'")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"bytes\":{},\"bodies\":[{}],\"skipped\":[{}],\"natives\":{{{}}}}}",
        w.bytes.len(),
        bodies,
        skipped,
        natives
    );
}
