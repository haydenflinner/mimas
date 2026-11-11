//! `wgbench <file.mim>` — time the interpreter against the IR-lane wasm on
//! the same program, so coverage numbers translate into a speed number.
//!
//! Interpreter: `mimas::Vm` (build this example --release — a debug VM is
//! several times slower and skews the comparison). Wasm: wasmtime hosting
//! `wfull::emit_waffle_ir`'s module with every `env` import stubbed — natives
//! return zero-typed values, except `print`, whose args are captured so the
//! wasm run stays observable and checkable against the interpreter's
//! stdout.
//!
//! Bodies the IR lane can't emit still `skip` — if body 0 skips, the
//! program can't run on wasm at all and there's nothing to time.

use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mimas::Vm;
use mimas_wasmgen::Opts;
use mimas_wasmgen::wfull::emit_waffle_ir;
use wasmtime::{Engine, Global, Linker, Module, Store, Val};

fn strip_print(src: &str) -> String {
    src.lines()
        .filter(|line| !line.trim_start().starts_with("print("))
        .collect::<Vec<_>>()
        .join("\n")
}

fn val_zero(ty: wasmtime::ValType) -> Val {
    match ty {
        wasmtime::ValType::I32 => Val::I32(0),
        wasmtime::ValType::I64 => Val::I64(0),
        wasmtime::ValType::F32 => Val::F32(0),
        wasmtime::ValType::F64 => Val::F64(0),
        _ => Val::I64(0),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: wgbench <file.mim> [--no-print]");
    let drop_print = std::env::args().any(|a| a == "--no-print");
    let src = fs::read_to_string(&path).expect("read script");
    let src = if drop_print { strip_print(&src) } else { src };

    // one compile feeds both lanes
    let (program, ir, sources) =
        Vm::compile_parts_ir(&[("main", src.as_str())], mimas::library::std)
            .unwrap_or_else(|e| panic!("compile: {e}"));
    // map native ids -> names: `print` is captured, the float methods get real
    // bodies (a stub returning 0 makes any program with `cos`/`atan2`/`clamp`
    // do different work on the two lanes, and the comparison lies)
    let lib = Vm::new().install_library(mimas::library::std);
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut math = mimas_wasmgen::MathNatives::new();
    for (id, e) in lib.natives() {
        let name = match e {
            api::ApiEntry::Function(f) => format!("{}::{}", f.module.join("::"), f.name),
            api::ApiEntry::Method(m) => format!("{}::{}", m.recv_ty, m.name),
            _ => format!("native {id:?}"),
        };
        let k = id.index() as u32;
        match name.as_str() {
            "float::abs" => {
                math.insert(k, mimas_wasmgen::MathOp::Abs);
            }
            "float::min" => {
                math.insert(k, mimas_wasmgen::MathOp::Min);
            }
            "float::max" => {
                math.insert(k, mimas_wasmgen::MathOp::Max);
            }
            "float::floor" => {
                math.insert(k, mimas_wasmgen::MathOp::Floor);
            }
            "float::to" => {
                math.insert(k, mimas_wasmgen::MathOp::Identity);
            }
            _ => {}
        }
        names.insert(k, name);
    }
    let w = emit_waffle_ir(
        &ir,
        &program.strs,
        &Opts::default(),
        None,
        Some(&math),
        0,
        0,
    )
    .unwrap_or_else(|e| panic!("emit_waffle_ir: {e}"));
    let total = ir.bodies.len();
    println!("waffle: {}/{} bodies emitted", w.bodies.len(), total);
    for s in &w.skipped {
        println!("  skip b{:<3} {}", s.body, s.reason);
    }
    if !w.bodies.iter().any(|b| b.body == 0) {
        eprintln!("body 0 skipped — program can't run on wasm, nothing to time");
        return;
    }

    // ---- wasm side: stub every env import, capture print args ----
    let prints: Arc<Mutex<Vec<Vec<Val>>>> = Arc::new(Mutex::new(Vec::new()));
    let stubbed: Arc<Mutex<std::collections::BTreeSet<String>>> =
        Arc::new(Mutex::new(Default::default()));
    let engine = Engine::default();
    let module = Module::new(&engine, &w.bytes[..]).unwrap_or_else(|e| panic!("wasm: {e}"));
    let mut linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    for imp in module.imports() {
        match imp.ty() {
            wasmtime::ExternType::Func(ft) => {
                let pname = imp
                    .name()
                    .strip_prefix('n')
                    .and_then(|s| s.split('_').next())
                    .and_then(|s| s.parse::<u32>().ok())
                    .and_then(|id| names.get(&id).cloned())
                    .unwrap_or_else(|| imp.name().to_string());
                if std::env::var("WG_IMPORTS").is_ok() {
                    println!("import {}.{} -> {pname} {:?}", imp.module(), imp.name(), ft);
                }
                let is_print = pname.ends_with("::print");
                let captured = prints.clone();
                // floats cross the import boundary either as f64 or as their
                // bit pattern in an i64 (the emitter's `Word` class for a
                // classless native arg); read both
                let fv = |v: &Val| match v {
                    Val::F64(b) => f64::from_bits(*b),
                    Val::I64(i) => f64::from_bits(*i as u64),
                    Val::F32(b) => f32::from_bits(*b) as f64,
                    Val::I32(i) => *i as f64,
                    _ => 0.0,
                };
                let f1 = move |p: &[Val]| fv(&p[0]);
                let f2 = move |p: &[Val]| fv(&p[1]);
                // the float methods a numeric program reaches for; anything
                // else is still a zero stub, and the summary names it
                let body: Option<Box<dyn Fn(&[Val]) -> f64 + Send + Sync>> = match pname.as_str() {
                    "float::sin" => Some(Box::new(move |p| f1(p).sin())),
                    "float::cos" => Some(Box::new(move |p| f1(p).cos())),
                    "float::tan" => Some(Box::new(move |p| f1(p).tan())),
                    "float::asin" => Some(Box::new(move |p| f1(p).asin())),
                    "float::acos" => Some(Box::new(move |p| f1(p).acos())),
                    "float::atan" => Some(Box::new(move |p| f1(p).atan())),
                    "float::atan2" => Some(Box::new(move |p| f1(p).atan2(f2(p)))),
                    "float::exp" => Some(Box::new(move |p| f1(p).exp())),
                    "float::ln" => Some(Box::new(move |p| f1(p).ln())),
                    "float::pow" => Some(Box::new(move |p| f1(p).powf(f2(p)))),
                    "float::hypot" => Some(Box::new(move |p| f1(p).hypot(f2(p)))),
                    "float::round" => Some(Box::new(move |p| f1(p).round())),
                    "float::ceil" => Some(Box::new(move |p| f1(p).ceil())),
                    "float::signum" => Some(Box::new(move |p| f1(p).signum())),
                    "float::clamp" => Some(Box::new(move |p| f1(p).clamp(f2(p), fv(&p[2])))),
                    "float::to" => Some(Box::new(move |p| f1(p))),
                    _ => None,
                };
                if body.is_none() && !is_print {
                    stubbed.lock().unwrap().insert(pname.clone());
                }
                // the result types come from the import's signature, not from
                // the placeholder values wasmtime hands over
                let ret_tys: Vec<wasmtime::ValType> = ft.results().collect();
                linker
                    .func_new(imp.module(), imp.name(), ft, move |_c, params, results| {
                        if is_print {
                            captured.lock().unwrap().push(params.to_vec());
                        }
                        for (i, r) in results.iter_mut().enumerate() {
                            let t = ret_tys[i].clone();
                            *r = match (&body, &t) {
                                (Some(f), wasmtime::ValType::F64) => Val::F64(f(params).to_bits()),
                                // a `Word` result carries a float as its bit pattern
                                (Some(f), wasmtime::ValType::I64) => {
                                    Val::I64(f(params).to_bits() as i64)
                                }
                                _ => val_zero(t),
                            };
                        }
                        Ok(())
                    })
                    .unwrap();
            }
            wasmtime::ExternType::Global(gt) => {
                let init = val_zero(gt.content().clone());
                let g = Global::new(&mut store, gt, init).unwrap();
                linker
                    .define(&mut store, imp.module(), imp.name(), g)
                    .unwrap();
            }
            _ => panic!("unstubbable import {} {:?}", imp.name(), imp.ty()),
        }
    }
    // one instance per run: the bump arena has no GC, so a program's garbage
    // accumulates for the life of an instance and a long enough series of
    // reps on one instance traps in `__alloc`. The high-water mark after a
    // run is the allocation volume the lane paid for that run.
    let run_once = |store: &mut Store<()>| -> Result<(f64, i32), String> {
        let inst = linker
            .instantiate(&mut *store, &module)
            .map_err(|e| format!("instantiate: {e}"))?;
        let b0 = inst.get_func(&mut *store, "b0").expect("b0 export");
        let fty = b0.ty(&*store);
        let params: Vec<Val> = fty.params().map(val_zero).collect();
        let mut results: Vec<Val> = fty.results().map(val_zero).collect();
        let t = Instant::now();
        let r = b0.call(&mut *store, &params, &mut results);
        let dt = t.elapsed().as_secs_f64();
        let hp = inst
            .get_global(&mut *store, "__hp")
            .map(|g| g.get(&mut *store).i32().unwrap_or(0))
            .unwrap_or(0);
        r.map_err(|e| e.to_string().lines().last().unwrap_or("trap").to_string())?;
        Ok((dt, hp))
    };
    // warmup + timed reps sized to ~2s of wasm work. A trap is reported, not
    // fatal, so the interpreter still gets timed.
    let mut wasm_med = None;
    match run_once(&mut store) {
        Err(e) => println!("wasm: trapped: {e}"),
        Ok((est, hp)) => {
            println!("wasm:   arena high-water {:.1} MB per run", hp as f64 / 1e6);
            let reps = (2.0 / est.max(1e-6)).ceil().clamp(1.0, 50.0) as usize;
            let mut wt = Vec::new();
            for _ in 0..reps {
                // a fresh store too: wasmtime keeps the old instances' memories alive otherwise
                let mut st = Store::new(&engine, ());
                match run_once(&mut st) {
                    Ok((dt, _)) => wt.push(dt),
                    Err(e) => {
                        println!("wasm: trapped on a rep: {e}");
                        break;
                    }
                }
            }
            if !wt.is_empty() {
                let m = median(wt.clone());
                println!("wasm:   median {m:.4}s over {} reps", wt.len());
                wasm_med = Some(m);
            }
        }
    }

    // ---- interpreter side: fresh Vm per rep, run() timed ----
    let mut it = Vec::new();
    let ireps = 5;
    for _ in 0..ireps {
        let mut vm = Vm::new();
        vm.load_prebuilt(program.clone(), sources.clone(), mimas::library::std);
        let t = Instant::now();
        vm.run().unwrap_or_else(|e| panic!("vm run: {e}"));
        it.push(t.elapsed().as_secs_f64());
    }
    let interp_med = median(it);
    println!("interp: median {interp_med:.4}s over {ireps} reps");
    if let Some(w) = wasm_med {
        println!("speedup: {:.1}x", interp_med / w);
    }

    // captured wasm prints — compare against what the interpreter printed
    let ps = prints.lock().unwrap();
    println!("wasm print calls: {}", ps.len());
    for (i, args) in ps.iter().take(10).enumerate() {
        let shown: Vec<String> = args
            .iter()
            .map(|v| match v {
                Val::F64(b) => format!("{}", f64::from_bits(*b)),
                // a float handed over as its bit pattern reads back as one
                Val::I64(i)
                    if (*i as u64) >> 52 & 0x7ff > 0x380 && (*i as u64) >> 52 & 0x7ff < 0x480 =>
                {
                    format!("{} (f64 bits)", f64::from_bits(*i as u64))
                }
                other => format!("{other:?}"),
            })
            .collect();
        println!("  print[{i}]: {}", shown.join(" "));
    }
    let st = stubbed.lock().unwrap();
    if !st.is_empty() {
        println!(
            "imports stubbed to zero ({}): {}",
            st.len(),
            st.iter().cloned().collect::<Vec<_>>().join(", ")
        );
    }
}
