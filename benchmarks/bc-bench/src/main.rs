//! Three-way benchmark over the `benchmarks/*.mim` suite:
//!
//!   vm    — the stock interpreter (`Vm::run`)
//!   bc    — the same `Program` with bcgen-specialized bodies installed
//!           (`Vm::install_bc`); identical semantics, generated dispatch
//!   jit   — Cranelift-JIT'd bodies through the same `install_bc` slot
//!           (`mimas_jit::compile` once per workload, timing amortized out)
//!   rust  — the rustgen-transpiled module from `host/corpus-gen/src/gen/`
//!           (typed native Rust against the mrt shim below)
//!
//! Usage: `bc-bench [workload ...] [iters]` — defaults to every workload,
//! best-of-3.

use std::time::{Duration, Instant};

// The rustgen-emitted modules `use crate::mrt::*`; the bench's mrt shim covers
// exactly the surface these five pages use.
pub mod mrt {
    pub type Arr<T> = imbl::Vector<T>;
    pub fn print<T: std::fmt::Display>(x: T) {
        println!("{x}");
    }
    pub fn to_float(x: i64) -> f64 {
        x as f64
    }
    pub fn new_filled<T: Clone>(v: T, n: i64) -> Arr<T> {
        (0..n.max(0)).map(|_| v.clone()).collect()
    }
}

mod bc {
    pub mod fib_iter {
        include!(concat!(env!("OUT_DIR"), "/fib_iter.rs"));
    }
    pub mod fib_rec {
        include!(concat!(env!("OUT_DIR"), "/fib_rec.rs"));
    }
    pub mod mandelbrot {
        include!(concat!(env!("OUT_DIR"), "/mandelbrot.rs"));
    }
    pub mod prime_numbers {
        include!(concat!(env!("OUT_DIR"), "/prime_numbers.rs"));
    }
    pub mod physics {
        include!(concat!(env!("OUT_DIR"), "/physics.rs"));
    }
}

mod rust {
    pub mod fib_iter {
        #![allow(warnings)]
        include!("../../../../host/corpus-gen/src/gen/mimas_benchmarks_fib_iter_fib_iter.rs");
    }
    pub mod fib_rec {
        #![allow(warnings)]
        include!("../../../../host/corpus-gen/src/gen/mimas_benchmarks_fib_rec_fib_rec.rs");
    }
    pub mod mandelbrot {
        #![allow(warnings)]
        include!("../../../../host/corpus-gen/src/gen/mimas_benchmarks_mandelbrot_mandelbrot.rs");
    }
    pub mod prime_numbers {
        #![allow(warnings)]
        include!("../../../../host/corpus-gen/src/gen/mimas_benchmarks_prime_numbers_prime_numbers.rs");
    }
    pub mod physics {
        #![allow(warnings)]
        include!("../../../../host/corpus-gen/src/gen/mimas_benchmarks_physics_physics.rs");
    }
}

fn source(name: &str) -> String {
    std::fs::read_to_string(format!("../{name}/{name}.mim")).expect("read .mim")
}

/// `install_bc` is applied per run — a fresh `Vm` each time so heap state
/// (arrays, interned strings) doesn't leak between iterations.
fn run_vm(source: &str, bodies: Option<Vec<Option<mimas::vm::bc::BodyFn>>>) -> Duration {
    let (program, sources) =
        mimas::Vm::compile_parts(&[("main", source)], mimas::library::std).expect("compiles");
    let mut vm = mimas::Vm::new();
    vm.load_prebuilt(program, sources, mimas::library::std);
    if let Some(bodies) = bodies {
        vm.install_bc(bodies);
    }
    let t = Instant::now();
    vm.run().expect("vm run");
    t.elapsed()
}

fn run_rust(f: fn()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}

/// `BC_LANES=bc,rust` selects which lanes run (default: all) — for profiling a
/// single lane without paying for the others.
fn lanes() -> Vec<String> {
    std::env::var("BC_LANES")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(|_| {
            ["vm", "bc", "jit", "llvm", "rust"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
}

fn bench(name: &str, bodies: fn() -> Vec<Option<mimas::vm::bc::BodyFn>>, rust_fn: fn(), iters: usize) {
    let src = source(name);
    let lanes = lanes();
    let mut vm_best = Duration::MAX;
    let mut bc_best = Duration::MAX;
    let mut jit_best = Duration::MAX;
    let mut llvm_best = Duration::MAX;
    let mut rust_best = Duration::MAX;
    if lanes.iter().any(|l| l == "vm") {
        for _ in 0..iters {
            vm_best = vm_best.min(run_vm(&src, None));
        }
    }
    if lanes.iter().any(|l| l == "bc") {
        for _ in 0..iters {
            let b = bodies();
            bc_best = bc_best.min(run_vm(&src, Some(b)));
        }
    }
    // The JIT module is compiled once (its `compile` time is a one-off cost,
    // reported separately); every iteration installs the same body table into a
    // fresh Vm, exactly like the bc lane.
    let mut jit_compile = Duration::ZERO;
    if lanes.iter().any(|l| l == "jit") {
        let (program, _s) =
            mimas::Vm::compile_parts(&[("main", src.as_str())], mimas::library::std)
                .expect("jit lane compile");
        let t = Instant::now();
        let jit = jit::compile(&program).expect("jit compile");
        jit_compile = t.elapsed();
        for _ in 0..iters {
            jit_best = jit_best.min(run_vm(&src, Some(jit.bodies())));
        }
    }
    // Same one-off-compile treatment as the Cranelift lane — the LLVM tier's
    // compile latency is reported separately since it decides whether this is
    // a hot-tier or a compile-once top tier.
    let mut llvm_compile = Duration::ZERO;
    if lanes.iter().any(|l| l == "llvm") {
        let (program, _s) =
            mimas::Vm::compile_parts(&[("main", src.as_str())], mimas::library::std)
                .expect("llvm lane compile");
        let t = Instant::now();
        let llvm = llvm_jit::compile(&program).expect("llvm-jit compile");
        llvm_compile = t.elapsed();
        for _ in 0..iters {
            llvm_best = llvm_best.min(run_vm(&src, Some(llvm.bodies())));
        }
    }
    if lanes.iter().any(|l| l == "rust") {
        for _ in 0..iters {
            rust_best = rust_best.min(run_rust(rust_fn));
        }
    }
    // Unselected lanes stay `Duration::MAX` — show them as `-` rather than a
    // garbage ratio (MAX as f64 prints 18446744073709551616.000s).
    let dur = |d: Duration| {
        if d == Duration::MAX {
            "-".to_string()
        } else {
            format!("{d:>9.3?}")
        }
    };
    let ratio = |a: Duration, b: Duration| {
        if a == Duration::MAX || b == Duration::MAX {
            "-".to_string()
        } else {
            format!("x{:.2}", a.as_secs_f64() / b.as_secs_f64())
        }
    };
    println!(
        "{name:>14}  vm {}  bc {} ({})  jit {} ({}, compile {})  llvm {} ({}, compile {})  rust {} ({} vs bc)",
        dur(vm_best),
        dur(bc_best),
        ratio(vm_best, bc_best),
        dur(jit_best),
        ratio(vm_best, jit_best),
        dur(jit_compile),
        dur(llvm_best),
        ratio(vm_best, llvm_best),
        dur(llvm_compile),
        dur(rust_best),
        ratio(bc_best, rust_best),
    );
}

fn main() {
    let mut names: Vec<&str> = Vec::new();
    let mut iters = 3usize;
    for arg in std::env::args().skip(1) {
        match arg.parse::<usize>() {
            Ok(n) => iters = n,
            Err(_) => names.push(Box::leak(arg.into_boxed_str())),
        }
    }
    let all: [(&str, fn() -> Vec<Option<mimas::vm::bc::BodyFn>>, fn()); 5] = [
        ("fib_iter", bc::fib_iter::bodies, rust::fib_iter::run_top_level),
        ("fib_rec", bc::fib_rec::bodies, rust::fib_rec::run_top_level),
        ("mandelbrot", bc::mandelbrot::bodies, rust::mandelbrot::run_top_level),
        ("prime_numbers", bc::prime_numbers::bodies, rust::prime_numbers::run_top_level),
        ("physics", bc::physics::bodies, rust::physics::run_top_level),
    ];
    println!("workload           vm          bc (speedup)      jit (speedup)            llvm (speedup)           rust (bc/rust)");
    for (name, bodies, rust_fn) in all {
        if names.is_empty() || names.contains(&name) {
            bench(name, bodies, rust_fn, iters);
        }
    }
}
