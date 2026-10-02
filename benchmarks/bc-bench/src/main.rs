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

fn bench(name: &str, bodies: fn() -> Vec<Option<mimas::vm::bc::BodyFn>>, rust_fn: fn(), iters: usize) {
    let src = source(name);
    let mut vm_best = Duration::MAX;
    let mut bc_best = Duration::MAX;
    let mut jit_best = Duration::MAX;
    let mut rust_best = Duration::MAX;
    for _ in 0..iters {
        vm_best = vm_best.min(run_vm(&src, None));
    }
    for _ in 0..iters {
        let b = bodies();
        bc_best = bc_best.min(run_vm(&src, Some(b)));
    }
    // The JIT module is compiled once (its `compile` time is a one-off cost,
    // reported separately); every iteration installs the same body table into a
    // fresh Vm, exactly like the bc lane.
    let (program, _s) =
        mimas::Vm::compile_parts(&[("main", src.as_str())], mimas::library::std)
            .expect("jit lane compile");
    let t = Instant::now();
    let jit = jit::compile(&program).expect("jit compile");
    let jit_compile = t.elapsed();
    for _ in 0..iters {
        jit_best = jit_best.min(run_vm(&src, Some(jit.bodies())));
    }
    for _ in 0..iters {
        rust_best = rust_best.min(run_rust(rust_fn));
    }
    println!(
        "{name:>14}  vm {:>9.3?}  bc {:>9.3?} (x{:.2})  jit {:>9.3?} (x{:.2}, compile {:>9.3?})  rust {:>9.3?} (x{:.2} vs bc)",
        vm_best,
        bc_best,
        vm_best.as_secs_f64() / bc_best.as_secs_f64(),
        jit_best,
        vm_best.as_secs_f64() / jit_best.as_secs_f64(),
        jit_compile,
        rust_best,
        bc_best.as_secs_f64() / rust_best.as_secs_f64(),
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
    println!("workload           vm          bc (speedup)      jit (speedup)            rust (bc/rust)");
    for (name, bodies, rust_fn) in all {
        if names.is_empty() || names.contains(&name) {
            bench(name, bodies, rust_fn, iters);
        }
    }
}
