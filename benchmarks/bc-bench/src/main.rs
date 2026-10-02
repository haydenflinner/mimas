//! Three-way benchmark over the `benchmarks/*.mim` suite:
//!
//!   vm    — the stock interpreter (`Vm::run`)
//!   bc    — the same `Program` with bcgen-specialized bodies installed
//!           (`Vm::install_bc`); identical semantics, generated dispatch
//!   jit   — Cranelift-JIT'd bodies through the same `install_bc` slot
//!           (`mimas_jit::compile` once per workload, timing amortized out)
//!
//! Usage: `bc-bench [workload ...] [iters]` — defaults to every workload,
//! best-of-3.

use std::time::{Duration, Instant};

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

/// `BC_LANES=bc,jit` selects which lanes run (default: all) — for profiling a
/// single lane without paying for the others.
fn lanes() -> Vec<String> {
    std::env::var("BC_LANES")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(|_| {
            ["vm", "bc", "jit"].iter().map(|s| s.to_string()).collect()
        })
}

fn bench(name: &str, bodies: fn() -> Vec<Option<mimas::vm::bc::BodyFn>>, iters: usize) {
    let src = source(name);
    let lanes = lanes();
    let mut vm_best = Duration::MAX;
    let mut bc_best = Duration::MAX;
    let mut jit_best = Duration::MAX;
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
        "{name:>14}  vm {}  bc {} ({})  jit {} ({}, compile {})",
        dur(vm_best),
        dur(bc_best),
        ratio(vm_best, bc_best),
        dur(jit_best),
        ratio(vm_best, jit_best),
        dur(jit_compile),
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
    let all: [(&str, fn() -> Vec<Option<mimas::vm::bc::BodyFn>>); 5] = [
        ("fib_iter", bc::fib_iter::bodies),
        ("fib_rec", bc::fib_rec::bodies),
        ("mandelbrot", bc::mandelbrot::bodies),
        ("prime_numbers", bc::prime_numbers::bodies),
        ("physics", bc::physics::bodies),
    ];
    println!("workload           vm          bc (speedup)      jit (speedup, compile)");
    for (name, bodies) in all {
        if names.is_empty() || names.contains(&name) {
            bench(name, bodies, iters);
        }
    }
}
