//! Three-way benchmark over the `benchmarks/*.mim` suite:
//!
//!   vm    — the stock interpreter (`Vm::run`)
//!   bc    — the same `Program` with bcgen-specialized bodies installed
//!           (`Vm::install_bc`); identical semantics, generated dispatch
//!   jit   — Cranelift-JIT'd bodies through the same `install_bc` slot
//!           (`mimas_jit::compile` once per workload, timing amortized out)
//!   native— handwritten idiomatic Rust (`src/native.rs`), asserted against
//!           the vm lane's captured `print` output each run
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

mod native;

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

fn run_native(f: fn() -> String) -> Duration {
    let t = Instant::now();
    println!("{}", f());
    t.elapsed()
}

/// The plain-interpreter lane's `print` output — the reference the
/// handwritten `native` lane is asserted against (see `bench`).
fn vm_output(source: &str) -> Vec<String> {
    let (program, sources) =
        mimas::Vm::compile_parts(&[("main", source)], mimas::library::std).expect("compiles");
    let mut vm = mimas::Vm::new();
    vm.load_prebuilt(program, sources, mimas::library::std);
    vm.run().expect("vm run");
    vm.fixture::<mimas::vm::fixtures::Prints>()
        .take()
        .into_iter()
        .map(|(_, _, line)| line)
        .collect()
}

/// `BC_LANES=bc,jit` selects which lanes run (default: all) — for profiling a
/// single lane without paying for the others.
fn lanes() -> Vec<String> {
    std::env::var("BC_LANES")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(|_| {
            ["vm", "bc", "jit", "native"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
}

fn bench(
    name: &str,
    mim: &str,
    bodies: Option<fn() -> Vec<Option<mimas::vm::bc::BodyFn>>>,
    native_fn: Option<fn() -> String>,
    iters: usize,
) {
    let src = source(mim);
    let lanes = lanes();
    let mut vm_best = Duration::MAX;
    let mut bc_best = Duration::MAX;
    let mut jit_best = Duration::MAX;
    let mut native_best = Duration::MAX;
    if lanes.iter().any(|l| l == "vm") && bodies.is_some() {
        for _ in 0..iters {
            vm_best = vm_best.min(run_vm(&src, None));
        }
    }
    if lanes.iter().any(|l| l == "bc") {
        if let Some(bodies) = bodies {
            for _ in 0..iters {
                let b = bodies();
                bc_best = bc_best.min(run_vm(&src, Some(b)));
            }
        }
    }
    // The JIT module is compiled once (its `compile` time is a one-off cost,
    // reported separately); every iteration installs the same body table into a
    // fresh Vm, exactly like the bc lane.
    let mut jit_compile = Duration::ZERO;
    if lanes.iter().any(|l| l == "jit") && bodies.is_some() {
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
    if lanes.iter().any(|l| l == "native") {
        if let Some(native_fn) = native_fn {
            // cross-lane check: the native lane must print what `vm` prints —
            // one untimed reference run of each, then the timed loop
            assert_eq!(
                vec![native_fn()],
                vm_output(&src),
                "{name}: native lane disagrees with the interpreter"
            );
            for _ in 0..iters {
                native_best = native_best.min(run_native(native_fn));
            }
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
        "{name:>14}  vm {}  bc {} ({})  jit {} ({}, compile {})  native {} ({} jit/native)",
        dur(vm_best),
        dur(bc_best),
        ratio(vm_best, bc_best),
        dur(jit_best),
        ratio(vm_best, jit_best),
        dur(jit_compile),
        dur(native_best),
        ratio(jit_best, native_best),
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
    #[allow(clippy::type_complexity)]
    let all: [(
        &str,
        &str,
        Option<fn() -> Vec<Option<mimas::vm::bc::BodyFn>>>,
        Option<fn() -> String>,
    ); 6] = [
        (
            "fib_iter",
            "fib_iter",
            Some(bc::fib_iter::bodies),
            Some(native::fib_iter),
        ),
        (
            "fib_rec",
            "fib_rec",
            Some(bc::fib_rec::bodies),
            Some(native::fib_rec),
        ),
        (
            "mandelbrot",
            "mandelbrot",
            Some(bc::mandelbrot::bodies),
            Some(native::mandelbrot),
        ),
        (
            "prime_numbers",
            "prime_numbers",
            Some(bc::prime_numbers::bodies),
            Some(native::prime_numbers),
        ),
        (
            "physics",
            "physics",
            Some(bc::physics::bodies),
            Some(native::physics),
        ),
        // SoA layout of the same physics program — native-lane only, verified
        // against physics.mim's vm output. The AoS/SoA delta is the reference
        // for what a struct-of-arrays `Instance` layout could buy mimas.
        ("physics_soa", "physics", None, Some(native::physics_soa)),
    ];
    println!("workload           vm          bc (speedup)      jit (speedup, compile)    native (jit/native)");
    for (name, mim, bodies, native_fn) in all {
        if names.is_empty() || names.contains(&name) {
            bench(name, mim, bodies, native_fn, iters);
        }
    }
}
