//! Handwritten-Rust lane for the five-workload bench — the same programs as
//! `benchmarks/<name>/<name>.mim`, transcribed to idiomatic Rust rather than
//! rustgen's `imbl`-shim output — plus `physics_soa`, a struct-of-arrays
//! variant of `physics.mim` kept as the AoS-vs-SoA reference for future
//! `Instance` layout work. Each entry returns the string its `.mim` twin
//! `print`s at the end; `bench()` asserts it equals the VM lane's captured
//! output, so a lane that silently disagrees fails loudly.

/// `fib_iter.mim` — i64 loop, `N = 20_000_000`, mod-1e9+7 accumulation.
pub fn fib_iter() -> String {
    const N: i64 = 20_000_000;
    const MOD: i64 = 1_000_000_007;
    let (mut a, mut b) = (0i64, 1i64);
    for _ in 0..N {
        let t = (a + b) % MOD;
        a = b;
        b = t;
    }
    a.to_string()
}

/// `fib_rec.mim` — plain recursion, `fib(30)` x `REPEAT = 20`. The arg goes
/// through `black_box` so LLVM can't hoist the pure call out of the loop
/// (otherwise this measures one `fib(30)`, not twenty).
pub fn fib_rec() -> String {
    const TARGET: i64 = 30;
    const REPEAT: i64 = 20;
    fn fib(n: i64) -> i64 {
        if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
    }
    let mut result = 0;
    for _ in 0..REPEAT {
        result = std::hint::black_box(fib(std::hint::black_box(TARGET)));
    }
    result.to_string()
}

/// `mandelbrot.mim` — f64 iteration loop over a 450x450 grid, checksum of
/// iteration counts.
pub fn mandelbrot() -> String {
    const WIDTH: i64 = 450;
    const HEIGHT: i64 = 450;
    const MAXITER: i64 = 400;
    let mut checksum = 0i64;
    for py in 0..HEIGHT {
        let cy = py as f64 / HEIGHT as f64 * 3.0 - 1.5;
        for px in 0..WIDTH {
            let cx = px as f64 / WIDTH as f64 * 3.0 - 2.0;
            let (mut zx, mut zy, mut iter) = (0.0f64, 0.0f64, 0i64);
            while zx * zx + zy * zy <= 4.0 && iter < MAXITER {
                let new_zx = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = new_zx;
                iter += 1;
            }
            checksum += iter;
        }
    }
    checksum.to_string()
}

/// `prime_numbers.mim` — honest `Vec<bool>` sieve over
/// `MAX_NUMBER_TO_CHECK = 7_000_000`; expects `476_648` primes. This is the
/// workload `ArrayStore::Bools` exists for: the mask is ~875KB bit-packed
/// here, ~7MB as bytes, ~112MB as `Vec<Val>`.
pub fn prime_numbers() -> String {
    const MAX_NUMBER_TO_CHECK: usize = 7_000_000;
    let mut prime_mask = vec![true; MAX_NUMBER_TO_CHECK + 1];
    prime_mask[0] = false;
    prime_mask[1] = false;
    let mut total_primes_found = 0i64;
    let mut i = 2usize;
    while i < MAX_NUMBER_TO_CHECK + 1 {
        if !prime_mask[i] {
            i += 1;
            continue;
        }
        total_primes_found += 1;
        let mut n = 2 * i;
        while n < MAX_NUMBER_TO_CHECK + 1 {
            prime_mask[n] = false;
            n += i;
        }
        i += 1;
    }
    total_primes_found.to_string()
}

/// `physics.mim` — `Ball { x, y, vx, vy }` structs in a `Vec`, 600 steps of
/// wall-bounce + pairwise collision on 250 balls.
pub fn physics() -> String {
    struct Ball {
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
    }
    const COUNT: usize = 250;
    const STEPS: usize = 600;
    const SIZE: f64 = 400.0;
    const TOUCH: f64 = 8.0;
    let mut balls: Vec<Ball> = Vec::new();
    for i in 0..COUNT as i64 {
        balls.push(Ball {
            x: (i % 25) as f64 * 16.0,
            y: (i % 17) as f64 * 23.0,
            vx: (i % 7) as f64 - 3.0,
            vy: (i % 5) as f64 - 2.0,
        });
    }
    for _ in 0..STEPS {
        // move each ball, reversing it whenever it reaches a wall
        for ball in &mut balls {
            ball.x += ball.vx;
            ball.y += ball.vy;
            if ball.x < 0.0 || ball.x > SIZE {
                ball.vx = -ball.vx;
            }
            if ball.y < 0.0 || ball.y > SIZE {
                ball.vy = -ball.vy;
            }
        }
        // when two balls touch, swap their velocities so they bounce apart
        for i in 0..COUNT {
            for j in i + 1..COUNT {
                let (lo, hi) = balls.split_at_mut(j);
                let (ball, other) = (&mut lo[i], &mut hi[0]);
                let dx = other.x - ball.x;
                let dy = other.y - ball.y;
                if (dx * dx + dy * dy).sqrt() < TOUCH {
                    let (vx, vy) = (ball.vx, ball.vy);
                    ball.vx = other.vx;
                    ball.vy = other.vy;
                    other.vx = vx;
                    other.vy = vy;
                }
            }
        }
    }
    let mut total = 0.0;
    for ball in &balls {
        total += ball.x + ball.y;
    }
    // `print` renders floats through `Display` — same `{}` format here
    format!("{total}")
}

/// `physics.mim`, struct-of-arrays style — `Balls { x, y, vx, vy: Vec<f64> }`
/// instead of `Vec<Ball>`. A separate row in the bench (native lane only,
/// verified against the same `physics.mim` output): the AoS-vs-SoA delta is
/// the reference for what a struct-of-arrays `Instance` layout could buy.
///
/// Operation order is identical to `physics()` — same f64 adds/compares in
/// the same sequence — so the printed total is bit-identical.
pub fn physics_soa() -> String {
    const COUNT: usize = 250;
    const STEPS: usize = 600;
    const SIZE: f64 = 400.0;
    const TOUCH: f64 = 8.0;
    let (mut x, mut y, mut vx, mut vy) = (
        Vec::with_capacity(COUNT),
        Vec::with_capacity(COUNT),
        Vec::with_capacity(COUNT),
        Vec::with_capacity(COUNT),
    );
    for i in 0..COUNT as i64 {
        x.push((i % 25) as f64 * 16.0);
        y.push((i % 17) as f64 * 23.0);
        vx.push((i % 7) as f64 - 3.0);
        vy.push((i % 5) as f64 - 2.0);
    }
    for _ in 0..STEPS {
        // move each ball, reversing it whenever it reaches a wall
        for i in 0..COUNT {
            x[i] += vx[i];
            y[i] += vy[i];
            if x[i] < 0.0 || x[i] > SIZE {
                vx[i] = -vx[i];
            }
            if y[i] < 0.0 || y[i] > SIZE {
                vy[i] = -vy[i];
            }
        }
        // when two balls touch, swap their velocities so they bounce apart
        for i in 0..COUNT {
            for j in i + 1..COUNT {
                let dx = x[j] - x[i];
                let dy = y[j] - y[i];
                if (dx * dx + dy * dy).sqrt() < TOUCH {
                    vx.swap(i, j);
                    vy.swap(i, j);
                }
            }
        }
    }
    let mut total = 0.0;
    for i in 0..COUNT {
        total += x[i] + y[i];
    }
    format!("{total}")
}
