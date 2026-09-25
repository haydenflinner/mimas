fn fib_iter(n: i64) -> i64 {
    const MOD: i64 = 1000000007;
    let (mut a, mut b) = (0i64, 1i64);
    for _ in 0..n {
        let t = (a + b) % MOD;
        a = b;
        b = t;
    }
    a
}

fn main() {
    println!("{}", fib_iter(20_000_000));
}
