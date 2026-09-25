fn fib(n: i64) -> i64 {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() {
    const TARGET: i64 = 30;
    const REPEAT: i64 = 20;
    let mut result = 0;
    for _ in 0..REPEAT {
        result = fib(TARGET);
    }
    println!("{}", result);
}
