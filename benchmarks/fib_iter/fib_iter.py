N = 20000000
MOD = 1000000007

def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % MOD
    return a

print(fib_iter(N))
