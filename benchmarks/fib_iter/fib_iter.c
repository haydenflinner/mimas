#include <stdio.h>
#include <stdint.h>

static int64_t fib_iter(int64_t n) {
    const int64_t MOD = 1000000007;
    int64_t a = 0, b = 1;
    for (int64_t i = 0; i < n; i++) {
        int64_t t = (a + b) % MOD;
        a = b;
        b = t;
    }
    return a;
}

int main(void) {
    printf("%lld\n", (long long)fib_iter(20000000));
    return 0;
}
