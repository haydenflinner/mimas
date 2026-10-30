#include <stdio.h>
#include <stdint.h>

static int64_t fib(int64_t n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    const int64_t TARGET = 30, REPEAT = 20;
    int64_t result = 0;
    for (int64_t i = 0; i < REPEAT; i++) result = fib(TARGET);
    printf("%lld\n", (long long)result);
    return 0;
}
