TARGET = 30
REPEAT = 20

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

result = 0
for _ in range(REPEAT):
    result = fib(TARGET)
print(result)
