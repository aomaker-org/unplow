# Path: unplow/tests/data/fib_solution.py
# Reference fibonacci implementation for replow evaluation
# Target: 80 cols, max 120 chars


def fib(n: int) -> int:
    """Compute n-th Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# end of file: unplow/tests/data/fib_solution.py
