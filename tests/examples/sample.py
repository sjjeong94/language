# 간단한 함수 예제
def add(a, b):
    return a + b


def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)


def main():
    x = 5
    y = 3
    result = add(x, y)

    fact = factorial(4)

    return result + fact
