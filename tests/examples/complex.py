# 더 복잡한 예제: 피보나치 수열과 조건문
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a = 0
        b = 1
        i = 2
        while i <= n:
            temp = a + b
            a = b
            b = temp
            i = i + 1
        return b


def is_even(num):
    return num % 2 == 0


def calculate_sum(limit):
    total = 0
    i = 1
    while i <= limit:
        if is_even(i):
            total = total + i
        i = i + 1
    return total
