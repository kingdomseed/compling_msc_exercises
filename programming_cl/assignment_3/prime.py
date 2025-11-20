
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def is_twin_prime(n):
    if not is_prime(n):
        return False
    return is_prime(n - 2) or is_prime(n + 2)
