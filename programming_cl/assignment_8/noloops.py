

def product(nums: list[int]) -> int:
    if nums == []:
        return 1
    el = nums[0]
    rest = nums[1:]
    rest_product = product(rest)
    return el * rest_product
    

def squares(nums: list[int]) -> list[int]:
    # Return a list of the squares of each element in nums.
    if nums == []:
        return []
    el = nums[0]
    rest = nums[1:]
    rest_squares = squares(rest)
    return [el * el] + rest_squares

# num_zeros(n) accepts a positive(!) integer n as a parameter, and returns the number of zeros in that
# number’s decimal representation recursively.
def num_zeros(n: int) -> int:
    n_abs = abs(n)

    # Base case(s)
    if n_abs == 0:
        return 1  # "0" has one zero digit
    if n_abs < 10:
        return 0  # single non-zero digit has no zero digits

    # Step 1: peel off last digit
    last_digit = n_abs % 10

    # Step 2: contribute 1 if that digit is zero, else 0
    zeros_in_last_digit = 0
    if last_digit == 0:
        zeros_in_last_digit = 1

    # Step 3: remove last digit using integer division
    remaining_digits = n_abs // 10

    # Step 4: recursive call on remaining digits and add them
    return zeros_in_last_digit + num_zeros(remaining_digits)