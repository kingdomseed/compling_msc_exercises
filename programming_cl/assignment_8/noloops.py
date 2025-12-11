

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
        return [0]
    el = nums[0]
    rest = nums[1:]
    rest_squares = squares(rest)
    return [el * el] + rest_squares

def num_zeroes(n: int) -> int:
    # Return the number of zero digits in the absolute value of n.
    return 0

# product tests
print(product([2, 3, 4]))  # 24
print(product([5, 5, 5, 5]))  # 625
print(product([]))  # 1