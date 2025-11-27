
def rotated(lst: list, n: int):
    rotated_list = []
    # a positive n means rotate to the right
    # a negative n means rotate to the left
    # if n is larger than the length of the list,
    # rotate it as many times as needed
    if n < 0:
        rotated_list = lst.copy()
        while n < 0:
            rotated_list.insert(0, rotated_list.pop())
            n += 1
    elif n > 0:
        rotated_list = lst.copy()
        while n > 0:
            rotated_list.append(rotated_list.pop(0))
            n -= 1
    
    return rotated_list

print(rotated([1, 2, 3, 4, 5], 1))
print(rotated([1, 2, 3, 4, 5], 100))
print(rotated([1, 2, 3, 4, 5], 0))
print(rotated([1, 2, 3, 4, 5], -100))