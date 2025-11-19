import math
# Returns the last element of list l
def last(l: list):
    return l[-1]

# Returns the middle element of list l with an odd number of elements
def middle(l: list):
    return l[len(l) // 2]

# Returns the product of a list of numbers
def product(l: list[int]):
    return math.prod(l)

# Returns the mean (average) of a list of numbers
def mean(l: list[int]):
    return sum(l) / len(l)

# returns the sum of the elements of a list with even indices
def even_sum(l: list[int]):
    return sum(l[::2])
