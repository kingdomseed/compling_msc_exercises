
def find_maxima(lst):
    """
    The function should return a new list with all the elements from lst that
    are bigger than all the elements which precede them. For example, a result
    of find_maxima([1, 3, 2, 7, 4]) should be [1, 3, 7] because 1 is first
    (so there are no preceding elements), 3 is bigger than 1 , and 7 is bigger
    than all [1, 3, 2].

    Must have complexity O(n)
    """
    if len(lst) < 3:
        return []

    local_maxima = [lst[0]]
    current_max = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > current_max:
            local_maxima.append(lst[i])
            current_max = lst[i]

    return local_maxima

# Example usage:


if __name__ == "__main__":
    print(find_maxima(range(100000)))
