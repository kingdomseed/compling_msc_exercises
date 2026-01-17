#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Programming for Computational Linguistics 2020/2021
    Time and Memory Issues

    Finds 'k' smallest numbers on the list lst
"""

def find_k_smallest(lst, k):
    result = [ ]

    for i in range(k):
        lst.sort()
        result.append(lst[i])

    return result

if __name__ == "__main__":
    print(find_k_smallest(list(range(100)), 100))
    print(find_k_smallest(list(range(1000000)), 1000))
