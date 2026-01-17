#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Programming for Computational Linguistics 2020/2021
    Time and Memory Issues
    
    Functions for calculating the memory usage of objects
"""

import sys

# elements of lst have to be basic types
def getsizeof_list(lst):
    result = sys.getsizeof(lst)
    for el in lst:
        result += sys.getsizeof(el)
    
    return result

# elements of dct have to be basic types
def getsizeof_dict(dct):
    result = sys.getsizeof(dct)
    for k, v in dct.items():
        result += sys.getsizeof(k)
        result += sys.getsizeof(v)
    
    return result
