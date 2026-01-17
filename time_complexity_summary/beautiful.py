#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Programming for Computational Linguistics 2020/2021
    Time and Memory Issues
    
    Reverses the list, removes even numbers, and then print a beautiful version of it.
"""

import datetime

def reverse_list(lst):
    result = [ ]
    for el in lst:
        result.insert(0, el)
    return result

def reverse_list_fixed(lst):
    result = [ ]
    for i in range(1, len(lst) + 1):
        result.append(lst[-i])
    return result

def filter_even_elements(lst):
    result = [ ]
    for el in lst:
        if el % 2 == 1:
            result.append(el)
    return result

def nice_str(lst):
    result = "<"
    for el in lst[:-1]:
        result += str(el) + ","
        
    if lst != [ ]:
        result += str(lst[-1])
        
    result += ">"
    return result

@profile
def print_beautiful(lst):
    rev = reverse_list(lst)
    no_even = filter_even_elements(rev)
    beautiful = nice_str(no_even)
    
    print(beautiful)

def print_beautiful_with_times(lst):
    start = datetime.datetime.now()
    rev = reverse_list(lst)
    print("Rev time:", 
          (datetime.datetime.now() - start).total_seconds())
    
    start = datetime.datetime.now() 
    no_even = filter_even_elements(rev)
    print("Even time:", 
          (datetime.datetime.now() - start).total_seconds())
    
    start = datetime.datetime.now()
    beautiful = nice_str(no_even)
    print("Str time:", 
          (datetime.datetime.now() - start).total_seconds())
    
    # commented out to see the times
    #print(beautiful)

if __name__ == "__main__":    
    lst = range(100000)
    print_beautiful(lst)
   
