#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Programming for Computational Linguistics 2020/2021
    Time and Memory Issues
    
    Returns a new list without duplicated elements
"""

import datetime

def remove_duplicates_1(lst):
    '''Uses a list'''
    
    result = [ ]
    for el in lst:
        if el not in result:
            result.append(el)
    return result

def remove_duplicates_2(lst):
    '''Uses a dictionary'''
    
    result = { }
    for el in lst:
        result[el] = 1
    return list(result.keys())


if __name__ == "__main__":
    start = datetime.datetime.now()
    remove_duplicates_1(range(10000))
    print("Time (list): ", ( datetime.datetime.now() - start).total_seconds ())
    
    start = datetime.datetime.now()
    remove_duplicates_2(range(10000))
    print("Time (dict): ", ( datetime.datetime.now() - start).total_seconds ())
