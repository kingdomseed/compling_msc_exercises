#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Programming for Computational Linguistics 2020/2021
    Time and Memory Issues
    
    Reads all bigrams from the given file
"""

import gzip
import time
import sys

def read_bigrams(file):
    result = [ ]
    
    for nr, line in enumerate(file):
        words = line.split()
        
        for i in range(len(words) - 1):
            first = words[i]
            second = words[i+1]
            
            result.append({"first" : first,
                           "second" : second})

        # slows down the process to be able to observe it in htop
        if nr % 500 == 0:
            time.sleep(0.01)

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Give the file to read as a parameter")
        sys.exit()

    filename = sys.argv[1]
    if filename.endswith("gz"):
        f = gzip.open(filename, "rb")
    else:
        f = open(filename, "r")
    
    bigrams = read_bigrams(f)
    print("All bigrams:", len(bigrams))
