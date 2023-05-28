#!/usr/bin/env python

'''
Utility for measuring python opencv API coverage by samples.
'''

# Python 2/3 compatibility
from __future__ import print_function

from glob import glob
import cv2 as cv
import re

def main():
    '''
    The main function that performs the operation of checking the cv API coverage.
    '''

    # Create a set of all callable functions in cv
    cv_callable = set(['cv.' + name for name in dir(cv) if callable(getattr(cv, name))])

    found = set()
    # Iterate over all python files in the current directory
    for fn in glob('*.py'):
        print(f' --- {fn}')
        with open(fn, 'r') as file:
            code = file.read()
        found |= set(re.findall('cv\.\w+', code))

    # Check for used and unused cv functions
    cv_used = found & cv_callable
    cv_unused = cv_callable - cv_used
    
    # Write the unused functions to a file
    with open('unused_api.txt', 'w') as f:
        f.write('\n'.join(sorted(cv_unused)))

    r = len(cv_used) / len(cv_callable)
    print(f'\ncv api coverage: {len(cv_used)} / {len(cv_callable)}  ({r*100:.1f}%)')


if __name__ == '__main__':
    main()
