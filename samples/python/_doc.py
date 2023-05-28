#!/usr/bin/env python

'''
Scans current directory for *.py files and reports
ones with missing __doc__ string.
'''

# Python 2/3 compatibility
from __future__ import print_function

from glob import glob

def main():
    '''
    The main function that performs the operation of checking for python files
    without docstrings in the current directory.
    '''

    print('--- undocumented files:')
    for fn in glob('*.py'):
        loc = {}
        try:
            # Compatibility for Python 2 and 3
            try:
                with open(fn) as file:
                    exec(file.read(), loc)  # Python 2/3
            except Exception:
                pass
        except Exception:
            pass
        if '__doc__' not in loc:
            print(fn)


if __name__ == '__main__':
    main()
