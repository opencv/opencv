#!/usr/bin/env python

'''
Scans current directory for *.py files and reports
ones with missing __doc__ string.
'''

# Python 2/3 compatibility
from __future__ import print_function

from glob import glob

if __name__ == '__main__':
    print('--- undocumented files:')
    for fn in glob('*.py'):
        loc = {}
        try:
            try:
                execfile(fn, loc)           # Python 2
            except NameError:
                exec(open(fn).read(), loc)  # Python 3
        except Exception:
            pass
        if '__doc__' not in loc:
            print(fn)
