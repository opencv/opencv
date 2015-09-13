#!/usr/bin/env python

'''
Scans current directory for *.py files and reports
ones with missing __doc__ string.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

from glob import glob

if __name__ == '__main__':
    print('--- undocumented files:')
    for fn in glob('*.py'):
        loc = {}
        if PY3:
            exec(open(fn).read(), loc)
        else:
            execfile(fn, loc)
        if '__doc__' not in loc:
            print(fn)
