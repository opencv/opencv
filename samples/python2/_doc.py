#!/usr/bin/env python

'''
Scans current directory for *.py files and reports
ones with missing __doc__ string.
'''

from glob import glob

if __name__ == '__main__':
    print '--- undocumented files:'
    for fn in glob('*.py'):
        loc = {}
        execfile(fn, loc)
        if '__doc__' not in loc:
            print fn
