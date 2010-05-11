#! /usr/bin/env python
"""
This script extracts #defines from those OpenCV headers that can't be
directly parsed by current SWIG versions and must be pre-filtered by
the C preprocessor (that erases all #defines).
"""

import sys, re

for fn in sys.argv[1:]:
    f = open( fn, "r" )
    in_define = 0
    for l in f.xreadlines():
        if re.match( r"^#define\s+(CV_|IPL_|cv)\w+\s+", l ):
            in_define = 1
        if re.match (r"^#define\s+CV_MAKETYPE", l):
            in_define = 1
        if re.match (r"^#define\s+CV_CN", l):
            in_define = 1
        if re.match (r"^#define\s+CV_MAT_TYPE", l):
            in_define = 1
        if re.match (r"^#define\s+CV_MAT_DEPTH", l):
            in_define = 1
        if in_define:
            print l[:l.find ('/*')]
            if not l.endswith( "\\\n" ):
                in_define = 0
                print
    f.close()

