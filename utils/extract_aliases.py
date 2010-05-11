#! /usr/bin/env python
"""
This script extracts macros #defines from those OpenCV headers that can't be
directly parsed by current SWIG versions and must be pre-filtered by
the C preprocessor (that erases all #defines).  Type information is missing in the 
macros, so C code can't be regenerated.  Instead, try to convert C to Python code.
C macros too complicated to represent in python using regexes are listed in EXCLUDE
"""

import sys, re

EXCLUDE = { } 

# force this to be part of cv module
# otherwise becomes cv.cvmacros
print "/** This file was automatically generated using util/extract_aliases.py script */"
print "%module cv"
print "%pythoncode %{"
for fn in sys.argv[1:]:
    f = open( fn, "r" )
    in_define = 0
    for l in f.xreadlines():
        m = re.match( r"^#define\s+((?:CV_|IPL_|cv)\w+)\s+((?:CV|IPL|cv)\w*)\s*$", l )
        if m and not l.endswith( "\\\n" ) and not EXCLUDE.has_key(m.group(1)):
            print "%s=%s" % (m.group(1), m.group(2))
    f.close()

print "%}"
