#! /usr/bin/env python

"""
This script checks HighGUI's time seeking functionality
on a DV-compressed .avi file.
"""

# name if this test and it's requirements
TESTNAME = "seek_time_dv_pal_progressive_avi"
REQUIRED = []
ERRORS=[0.051,0.047,0.051,0.050,0.052,0.049,0.051,0.050,0.050,0.051,0.054,0.052,0.053,0.052,0.055,0.052,0.053,0.052,0.053,0.052,0.056,0.055,0.056,0.055,0.058,0.055,0.056,0.055,0.056]

# needed for sys.exit(int), .works file handling and check routine
import sys
import works
import seek_test

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED,TESTNAME):
	sys.exit(77)

# name of file we check here
FILENAME='dv_pal_progressive.avi'

# run check routine
result=seek_test.seek_time_ok(FILENAME,ERRORS)

# create flag file for following tests
works.set_file(TESTNAME)

 # return result of test routine
sys.exit(result)
