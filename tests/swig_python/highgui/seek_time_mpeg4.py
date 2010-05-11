#! /usr/bin/env python

"""
This script checks HighGUI's time seeking functionality
on an MPEG4-compressed .mp4 file.
"""

# name if this test and it's requirements
TESTNAME = "seek_time_mpeg4"
REQUIRED = []
ERRORS=[0.042,0.025,0.026,0.025,0.024,0.024,0.026,0.024,0.025,0.024,0.028,0.023,0.024,0.024,0.024,0.024,0.025,0.023,0.027,0.024,0.030,0.025,0.026,0.026,0.026,0.026,0.026,0.024,0.027]

# needed for sys.exit(int), .works file handling and check routine
import sys
import works
import seek_test

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED,TESTNAME):
	sys.exit(77)

# name of file we check here
FILENAME='mpeg4.mp4'

# run check routine
result=seek_test.seek_time_ok(FILENAME,ERRORS)

# create flag file for following tests
works.set_file(TESTNAME)

 # return result of test routine
sys.exit(result)
