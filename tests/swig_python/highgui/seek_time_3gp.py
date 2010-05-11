#! /usr/bin/env python

"""
This script checks HighGUI's time seeking functionality
on a 3GP-compressed .3gp file.
"""

# name if this test and it's requirements
TESTNAME = "seek_time_3gp"
REQUIRED = []
ERRORS=[0.043,0.031,0.032,0.031,0.029,0.030,0.030,0.031,0.030,0.029,0.034,0.027,0.029,0.029,0.029,0.029,0.029,0.028,0.031,0.030,0.035,0.031,0.031,0.032,0.031,0.032,0.033,0.031,0.033]

# needed for sys.exit(int), .works file handling and check routine
import sys
import works
import seek_test

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED,TESTNAME):
	sys.exit(77)

# name of file we check here
FILENAME='3gp.3gp'

# run check routine
result=seek_test.seek_time_ok(FILENAME,ERRORS)

# create flag file for following tests
works.set_file(TESTNAME)

 # return result of test routine
sys.exit(result)
