#! /usr/bin/env python

"""
This script checks HighGUI's cvGetCaptureProperty functionality for correct return
of the frame width and height of a DivX-compressed .avi file.
"""

# name if this test and it's requirements
TESTNAME = "size_divx"
REQUIRED = []

# needed for sys.exit(int), .works file handling and check routine
import sys
import works
import size_test

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED,TESTNAME):
	sys.exit(77)

# name of file we check here
FILENAME='divx.avi'

# run check routine
result=size_test.size_ok(FILENAME)

# create flag file for following tests
works.set_file(TESTNAME)

 # return result of test routine
sys.exit(result)
