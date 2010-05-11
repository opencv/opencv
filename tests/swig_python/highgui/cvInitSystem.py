#! /usr/bin/env python

"""
This script will test highgui's InitSystem function
ATTENTION: This test doesn't do much, yet, but cvInitSystem
is called with default parameters on the first highgui function call anyway.
"""

# name of this test and it's requirements
TESTNAME = "cvInitSystem"
REQUIRED = []

# needed for sys.exit(int) and .works file handling
import sys
import works

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED, TESTNAME):
	sys.exit(77)

# import the necessary things for OpenCV
import highgui

# try to initialize the highgui system
# res = highgui.cvInitSystem(globals["0,characs)
# if res != 0:
#	sys.exit(1)

# create flag file for the following tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
