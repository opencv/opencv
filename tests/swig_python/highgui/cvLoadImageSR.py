#! /usr/bin/env python
"""
This script will test highgui's image loading functionality
for .sr files
"""

# file extension to check
EXTENSION  = "sr"

# import check routine
import cvLoadImage
import sys

# check image file of extension EXTENSION,
# the function also exits and returns
# 0,1 or 77 accordingly.

if cvLoadImage.image_ok(EXTENSION):
	sys.exit(0)
else:
	sys.exit(1)

