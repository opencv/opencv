#! /usr/bin/env python
"""
This script will test highgui's video reading functionality
for RAW YUY2 .avi files
"""

# pixel format to check
FORMAT   = "YUY2"

# import check routine
import cvCreateFileCapture

# check video file of format FORMAT,
# the function also exits and returns
# 0,1 or 77 accordingly.

cvCreateFileCapture.video_ok(FORMAT)
