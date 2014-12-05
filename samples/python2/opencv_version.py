#!/usr/bin/env python

import cv2

if __name__ == '__main__':
    import sys
    try:
        param = sys.argv[1]
    except:
        param = ""

    if ("--build" == param):
        print cv2.getBuildInformation()
    elif ("--help" == param):
        print "\t--build\n\t\tprint complete build info"
        print "\t--help\n\t\tprint this help"
    else:
        print "Welcome to OpenCV"
