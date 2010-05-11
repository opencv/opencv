import cv
import urllib2
from sys import argv

def load_sample(name=None):
    if len(argv) > 1:
        img0 = cv.LoadImage(argv[1], cv.CV_LOAD_IMAGE_COLOR)
    elif name is not None:
        try:
            img0 = cv.LoadImage(name, cv.CV_LOAD_IMAGE_COLOR)
        except IOError:
            urlbase = 'https://code.ros.org/svn/opencv/trunk/opencv/samples/c/'
            file = name.split('/')[-1]
            filedata = urllib2.urlopen(urlbase+file).read()
            imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
            cv.SetData(imagefiledata, filedata, len(filedata))
            img0 = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)
    return img0
