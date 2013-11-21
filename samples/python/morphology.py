#!/usr/bin/python
import sys
import urllib2
import cv2.cv as cv

src = 0
image = 0
dest = 0
element_shape = cv.CV_SHAPE_RECT

def Opening(pos):
    element = cv.CreateStructuringElementEx(pos*2+1, pos*2+1, pos, pos, element_shape)
    cv.Erode(src, image, element, 1)
    cv.Dilate(image, dest, element, 1)
    cv.ShowImage("Opening & Closing", dest)
def Closing(pos):
    element = cv.CreateStructuringElementEx(pos*2+1, pos*2+1, pos, pos, element_shape)
    cv.Dilate(src, image, element, 1)
    cv.Erode(image, dest, element, 1)
    cv.ShowImage("Opening & Closing", dest)
def Erosion(pos):
    element = cv.CreateStructuringElementEx(pos*2+1, pos*2+1, pos, pos, element_shape)
    cv.Erode(src, dest, element, 1)
    cv.ShowImage("Erosion & Dilation", dest)
def Dilation(pos):
    element = cv.CreateStructuringElementEx(pos*2+1, pos*2+1, pos, pos, element_shape)
    cv.Dilate(src, dest, element, 1)
    cv.ShowImage("Erosion & Dilation", dest)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'https://raw.github.com/Itseez/opencv/master/samples/c/fruits.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        src = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)

    image = cv.CloneImage(src)
    dest = cv.CloneImage(src)
    cv.NamedWindow("Opening & Closing", 1)
    cv.NamedWindow("Erosion & Dilation", 1)
    cv.ShowImage("Opening & Closing", src)
    cv.ShowImage("Erosion & Dilation", src)
    cv.CreateTrackbar("Open", "Opening & Closing", 0, 10, Opening)
    cv.CreateTrackbar("Close", "Opening & Closing", 0, 10, Closing)
    cv.CreateTrackbar("Dilate", "Erosion & Dilation", 0, 10, Dilation)
    cv.CreateTrackbar("Erode", "Erosion & Dilation", 0, 10, Erosion)
    cv.WaitKey(0)
    cv.DestroyWindow("Opening & Closing")
    cv.DestroyWindow("Erosion & Dilation")
