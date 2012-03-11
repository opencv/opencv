#!/usr/bin/python
import cv2.cv as cv
import sys
import urllib2

hist_size = 64
range_0 = [0, 256]
ranges = [ range_0 ]

class DemHist:

    def __init__(self, src_image):
        self.src_image = src_image
        self.dst_image = cv.CloneMat(src_image)
        self.hist_image = cv.CreateImage((320, 200), 8, 1)
        self.hist = cv.CreateHist([hist_size], cv.CV_HIST_ARRAY, ranges, 1)

        self.brightness = 0
        self.contrast = 0

        cv.NamedWindow("image", 0)
        cv.NamedWindow("histogram", 0)
        cv.CreateTrackbar("brightness", "image", 100, 200, self.update_brightness)
        cv.CreateTrackbar("contrast", "image", 100, 200, self.update_contrast)

        self.update_brightcont()

    def update_brightness(self, val):
        self.brightness = val - 100
        self.update_brightcont()

    def update_contrast(self, val):
        self.contrast = val - 100
        self.update_brightcont()

    def update_brightcont(self):
        # The algorithm is by Werner D. Streidt
        # (http://visca.com/ffactory/archives/5-99/msg00021.html)

        if self.contrast > 0:
            delta = 127. * self.contrast / 100
            a = 255. / (255. - delta * 2)
            b = a * (self.brightness - delta)
        else:
            delta = -128. * self.contrast / 100
            a = (256. - delta * 2) / 255.
            b = a * self.brightness + delta

        cv.ConvertScale(self.src_image, self.dst_image, a, b)
        cv.ShowImage("image", self.dst_image)

        cv.CalcArrHist([self.dst_image], self.hist)
        (min_value, max_value, _, _) = cv.GetMinMaxHistValue(self.hist)
        cv.Scale(self.hist.bins, self.hist.bins, float(self.hist_image.height) / max_value, 0)

        cv.Set(self.hist_image, cv.ScalarAll(255))
        bin_w = round(float(self.hist_image.width) / hist_size)

        for i in range(hist_size):
            cv.Rectangle(self.hist_image, (int(i * bin_w), self.hist_image.height),
                         (int((i + 1) * bin_w), self.hist_image.height - cv.Round(self.hist.bins[i])),
                         cv.ScalarAll(0), -1, 8, 0)
       
        cv.ShowImage("histogram", self.hist_image)

if __name__ == "__main__":
    # Load the source image.
    if len(sys.argv) > 1:
        src_image = cv.GetMat(cv.LoadImage(sys.argv[1], 0))
    else:
        url = 'http://code.opencv.org/svn/opencv/trunk/opencv/samples/c/baboon.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        src_image = cv.DecodeImageM(imagefiledata, 0)

    dh = DemHist(src_image)

    cv.WaitKey(0)
