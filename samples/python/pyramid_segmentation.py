#!/usr/bin/python
import cv2.cv as cv

class PyrSegmentation:
    def __init__(self, img0):
        self.thresh1 = 255
        self.thresh2 = 30
        self.level =4
        self.storage = cv.CreateMemStorage()
        cv.NamedWindow("Source", 0)
        cv.ShowImage("Source", img0)
        cv.NamedWindow("Segmentation", 0)
        cv.CreateTrackbar("Thresh1", "Segmentation", self.thresh1, 255, self.set_thresh1)
        cv.CreateTrackbar("Thresh2", "Segmentation",  self.thresh2, 255, self.set_thresh2)
        self.image0 = cv.CloneImage(img0)
        self.image1 = cv.CloneImage(img0)
        cv.ShowImage("Segmentation", self.image1)

    def set_thresh1(self, val):
        self.thresh1 = val
        self.on_segment()

    def set_thresh2(self, val):
        self.thresh2 = val
        self.on_segment()

    def on_segment(self):
        comp = cv.PyrSegmentation(self.image0, self.image1, self.storage, \
                            self.level, self.thresh1+1, self.thresh2+1)
        cv.ShowImage("Segmentation", self.image1)
    
    def run(self):
        self.on_segment()
        cv.WaitKey(0)

if __name__ == "__main__":
    img0 = cv.LoadImage("../c/fruits.jpg", 1)

    # segmentation of the color image
    PyrSegmentation(img0).run()
