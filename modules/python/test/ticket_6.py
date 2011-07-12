import urllib
import cv2.cv as cv
import Image
import unittest

class TestLoadImage(unittest.TestCase):
    def setUp(self):
        open("large.jpg", "w").write(urllib.urlopen("http://www.cs.ubc.ca/labs/lci/curious_george/img/ROS_bug_imgs/IMG_3560.jpg").read())

    def test_load(self):
        pilim = Image.open("large.jpg")
        cvim = cv.LoadImage("large.jpg")
        self.assert_(len(pilim.tostring()) == len(cvim.tostring()))

if __name__ == '__main__':
    unittest.main()
