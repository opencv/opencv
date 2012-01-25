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

class Creating(unittest.TestCase):
    size=(640, 480)
    repeat=100
    def test_0_Create(self):
        image = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, 1)
        cnt=cv.CountNonZero(image)
        self.assertEqual(cnt, 0, msg="Created image is not black. CountNonZero=%i" % cnt)

    def test_2_CreateRepeat(self):
        cnt=0
        for i in range(self.repeat):
            image = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, 1)
            cnt+=cv.CountNonZero(image)
        self.assertEqual(cnt, 0, msg="Created images are not black. Mean CountNonZero=%.3f" % (1.*cnt/self.repeat))

    def test_2a_MemCreated(self):
        cnt=0
        v=[]
        for i in range(self.repeat):
            image = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, 1)
            cv.FillPoly(image, [[(0, 0), (0, 100), (100, 0)]], 0)
            cnt+=cv.CountNonZero(image)
            v.append(image)
        self.assertEqual(cnt, 0, msg="Memorized images are not black. Mean CountNonZero=%.3f" % (1.*cnt/self.repeat))

    def test_3_tostirng(self):
        image = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, 1)
        image.tostring()
        cnt=cv.CountNonZero(image)
        self.assertEqual(cnt, 0, msg="After tostring(): CountNonZero=%i" % cnt)

    def test_40_tostringRepeat(self):
        cnt=0
        image = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, 1)
        cv.Set(image, cv.Scalar(0,0,0,0))
        for i in range(self.repeat*100):
            image.tostring()
        cnt=cv.CountNonZero(image)
        self.assertEqual(cnt, 0, msg="Repeating tostring(): Mean CountNonZero=%.3f" % (1.*cnt/self.repeat))

    def test_41_CreateToStringRepeat(self):
        cnt=0
        for i in range(self.repeat*100):
            image = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, 1)
            cv.Set(image, cv.Scalar(0,0,0,0))
            image.tostring()
            cnt+=cv.CountNonZero(image)
        self.assertEqual(cnt, 0, msg="Repeating create and tostring(): Mean CountNonZero=%.3f" % (1.*cnt/self.repeat))

    def test_4a_MemCreatedToString(self):
        cnt=0
        v=[]
        for i in range(self.repeat):
            image = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, 1)
            cv.Set(image, cv.Scalar(0,0,0,0))
            image.tostring()
            cnt+=cv.CountNonZero(image)
            v.append(image)
        self.assertEqual(cnt, 0, msg="Repeating and memorizing after tostring(): Mean CountNonZero=%.3f" % (1.*cnt/self.repeat))

if __name__ == '__main__':
    unittest.main()
