package org.opencv.test.features2d;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.KeyPoint;
import org.opencv.features2d.StarDetector;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

import java.util.LinkedList;
import java.util.List;

public class StarDetectorTest extends OpenCVTestCase {

    public void test_1() {
        super.test_1("FEATURES2D.StarDetector");
    }
    
    private Mat getStarImg()
    {
        Scalar color = new Scalar(0);
        int center = 100;
        int radius = 5;
        int offset = 40;
        
        Mat img = new Mat(200, 200, CvType.CV_8U, new Scalar(255));
        Core.circle(img, new Point(center - offset, center), radius, color, -1);
        Core.circle(img, new Point(center + offset, center), radius, color, -1);
        Core.circle(img, new Point(center, center - offset), radius, color, -1);
        Core.circle(img, new Point(center, center + offset), radius, color, -1);
        Core.circle(img, new Point(center, center), radius, color, -1);
        return img;
    }

    public void testDetect() {
        Mat img = getStarImg();
        List<KeyPoint> keypoints = new LinkedList<KeyPoint>();
        StarDetector star = new StarDetector();
        
        star.detect(img, keypoints);
        
        KeyPoint truth = new KeyPoint(100, 100, 8, -1,-223.40334f, 0, -1);
        assertEquals(1, keypoints.size());
        assertKeyPointEqual(truth, keypoints.get(0), EPS);
    }

    public void testStarDetector() {
        StarDetector star = new StarDetector();
        assertNotNull(star);
    }

    public void testStarDetectorIntIntIntIntInt() {
        StarDetector star = new StarDetector(45, 30, 10, 8, 5);
        assertNotNull(star);
    }

}
