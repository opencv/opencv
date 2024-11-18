package org.opencv.test.features;

import org.junit.Assert;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features.Features;
import org.opencv.features.ORB;
import org.opencv.test.OpenCVTestCase;

public class ORBFeatureDetectorTest extends OpenCVTestCase {

    public void testCreate() {
        fail("Not yet implemented");
    }

    public void testDetectListOfMatListOfListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPointMat() {
        fail("Not yet implemented");
    }

    public void testEmpty() {
        fail("Not yet implemented");
    }

    public void testRead() {
        fail("Not yet implemented");
    }

    public void testWrite() {
        fail("Not yet implemented");
    }

    public void testDetectTwoPoints() {
        Mat img = new Mat(256,256, CvType.CV_8UC3, new Scalar(0,0,0));
        img.put(35, 40, 255,255, 255);
        img.put(152, 98, 200,0, 0);

        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        ORB orb = ORB.create();
        Mat descriptors = new Mat();
        orb.detectAndCompute(img, new Mat(), keypoints, descriptors);

        KeyPoint[] keypointsArray = keypoints.toArray();
        assertEquals(2, keypointsArray.length);

        long x1 = Math.round(keypointsArray[0].pt.x);
        long y1 = Math.round(keypointsArray[0].pt.y);
        long x2 = Math.round(keypointsArray[1].pt.x);
        long y2 = Math.round(keypointsArray[1].pt.y);

        if (x2 > x1) {
            assertEquals(40, x1);
            assertEquals(35, y1);
            assertEquals(98, x2);
            assertEquals(152, y2);
        } else {
            assertEquals(40, x2);
            assertEquals(35, y2);
            assertEquals(98, x1);
            assertEquals(152, y1);
        }
    }

}
