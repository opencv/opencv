package org.opencv.test.imgproc;

import org.opencv.test.OpenCVTestCase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class MomentsTest extends OpenCVTestCase {

    Mat data;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        data = new Mat(3,3, CvType.CV_8UC1, new Scalar(1));
        data.row(1).setTo(new Scalar(5));
    }

    public void testAll() {
        Moments res = Imgproc.moments(data);
        assertEquals(res.m00, 21.0, EPS);
        assertEquals(res.m10, 21.0, EPS);
        assertEquals(res.m01, 21.0, EPS);
        assertEquals(res.m20, 35.0, EPS);
        assertEquals(res.m11, 21.0, EPS);
        assertEquals(res.m02, 27.0, EPS);
        assertEquals(res.m30, 63.0, EPS);
        assertEquals(res.m21, 35.0, EPS);
        assertEquals(res.m12, 27.0, EPS);
        assertEquals(res.m03, 39.0, EPS);
        assertEquals(res.mu20, 14.0, EPS);
        assertEquals(res.mu11, 0.0, EPS);
        assertEquals(res.mu02, 6.0, EPS);
        assertEquals(res.mu30, 0.0, EPS);
        assertEquals(res.mu21, 0.0, EPS);
        assertEquals(res.mu12, 0.0, EPS);
        assertEquals(res.mu03, 0.0, EPS);
        assertEquals(res.nu20, 0.031746031746031744, EPS);
        assertEquals(res.nu11, 0.0, EPS);
        assertEquals(res.nu02, 0.013605442176870746, EPS);
        assertEquals(res.nu30, 0.0, EPS);
        assertEquals(res.nu21, 0.0, EPS);
        assertEquals(res.nu12, 0.0, EPS);
        assertEquals(res.nu03, 0.0, EPS);
    }

}
