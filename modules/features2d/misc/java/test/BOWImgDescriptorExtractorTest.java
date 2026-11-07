package org.opencv.test.features2d;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.KeyPoint;
import org.opencv.features2d.ORB;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.BOWImgDescriptorExtractor;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;

public class BOWImgDescriptorExtractorTest extends OpenCVTestCase {

    ORB extractor;
    DescriptorMatcher matcher;
    int matSize;

    public static void assertDescriptorsClose(Mat expected, Mat actual, int allowedDistance) {
        double distance = Core.norm(expected, actual, Core.NORM_HAMMING);
        assertTrue("expected:<" + allowedDistance + "> but was:<" + distance + ">", distance <= allowedDistance);
    }

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Imgproc.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Imgproc.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = ORB.create();
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        matSize = 100;
    }

    public void testCreate() {
        BOWImgDescriptorExtractor bow = new BOWImgDescriptorExtractor(extractor, matcher);
    }

}
