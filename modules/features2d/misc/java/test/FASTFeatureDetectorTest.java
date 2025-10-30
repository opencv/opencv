package org.opencv.test.features2d;

import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.core.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;

public class FASTFeatureDetectorTest extends OpenCVTestCase {

    FastFeatureDetector detector;
    KeyPoint[] truth;

    private Mat getMaskImg() {
        Mat mask = new Mat(100, 100, CvType.CV_8U, new Scalar(255));
        Mat right = mask.submat(0, 100, 50, 100);
        right.setTo(new Scalar(0));
        return mask;
    }

    private Mat getTestImg() {
        Mat img = new Mat(100, 100, CvType.CV_8U, new Scalar(255));
        Imgproc.line(img, new Point(30, 30), new Point(70, 70), new Scalar(0), 8);
        return img;
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        detector = FastFeatureDetector.create();
        truth = new KeyPoint[] { new KeyPoint(32, 27, 7, -1, 254, 0, -1), new KeyPoint(27, 32, 7, -1, 254, 0, -1), new KeyPoint(73, 68, 7, -1, 254, 0, -1),
                new KeyPoint(68, 73, 7, -1, 254, 0, -1) };
    }

    public void testCreate() {
        assertNotNull(detector);
    }

    public void testDetectListOfMatListOfListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPoint() {
        Mat img = getTestImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();

        detector.detect(img, keypoints);

        assertListKeyPointEquals(Arrays.asList(truth), keypoints.toList(), EPS);

        // OpenCVTestRunner.Log("points found: " + keypoints.size());
        // for (KeyPoint kp : keypoints)
        // OpenCVTestRunner.Log(kp.toString());
    }

    public void testDetectMatListOfKeyPointMat() {
        Mat img = getTestImg();
        Mat mask = getMaskImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();

        detector.detect(img, keypoints, mask);

        assertListKeyPointEquals(Arrays.asList(truth[0], truth[1]), keypoints.toList(), EPS);
    }

    public void testEmpty() {
//        assertFalse(detector.empty());
        fail("Not yet implemented"); // FAST does not override empty() method
    }

    public void testRead() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        writeFile(filename, "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.FastFeatureDetector</name>\n<threshold>10</threshold>\n<nonmaxSuppression>1</nonmaxSuppression>\n<type>2</type>\n</opencv_storage>\n");
        detector.read(filename);

        assertEquals(10, detector.getThreshold());
        assertEquals(true, detector.getNonmaxSuppression());
        assertEquals(2, detector.getType());

        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();

        detector.detect(grayChess, keypoints1);

        writeFile(filename, "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.FastFeatureDetector</name>\n<threshold>150</threshold>\n<nonmaxSuppression>1</nonmaxSuppression>\n<type>2</type>\n</opencv_storage>\n");
        detector.read(filename);

        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();

        detector.detect(grayChess, keypoints2);

        assertTrue(keypoints2.total() <= keypoints1.total());
    }

    public void testReadYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        writeFile(filename, "%YAML:1.0\n---\nthreshold: 130\nnonmaxSuppression: 1\ntype: 2\n");
        detector.read(filename);

        assertEquals(130, detector.getThreshold());
        assertEquals(true, detector.getNonmaxSuppression());
        assertEquals(2, detector.getType());

        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();

        detector.detect(grayChess, keypoints1);

        writeFile(filename, "%YAML:1.0\n---\nthreshold: 150\nnonmaxSuppression: 1\ntype: 2\n");
        detector.read(filename);

        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();

        detector.detect(grayChess, keypoints2);

        assertTrue(keypoints2.total() <= keypoints1.total());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.FastFeatureDetector\"\nthreshold: 10\nnonmaxSuppression: 1\ntype: 2\n";
        String data = readFile(filename);

        assertEquals(truth, data);
    }
}
