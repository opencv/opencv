package org.opencv.test.features2d;

import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.FeatureDetector;
import org.opencv.core.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class STARFeatureDetectorTest extends OpenCVTestCase {

    FeatureDetector detector;
    int matSize;
    KeyPoint[] truth;

    private Mat getMaskImg() {
        Mat mask = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Mat right = mask.submat(0, matSize, matSize / 2, matSize);
        right.setTo(new Scalar(0));
        return mask;
    }

    private Mat getTestImg() {
        Scalar color = new Scalar(0);
        int center = matSize / 2;
        int radius = 6;
        int offset = 40;

        Mat img = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.circle(img, new Point(center - offset, center), radius, color, -1);
        Core.circle(img, new Point(center + offset, center), radius, color, -1);
        Core.circle(img, new Point(center, center - offset), radius, color, -1);
        Core.circle(img, new Point(center, center + offset), radius, color, -1);
        Core.circle(img, new Point(center, center), radius, color, -1);
        return img;
    }

    protected void setUp() throws Exception {
        super.setUp();
        detector = FeatureDetector.create(FeatureDetector.STAR);
        matSize = 200;
        truth = new KeyPoint[] {
                new KeyPoint( 95,  80, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(105,  80, 22, -1, 31.5957f, 0, -1),
                new KeyPoint( 80,  95, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(120,  95, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(100, 100,  8, -1, 30.f,     0, -1),
                new KeyPoint( 80, 105, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(120, 105, 22, -1, 31.5957f, 0, -1),
                new KeyPoint( 95, 120, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(105, 120, 22, -1, 31.5957f, 0, -1)
            };
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
    }

    public void testDetectMatListOfKeyPointMat() {
        Mat img = getTestImg();
        Mat mask = getMaskImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();

        detector.detect(img, keypoints, mask);

        assertListKeyPointEquals(Arrays.asList(truth[0], truth[2], truth[5], truth[7]), keypoints.toList(), EPS);
    }

    public void testEmpty() {
        assertFalse(detector.empty());
    }

    public void testRead() {
        Mat img = getTestImg();

        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        detector.detect(img, keypoints1);

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nmaxSize: 45\nresponseThreshold: 150\nlineThresholdProjected: 10\nlineThresholdBinarized: 8\nsuppressNonmaxSize: 5\n");
        detector.read(filename);

        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        detector.detect(img, keypoints2);

        assertTrue(keypoints2.total() <= keypoints1.total());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        detector.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.STAR</name>\n<lineThresholdBinarized>8</lineThresholdBinarized>\n<lineThresholdProjected>10</lineThresholdProjected>\n<maxSize>45</maxSize>\n<responseThreshold>30</responseThreshold>\n<suppressNonmaxSize>5</suppressNonmaxSize>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

        String truth = "%YAML:1.0\nname: \"Feature2D.STAR\"\nlineThresholdBinarized: 8\nlineThresholdProjected: 10\nmaxSize: 45\nresponseThreshold: 30\nsuppressNonmaxSize: 5\n";
        assertEquals(truth, readFile(filename));
    }

}
