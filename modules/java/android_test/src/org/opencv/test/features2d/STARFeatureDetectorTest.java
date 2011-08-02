package org.opencv.test.features2d;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

import java.util.ArrayList;
import java.util.List;

public class STARFeatureDetectorTest extends OpenCVTestCase {

    FeatureDetector detector;
    KeyPoint[] truth;
    int matSize;

    protected void setUp() throws Exception {
        detector = FeatureDetector.create(FeatureDetector.STAR);

        matSize = 200;

        truth = new KeyPoint[] {
                new KeyPoint(95, 80, 22, -1, 31.595734f, 0, -1),
                new KeyPoint(105, 80, 22, -1, 31.595734f, 0, -1),
                new KeyPoint(80, 95, 22, -1, 31.595734f, 0, -1),
                new KeyPoint(120, 95, 22, -1, 31.595734f, 0, -1),
                new KeyPoint(100, 100, 8, -1, -219.90825f, 0, -1),
                new KeyPoint(80, 105, 22, -1, 31.595734f, 0, -1),
                new KeyPoint(120, 105, 22, -1, 31.595734f, 0, -1),
                new KeyPoint(95, 120, 22, -1, 31.595734f, 0, -1),
                new KeyPoint(105, 120, 22, -1, 31.595734f, 0, -1) };

        super.setUp();
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

    private Mat getMaskImg() {
        Mat mask = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Mat right = mask.submat(0, matSize, matSize / 2, matSize);
        right.setTo(new Scalar(0));
        return mask;
    }

    public void testCreate() {
        assertNotNull(detector);
    }

    public void testDetectMatListOfKeyPointMat() {
        Mat img = getTestImg();
        Mat mask = getMaskImg();
        List<KeyPoint> keypoints = new ArrayList<KeyPoint>();

        detector.detect(img, keypoints, mask);

        KeyPoint[] _truth = new KeyPoint[] { truth[0], truth[2], truth[5], truth[7] };

        assertEquals(_truth.length, keypoints.size());
        for (int i = 0; i < _truth.length; i++)
            assertKeyPointEqual(_truth[i], keypoints.get(i), EPS);
    }

    public void testDetectMatListOfKeyPoint() {
        Mat img = getTestImg();
        List<KeyPoint> keypoints = new ArrayList<KeyPoint>();

        detector.detect(img, keypoints);

        assertEquals(truth.length, keypoints.size());
        for (int i = 0; i < truth.length; i++)
            assertKeyPointEqual(truth[i], keypoints.get(i), EPS);
    }

    public void testEmpty() {
        assertFalse(detector.empty());
    }

    public void testRead() {
        Mat img = getTestImg();
        
        List<KeyPoint> keypoints1 = new ArrayList<KeyPoint>();
        detector.detect(img, keypoints1);
        
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nmaxSize: 45\nresponseThreshold: 150\nlineThresholdProjected: 10\nlineThresholdBinarized: 8\nsuppressNonmaxSize: 5\n");
        detector.read(filename);
        
        List<KeyPoint> keypoints2 = new ArrayList<KeyPoint>();
        detector.detect(img, keypoints2);
        
        assertTrue(keypoints2.size() <= keypoints1.size());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        detector.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<maxSize>45</maxSize>\n<responseThreshold>30</responseThreshold>\n<lineThresholdProjected>10</lineThresholdProjected>\n<lineThresholdBinarized>8</lineThresholdBinarized>\n<suppressNonmaxSize>5</suppressNonmaxSize>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

        String truth = "%YAML:1.0\nmaxSize: 45\nresponseThreshold: 30\nlineThresholdProjected: 10\nlineThresholdBinarized: 8\nsuppressNonmaxSize: 5\n";
        assertEquals(truth, readFile(filename));
    }

}
