package org.opencv.test.features2d;

import java.util.Arrays;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.features2d.SimpleBlobDetector_Params;

public class SIMPLEBLOBFeatureDetectorTest extends OpenCVTestCase {

    SimpleBlobDetector detector;
    int matSize;
    KeyPoint[] truth;

    private Mat getMaskImg() {
        Mat mask = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Mat right = mask.submat(0, matSize, matSize / 2, matSize);
        right.setTo(new Scalar(0));
        return mask;
    }

    private Mat getTestImg() {

        int center = matSize / 2;
        int offset = 40;

        Mat img = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Imgproc.circle(img, new Point(center - offset, center), 24, new Scalar(0), -1);
        Imgproc.circle(img, new Point(center + offset, center), 20, new Scalar(50), -1);
        Imgproc.circle(img, new Point(center, center - offset), 18, new Scalar(100), -1);
        Imgproc.circle(img, new Point(center, center + offset), 14, new Scalar(150), -1);
        Imgproc.circle(img, new Point(center, center), 10, new Scalar(200), -1);
        return img;
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        detector = SimpleBlobDetector.create();
        matSize = 200;
        truth = new KeyPoint[] {
                new KeyPoint(140, 100, 41.036568f, -1, 0, 0, -1),
                new KeyPoint(60, 100, 48.538486f, -1, 0, 0, -1),
                new KeyPoint(100, 60, 36.769554f, -1, 0, 0, -1),
                new KeyPoint(100, 140, 28.635643f, -1, 0, 0, -1),
                new KeyPoint(100, 100, 20.880613f, -1, 0, 0, -1)
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

        assertListKeyPointEquals(Arrays.asList(truth[1]), keypoints.toList(), EPS);
    }

    public void testEmpty() {
//        assertFalse(detector.empty());
        fail("Not yet implemented");
    }

    public void testReadYml() {
        Mat img = getTestImg();

        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        detector.detect(img, keypoints1);

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nthresholdStep: 10.0\nminThreshold: 50\nmaxThreshold: 220\nminRepeatability: 2\nminDistBetweenBlobs: 10.\nfilterByColor: 1\nblobColor: 0\nfilterByArea: 1\nminArea: 800\nmaxArea: 6000\nfilterByCircularity: 0\nminCircularity: 0.7\nmaxCircularity: 10.\nfilterByInertia: 1\nminInertiaRatio: 0.2\nmaxInertiaRatio: 11.\nfilterByConvexity: true\nminConvexity: 0.9\nmaxConvexity: 12.\n");
        detector.read(filename);

        SimpleBlobDetector_Params params = detector.getParams();
        assertEquals(10.0f, params.get_thresholdStep());
        assertEquals(50f, params.get_minThreshold());
        assertEquals(220f, params.get_maxThreshold());
        assertEquals(2, params.get_minRepeatability());
        assertEquals(10.0f, params.get_minDistBetweenBlobs());
        assertEquals(true, params.get_filterByColor());
        // FIXME: blobColor field has uchar type in C++ and cannot be automatically wrapped to Java as it does not support unsigned types
        //assertEquals(0, params.get_blobColor());
        assertEquals(true, params.get_filterByArea());
        assertEquals(800f, params.get_minArea());
        assertEquals(6000f, params.get_maxArea());
        assertEquals(false, params.get_filterByCircularity());
        assertEquals(0.7f, params.get_minCircularity());
        assertEquals(10.0f, params.get_maxCircularity());
        assertEquals(true, params.get_filterByInertia());
        assertEquals(0.2f, params.get_minInertiaRatio());
        assertEquals(11.0f, params.get_maxInertiaRatio());
        assertEquals(true, params.get_filterByConvexity());
        assertEquals(0.9f, params.get_minConvexity());
        assertEquals(12.0f, params.get_maxConvexity());

        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        detector.detect(img, keypoints2);

        assertTrue(keypoints2.total() <= keypoints1.total());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        detector.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<format>3</format>\n<thresholdStep>10.</thresholdStep>\n<minThreshold>50.</minThreshold>\n<maxThreshold>220.</maxThreshold>\n<minRepeatability>2</minRepeatability>\n<minDistBetweenBlobs>10.</minDistBetweenBlobs>\n<filterByColor>1</filterByColor>\n<blobColor>0</blobColor>\n<filterByArea>1</filterByArea>\n<minArea>25.</minArea>\n<maxArea>5000.</maxArea>\n<filterByCircularity>0</filterByCircularity>\n<minCircularity>8.0000001192092896e-01</minCircularity>\n<maxCircularity>3.4028234663852886e+38</maxCircularity>\n<filterByInertia>1</filterByInertia>\n<minInertiaRatio>1.0000000149011612e-01</minInertiaRatio>\n<maxInertiaRatio>3.4028234663852886e+38</maxInertiaRatio>\n<filterByConvexity>1</filterByConvexity>\n<minConvexity>9.4999998807907104e-01</minConvexity>\n<maxConvexity>3.4028234663852886e+38</maxConvexity>\n<collectContours>0</collectContours>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }
}
