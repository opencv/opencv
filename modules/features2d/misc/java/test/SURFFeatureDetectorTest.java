package org.opencv.test.features2d;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.Feature2D;

public class SURFFeatureDetectorTest extends OpenCVTestCase {

    Feature2D detector;
    int matSize;
    KeyPoint[] truth;

    private Mat getMaskImg() {
        Mat mask = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Mat right = mask.submat(0, matSize, matSize / 2, matSize);
        right.setTo(new Scalar(0));
        return mask;
    }

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Imgproc.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Imgproc.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    private void order(List<KeyPoint> points) {
        Collections.sort(points, new Comparator<KeyPoint>() {
            public int compare(KeyPoint p1, KeyPoint p2) {
                if (p1.angle < p2.angle)
                    return -1;
                if (p1.angle > p2.angle)
                    return 1;
                return 0;
            }
        });
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        detector = createClassInstance(XFEATURES2D+"SURF", DEFAULT_FACTORY, null, null);
        matSize = 100;
        truth = new KeyPoint[] {
                new KeyPoint(55.775578f, 55.775578f, 16, 80.245735f, 8617.8633f, 0, -1),
                new KeyPoint(44.224422f, 55.775578f, 16, 170.24574f, 8617.8633f, 0, -1),
                new KeyPoint(44.224422f, 44.224422f, 16, 260.24573f, 8617.8633f, 0, -1),
                new KeyPoint(55.775578f, 44.224422f, 16, 350.24573f, 8617.8633f, 0, -1)
            };
    }

    public void testCreate() {
        assertNotNull(detector);
    }

    public void testDetectListOfMatListOfListOfKeyPoint() {

        setProperty(detector, "hessianThreshold", "double", 8000);
        setProperty(detector, "nOctaves", "int", 3);
        setProperty(detector, "nOctaveLayers", "int", 4);
        setProperty(detector, "upright", "boolean", false);

        List<MatOfKeyPoint> keypoints = new ArrayList<MatOfKeyPoint>();
        Mat cross = getTestImg();
        List<Mat> crosses = new ArrayList<Mat>(3);
        crosses.add(cross);
        crosses.add(cross);
        crosses.add(cross);

        detector.detect(crosses, keypoints);

        assertEquals(3, keypoints.size());

        for (MatOfKeyPoint mkp : keypoints) {
            List<KeyPoint> lkp = mkp.toList();
            order(lkp);
            assertListKeyPointEquals(Arrays.asList(truth), lkp, EPS);
        }
    }

    public void testDetectListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPoint() {

        setProperty(detector, "hessianThreshold", "double", 8000);
        setProperty(detector, "nOctaves", "int", 3);
        setProperty(detector, "nOctaveLayers", "int", 4);
        setProperty(detector, "upright", "boolean", false);

        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat cross = getTestImg();

        detector.detect(cross, keypoints);

        List<KeyPoint> lkp = keypoints.toList();
        order(lkp);
        assertListKeyPointEquals(Arrays.asList(truth), lkp, EPS);
    }

    public void testDetectMatListOfKeyPointMat() {

        setProperty(detector, "hessianThreshold", "double", 8000);
        setProperty(detector, "nOctaves", "int", 3);
        setProperty(detector, "nOctaveLayers", "int", 4);
        setProperty(detector, "upright", "boolean", false);

        Mat img = getTestImg();
        Mat mask = getMaskImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();

        detector.detect(img, keypoints, mask);

        List<KeyPoint> lkp = keypoints.toList();
        order(lkp);
        assertListKeyPointEquals(Arrays.asList(truth[1], truth[2]), lkp, EPS);
    }

    public void testEmpty() {
//        assertFalse(detector.empty());
        fail("Not yet implemented");
    }

    public void testRead() {
        Mat cross = getTestImg();

        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        detector.detect(cross, keypoints1);

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n---\nhessianThreshold: 8000.\noctaves: 3\noctaveLayers: 4\nupright: 0\n");
        detector.read(filename);

        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        detector.detect(cross, keypoints2);

        assertTrue(keypoints2.total() <= keypoints1.total());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        detector.write(filename);

//        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.SURF</name>\n<extended>0</extended>\n<hessianThreshold>100.</hessianThreshold>\n<nOctaveLayers>3</nOctaveLayers>\n<nOctaves>4</nOctaves>\n<upright>0</upright>\n</opencv_storage>\n";
        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

//        String truth = "%YAML:1.0\n---\nname: \"Feature2D.SURF\"\nextended: 0\nhessianThreshold: 100.\nnOctaveLayers: 3\nnOctaves: 4\nupright: 0\n";
        String truth = "%YAML:1.0\n---\n";
        assertEquals(truth, readFile(filename));
    }

}
