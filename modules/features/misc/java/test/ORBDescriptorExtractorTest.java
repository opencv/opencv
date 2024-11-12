package org.opencv.test.features;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.KeyPoint;
import org.opencv.features.ORB;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;

public class ORBDescriptorExtractorTest extends OpenCVTestCase {

    ORB extractor;
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
        matSize = 100;
    }

    public void testComputeListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testComputeMatListOfKeyPointMat() {
        KeyPoint point = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        MatOfKeyPoint keypoints = new MatOfKeyPoint(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 32, CvType.CV_8UC1) {
            {
                put(0, 0,
                        6, 74, 6, 129, 2, 130, 56, 0, 44, 132, 66, 165, 172, 6, 3, 72, 102, 61, 171, 214, 0, 144, 65, 232, 4, 32, 138, 131, 4, 21, 37, 217);
            }
        };
        assertDescriptorsClose(truth, descriptors, 1);
    }

    public void testCreate() {
        assertNotNull(extractor);
    }

    public void testDescriptorSize() {
        assertEquals(32, extractor.descriptorSize());
    }

    public void testDescriptorType() {
        assertEquals(CvType.CV_8U, extractor.descriptorType());
    }

    public void testEmpty() {
//        assertFalse(extractor.empty());
        fail("Not yet implemented"); // ORB does not override empty() method
    }

    public void testReadYml() {
        KeyPoint point = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        MatOfKeyPoint keypoints = new MatOfKeyPoint(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n---\nnfeatures: 500\nscaleFactor: 1.1\nnlevels: 3\nedgeThreshold: 31\nfirstLevel: 0\nwta_k: 2\nscoreType: 0\npatchSize: 31\nfastThreshold: 20\n");
        extractor.read(filename);

        assertEquals(500, extractor.getMaxFeatures());
        assertEquals(1.1, extractor.getScaleFactor());
        assertEquals(3, extractor.getNLevels());
        assertEquals(31, extractor.getEdgeThreshold());
        assertEquals(0, extractor.getFirstLevel());
        assertEquals(2, extractor.getWTA_K());
        assertEquals(0, extractor.getScoreType());
        assertEquals(31, extractor.getPatchSize());
        assertEquals(20, extractor.getFastThreshold());

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 32, CvType.CV_8UC1) {
            {
                put(0, 0,
                        6, 10, 22, 5, 2, 130, 56, 0, 44, 164, 66, 165, 140, 6, 1, 72, 38, 61, 163, 210, 0, 208, 1, 104, 4, 32, 74, 131, 0, 37, 37, 67);
            }
        };
        assertDescriptorsClose(truth, descriptors, 1);
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.ORB\"\nnfeatures: 500\nscaleFactor: 1.2000000476837158\nnlevels: 8\nedgeThreshold: 31\nfirstLevel: 0\nwta_k: 2\nscoreType: 0\npatchSize: 31\nfastThreshold: 20\n";
//        String truth = "%YAML:1.0\n---\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e\\+000", "e+00"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
