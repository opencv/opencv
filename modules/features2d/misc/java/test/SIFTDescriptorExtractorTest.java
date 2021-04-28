package org.opencv.test.features2d;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.KeyPoint;
import org.opencv.features2d.SIFT;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.Feature2D;

public class SIFTDescriptorExtractorTest extends OpenCVTestCase {

    Feature2D extractor;
    KeyPoint keypoint;
    int matSize;
    Mat truth;

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Imgproc.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Imgproc.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = SIFT.create();
        keypoint = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        matSize = 100;
        truth = new Mat(1, 128, CvType.CV_32FC1) {
            {
                put(0, 0,
                0, 0, 0, 1, 2, 0, 0, 0, 14, 22, 21, 21, 23,
                1, 0, 0, 7, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 25, 15, 11, 2, 0, 0, 117, 87, 73,
                71, 117, 40, 5, 5, 79, 63, 117, 25, 10, 2, 27, 19,
                10, 13, 20, 2, 0, 0, 5, 8, 0, 0, 75, 58, 35, 30, 94,
                15, 91, 51, 117, 96, 117, 110, 117, 75, 117, 59, 117,
                27, 30, 24, 117, 117, 16, 12, 15, 1, 0, 0, 24, 27, 0,
                0, 0, 0, 1, 5, 16, 2, 47, 8, 0, 0, 82, 57, 33, 16, 84,
                13, 0, 0, 28, 24, 18, 23, 0, 0, 0, 0, 0, 0, 0, 0);
            }
        };
    }

    public void testComputeListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testComputeMatListOfKeyPointMat() {
        MatOfKeyPoint keypoints = new MatOfKeyPoint(keypoint);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        extractor.compute(img, keypoints, descriptors);

        assertMatEqual(truth, descriptors, EPS);
    }

    public void testCreate() {
        assertNotNull(extractor);
    }

    public void testDescriptorSize() {
        assertEquals(128, extractor.descriptorSize());
    }

    public void testDescriptorType() {
        assertEquals(CvType.CV_32F, extractor.descriptorType());
    }

    public void testEmpty() {
//        assertFalse(extractor.empty());
        fail("Not yet implemented"); //SIFT does not override empty() method
    }

    public void testRead() {
        fail("Not yet implemented");
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

//        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.SIFT</name>\n<contrastThreshold>4.0000000000000001e-02</contrastThreshold>\n<edgeThreshold>10.</edgeThreshold>\n<nFeatures>0</nFeatures>\n<nOctaveLayers>3</nOctaveLayers>\n<sigma>1.6000000000000001e+00</sigma>\n</opencv_storage>\n";
        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n</opencv_storage>\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

//        String truth = "%YAML:1.0\n---\nname: \"Feature2D.SIFT\"\ncontrastThreshold: 4.0000000000000001e-02\nedgeThreshold: 10.\nnFeatures: 0\nnOctaveLayers: 3\nsigma: 1.6000000000000001e+00\n";
        String truth = "%YAML:1.0\n---\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
