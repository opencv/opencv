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
import org.opencv.features2d.SIFT;

public class SIFTDescriptorExtractorTest extends OpenCVTestCase {

    SIFT extractor;
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
                          0, 0, 0, 1, 3, 0, 0, 0, 15, 23, 22, 20, 24, 2, 0, 0, 7, 8, 2, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 16, 13, 2, 0, 0, 117,
                          86, 79, 68, 117, 42, 5, 5, 79, 60, 117, 25, 9, 2, 28, 19, 11, 13,
                          20, 2, 0, 0, 5, 8, 0, 0, 76, 58, 34, 31, 97, 16, 95, 49, 117, 92,
                          117, 112, 117, 76, 117, 54, 117, 25, 29, 22, 117, 117, 16, 11, 14,
                          1, 0, 0, 22, 26, 0, 0, 0, 0, 1, 4, 15, 2, 47, 8, 0, 0, 82, 56, 31,
                          17, 81, 12, 0, 0, 26, 23, 18, 23, 0, 0, 0, 0, 0, 0, 0, 0
                    );
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
        fail("Not yet implemented"); // SIFT does not override empty() method
    }

    public void testReadYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.SIFT\"\nnfeatures: 100\nnOctaveLayers: 4\ncontrastThreshold: 5.0000000000000001e-02\nedgeThreshold: 11\nsigma: 1.7\ndescriptorType: 5\n");

        extractor.read(filename);

        assertEquals(128, extractor.descriptorSize());

        assertEquals(100, extractor.getNFeatures());
        assertEquals(4, extractor.getNOctaveLayers());
        assertEquals(0.05, extractor.getContrastThreshold());
        assertEquals(11., extractor.getEdgeThreshold());
        assertEquals(1.7, extractor.getSigma());
        assertEquals(5, extractor.descriptorType());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.SIFT\"\nnfeatures: 0\nnOctaveLayers: 3\ncontrastThreshold: 0.040000000000000001\nedgeThreshold: 10.\nsigma: 1.6000000000000001\ndescriptorType: 5\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
