package org.opencv.test.features2d;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class SIFTDescriptorExtractorTest extends OpenCVTestCase {

    DescriptorExtractor extractor;
    KeyPoint keypoint;
    int matSize;
    Mat truth;

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Core.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    @Override
    protected void setUp() throws Exception {
        extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        keypoint = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        matSize = 100;
        truth = new Mat(1, 128, CvType.CV_32FC1) {
            {
                put(0, 0,
                		0, 0, 0, 0, 0, 0, 0, 0, 16, 12, 17, 28, 26, 0, 0, 2, 23, 14, 12, 9, 6, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                		14, 88, 23, 17, 24, 29, 0, 117, 54, 117, 116, 117, 22, 29, 27, 117, 59, 76, 19, 30, 2, 9, 26, 2, 7, 6, 0, 0,
                		0, 0, 0, 0, 8, 50, 16, 30, 58, 89, 0, 117, 49, 95, 75, 117, 112, 117, 93, 81, 86, 117, 5, 5, 39, 117, 71, 20,
                		20, 12, 0, 0, 1, 20, 19, 0, 0, 0, 2, 14, 4, 1, 0, 69, 0, 0, 14, 90, 31, 35, 56, 25, 0, 0, 0, 0, 2, 12, 16, 0,
                		0, 0, 0, 0, 0, 2, 1);
            }
        };

        super.setUp();
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
        assertFalse(extractor.empty());
    }

    public void testRead() {
        MatOfKeyPoint keypoints =new MatOfKeyPoint(keypoint);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename,
                "%YAML:1.0\nmagnification: 3.\nisNormalize: 1\nrecalculateAngles: 1\nnOctaves: 6\nnOctaveLayers: 4\nfirstOctave: -1\nangleMode: 0\n");

        extractor.read(filename);

        extractor.compute(img, keypoints, descriptors);
        assertMatNotEqual(truth, descriptors, EPS);
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.SIFT</name>\n<contrastThreshold>4.0000000000000001e-02</contrastThreshold>\n<edgeThreshold>10.</edgeThreshold>\n<nFeatures>0</nFeatures>\n<nOctaveLayers>3</nOctaveLayers>\n<sigma>1.6000000000000001e+00</sigma>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\nname: \"Feature2D.SIFT\"\ncontrastThreshold: 4.0000000000000001e-02\nedgeThreshold: 10.\nnFeatures: 0\nnOctaveLayers: 3\nsigma: 1.6000000000000001e+00\n";
        assertEquals(truth, readFile(filename));
    }

}
