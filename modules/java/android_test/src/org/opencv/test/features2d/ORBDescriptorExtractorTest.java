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

public class ORBDescriptorExtractorTest extends OpenCVTestCase {

    DescriptorExtractor extractor;
    int matSize;

    public static void assertDescriptorsClose(Mat expected, Mat actual, int allowedDistance) {
        double distance = Core.norm(expected, actual, Core.NORM_HAMMING);
        assertTrue("expected:<" + allowedDistance + "> but was:<" + distance + ">", distance <= allowedDistance);
    }

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Core.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
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
                        6, 74, 6, 129, 2, 130, 56, 0, 36, 132, 66, 165, 172, 6, 3, 72, 102, 61, 163, 214, 0, 144, 65, 232, 4, 32, 138, 129, 4, 21, 37, 88);
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
        assertFalse(extractor.empty());
    }

    public void testRead() {
        KeyPoint point = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        MatOfKeyPoint keypoints = new MatOfKeyPoint(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nscaleFactor: 1.1\nnLevels: 3\nfirstLevel: 0\nedgeThreshold: 31\npatchSize: 31\n");
        extractor.read(filename);

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 32, CvType.CV_8UC1) {
            {
                put(0, 0,
                        6, 10, 22, 5, 2, 130, 56, 0, 44, 164, 66, 165, 140, 6, 1, 72, 38, 61, 163, 210, 0, 208, 1, 104, 4, 32, 10, 131, 0, 37, 37, 67);
            }
        };
        assertDescriptorsClose(truth, descriptors, 1);
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.ORB</name>\n<WTA_K>2</WTA_K>\n<edgeThreshold>31</edgeThreshold>\n<firstLevel>0</firstLevel>\n<nFeatures>500</nFeatures>\n<nLevels>8</nLevels>\n<patchSize>31</patchSize>\n<scaleFactor>1.2000000476837158e+00</scaleFactor>\n<scoreType>0</scoreType>\n</opencv_storage>\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e\\+000", "e+00"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\nname: \"Feature2D.ORB\"\nWTA_K: 2\nedgeThreshold: 31\nfirstLevel: 0\nnFeatures: 500\nnLevels: 8\npatchSize: 31\nscaleFactor: 1.2000000476837158e+00\nscoreType: 0\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e\\+000", "e+00"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
