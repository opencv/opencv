package org.opencv.test.features2d;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

import java.util.Arrays;
import java.util.List;

public class ORBDescriptorExtractorTest extends OpenCVTestCase {

    DescriptorExtractor extractor;
    int matSize;

    private Mat getTestImg() {
        Mat cross = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Core.line(cross, new Point(20, matSize / 2), new Point(matSize - 21, matSize / 2), new Scalar(100), 2);
        Core.line(cross, new Point(matSize / 2, 20), new Point(matSize / 2, matSize - 21), new Scalar(100), 2);

        return cross;
    }

    @Override
    protected void setUp() throws Exception {
        extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        matSize = 100;

        super.setUp();
    }

    public void testComputeListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testComputeMatListOfKeyPointMat() {
        KeyPoint point = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        List<KeyPoint> keypoints = Arrays.asList(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 32, CvType.CV_8UC1) {
            {
                put(0, 0, 20, 51, 88, 22, 14, 181, 78, 111, 36, 144, 62, 0, 188, 196, 4, 8, 133, 80, 96, 18, 64, 29, 0,
                        254, 230, 247, 12, 2, 78, 129, 70, 145);
            }
        };
        assertMatEqual(truth, descriptors);
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
        List<KeyPoint> keypoints = Arrays.asList(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nscaleFactor: 1.1\nnLevels: 3\nfirstLevel: 0\nedgeThreshold: 31\npatchSize: 31\n");
        extractor.read(filename);

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 32, CvType.CV_8UC1) {
            {
                put(0, 0, 20, 55, 88, 20, 14, 49, 70, 111, 148, 144, 30, 16, 252, 133, 0, 8, 5, 85, 32, 0, 74, 25, 0,
                        252, 119, 191, 4, 2, 66, 1, 66, 145);
            }
        };
        assertMatEqual(truth, descriptors);
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<scaleFactor>1.2000000476837158e+00</scaleFactor>\n<nLevels>3</nLevels>\n<firstLevel>0</firstLevel>\n<edgeThreshold>31</edgeThreshold>\n<patchSize>31</patchSize>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\nscaleFactor: 1.2000000476837158e+00\nnLevels: 3\nfirstLevel: 0\nedgeThreshold: 31\npatchSize: 31\n";
        assertEquals(truth, readFile(filename));
    }

}
