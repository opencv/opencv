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

public class BRIEFDescriptorExtractorTest extends OpenCVTestCase {

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
        extractor = DescriptorExtractor.create(DescriptorExtractor.BRIEF);
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
                put(0, 0, 96, 0, 76, 24, 47, 182, 68, 137, 149, 195, 67, 16, 187, 224, 74, 8, 82, 169, 87, 70, 44, 4, 192, 56, 13, 128, 44, 106, 146, 72, 194,
                        245);
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
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\ndescriptorSize: 64\n");

        extractor.read(filename);

        assertEquals(64, extractor.descriptorSize());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<descriptorSize>32</descriptorSize>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\ndescriptorSize: 32\n";
        assertEquals(truth, readFile(filename));
    }

}
