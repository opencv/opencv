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

public class SURFDescriptorExtractorTest extends OpenCVTestCase {

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
        extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        matSize = 100;

        super.setUp();
    }

    public void testCompute() {
        KeyPoint point = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        List<KeyPoint> keypoints = Arrays.asList(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 64, CvType.CV_32FC1) {
            {
                put(0, 0, 0, 0, 0, 0, 0.011540107, 0.0029440077, 0.095483348, 0.018144149, 0.00014820647, 0, 0.00014820647, 0, 0, 0, 0, 0, 0, -0.00014820647,
                        0, 0.00014820647, 0.10196275, 0.0099145742, 0.57075155, 0.047922116, 0, 0, 0, 0, 0, 0, 0, 0, 0.0029440068, -0.011540107, 0.018144149,
                        0.095483348, 0.085385554, -0.054076977, 0.34105155, 0.47911066, 0.023395451, -0.11012388, 0.088196531, 0.50863767, 0.0031790689,
                        -0.019882837, 0.0089476965, 0.054817006, -0.0033560959, -0.0011770058, 0.0033560959, 0.0011770058, 0.019882834, 0.0031790687,
                        0.054817006, 0.0089476984, 0, 0, 0, 0, -0.0011770058, 0.0033560959, 0.0011770058, 0.0033560959);
            }
        };

        assertMatEqual(truth, descriptors, EPS);
    }

    public void testCreate() {
        assertNotNull(extractor);
    }

    public void testDescriptorSize() {
        assertEquals(64, extractor.descriptorSize());
    }

    public void testDescriptorType() {
        assertEquals(CvType.CV_32F, extractor.descriptorType());
    }

    public void testEmpty() {
        assertFalse(extractor.empty());
    }

    public void testRead() {
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nnOctaves: 4\nnOctaveLayers: 2\nextended: 1\nupright: 0\n");

        extractor.read(filename);

        assertEquals(128, extractor.descriptorSize());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<nOctaves>4</nOctaves>\n<nOctaveLayers>2</nOctaveLayers>\n<extended>0</extended>\n<upright>0</upright>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\nnOctaves: 4\nnOctaveLayers: 2\nextended: 0\nupright: 0\n";
        assertEquals(truth, readFile(filename));
    }

}
