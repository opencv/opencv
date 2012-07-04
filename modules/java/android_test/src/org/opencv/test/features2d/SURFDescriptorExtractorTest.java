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

    public void testComputeListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testComputeMatListOfKeyPointMat() {
        KeyPoint point = new KeyPoint(55.775577545166016f, 44.224422454833984f, 16, 9.754629f, 8617.863f, 1, -1);
        MatOfKeyPoint keypoints = new MatOfKeyPoint(point);
        Mat img = getTestImg();
        Mat descriptors = new Mat();

        extractor.compute(img, keypoints, descriptors);

        Mat truth = new Mat(1, 128, CvType.CV_32FC1) {
            {
                put(0, 0,
                        -0.0041138371, 0.0041138371, 0, 0, 0, 0, 0.0014427509, 0.0014427509, -0.0081971241, 0.034624498, 0.032569118,
                        0.032569118, -0.007222258, 0.0076424959, 0.0033254174, 0.0033254174, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.10815519, 0.38033518, 0.24314292, 0.24314292, -0.068393648, 0.068393648,
                        0.039715949, 0.039715949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.7263528e-05, 8.7263528e-05, -6.0081031e-05,
                        6.0081031e-05, -0.00012158759, 0.00012158759, 0.0033254174, 0.0033254174, -0.007222258, 0.0076424964,
                        0.0081971241, 0.034624498, -0.032569118, 0.032569118, -0.077379324, 0.27552885, 0.14366581, 0.31175563,
                        -0.013609707, 0.24329227, -0.091054246, 0.17476201, 0.022970313, 0.022970313, -0.035123408, 0.035771687,
                        0.1907353, 0.3838968, -0.31571922, 0.31571922, 0.0092833797, 0.0092833797, -0.012892088, 0.012957365,
                        0.029558292, 0.073337689, -0.043703932, 0.043703932, 0.0014427509, 0.0014427509, 0, 0, 0.0041138371,
                        0.0041138371, 0, 0, -0.02955829, 0.073337704, 0.043703932, 0.043703932, -0.012892087, 0.012957364,
                        0.0092833797,0.0092833797, 6.0081031e-05, 6.0081031e-05, 0.00012158759, 0.00012158759, -8.7263528e-05,
                        8.7263528e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        );
            }
        };

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
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\nnOctaves: 4\nnOctaveLayers: 2\nextended: 1\nupright: 0\n");

        extractor.read(filename);

        assertEquals(128, extractor.descriptorSize());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        extractor.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.SURF</name>\n<extended>1</extended>\n<hessianThreshold>100.</hessianThreshold>\n<nOctaveLayers>2</nOctaveLayers>\n<nOctaves>4</nOctaves>\n<upright>0</upright>\n</opencv_storage>\n";
        assertEquals(truth, readFile(filename));
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\nname: \"Feature2D.SURF\"\nextended: 1\nhessianThreshold: 100.\nnOctaveLayers: 2\nnOctaves: 4\nupright: 0\n";
        assertEquals(truth, readFile(filename));
    }

}
