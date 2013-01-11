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
        super.setUp();
        extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
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

        Mat truth = new Mat(1, 128, CvType.CV_32FC1) {
            {
                put(0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0.045382127, 0.075976953, -0.031969212, 0.035002094, 0.012224297,
                          0.012286193, -0.0088025155, 0.0088025155, 0.00017225844, 0.00017225844, 0, 0, 8.2743405e-05,
                          8.2743405e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8.2743405e-05, 8.2743405e-05, -0.00017225844,
                          0.00017225844, 0, 0, 0.31723264, 0.42715758, -0.19872268, 0.23621935, 0.033304065, 0.033918764,
                          -0.021780485, 0.021780485, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0088025145,
                          0.0088025145, 0.012224296, 0.012286192, -0.045382123, 0.075976953, 0.031969212, 0.035002094,
                          0.10047197, 0.21463872, -0.0012294546, 0.18176091, -0.075555265, 0.35627601, 0.01270232,
                          0.20058797, -0.037658721, 0.037658721, 0.064850949, 0.064850949, -0.27688536, 0.44229308,
                          0.14888979, 0.14888979, -0.0031531656, 0.0031531656, 0.0068481555, 0.0072466261, -0.034193151,
                          0.040314503, 0.01108359, 0.023398584, -0.00071876607, 0.00071876607, -0.0031819802,
                          0.0031819802, 0, 0, -0.0013680183, 0.0013680183, 0.034193147, 0.040314503, -0.01108359,
                          0.023398584, 0.006848156, 0.0072466265, -0.0031531656, 0.0031531656, 0, 0, 0, 0, 0, 0, 0, 0,
                         -0.0013680183, 0.0013680183, 0, 0, 0.00071876607, 0.00071876607, 0.0031819802, 0.0031819802
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
