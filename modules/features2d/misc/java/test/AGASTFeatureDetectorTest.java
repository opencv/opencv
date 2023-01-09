package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.features2d.AgastFeatureDetector;

public class AGASTFeatureDetectorTest extends OpenCVTestCase {

    AgastFeatureDetector detector;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        detector = AgastFeatureDetector.create(); // default (10,true,3)
    }

    public void testCreate() {
        assertNotNull(detector);
    }

    public void testDetectListOfMatListOfListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPointMat() {
        fail("Not yet implemented");
    }

    public void testEmpty() {
        fail("Not yet implemented");
    }

    public void testRead() {
        String filename = OpenCVTestRunner.getTempFileName("xml");
        writeFile(filename, "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.AgastFeatureDetector</name>\n<threshold>11</threshold>\n<nonmaxSuppression>0</nonmaxSuppression>\n<type>2</type>\n</opencv_storage>\n");

        detector.read(filename);

        assertEquals(11, detector.getThreshold());
        assertEquals(false, detector.getNonmaxSuppression());
        assertEquals(2, detector.getType());
    }

    public void testReadYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.AgastFeatureDetector\"\nthreshold: 11\nnonmaxSuppression: 0\ntype: 2\n");

        detector.read(filename);

        assertEquals(11, detector.getThreshold());
        assertEquals(false, detector.getNonmaxSuppression());
        assertEquals(2, detector.getType());
    }

    public void testWrite() {
        String filename = OpenCVTestRunner.getTempFileName("xml");

        detector.write(filename);

        String truth = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<name>Feature2D.AgastFeatureDetector</name>\n<threshold>10</threshold>\n<nonmaxSuppression>1</nonmaxSuppression>\n<type>3</type>\n</opencv_storage>\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.AgastFeatureDetector\"\nthreshold: 10\nnonmaxSuppression: 1\ntype: 3\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
