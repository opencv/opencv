package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.features2d.GFTTDetector;

public class GFTTFeatureDetectorTest extends OpenCVTestCase {

    GFTTDetector detector;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        detector = GFTTDetector.create(); // default constructor have (1000, 0.01, 1, 3, 3, false, 0.04)
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

    public void testReadYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.GFTTDetector\"\nnfeatures: 500\nqualityLevel: 2.0000000000000000e-02\nminDistance: 2.\nblockSize: 4\ngradSize: 5\nuseHarrisDetector: 1\nk: 5.0000000000000000e-02\n");
        detector.read(filename);

        assertEquals(500, detector.getMaxFeatures());
        assertEquals(0.02, detector.getQualityLevel());
        assertEquals(2.0, detector.getMinDistance());
        assertEquals(4, detector.getBlockSize());
        assertEquals(5, detector.getGradientSize());
        assertEquals(true, detector.getHarrisDetector());
        assertEquals(0.05, detector.getK());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.GFTTDetector\"\nnfeatures: 1000\nqualityLevel: 1.0000000000000000e-02\nminDistance: 1.\nblockSize: 3\ngradSize: 3\nuseHarrisDetector: 0\nk: 4.0000000000000001e-02\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
