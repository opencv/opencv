package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.features2d.MSER;

public class MSERFeatureDetectorTest extends OpenCVTestCase {

    MSER detector;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        detector = MSER.create(); // default constructor have (5, 60, 14400, .25, .2, 200, 1.01, .003, 5)
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

        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.MSER\"\ndelta: 6\nminArea: 62\nmaxArea: 14402\nmaxVariation: .26\nminDiversity: .3\nmaxEvolution: 201\nareaThreshold: 1.02\nminMargin: 3.0e-3\nedgeBlurSize: 3\npass2Only: 1\n");
        detector.read(filename);

        assertEquals(6, detector.getDelta());
        assertEquals(62, detector.getMinArea());
        assertEquals(14402, detector.getMaxArea());
        assertEquals(.26, detector.getMaxVariation());
        assertEquals(.3, detector.getMinDiversity());
        assertEquals(201, detector.getMaxEvolution());
        assertEquals(1.02, detector.getAreaThreshold());
        assertEquals(0.003, detector.getMinMargin());
        assertEquals(3, detector.getEdgeBlurSize());
        assertEquals(true, detector.getPass2Only());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.MSER\"\ndelta: 5\nminArea: 60\nmaxArea: 14400\nmaxVariation: 0.25\nminDiversity: 0.20000000000000001\nmaxEvolution: 200\nareaThreshold: 1.01\nminMargin: 0.0030000000000000001\nedgeBlurSize: 5\npass2Only: 0\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
