package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.features2d.BRISK;

public class BRISKDescriptorExtractorTest extends OpenCVTestCase {

    BRISK extractor;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = BRISK.create(); // default (30,3,1)
    }

    public void testCreate() {
        assertNotNull(extractor);
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
        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.BRISK\"\nthreshold: 31\noctaves: 4\npatternScale: 1.1\n");

        extractor.read(filename);

        assertEquals(31, extractor.getThreshold());
        assertEquals(4, extractor.getOctaves());
        assertEquals(1.1f, extractor.getPatternScale());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.BRISK\"\nthreshold: 30\noctaves: 3\npatternScale: 1.\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
