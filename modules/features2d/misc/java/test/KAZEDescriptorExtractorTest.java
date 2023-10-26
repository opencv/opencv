package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.features2d.KAZE;

public class KAZEDescriptorExtractorTest extends OpenCVTestCase {

    KAZE extractor;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = KAZE.create(); // default (false,false,0.001f,4,4,1)
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
        writeFile(filename, "%YAML:1.0\n---\nformat: 3\nname: \"Feature2D.KAZE\"\nextended: 1\nupright: 1\nthreshold: 0.125\noctaves: 3\nsublevels: 5\ndiffusivity: 2\n");

        extractor.read(filename);

        assertEquals(true, extractor.getExtended());
        assertEquals(true, extractor.getUpright());
        assertEquals(0.125, extractor.getThreshold());
        assertEquals(3, extractor.getNOctaves());
        assertEquals(5, extractor.getNOctaveLayers());
        assertEquals(2, extractor.getDiffusivity());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\n---\nformat: 3\nname: \"Feature2D.KAZE\"\nextended: 0\nupright: 0\nthreshold: 1.0000000474974513e-03\noctaves: 4\nsublevels: 4\ndiffusivity: 1\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
