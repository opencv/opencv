package org.opencv.test.objdetect;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.objdetect.QRCodeDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.test.OpenCVTestCase;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class QRCodeDetectorTest extends OpenCVTestCase {

    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";
    private String testDataPath;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        testDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);
        if (testDataPath == null)
            throw new Exception(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!");
    }

    public void testDetectAndDecode() {
        Mat img = Imgcodecs.imread(testDataPath + "/cv/qrcode/link_ocv.jpg");
        assertFalse(img.empty());
        QRCodeDetector detector = new QRCodeDetector();
        assertNotNull(detector);
        String output = detector.detectAndDecode(img);
        assertEquals(output, "https://opencv.org/");
    }

    public void testDetectAndDecodeMulti() {
        Mat img = Imgcodecs.imread(testDataPath + "/cv/qrcode/multiple/6_qrcodes.png");
        assertFalse(img.empty());
        QRCodeDetector detector = new QRCodeDetector();
        assertNotNull(detector);
        List < String > output = new ArrayList< String >();
        boolean result = detector.detectAndDecodeMulti(img, output);
        assertTrue(result);
        assertEquals(output.size(), 6);
        List < String > expectedResults = Arrays.asList("SKIP", "EXTRA", "TWO STEPS FORWARD", "STEP BACK", "QUESTION", "STEP FORWARD");
        assertEquals(new HashSet<String>(output), new HashSet<String>(expectedResults));
    }
}
