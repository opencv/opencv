package org.opencv.test.objdetect;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.objdetect.QRCodeDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.test.OpenCVTestCase;

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
        QRCodeDetector detector = new QRCodeDetector();
        String output = detector.detectAndDecode(img);
        assertEquals(output, "https://opencv.org/");
    }

    public void testMultipleDetectAndDecode() {
        Mat img = Imgcodecs.imread(testDataPath + "/cv/qrcode/multiple/6_qrcodes.png");
        QRCodeDetector detector = new QRCodeDetector();
        List < String > output = detector.multipleDetectAndDecode(img);
        System.out.println(output);
        assertEquals(output.get(0), "STEP BACK");
        assertEquals(output.get(1), "SKIP");
        assertEquals(output.get(2), "STEP FORWARD");
        assertEquals(output.get(3), "EXTRA");
        assertEquals(output.get(4), "QUESTION");
        assertEquals(output.get(5), "TWO STEPS FORWARD");
    }
}
