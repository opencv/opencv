package org.opencv.test.barcode;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.objdetect.BarcodeDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.test.OpenCVTestCase;
import java.util.ArrayList;

public class BarcodeDetectorTest extends OpenCVTestCase {

    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";
    private String testDataPath;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        // relys on https://developer.android.com/reference/java/lang/System
        isTestCaseEnabled = System.getProperties().getProperty("java.vm.name") != "Dalvik";

        if (isTestCaseEnabled) {
            testDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);
            if (testDataPath == null)
                throw new Exception(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!");
        }
    }

    public void testDetectAndDecode() {
        Mat img = Imgcodecs.imread(testDataPath + "/cv/barcode/multiple/4_barcodes.jpg");
        assertFalse(img.empty());
        BarcodeDetector detector = new BarcodeDetector();
        assertNotNull(detector);
        List < String > infos = new ArrayList< String >();
        List < String > types = new ArrayList< String >();

        boolean result = detector.detectAndDecodeWithType(img, infos, types);
        assertTrue(result);
        assertEquals(infos.size(), 4);
        assertEquals(types.size(), 4);
        final String[]  correctResults = {"9787122276124", "9787118081473", "9787564350840", "9783319200064"};
        for (int i = 0; i < 4; i++) {
            assertEquals(types.get(i), "EAN_13");
            result = false;
            for (int j = 0; j < 4; j++) {
                if (correctResults[j].equals(infos.get(i))) {
                    result = true;
                    break;
                }
            }
            assertTrue(result);
        }

    }
}
