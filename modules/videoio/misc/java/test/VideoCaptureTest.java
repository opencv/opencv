package org.opencv.test.videoio;

import java.util.List;

import org.opencv.core.Size;
import org.opencv.videoio.Videoio;
import org.opencv.videoio.VideoCapture;

import org.opencv.test.OpenCVTestCase;

public class VideoCaptureTest extends OpenCVTestCase {

    private VideoCapture capture;
    private boolean isOpened;
    private boolean isSucceed;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        capture = null;
        isTestCaseEnabled = false;
        isSucceed = false;
        isOpened = false;
    }

    public void testGrab() {
        capture = new VideoCapture();
        isSucceed = capture.grab();
        assertFalse(isSucceed);
    }

    public void testIsOpened() {
        capture = new VideoCapture();
        assertFalse(capture.isOpened());
    }

    public void testVideoCapture() {
        capture = new VideoCapture();
        assertNotNull(capture);
        assertFalse(capture.isOpened());
    }

}
