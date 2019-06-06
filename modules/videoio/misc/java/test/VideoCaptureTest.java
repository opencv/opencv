package org.opencv.test.videoio;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

import org.junit.Test;
import org.opencv.test.OpenCVTestCase;
import org.opencv.videoio.VideoCapture;

public class VideoCaptureTest extends OpenCVTestCase {

    private VideoCapture capture;
    private boolean isSucceed;

    @Test
    public void testGrab() {
        capture = new VideoCapture();
        isSucceed = capture.grab();
        assertFalse(isSucceed);
    }

    @Test
    public void testIsOpened() {
        capture = new VideoCapture();
        assertFalse(capture.isOpened());
    }

    @Test
    public void testVideoCapture() {
        capture = new VideoCapture();
        assertNotNull(capture);
        assertFalse(capture.isOpened());
    }

}
