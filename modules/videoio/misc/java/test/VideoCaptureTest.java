package org.opencv.test.videoio;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assume.assumeTrue;

import org.junit.Test;
import org.opencv.videoio.VideoCapture;

import org.opencv.test.OpenCVTestCase;

public class VideoCaptureTest extends OpenCVTestCase {

    private VideoCapture capture;
    private boolean isOpened;
    private boolean isSucceed;

    @Override
    public void setUp() throws Exception {
        super.setUp();

        capture = null;
        assumeTrue(false);
        isSucceed = false;
        isOpened = false;
    }

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
