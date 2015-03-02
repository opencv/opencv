package org.opencv.test.highgui;

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

    public void testGet() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            double frameWidth = capture.get(Videoio.CV_CAP_PROP_FRAME_WIDTH);
            assertTrue(0 != frameWidth);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testGetSupportedPreviewSizes() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            List<Size> sizes = capture.getSupportedPreviewSizes();
            assertNotNull(sizes);
            assertFalse(sizes.isEmpty());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testGrab() {
        capture = new VideoCapture();
        isSucceed = capture.grab();
        assertFalse(isSucceed);
    }

    public void testGrabFromRealCamera() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            isSucceed = capture.grab();
            assertTrue(isSucceed);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testIsOpened() {
        capture = new VideoCapture();
        assertFalse(capture.isOpened());
    }

    public void testIsOpenedRealCamera() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            isOpened = capture.isOpened();
            assertTrue(isOpened);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testOpen() {
        try {
            capture = new VideoCapture();
            capture.open(Videoio.CV_CAP_ANDROID);
            isOpened = capture.isOpened();
            assertTrue(isOpened);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testRead() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            isSucceed = capture.read(dst);
            assertTrue(isSucceed);
            assertFalse(dst.empty());
            assertEquals(3, dst.channels());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testRelease() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            capture.release();
            assertFalse(capture.isOpened());
            capture = null;
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testRetrieveMat() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            capture.grab();
            isSucceed = capture.retrieve(dst);
            assertTrue(isSucceed);
            assertFalse(dst.empty());
            assertEquals(3, dst.channels());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testRetrieveMatInt() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            capture.grab();
            isSucceed = capture.retrieve(dst, Videoio.CV_CAP_ANDROID_GREY_FRAME);
            assertTrue(isSucceed);
            assertFalse(dst.empty());
            assertEquals(1, dst.channels());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testSet() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            capture.set(Videoio.CV_CAP_PROP_FRAME_WIDTH, 640);
            capture.set(Videoio.CV_CAP_PROP_FRAME_HEIGHT, 480);
            double frameWidth = capture.get(Videoio.CV_CAP_PROP_FRAME_WIDTH);
            capture.read(dst);
            assertEquals(640.0, frameWidth);
            assertEquals(640, dst.cols());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testVideoCapture() {
        capture = new VideoCapture();
        assertNotNull(capture);
        assertFalse(capture.isOpened());
    }

    public void testVideoCaptureInt() {
        try {
            capture = new VideoCapture(Videoio.CV_CAP_ANDROID);
            assertNotNull(capture);
            assertTrue(capture.isOpened());
        } finally {
            if (capture != null) capture.release();
        }
    }
}
