package org.opencv.test.highgui;

import java.util.List;

import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

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
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            double frameWidth = capture.get(Highgui.CV_CAP_PROP_FRAME_WIDTH);
            assertTrue(0 != frameWidth);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testGetSupportedPreviewSizes() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            List<Size> sizes = capture.getSupportedPreviewSizes();
            assertNotNull(sizes);
            assertFalse(sizes.isEmpty());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testGrab() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        capture = new VideoCapture();
        isSucceed = capture.grab();
        assertFalse(isSucceed);
    }

    public void testGrabFromRealCamera() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            isSucceed = capture.grab();
            assertTrue(isSucceed);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testIsOpened() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        capture = new VideoCapture();
        assertFalse(capture.isOpened());
    }

    public void testIsOpenedRealCamera() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            isOpened = capture.isOpened();
            assertTrue(isOpened);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testOpen() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture();
            capture.open(Highgui.CV_CAP_ANDROID);
            isOpened = capture.isOpened();
            assertTrue(isOpened);
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testRead() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            isSucceed = capture.read(dst);
            assertTrue(isSucceed);
            assertFalse(dst.empty());
            assertEquals(3, dst.channels());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testRelease() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            capture.release();
            assertFalse(capture.isOpened());
            capture = null;
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testRetrieveMat() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
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
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            capture.grab();
            isSucceed = capture.retrieve(dst, Highgui.CV_CAP_ANDROID_GREY_FRAME);
            assertTrue(isSucceed);
            assertFalse(dst.empty());
            assertEquals(1, dst.channels());
        } finally {
            if (capture != null) capture.release();
        }
    }

    public void testSet() {
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            capture.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, 640);
            capture.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, 480);
            double frameWidth = capture.get(Highgui.CV_CAP_PROP_FRAME_WIDTH);
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
        if (!isTestCaseEnabled) {
            fail("Not yet implemented");
        }

        try {
            capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
            assertNotNull(capture);
            assertTrue(capture.isOpened());
        } finally {
            if (capture != null) capture.release();
        }
    }
}
