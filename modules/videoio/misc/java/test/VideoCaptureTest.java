package org.opencv.test.videoio;

import java.util.List;

import org.opencv.core.Size;
import org.opencv.videoio.Videoio;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.IStreamReader;

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

    public void testDefaultConstructor() {
        capture = new VideoCapture();
        assertNotNull(capture);
        assertFalse(capture.isOpened());
    }

    public void testConstructorWithFilename() {
        capture = new VideoCapture("some_file.avi");
        assertNotNull(capture);
    }

    public void testConstructorWithFilenameAndExplicitlySpecifiedAPI() {
        capture = new VideoCapture("some_file.avi", Videoio.CAP_ANY);
        assertNotNull(capture);
    }

    public void testConstructorWithIndex() {
        capture = new VideoCapture(0);
        assertNotNull(capture);
    }

    public void testConstructorWithIndexAndExplicitlySpecifiedAPI() {
        capture = new VideoCapture(0, Videoio.CAP_ANY);
        assertNotNull(capture);
    }

    public void testConstructorStream() {
        IStreamReader stream = new IStreamReader() {
            // @Override
            // public int read(String buffer, int size) {
            //     return 0;
            // }

            // @Override
            // public int seek(int offset, int origin) {
            //     return 0;
            // }
        };
        capture = new VideoCapture(stream, Videoio.CAP_ANY, null);
        assertNotNull(capture);
        assertTrue(capture.isOpened());
    }
}
