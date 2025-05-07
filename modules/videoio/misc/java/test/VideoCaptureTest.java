package org.opencv.test.videoio;

import java.util.List;
import java.io.File;
import java.io.RandomAccessFile;
import java.io.IOException;
import java.io.FileNotFoundException;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.MatOfInt;
import org.opencv.videoio.Videoio;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.IStreamReader;

import org.opencv.test.OpenCVTestCase;

public class VideoCaptureTest extends OpenCVTestCase {
    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";

    private VideoCapture capture;
    private boolean isOpened;
    private boolean isSucceed;
    private File testDataPath;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        capture = null;
        isSucceed = false;
        isOpened = false;

        String envTestDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);

        if(envTestDataPath == null) throw new Exception(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!");

        testDataPath = new File(envTestDataPath);
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

    public void testConstructorStream() throws FileNotFoundException {
        // Check backend is available
        Integer apiPref = Videoio.CAP_ANY;
        for (Integer backend : Videoio.getStreamBufferedBackends())
        {
            if (!Videoio.hasBackend(backend))
                continue;
            if (!Videoio.isBackendBuiltIn(backend))
            {
                int[] abi = new int[1], api = new int[1];
                Videoio.getStreamBufferedBackendPluginVersion(backend, abi, api);
                if (abi[0] < 1 || (abi[0] == 1 && api[0] < 2))
                    continue;
            }
            apiPref = backend;
            break;
        }
        if (apiPref == Videoio.CAP_ANY)
        {
            throw new TestSkipException();
        }

        RandomAccessFile f = new RandomAccessFile(new File(testDataPath, "cv/video/768x576.avi"), "r");

        IStreamReader stream = new IStreamReader(0)
        {
            @Override
            public long read(byte[] buffer, long size)
            {
                try
                {
                    return Math.max(f.read(buffer), 0);
                }
                catch (IOException e)
                {
                    System.out.println(e.getMessage());
                    return 0;
                }
            }

            @Override
            public long seek(long offset, int origin)
            {
                try
                {
                    if (origin == 0)
                        f.seek(offset);
                    return f.getFilePointer();
                }
                catch (IOException e)
                {
                    System.out.println(e.getMessage());
                    return 0;
                }
            }
        };
        capture = new VideoCapture(stream, apiPref, new MatOfInt());
        assertNotNull(capture);
        assertTrue(capture.isOpened());

        Mat frame = new Mat();
        assertTrue(capture.read(frame));
        assertEquals(frame.rows(), 576);
        assertEquals(frame.cols(), 768);
    }
}
