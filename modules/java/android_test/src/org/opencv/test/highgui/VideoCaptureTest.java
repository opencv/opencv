package org.opencv.test.highgui;

import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import org.opencv.test.OpenCVTestCase;


public class VideoCaptureTest extends OpenCVTestCase {
	
	private VideoCapture capture;
	private boolean isSucceed;
	private boolean isOpened; 
	
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        capture = null;
        isSucceed = false;
        isOpened = false;
    }
	
	public void test_1() {
		super.test_1("HIGHGUI.VideoCapture");
	}

	public void testGet() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		double frameWidth = capture.get(Highgui.CV_CAP_PROP_FRAME_WIDTH);
		capture.release();
		assertTrue(0 != frameWidth);
	}

	public void testGrab() {
		capture = new VideoCapture();
		isSucceed = capture.grab();
		assertFalse(isSucceed);
	}
	
	public void testGrabFromRealCamera() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		isSucceed = capture.grab();
		capture.release();
		assertTrue(isSucceed);
	}

	public void testIsOpened() {
		capture = new VideoCapture();
		assertFalse(capture.isOpened());
	}
	
	public void testIsOpenedRealCamera() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		isOpened = capture.isOpened();
		capture.release();
		assertTrue(isOpened);
	}

	public void testOpenInt() {
		capture = new VideoCapture();
		capture.open(Highgui.CV_CAP_ANDROID);
		isOpened = capture.isOpened();
		capture.release();
		assertTrue(isOpened);
	}

	public void testOpenString() {
		fail("Not yet implemented");
	}

	public void testRead() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		isSucceed = capture.read(dst);
		capture.release();
		assertTrue(isSucceed);
		assertFalse(dst.empty());
		assertEquals(3, dst.channels());
	}

	public void testRelease() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		capture.release();
		assertFalse(capture.isOpened());
	}

	public void testRetrieveMat() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		capture.grab();
		isSucceed = capture.retrieve(dst);
		capture.release();
		assertTrue(isSucceed);
		assertFalse(dst.empty());
		assertEquals(3, dst.channels());
	}

	public void testRetrieveMatInt() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		capture.grab();
		isSucceed = capture.retrieve(dst, 1);
		capture.release();
		assertTrue(isSucceed);
		assertFalse(dst.empty());
		//OpenCVTestRunner.Log(dst.toString());
		assertEquals(1, dst.channels());
	}

	public void testSet() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		capture.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, 640.0);
		double frameWidth = capture.get(Highgui.CV_CAP_PROP_FRAME_WIDTH);
		capture.read(dst);
		capture.release();
		assertEquals(640.0, frameWidth);
		assertEquals(640, dst.cols());
	}

	public void testVideoCapture() {
		capture = new VideoCapture();
		assertTrue(null != capture);
	}

	public void testVideoCaptureInt() {
		capture = new VideoCapture(Highgui.CV_CAP_ANDROID);
		assertTrue(null != capture);
		isOpened = capture.isOpened();
		capture.release();
		assertTrue(isOpened);
	}

	public void testVideoCaptureString() {
		fail("Not yet implemented");
	}

}
