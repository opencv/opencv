package org.opencv.test.highgui;

import org.opencv.highgui;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;


public class highguiTest extends OpenCVTestCase {

	public void testDestroyAllWindows() {
		//XXX: highgui.destroyAllWindows()
		fail("Not yet implemented");
	}
	
	public void testDestroyWindow() {
		//XXX: highgui.destroyWindow(winname)
		fail("Not yet implemented");
	}

	public void testGetTrackbarPos() {
		//XXX: highgui.getTrackbarPos(trackbarname, winname)
		fail("Not yet implemented");
	}

	public void testGetWindowProperty() {
		//XXX: highgui.getWindowProperty(winname, prop_id)
		fail("Not yet implemented");
	}

	public void testImdecode() {
		//XXX: highgui.imdecode(buf, flags)
		fail("Not yet implemented");
	}
	
	public void testImreadString() {		
		dst = highgui.imread(OpenCVTestRunner.LENA_PATH);
		assertTrue(!dst.empty());
		assertEquals(3, dst.channels());
		assertTrue(512 == dst.cols());
		assertTrue(512 == dst.rows());
	}

	public void testImreadStringInt() {
		dst = highgui.imread(OpenCVTestRunner.LENA_PATH, 0);
		assertTrue(!dst.empty());
		assertEquals(1, dst.channels());
		assertTrue(512 == dst.cols());
		assertTrue(512 == dst.rows());
	}

	public void testImshow() {
		//XXX: highgui.imshow(winname, mat)
		fail("Not yet implemented");
	}

	public void testNamedWindowString() {
		//XXX: highgui.namedWindow(winname)
		fail("Do not export this function");
	}

	public void testNamedWindowStringInt() {
		//XXX: highgui.namedWindow(winname, flags)
		fail("Do not export this function");
	}

	public void testSetTrackbarPos() {
		//XXX: highgui.setTrackbarPos(trackbarname, winname, pos)
		fail("Not yet implemented");
	}

	public void testSetWindowProperty() {
		//XXX: highgui.setWindowProperty(winname, prop_id, prop_value)
		fail("Not yet implemented");
	}

	public void testStartWindowThread() {
		//XXX: highgui.startWindowThread()
		fail("Do not export this function");
	}

	public void testWaitKey() {
		//XXX: highgui.waitKey()
		fail("Not yet implemented");
	}

	public void testWaitKeyInt() {
		//XXX: highgui.waitKey(delay)
		fail("Not yet implemented");
	}

}
