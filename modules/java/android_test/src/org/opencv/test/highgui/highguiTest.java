package org.opencv.test.highgui;

import org.opencv.highgui;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;


public class highguiTest extends OpenCVTestCase {

	public void testDestroyAllWindows() {
		//XXX: do not export this function
		fail("Do not export this function");
	}
	
	public void testDestroyWindow() {
		//XXX: do not export this function
		fail("Do not export this function");
	}

	public void testGetTrackbarPos() {
		//XXX: do we need this function?
		fail("Not yet implemented");
	}

	public void testGetWindowProperty() {
		//XXX: do we need this function?
		fail("Not yet implemented");
	}

	public void testImdecode() {
		//XXX: do we need this function?
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
		//XXX: do we need this function?
		fail("Not yet implemented");
	}

	public void testNamedWindowString() {
		//XXX: do not export this function
		fail("Do not export this function");
	}

	public void testNamedWindowStringInt() {
		//XXX: do not export this function
		fail("Do not export this function");
	}

	public void testSetTrackbarPos() {
		//XXX: do we need this function?
		fail("Not yet implemented");
	}

	public void testSetWindowProperty() {
		//XXX: do we need this function?
		fail("Not yet implemented");
	}

	public void testStartWindowThread() {
		//XXX: do not export this function
		fail("Do not export this function");
	}

	public void testWaitKey() {
		//XXX: we need this function if only imshow will be implemented
		fail("Not yet implemented");
	}

	public void testWaitKeyInt() {
		//XXX: we need this function if only imshow will be implemented
		fail("Not yet implemented");
	}

}
