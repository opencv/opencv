package org.opencv.test.highgui;

import org.opencv.highgui.Highgui;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;


public class highguiTest extends OpenCVTestCase {

	public void testImreadString() {		
		dst = Highgui.imread(OpenCVTestRunner.LENA_PATH);
		assertTrue(!dst.empty());
		assertEquals(3, dst.channels());
		assertTrue(512 == dst.cols());
		assertTrue(512 == dst.rows());
	}

	public void testImreadStringInt() {
		dst = Highgui.imread(OpenCVTestRunner.LENA_PATH, 0);
		assertTrue(!dst.empty());
		assertEquals(1, dst.channels());
		assertTrue(512 == dst.cols());
		assertTrue(512 == dst.rows());
	}

}
