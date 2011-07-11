package org.opencv.test.features2d;

import org.opencv.features2d.StarDetector;
import org.opencv.test.OpenCVTestCase;

public class StarDetectorTest extends OpenCVTestCase {
	
	private StarDetector star;
	
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        star = null;
    }
	
	public void test_1() {
		super.test_1("FEATURES2D.StarDetector");
	}

	public void testStarDetector() {
		star = new StarDetector();
		assertTrue(null != star);
	}
	
	public void testStarDetectorIntIntIntIntInt() {
		star = new StarDetector(45, 30, 10, 8, 5);
		assertTrue(null != star);
	}

}
