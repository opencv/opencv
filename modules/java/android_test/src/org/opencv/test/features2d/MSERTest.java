package org.opencv.test.features2d;

import org.opencv.features2d.MSER;
import org.opencv.test.OpenCVTestCase;

public class MSERTest extends OpenCVTestCase {
	
	private MSER mser;
	
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        mser = null;
    }
	
	public void test_1() {
		super.test_1("FEATURES2D.MSER");
	}

	public void testMSER() {
		mser = new MSER();
		assertTrue(null != mser);
	}

	public void testMSERIntIntIntDoubleDoubleIntDoubleDoubleInt() {
		mser = new MSER(5, 60, 14400, .25f, .2f, 200, 1.01, .003, 5);
		assertTrue(null != mser);
	}

}
