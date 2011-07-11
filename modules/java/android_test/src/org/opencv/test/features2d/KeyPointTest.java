package org.opencv.test.features2d;

import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;

public class KeyPointTest extends OpenCVTestCase {
	
	private KeyPoint keyPoint;
	private float x;
	private float y;
	private float size;
	
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        keyPoint = null;
    	x = 1.0f;
    	y = 2.0f;
    	size = 3.0f;
    }
	
	public void test_1() {
		super.test_1("FEATURES2D.KeyPoint");
	}

	public void testKeyPoint() {
		keyPoint = new KeyPoint();
		assertTrue(null != keyPoint);
	}

	public void testKeyPointFloatFloatFloat() {
		keyPoint = new KeyPoint(x, y, size);
		assertTrue(null != keyPoint);
	}

	public void testKeyPointFloatFloatFloatFloat() {
		keyPoint = new KeyPoint(x, y, size, 10.0f);
		assertTrue(null != keyPoint);
	}

	public void testKeyPointFloatFloatFloatFloatFloat() {
		keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f);
		assertTrue(null != keyPoint);
	}

	public void testKeyPointFloatFloatFloatFloatFloatInt() {
		keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1);
		assertTrue(null != keyPoint);
	}

	public void testKeyPointFloatFloatFloatFloatFloatIntInt() {
		keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1, 1);
		assertTrue(null != keyPoint);
	}

}
