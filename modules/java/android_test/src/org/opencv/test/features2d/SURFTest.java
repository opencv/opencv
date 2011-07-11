package org.opencv.test.features2d;

import org.opencv.features2d.SURF;
import org.opencv.test.OpenCVTestCase;

public class SURFTest extends OpenCVTestCase {
	
	private SURF surf;
	
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        surf = null;
    }
    
	public void test_1() {
		super.test_1("FEATURES2D.SURF");
	}

	public void testDescriptorSize() {
		surf = new SURF(500.0, 4, 2, false);
		assertEquals(64, surf.descriptorSize());
		
		surf = new SURF(500.0, 4, 2, true);
		assertEquals(128, surf.descriptorSize());
	}

	public void testSURF() {
		surf = new SURF();
		assertTrue(null != surf);
	}
	
	public void testSURFDouble() {
		surf = new SURF(500.0);
		assertTrue(null != surf);
	}

	public void testSURFDoubleInt() {
		surf = new SURF(500.0, 4);
		assertTrue(null != surf);
	}

	public void testSURFDoubleIntInt() {
		surf = new SURF(500.0, 4, 2);
		assertTrue(null != surf);
	}

	public void testSURFDoubleIntIntBoolean() {

	}

	public void testSURFDoubleIntIntBooleanBoolean() {
		surf = new SURF(500.0, 4, 2, false, false);
		assertTrue(null != surf);
	}

}
