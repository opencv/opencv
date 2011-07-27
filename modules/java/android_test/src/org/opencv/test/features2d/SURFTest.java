package org.opencv.test.features2d;

import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.features2d.KeyPoint;
import org.opencv.features2d.SURF;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class SURFTest extends OpenCVTestCase {
	
	private SURF surf;

	@Override
    protected void setUp() throws Exception {
        super.setUp();
        
        surf = null;
    }

	public void test_1() {
		super.test_1("features2d.SURF");
	}
	
	public void testDescriptorSize() {
		surf = new SURF(500.0, 4, 2, false);
		assertEquals(64, surf.descriptorSize());
		
		surf = new SURF(500.0, 4, 2, true);
		assertEquals(128, surf.descriptorSize());
	}
	
    public void testDetectMatMatListOfKeyPoint() {
    	surf = new SURF();
    	List<KeyPoint> keypoints = new LinkedList<KeyPoint>();
    	
    	surf.detect(grayChess, new Mat(), keypoints);    	
    	OpenCVTestRunner.Log("" + keypoints.size());
		fail("Not yet implemented");
	}
    
	public void testDetectMatMatListOfKeyPointListOfFloat() {
		fail("Not yet implemented");
	}

	public void testDetectMatMatListOfKeyPointListOfFloatBoolean() {
		fail("Not yet implemented");
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
