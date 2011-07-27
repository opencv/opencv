package org.opencv.test.core;

import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.test.OpenCVTestCase;

public class RectTest extends OpenCVTestCase {

	protected void setUp() throws Exception {
		super.setUp();
	}

	public void testArea() {
		fail("Not yet implemented");
	}

	public void testBr() {
		fail("Not yet implemented");
	}

	public void testClone() {
		fail("Not yet implemented");
	}

	public void testContains() {
		Rect r = new Rect(0,0,10,10);
		Point p_inner = new Point(5,5);
		Point p_outer = new Point(5,55);
		Point p_bl = new Point(0,0);
		Point p_br = new Point(10,0);
		Point p_tl = new Point(0,10);
		Point p_tr = new Point(10,10);
		
		assertTrue(r.contains(p_inner));
		assertTrue(r.contains(p_bl));
		
		assertFalse(r.contains(p_outer));
		assertFalse(r.contains(p_br));
		assertFalse(r.contains(p_tl));
		assertFalse(r.contains(p_tr));
	}

	public void testEqualsObject() {
		fail("Not yet implemented");
	}

	public void testRect() {
		fail("Not yet implemented");
	}

	public void testRectDoubleArray() {
		fail("Not yet implemented");
	}

	public void testRectIntIntIntInt() {
		fail("Not yet implemented");
	}

	public void testRectPointPoint() {
		fail("Not yet implemented");
	}

	public void testRectPointSize() {
		fail("Not yet implemented");
	}

	public void testSet() {
		fail("Not yet implemented");
	}

	public void testSize() {
		fail("Not yet implemented");
	}

	public void testTl() {
		fail("Not yet implemented");
	}

	public void testToString() {
		fail("Not yet implemented");
	}

}
