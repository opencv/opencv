package org.opencv.test.core;

import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.test.OpenCVTestCase;

public class PointTest extends OpenCVTestCase {
    
    private Point p1;
    private Point p2;    

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        p1 = new Point(2, 2);
        p2 = new Point(1, 1);
    }

    public void testClone() {
        Point truth = new Point(1, 1);
        Point dstPoint = truth.clone();
        assertEquals(truth, dstPoint);
    }

    public void testDot() {
        double result = p1.dot(p2);
        assertEquals(4.0, result);
    }

    public void testEqualsObject() {
        boolean flag = p1.equals(p1);
        assertTrue(flag);

        flag = p1.equals(p2);
        assertFalse(flag);
    }

    public void testInside() {
        Rect rect = new Rect(0, 0, 5, 3);
        assertTrue(p1.inside(rect));

        Point p2 = new Point(3, 3);
        assertFalse(p2.inside(rect));
    }

    public void testPoint() {
        Point p = new Point();
        
        assertNotNull(p);
        assertEquals(0.0, p.x);
        assertEquals(0.0, p.y);
    }

    public void testPointDoubleArray() {
        double[] vals = { 2, 4 };
        Point p = new Point(vals);

        assertEquals(2.0, p.x);
        assertEquals(4.0, p.y);
    }

    public void testPointDoubleDouble() {
        p1 = new Point(7, 5);
        
        assertNotNull(p1);
        assertEquals(7.0, p1.x);
        assertEquals(5.0, p1.y);
    }

    public void testSet() {
        double[] vals1 = {};
        p1.set(vals1);
        assertEquals(0.0, p1.x);
        assertEquals(0.0, p1.y);

        double[] vals2 = { 6, 10 };
        p2.set(vals2);
        assertEquals(6.0, p2.x);
        assertEquals(10.0, p2.y);
    }

    public void testToString() {
        String actual = p1.toString();
        String expected = "{2.0, 2.0}";
        assertEquals(expected, actual);
    }

}
