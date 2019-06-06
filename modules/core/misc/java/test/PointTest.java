package org.opencv.test.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.test.OpenCVTestCase;

public class PointTest extends OpenCVTestCase {

    private Point p1;
    private Point p2;

    @Override
    public void setUp() throws Exception {
        super.setUp();

        p1 = new Point(2, 2);
        p2 = new Point(1, 1);
    }

    @Test
    public void testClone() {
        Point truth = new Point(1, 1);
        Point dstPoint = truth.clone();
        assertEquals(truth, dstPoint);
    }

    @Test
    public void testDot() {
        double result = p1.dot(p2);
        assertEquals(4.0, result, 0);
    }

    @Test
    public void testEqualsObject() {
        boolean flag = p1.equals(p1);
        assertTrue(flag);

        flag = p1.equals(p2);
        assertFalse(flag);
    }

    @Test
    public void testHashCode() {
        assertEquals(p1.hashCode(), p1.hashCode());
    }

    @Test
    public void testInside() {
        Rect rect = new Rect(0, 0, 5, 3);
        assertTrue(p1.inside(rect));

        Point p2 = new Point(3, 3);
        assertFalse(p2.inside(rect));
    }

    @Test
    public void testPoint() {
        Point p = new Point();

        assertNotNull(p);
        assertEquals(0.0, p.x, 0);
        assertEquals(0.0, p.y, 0);
    }

    @Test
    public void testPointDoubleArray() {
        double[] vals = { 2, 4 };
        Point p = new Point(vals);

        assertEquals(2.0, p.x, 0);
        assertEquals(4.0, p.y, 0);
    }

    @Test
    public void testPointDoubleDouble() {
        p1 = new Point(7, 5);

        assertNotNull(p1);
        assertEquals(7.0, p1.x, 0);
        assertEquals(5.0, p1.y, 0);
    }

    @Test
    public void testSet() {
        double[] vals1 = {};
        p1.set(vals1);
        assertEquals(0.0, p1.x, 0);
        assertEquals(0.0, p1.y, 0);

        double[] vals2 = { 6, 10 };
        p2.set(vals2);
        assertEquals(6.0, p2.x, 0);
        assertEquals(10.0, p2.y, 0);
    }

    @Test
    public void testToString() {
        String actual = p1.toString();
        String expected = "{2.0, 2.0}";
        assertEquals(expected, actual);
    }

}
