package org.opencv.test.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.test.OpenCVTestCase;

public class Point3Test extends OpenCVTestCase {

    private Point3 p1;
    private Point3 p2;

    @Override
    public void setUp() throws Exception {
        super.setUp();

        p1 = new Point3(2, 2, 2);
        p2 = new Point3(1, 1, 1);
    }

    @Test
    public void testClone() {
        Point3 truth = new Point3(1, 1, 1);
        p1 = truth.clone();
        assertEquals(truth, p1);
    }

    @Test
    public void testCross() {
        Point3 dstPoint = p1.cross(p2);
        Point3 truth = new Point3(0, 0, 0);
        assertEquals(truth, dstPoint);
    }

    @Test
    public void testDot() {
        double result = p1.dot(p2);
        assertEquals(6.0, result, 0);
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
    public void testPoint3() {
        p1 = new Point3();

        assertNotNull(p1);
        assertTrue(0 == p1.x);
        assertTrue(0 == p1.y);
        assertTrue(0 == p1.z);
    }

    @Test
    public void testPoint3DoubleArray() {
        double[] vals = { 1, 2, 3 };
        p1 = new Point3(vals);

        assertTrue(1 == p1.x);
        assertTrue(2 == p1.y);
        assertTrue(3 == p1.z);
    }

    @Test
    public void testPoint3DoubleDoubleDouble() {
        p1 = new Point3(1, 2, 3);

        assertEquals(1., p1.x, 0);
        assertEquals(2., p1.y, 0);
        assertEquals(3., p1.z, 0);
    }

    @Test
    public void testPoint3Point() {
        Point p = new Point(2, 3);
        p1 = new Point3(p);

        assertEquals(2., p1.x, 0);
        assertEquals(3., p1.y, 0);
        assertEquals(0., p1.z, 0);
    }

    @Test
    public void testSet() {
        double[] vals1 = {};
        p1.set(vals1);

        assertEquals(0., p1.x, 0);
        assertEquals(0., p1.y, 0);
        assertEquals(0., p1.z, 0);

        double[] vals2 = { 3, 6, 10 };
        p1.set(vals2);

        assertEquals(3., p1.x, 0);
        assertEquals(6., p1.y, 0);
        assertEquals(10., p1.z, 0);
    }

    @Test
    public void testToString() {
        String actual = p1.toString();
        String expected = "{2.0, 2.0, 2.0}";
        assertEquals(expected, actual);
    }

}
