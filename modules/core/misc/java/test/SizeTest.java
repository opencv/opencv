package org.opencv.test.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;

public class SizeTest extends OpenCVTestCase {

    Size dstSize;
    Size sz1;
    Size sz2;

    @Override
    public void setUp() throws Exception {
        super.setUp();

        sz1 = new Size(10.0, 10.0);
        sz2 = new Size(-1, -1);
        dstSize = null;
    }

    @Test
    public void testArea() {
        double area = sz1.area();
        assertEquals(100.0, area, 0);
    }

    @Test
    public void testClone() {
        dstSize = sz1.clone();
        assertEquals(sz1, dstSize);
    }

    @Test
    public void testEqualsObject() {
        assertFalse(sz1.equals(sz2));

        sz2 = sz1.clone();
        assertTrue(sz1.equals(sz2));
    }

    @Test
    public void testHashCode() {
        assertEquals(sz1.hashCode(), sz1.hashCode());
    }

    @Test
    public void testSet() {
        double[] vals1 = {};
        sz2.set(vals1);
        assertEquals(0., sz2.width, 0);
        assertEquals(0., sz2.height, 0);

        double[] vals2 = { 9, 12 };
        sz1.set(vals2);
        assertEquals(9., sz1.width, 0);
        assertEquals(12., sz1.height, 0);
    }

    @Test
    public void testSize() {
        dstSize = new Size();

        assertNotNull(dstSize);
        assertEquals(0., dstSize.width, 0);
        assertEquals(0., dstSize.height, 0);
    }

    @Test
    public void testSizeDoubleArray() {
        double[] vals = { 10, 20 };
        sz2 = new Size(vals);

        assertEquals(10., sz2.width, 0);
        assertEquals(20., sz2.height, 0);
    }

    @Test
    public void testSizeDoubleDouble() {
        assertNotNull(sz1);

        assertEquals(10.0, sz1.width, 0);
        assertEquals(10.0, sz1.height, 0);
    }

    @Test
    public void testSizePoint() {
        Point p = new Point(2, 4);
        sz1 = new Size(p);

        assertNotNull(sz1);
        assertEquals(2.0, sz1.width, 0);
        assertEquals(4.0, sz1.height, 0);
    }

    @Test
    public void testToString() {
        String actual = sz1.toString();
        String expected = "10x10";
        assertEquals(expected, actual);
    }

}
