package org.opencv.test.core;

import org.opencv.core.Range;
import org.opencv.test.OpenCVTestCase;

public class RangeTest extends OpenCVTestCase {

    Range r1;
    Range r2;
    Range range;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        range = new Range();
        r1 = new Range(1, 11);
        r2 = new Range(1, 1);
    }

    public void testAll() {
        range = Range.all();
        assertEquals(Integer.MIN_VALUE, range.start);
        assertEquals(Integer.MAX_VALUE, range.end);
    }

    public void testClone() {
        Range dstRange = new Range();
        dstRange = r1.clone();
        assertEquals(r1, dstRange);
    }

    public void testEmpty() {
        boolean flag;

        flag = r1.empty();
        assertFalse(flag);

        flag = r2.empty();
        assertTrue(flag);
    }

    public void testEqualsObject() {
        assertFalse(r2.equals(r1));

        range = r1.clone();
        assertTrue(r1.equals(range));
    }

    public void testHashCode() {
        assertEquals(r1.hashCode(), r1.hashCode());
    }

    public void testIntersection() {
        range = r1.intersection(r2);
        assertEquals(r2, range);
    }

    public void testRange() {
        range = new Range();

        assertNotNull(range);
        assertEquals(0, range.start);
        assertEquals(0, range.end);
    }

    public void testRangeDoubleArray() {
        double[] vals = { 2, 4 };
        Range r = new Range(vals);

        assertTrue(2 == r.start);
        assertTrue(4 == r.end);
    }

    public void testRangeIntInt() {
        r1 = new Range(12, 13);

        assertNotNull(r1);
        assertEquals(12, r1.start);
        assertEquals(13, r1.end);
    }

    public void testSet() {
        double[] vals1 = {};
        r1.set(vals1);
        assertEquals(0, r1.start);
        assertEquals(0, r1.end);

        double[] vals2 = { 6, 10 };
        r2.set(vals2);
        assertEquals(6, r2.start);
        assertEquals(10, r2.end);
    }

    public void testShift() {
        int delta = 1;
        range = range.shift(delta);
        assertEquals(r2, range);
    }

    public void testSize() {
        assertEquals(10, r1.size());

        assertEquals(0, r2.size());
    }

    public void testToString() {
        String actual = r1.toString();
        String expected = "[1, 11)";
        assertEquals(expected, actual);
    }

}
