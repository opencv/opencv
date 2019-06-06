package org.opencv.test.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.opencv.core.Range;
import org.opencv.test.OpenCVTestCase;

public class RangeTest extends OpenCVTestCase {

    Range r1;
    Range r2;
    Range range;

    @Override
    public void setUp() throws Exception {
        super.setUp();

        range = new Range();
        r1 = new Range(1, 11);
        r2 = new Range(1, 1);
    }

    @Test
    public void testAll() {
        range = Range.all();
        assertEquals(Integer.MIN_VALUE, range.start);
        assertEquals(Integer.MAX_VALUE, range.end);
    }

    @Test
    public void testClone() {
        Range dstRange = new Range();
        dstRange = r1.clone();
        assertEquals(r1, dstRange);
    }

    @Test
    public void testEmpty() {
        boolean flag;

        flag = r1.empty();
        assertFalse(flag);

        flag = r2.empty();
        assertTrue(flag);
    }

    @Test
    public void testEqualsObject() {
        assertFalse(r2.equals(r1));

        range = r1.clone();
        assertTrue(r1.equals(range));
    }

    @Test
    public void testHashCode() {
        assertEquals(r1.hashCode(), r1.hashCode());
    }

    @Test
    public void testIntersection() {
        range = r1.intersection(r2);
        assertEquals(r2, range);
    }

    @Test
    public void testRange() {
        range = new Range();

        assertNotNull(range);
        assertEquals(0, range.start);
        assertEquals(0, range.end);
    }

    @Test
    public void testRangeDoubleArray() {
        double[] vals = { 2, 4 };
        Range r = new Range(vals);

        assertTrue(2 == r.start);
        assertTrue(4 == r.end);
    }

    @Test
    public void testRangeIntInt() {
        r1 = new Range(12, 13);

        assertNotNull(r1);
        assertEquals(12, r1.start);
        assertEquals(13, r1.end);
    }

    @Test
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

    @Test
    public void testShift() {
        int delta = 1;
        range = range.shift(delta);
        assertEquals(r2, range);
    }

    @Test
    public void testSize() {
        assertEquals(10, r1.size());

        assertEquals(0, r2.size());
    }

    @Test
    public void testToString() {
        String actual = r1.toString();
        String expected = "[1, 11)";
        assertEquals(expected, actual);
    }

}
