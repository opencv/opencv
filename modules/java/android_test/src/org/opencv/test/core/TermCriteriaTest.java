package org.opencv.test.core;

import org.opencv.core.TermCriteria;
import org.opencv.test.OpenCVTestCase;

public class TermCriteriaTest extends OpenCVTestCase {

    private TermCriteria tc1;
    private TermCriteria tc2;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        tc1 = new TermCriteria();
        tc2 = new TermCriteria(2, 4, EPS);
    }

    public void testClone() {
        tc1 = tc2.clone();
        assertEquals(tc2, tc1);
    }

    public void testEqualsObject() {
        assertFalse(tc2.equals(tc1));

        tc1 = tc2.clone();
        assertTrue(tc2.equals(tc1));
    }

    public void testHashCode() {
        assertEquals(tc2.hashCode(), tc2.hashCode());
    }

    public void testSet() {
        double[] vals1 = {};
        tc1.set(vals1);

        assertEquals(0, tc1.type);
        assertEquals(0, tc1.maxCount);
        assertEquals(0.0, tc1.epsilon);

        double[] vals2 = { 9, 8, 0.002 };
        tc2.set(vals2);

        assertEquals(9, tc2.type);
        assertEquals(8, tc2.maxCount);
        assertEquals(0.002, tc2.epsilon);
    }

    public void testTermCriteria() {
        tc1 = new TermCriteria();

        assertNotNull(tc1);
        assertEquals(0, tc1.type);
        assertEquals(0, tc1.maxCount);
        assertEquals(0.0, tc1.epsilon);
    }

    public void testTermCriteriaDoubleArray() {
        double[] vals = { 3, 2, 0.007 };
        tc1 = new TermCriteria(vals);

        assertEquals(3, tc1.type);
        assertEquals(2, tc1.maxCount);
        assertEquals(0.007, tc1.epsilon);
    }

    public void testTermCriteriaIntIntDouble() {
        tc1 = new TermCriteria(2, 4, EPS);

        assertNotNull(tc2);
        assertEquals(2, tc2.type);
        assertEquals(4, tc2.maxCount);
        assertEquals(EPS, tc2.epsilon);
    }

    public void testToString() {
        String actual = tc2.toString();
        double eps = EPS;
        String expected = "{ type: 2, maxCount: 4, epsilon: " + eps + "}";

        assertEquals(expected, actual);
    }

}
