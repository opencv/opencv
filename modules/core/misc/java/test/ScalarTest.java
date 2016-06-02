package org.opencv.test.core;

import org.opencv.core.Scalar;
import org.opencv.test.OpenCVTestCase;

public class ScalarTest extends OpenCVTestCase {

    private Scalar dstScalar;
    private Scalar s1;
    private Scalar s2;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        s1 = new Scalar(1.0);
        s2 = Scalar.all(1.0);
        dstScalar = null;
    }

    public void testAll() {
        dstScalar = Scalar.all(2.0);
        Scalar truth = new Scalar(2.0, 2.0, 2.0, 2.0);
        assertEquals(truth, dstScalar);
    }

    public void testClone() {
        dstScalar = s2.clone();
        assertEquals(s2, dstScalar);
    }

    public void testConj() {
        dstScalar = s2.conj();
        Scalar truth = new Scalar(1, -1, -1, -1);
        assertEquals(truth, dstScalar);
    }

    public void testEqualsObject() {
        dstScalar = s2.clone();
        assertTrue(s2.equals(dstScalar));

        assertFalse(s2.equals(s1));
    }

    public void testHashCode() {
        assertEquals(s2.hashCode(), s2.hashCode());
    }

    public void testIsReal() {
        assertTrue(s1.isReal());

        assertFalse(s2.isReal());
    }

    public void testMulScalar() {
        dstScalar = s2.mul(s1);
        assertEquals(s1, dstScalar);
    }

    public void testMulScalarDouble() {
        double multiplier = 2.0;
        dstScalar = s2.mul(s1, multiplier);
        Scalar truth = new Scalar(2);
        assertEquals(truth, dstScalar);
    }

    public void testScalarDouble() {
        Scalar truth = new Scalar(1);
        assertEquals(truth, s1);
    }

    public void testScalarDoubleArray() {
        double[] vals = { 2.0, 4.0, 5.0, 3.0 };
        dstScalar = new Scalar(vals);

        Scalar truth = new Scalar(2.0, 4.0, 5.0, 3.0);
        assertEquals(truth, dstScalar);
    }

    public void testScalarDoubleDouble() {
        dstScalar = new Scalar(2, 5);
        Scalar truth = new Scalar(2.0, 5.0, 0.0, 0.0);
        assertEquals(truth, dstScalar);
    }

    public void testScalarDoubleDoubleDouble() {
        dstScalar = new Scalar(2.0, 5.0, 5.0);
        Scalar truth = new Scalar(2.0, 5.0, 5.0, 0.0);
        assertEquals(truth, dstScalar);
    }

    public void testScalarDoubleDoubleDoubleDouble() {
        dstScalar = new Scalar(2.0, 5.0, 5.0, 9.0);
        Scalar truth = new Scalar(2.0, 5.0, 5.0, 9.0);
        assertEquals(truth, dstScalar);
    }

    public void testSet() {
        double[] vals = { 1.0, 1.0, 1.0, 1.0 };
        s1.set(vals);
        assertEquals(s2, s1);
    }

    public void testToString() {
        String actual = s2.toString();
        String expected = "[1.0, 1.0, 1.0, 1.0]";
        assertEquals(expected, actual);
    }

}
