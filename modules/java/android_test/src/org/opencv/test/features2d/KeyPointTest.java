package org.opencv.test.features2d;

import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;

public class KeyPointTest extends OpenCVTestCase {

    private KeyPoint keyPoint;
    private float size;
    private float x;
    private float y;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        keyPoint = null;
        x = 1.0f;
        y = 2.0f;
        size = 3.0f;
    }

    public void testGet_angle() {
        fail("Not yet implemented");
    }

    public void testGet_class_id() {
        fail("Not yet implemented");
    }

    public void testGet_octave() {
        fail("Not yet implemented");
    }

    public void testGet_pt() {
        fail("Not yet implemented");
    }

    public void testGet_response() {
        fail("Not yet implemented");
    }

    public void testGet_size() {
        fail("Not yet implemented");
    }

    public void testKeyPoint() {
        keyPoint = new KeyPoint();
        assertTrue(null != keyPoint);
    }

    public void testKeyPointFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size);
        assertTrue(null != keyPoint);
    }

    public void testKeyPointFloatFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size, 10.0f);
        assertTrue(null != keyPoint);
    }

    public void testKeyPointFloatFloatFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f);
        assertTrue(null != keyPoint);
    }

    public void testKeyPointFloatFloatFloatFloatFloatInt() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1);
        assertTrue(null != keyPoint);
    }

    public void testKeyPointFloatFloatFloatFloatFloatIntInt() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1, 1);
        assertTrue(null != keyPoint);
    }

    public void testSet_angle() {
        fail("Not yet implemented");
    }

    public void testSet_class_id() {
        fail("Not yet implemented");
    }

    public void testSet_octave() {
        fail("Not yet implemented");
    }

    public void testSet_pt() {
        fail("Not yet implemented");
    }

    public void testSet_response() {
        fail("Not yet implemented");
    }

    public void testSet_size() {
        fail("Not yet implemented");
    }

    public void testToString() {
        fail("Not yet implemented");
    }

}
