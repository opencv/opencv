package org.opencv.test.core;

import org.opencv.core.Point;
import org.opencv.core.KeyPoint;
import org.opencv.test.OpenCVTestCase;

public class KeyPointTest extends OpenCVTestCase {

    private float angle;
    private int classId;
    private KeyPoint keyPoint;
    private int octave;
    private float response;
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
        angle = 30.0f;
        response = 2.0f;
        octave = 1;
        classId = 1;
    }

    public void testKeyPoint() {
        keyPoint = new KeyPoint();
        assertPointEquals(new Point(0, 0), keyPoint.pt, EPS);
    }

    public void testKeyPointFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size);
        assertPointEquals(new Point(1, 2), keyPoint.pt, EPS);
    }

    public void testKeyPointFloatFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size, 10.0f);
        assertEquals(10.0f, keyPoint.angle);
    }

    public void testKeyPointFloatFloatFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f);
        assertEquals(1.0f, keyPoint.response);
    }

    public void testKeyPointFloatFloatFloatFloatFloatInt() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1);
        assertEquals(1, keyPoint.octave);
    }

    public void testKeyPointFloatFloatFloatFloatFloatIntInt() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1, 1);
        assertEquals(1, keyPoint.class_id);
    }

    public void testToString() {
        keyPoint = new KeyPoint(x, y, size, angle, response, octave, classId);

        String actual = keyPoint.toString();

        String expected = "KeyPoint [pt={1.0, 2.0}, size=3.0, angle=30.0, response=2.0, octave=1, class_id=1]";
        assertEquals(expected, actual);
    }

}
