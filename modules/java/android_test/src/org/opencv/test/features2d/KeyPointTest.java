package org.opencv.test.features2d;

import org.opencv.core.Point;
import org.opencv.features2d.KeyPoint;
import org.opencv.test.OpenCVTestCase;

public class KeyPointTest extends OpenCVTestCase {

    private KeyPoint keyPoint;
    private float size;
    private float x;
    private float y;
    private float angle;
    private float response;
    private int octave;
    private int classId;

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

    public void testGet_angle() {
        keyPoint = new KeyPoint(x, y, size, angle);
        assertEquals(30.0f, keyPoint.angle);
    }

    public void testGet_class_id() {
        keyPoint = new KeyPoint(x, y, size, angle, response, octave, classId);
        assertEquals(1, keyPoint.class_id);
    }

    public void testGet_octave() {
        keyPoint = new KeyPoint(x, y, size, angle, response, octave);
        assertEquals(1, keyPoint.octave);
    }

    public void testGet_pt() {
        keyPoint = new KeyPoint(x, y, size);
        assertPointEquals(new Point(1, 2), keyPoint.pt, EPS);
    }

    public void testGet_response() {
        keyPoint = new KeyPoint(x, y, size, angle, response);
        assertEquals(2.0f, keyPoint.response);
    }

    public void testGet_size() {
        keyPoint = new KeyPoint(x, y, size);
        assertEquals(3.0f, keyPoint.size);
    }

    public void testKeyPoint() {
        keyPoint = new KeyPoint();
    }

    public void testKeyPointFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size);
    }

    public void testKeyPointFloatFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size, 10.0f);
    }

    public void testKeyPointFloatFloatFloatFloatFloat() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f);
    }

    public void testKeyPointFloatFloatFloatFloatFloatInt() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1);
    }

    public void testKeyPointFloatFloatFloatFloatFloatIntInt() {
        keyPoint = new KeyPoint(x, y, size, 1.0f, 1.0f, 1, 1);
    }

    public void testSet_angle() {
        keyPoint = new KeyPoint(x, y, size, angle);
        keyPoint.angle = 10f;
    }

    public void testSet_class_id() {
        keyPoint = new KeyPoint(x, y, size, angle, response, octave, classId);
        keyPoint.class_id = 2;
    }

    public void testSet_octave() {
        keyPoint = new KeyPoint(x, y, size, angle, response, octave);
        keyPoint.octave = 0;
    }

    public void testSet_pt() {
        keyPoint = new KeyPoint(x, y, size);
        keyPoint.pt = new Point(4, 3);
        assertPointEquals(new Point(4, 3), keyPoint.pt, EPS);
    }

    public void testSet_response() {
        keyPoint = new KeyPoint(x, y, size, angle, response);
        keyPoint.response = 1.5f;
    }

    public void testSet_size() {
        keyPoint = new KeyPoint(x, y, size);
        keyPoint.size = 5.0f;
    }

    public void testToString() {
        keyPoint = new KeyPoint(x, y, size, angle, response, octave, classId);

        String actual = keyPoint.toString();

        String expected = "KeyPoint [pt={1.0, 2.0}, size=3.0, angle=30.0, response=2.0, octave=1, class_id=1]";
        assertEquals(expected, actual);
    }

}
