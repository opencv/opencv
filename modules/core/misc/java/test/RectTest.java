package org.opencv.test.core;

import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;

public class RectTest extends OpenCVTestCase {

    private Rect r;
    private Rect rect;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        r = new Rect();
        rect = new Rect(0, 0, 10, 10);
    }

    public void testArea() {
        double area;
        area = rect.area();
        assertEquals(100.0, area);
    }

    public void testBr() {
        Point p_br = new Point();
        p_br = rect.br();
        Point truth = new Point(10, 10);
        assertEquals(truth, p_br);
    }

    public void testClone() {
        r = rect.clone();
        assertEquals(rect, r);
    }

    public void testContains() {
        Rect rect = new Rect(0, 0, 10, 10);

        Point p_inner = new Point(5, 5);
        Point p_outer = new Point(5, 55);
        Point p_bl = new Point(0, 0);
        Point p_br = new Point(10, 0);
        Point p_tl = new Point(0, 10);
        Point p_tr = new Point(10, 10);

        assertTrue(rect.contains(p_inner));
        assertTrue(rect.contains(p_bl));

        assertFalse(rect.contains(p_outer));
        assertFalse(rect.contains(p_br));
        assertFalse(rect.contains(p_tl));
        assertFalse(rect.contains(p_tr));
    }

    public void testEqualsObject() {
        boolean flag;
        flag = rect.equals(r);
        assertFalse(flag);

        r = rect.clone();
        flag = rect.equals(r);
        assertTrue(flag);
    }

    public void testHashCode() {
        assertEquals(rect.hashCode(), rect.hashCode());
    }

    public void testRect() {
        r = new Rect();

        assertEquals(0, r.x);
        assertEquals(0, r.y);
        assertEquals(0, r.width);
        assertEquals(0, r.height);
    }

    public void testRectDoubleArray() {
        double[] vals = { 1, 3, 5, 2 };
        r = new Rect(vals);

        assertEquals(1, r.x);
        assertEquals(3, r.y);
        assertEquals(5, r.width);
        assertEquals(2, r.height);
    }

    public void testRectIntIntIntInt() {
        r = new Rect(1, 3, 5, 2);

        assertNotNull(rect);
        assertEquals(0, rect.x);
        assertEquals(0, rect.y);
        assertEquals(10, rect.width);
        assertEquals(10, rect.height);
    }

    public void testRectPointPoint() {
        Point p1 = new Point(4, 4);
        Point p2 = new Point(2, 3);

        r = new Rect(p1, p2);
        assertNotNull(r);
        assertEquals(2, r.x);
        assertEquals(3, r.y);
        assertEquals(2, r.width);
        assertEquals(1, r.height);
    }

    public void testRectPointSize() {
        Point p1 = new Point(4, 4);
        Size sz = new Size(3, 1);
        r = new Rect(p1, sz);

        assertEquals(4, r.x);
        assertEquals(4, r.y);
        assertEquals(3, r.width);
        assertEquals(1, r.height);
    }

    public void testSet() {
        double[] vals1 = {};
        Rect r1 = new Rect(vals1);

        assertEquals(0, r1.x);
        assertEquals(0, r1.y);
        assertEquals(0, r1.width);
        assertEquals(0, r1.height);

        double[] vals2 = { 2, 2, 10, 5 };
        r = new Rect(vals2);

        assertEquals(2, r.x);
        assertEquals(2, r.y);
        assertEquals(10, r.width);
        assertEquals(5, r.height);
    }

    public void testSize() {
        Size s1 = new Size(0, 0);
        assertEquals(s1, r.size());

        Size s2 = new Size(10, 10);
        assertEquals(s2, rect.size());
    }

    public void testTl() {
        Point p_tl = new Point();
        p_tl = rect.tl();
        Point truth = new Point(0, 0);
        assertEquals(truth, p_tl);
    }

    public void testToString() {
        String actual = rect.toString();
        String expected = "{0, 0, 10x10}";
        assertEquals(expected, actual);
    }

}
