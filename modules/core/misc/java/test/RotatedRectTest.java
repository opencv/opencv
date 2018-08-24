package org.opencv.test.core;

import org.opencv.core.CvType;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.MatOfRotatedRect;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;

import java.util.Arrays;
import java.util.List;

public class RotatedRectTest extends OpenCVTestCase {

    private double angle;
    private Point center;
    private Size size;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        center = new Point(matSize / 2, matSize / 2);
        size = new Size(matSize / 4, matSize / 2);
        angle = 40;
    }

    public void testBoundingRect() {
        size = new Size(matSize / 2, matSize / 2);
        assertEquals(size.height, size.width);
        double length = size.height;

        angle = 45;
        RotatedRect rr = new RotatedRect(center, size, angle);

        Rect r = rr.boundingRect();
        double halfDiagonal = length * Math.sqrt(2) / 2;

        assertTrue((r.x == Math.floor(center.x - halfDiagonal)) && (r.y == Math.floor(center.y - halfDiagonal)));

        assertTrue((r.br().x >= Math.ceil(center.x + halfDiagonal)) && (r.br().y >= Math.ceil(center.y + halfDiagonal)));

        assertTrue((r.br().x - Math.ceil(center.x + halfDiagonal)) <= 1 && (r.br().y - Math.ceil(center.y + halfDiagonal)) <= 1);
    }

    public void testClone() {
        RotatedRect rrect = new RotatedRect(center, size, angle);
        RotatedRect clone = rrect.clone();

        assertTrue(clone != null);
        assertTrue(rrect.center.equals(clone.center));
        assertTrue(rrect.size.equals(clone.size));
        assertTrue(rrect.angle == clone.angle);
    }

    public void testEqualsObject() {
        Point center2 = new Point(matSize / 3, matSize / 1.5);
        Size size2 = new Size(matSize / 2, matSize / 4);
        double angle2 = 0;

        RotatedRect rrect1 = new RotatedRect(center, size, angle);
        RotatedRect rrect2 = new RotatedRect(center2, size2, angle2);
        RotatedRect rrect3 = rrect1;
        RotatedRect clone1 = rrect1.clone();
        RotatedRect clone2 = rrect2.clone();

        assertTrue(rrect1.equals(rrect3));
        assertTrue(!rrect1.equals(rrect2));

        assertTrue(rrect2.equals(clone2));
        clone2.angle = 10;
        assertTrue(!rrect2.equals(clone2));

        assertTrue(rrect1.equals(clone1));

        clone1.center.x += 1;
        assertTrue(!rrect1.equals(clone1));

        clone1.center.x -= 1;
        assertTrue(rrect1.equals(clone1));

        clone1.size.width += 1;
        assertTrue(!rrect1.equals(clone1));

        assertTrue(!rrect1.equals(size));
    }

    public void testHashCode() {
        RotatedRect rr = new RotatedRect(center, size, angle);
        assertEquals(rr.hashCode(), rr.hashCode());
    }

    public void testPoints() {
        RotatedRect rrect = new RotatedRect(center, size, angle);

        Point p[] = new Point[4];
        rrect.points(p);

        boolean is_p0_irrational = (100 * p[0].x != (int) (100 * p[0].x)) && (100 * p[0].y != (int) (100 * p[0].y));
        boolean is_p1_irrational = (100 * p[1].x != (int) (100 * p[1].x)) && (100 * p[1].y != (int) (100 * p[1].y));
        boolean is_p2_irrational = (100 * p[2].x != (int) (100 * p[2].x)) && (100 * p[2].y != (int) (100 * p[2].y));
        boolean is_p3_irrational = (100 * p[3].x != (int) (100 * p[3].x)) && (100 * p[3].y != (int) (100 * p[3].y));

        assertTrue(is_p0_irrational && is_p1_irrational && is_p2_irrational && is_p3_irrational);

        assertTrue("Symmetric points 0 and 2",
                Math.abs((p[0].x + p[2].x) / 2 - center.x) + Math.abs((p[0].y + p[2].y) / 2 - center.y) < EPS);

        assertTrue("Symmetric points 1 and 3",
                Math.abs((p[1].x + p[3].x) / 2 - center.x) + Math.abs((p[1].y + p[3].y) / 2 - center.y) < EPS);

        assertTrue("Orthogonal vectors 01 and 12",
                Math.abs((p[1].x - p[0].x) * (p[2].x - p[1].x) +
                        (p[1].y - p[0].y) * (p[2].y - p[1].y)) < EPS);

        assertTrue("Orthogonal vectors 12 and 23",
                Math.abs((p[2].x - p[1].x) * (p[3].x - p[2].x) +
                        (p[2].y - p[1].y) * (p[3].y - p[2].y)) < EPS);

        assertTrue("Orthogonal vectors 23 and 30",
                Math.abs((p[3].x - p[2].x) * (p[0].x - p[3].x) +
                        (p[3].y - p[2].y) * (p[0].y - p[3].y)) < EPS);

        assertTrue("Orthogonal vectors 30 and 01",
                Math.abs((p[0].x - p[3].x) * (p[1].x - p[0].x) +
                        (p[0].y - p[3].y) * (p[1].y - p[0].y)) < EPS);

        assertTrue("Length of the vector 01",
                Math.abs((p[1].x - p[0].x) * (p[1].x - p[0].x) +
                        (p[1].y - p[0].y) * (p[1].y - p[0].y) - size.height * size.height) < EPS);

        assertTrue("Length of the vector 21",
                Math.abs((p[1].x - p[2].x) * (p[1].x - p[2].x) +
                        (p[1].y - p[2].y) * (p[1].y - p[2].y) - size.width * size.width) < EPS);

        assertTrue("Angle of the vector 21 with the axes", Math.abs((p[2].x - p[1].x) / size.width - Math.cos(angle * Math.PI / 180)) < EPS);
    }

    public void testRotatedRect() {
        RotatedRect rr = new RotatedRect();

        assertTrue(rr != null);
        assertTrue(rr.center != null);
        assertTrue(rr.size != null);
        assertTrue(rr.angle == 0.0);
    }

    public void testRotatedRectDoubleArray() {
        double[] vals = { 1.5, 2.6, 3.7, 4.2, 5.1 };
        RotatedRect rr = new RotatedRect(vals);

        assertNotNull(rr);
        assertEquals(1.5, rr.center.x);
        assertEquals(2.6, rr.center.y);
        assertEquals(3.7, rr.size.width);
        assertEquals(4.2, rr.size.height);
        assertEquals(5.1, rr.angle);
    }

    public void testRotatedRectPointSizeDouble() {
        RotatedRect rr = new RotatedRect(center, size, 40);

        assertTrue(rr != null);
        assertTrue(rr.center != null);
        assertTrue(rr.size != null);
        assertTrue(rr.angle == 40.0);
    }

    public void testSet() {
        double[] vals1 = {};
        RotatedRect r1 = new RotatedRect(center, size, 40);

        r1.set(vals1);

        assertEquals(0., r1.angle);
        assertPointEquals(new Point(0, 0), r1.center, EPS);
        assertSizeEquals(new Size(0, 0), r1.size, EPS);

        double[] vals2 = { 1, 2, 3, 4, 5 };
        RotatedRect r2 = new RotatedRect(center, size, 40);

        r2.set(vals2);

        assertEquals(5., r2.angle);
        assertPointEquals(new Point(1, 2), r2.center, EPS);
        assertSizeEquals(new Size(3, 4), r2.size, EPS);
    }

    public void testToString() {
        String actual = new RotatedRect(new Point(1, 2), new Size(10, 12), 4.5).toString();
        String expected = "{ {1.0, 2.0} 10x12 * 4.5 }";
        assertEquals(expected, actual);
    }

    public void testMatOfRotatedRect() {
        RotatedRect a = new RotatedRect(new Point(1,2),new Size(3,4),5.678);
        RotatedRect b = new RotatedRect(new Point(9,8),new Size(7,6),5.432);
        MatOfRotatedRect m = new MatOfRotatedRect(a,b,a,b,a,b,a,b);
        assertEquals(m.rows(), 8);
        assertEquals(m.cols(), 1);
        assertEquals(m.type(), CvType.CV_32FC(5));
        RotatedRect[] arr = m.toArray();
        assertEquals(arr[2].angle, a.angle, EPS);
        assertEquals(arr[3].center.x, b.center.x);
        assertEquals(arr[3].size.width, b.size.width);
        List<RotatedRect> li = m.toList();
        assertEquals(li.size(), 8);
        RotatedRect rr = li.get(7);
        assertEquals(rr.angle, b.angle, EPS);
        assertEquals(rr.center.y, b.center.y);
    }
}
