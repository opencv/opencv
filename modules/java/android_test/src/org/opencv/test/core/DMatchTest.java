package org.opencv.test.core;

import org.opencv.core.DMatch;

import junit.framework.TestCase;

public class DMatchTest extends TestCase {
    public void testDMatch() {
        new DMatch();
    }

    public void testDMatchIntIntFloat() {
        DMatch dm1 = new DMatch(1, 4, 4.0f);

        assertEquals(1, dm1.queryIdx);
        assertEquals(4, dm1.trainIdx);
        assertEquals(4.0f, dm1.distance);
    }

    public void testDMatchIntIntIntFloat() {
        DMatch dm2 = new DMatch(2, 6, -1, 8.0f);

        assertEquals(2, dm2.queryIdx);
        assertEquals(6, dm2.trainIdx);
        assertEquals(-1, dm2.imgIdx);
        assertEquals(8.0f, dm2.distance);
    }

    public void testLessThan() {
        DMatch dm1 = new DMatch(1, 4, 4.0f);
        DMatch dm2 = new DMatch(2, 6, -1, 8.0f);
        assertTrue(dm1.lessThan(dm2));
    }

    public void testToString() {
        DMatch dm2 = new DMatch(2, 6, -1, 8.0f);

        String actual = dm2.toString();

        String expected = "DMatch [queryIdx=2, trainIdx=6, imgIdx=-1, distance=8.0]";
        assertEquals(expected, actual);
    }

}
