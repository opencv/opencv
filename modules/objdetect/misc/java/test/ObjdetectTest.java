package org.opencv.test.objdetect;

import org.opencv.test.OpenCVTestCase;

public class ObjdetectTest extends OpenCVTestCase {

    public void testGroupRectanglesListOfRectListOfIntegerInt() {
        fail("Not yet implemented");
        /*
        final int NUM = 10;
        MatOfRect rects = new MatOfRect();
        rects.alloc(NUM);

        for (int i = 0; i < NUM; i++)
            rects.put(i, 0, 10, 10, 20, 20);

        int groupThreshold = 1;
        Objdetect.groupRectangles(rects, null, groupThreshold);//TODO: second parameter should not be null
        assertEquals(1, rects.total());
        */
    }

    public void testGroupRectanglesListOfRectListOfIntegerIntDouble() {
        fail("Not yet implemented");
        /*
        final int NUM = 10;
        MatOfRect rects = new MatOfRect();
        rects.alloc(NUM);

        for (int i = 0; i < NUM; i++)
            rects.put(i, 0, 10, 10, 20, 20);

        for (int i = 0; i < NUM; i++)
            rects.put(i, 0, 10, 10, 25, 25);

        int groupThreshold = 1;
        double eps = 0.2;
        Objdetect.groupRectangles(rects, null, groupThreshold, eps);//TODO: second parameter should not be null
        assertEquals(2, rects.size());
        */
    }
}
