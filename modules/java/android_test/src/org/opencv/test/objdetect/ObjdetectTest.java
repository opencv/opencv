package org.opencv.test.objdetect;

import java.util.ArrayList;

import org.opencv.core.Rect;
import org.opencv.objdetect.Objdetect;
import org.opencv.test.OpenCVTestCase;

public class ObjdetectTest extends OpenCVTestCase {

    public void testGroupRectanglesListOfRectListOfIntegerInt() {
        fail("Not yet implemented");
        Rect r = new Rect(10, 10, 20, 20);
        ArrayList<Rect> rects = new ArrayList<Rect>();

        for (int i = 0; i < 10; i++)
            rects.add(r);

        int groupThreshold = 1;
        Objdetect.groupRectangles(rects, null, groupThreshold);//TODO: second parameter should not be null
        assertEquals(1, rects.size());
    }

    public void testGroupRectanglesListOfRectListOfIntegerIntDouble() {
        fail("Not yet implemented");
        Rect r1 = new Rect(10, 10, 20, 20);
        Rect r2 = new Rect(10, 10, 25, 25);
        ArrayList<Rect> rects = new ArrayList<Rect>();

        for (int i = 0; i < 10; i++)
            rects.add(r1);
        for (int i = 0; i < 10; i++)
            rects.add(r2);

        int groupThreshold = 1;
        double eps = 0.2;
        Objdetect.groupRectangles(rects, null, groupThreshold, eps);//TODO: second parameter should not be null
        assertEquals(2, rects.size());
    }
}
