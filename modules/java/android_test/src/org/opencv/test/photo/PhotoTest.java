package org.opencv.test.photo;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Point;
import org.opencv.photo.Photo;
import org.opencv.test.OpenCVTestCase;

public class PhotoTest extends OpenCVTestCase {

    public void testInpaint() {
        Point p = new Point(matSize / 2, matSize / 2);
        Core.circle(gray255, p, 2, colorBlack, Core.FILLED);
        Core.circle(gray0,   p, 2, colorWhite, Core.FILLED);

        Photo.inpaint(gray255, gray0, dst, 3, Photo.INPAINT_TELEA);

        assertMatEqual(getMat(CvType.CV_8U, 255), dst);
    }

}
