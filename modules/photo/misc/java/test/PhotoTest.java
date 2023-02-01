package org.opencv.test.photo;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Point;
import org.opencv.photo.Photo;
import org.opencv.test.OpenCVTestCase;
import org.opencv.imgproc.Imgproc;

public class PhotoTest extends OpenCVTestCase {

    public void testInpaint() {
        Point p = new Point(matSize / 2, matSize / 2);
        Imgproc.circle(gray255, p, 2, colorBlack, Imgproc.FILLED);
        Imgproc.circle(gray0,   p, 2, colorWhite, Imgproc.FILLED);

        Photo.inpaint(gray255, gray0, dst, 3, Photo.INPAINT_TELEA);

        assertMatEqual(getMat(CvType.CV_8U, 255), dst);
    }

}
