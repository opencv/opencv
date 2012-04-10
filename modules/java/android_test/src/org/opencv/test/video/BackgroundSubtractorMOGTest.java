package org.opencv.test.video;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.test.OpenCVTestCase;
import org.opencv.video.BackgroundSubtractorMOG;

public class BackgroundSubtractorMOGTest extends OpenCVTestCase {

    public void testApplyMatMat() {
        fail("Not yet implemented");
        /*
        BackgroundSubtractorMOG backGroundSubtract = new BackgroundSubtractorMOG();

        Point bottomRight = new Point(rgbLena.cols() / 2, rgbLena.rows() / 2);
        Point topLeft = new Point(0, 0);
        Scalar color = new Scalar(128);
        Mat mask = new Mat(rgbLena.size(), CvType.CV_16UC3, new Scalar(1));

        Core.rectangle(rgbLena, bottomRight, topLeft, color, Core.FILLED);

        backGroundSubtract.apply(rgbLena, mask);

        Mat truth = new Mat(rgbLena.size(), rgbLena.type(), new Scalar(0));
        Core.rectangle(truth, bottomRight, topLeft, color, Core.FILLED);
        assertMatEqual(truth, rgbLena);
        */
    }

    public void testApplyMatMatDouble() {
        fail("Not yet implemented");
    }

    public void testBackgroundSubtractorMOG() {
        fail("Not yet implemented");
    }

    public void testBackgroundSubtractorMOGIntIntDouble() {
        fail("Not yet implemented");
    }

    public void testBackgroundSubtractorMOGIntIntDoubleDouble() {
        fail("Not yet implemented");
    }

}
