package org.opencv.test.objdetect;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.objdetect.Objdetect;
import org.opencv.test.OpenCVTestCase;
import org.opencv.imgproc.Imgproc;

public class ObjdetectTest extends OpenCVTestCase {

    Size size;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        size = new Size(3, 3);
    }

    public void testFindChessboardCornersMatSizeMat() {
        Size patternSize = new Size(9, 6);
        MatOfPoint2f corners = new MatOfPoint2f();
        Objdetect.findChessboardCorners(grayChess, patternSize, corners);
        assertFalse(corners.empty());
    }

    public void testFindChessboardCornersMatSizeMatInt() {
        Size patternSize = new Size(9, 6);
        MatOfPoint2f corners = new MatOfPoint2f();
        Objdetect.findChessboardCorners(grayChess, patternSize, corners, Objdetect.CALIB_CB_ADAPTIVE_THRESH + Objdetect.CALIB_CB_NORMALIZE_IMAGE
                + Objdetect.CALIB_CB_FAST_CHECK);
        assertFalse(corners.empty());
    }

    public void testFind4QuadCornerSubpix() {
        Size patternSize = new Size(9, 6);
        MatOfPoint2f corners = new MatOfPoint2f();
        Size region_size = new Size(5, 5);
        Objdetect.findChessboardCorners(grayChess, patternSize, corners);
        Objdetect.find4QuadCornerSubpix(grayChess, corners, region_size);
        assertFalse(corners.empty());
    }

    public void testFindCirclesGridMatSizeMat() {
        int size = 300;
        Mat img = new Mat(size, size, CvType.CV_8U);
        img.setTo(new Scalar(255));
        Mat centers = new Mat();

        assertFalse(Objdetect.findCirclesGrid(img, new Size(5, 5), centers));

        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) {
                Point pt = new Point(size * (2 * i + 1) / 10, size * (2 * j + 1) / 10);
                Imgproc.circle(img, pt, 10, new Scalar(0), -1);
            }

        assertTrue(Objdetect.findCirclesGrid(img, new Size(5, 5), centers));

        assertEquals(1, centers.rows());
        assertEquals(25, centers.cols());
        assertEquals(CvType.CV_32FC2, centers.type());
    }

    public void testFindCirclesGridMatSizeMatInt() {
        int size = 300;
        Mat img = new Mat(size, size, CvType.CV_8U);
        img.setTo(new Scalar(255));
        Mat centers = new Mat();

        assertFalse(Objdetect.findCirclesGrid(img, new Size(3, 5), centers, Objdetect.CALIB_CB_CLUSTERING
                | Objdetect.CALIB_CB_ASYMMETRIC_GRID));

        int step = size * 2 / 15;
        int offsetx = size / 6;
        int offsety = (size - 4 * step) / 2;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 5; j++) {
                Point pt = new Point(offsetx + (2 * i + j % 2) * step, offsety + step * j);
                Imgproc.circle(img, pt, 10, new Scalar(0), -1);
            }

        assertTrue(Objdetect.findCirclesGrid(img, new Size(3, 5), centers, Objdetect.CALIB_CB_CLUSTERING
                | Objdetect.CALIB_CB_ASYMMETRIC_GRID));

        assertEquals(1, centers.rows());
        assertEquals(15, centers.cols());
        assertEquals(CvType.CV_32FC2, centers.type());
    }
}
