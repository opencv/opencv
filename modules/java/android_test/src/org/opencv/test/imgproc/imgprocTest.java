package org.opencv.test.imgproc;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.test.OpenCVTestCase;

public class imgprocTest extends OpenCVTestCase {

    private Mat gray_64f_2;
    private Mat dst64F;
    private Mat mask;
    private int imgprocSz;
    Point anchorPoint;
    Size size;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        imgprocSz = 2;

        gray_64f_2 = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(2));
        dst64F = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(0));
        mask = new Mat(imgprocSz, imgprocSz, CvType.CV_8U, new Scalar(1));
        anchorPoint = new Point(2, 2);
        size = new Size(3, 3);
    }

    public void test_1() {
        super.test_1("imgproc");
    }

    public void testAccumulateMatMat() {
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(2));
        Imgproc.accumulate(gray_64f_2, dst64F);
        assertMatEqual(truth, dst64F, EPS);

        dst = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(0));
        Imgproc.accumulate(gray1_32f, dst);
        assertMatEqual(gray1_32f, dst, EPS);
    }

    public void testAccumulateMatMatMat() {
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(2));

        Imgproc.accumulate(gray_64f_2, dst64F, mask); // TODO: use better mask
        assertMatEqual(truth, dst64F, EPS);
    }

    public void testAccumulateProductMatMatMat() {
        Mat src1 = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(1));

        Mat src2 = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        src2.put(0, 0, 2, 1);
        src2.put(1, 0, 1, 2);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(2));
        truth.put(0, 0, 2, 1);
        truth.put(1, 0, 1, 2);

        Mat dstImage = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(
                0));
        Imgproc.accumulateProduct(src1, src2, dstImage);
        assertMatEqual(truth, dstImage, EPS);
    }

    public void testAccumulateProductMatMatMatMat() {
        Mat src1 = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        src1.put(0, 0, 1, 1);
        src1.put(1, 0, 0, 1);

        Mat src2 = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        src2.put(0, 0, 2, 1);
        src2.put(1, 0, 1, 2);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(0));
        truth.put(0, 0, 2, 1);
        truth.put(1, 0, 0, 2);

        Imgproc.accumulateProduct(src1, src2, dst64F, mask);
        assertMatEqual(truth, dst64F, EPS);
    }

    public void testAccumulateSquareMatMat() {
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(4));

        Imgproc.accumulateSquare(gray_64f_2, dst64F);
        assertMatEqual(truth, dst64F, EPS);
    }

    public void testAccumulateSquareMatMatMat() {
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(4));

        Imgproc.accumulateSquare(gray_64f_2, dst64F, mask);
        assertMatEqual(truth, dst64F, EPS);
    }

    public void testAccumulateWeightedMatMatDouble() {
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(4));

        Imgproc.accumulateWeighted(gray_64f_2, dst64F, 2.0);
        assertMatEqual(truth, dst64F, EPS);
    }

    public void testAccumulateWeightedMatMatDoubleMat() {
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_64F, new Scalar(8));
        Imgproc.accumulateWeighted(gray_64f_2, dst64F, 4.0, mask);
        assertMatEqual(truth, dst64F, EPS);
    }

    public void testAdaptiveThreshold() {
        Imgproc.adaptiveThreshold(gray0, dst, 2.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 3, 0);
        assertMatEqual(gray0, dst);
    }

    public void testApproxPolyDP() {
        Mat curve = new Mat(1, 5, CvType.CV_32FC2);
        curve.put(0, 0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 4.0, 5.0, 3.0);

        Mat approxCurve = new Mat(3, 1, CvType.CV_32FC2);
        approxCurve.put(0, 0, 1.0, 3.0, 3.0, 5.0, 5.0, 3.0);

        Imgproc.approxPolyDP(curve, dst, EPS, true);
        assertMatEqual(approxCurve, dst, EPS);
    }

    public void testArcLength() {
        Mat curve = new Mat(1, 5, CvType.CV_32FC2);
        curve.put(0, 0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 4.0, 5.0, 3.0);

        double arcLength = Imgproc.arcLength(curve, false);
        double expectedLength = 5.656854152679443;
        assertEquals(expectedLength, arcLength);
    }

    public void testBilateralFilterMatMatIntDoubleDouble() {
        Imgproc.bilateralFilter(gray255, dst, 5, 10.0, 5.0);
        assertMatEqual(gray255, dst);
    }

    public void testBilateralFilterMatMatIntDoubleDoubleInt() {
        Imgproc.bilateralFilter(gray255, dst, 5, 10.0, 5.0,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(gray255, dst);
    }

    public void testBlurMatMatSize() {
        Imgproc.blur(gray0, dst, size);
        assertMatEqual(gray0, dst);

        Imgproc.blur(gray255, dst, size);
        assertMatEqual(gray255, dst);
    }

    public void testBlurMatMatSizePoint() {
        Imgproc.blur(gray0, dst, size, anchorPoint);
        assertMatEqual(gray0, dst);
    }

    public void testBlurMatMatSizePointInt() {
        Imgproc.blur(gray0, dst, size, anchorPoint, Imgproc.BORDER_REFLECT);
        assertMatEqual(gray0, dst);
    }

    public void testBorderInterpolate() {
        float val1 = Imgproc.borderInterpolate(100, 150,
                Imgproc.BORDER_REFLECT_101);
        assertEquals(100.0f, val1);

        float val2 = Imgproc.borderInterpolate(-5, 10, Imgproc.BORDER_WRAP);
        assertEquals(5.0f, val2);
    }

    public void testBoundingRect() {
        Mat points = new Mat(1, 4, CvType.CV_32FC2);
        points.put(0, 0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 4.0, 4.0);

        Point p1 = new Point(1, 1);
        Point p2 = new Point(-5, -2);

        Rect bbox = Imgproc.boundingRect(points);
        assertTrue(bbox.contains(p1));
        assertFalse(bbox.contains(p2));
    }

    public void testBoxFilterMatMatIntSize() {
        Size size = new Size(3, 3);
        Imgproc.boxFilter(gray0, dst, 8, size);
        assertMatEqual(gray0, dst);
    }

    public void testBoxFilterMatMatIntSizePoint() {
        Imgproc.boxFilter(gray0, dst, 8, size, anchorPoint);
        assertMatEqual(gray0, dst);
    }

    public void testBoxFilterMatMatIntSizePointBoolean() {
        Imgproc.boxFilter(gray255, dst, 8, size, anchorPoint, false);
        assertMatEqual(gray255, dst);
    }

    public void testBoxFilterMatMatIntSizePointBooleanInt() {
        Imgproc.boxFilter(gray255, dst, 8, size, anchorPoint, false,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(gray255, dst);
    }

    public void testCalcBackProject() {
        ArrayList<Mat> images = new ArrayList<Mat>();
        List<Integer> channels = new ArrayList<Integer>();
        List<Integer> histSize = new ArrayList<Integer>();
        List<Float> ranges = new ArrayList<Float>();

        images.add(grayChess);
        channels.add(0);
        histSize.add(10);
        ranges.add(0.0f);
        ranges.add(256.0f);

        Mat hist = new Mat();
        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
        Core.normalize(hist, hist);

        Imgproc.calcBackProject(images, channels, hist, dst, ranges, 255);
        assertTrue(grayChess.size().equals(dst.size()));
        assertEquals(grayChess.depth(), dst.depth());
        assertTrue(0 != Core.countNonZero(dst));
    }

    public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat() {
        ArrayList<Mat> images = new ArrayList<Mat>();
        List<Integer> channels = new ArrayList<Integer>();
        List<Integer> histSize = new ArrayList<Integer>();
        List<Float> ranges = new ArrayList<Float>();

        images.add(gray128);
        channels.add(0);
        histSize.add(10);
        ranges.add(0.0f);
        ranges.add(256.0f);

        truth = new Mat(10, 1, CvType.CV_32F, Scalar.all(0.0));
        truth.put(5, 0, 100.0);

        Mat hist = new Mat();
        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
        assertMatEqual(truth, hist, EPS);
    }

    public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat2d() {
        ArrayList<Mat> images = new ArrayList<Mat>();
        List<Integer> channels = new ArrayList<Integer>();
        List<Integer> histSize = new ArrayList<Integer>();
        List<Float> ranges = new ArrayList<Float>();

        images.add(gray255);
        images.add(gray128);

        channels.add(0);
        channels.add(1);

        histSize.add(10);
        histSize.add(10);

        ranges.add(0.0f);
        ranges.add(256.0f);
        ranges.add(0.0f);
        ranges.add(256.0f);

        truth = new Mat(10, 10, CvType.CV_32F, Scalar.all(0.0));
        truth.put(9, 5, 100.0);

        Mat hist = new Mat();
        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
        assertMatEqual(truth, hist, EPS);
    }

    public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloatBoolean() {
        ArrayList<Mat> images = new ArrayList<Mat>();
        List<Integer> channels = new ArrayList<Integer>();
        List<Integer> histSize = new ArrayList<Integer>();
        List<Float> ranges = new ArrayList<Float>();
        Mat hist = new Mat();

        images.add(gray255);
        images.add(gray128);

        channels.add(0);
        channels.add(1);

        histSize.add(10);
        histSize.add(10);

        ranges.add(0.0f);
        ranges.add(256.0f);
        ranges.add(0.0f);
        ranges.add(256.0f);

        truth = new Mat(10, 10, CvType.CV_32F, Scalar.all(0.0));
        truth.put(9, 5, 100.0);
        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges,
                true);
        assertMatEqual(truth, hist, EPS);
    }

    public void testCannyMatMatDoubleDouble() {
        Imgproc.Canny(gray255, dst, 5.0, 10.0);
        assertMatEqual(gray0, dst);
    }

    public void testCannyMatMatDoubleDoubleInt() {
        Imgproc.Canny(gray255, dst, 5.0, 10.0, 5);
        assertMatEqual(gray0, dst);
    }

    public void testCannyMatMatDoubleDoubleIntBoolean() {
        Imgproc.Canny(gray0, dst, 5.0, 10.0, 5, true);
        assertMatEqual(gray0, dst);
    }

    public void testCompareHist() {
        Mat H1 = new Mat(3, 1, CvType.CV_32F);
        Mat H2 = new Mat(3, 1, CvType.CV_32F);

        H1.put(0, 0, 1, 2, 3);
        H2.put(0, 0, 4, 5, 6);

        double comparator = Imgproc.compareHist(H1, H2, Imgproc.CV_COMP_CORREL);
        assertEquals(1.0, comparator);
    }

    public void testContourAreaMat() {
        Mat contour = new Mat(1, 4, CvType.CV_32FC2);
        contour.put(0, 0, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 5.0, 4.0);

        double area = Imgproc.contourArea(contour);
        assertEquals(45.0, area);
    }

    public void testContourAreaMatBoolean() {
        Mat contour = new Mat(1, 4, CvType.CV_32FC2);
        contour.put(0, 0, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 5.0, 4.0);

        double area = Imgproc.contourArea(contour, true);
        assertEquals(45.0, area);
    }

    public void testConvertMapsMatMatMatMatInt() {
        Mat map1 = new Mat(1, 4, CvType.CV_32FC1, new Scalar(1));
        Mat map2 = new Mat(1, 4, CvType.CV_32FC1, new Scalar(1));
        Mat dstmap1 = new Mat(1, 4, CvType.CV_16SC2);
        Mat dstmap2 = new Mat(1, 4, CvType.CV_16UC1);

        // FIXME: dstmap1 - Documentation says Cvtype but requires integer
        Imgproc.convertMaps(map1, map2, dstmap1, dstmap2, CvType.CV_32F);
        fail("Not yet implemented");
    }

    public void testConvertMapsMatMatMatMatIntBoolean() {
        fail("Not yet implemented");
    }

    public void testConvexHullMatMat() {
        Mat points = new Mat(1, 6, CvType.CV_32FC2);
        points.put(0, 0, 2.0, 0.0, 4.0, 0.0, 3.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0,
                1.0);

        Mat expHull = new Mat(4, 1, CvType.CV_32FC2);
        expHull.put(0, 0, 4, 0, 3, 2, 0, 2, 2, 0);

        Imgproc.convexHull(points, dst);
        assertMatEqual(expHull, dst, EPS);
    }

    public void testConvexHullMatMatBoolean() {
        Mat points = new Mat(1, 6, CvType.CV_32FC2);
        points.put(0, 0, 2.0, 0.0, 4.0, 0.0, 3.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0,
                1.0);

        Mat expHull = new Mat(4, 1, CvType.CV_32FC2);
        expHull.put(0, 0, 0, 2, 3, 2, 4, 0, 2, 0);

        Imgproc.convexHull(points, dst, true);
        assertMatEqual(expHull, dst, EPS);
    }

    public void testConvexHullMatMatBooleanBoolean() {
        Mat points = new Mat(1, 6, CvType.CV_32FC2);
        points.put(0, 0, 2.0, 0.0, 4.0, 0.0, 3.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0,
                1.0);

        Mat expHull = new Mat(4, 1, CvType.CV_32FC2);
        expHull.put(0, 0, 0, 2, 3, 2, 4, 0, 2, 0);

        Imgproc.convexHull(points, dst, true, true);
        assertMatEqual(expHull, dst, EPS);
    }

    public void testCopyMakeBorderMatMatIntIntIntIntInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(1));
        truth = new Mat(6, 6, CvType.CV_32F, new Scalar(1));
        int border = 2;

        Imgproc.copyMakeBorder(src, dst, border, border, border, border,
                Imgproc.BORDER_REPLICATE);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCopyMakeBorderMatMatIntIntIntIntIntScalar() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(1.0));

        truth = new Mat(6, 6, CvType.CV_32F, new Scalar(1));
        Scalar value = new Scalar(0);
        int border = 2;

        Imgproc.copyMakeBorder(src, dst, border, border, border, border,
                Imgproc.BORDER_REPLICATE, value);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerEigenValsAndVecsMatMatIntInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1);
        src.put(0, 0, 1, 2);
        src.put(1, 0, 4, 2);

        int blockSize = 3;
        int ksize = 5;

        // TODO: eigen vals and vectors returned = 0 for most src matrices
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC(6), new Scalar(0));
        Imgproc.cornerEigenValsAndVecs(src, dst, blockSize, ksize);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerEigenValsAndVecsMatMatIntIntInt() {
        Mat src = new Mat(4, 4, CvType.CV_32FC1, new Scalar(128));

        int blockSize = 3;
        int ksize = 5;

        truth = new Mat(4, 4, CvType.CV_32FC(6), new Scalar(0));

        Imgproc.cornerEigenValsAndVecs(src, dst, blockSize, ksize,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerHarrisMatMatIntIntDouble() {
        truth = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(0));
        int blockSize = 5;
        int ksize = 7;
        double k = 0.1;
        Imgproc.cornerHarris(gray128, dst, blockSize, ksize, k);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerHarrisMatMatIntIntDoubleInt() {
        truth = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(0));
        int blockSize = 5;
        int ksize = 7;
        double k = 0.1;
        Imgproc.cornerHarris(gray255, dst, blockSize, ksize, k,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerMinEigenValMatMatInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1);
        src.put(0, 0, 1, 2);
        src.put(1, 0, 2, 1);

        int blockSize = 5;

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1, new Scalar(0));
        Imgproc.cornerMinEigenVal(src, dst, blockSize);
        assertMatEqual(truth, dst, EPS);

        Mat truth1 = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(0));
        Imgproc.cornerMinEigenVal(gray255, dst, blockSize);
        assertMatEqual(truth1, dst, EPS);
    }

    public void testCornerMinEigenValMatMatIntInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32FC1);

        truth = new Mat(3, 3, CvType.CV_32FC1, new Scalar(0));
        int blockSize = 3;
        int ksize = 5;

        truth.put(0, 0, 0.055555549, 0.027777772, 0.055555549);
        truth.put(1, 0, 0.027777772, 0.055555549, 0.027777772);
        truth.put(2, 0, 0.055555549, 0.027777772, 0.055555549);

        Imgproc.cornerMinEigenVal(src, dst, blockSize, ksize);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerMinEigenValMatMatIntIntInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32FC1);

        truth = new Mat(3, 3, CvType.CV_32FC1, new Scalar(0));
        int blockSize = 3;
        int ksize = 5;

        truth.put(0, 0, 0.68055558, 0.92708349, 0.5868057);
        truth.put(1, 0, 0.92708343, 0.92708343, 0.92708343);
        truth.put(2, 0, 0.58680564, 0.92708343, 0.68055564);

        Imgproc.cornerMinEigenVal(src, dst, blockSize, ksize,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerSubPix() {
        fail("Not yet implemented");
    }

    public void testCvtColorMatMatInt() {
        Imgproc.cvtColor(rgba0, dst, 2);
        assertMatEqual(rgba0, dst);
    }

    public void testCvtColorMatMatIntInt() {
        Imgproc.cvtColor(rgba128, dst, 2, 1);
        assertMatEqual(rgba128, dst);
    }

    public void testDilateMatMatMat() {
        Mat kernel = new Mat();
        Imgproc.dilate(gray255, dst, kernel);
        assertMatEqual(gray255, dst);

        Imgproc.dilate(gray1, dst, kernel);
        assertMatEqual(gray1, dst);
    }

    public void testDilateMatMatMatPoint() {
        Mat kernel = new Mat();

        Imgproc.dilate(gray255, dst, kernel, anchorPoint);
        assertMatEqual(gray255, dst);
    }

    public void testDilateMatMatMatPointInt() {
        Mat kernel = new Mat();

        Imgproc.dilate(gray255, dst, kernel, anchorPoint, 10);
        assertMatEqual(gray255, dst);
    }

    public void testDilateMatMatMatPointIntInt() {
        Mat kernel = new Mat();

        Imgproc.dilate(gray255, dst, kernel, anchorPoint, 10,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(gray255, dst);
    }

    public void testDilateMatMatMatPointIntIntScalar() {
        Mat kernel = new Mat();
        Scalar value = new Scalar(0);

        Imgproc.dilate(gray255, dst, kernel, anchorPoint, 10,
                Imgproc.BORDER_REFLECT, value);
        assertMatEqual(gray255, dst);
    }

    public void testDistanceTransform() {
        truth = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(8192));
        Mat dstLables = new Mat(matSize, matSize, CvType.CV_32SC1,
                new Scalar(0));

        Mat labels = new Mat();
        Imgproc.distanceTransform(gray128, dst, labels, Imgproc.CV_DIST_L2, 3);
        assertMatEqual(truth, dst, EPS);
        assertMatEqual(dstLables, labels);
    }

    public void testDrawContoursMatListOfMatIntScalar() {
        List<Mat> contours = new ArrayList<Mat>(5);
        Mat hierarchy = dst;
        Core.rectangle(gray0, new Point(1, 2), new Point(7, 8), new Scalar(100));
        Imgproc.findContours(gray0, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
    	assertTrue(1 == contours.size());
        
    	assertFalse(0 == Core.countNonZero(gray0));
    	Imgproc.drawContours(gray0, contours, -1, new Scalar(0));
    	assertTrue(0 == Core.countNonZero(gray0));
    }

    public void testDrawContoursMatListOfMatIntScalarInt() {
        fail("Not yet implemented");
    }

    public void testDrawContoursMatListOfMatIntScalarIntInt() {
        fail("Not yet implemented");
    }

    public void testDrawContoursMatListOfMatIntScalarIntIntMat() {
        fail("Not yet implemented");
    }

    public void testDrawContoursMatListOfMatIntScalarIntIntMatInt() {
        fail("Not yet implemented");
    }

    public void testDrawContoursMatListOfMatIntScalarIntIntMatIntPoint() {
        fail("Not yet implemented");
    }

    public void testEqualizeHist() {
        Imgproc.equalizeHist(gray0, dst);
        assertMatEqual(gray0, dst);

        Imgproc.equalizeHist(gray255, dst);
        assertMatEqual(gray255, dst);
    }

    public void testErodeMatMatMat() {
        Mat kernel = new Mat();
        Imgproc.erode(gray128, dst, kernel);
        assertMatEqual(gray128, dst);
    }

    public void testErodeMatMatMatPoint() {
        Mat src = new Mat(3, 3, CvType.CV_8U);
        src.put(0, 0, 1, 4, 8);
        src.put(1, 0, 2, 0, 1);
        src.put(2, 0, 3, 4, 6);

        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(0.0));
        Mat kernel = new Mat();

        Imgproc.erode(src, dst, kernel, anchorPoint);
        assertMatEqual(truth, dst);
    }

    public void testErodeMatMatMatPointInt() {
        Mat src = new Mat(3, 3, CvType.CV_8U);
        src.put(0, 0, 15, 9, 10);
        src.put(1, 0, 10, 8, 12);
        src.put(2, 0, 12, 20, 25);

        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(8.0));
        Mat kernel = new Mat();

        Imgproc.erode(src, dst, kernel, anchorPoint, 10);
        assertMatEqual(truth, dst);
    }

    public void testErodeMatMatMatPointIntInt() {
        Mat src = new Mat(3, 3, CvType.CV_8U);
        src.put(0, 0, 15, 9, 10);
        src.put(1, 0, 10, 8, 12);
        src.put(2, 0, 12, 20, 25);

        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(8.0));
        Mat kernel = new Mat();

        Imgproc.erode(src, dst, kernel, anchorPoint, 10, Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst);
    }

    public void testErodeMatMatMatPointIntIntScalar() {
        Mat src = new Mat(3, 3, CvType.CV_8U);
        src.put(0, 0, 15, 9, 10);
        src.put(1, 0, 10, 8, 12);
        src.put(2, 0, 12, 20, 25);

        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(8.0));
        Mat kernel = new Mat();
        Scalar sc = new Scalar(3, 3);

        Imgproc.erode(src, dst, kernel, anchorPoint, 10,
                Imgproc.BORDER_REFLECT, sc);
        assertMatEqual(truth, dst);
    }

    public void testFilter2DMatMatIntMat() {
        Mat src = Mat.eye(4, 4, CvType.CV_32F);
        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(
                1.0));

        truth = Mat.eye(4, 4, CvType.CV_32F);
        truth.put(0, 0, 2, 2, 1, 0);
        truth.put(1, 0, 2, 2, 1, 0);
        truth.put(2, 0, 1, 1, 2, 1);
        truth.put(3, 0, 0, 0, 1, 2);

        Imgproc.filter2D(src, dst, -1, kernel);
        assertMatEqual(truth, dst, EPS);
    }

    public void testFilter2DMatMatIntMatPoint() {
        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(
                1.0));
        Point point = new Point(0, 0);

        Imgproc.filter2D(gray128, dst, -1, kernel, point);
        assertMatEqual(gray255, dst);
    }

    public void testFilter2DMatMatIntMatPointDoubleInt() {
        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(
                0.0));
        Point point = new Point(0, 0);

        Imgproc.filter2D(gray128, dst, -1, kernel, point, 2.0,
                Imgproc.BORDER_CONSTANT);
        assertMatEqual(gray2, dst);
    }

    public void testFindContoursMatListOfMatMatIntInt() {
        Mat img = new Mat(50, 50, CvType.CV_8UC1, new Scalar(0));
        List<Mat> contours = new ArrayList<Mat>(5);
        Mat hierarchy = dst;
        
        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        // no contours on empty image
        assertEquals(contours.size(), 0);
        assertEquals(contours.size(), hierarchy.total());

        Core.rectangle(img, new Point(10, 20), new Point(20, 30), new Scalar(100), 3, 16 /*CV_AA*/);
        Core.rectangle(img, new Point(30, 35), new Point(40, 45), new Scalar(200));
        
        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        // two contours of two rectangles
        assertEquals(contours.size(), 2);
        assertEquals(contours.size(), hierarchy.total());
    }

    public void testFindContoursMatListOfMatMatIntIntPoint() {
        Mat img  = new Mat(50, 50, CvType.CV_8UC1, new Scalar(0));
        Mat img2 = img.submat(5, 50, 3, 50); 
        List<Mat> contours  = new ArrayList<Mat>();
        List<Mat> contours2 = new ArrayList<Mat>();
        Mat hierarchy = dst;

        Core.rectangle(img, new Point(10, 20), new Point(20, 30), new Scalar(100), 3, 16 /*CV_AA*/);
        Core.rectangle(img, new Point(30, 35), new Point(40, 45), new Scalar(200));
        
        Imgproc.findContours(img,  contours,  hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(img2, contours2, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(3, 5));

        assertEquals(contours.size(), contours2.size());
        assertMatEqual(contours.get(0), contours2.get(0));
    }

    public void testFitEllipse() {
        Mat points = new Mat(1, 5, CvType.CV_32FC2); // TODO: use the list of
                                                     // Points
        points.put(0, 0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0);

        RotatedRect rrect = new RotatedRect();
        rrect = Imgproc.fitEllipse(points);
        assertEquals(0.0, rrect.center.x);
        assertEquals(0.0, rrect.center.y);
        assertEquals(2.0, rrect.size.width);
        assertEquals(2.0, rrect.size.height);
    }

    public void testFitLine() {
        Mat points = new Mat(1, 4, CvType.CV_32FC2);
        points.put(0, 0, 0.0, 0.0, 2.0, 3.0, 3.0, 4.0, 5.0, 8.0);

        Mat linePoints = new Mat(4, 1, CvType.CV_32FC1);
        linePoints.put(0, 0, 0.53196341, 0.84676737, 2.496531, 3.7467217);

        Imgproc.fitLine(points, dst, Imgproc.CV_DIST_L12, 0, 0.01, 0.01);
        assertMatEqual(linePoints, dst, EPS);
    }

    public void testFloodFillMatMatPointScalar() {
        Mat mask = new Mat(matSize + 2, matSize + 2, CvType.CV_8U);
        Mat img = gray0;
        
        img.setTo(new Scalar(0));
        mask.setTo(new Scalar(0));
        
        Core.circle(mask, new Point(matSize / 2 + 1, matSize / 2 + 1), 3, new Scalar(2));
        
        int retval = Imgproc.floodFill(img, mask, new Point(matSize / 2, matSize / 2), new Scalar(1));
        
        assertEquals(Core.countNonZero(img), retval);
        
        Core.circle(mask, new Point(matSize / 2 + 1, matSize / 2 + 1), 3, new Scalar(0));
        
        assertEquals(retval + 4 * (matSize + 1), Core.countNonZero(mask));
        
        assertMatEqual(mask.submat(1, matSize+1, 1, matSize+1), img);
    }
    
    public void testFloodFillMatMatPointScalar_WithoutMask() {
        Mat img = gray0;
        
        Core.circle(img, new Point(matSize / 2, matSize / 2), 3, new Scalar(2));
        
        //TODO: ideally we should pass null instead of "new Mat()"
        int retval = Imgproc.floodFill(img, new Mat(), new Point(matSize / 2, matSize / 2), new Scalar(1));
        
        Core.circle(img, new Point(matSize / 2, matSize / 2), 3, new Scalar(0));
        
        assertEquals(Core.countNonZero(img), retval);
    }

    public void testFloodFillMatMatPointScalarRect() {
        fail("Not yet implemented");
    }

    public void testFloodFillMatMatPointScalarRectScalar() {
        fail("Not yet implemented");
    }

    public void testFloodFillMatMatPointScalarRectScalarScalar() {
        fail("Not yet implemented");
    }

    public void testFloodFillMatMatPointScalarRectScalarScalarInt() {
        fail("Not yet implemented");
    }

    public void testGaussianBlurMatMatSizeDouble() {
        Imgproc.GaussianBlur(gray0, dst, size, 1.0);
        assertMatEqual(gray0, dst);

        Imgproc.GaussianBlur(gray2, dst, size, 1.0);
        assertMatEqual(gray2, dst);

    }

    public void testGaussianBlurMatMatSizeDoubleDouble() {
        Imgproc.GaussianBlur(gray2, dst, size, 0.0, 0.0);
        assertMatEqual(gray2, dst);
    }

    public void testGaussianBlurMatMatSizeDoubleDoubleInt() {
        Imgproc.GaussianBlur(gray2, dst, size, 1.0, 3.0, Imgproc.BORDER_REFLECT);
        assertMatEqual(gray2, dst);
    }

    public void testGetAffineTransform() {
        fail("Not yet implemented");
    }

    public void testGetDefaultNewCameraMatrixMat() {
        truth = new Mat();

        truth = Imgproc.getDefaultNewCameraMatrix(gray0);
        assertTrue(0 == Core.countNonZero(truth));
        assertFalse(truth.empty());
    }

    public void testGetDefaultNewCameraMatrixMatSize() {
        truth = new Mat();

        truth = Imgproc.getDefaultNewCameraMatrix(gray0, size);
        assertTrue(0 == Core.countNonZero(truth));
        assertFalse(truth.empty());
    }

    public void testGetDefaultNewCameraMatrixMatSizeBoolean() {
        truth = new Mat();

        truth = Imgproc.getDefaultNewCameraMatrix(gray0, size, true);
        assertTrue(0 != Core.countNonZero(truth));
        assertFalse(truth.empty());
    }

    public void testGetDerivKernelsMatMatIntIntInt() {
        Mat kx = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        Mat ky = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        Mat expKx = new Mat(3, 1, CvType.CV_32F);
        Mat expKy = new Mat(3, 1, CvType.CV_32F);

        kx.put(0, 0, 1, 1);
        kx.put(1, 0, 1, 1);

        ky.put(0, 0, 2, 2);
        ky.put(1, 0, 2, 2);

        expKx.put(0, 0, 1, -2, 1);
        expKy.put(0, 0, 1, -2, 1);

        Imgproc.getDerivKernels(kx, ky, 2, 2, 3);
        assertMatEqual(expKx, kx, EPS);
        assertMatEqual(expKy, ky, EPS);
    }

    public void testGetDerivKernelsMatMatIntIntIntBoolean() {
        Mat kx = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        Mat ky = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        Mat expKx = new Mat(3, 1, CvType.CV_32F);
        Mat expKy = new Mat(3, 1, CvType.CV_32F);

        kx.put(0, 0, 1, 1);
        kx.put(1, 0, 1, 1);

        ky.put(0, 0, 2, 2);
        ky.put(1, 0, 2, 2);

        expKx.put(0, 0, 1, -2, 1);
        expKy.put(0, 0, 1, -2, 1);

        Imgproc.getDerivKernels(kx, ky, 2, 2, 3, true);
        assertMatEqual(expKx, kx, EPS);
        assertMatEqual(expKy, ky, EPS);
    }

    public void testGetDerivKernelsMatMatIntIntIntBooleanInt() {
        Mat kx = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        Mat ky = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        Mat expKx = new Mat(3, 1, CvType.CV_32F);
        Mat expKy = new Mat(3, 1, CvType.CV_32F);

        kx.put(0, 0, 1, 1);
        kx.put(1, 0, 1, 1);

        ky.put(0, 0, 2, 2);
        ky.put(1, 0, 2, 2);

        expKx.put(0, 0, 1, -2, 1);
        expKy.put(0, 0, 1, -2, 1);

        Imgproc.getDerivKernels(kx, ky, 2, 2, 3, true, CvType.CV_32F);
        assertMatEqual(expKx, kx, EPS);
        assertMatEqual(expKy, ky, EPS);
    }

    public void testGetGaussianKernelIntDouble() {
        truth = new Mat(1, 1, CvType.CV_64FC1, new Scalar(1.0));

        dst = Imgproc.getGaussianKernel(1, 0.5);
        assertMatEqual(truth, dst, EPS);
    }

    public void testGetGaussianKernelIntDoubleInt() {
        truth = new Mat(3, 1, CvType.CV_32F);
        truth.put(0, 0, 0.23899426, 0.52201146, 0.23899426);

        dst = Imgproc.getGaussianKernel(3, 0.8, CvType.CV_32F);
        assertMatEqual(truth, dst, EPS);
    }

    public void testGetRectSubPixMatSizePointMat() {
        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(255));
        Size size = new Size(3, 3);
        Point center = new Point(gray255.cols() / 2, gray255.rows() / 2);

        Imgproc.getRectSubPix(gray255, size, center, dst);
        assertMatEqual(truth, dst);
    }

    public void testGetRectSubPixMatSizePointMatInt() {
        Mat src = new Mat(10, 10, CvType.CV_32F, new Scalar(2));
        truth = new Mat(5, 5, CvType.CV_32F, new Scalar(2));
        Size patchSize = new Size(5, 5);
        Point center = new Point(src.cols() / 2, src.rows() / 2);

        Imgproc.getRectSubPix(src, patchSize, center, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testGetRotationMatrix2D() {
        truth = new Mat(2, 3, CvType.CV_64F);
        truth.put(0, 0, 1, 0, 0);
        truth.put(1, 0, 0, 1, 0);
        Point center = new Point(0, 0);
        dst = Imgproc.getRotationMatrix2D(center, 0.0, 1.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testGetStructuringElementIntSize() {
        truth = new Mat(3, 3, CvType.CV_8UC1, new Scalar(1.0));

        dst = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size);
        assertMatEqual(truth, dst);
    }

    public void testGetStructuringElementIntSizePoint() {
        truth = new Mat(3, 3, CvType.CV_8UC1);

        truth.put(0, 0, 0, 0, 1);
        truth.put(1, 0, 0, 0, 1);
        truth.put(2, 0, 1, 1, 1);

        dst = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, size,
                anchorPoint);
        assertMatEqual(truth, dst);
    }

    public void testGoodFeaturesToTrackMatMatIntDoubleDouble() {
        Mat src = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(2.0));
        Mat corners = new Mat(1, 4, CvType.CV_32FC2);
        corners.put(0, 0, 1.0, 1.0, 6.0, 1.0, 6.0, 1.0, 6.0, 6.0);

        Imgproc.goodFeaturesToTrack(src, dst, 100, 0.01, 5.0);
        // TODO : How do we test this?
        fail("Not yet implemented");
    }

    public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMat() {
        fail("Not yet implemented");
    }

    public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMatInt() {
        fail("Not yet implemented");
    }

    public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMatIntBoolean() {
        fail("Not yet implemented");
    }

    public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMatIntBooleanDouble() {
        fail("Not yet implemented");
    }

    public void testGrabCutMatMatRectMatMatInt() {
        fail("Not yet implemented");
    }

    public void testGrabCutMatMatRectMatMatIntInt() {
        fail("Not yet implemented");
    }

    public void testHoughCirclesMatMatIntDoubleDouble() {
        // double minDist = gray255.row(0)/4;
        // Imgproc.HoughCircles(gray255, dst, Imgproc.CV_HOUGH_GRADIENT, 2.0,
        // 0.5);
        // TODO : How do we test this?

        fail("Not yet implemented");
    }

    public void testHoughCirclesMatMatIntDoubleDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testHoughCirclesMatMatIntDoubleDoubleDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testHoughCirclesMatMatIntDoubleDoubleDoubleDoubleInt() {
        fail("Not yet implemented");
    }

    public void testHoughCirclesMatMatIntDoubleDoubleDoubleDoubleIntInt() {
        fail("Not yet implemented");
    }

    public void testHoughLinesMatMatDoubleDoubleInt() {
        fail("Not yet implemented");
    }

    public void testHoughLinesMatMatDoubleDoubleIntDouble() {
        fail("Not yet implemented");
    }

    public void testHoughLinesMatMatDoubleDoubleIntDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testHoughLinesPMatMatDoubleDoubleInt() {
        fail("Not yet implemented");
    }

    public void testHoughLinesPMatMatDoubleDoubleIntDouble() {
        fail("Not yet implemented");
    }

    public void testHoughLinesPMatMatDoubleDoubleIntDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testHuMoments() {
        fail("Not yet implemented");
    }

    public void testInitUndistortRectifyMap() {
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 1);
        cameraMatrix.put(2, 0, 0, 0, 1);

        Mat R = new Mat(3, 3, CvType.CV_32F, new Scalar(2.0));
        Mat newCameraMatrix = new Mat(3, 3, CvType.CV_32F, new Scalar(3.0));

        Mat distCoeffs = new Mat();
        Mat map1 = new Mat();
        Mat map2 = new Mat();

        // FIXME: dstmap1 - Documentation says Cvtype but requires integer
        Imgproc.initUndistortRectifyMap(cameraMatrix, distCoeffs, R,
                newCameraMatrix, size, CvType.CV_32F, map1, map2);
        fail("Not yet implemented");
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMat() {
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F);
        // Size imageSize = new Size(2, 2);

        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 2);
        cameraMatrix.put(2, 0, 0, 0, 1);

        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 0, 0);
        truth.put(2, 0, 0, 3, 0);

        distCoeffs.put(0, 0, 1.0, 3.0, 2.0, 4);
        // TODO: No documentation for this function
        // Imgproc.initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize,
        // 5.0, m1type, truthput1, truthput2);
        fail("Not yet implemented");
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMatInt() {
        fail("Not yet implemented");
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMatIntDouble() {
        fail("Not yet implemented");
    }

    public void testInpaint() {
        Imgproc.inpaint(gray255, gray128, dst, 3.0, Imgproc.INPAINT_TELEA);
        assertMatEqual(gray255, dst);
    }

    public void testIntegral2MatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3.0));
        Mat expSum = new Mat(4, 4, CvType.CV_64F);
        Mat expSqsum = new Mat(4, 4, CvType.CV_64F);
        Mat sum = new Mat();
        Mat sqsum = new Mat();

        expSum.put(0, 0, 0, 0, 0, 0);
        expSum.put(1, 0, 0, 3, 6, 9);
        expSum.put(2, 0, 0, 6, 12, 18);
        expSum.put(3, 0, 0, 9, 18, 27);

        expSqsum.put(0, 0, 0, 0, 0, 0);
        expSqsum.put(1, 0, 0, 9, 18, 27);
        expSqsum.put(2, 0, 0, 18, 36, 54);
        expSqsum.put(3, 0, 0, 27, 54, 81);

        Imgproc.integral2(src, sum, sqsum);
        assertMatEqual(expSum, sum, EPS);
        assertMatEqual(expSqsum, sqsum, EPS);
    }

    public void testIntegral2MatMatMatInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3.0));
        Mat expSum = new Mat(4, 4, CvType.CV_64F);
        Mat expSqsum = new Mat(4, 4, CvType.CV_64F);
        Mat sum = new Mat();
        Mat sqsum = new Mat();

        expSum.put(0, 0, 0, 0, 0, 0);
        expSum.put(1, 0, 0, 3, 6, 9);
        expSum.put(2, 0, 0, 6, 12, 18);
        expSum.put(3, 0, 0, 9, 18, 27);

        expSqsum.put(0, 0, 0, 0, 0, 0);
        expSqsum.put(1, 0, 0, 9, 18, 27);
        expSqsum.put(2, 0, 0, 18, 36, 54);
        expSqsum.put(3, 0, 0, 27, 54, 81);

        Imgproc.integral2(src, sum, sqsum, CvType.CV_64F);
        assertMatEqual(expSum, sum, EPS);
        assertMatEqual(expSqsum, sqsum, EPS);
    }

    public void testIntegral3MatMatMatMat() {
        Mat src = new Mat(1, 1, CvType.CV_32F, new Scalar(1.0));
        Mat expSum = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        Mat expSqsum = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        Mat expTilted = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        Mat sum = new Mat();
        Mat sqsum = new Mat();
        Mat tilted = new Mat();

        expSum.put(0, 0, 0, 0);
        expSum.put(1, 0, 0, 1);

        expSqsum.put(0, 0, 0, 0);
        expSqsum.put(1, 0, 0, 1);

        expTilted.put(0, 0, 0, 0);
        expTilted.put(1, 0, 0, 1);

        Imgproc.integral3(src, sum, sqsum, tilted);
        assertMatEqual(expSum, sum, EPS);
        assertMatEqual(expSqsum, sqsum, EPS);
        assertMatEqual(expTilted, tilted, EPS);

    }

    public void testIntegral3MatMatMatMatInt() {
        Mat src = new Mat(1, 1, CvType.CV_32F, new Scalar(1.0));
        Mat expSum = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        Mat expSqsum = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        Mat expTilted = new Mat(imgprocSz, imgprocSz, CvType.CV_64F);
        Mat sum = new Mat();
        Mat sqsum = new Mat();
        Mat tilted = new Mat();

        expSum.put(0, 0, 0, 0);
        expSum.put(1, 0, 0, 1);

        expSqsum.put(0, 0, 0, 0);
        expSqsum.put(1, 0, 0, 1);

        expTilted.put(0, 0, 0, 0);
        expTilted.put(1, 0, 0, 1);

        Imgproc.integral3(src, sum, sqsum, tilted, CvType.CV_64F);
        assertMatEqual(expSum, sum, EPS);
        assertMatEqual(expSqsum, sqsum, EPS);
        assertMatEqual(expTilted, tilted, EPS);
    }

    public void testIntegralMatMat() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2.0));
        truth = new Mat(3, 3, CvType.CV_64F);

        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 2, 4);
        truth.put(2, 0, 0, 4, 8);

        Imgproc.integral(src, dst);
        assertMatEqual(truth, dst, EPS);

    }

    public void testIntegralMatMatInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2.0));
        truth = new Mat(3, 3, CvType.CV_64F);

        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 2, 4);
        truth.put(2, 0, 0, 4, 8);

        Imgproc.integral(src, dst, CvType.CV_64F);
        assertMatEqual(truth, dst, EPS);
    }

    public void testInvertAffineTransform() {
        Mat src = new Mat(2, 3, CvType.CV_64F, new Scalar(1));
        truth = new Mat(2, 3, CvType.CV_64F, new Scalar(0));

        Imgproc.invertAffineTransform(src, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testIsContourConvex() {
        Mat contour1 = new Mat(1, 4, CvType.CV_32FC2);
        contour1.put(0, 0, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 5.0, 4.0);
        assertFalse(Imgproc.isContourConvex(contour1));

        Mat contour2 = new Mat(1, 2, CvType.CV_32FC2);
        contour2.put(0, 0, 1.0, 1.0, 5.0, 1.0);
        assertFalse(Imgproc.isContourConvex(contour2));
    }

    public void testLaplacianMatMatInt() {
        Imgproc.Laplacian(gray0, dst, CvType.CV_8U);
        assertMatEqual(gray0, dst);
    }

    public void testLaplacianMatMatIntInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(2.0));
        truth = new Mat(3, 3, CvType.CV_32F, new Scalar(0.0));

        Imgproc.Laplacian(src, dst, CvType.CV_32F, 1);
        assertMatEqual(truth, dst, EPS);
    }

    public void testLaplacianMatMatIntIntDouble() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);

        truth.put(0, 0, -8, 8);
        truth.put(1, 0, 8, -8);

        Imgproc.Laplacian(src, dst, CvType.CV_32F, 1, 2.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testLaplacianMatMatIntIntDoubleDouble() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);

        truth.put(0, 0, -7.9990001, 8.0009995);
        truth.put(1, 0, 8.0009995, -7.9990001);

        Imgproc.Laplacian(src, dst, CvType.CV_32F, 1, 2.0, EPS);
        assertMatEqual(truth, dst, EPS);
    }

    public void testLaplacianMatMatIntIntDoubleDoubleInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(2.0));
        truth = new Mat(3, 3, CvType.CV_32F, new Scalar(0.00099945068));

        Imgproc.Laplacian(src, dst, CvType.CV_32F, 1, 2.0, EPS,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testMatchShapes() {
        Mat contour1 = new Mat(1, 4, CvType.CV_32FC2);
        Mat contour2 = new Mat(1, 4, CvType.CV_32FC2);

        contour1.put(0, 0, 1.0, 1.0, 5.0, 1.0, 4.0, 3.0, 6.0, 2.0);
        contour1.put(0, 0, 1.0, 1.0, 6.0, 1.0, 4.0, 1.0, 2.0, 5.0);

        // TODO: returns random comparers
        double comparer = Imgproc.matchShapes(contour1, contour2,
                Imgproc.CV_CONTOURS_MATCH_I1, 1.0);
        double expComparer = 2.98;
        assertEquals(expComparer, comparer);
    }

    public void testMatchTemplate() {
        Mat image = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        Mat templ = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);

        image.put(0, 0, 1, 2, 3, 4);
        templ.put(0, 0, 5, 6, 7, 8);

        truth = new Mat(1, 1, CvType.CV_32F, new Scalar(70));
        Imgproc.matchTemplate(image, templ, dst, Imgproc.TM_CCORR);
        assertMatEqual(truth, dst, EPS);

        truth = new Mat(1, 1, CvType.CV_32F, new Scalar(0));
        Imgproc.matchTemplate(gray255, gray0, dst, Imgproc.TM_CCORR);
        assertMatEqual(truth, dst, EPS);
    }

    public void testMedianBlur() {
        Imgproc.medianBlur(gray255, dst, 5);
        assertMatEqual(gray255, dst);

        Imgproc.medianBlur(gray2, dst, 3);
        assertMatEqual(gray2, dst);
    }

    public void testMinAreaRect() {
        Mat points = new Mat(1, 4, CvType.CV_32FC2);
        points.put(0, 0, 1.0, 1.0, 5.0, 1.0, 4.0, 3.0, 6.0, 2.0);

        RotatedRect rrect = Imgproc.minAreaRect(points);
        // TODO - how to test rotated rectangle
        fail("Not yet implemented");
    }

    public void testMinEnclosingCircle() {
        Mat points = new Mat(1, 4, CvType.CV_32FC2);
        Point actualCenter = new Point();
        Point expCenter = new Point(0, 0);
        float radius = 0.0f;
        // float expectedRadius = 1.0f;
        points.put(0, 0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0);
        // TODO: Unexpected radius is returned i.e 0
        Imgproc.minEnclosingCircle(points, actualCenter, radius);
        assertEquals(expCenter, actualCenter);
        // assertEquals(expectedRadius, radius);
        fail("Not yet implemented");
    }

    public void testMomentsMat() {
        fail("Not yet implemented");
    }

    public void testMomentsMatBoolean() {
        fail("Not yet implemented");
    }

    public void testMorphologyExMatMatIntMat() {
        Imgproc.morphologyEx(gray255, dst, Imgproc.MORPH_GRADIENT, gray0);
        assertMatEqual(gray0, dst);
    }

    public void testMorphologyExMatMatIntMatPoint() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_8U);

        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_8U, new Scalar(0));
        Point point = new Point(0, 0);

        truth = Mat.eye(imgprocSz, imgprocSz, CvType.CV_8U);

        Imgproc.morphologyEx(src, dst, Imgproc.MORPH_OPEN, kernel, point);
        assertMatEqual(truth, dst);
    }

    public void testMorphologyExMatMatIntMatPointInt() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_8U);

        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_8U, new Scalar(0));
        Point point = new Point(0, 0);

        Imgproc.morphologyEx(src, dst, Imgproc.MORPH_CLOSE, kernel, point, 10);
        truth = Mat.eye(imgprocSz, imgprocSz, CvType.CV_8U);
        assertMatEqual(truth, dst);
    }

    public void testMorphologyExMatMatIntMatPointIntInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        src.put(0, 0, 2, 1);
        src.put(1, 0, 2, 1);

        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_8U, new Scalar(1));
        Point point = new Point(1, 1);

        Imgproc.morphologyEx(src, dst, Imgproc.MORPH_TOPHAT, kernel, point, 10,
                Imgproc.BORDER_REFLECT);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        truth.put(0, 0, 1, 0);
        truth.put(1, 0, 1, 0);
        assertMatEqual(truth, dst);
    }

    public void testMorphologyExMatMatIntMatPointIntIntScalar() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        src.put(0, 0, 2, 1);
        src.put(1, 0, 2, 1);

        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_8U, new Scalar(1));
        Point point = new Point(1, 1);
        Scalar sc = new Scalar(3, 3);

        Imgproc.morphologyEx(src, dst, Imgproc.MORPH_TOPHAT, kernel, point, 10,
                Imgproc.BORDER_REFLECT, sc);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        truth.put(0, 0, 1, 0);
        truth.put(1, 0, 1, 0);
        assertMatEqual(truth, dst);
    }

    public void testPointPolygonTest() {
        Mat contour1 = new Mat(1, 5, CvType.CV_32FC2);
        contour1.put(0, 0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0);
        Point pt1 = new Point(contour1.cols() / 2, contour1.rows() / 2);

        double sign1 = Imgproc.pointPolygonTest(contour1, pt1, false);
        assertTrue(sign1 < 0);

        Mat contour2 = new Mat(1, 3, CvType.CV_32FC2);
        contour2.put(0, 0, 0.0, 0.0, 2.0, 0.0, 1.0, 3.0);
        Point pt2 = new Point(1, 1);

        double sign2 = Imgproc.pointPolygonTest(contour2, pt2, false);
        assertEquals(100.0, sign2);
    }

    public void testPreCornerDetectMatMatInt() {
        Mat src = new Mat(4, 4, CvType.CV_32F, new Scalar(1));
        truth = new Mat(4, 4, CvType.CV_32F, new Scalar(0));
        int ksize = 3;

        Imgproc.preCornerDetect(src, dst, ksize);
        assertMatEqual(truth, dst, EPS);
    }

    public void testPreCornerDetectMatMatIntInt() {
        Mat src = new Mat(4, 4, CvType.CV_32F, new Scalar(1));
        truth = new Mat(4, 4, CvType.CV_32F, new Scalar(0));
        int ksize = 3;

        Imgproc.preCornerDetect(src, dst, ksize, Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testPyrDownMatMat() {
        Mat src = new Mat(4, 4, CvType.CV_32F);

        src.put(0, 0, 2, 1, 4, 2);
        src.put(1, 0, 3, 2, 6, 8);
        src.put(2, 0, 4, 6, 8, 10);
        src.put(3, 0, 12, 32, 6, 18);

        Imgproc.pyrDown(src, dst);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        truth.put(0, 0, 2.78125, 4.609375);
        truth.put(1, 0, 8.546875, 8.8515625);
        assertMatEqual(truth, dst, EPS);
    }

    public void testPyrDownMatMatSize() {
        Mat src = new Mat(4, 4, CvType.CV_32F);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        Size dstSize = new Size(2, 2);

        src.put(0, 0, 2, 1, 4, 2);
        src.put(1, 0, 3, 2, 6, 8);
        src.put(2, 0, 4, 6, 8, 10);
        src.put(3, 0, 12, 32, 6, 18);

        truth.put(0, 0, 2.78125, 4.609375);
        truth.put(1, 0, 8.546875, 8.8515625);

        Imgproc.pyrDown(src, dst, dstSize);
        assertMatEqual(truth, dst, EPS);
    }

    public void testPyrMeanShiftFilteringMatMatDoubleDouble() {
        Mat src = new Mat(matSize, matSize, CvType.CV_8UC3, new Scalar(0.0));
        Imgproc.pyrMeanShiftFiltering(src, dst, 2.0, 4.0);
        // TODO : size of destination matrix not understandable
        fail("Not yet implemented");
    }

    public void testPyrMeanShiftFilteringMatMatDoubleDoubleInt() {
        fail("Not yet implemented");
    }

    public void testPyrMeanShiftFilteringMatMatDoubleDoubleIntTermCriteria() {
        fail("Not yet implemented");
    }

    public void testPyrUpMatMat() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        src.put(0, 0, 2, 1);
        src.put(1, 0, 3, 2);

        truth = new Mat(4, 4, CvType.CV_32F);
        truth.put(0, 0, 2, 1.75, 1.375, 1.25);
        truth.put(1, 0, 2.25, 2, 1.625, 1.5);
        truth.put(2, 0, 2.5, 2.25, 1.875, 1.75);
        truth.put(3, 0, 2.25, 2, 1.625, 1.5);

        Imgproc.pyrUp(src, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testPyrUpMatMatSize() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        src.put(0, 0, 2, 1);
        src.put(1, 0, 3, 2);

        Size dstSize = new Size(4, 4);
        truth = new Mat(4, 4, CvType.CV_32F);
        truth.put(0, 0, 2, 1.75, 1.375, 1.25);
        truth.put(1, 0, 2.25, 2, 1.625, 1.5);
        truth.put(2, 0, 2.5, 2.25, 1.875, 1.75);
        truth.put(3, 0, 2.25, 2, 1.625, 1.5);

        Imgproc.pyrUp(src, dst, dstSize);
        assertMatEqual(truth, dst, EPS);
    }

    public void testRemapMatMatMatMatInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2.0));
        Mat map1 = new Mat(1, 3, CvType.CV_32FC1);
        Mat map2 = new Mat(1, 3, CvType.CV_32FC1);

        map1.put(0, 0, 3.0, 6.0, 5, 0);
        map2.put(0, 0, 4.0, 8.0, 12.0);

        truth = new Mat(1, 3, CvType.CV_32F, new Scalar(0));

        Imgproc.remap(src, dst, map1, map2, Imgproc.INTER_LINEAR);
        assertMatEqual(truth, dst, EPS);
    }

    public void testRemapMatMatMatMatIntInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2.0));
        Mat map1 = new Mat(1, 3, CvType.CV_32FC1);
        Mat map2 = new Mat(1, 3, CvType.CV_32FC1);

        map1.put(0, 0, 3.0, 6.0, 5, 0);
        map2.put(0, 0, 4.0, 8.0, 12.0);

        truth = new Mat(1, 3, CvType.CV_32F, new Scalar(2));

        Imgproc.remap(src, dst, map1, map2, Imgproc.INTER_LINEAR,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testRemapMatMatMatMatIntIntScalar() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2.0));
        Mat map1 = new Mat(1, 3, CvType.CV_32FC1);
        Mat map2 = new Mat(1, 3, CvType.CV_32FC1);

        Scalar sc = new Scalar(0.0);

        map1.put(0, 0, 3.0, 6.0, 5, 0);
        map2.put(0, 0, 4.0, 8.0, 12.0);

        truth = new Mat(1, 3, CvType.CV_32F, new Scalar(2));

        Imgproc.remap(src, dst, map1, map2, Imgproc.INTER_LINEAR,
                Imgproc.BORDER_REFLECT, sc);
        assertMatEqual(truth, dst, EPS);
    }

    public void testResizeMatMatSize() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_8UC1, new Scalar(1.0));
        truth = new Mat(1, 1, CvType.CV_8UC1, new Scalar(1.0));
        Size dsize = new Size(1, 1);

        Imgproc.resize(src, dst, dsize);
        assertMatEqual(truth, dst);
    }

    public void testResizeMatMatSizeDouble() {
        Size dsize = new Size(2, 2);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_8UC1, new Scalar(255));

        Imgproc.resize(gray255, dst, dsize, 0.5);
        assertMatEqual(truth, dst);
    }

    public void testResizeMatMatSizeDoubleDouble() {
        Size dsize = new Size(2, 2);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_8UC1, new Scalar(255));

        Imgproc.resize(gray255, dst, dsize, 0.0, 0.0);
        assertMatEqual(truth, dst);
    }

    public void testResizeMatMatSizeDoubleDoubleInt() {
        Size dsize = new Size(2, 2);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_8UC1, new Scalar(255));

        Imgproc.resize(gray255, dst, dsize, 1.5, 1.5, Imgproc.INTER_AREA);
        assertMatEqual(truth, dst);
    }

    public void testScharrMatMatIntIntInt() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(0));
        Imgproc.Scharr(src, dst, CvType.CV_32F, 1, 0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testScharrMatMatIntIntIntDouble() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(0));
        Imgproc.Scharr(src, dst, CvType.CV_32F, 0, 1, 1.5);
        assertMatEqual(truth, dst, EPS);
    }

    public void testScharrMatMatIntIntIntDoubleDouble() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(0.001));
        Imgproc.Scharr(src, dst, CvType.CV_32F, 1, 0, 1.5, 0.001);
        assertMatEqual(truth, dst, EPS);
    }

    public void testScharrMatMatIntIntIntDoubleDoubleInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32F);

        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, -15, -19.5, -4.5);
        truth.put(1, 0, 10.5, 0, -10.5);
        truth.put(2, 0, 4.5, 19.5, 15);

        Imgproc.Scharr(src, dst, CvType.CV_32F, 1, 0, 1.5, 0.0,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testSepFilter2DMatMatIntMatMat() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2.0));
        Mat kernelX = new Mat(1, 3, CvType.CV_32FC1);
        Mat kernelY = new Mat(1, 3, CvType.CV_32FC1);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(420));

        kernelX.put(0, 0, 4.0, 3.0, 7.0);
        kernelY.put(0, 0, 9.0, 4.0, 2.0);

        Imgproc.sepFilter2D(src, dst, CvType.CV_32F, kernelX, kernelY);
        assertMatEqual(truth, dst, EPS);
    }

    public void testSepFilter2DMatMatIntMatMatPoint() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1,
                new Scalar(2.0));
        Mat kernelX = new Mat(1, 3, CvType.CV_32FC1);
        Mat kernelY = new Mat(1, 3, CvType.CV_32FC1);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(36.0));

        kernelX.put(0, 0, 2.0, 2.0, 2.0);
        kernelY.put(0, 0, 1.0, 1.0, 1.0);

        Imgproc.sepFilter2D(src, dst, CvType.CV_32F, kernelX, kernelY,
                anchorPoint);
        assertMatEqual(truth, dst, EPS);
    }

    public void testSepFilter2DMatMatIntMatMatPointDouble() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1,
                new Scalar(2.0));

        Mat kernelX = new Mat(1, 3, CvType.CV_32FC1);
        kernelX.put(0, 0, 2.0, 2.0, 2.0);

        Mat kernelY = new Mat(1, 3, CvType.CV_32FC1);
        kernelY.put(0, 0, 1.0, 1.0, 1.0);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(36.001));
        Imgproc.sepFilter2D(src, dst, CvType.CV_32F, kernelX, kernelY,
                anchorPoint, EPS);
        assertMatEqual(truth, dst, EPS);
    }

    public void testSepFilter2DMatMatIntMatMatPointDoubleInt() {
        Mat kernelX = new Mat(1, 3, CvType.CV_32FC1);
        kernelX.put(0, 0, 2.0, 2.0, 2.0);

        Mat kernelY = new Mat(1, 3, CvType.CV_32FC1);
        kernelY.put(0, 0, 1.0, 1.0, 1.0);

        truth = new Mat(10, 10, CvType.CV_32F, new Scalar(0.001));
        Imgproc.sepFilter2D(gray0, dst, CvType.CV_32F, kernelX, kernelY,
                anchorPoint, EPS, Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testSobelMatMatIntIntInt() {
        Imgproc.Sobel(gray0, dst, CvType.CV_8U, 2, 0);
        assertMatEqual(gray0, dst);
    }

    public void testSobelMatMatIntIntIntInt() {
        Imgproc.Sobel(gray255, dst, CvType.CV_8U, 1, 0, 3);
        assertMatEqual(gray0, dst);
    }

    public void testSobelMatMatIntIntIntIntDouble() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        src.put(0, 0, 2, 0, 1);
        src.put(1, 0, 3, 0, -10);
        src.put(2, 0, -4, 0, 3);

        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, 0, -56, 0);
        truth.put(1, 0, 0, -40, 0);
        truth.put(2, 0, 0, -24, 0);

        Imgproc.Sobel(src, dst, CvType.CV_32F, 1, 0, 3, 2.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testSobelMatMatIntIntIntIntDoubleDouble() {
        Imgproc.Sobel(gray255, dst, CvType.CV_8U, 1, 0, 3, 2.0, 0.001);
        assertMatEqual(gray0, dst);
    }

    public void testSobelMatMatIntIntIntIntDoubleDoubleInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        src.put(0, 0, 2, 0, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 1, 0, 2);

        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, -16, -12, 4);
        truth.put(1, 0, -14, -12, 2);
        truth.put(2, 0, -10, 0, 10);

        Imgproc.Sobel(src, dst, CvType.CV_32F, 1, 0, 3, 2.0, 0.0,
                Imgproc.BORDER_REPLICATE);
        assertMatEqual(truth, dst, EPS);
    }

    public void testThreshold() {
        Imgproc.threshold(gray0, dst, 0.25, 255.0, Imgproc.THRESH_TRUNC);
        assertMatEqual(gray0, dst);

        Imgproc.threshold(gray1, dst, 0.25, 255.0, Imgproc.THRESH_BINARY);
        assertMatEqual(gray255, dst);

        Imgproc.threshold(gray0, dst, 0.25, 255.0, Imgproc.THRESH_BINARY_INV);
        assertMatEqual(gray255, dst);
    }

    public void testUndistortMatMatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3.0));
        truth = new Mat(3, 3, CvType.CV_32F);
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F);

        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 2);
        cameraMatrix.put(2, 0, 0, 0, 1);

        distCoeffs.put(0, 0, 1.0, 3.0, 2.0, 4.0);

        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 0, 0);
        truth.put(2, 0, 0, 3, 0);

        Imgproc.undistort(src, dst, cameraMatrix, distCoeffs);
        assertMatEqual(truth, dst, EPS);
    }

    public void testUndistortMatMatMatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3.0));

        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        Mat newCameraMatrix = new Mat(3, 3, CvType.CV_32F, new Scalar(1.0));
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F);

        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 2);
        cameraMatrix.put(2, 0, 0, 0, 1);

        distCoeffs.put(0, 0, 2.0, 1.0, 4.0, 5.0);

        truth = new Mat(3, 3, CvType.CV_32F, new Scalar(3));

        Imgproc.undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpAffineMatMatMatSize() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        ;
        Mat M = new Mat(2, 3, CvType.CV_32F);

        src.put(0, 0, 2, 0, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 1, 0, 2);

        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 2, 0);
        truth.put(2, 0, 0, 6, 4);

        M.put(0, 0, 1, 0, 1);
        M.put(1, 0, 0, 1, 1);

        Imgproc.warpAffine(src, dst, M, size);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpAffineMatMatMatSizeInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        Size dsize = new Size(2, 2);
        Mat M = new Mat(2, 3, CvType.CV_32F);

        src.put(0, 0, 2, 4, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 0, 2, 2);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        truth.put(0, 0, 6, 4);
        truth.put(1, 0, 6, 4);

        M.put(0, 0, 1, 0, 0);
        M.put(1, 0, 0, 0, 1);

        Imgproc.warpAffine(src, dst, M, dsize, Imgproc.WARP_INVERSE_MAP);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpAffineMatMatMatSizeIntInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F);

        Size dsize = new Size(2, 2);
        Mat M = Mat.eye(2, 3, CvType.CV_32F);

        src.put(0, 0, 2, 4, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 0, 2, 2);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        truth.put(0, 0, 2, 4);
        truth.put(1, 0, 6, 4);

        Imgproc.warpAffine(src, dst, M, dsize, Imgproc.WARP_INVERSE_MAP,
                Imgproc.BORDER_TRANSPARENT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpAffineMatMatMatSizeIntIntScalar() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        Size dsize = new Size(2, 2);
        Scalar sc = new Scalar(1.0);

        src.put(0, 0, 2, 4, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 0, 2, 2);

        Mat M = new Mat(2, 3, CvType.CV_32F);
        M.put(0, 0, 1, 0, 0);
        M.put(1, 0, 0, 0, 1);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        truth.put(0, 0, 6, 4);
        truth.put(1, 0, 6, 4);

        Imgproc.warpAffine(src, dst, M, dsize, Imgproc.WARP_INVERSE_MAP,
                Imgproc.BORDER_CONSTANT, sc);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpPerspectiveMatMatMatSize() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        Mat M = new Mat(3, 3, CvType.CV_32F);

        src.put(0, 0, 2, 4, 1);
        src.put(1, 0, 0, 4, 5);
        src.put(2, 0, 1, 2, 2);

        M.put(0, 0, 1, 0, 1);
        M.put(1, 0, 0, 1, 1);
        M.put(2, 0, 0, 0, 1);

        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 2, 4);
        truth.put(2, 0, 0, 0, 4);

        Imgproc.warpPerspective(src, dst, M, size);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpPerspectiveMatMatMatSizeInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        src.put(0, 0, 2, 4, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 0, 2, 2);

        Size dsize = new Size(2, 2);
        Mat M = new Mat(3, 3, CvType.CV_32F);
        M.put(0, 0, 1, 0, 0);
        M.put(1, 0, 0, 0, 1);
        M.put(2, 0, 0, 0, 1);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        truth.put(0, 0, 6, 4);
        truth.put(1, 0, 6, 4);

        Imgproc.warpPerspective(src, dst, M, dsize, Imgproc.WARP_INVERSE_MAP);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpPerspectiveMatMatMatSizeIntInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        src.put(0, 0, 2, 4, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 0, 2, 2);

        Size dsize = new Size(2, 2);
        Mat M = new Mat(3, 3, CvType.CV_32F);
        M.put(0, 0, 1, 0, 0);
        M.put(1, 0, 0, 0, 1);
        M.put(2, 0, 0, 0, 1);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        truth.put(0, 0, 6, 4);
        truth.put(1, 0, 6, 4);

        Imgproc.warpPerspective(src, dst, M, dsize, Imgproc.WARP_INVERSE_MAP,
                Imgproc.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpPerspectiveMatMatMatSizeIntIntScalar() {
        Mat src = new Mat(3, 3, CvType.CV_32F);
        src.put(0, 0, 2, 4, 1);
        src.put(1, 0, 6, 4, 3);
        src.put(2, 0, 0, 2, 2);

        Size dsize = new Size(2, 2);
        Mat M = Mat.eye(3, 3, CvType.CV_32F);
        Scalar sc = new Scalar(1.0);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F);
        truth.put(0, 0, 2, 4);
        truth.put(1, 0, 6, 4);

        Imgproc.warpPerspective(src, dst, M, dsize, Imgproc.WARP_INVERSE_MAP,
                Imgproc.BORDER_REFLECT, sc);
        assertMatEqual(truth, dst, EPS);
    }

    public void testWatershed() {
        Mat image = Mat.eye(4, 4, CvType.CV_8UC(3));
        Mat markers = new Mat(4, 4, CvType.CV_32SC1, new Scalar(0));

        truth = new Mat(4, 4, CvType.CV_32SC1);
        truth.put(0, 0, -1, -1, -1, -1);
        truth.put(1, 0, -1, 0, 0, -1);
        truth.put(2, 0, -1, 0, 0, -1);
        truth.put(3, 0, -1, -1, -1, -1);

        Imgproc.watershed(image, markers);
        assertMatEqual(truth, markers);
    }

}