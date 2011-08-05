package org.opencv.test.core;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.test.OpenCVTestCase;
import org.opencv.utils.Converters;

public class CoreTest extends OpenCVTestCase {

    public void testAbsdiff() {
        Core.absdiff(gray128, gray255, dst);
        assertMatEqual(gray127, dst);
    }

    public void testAddMatMatMat() {
        Core.add(gray128, gray128, dst);
        assertMatEqual(gray255, dst);
    }

    public void testAddMatMatMatMat() {
        Core.add(gray0, gray1, dst, gray1);
        assertMatEqual(gray1, dst);

        dst.setTo(new Scalar(127));
        Core.add(gray0, gray1, dst, gray0);
        assertMatEqual(gray127, dst);
    }

    public void testAddMatMatMatMatInt() {
        Core.add(gray0, gray1, dst, gray1, CvType.CV_32F);
        assertTrue(CvType.CV_32F == dst.depth());
        assertMatEqual(gray1_32f, dst, EPS);
    }

    public void testAddWeightedMatDoubleMatDoubleDoubleMat() {
        Core.addWeighted(gray1, 126.0, gray127, 1.0, 2.0, dst);
        assertMatEqual(gray255, dst);
    }

    public void testAddWeightedMatDoubleMatDoubleDoubleMatInt() {
        Core.addWeighted(gray1, 126.0, gray127, 1.0, 2.0, dst, CvType.CV_32F);
        assertTrue(CvType.CV_32F == dst.depth());
        assertMatEqual(gray255_32f, dst, EPS);
    }

    public void testBitwise_andMatMatMat() {
        Core.bitwise_and(gray3, gray2, dst);
        assertMatEqual(gray2, dst);
    }

    public void testBitwise_andMatMatMatMat() {
        Core.bitwise_and(gray0, gray1, dst, gray255);
        assertMatEqual(gray0, dst);
    }

    public void testBitwise_notMatMat() {
        Core.bitwise_not(gray255, dst);
        assertMatEqual(gray0, dst);
    }

    public void testBitwise_notMatMatMat() {
        Core.bitwise_not(gray255, dst, gray255);
        assertMatEqual(gray0, dst);
    }

    public void testBitwise_orMatMatMat() {
        Core.bitwise_or(gray3, gray2, dst);
        assertMatEqual(gray3, dst);
    }

    public void testBitwise_orMatMatMatMat() {
        Core.bitwise_or(gray127, gray128, dst, gray255);
        assertMatEqual(gray255, dst);
    }

    public void testBitwise_xorMatMatMat() {
        Core.bitwise_xor(gray3, gray2, dst);
        assertMatEqual(gray1, dst);
    }

    public void testBitwise_xorMatMatMatMat() {
        Core.bitwise_or(gray127, gray128, dst, gray255);
        assertMatEqual(gray255, dst);
    }

    public void testCalcCovarMatrixMatMatMatInt() {
        Mat covar = new Mat(matSize, matSize, CvType.CV_64FC1);
        Mat mean = new Mat(1, matSize, CvType.CV_64FC1);

        Core.calcCovarMatrix(gray0_32f, covar, mean, Core.COVAR_ROWS | Core.COVAR_NORMAL);

        assertMatEqual(gray0_64f, covar, EPS);
        assertMatEqual(gray0_64f_1d, mean, EPS);
    }

    public void testCalcCovarMatrixMatMatMatIntInt() {
        Mat covar = new Mat(matSize, matSize, CvType.CV_32F);
        Mat mean = new Mat(1, matSize, CvType.CV_32F);

        Core.calcCovarMatrix(gray0_32f, covar, mean, Core.COVAR_ROWS | Core.COVAR_NORMAL, CvType.CV_32F);
        assertMatEqual(gray0_32f, covar, EPS);
        assertMatEqual(gray0_32f_1d, mean, EPS);
    }

    public void testCartToPolarMatMatMatMat() {
        Mat x = new Mat(1, 3, CvType.CV_32F);
        Mat y = new Mat(1, 3, CvType.CV_32F);
        x.put(0, 0, 3.0, 6.0, 5, 0);
        y.put(0, 0, 4.0, 8.0, 12.0);

        Mat magnitude = new Mat(1, 3, CvType.CV_32F);
        Mat angle = new Mat(1, 3, CvType.CV_32F);
        magnitude.put(0, 0, 5.0, 10.0, 13.0);
        angle.put(0, 0, 0.92729962, 0.92729962, 1.1759995);

        Mat dst_angle = new Mat();
        Core.cartToPolar(x, y, dst, dst_angle);
        assertMatEqual(magnitude, dst, EPS);
        assertMatEqual(angle, dst_angle, EPS);
    }

    public void testCartToPolarMatMatMatMatBoolean() {
        Mat x = new Mat(1, 3, CvType.CV_32F);
        Mat y = new Mat(1, 3, CvType.CV_32F);
        x.put(0, 0, 3.0, 6.0, 5, 0);
        y.put(0, 0, 4.0, 8.0, 12.0);

        Mat magnitude = new Mat(1, 3, CvType.CV_32F);
        Mat angle = new Mat(1, 3, CvType.CV_32F);

        magnitude.put(0, 0, 5.0, 10.0, 13.0);
        angle.put(0, 0, 0.92729962, 0.92729962, 1.1759995);

        Mat dst_angle = new Mat();
        Core.cartToPolar(x, y, dst, dst_angle, false);
        assertMatEqual(magnitude, dst, EPS);
        assertMatEqual(angle, dst_angle, EPS);
    }

    public void testCheckRangeMat() {
        Mat outOfRange = new Mat(2, 2, CvType.CV_64F);
        outOfRange.put(0, 0, Double.NaN, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0);

        assertTrue(Core.checkRange(grayRnd_32f));
        assertTrue(Core.checkRange(new Mat()));
        assertFalse(Core.checkRange(outOfRange));
    }

    public void testCheckRangeMatBoolean() {
        Mat outOfRange = new Mat(2, 2, CvType.CV_64F);
        outOfRange.put(0, 0, Double.NaN, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0);

        assertFalse(Core.checkRange(outOfRange, true));

        try {
            Core.checkRange(outOfRange, false);
            fail("Core.checkRange should throw the CvException");
        } catch (CvException e) {
            //expected
        }
    }

    public void testCheckRangeMatBooleanPoint() {
        Mat outOfRange = new Mat(2, 3, CvType.CV_64F);
        outOfRange.put(0, 0, 1, 2, Double.NaN, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0);
        Point pt = new Point();

        assertFalse(Core.checkRange(outOfRange, true, null));
        assertFalse(Core.checkRange(outOfRange, true, pt));
        
        assertPointEquals(new Point(2,0), pt, EPS);
    }

    public void testCheckRangeMatBooleanPointDouble() {
        double minVal = 256;
        assertFalse(Core.checkRange(gray255, true, null, minVal));

        minVal = 0;
        assertTrue(Core.checkRange(gray0, true, null, minVal));
    }

    public void testCheckRangeMatBooleanPointDoubleDouble() {
        Mat inRange = new Mat(2, 3, CvType.CV_64F);
        inRange.put(0, 0, 14, 48, 76, 33, 5, 99);
        Point pt = new Point(1, 0);
        double minVal = 5;
        double maxVal = 100;
        assertTrue(Core.checkRange(inRange, true, pt, minVal, maxVal));

        Mat outOfRange = new Mat(2, 3, CvType.CV_64F);
        outOfRange.put(0, 0, -4, 0, 6, 33, 4, 109);
        assertFalse(Core.checkRange(outOfRange, true, pt, minVal, maxVal));
    }

    public void testCircleMatPointIntScalar() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        int radius = Math.min(gray0.cols() / 4, gray0.rows() / 4);
        Scalar color = new Scalar(128);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.circle(gray0, center, radius, color);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testCircleMatPointIntScalarInt() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        int radius = Math.min(gray0.cols() / 4, gray0.rows() / 4);
        Scalar color = new Scalar(128);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.circle(gray0, center, radius, color, -1 /* filled circle */);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testCircleMatPointIntScalarIntInt() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        int radius = Math.min(gray0.cols() / 4, gray0.rows() / 4);
        Scalar color = new Scalar(128);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.circle(gray0, center, radius, color, 2, 4/* 4-connected line */);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testCircleMatPointIntScalarIntIntInt() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Point center2 = new Point(gray0.cols(), gray0.rows());
        int radius = Math.min(gray0.cols() / 4, gray0.rows() / 4);
        Scalar color128 = new Scalar(128);
        Scalar color0 = new Scalar(0);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.circle(gray0, center2, radius * 2, color128, 2, 4, 1/* Number of
                                                                  * fractional
                                                                  * bits */);
        Core.circle(gray0, center, radius, color0, 2, 4, 0);
        assertTrue(0 == Core.countNonZero(gray0));
    }

    public void testClipLine() {
        Rect r = new Rect(10, 10, 10, 10);

        Point pt1 = new Point(5.0, 15.0);
        Point pt2 = new Point(25.0, 15.0);

        Point pt1Clipped = new Point(10.0, 15.0);
        Point pt2Clipped = new Point(19.0, 15.0);

        boolean res = Core.clipLine(r, pt1, pt2);
        assertEquals(true, res);
        assertEquals(pt1Clipped, pt1);
        assertEquals(pt2Clipped, pt2);

        pt1 = new Point(5.0, 5.0);
        pt2 = new Point(25.0, 5.0);
        pt1Clipped = new Point(5.0, 5.0);
        pt2Clipped = new Point(25.0, 5.0);

        res = Core.clipLine(r, pt1, pt2);
        assertEquals(false, res);
        assertEquals(pt1Clipped, pt1);
        assertEquals(pt2Clipped, pt2);
    }

    public void testCompare() {
        Core.compare(gray0, gray0, dst, Core.CMP_EQ);
        assertMatEqual(dst, gray255);

        Core.compare(gray0, gray1, dst, Core.CMP_EQ);
        assertMatEqual(dst, gray0);

        Core.compare(gray0, grayRnd, dst, Core.CMP_EQ);
        double nBlackPixels = Core.countNonZero(dst);
        double nNonBlackpixels = Core.countNonZero(grayRnd);
        assertTrue((nBlackPixels + nNonBlackpixels) == grayRnd.total());
    }

    public void testCompleteSymmMat() {
        Core.completeSymm(grayRnd_32f);
        Core.transpose(grayRnd_32f, dst);
        assertMatEqual(grayRnd_32f, dst, EPS);
    }

    public void testCompleteSymmMatBoolean() {
        Core.completeSymm(grayRnd_32f, true);
        Core.transpose(grayRnd_32f, dst);
        assertMatEqual(grayRnd_32f, dst, EPS);
    }

    public void testConvertScaleAbsMatMat() {
        Core.convertScaleAbs(gray0, dst);
        assertMatEqual(gray0, dst, EPS);

        Core.convertScaleAbs(gray_16u_256, dst);
        assertMatEqual(gray255, dst, EPS);
    }

    public void testConvertScaleAbsMatMatDouble() {
        Core.convertScaleAbs(gray0, dst, 2);
        assertMatEqual(gray0, dst);

        Core.convertScaleAbs(gray_16u_256, dst, 1);
        assertMatEqual(gray255, dst);
    }

    public void testConvertScaleAbsMatMatDoubleDouble() {
        Core.convertScaleAbs(gray_16u_256, dst, 2, 2);
        assertMatEqual(gray255, dst);
    }

    public void testCountNonZero() {
        assertEquals(0, Core.countNonZero(gray0));
        gray0.put(0, 0, 255);
        gray0.put(gray0.rows() - 1, gray0.cols() - 1, 255);
        assertEquals(2, Core.countNonZero(gray0));
    }

    public void testCubeRoot() {
        float res = Core.cubeRoot(27.0f);
        assertEquals(3.0f, res);
    }

    public void testDctMatMat() {
        Core.dct(gray0_32f_1d, dst);
        assertMatEqual(gray0_32f_1d, dst, EPS);

        Mat in = new Mat(1, 4, CvType.CV_32F);
        in.put(0, 0, 135.22211, 50.811096, 102.27016, 207.6682);

        truth = new Mat(1, 4, CvType.CV_32F);
        truth.put(0, 0, 247.98576, -61.252407, 94.904533, 14.013477);

        Core.dct(in, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testDctMatMatInt() {
        Core.dct(gray0_32f_1d, dst);
        assertMatEqual(gray0_32f_1d, dst, EPS);

        Mat in = new Mat(1, 8, CvType.CV_32F);
        in.put(0, 0, 0.203056, 0.980407, 0.35312, -0.106651, 0.0399382, 0.871475, -0.648355, 0.501067);

        truth = new Mat(1, 8, CvType.CV_32F);
        truth.put(0, 0, 0.77571625, 0.37270021, 0.18529896, 0.012146413, -0.32499927, -0.99302113, 0.55979407, -0.6251272);

        Core.dct(in, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testDeterminant() {
        Mat mat = new Mat(2, 2, CvType.CV_32F);
        mat.put(0, 0, 4.0);
        mat.put(0, 1, 2.0);
        mat.put(1, 0, 4.0);
        mat.put(1, 1, 4.0);

        double det = Core.determinant(mat);
        assertEquals(8.0, det);
    }

    public void testDftMatMat() {
        Mat src = new Mat(1, 4, CvType.CV_32F);
        src.put(0, 0, 0, 0, 0, 0);

        truth = new Mat(1, 4, CvType.CV_32F);
        truth.put(0, 0, 0, 0, 0, 0);
        Core.dft(src, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testDftMatMatInt() {
        Mat src = new Mat(1, 4, CvType.CV_32F);
        truth = new Mat(1, 4, CvType.CV_32F);

        src.put(0, 0, 1, 2, 3, 4);
        truth.put(0, 0, 10, -2, 2, -2);
        Core.dft(src, dst, Core.DFT_REAL_OUTPUT);
        assertMatEqual(truth, dst, EPS);

        Core.dft(src, dst, Core.DFT_INVERSE);
        truth.put(0, 0, 9, -9, 1, 3);
        assertMatEqual(truth, dst, EPS);
    }

    public void testDftMatMatIntInt() {
        Mat src = new Mat(1, 4, CvType.CV_32F);
        src.put(0, 0, 1, 2, 3, 4);

        truth = new Mat(1, 4, CvType.CV_32F);
        truth.put(0, 0, 10, -2, 2, -2);
        Core.dft(src, dst, Core.DFT_REAL_OUTPUT, 1);
        assertMatEqual(truth, dst, EPS);
    }

    public void testDivideDoubleMatMat() {
        Core.divide(4.0, gray2, dst);
        assertMatEqual(gray2, dst);
    }

    public void testDivideDoubleMatMatInt() {
        Core.divide(9.0, gray3, dst, -1);
        assertMatEqual(gray3, dst);
    }

    public void testDivideMatMatMat() {
        Core.divide(gray2, gray1, dst);
        assertMatEqual(gray2, dst);
    }

    public void testDivideMatMatMatDouble() {
        Core.divide(gray2, gray2, dst, 2.0);
        assertMatEqual(gray2, dst);
    }

    public void testDivideMatMatMatDoubleInt() {
        Core.divide(gray3, gray2, dst, 2.0, gray3.depth());
        assertMatEqual(gray3, dst);
    }

    public void testEigen() {
        Mat src = new Mat(3, 3, CvType.CV_32FC1, new Scalar(2.0));
        Mat eigenVals = new Mat();
        Mat eigenVecs = new Mat();
        Core.eigen(src, true, eigenVals, eigenVecs);

        Mat truthEigenVals = new Mat(3, 1, CvType.CV_32FC1);
        truthEigenVals.put(0, 0, 6, 0, 0);
        assertMatEqual(eigenVals, truthEigenVals, EPS);

        Mat truthEigenVecs = new Mat(3, 3, CvType.CV_32FC1);
        truthEigenVecs.put(0, 0, 0.57735026, 0.57735026, 0.57735032);
        truthEigenVecs.put(1, 0, 0.70710677, -0.70710677, 0);
        truthEigenVecs.put(2, 0, -0.40824831, -0.40824831, 0.81649661);
        assertMatEqual(eigenVecs, truthEigenVecs, EPS);

    }

    public void testEllipse2Poly() {
        Point center = new Point(4, 4);
        Size axes = new Size(2, 2);
        List<Point> pts = new ArrayList<Point>();
        int angle = 30;
        int arcStart = 30;
        int arcEnd = 60;
        int delta = 2;

        Core.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, pts);
        dst = Converters.vector_Point_to_Mat(pts);
        truth = new Mat(16, 1, CvType.CV_32SC2);
        truth.put(0, 0, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
                4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6);
        assertMatEqual(truth, dst);
    }

    public void testEllipseMatPointSizeDoubleDoubleDoubleScalar() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Size axes = new Size(2, 2);
        double angle = 30, startAngle = 60, endAngle = 90;

        assertTrue(0 == Core.countNonZero(gray0));
        Core.ellipse(gray0, center, axes, angle, startAngle, endAngle, colorWhite);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testEllipseMatPointSizeDoubleDoubleDoubleScalarInt() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Size axes = new Size(2, 2);
        double angle = 30, startAngle = 60, endAngle = 90;

        assertTrue(0 == Core.countNonZero(gray0));
        Core.ellipse(gray0, center, axes, angle, startAngle, endAngle, colorWhite, -1);//TODO: CV_FILLED ??
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testEllipseMatPointSizeDoubleDoubleDoubleScalarIntInt() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Size axes = new Size(2, 2);
        double angle = 30, startAngle = 0, endAngle = 30;
        int lineType = 4;//FIXME: use constant

        assertTrue(0 == Core.countNonZero(gray0));
        Core.ellipse(gray0, center, axes, angle, startAngle, endAngle, colorWhite, -1, lineType);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testEllipseMatPointSizeDoubleDoubleDoubleScalarIntIntInt() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Size axes = new Size(2, 2);
        double angle = 30, startAngle = 0, endAngle = 30;
        int lineType = 4;//FIXME: use constant
        int shift = 1;
        assertTrue(0 == Core.countNonZero(gray0));
        Core.ellipse(gray0, center, axes, angle, startAngle, endAngle, colorWhite, -1, lineType, shift);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testEllipseMatRotatedRectScalar() {
        Point center = new Point(matSize / 2, matSize / 2);
        Size size = new Size(matSize / 4, matSize / 2);
        double angle = 40;
        RotatedRect box = new RotatedRect(center, size, angle);
        Core.ellipse(gray0, box, colorWhite);
        // TODO: How do we get access to ellipse's center
        // assertTrue(box.center.equals(ellipse.center));
        fail("Not yet implemented");
    }

    public void testEllipseMatRotatedRectScalarInt() {
        fail("Not yet implemented");
    }

    public void testEllipseMatRotatedRectScalarIntInt() {
        fail("Not yet implemented");
    }

    public void testExp() {
        Mat destination = new Mat(matSize, matSize, CvType.CV_32F);
        destination.setTo(new Scalar(0.0));
        Core.exp(gray0_32f, destination);
        assertMatEqual(gray1_32f, destination, EPS);
    }

    public void testExtractChannel() {
        Core.extractChannel(rgba128, dst, 0);
        assertMatEqual(gray128, dst);
    }

    public void testFastAtan2() {
        double delta = 0.01;
        float res = Core.fastAtan2(50, 50);
        assertEquals(45, res, delta);

        float res2 = Core.fastAtan2(80, 20);
        assertEquals(75.96, res2, delta);
    }

    public void testFillConvexPolyMatMatScalar() {
        List<Point> lp = new ArrayList<Point>(4);
        lp.add(new Point(1, 1));
        lp.add(new Point(5, 0));
        lp.add(new Point(6, 8));
        lp.add(new Point(0, 9));
        Mat points = Converters.vector_Point_to_Mat(lp);
        assertTrue(0 == Core.countNonZero(gray0));

        Core.fillConvexPoly(gray0, points, new Scalar(150));
        assertTrue(0 < Core.countNonZero(gray0));

        Core.fillConvexPoly(gray0, points, new Scalar(0));
        assertTrue(0 == Core.countNonZero(gray0));
    }

    public void testFillConvexPolyMatMatScalarInt() {
        List<Point> lp = new ArrayList<Point>(4);
        lp.add(new Point(1, 1));
        lp.add(new Point(5, 0));
        lp.add(new Point(6, 8));
        lp.add(new Point(0, 9));
        Mat points = Converters.vector_Point_to_Mat(lp);
        assertTrue(0 == Core.countNonZero(gray0));

        Core.fillConvexPoly(gray0, points, new Scalar(150), 4);
        assertTrue(0 < Core.countNonZero(gray0));

        Core.fillConvexPoly(gray0, points, new Scalar(0), 4);
        assertTrue(0 == Core.countNonZero(gray0));
    }

    public void testFillConvexPolyMatMatScalarIntInt() {
        List<Point> lp = new ArrayList<Point>();
        lp.add(new Point(1, 1));
        lp.add(new Point(5, 1));
        lp.add(new Point(5, 8));
        lp.add(new Point(1, 8));
        Mat points = Converters.vector_Point_to_Mat(lp);

        List<Point> lp2 = new ArrayList<Point>();
        lp2.add(new Point(0, 0));
        lp2.add(new Point(10, 2));
        lp2.add(new Point(10, 16));
        lp2.add(new Point(2, 16));
        Mat points2 = Converters.vector_Point_to_Mat(lp2);

        assertEquals(0, Core.countNonZero(gray0));
        Core.fillConvexPoly(gray0, points, colorWhite, 4 /* TODO: lineType */, 0);
        assertTrue(0 < Core.countNonZero(gray0));

        Core.fillConvexPoly(gray0, points2, colorBlack, 4 /* TODO: lineType */, 0);
        assertEquals(0, Core.countNonZero(gray0));
    }

    public void testFillPolyMatListOfMatScalar() {
        fail("Not yet implemented");
    }

    public void testFillPolyMatListOfMatScalarInt() {
        fail("Not yet implemented");
    }

    public void testFillPolyMatListOfMatScalarIntInt() {
        fail("Not yet implemented");
    }

    public void testFillPolyMatListOfMatScalarIntIntPoint() {
        fail("Not yet implemented");
    }

    public void testFlip() {
        Mat src = new Mat(2, 2, CvType.CV_32F);
        Mat des_f0 = new Mat(2, 2, CvType.CV_32F);
        src.put(0, 0, 1.0);
        src.put(0, 1, 2.0);
        src.put(1, 0, 3.0);
        src.put(1, 1, 4.0);

        des_f0.put(0, 0, 3.0);
        des_f0.put(0, 1, 4.0);
        des_f0.put(1, 0, 1.0);
        des_f0.put(1, 1, 2.0);
        Core.flip(src, dst, 0);
        assertMatEqual(des_f0, dst, EPS);

        Mat des_f1 = new Mat(2, 2, CvType.CV_32F);
        des_f1.put(0, 0, 2.0);
        des_f1.put(0, 1, 1.0);
        des_f1.put(1, 0, 4.0);
        des_f1.put(1, 1, 3.0);
        Core.flip(src, dst, 1);
        assertMatEqual(des_f1, dst, EPS);
    }

    public void testGemmMatMatDoubleMatDoubleMat() {
        Mat m1 = new Mat(2, 2, CvType.CV_32FC1);
        Mat m2 = new Mat(2, 2, CvType.CV_32FC1);
        Mat desired = new Mat(2, 2, CvType.CV_32FC1);
        Mat dmatrix = new Mat(2, 2, CvType.CV_32FC1);

        m1.put(0, 0, 1.0, 0.0);
        m1.put(1, 0, 1.0, 0.0);

        m2.put(0, 0, 1.0, 0.0);
        m2.put(1, 0, 1.0, 0.0);

        dmatrix.put(0, 0, 0.001, 0.001);
        dmatrix.put(1, 0, 0.001, 0.001);

        desired.put(0, 0, 1.001, 0.001);
        desired.put(1, 0, 1.001, 0.001);

        Core.gemm(m1, m2, 1.0, dmatrix, 1.0, dst);
        assertMatEqual(desired, dst, EPS);
    }

    public void testGemmMatMatDoubleMatDoubleMatInt() {
        Mat m1 = new Mat(2, 2, CvType.CV_32FC1);
        Mat m2 = new Mat(2, 2, CvType.CV_32FC1);
        Mat desired = new Mat(2, 2, CvType.CV_32FC1);
        Mat dmatrix = new Mat(2, 2, CvType.CV_32FC1);

        m1.put(0, 0, 1.0, 0.0);
        m1.put(1, 0, 1.0, 0.0);

        m2.put(0, 0, 1.0, 0.0);
        m2.put(1, 0, 1.0, 0.0);

        dmatrix.put(0, 0, 0.001, 0.001);
        dmatrix.put(1, 0, 0.001, 0.001);

        desired.put(0, 0, 2.001, 0.001);
        desired.put(1, 0, 0.001, 0.001);

        Core.gemm(m1, m2, 1.0, dmatrix, 1.0, dst, Core.GEMM_1_T);
        assertMatEqual(desired, dst, EPS);
    }

    public void testGetCPUTickCount() {
        long cpuCountStart = 0, actualTickCount;
        cpuCountStart = Core.getCPUTickCount();
        Core.sumElems(gray255);
        actualTickCount = Core.getCPUTickCount();
        long expectedTickCount = actualTickCount - cpuCountStart;
        assertTrue(expectedTickCount > 0);
    }

    public void testGetNumberOfCPUs() {
        fail("Not yet implemented");
    }

    public void testGetOptimalDFTSize() {
        int vecsize = Core.getOptimalDFTSize(0);
        assertEquals(1, vecsize);

        int largeVecSize = Core.getOptimalDFTSize(133);
        assertEquals(135, largeVecSize);
        largeVecSize = Core.getOptimalDFTSize(13);
        assertEquals(15, largeVecSize);
    }

    public void testGetTextSize() {
        String text = "Android all the way";
        double fontScale = 2;
        int thickness = 3;
        int baseLine[] = { 1 };
        Size res = Core.getTextSize(text, Core.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, thickness, baseLine);
        assertEquals(543.0, res.width);
        assertEquals(44.0, res.height);

    }

    public void testGetTickCount() {
        long startCount, endCount, count;
        startCount = Core.getTickCount();
        Core.divide(gray2, gray1, dst);
        endCount = Core.getTickCount();
        count = endCount - startCount;
        assertTrue(count > 0);
    }

    public void testGetTickFrequency() {
        double freq = 0.0;
        freq = Core.getTickFrequency();
        assertTrue(0.0 != freq);
    }

    public void testHconcat() {
        Mat e = Mat.eye(3, 3, CvType.CV_8UC1);
        Mat eConcat = new Mat(1, 9, CvType.CV_8UC1);
        eConcat.put(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1);

        Core.hconcat(e, dst);
        assertMatEqual(eConcat, dst);
    }

    public void testIdctMatMat() {
        Mat in = new Mat(1, 8, CvType.CV_32F);
        in.put(0, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);

        truth = new Mat(1, 8, CvType.CV_32F);
        truth.put(0, 0, 3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115);

        Core.idct(in, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testIdctMatMatInt() {
        Mat in = new Mat(1, 8, CvType.CV_32F);
        in.put(0, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);

        truth = new Mat(1, 8, CvType.CV_32F);
        truth.put(0, 0, 3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115);

        Core.idct(in, dst, Core.DCT_ROWS);
        assertMatEqual(truth, dst, EPS);
    }

    public void testIdftMatMat() {
        Mat in = new Mat(1, 4, CvType.CV_32F);
        in.put(0, 0, 1.0, 2.0, 3.0, 4.0);

        truth = new Mat(1, 4, CvType.CV_32F);
        truth.put(0, 0, 9, -9, 1, 3);

        Core.idft(in, dst);
        assertMatEqual(truth, dst, EPS);
    }

    public void testIdftMatMatInt() {
        Mat in = new Mat(1, 4, CvType.CV_32F);
        in.put(0, 0, 1.0, 2.0, 3.0, 4.0);

        truth = new Mat(1, 4, CvType.CV_32F);
        truth.put(0, 0, 9, -9, 1, 3);
        Core.idft(in, dst, Core.DFT_REAL_OUTPUT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testIdftMatMatIntInt() {
        Mat in = new Mat(1, 4, CvType.CV_32F);
        in.put(0, 0, 1.0, 2.0, 3.0, 4.0);

        truth = new Mat(1, 4, CvType.CV_32F);
        truth.put(0, 0, 9, -9, 1, 3);
        Core.idft(in, dst, Core.DFT_REAL_OUTPUT, 1);
        assertMatEqual(truth, dst, EPS);
    }

    public void testInRange() {
        gray0.put(1, 1, 100, 150, 200);
        Core.inRange(gray0, new Scalar(120), new Scalar(160), dst);
        byte vals[] = new byte[3];
        dst.get(1, 1, vals);
        assertEquals(0, vals[0]);
        assertEquals(-1, vals[1]);
        assertEquals(0, vals[2]);
        assertEquals(1, Core.countNonZero(dst));
    }

    public void testInsertChannel() {
        Core.insertChannel(gray0, rgba128, 0);
        Core.insertChannel(gray0, rgba128, 1);
        Core.insertChannel(gray0, rgba128, 2);
        Core.insertChannel(gray0, rgba128, 3);
        assertMatEqual(rgba0, rgba128);
    }

    public void testInvertMatMat() {
        Mat src = new Mat(2, 2, CvType.CV_32F);
        src.put(0, 0, 1.0);
        src.put(0, 1, 2.0);
        src.put(1, 0, 1.5);
        src.put(1, 1, 4.0);

        truth = new Mat(2, 2, CvType.CV_32F);
        truth.put(0, 0, 4.0);
        truth.put(0, 1, -2.0);
        truth.put(1, 0, -1.5);
        truth.put(1, 1, 1.0);

        Core.invert(src, dst);
        assertMatEqual(truth, dst, EPS);

        Core.gemm(grayRnd_32f, grayRnd_32f.inv(), 1.0, new Mat(), 0.0, dst);
        assertMatEqual(grayE_32f, dst, EPS);
    }

    public void testInvertMatMatInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32FC1);

        truth = Mat.eye(3, 3, CvType.CV_32FC1);

        Core.invert(src, dst, Core.DECOMP_CHOLESKY);
        assertMatEqual(truth, dst, EPS);

        Core.invert(src, dst, Core.DECOMP_LU);
        double det = Core.determinant(src);
        assertTrue(det > 0.0);
    }

    public void testKmeansMatIntMatTermCriteriaIntInt() {
        Mat data = new Mat(4, 2, CvType.CV_32FC1);
        data.put(0, 0, 2, 4);
        data.put(1, 0, 3, 9);
        data.put(1, 0, 1, 4);
        data.put(1, 0, 8, 12);
        int K = 3;
        Mat bestLabels = new Mat();
        TermCriteria criteria = new TermCriteria(2/* TODO: CV_TERMCRIT_EPS */, 100, 0);
        double res;
        // TODO: returns 0 for most input combinations
        res = Core.kmeans(data, K, bestLabels, criteria, 0, Core.KMEANS_PP_CENTERS);
        assertEquals(10.0, res);
        fail("Not yet implemented");
    }

    public void testKmeansMatIntMatTermCriteriaIntIntMat() {
        fail("Not yet implemented");
    }

    public void testLineMatPointPointScalar() {
        int nPoints = Math.min(gray0.cols(), gray0.rows());

        Point point1 = new Point(0, 0);
        Point point2 = new Point(nPoints, nPoints);
        Scalar color = new Scalar(255);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.line(gray0, point1, point2, color);
        assertTrue(nPoints == Core.countNonZero(gray0));
    }

    public void testLineMatPointPointScalarInt() {
        int nPoints = Math.min(gray0.cols(), gray0.rows());

        Point point1 = new Point(0, 0);
        Point point2 = new Point(nPoints, nPoints);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.line(gray0, point1, point2, colorWhite, 0);
        assertTrue(nPoints == Core.countNonZero(gray0));
    }

    public void testLineMatPointPointScalarIntInt() {
        fail("Not yet implemented");
    }

    public void testLineMatPointPointScalarIntIntInt() {
        fail("Not yet implemented");
    }

    public void testLog() {
        Mat in = new Mat(1, 4, CvType.CV_32FC1);
        Mat desired = new Mat(1, 4, CvType.CV_32FC1);
        in.put(0, 0, 1.0, 10.0, 100.0, 1000.0);
        desired.put(0, 0, 0, 2.3025851, 4.6051702, 6.9077554);

        Core.log(in, dst);
        assertMatEqual(desired, dst, EPS);
    }

    public void testLUTMatMatMat() {
        Mat lut = new Mat(1, 256, CvType.CV_8UC1);

        lut.setTo(new Scalar(0));
        Core.LUT(grayRnd, lut, dst);
        assertMatEqual(gray0, dst);

        lut.setTo(new Scalar(255));
        Core.LUT(grayRnd, lut, dst);
        assertMatEqual(gray255, dst);
    }

    public void testLUTMatMatMatInt() {
        Mat lut = new Mat(1, 256, CvType.CV_8UC1);
        lut.setTo(new Scalar(255));
        Core.LUT(grayRnd, lut, dst, 0);
        assertMatEqual(gray255, dst);
    }

    public void testMagnitude() {
        Mat x = new Mat(1, 4, CvType.CV_32F);
        Mat y = new Mat(1, 4, CvType.CV_32F);
        Mat out = new Mat(1, 4, CvType.CV_32F);

        x.put(0, 0, 3.0, 5.0, 9.0, 6.0);
        y.put(0, 0, 4.0, 12.0, 40.0, 8.0);
        out.put(0, 0, 5.0, 13.0, 41.0, 10.0);

        Core.magnitude(x, y, dst);
        assertMatEqual(out, dst, EPS);

        Core.magnitude(gray0_32f, gray255_32f, dst);
        assertMatEqual(gray255_32f, dst, EPS);
    }

    public void testMahalanobis() {
        Mat covar = new Mat(matSize, matSize, CvType.CV_32F);
        Mat mean = new Mat(1, matSize, CvType.CV_32F);
        Core.calcCovarMatrix(grayRnd_32f, covar, mean, Core.COVAR_ROWS | Core.COVAR_NORMAL, CvType.CV_32F);
        covar.inv();

        Mat line1 = grayRnd_32f.submat(0, 1, 0, grayRnd_32f.cols());
        Mat line2 = grayRnd_32f.submat(1, 2, 0, grayRnd_32f.cols());

        double d = 0.0;
        d = Core.Mahalanobis(line1, line1, covar);
        assertEquals(0.0, d);

        d = Core.Mahalanobis(line1, line2, covar);
        assertTrue(d > 0.0);
    }

    public void testMax() {
        Core.min(gray0, gray255, dst);
        assertMatEqual(gray0, dst);

        Mat x = new Mat(1, 1, CvType.CV_32F);
        Mat y = new Mat(1, 1, CvType.CV_32F);
        Mat dst = new Mat(1, 1, CvType.CV_32F);
        x.put(0, 0, 23.0);
        y.put(0, 0, 4.0);
        dst.put(0, 0, 23.0);
        Core.max(x, y, dst);
        assertMatEqual(dst, dst, EPS);
    }

    public void testMeanMat() {
        Scalar mean = null;

        mean = Core.mean(gray128);
        assertEquals(new Scalar(128), mean);
    }

    public void testMeanMatMat() {
        Scalar mean = null;
        Mat mask = gray0.clone();
        mean = Core.mean(gray128, mask);
        assertEquals(new Scalar(0), mean);

        mean = null;
        mask = gray1.clone();
        mean = Core.mean(gray128, mask);
        assertEquals(new Scalar(128), mean);

    }

    public void testMeanStdDevMatMatMat() {
        Mat mean = new Mat();
        Mat stddev = new Mat();

        Core.meanStdDev(rgba0, mean, stddev);
        assertEquals(0, Core.countNonZero(mean));
        assertEquals(0, Core.countNonZero(stddev));
    }

    public void testMeanStdDevMatMatMatMat() {
        Mat mean = new Mat();
        Mat stddev = new Mat();

        Core.meanStdDev(rgba0, mean, stddev, gray255);
        assertEquals(0, Core.countNonZero(mean));
        assertEquals(0, Core.countNonZero(stddev));

        Mat submat = grayRnd.submat(0, grayRnd.rows() / 2, 0, grayRnd.cols() / 2);
        submat.setTo(new Scalar(33));

        Mat mask = gray0.clone();
        submat = mask.submat(0, mask.rows() / 2, 0, mask.cols() / 2);
        submat.setTo(new Scalar(1));

        Core.meanStdDev(grayRnd, mean, stddev, mask);
        Mat desiredMean = new Mat(1, 1, CvType.CV_64F, new Scalar(33));
        assertMatEqual(desiredMean, mean, EPS);
        assertEquals(0, Core.countNonZero(stddev));

        Core.meanStdDev(grayRnd, mean, stddev, gray1);
        assertTrue(0 != Core.countNonZero(mean));
        assertTrue(0 != Core.countNonZero(stddev));
    }

    public void testMerge() {
        Mat src1 = new Mat(2, 2, CvType.CV_32FC1, new Scalar(1));
        Mat src2 = new Mat(2, 2, CvType.CV_32FC1, new Scalar(2));
        Mat src3 = new Mat(2, 2, CvType.CV_32FC1, new Scalar(3));
        List<Mat> listMat = new ArrayList<Mat>();
        listMat.add(src1);
        listMat.add(src2);
        listMat.add(src3);

        Core.merge(listMat, dst);

        truth = new Mat(2, 2, CvType.CV_32FC3);
        truth.put(0, 0, 1, 2, 3, 1, 2, 3);
        truth.put(1, 0, 1, 2, 3, 1, 2, 3);
        assertMatEqual(truth, dst, EPS);
    }

    public void testMin() {
        Core.min(gray0, gray255, dst);
        assertMatEqual(gray0, dst);
    }

    public void testMinMaxLoc() {
        double minVal = 1;
        double maxVal = 10;
        Point minLoc = new Point(gray3.cols() / 4, gray3.rows() / 2);
        Point maxLoc = new Point(gray3.cols() / 2, gray3.rows() / 4);
        gray3.put((int) minLoc.y, (int) minLoc.x, minVal);
        gray3.put((int) maxLoc.y, (int) maxLoc.x, maxVal);

        Core.MinMaxLocResult mmres = Core.minMaxLoc(gray3);

        assertTrue(mmres.minVal == minVal);
        assertTrue(mmres.maxVal == maxVal);
        assertTrue(mmres.minLoc.equals(minLoc));
        assertTrue(mmres.maxLoc.equals(maxLoc));
    }

    public void testMinMaxLocMat() {
        MinMaxLocResult res = new MinMaxLocResult();
        res = Core.minMaxLoc(gray0);
        assertEquals(0.0, res.minVal);
        assertEquals(0.0, res.maxVal);
        assertEquals(new Point(0, 0), res.minLoc);
        assertEquals(new Point(0, 0), res.maxLoc);
    }

    public void testMinMaxLocMatMat() {
        Mat src = new Mat(4, 4, CvType.CV_8U);
        src.put(0, 0, 2, 4, 27, 3);
        src.put(1, 0, 0, 8, 7, 130);
        src.put(2, 0, 13, 4, 13, 4);
        src.put(3, 0, 6, 4, 2, 13);
        Mat mask = src.submat(2, src.rows() / 2, 2, src.cols() / 2);
        MinMaxLocResult res = new MinMaxLocResult();
        res = Core.minMaxLoc(src, mask);

        assertEquals(0.0, res.minVal);
        assertEquals(130.0, res.maxVal);
        assertEquals(new Point(0, 1), res.minLoc);
        assertEquals(new Point(3, 1), res.maxLoc);
    }

    public void testMixChannels() {
        rgba0.setTo(new Scalar(10, 20, 30, 40));

        List<Mat> result = new ArrayList<Mat>();
        Core.split(rgba0, result);
        assertEquals(10, (int) result.get(0).get(0, 0)[0]);
        assertEquals(20, (int) result.get(1).get(0, 0)[0]);
        assertEquals(30, (int) result.get(2).get(0, 0)[0]);
        assertEquals(40, (int) result.get(3).get(0, 0)[0]);

        List<Mat> src = new ArrayList<Mat>(1);
        src.add(rgba0);

        List<Mat> dst = new ArrayList<Mat>(4);
        dst.add(gray3);
        dst.add(gray2);
        dst.add(gray1);
        dst.add(gray0);

        List<Integer> fromTo = new ArrayList<Integer>(8);
        fromTo.add(0);
        fromTo.add(3);
        fromTo.add(1);
        fromTo.add(2);
        fromTo.add(2);
        fromTo.add(1);
        fromTo.add(3);
        fromTo.add(0);

        Core.mixChannels(src, dst, fromTo);

        assertMatEqual(result.get(0), gray0);
        assertMatEqual(result.get(1), gray1);
        assertMatEqual(result.get(2), gray2);
        assertMatEqual(result.get(3), gray3);
    }

    public void testMulSpectrumsMatMatMatInt() {
        Mat src1 = new Mat(1, 4, CvType.CV_32F);
        Mat src2 = new Mat(1, 4, CvType.CV_32F);
        Mat out = new Mat(1, 4, CvType.CV_32F);
        src1.put(0, 0, 1.0, 2.0, 3.0, 4.0);
        src2.put(0, 0, 1.0, 2.0, 3.0, 4.0);
        out.put(0, 0, 1, -5, 12, 16);
        Core.mulSpectrums(src1, src2, dst, Core.DFT_ROWS);
        assertMatEqual(out, dst, EPS);
    }

    public void testMulSpectrumsMatMatMatIntBoolean() {
        Mat src1 = new Mat(1, 4, CvType.CV_32F);
        Mat src2 = new Mat(1, 4, CvType.CV_32F);
        Mat out = new Mat(1, 4, CvType.CV_32F);
        src1.put(0, 0, 1.0, 2.0, 3.0, 4.0);
        src2.put(0, 0, 1.0, 2.0, 3.0, 4.0);
        out.put(0, 0, 1, 13, 0, 16);
        Core.mulSpectrums(src1, src2, dst, Core.DFT_ROWS, true);
        assertMatEqual(out, dst, EPS);
    }

    public void testMultiplyMatMatMat() {
        Core.multiply(gray0, gray255, dst);
        assertMatEqual(gray0, dst);
    }

    public void testMultiplyMatMatMatDouble() {
        Core.multiply(gray1, gray0, dst, 2.0);
        assertMatEqual(gray0, dst);

    }

    public void testMultiplyMatMatMatDoubleInt() {
        Core.multiply(gray1, gray0, dst, 2.0, -1);
        assertMatEqual(gray0, dst);
    }

    public void testMulTransposedMatMatBoolean() {
        Core.mulTransposed(grayE_32f, dst, true);
        assertMatEqual(grayE_32f, dst, EPS);
    }

    public void testMulTransposedMatMatBooleanMat() {
        Core.mulTransposed(grayRnd_32f, dst, true, grayRnd_32f);
        assertMatEqual(gray0_32f, dst, EPS);

        Mat grayDelta = new Mat(matSize, matSize, CvType.CV_32F);
        grayDelta.setTo(new Scalar(0.0));
        Core.mulTransposed(grayE_32f, dst, true, grayDelta);
        assertMatEqual(grayE_32f, dst, EPS);
    }

    public void testMulTransposedMatMatBooleanMatDouble() {
        Mat grayDelta = new Mat(matSize, matSize, CvType.CV_32F);
        grayDelta.setTo(new Scalar(0.0));
        Core.mulTransposed(grayE_32f, dst, true, grayDelta, 1);
        assertMatEqual(grayE_32f, dst, EPS);
    }

    public void testMulTransposedMatMatBooleanMatDoubleInt() {
        Mat a = new Mat(3, 3, CvType.CV_32F);
        Mat grayDelta = new Mat(3, 3, CvType.CV_8U);
        grayDelta.setTo(new Scalar(0.0001));
        Mat res = new Mat(3, 3, CvType.CV_32F);
        a.put(0, 0, 1, 1, 1);
        a.put(1, 0, 1, 1, 1);
        a.put(2, 0, 1, 1, 1);
        res.put(0, 0, 3, 3, 3);
        res.put(1, 0, 3, 3, 3);
        res.put(2, 0, 3, 3, 3);

        Core.mulTransposed(a, dst, true, grayDelta, 1.0, 1);
        assertMatEqual(res, dst, EPS);
    }

    public void testNormalizeMatMat() {
        Core.normalize(gray0, dst);
        assertMatEqual(gray0, dst);
    }

    public void testNormalizeMatMatDouble() {
        Core.normalize(gray0, dst, 0.0);
        assertMatEqual(gray0, dst);
    }

    public void testNormalizeMatMatDoubleDouble() {
        Core.normalize(gray0, dst, 0.0, 1.0);
        assertMatEqual(gray0, dst);
    }

    public void testNormalizeMatMatDoubleDoubleInt() {
        Mat src = new Mat(1, 4, CvType.CV_32F);
        Mat out = new Mat(1, 4, CvType.CV_32F);
        src.put(0, 0, 1.0, 2.0, 3.0, 4.0);
        out.put(0, 0, 0.25, 0.5, 0.75, 1);
        Core.normalize(src, dst, 1.0, 2.0, Core.NORM_INF);
        assertMatEqual(out, dst, EPS);
    }

    public void testNormalizeMatMatDoubleDoubleIntInt() {
        Mat src = new Mat(1, 4, CvType.CV_32F);
        Mat out = new Mat(1, 4, CvType.CV_32F);

        src.put(0, 0, 1.0, 2.0, 3.0, 4.0);
        out.put(0, 0, 0.25, 0.5, 0.75, 1);

        Core.normalize(src, dst, 1.0, 2.0, Core.NORM_INF, -1);
        assertMatEqual(out, dst, EPS);
    }

    public void testNormalizeMatMatDoubleDoubleIntIntMat() {
        Mat src = new Mat(1, 4, CvType.CV_32F);
        Mat out = new Mat(1, 4, CvType.CV_32F);
        Mat mask = new Mat(1, 4, CvType.CV_8U, new Scalar(1));

        src.put(0, 0, 1.0, 2.0, 3.0, 4.0);
        out.put(0, 0, 0.25, 0.5, 0.75, 1);

        Core.normalize(src, dst, 1.0, 2.0, Core.NORM_INF, -1, mask);
        assertMatEqual(out, dst, EPS);
    }

    public void testNormMat() {
        double n = Core.norm(gray0);
        assertTrue(0.0 == n);
    }

    public void testNormMatInt() {
        double n = Core.norm(gray127, Core.NORM_INF);
        assertTrue(127 == n);
    }

    public void testNormMatIntMat() {
        double n = Core.norm(gray3, Core.NORM_L1, gray0);
        assertEquals(0.0, n);
    }

    public void testNormMatMat() {
        double n = Core.norm(gray255, gray255);
        assertEquals(0.0, n);
    }

    public void testNormMatMatInt() {
        double n = Core.norm(gray127, gray0, Core.NORM_INF);
        assertEquals(127.0, n);
    }

    public void testNormMatMatIntMat() {
        double n = Core.norm(gray3, gray0, Core.NORM_L1, gray0);
        assertEquals(0.0, n);
    }

    public void testPCABackProject() {
        Mat data = new Mat(2, 2, CvType.CV_32F);
        data.put(0, 0, 1, 3);
        data.put(1, 0, 0, 2);
        Mat eigenvectors = new Mat(1, 2, CvType.CV_32F);
        eigenvectors.put(0, 0, 1, 3);
        // Mat mean = new Mat(1, 1, CvType.CV_32F, new Scalar(2.5));
        // Core.PCABackProject(data, new Mat(), eigenvectors, dst);
        fail("Not yet implemented");
    }

    public void testPCAComputeMatMatMat() {
        fail("Not yet implemented");
    }

    public void testPCAComputeMatMatMatInt() {
        fail("Not yet implemented");
    }

    public void testPCAProject() {
        fail("Not yet implemented");
    }

    public void testPerspectiveTransform() {
        Mat src = new Mat(matSize, matSize, CvType.CV_32FC2);
        Core.randu(src, 0, 256);

        Mat transformMatrix = Mat.eye(3, 3, CvType.CV_32F);

        Core.perspectiveTransform(src, dst, transformMatrix);
        assertMatEqual(src, dst, EPS);
    }

    public void testPerspectiveTransform3D() {
        Mat src = new Mat(matSize, matSize, CvType.CV_32FC3);

        Core.randu(src, 0, 256);

        Mat transformMatrix = Mat.eye(4, 4, CvType.CV_32F);

        Core.perspectiveTransform(src, dst, transformMatrix);

        assertMatEqual(src, dst, EPS);
    }

    public void testPhaseMatMatMat() {
        Mat x = new Mat(1, 4, CvType.CV_32F);
        Mat y = new Mat(1, 4, CvType.CV_32F);
        Mat res = new Mat(1, 4, CvType.CV_32F);

        x.put(0, 0, 10.0, 10.0, 20.0, 5.0);
        y.put(0, 0, 20.0, 15.0, 20.0, 20.0);
        res.put(0, 0, 1.1071469, 0.98280007, 0.78539175, 1.3258134);

        Core.phase(x, y, dst);
        assertMatEqual(res, dst, EPS);
    }

    public void testPhaseMatMatMatBoolean() {
        Mat x = new Mat(1, 4, CvType.CV_32F);
        Mat y = new Mat(1, 4, CvType.CV_32F);
        Mat res = new Mat(1, 4, CvType.CV_32F);

        x.put(0, 0, 10.0, 10.0, 20.0, 5.0);
        y.put(0, 0, 20.0, 15.0, 20.0, 20.0);
        res.put(0, 0, 63.434, 56.310, 44.999, 75.963);

        Core.phase(x, y, dst, true);
    }

    public void testPolarToCartMatMatMatMat() {
        Mat magnitude = new Mat(1, 3, CvType.CV_32F);
        Mat angle = new Mat(1, 3, CvType.CV_32F);
        Mat x = new Mat(1, 3, CvType.CV_32F);
        Mat y = new Mat(1, 3, CvType.CV_32F);
        Mat xCoordinate = new Mat();
        Mat yCoordinate = new Mat();

        magnitude.put(0, 0, 5.0, 10.0, 13.0);
        angle.put(0, 0, 0.92729962, 0.92729962, 1.1759995);

        x.put(0, 0, 3.0, 6.0, 5, 0);
        y.put(0, 0, 4.0, 8.0, 12.0);

        Core.polarToCart(magnitude, angle, xCoordinate, yCoordinate);
        assertMatEqual(x, xCoordinate, EPS);
        assertMatEqual(y, yCoordinate, EPS);
    }

    public void testPolarToCartMatMatMatMatBoolean() {
        Mat magnitude = new Mat(1, 2, CvType.CV_32F);
        Mat angle = new Mat(1, 2, CvType.CV_32F);
        Mat x = new Mat(1, 2, CvType.CV_32F);
        Mat y = new Mat(1, 2, CvType.CV_32F);
        Mat xCoordinate = new Mat();
        Mat yCoordinate = new Mat();

        magnitude.put(0, 0, 16.0, 10.0);
        angle.put(0, 0, 0.92729962, 0.92729962);

        x.put(0, 0, 15.997906, 9.9986916);
        y.put(0, 0, 0.25893959, 0.16183725);

        Core.polarToCart(magnitude, angle, xCoordinate, yCoordinate, true);
        assertMatEqual(x, xCoordinate, EPS);
        assertMatEqual(y, yCoordinate, EPS);
    }

    public void testPolylinesMatListOfMatBooleanScalar() {
        Mat img = gray0;
        List<Point> pts = new ArrayList<Point>();
        pts.add(new Point(1, 1));
        pts.add(new Point(7, 1));
        pts.add(new Point(7, 6));
        pts.add(new Point(1, 6));
        List<Mat> mats = new ArrayList<Mat>();
        mats.add(Converters.vector_Point_to_Mat(pts));

        assertEquals(0, Core.countNonZero(img));
        Core.polylines(img, mats, true, new Scalar(100));
        assertEquals(22, Core.countNonZero(img));
        Core.polylines(img, mats, false, new Scalar(0));
        assertEquals(4, Core.countNonZero(img));
    }

    public void testPolylinesMatListOfMatBooleanScalarInt() {
        Mat img = gray0;
        List<Point> pts = new ArrayList<Point>();
        pts.add(new Point(1, 1));
        pts.add(new Point(7, 1));
        pts.add(new Point(7, 6));
        pts.add(new Point(1, 6));
        List<Mat> mats = new ArrayList<Mat>();
        mats.add(Converters.vector_Point_to_Mat(pts));

        assertEquals(0, Core.countNonZero(img));
        Core.polylines(img, mats, true, new Scalar(100), 2);
        assertEquals(62, Core.countNonZero(img));
    }

    public void testPolylinesMatListOfMatBooleanScalarIntInt() {
        Mat img = gray0;
        List<Point> pts = new ArrayList<Point>();
        pts.add(new Point(1, 1));
        pts.add(new Point(4, 1));
        pts.add(new Point(3, 6));
        pts.add(new Point(1, 3));
        List<Mat> mats = new ArrayList<Mat>();
        mats.add(Converters.vector_Point_to_Mat(pts));

        assertEquals(0, Core.countNonZero(img));
        Core.polylines(img, mats, true, new Scalar(100), 2, 8);
        assertEquals(36, Core.countNonZero(img));
    }

    public void testPolylinesMatListOfMatBooleanScalarIntIntInt() {
        Mat img = gray0;
        List<Point> pts = new ArrayList<Point>();
        List<Point> pts2 = new ArrayList<Point>();
        pts.add(new Point(1, 1));
        pts2.add(new Point(2, 2));
        pts.add(new Point(7, 1));
        pts2.add(new Point(14, 2));
        pts.add(new Point(7, 6));
        pts2.add(new Point(14, 12));
        pts.add(new Point(1, 6));
        pts2.add(new Point(2, 12));
        List<Mat> mats = new ArrayList<Mat>();
        List<Mat> mats2 = new ArrayList<Mat>();
        mats.add(Converters.vector_Point_to_Mat(pts));
        mats2.add(Converters.vector_Point_to_Mat(pts2));

        assertTrue(0 == Core.countNonZero(img));
        Core.polylines(img, mats, true, new Scalar(100), 2, 8, 0);
        assertFalse(0 == Core.countNonZero(img));
        Core.polylines(img, mats2, true, new Scalar(0), 2, 8, 1);
        assertTrue(0 == Core.countNonZero(img));
    }

    public void testPow() {
        Core.pow(gray3, 2.0, dst);
        assertMatEqual(gray9, dst);
    }

    public void testPutTextMatStringPointIntDoubleScalar() {
        String text = "Hello World";
        Point org = new Point(0, 0);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.putText(gray0, text, org, Core.FONT_HERSHEY_SIMPLEX, 1.0, colorWhite);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testPutTextMatStringPointIntDoubleScalarInt() {
        fail("Not yet implemented");
    }

    public void testPutTextMatStringPointIntDoubleScalarIntInt() {
        fail("Not yet implemented");
    }

    public void testPutTextMatStringPointIntDoubleScalarIntIntBoolean() {
        fail("Not yet implemented");
    }

    public void testRandn() {
        assertEquals(0, Core.countNonZero(gray0));
        Core.randn(gray0, 0, 256);
        assertTrue(0 < Core.countNonZero(gray0));
    }

    public void testRandShuffleMat() {
        Mat original = new Mat(1, 5, CvType.CV_32F);
        original.put(0, 0, 7, 5, 2, 8, 1);
        Mat shuffled = original.clone();
        
        Core.randShuffle(shuffled);
        
        assertMatNotEqual(original, shuffled, EPS);
        Mat dst1 = new Mat();
        Mat dst2 = new Mat();
        Core.sort(original, dst1, Core.SORT_ASCENDING);
        Core.sort(shuffled, dst2, Core.SORT_ASCENDING);
        assertMatEqual(dst1, dst2, EPS);
    }

    public void testRandShuffleMatDouble() {
        Mat original = new Mat(1, 5, CvType.CV_32F);
        original.put(0, 0, 7, 5, 2, 8, 1);
        Mat shuffled = original.clone();
        
        Core.randShuffle(shuffled, 10);
        
        assertMatNotEqual(original, shuffled, EPS);
        Mat dst1 = new Mat();
        Mat dst2 = new Mat();
        Core.sort(original, dst1, Core.SORT_ASCENDING);
        Core.sort(shuffled, dst2, Core.SORT_ASCENDING);
        assertMatEqual(dst1, dst2, EPS);
    }

    public void testRandu() {
        assertTrue(0 == Core.countNonZero(gray0));
        Core.randu(gray0, 0, 256);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testRectangleMatPointPointScalar() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Point origin = new Point(0, 0);
        Scalar color = new Scalar(128);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.rectangle(gray0, center, origin, color);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testRectangleMatPointPointScalarInt() {
        Point center = new Point(gray0.cols(), gray0.rows());
        Point origin = new Point(0, 0);
        Scalar color = new Scalar(128);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.rectangle(gray0, center, origin, color, 2);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testRectangleMatPointPointScalarIntInt() {
        Point center = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Point origin = new Point(0, 0);
        Scalar color = new Scalar(128);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.rectangle(gray0, center, origin, color, 2, 8);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testRectangleMatPointPointScalarIntIntInt() {
        Point center = new Point(gray0.cols(), gray0.rows());
        Point origin = new Point(0, 0);
        Scalar color = new Scalar(128);

        assertTrue(0 == Core.countNonZero(gray0));
        Core.rectangle(gray0, center, origin, color, 2, 4, 2);
        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testReduceMatMatIntInt() {
        Mat src = new Mat(2, 2, CvType.CV_32F);
        Mat out = new Mat(1, 2, CvType.CV_32F);
        src.put(0, 0, 1, 0);
        src.put(1, 0, 1, 0);

        out.put(0, 0, 1, 0);

        Core.reduce(src, dst, 0, 2);
        assertMatEqual(out, dst, EPS);
    }

    public void testReduceMatMatIntIntInt() {
        Mat src = new Mat(2, 2, CvType.CV_32F);
        Mat out = new Mat(1, 2, CvType.CV_32F);
        src.put(0, 0, 1, 0);
        src.put(1, 0, 1, 0);

        out.put(0, 0, 1, 0);

        Core.reduce(src, dst, 0, 2, -1);
        assertMatEqual(out, dst, EPS);
    }

    public void testRepeat() {
        Mat src = new Mat(1, 3, CvType.CV_32F);
        Mat des1 = new Mat(1, 3, CvType.CV_32F);
        Mat des2 = new Mat(1, 6, CvType.CV_32F);
        src.put(0, 0, 1, 2, 3);

        des1.put(0, 0, 1, 2, 3);
        des2.put(0, 0, 1, 2, 3, 1, 2, 3);

        Core.repeat(src, 1, 1, dst);
        assertMatEqual(des1, dst, EPS);
        Core.repeat(src, 1, 2, dst);
        assertMatEqual(des2, dst, EPS);
    }

    public void testScaleAdd() {
        Core.scaleAdd(gray3, 2.0, gray3, dst);
        assertMatEqual(dst, gray9);
    }

    public void testSetIdentityMat() {
        Core.setIdentity(gray0_32f);
        assertMatEqual(grayE_32f, gray0_32f, EPS);
    }

    public void testSetIdentityMatScalar() {
        Core.gemm(grayE_32f, grayE_32f, 5.0, new Mat(), 0.0, dst);
        Core.setIdentity(gray0_32f, new Scalar(5));
        assertMatEqual(dst, gray0_32f, EPS);
    }

    public void testSolveCubic() {
        Mat coeffs = new Mat(1, 4, CvType.CV_32F);
        Mat roots = new Mat(3, 1, CvType.CV_32F);
        coeffs.put(0, 0, 1, 6, 11, 6);
        roots.put(0, 0, -3, -1, -2);
        Core.solveCubic(coeffs, dst);
        assertMatEqual(roots, dst, EPS);
    }

    public void testSolveMatMatMat() {
        Mat a = new Mat(3, 3, CvType.CV_32F);
        Mat b = new Mat(3, 1, CvType.CV_32F);
        Mat res = new Mat(3, 1, CvType.CV_32F);
        a.put(0, 0, 1, 1, 1);
        a.put(1, 0, 1, -2, 2);
        a.put(2, 0, 1, 2, 1);

        b.put(0, 0, 0, 4, 2);
        res.put(0, 0, -12, 2, 10);

        Core.solve(a, b, dst);
        assertMatEqual(res, dst, EPS);
    }

    public void testSolveMatMatMatInt() {
        Mat a = new Mat(3, 3, CvType.CV_32F);
        Mat b = new Mat(3, 1, CvType.CV_32F);
        Mat res = new Mat(3, 1, CvType.CV_32F);

        a.put(0, 0, 1, 1, 1);
        a.put(1, 0, 1, -2, 2);
        a.put(2, 0, 1, 2, 1);

        b.put(0, 0, 0, 4, 2);
        res.put(0, 0, -12, 2, 10);

        Core.solve(a, b, dst, 3);
        assertMatEqual(res, dst, EPS);
    }

    public void testSolvePolyMatMat() {
        Mat coeffs = new Mat(4, 1, CvType.CV_32F);
        Mat roots = new Mat(3, 1, CvType.CV_32F);

        coeffs.put(0, 0, -6, 11, -6, 1);

        truth = new Mat(3, 1, CvType.CV_32FC2);
        truth.put(0, 0, 1, 0, 2, 0, 3, 0);

        Core.solvePoly(coeffs, roots);
        assertMatEqual(truth, roots, EPS);
    }

    public void testSolvePolyMatMatInt() {
        Mat coeffs = new Mat(4, 1, CvType.CV_32F);
        Mat roots = new Mat(3, 1, CvType.CV_32F);

        coeffs.put(0, 0, -6, 11, -6, 1);

        truth = new Mat(3, 1, CvType.CV_32FC2);
        truth.put(0, 0, 1, 0, -1, 2, -2, 12);

        Core.solvePoly(coeffs, roots, 1);
        assertMatEqual(truth, roots, EPS);
    }

    public void testSort() {
        Mat submat = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols() / 2);
        submat.setTo(new Scalar(1.0));

        Core.sort(gray0, dst, Core.SORT_EVERY_ROW);
        submat = dst.submat(0, dst.rows() / 2, dst.cols() / 2, dst.cols());
        assertTrue(submat.total() == Core.countNonZero(submat));

        Core.sort(gray0, dst, Core.SORT_EVERY_COLUMN);
        submat = dst.submat(dst.rows() / 2, dst.rows(), 0, dst.cols() / 2);
        assertTrue(submat.total() == Core.countNonZero(submat));
    }

    public void testSortIdx() {
        Mat a = Mat.eye(3, 3, CvType.CV_8UC1);
        Mat b = new Mat();

        truth = new Mat(3, 3, CvType.CV_32SC1);
        truth.put(0, 0, 1, 2, 0);
        truth.put(1, 0, 0, 2, 1);
        truth.put(2, 0, 0, 1, 2);

        Core.sortIdx(a, b, Core.SORT_EVERY_ROW + Core.SORT_ASCENDING);
        assertMatEqual(truth, b);
    }

    public void testSplit() {
        ArrayList<Mat> cois = new ArrayList<Mat>();
        Core.split(rgba0, cois);
        for (Mat coi : cois) {
            assertMatEqual(gray0, coi);
        }
    }

    public void testSqrt() {
        Core.sqrt(gray9_32f, dst);
        assertMatEqual(gray3_32f, dst, EPS);

        Mat rgba144 = new Mat(matSize, matSize, CvType.CV_32FC4);
        Mat rgba12 = new Mat(matSize, matSize, CvType.CV_32FC4);
        rgba144.setTo(Scalar.all(144));
        rgba12.setTo(Scalar.all(12));

        Core.sqrt(rgba144, dst);
        assertMatEqual(rgba12, dst, EPS);
    }

    public void testSubtractMatMatMat() {
        Core.subtract(gray128, gray1, dst);
        assertMatEqual(gray127, dst);
    }

    public void testSubtractMatMatMatMat() {
        Mat mask = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(0));
        Mat submask = mask.submat(0, mask.rows() / 2, 0, mask.cols() / 2);
        submask.setTo(new Scalar(1));

        dst = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(0));
        Core.subtract(gray3, gray2, dst, mask);
        assertTrue(submask.total() == Core.countNonZero(dst));
    }

    public void testSubtractMatMatMatMatInt() {
        Core.subtract(gray3, gray2, dst, gray1, CvType.CV_32F);
        assertTrue(CvType.CV_32F == dst.depth());
        assertMatEqual(gray1_32f, dst, EPS);
    }

    public void testSumElems() {
        Mat src = new Mat(4, 4, CvType.CV_8U, new Scalar(10));
        Scalar res1 = Core.sumElems(src);
        assertEquals(new Scalar(160), res1);

        Scalar res2 = Core.sumElems(gray0);
        assertEquals(new Scalar(0), res2);
    }

    public void testSVBackSubst() {
        Mat w = new Mat(2, 2, CvType.CV_32FC1, new Scalar(2));
        Mat u = new Mat(2, 2, CvType.CV_32FC1, new Scalar(4));
        Mat vt = new Mat(2, 2, CvType.CV_32FC1, new Scalar(2));
        Mat rhs = new Mat(2, 2, CvType.CV_32FC1, new Scalar(1));

        Core.SVBackSubst(w, u, vt, rhs, dst);

        Mat truth = new Mat(2, 2, CvType.CV_32FC1, new Scalar(16));
        assertMatEqual(truth, dst, EPS);
    }

    public void testSVDecompMatMatMatMat() {
        Mat src = new Mat(1, 4, CvType.CV_32FC1);
        src.put(0, 0, 1, 4, 8, 6);
        Mat w = new Mat();
        Mat u = new Mat();
        Mat vt = new Mat();
        Core.SVDecomp(src, w, u, vt);

        Mat truthW = new Mat(1, 1, CvType.CV_32FC1, new Scalar(10.816654));
        assertMatEqual(truthW, w, EPS);

        Mat truthU = new Mat(1, 1, CvType.CV_32FC1, new Scalar(1));
        assertMatEqual(truthU, u, EPS);

        Mat truthVT = new Mat(1, 4, CvType.CV_32FC1);
        truthVT.put(0, 0, 0.09245003, 0.36980012, 0.73960024, 0.5547002);
        assertMatEqual(truthVT, vt, EPS);
    }

    public void testSVDecompMatMatMatMatInt() {
        Mat src = new Mat(1, 4, CvType.CV_32FC1);
        src.put(0, 0, 1, 4, 8, 6);
        Mat w = new Mat();
        Mat u = new Mat();
        Mat vt = new Mat();
        Core.SVDecomp(src, w, u, vt, 1/* TODO: SVD::MODIFY_A */);

        Mat truthW = new Mat(1, 1, CvType.CV_32FC1, new Scalar(10.816654));
        assertMatEqual(truthW, w, EPS);

        Mat truthU = new Mat(1, 1, CvType.CV_32FC1, new Scalar(1));
        assertMatEqual(truthU, u, EPS);

        Mat truthVT = new Mat(1, 4, CvType.CV_32FC1);
        truthVT.put(0, 0, 0.09245003, 0.36980012, 0.73960024, 0.5547002);
        assertMatEqual(truthVT, vt, EPS);

    }

    public void testTrace() {
        Scalar s = Core.trace(gray0);
        assertEquals(new Scalar(0), s);
    }

    public void testTransform() {
        Mat src = new Mat(2, 2, CvType.CV_32F, new Scalar(55));
        Mat m = Mat.eye(2, 2, CvType.CV_32FC1);

        Core.transform(src, dst, m);
        truth = new Mat(2, 2, CvType.CV_32FC2, new Scalar(55, 1));
        assertMatEqual(truth, dst, EPS);
    }

    public void testTranspose() {
        Mat subgray0 = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols());
        Mat destination = new Mat(matSize, matSize, CvType.CV_8U);
        destination.setTo(new Scalar(0));
        Mat subdst = destination.submat(0, destination.rows(), 0, destination.cols() / 2);
        subgray0.setTo(new Scalar(1));
        Core.transpose(gray0, destination);
        assertTrue(subdst.total() == Core.countNonZero(subdst));
    }

}
