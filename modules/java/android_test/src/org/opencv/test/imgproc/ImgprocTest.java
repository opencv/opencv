package org.opencv.test.imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.test.OpenCVTestCase;

public class ImgprocTest extends OpenCVTestCase {

    Point anchorPoint;
    private int imgprocSz;
    Size size;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        imgprocSz = 2;
        anchorPoint = new Point(2, 2);
        size = new Size(3, 3);
    }

    public void testAccumulateMatMat() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat dst = getMat(CvType.CV_64F, 0);
        Mat dst2 = src.clone();

        Imgproc.accumulate(src, dst);
        Imgproc.accumulate(src, dst2);

        assertMatEqual(src, dst, EPS);
        assertMatEqual(getMat(CvType.CV_64F, 4), dst2, EPS);
    }

    public void testAccumulateMatMatMat() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat mask = makeMask(getMat(CvType.CV_8U, 1));
        Mat dst = getMat(CvType.CV_64F, 0);
        Mat dst2 = src.clone();

        Imgproc.accumulate(src, dst, mask);
        Imgproc.accumulate(src, dst2, mask);

        assertMatEqual(makeMask(getMat(CvType.CV_64F, 2)), dst, EPS);
        assertMatEqual(makeMask(getMat(CvType.CV_64F, 4), 2), dst2, EPS);
    }

    public void testAccumulateProductMatMatMat() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat dst = getMat(CvType.CV_64F, 0);
        Mat dst2 = src.clone();

        Imgproc.accumulateProduct(src, src, dst);
        Imgproc.accumulateProduct(src, dst, dst2);

        assertMatEqual(getMat(CvType.CV_64F, 4), dst, EPS);
        assertMatEqual(getMat(CvType.CV_64F, 10), dst2, EPS);
    }

    public void testAccumulateProductMatMatMatMat() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat mask = makeMask(getMat(CvType.CV_8U, 1));
        Mat dst = getMat(CvType.CV_64F, 0);
        Mat dst2 = src.clone();

        Imgproc.accumulateProduct(src, src, dst, mask);
        Imgproc.accumulateProduct(src, dst, dst2, mask);

        assertMatEqual(makeMask(getMat(CvType.CV_64F, 4)), dst, EPS);
        assertMatEqual(makeMask(getMat(CvType.CV_64F, 10), 2), dst2, EPS);
    }

    public void testAccumulateSquareMatMat() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat dst = getMat(CvType.CV_64F, 0);
        Mat dst2 = src.clone();

        Imgproc.accumulateSquare(src, dst);
        Imgproc.accumulateSquare(src, dst2);

        assertMatEqual(getMat(CvType.CV_64F, 4), dst, EPS);
        assertMatEqual(getMat(CvType.CV_64F, 6), dst2, EPS);
    }

    public void testAccumulateSquareMatMatMat() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat mask = makeMask(getMat(CvType.CV_8U, 1));
        Mat dst = getMat(CvType.CV_64F, 0);
        Mat dst2 = src.clone();

        Imgproc.accumulateSquare(src, dst, mask);
        Imgproc.accumulateSquare(src, dst2, mask);

        assertMatEqual(makeMask(getMat(CvType.CV_64F, 4)), dst, EPS);
        assertMatEqual(makeMask(getMat(CvType.CV_64F, 6), 2), dst2, EPS);
    }

    public void testAccumulateWeightedMatMatDouble() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat dst = getMat(CvType.CV_64F, 4);
        Mat dst2 = src.clone();

        Imgproc.accumulateWeighted(src, dst, 0.5);
        Imgproc.accumulateWeighted(src, dst2, 2);

        assertMatEqual(getMat(CvType.CV_64F, 3), dst, EPS);
        assertMatEqual(getMat(CvType.CV_64F, 2), dst2, EPS);
    }

    public void testAccumulateWeightedMatMatDoubleMat() {
        Mat src = getMat(CvType.CV_64F, 2);
        Mat mask = makeMask(getMat(CvType.CV_8U, 1));
        Mat dst = getMat(CvType.CV_64F, 4);
        Mat dst2 = src.clone();

        Imgproc.accumulateWeighted(src, dst, 0.5, mask);
        Imgproc.accumulateWeighted(src, dst2, 2, mask);

        assertMatEqual(makeMask(getMat(CvType.CV_64F, 3), 4), dst, EPS);
        assertMatEqual(getMat(CvType.CV_64F, 2), dst2, EPS);
    }

    public void testAdaptiveThreshold() {
        Mat src = makeMask(getMat(CvType.CV_8U, 50), 20);
        Mat dst = new Mat();

        Imgproc.adaptiveThreshold(src, dst, 1, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 3, 0);

        assertEquals(src.rows(), Core.countNonZero(dst));
    }

    public void testApproxPolyDP() {
        MatOfPoint2f curve = new MatOfPoint2f(new Point(1, 3), new Point(2, 4), new Point(3, 5), new Point(4, 4), new Point(5, 3));

        MatOfPoint2f approxCurve = new MatOfPoint2f();

        Imgproc.approxPolyDP(curve, approxCurve, EPS, true);

        List<Point> approxCurveGold =  new ArrayList<Point>(3);
        approxCurveGold.add(new Point(1, 3));
        approxCurveGold.add(new Point(3, 5));
        approxCurveGold.add(new Point(5, 3));

        assertListPointEquals(approxCurve.toList(), approxCurveGold, EPS);
    }

    public void testArcLength() {
        MatOfPoint2f curve = new MatOfPoint2f(new Point(1, 3), new Point(2, 4), new Point(3, 5), new Point(4, 4), new Point(5, 3));

        double arcLength = Imgproc.arcLength(curve, false);

        assertEquals(5.656854152679443, arcLength);
    }

    public void testBilateralFilterMatMatIntDoubleDouble() {
        Imgproc.bilateralFilter(gray255, dst, 5, 10, 5);

        assertMatEqual(gray255, dst);
        // TODO_: write better test
    }

    public void testBilateralFilterMatMatIntDoubleDoubleInt() {
        Imgproc.bilateralFilter(gray255, dst, 5, 10, 5, Core.BORDER_REFLECT);

        assertMatEqual(gray255, dst);
        // TODO_: write better test
    }

    public void testBlurMatMatSize() {
        Imgproc.blur(gray0, dst, size);
        assertMatEqual(gray0, dst);

        Imgproc.blur(gray255, dst, size);
        assertMatEqual(gray255, dst);
        // TODO_: write better test
    }

    public void testBlurMatMatSizePoint() {
        Imgproc.blur(gray0, dst, size, anchorPoint);
        assertMatEqual(gray0, dst);
        // TODO_: write better test
    }

    public void testBlurMatMatSizePointInt() {
        Imgproc.blur(gray0, dst, size, anchorPoint, Core.BORDER_REFLECT);
        assertMatEqual(gray0, dst);
        // TODO_: write better test
    }

    public void testBoundingRect() {
        MatOfPoint points = new MatOfPoint(new Point(0, 0), new Point(0, 4), new Point(4, 0), new Point(4, 4));
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
        // TODO_: write better test
    }

    public void testBoxFilterMatMatIntSizePointBoolean() {
        Imgproc.boxFilter(gray255, dst, 8, size, anchorPoint, false);
        assertMatEqual(gray255, dst);
        // TODO_: write better test
    }

    public void testBoxFilterMatMatIntSizePointBooleanInt() {
        Imgproc.boxFilter(gray255, dst, 8, size, anchorPoint, false, Core.BORDER_REFLECT);
        assertMatEqual(gray255, dst);
        // TODO_: write better test
    }

    public void testCalcBackProject() {
        List<Mat> images = Arrays.asList(grayChess);
        MatOfInt channels = new MatOfInt(0);
        MatOfInt histSize = new MatOfInt(10);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);

        Mat hist = new Mat();
        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
        Core.normalize(hist, hist);

        Imgproc.calcBackProject(images, channels, hist, dst, ranges, 255);

        assertEquals(grayChess.size(), dst.size());
        assertEquals(grayChess.depth(), dst.depth());
        assertFalse(0 == Core.countNonZero(dst));
    }

    public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat() {
        List<Mat> images = Arrays.asList(gray128);
        MatOfInt channels = new MatOfInt(0);
        MatOfInt histSize = new MatOfInt(10);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        Mat hist = new Mat();

        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);

        truth = new Mat(10, 1, CvType.CV_32F, Scalar.all(0)) {
            {
                put(5, 0, 100);
            }
        };
        assertMatEqual(truth, hist, EPS);
    }

    public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat2D() {
        List<Mat> images = Arrays.asList(gray255, gray128);
        MatOfInt channels = new MatOfInt(0, 1);
        MatOfInt histSize = new MatOfInt(10, 10);
        MatOfFloat ranges = new MatOfFloat(0f, 256f, 0f, 256f);
        Mat hist = new Mat();

        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);

        truth = new Mat(10, 10, CvType.CV_32F, Scalar.all(0)) {
            {
                put(9, 5, 100);
            }
        };
        assertMatEqual(truth, hist, EPS);
    }

    public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat3D() {
        List<Mat> images = Arrays.asList(rgbLena);

        Mat hist3D = new Mat();
        List<Mat> histList = Arrays.asList( new Mat[] {new Mat(), new Mat(), new Mat()} );

        MatOfInt histSize = new MatOfInt(10);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);

        for(int i=0; i<rgbLena.channels(); i++)
        {
            Imgproc.calcHist(images, new MatOfInt(i), new Mat(), histList.get(i), histSize, ranges);

            assertEquals(10, histList.get(i).checkVector(1));
        }

        Core.merge(histList, hist3D);

        assertEquals(CvType.CV_32FC3, hist3D.type());
        assertEquals(10, hist3D.checkVector(3));

        Mat truth = new Mat(10, 1, CvType.CV_32FC3);
        truth.put(0, 0,
                 0, 24870, 0,
                 1863, 31926, 1,
                 56682, 37677, 2260,
                 77278, 44751, 32436,
                 69397, 41343, 18526,
                 27180, 40407, 18658,
                 21101, 15993, 32042,
                 8343, 18585, 47786,
                 300, 6567, 80988,
                 0, 25, 29447
                );

        assertMatEqual(truth, hist3D, EPS);
    }

    public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloatBoolean() {
        List<Mat> images = Arrays.asList(gray255, gray128);
        MatOfInt channels = new MatOfInt(0, 1);
        MatOfInt histSize = new MatOfInt(10, 10);
        MatOfFloat ranges = new MatOfFloat(0f, 256f, 0f, 256f);
        Mat hist = new Mat();

        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges, true);

        truth = new Mat(10, 10, CvType.CV_32F, Scalar.all(0)) {
            {
                put(9, 5, 100);
            }
        };
        assertMatEqual(truth, hist, EPS);
    }

    public void testCannyMatMatDoubleDouble() {
        Imgproc.Canny(gray255, dst, 5, 10);
        assertMatEqual(gray0, dst);
        // TODO_: write better test
    }

    public void testCannyMatMatDoubleDoubleIntBoolean() {
        Imgproc.Canny(gray0, dst, 5, 10, 5, true);
        assertMatEqual(gray0, dst);
        // TODO_: write better test
    }

    public void testCompareHist() {
        Mat H1 = new Mat(3, 1, CvType.CV_32F);
        Mat H2 = new Mat(3, 1, CvType.CV_32F);
        H1.put(0, 0, 1, 2, 3);
        H2.put(0, 0, 4, 5, 6);

        double distance = Imgproc.compareHist(H1, H2, Imgproc.CV_COMP_CORREL);

        assertEquals(1., distance);
    }

    public void testContourAreaMat() {
        Mat contour = new Mat(1, 4, CvType.CV_32FC2);
        contour.put(0, 0, 0, 0, 10, 0, 10, 10, 5, 4);

        double area = Imgproc.contourArea(contour);

        assertEquals(45., area);
    }

    public void testContourAreaMatBoolean() {
        Mat contour = new Mat(1, 4, CvType.CV_32FC2);
        contour.put(0, 0, 0, 0, 10, 0, 10, 10, 5, 4);

        double area = Imgproc.contourArea(contour, true);

        assertEquals(45., area);
        // TODO_: write better test
    }

    public void testConvertMapsMatMatMatMatInt() {
        Mat map1 = new Mat(1, 4, CvType.CV_32FC1, new Scalar(1));
        Mat map2 = new Mat(1, 4, CvType.CV_32FC1, new Scalar(2));
        Mat dstmap1 = new Mat(1, 4, CvType.CV_16SC2);
        Mat dstmap2 = new Mat(1, 4, CvType.CV_16UC1);

        Imgproc.convertMaps(map1, map2, dstmap1, dstmap2, CvType.CV_16SC2);

        Mat truthMap1 = new Mat(1, 4, CvType.CV_16SC2);
        truthMap1.put(0, 0, 1, 2, 1, 2, 1, 2, 1, 2);
        assertMatEqual(truthMap1, dstmap1);
        Mat truthMap2 = new Mat(1, 4, CvType.CV_16UC1, new Scalar(0));
        assertMatEqual(truthMap2, dstmap2);
    }

    public void testConvertMapsMatMatMatMatIntBoolean() {
        Mat map1 = new Mat(1, 3, CvType.CV_32FC1, new Scalar(2));
        Mat map2 = new Mat(1, 3, CvType.CV_32FC1, new Scalar(4));
        Mat dstmap1 = new Mat(1, 3, CvType.CV_16SC2);
        Mat dstmap2 = new Mat(1, 3, CvType.CV_16UC1);

        Imgproc.convertMaps(map1, map2, dstmap1, dstmap2, CvType.CV_16SC2, false);
        // TODO_: write better test (last param == true)

        Mat truthMap1 = new Mat(1, 3, CvType.CV_16SC2);
        truthMap1.put(0, 0, 2, 4, 2, 4, 2, 4);
        assertMatEqual(truthMap1, dstmap1);
        Mat truthMap2 = new Mat(1, 3, CvType.CV_16UC1, new Scalar(0));
        assertMatEqual(truthMap2, dstmap2);
    }

    public void testConvexHullMatMat() {
        MatOfPoint points = new MatOfPoint(
                new Point(20, 0),
                new Point(40, 0),
                new Point(30, 20),
                new Point(0,  20),
                new Point(20, 10),
                new Point(30, 10)
        );

        MatOfInt hull = new MatOfInt();

        Imgproc.convexHull(points, hull);

        MatOfInt expHull = new MatOfInt(
                1, 2, 3, 0
        );
        assertMatEqual(expHull, hull, EPS);
    }

    public void testConvexHullMatMatBooleanBoolean() {
        MatOfPoint points = new MatOfPoint(
                new Point(2, 0),
                new Point(4, 0),
                new Point(3, 2),
                new Point(0, 2),
                new Point(2, 1),
                new Point(3, 1)
        );

        MatOfInt hull = new MatOfInt();

        Imgproc.convexHull(points, hull, true);

        MatOfInt expHull = new MatOfInt(
                3, 2, 1, 0
        );
        assertMatEqual(expHull, hull, EPS);
    }

    public void testConvexityDefects() {
        MatOfPoint points = new MatOfPoint(
                new Point(20, 0),
                new Point(40, 0),
                new Point(30, 20),
                new Point(0,  20),
                new Point(20, 10),
                new Point(30, 10)
        );

        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(points, hull);

        MatOfInt4 convexityDefects = new MatOfInt4();
        Imgproc.convexityDefects(points, hull, convexityDefects);

        assertMatEqual(new MatOfInt4(3, 0, 5, 3620), convexityDefects);
    }

    public void testCornerEigenValsAndVecsMatMatIntInt() {
        fail("Not yet implemented");
        // TODO: write better test
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1);
        src.put(0, 0, 1, 2);
        src.put(1, 0, 4, 2);

        int blockSize = 3;
        int ksize = 5;

        // TODO: eigen vals and vectors returned = 0 for most src matrices
        Imgproc.cornerEigenValsAndVecs(src, dst, blockSize, ksize);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC(6), new Scalar(0));
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerEigenValsAndVecsMatMatIntIntInt() {
        fail("Not yet implemented");
        // TODO: write better test
        Mat src = new Mat(4, 4, CvType.CV_32FC1, new Scalar(128));

        int blockSize = 3;
        int ksize = 5;

        truth = new Mat(4, 4, CvType.CV_32FC(6), new Scalar(0));

        Imgproc.cornerEigenValsAndVecs(src, dst, blockSize, ksize, Core.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerHarrisMatMatIntIntDouble() {
        fail("Not yet implemented");
        // TODO: write better test

        truth = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(0));
        int blockSize = 5;
        int ksize = 7;
        double k = 0.1;
        Imgproc.cornerHarris(gray128, dst, blockSize, ksize, k);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerHarrisMatMatIntIntDoubleInt() {
        fail("Not yet implemented");
        // TODO: write better test

        truth = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(0));
        int blockSize = 5;
        int ksize = 7;
        double k = 0.1;
        Imgproc.cornerHarris(gray255, dst, blockSize, ksize, k, Core.BORDER_REFLECT);
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerMinEigenValMatMatInt() {
        fail("Not yet implemented");
        // TODO: write better test

        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1);
        src.put(0, 0, 1, 2);
        src.put(1, 0, 2, 1);
        int blockSize = 5;

        Imgproc.cornerMinEigenVal(src, dst, blockSize);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1, new Scalar(0));
        assertMatEqual(truth, dst, EPS);

        Imgproc.cornerMinEigenVal(gray255, dst, blockSize);

        truth = new Mat(matSize, matSize, CvType.CV_32FC1, new Scalar(0));
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerMinEigenValMatMatIntInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32FC1);
        int blockSize = 3;
        int ksize = 5;

        Imgproc.cornerMinEigenVal(src, dst, blockSize, ksize);

        truth = new Mat(3, 3, CvType.CV_32FC1) {
            {
                put(0, 0, 1. / 18, 1. / 36, 1. / 18);
                put(1, 0, 1. / 36, 1. / 18, 1. / 36);
                put(2, 0, 1. / 18, 1. / 36, 1. / 18);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerMinEigenValMatMatIntIntInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32FC1);
        int blockSize = 3;
        int ksize = 5;

        Imgproc.cornerMinEigenVal(src, dst, blockSize, ksize, Core.BORDER_REFLECT);

        truth = new Mat(3, 3, CvType.CV_32FC1) {
            {
                put(0, 0, 0.68055558, 0.92708349, 0.5868057);
                put(1, 0, 0.92708343, 0.92708343, 0.92708343);
                put(2, 0, 0.58680564, 0.92708343, 0.68055564);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testCornerSubPix() {
        Mat img = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(128));
        Point truthPosition = new Point(img.cols() / 2, img.rows() / 2);

        Rect r = new Rect(new Point(0, 0), truthPosition);
        Core.rectangle(img, r.tl(), r.br(), new Scalar(0), Core.FILLED);
        MatOfPoint2f corners = new MatOfPoint2f(new Point(truthPosition.x + 1, truthPosition.y + 1));
        Size winSize = new Size(2, 2);
        Size zeroZone = new Size(-1, -1);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS, 0, 0.01);

        Imgproc.cornerSubPix(img, corners, winSize, zeroZone, criteria);

        assertPointEquals(truthPosition, corners.toList().get(0), weakEPS);
    }

    public void testCvtColorMatMatInt() {
        fail("Not yet implemented");
    }

    public void testCvtColorMatMatIntInt() {
        fail("Not yet implemented");
    }

    public void testDilateMatMatMat() {
        Mat kernel = new Mat();

        Imgproc.dilate(gray255, dst, kernel);

        assertMatEqual(gray255, dst);

        Imgproc.dilate(gray1, dst, kernel);

        assertMatEqual(gray1, dst);
        // TODO_: write better test
    }

    public void testDilateMatMatMatPoint() {
        fail("Not yet implemented");
    }

    public void testDilateMatMatMatPointInt() {
        fail("Not yet implemented");
    }

    public void testDilateMatMatMatPointIntInt() {
        fail("Not yet implemented");
    }

    public void testDilateMatMatMatPointIntIntScalar() {
        fail("Not yet implemented");
    }

    public void testDistanceTransformWithLabels() {
        Mat dstLables = getMat(CvType.CV_32SC1, 0);
        Mat labels = new Mat();

        Imgproc.distanceTransformWithLabels(gray128, dst, labels, Imgproc.CV_DIST_L2, 3);

        assertMatEqual(dstLables, labels);
        assertMatEqual(getMat(CvType.CV_32FC1, 8192), dst, EPS);
    }

    public void testDrawContoursMatListOfMatIntScalar() {
        Core.rectangle(gray0, new Point(1, 2), new Point(7, 8), new Scalar(100));
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Core.drawContours(gray0, contours, -1, new Scalar(0));

        assertEquals(0, Core.countNonZero(gray0));
    }

    public void testDrawContoursMatListOfMatIntScalarInt() {
        Core.rectangle(gray0, new Point(1, 2), new Point(7, 8), new Scalar(100));
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Core.drawContours(gray0, contours, -1, new Scalar(0), Core.FILLED);

        assertEquals(0, Core.countNonZero(gray0));
    }


    public void testDrawContoursMatListOfMatIntScalarIntIntMatIntPoint() {
        fail("Not yet implemented");
    }

    public void testEqualizeHist() {
        Imgproc.equalizeHist(gray0, dst);
        assertMatEqual(gray0, dst);

        Imgproc.equalizeHist(gray255, dst);
        assertMatEqual(gray255, dst);
        // TODO_: write better test
    }

    public void testErodeMatMatMat() {
        Mat kernel = new Mat();

        Imgproc.erode(gray128, dst, kernel);

        assertMatEqual(gray128, dst);
    }

    public void testErodeMatMatMatPointInt() {
        Mat src = new Mat(3, 3, CvType.CV_8U) {
            {
                put(0, 0, 15, 9, 10);
                put(1, 0, 10, 8, 12);
                put(2, 0, 12, 20, 25);
            }
        };
        Mat kernel = new Mat();

        Imgproc.erode(src, dst, kernel, anchorPoint, 10);

        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(8));
        assertMatEqual(truth, dst);
    }

    public void testErodeMatMatMatPointIntIntScalar() {
        Mat src = new Mat(3, 3, CvType.CV_8U) {
            {
                put(0, 0, 15, 9, 10);
                put(1, 0, 10, 8, 12);
                put(2, 0, 12, 20, 25);
            }
        };
        Mat kernel = new Mat();
        Scalar sc = new Scalar(3, 3);

        Imgproc.erode(src, dst, kernel, anchorPoint, 10, Core.BORDER_REFLECT, sc);

        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(8));
        assertMatEqual(truth, dst);
    }

    public void testFilter2DMatMatIntMat() {
        Mat src = Mat.eye(4, 4, CvType.CV_32F);
        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(1));

        Imgproc.filter2D(src, dst, -1, kernel);

        truth = new Mat(4, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 2, 1, 0);
                put(1, 0, 2, 2, 1, 0);
                put(2, 0, 1, 1, 2, 1);
                put(3, 0, 0, 0, 1, 2);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testFilter2DMatMatIntMatPointDouble() {
        fail("Not yet implemented");
    }

    public void testFilter2DMatMatIntMatPointDoubleInt() {
        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(0));
        Point point = new Point(0, 0);

        Imgproc.filter2D(gray128, dst, -1, kernel, point, 2, Core.BORDER_CONSTANT);

        assertMatEqual(gray2, dst);
    }

    public void testFindContoursMatListOfMatMatIntInt() {
        Mat img = new Mat(50, 50, CvType.CV_8UC1, new Scalar(0));
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>(5);
        Mat hierarchy = new Mat();

        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // no contours on empty image
        assertEquals(contours.size(), 0);
        assertEquals(contours.size(), hierarchy.total());

        Core.rectangle(img, new Point(10, 20), new Point(20, 30), new Scalar(100), 3, Core.LINE_AA, 0);
        Core.rectangle(img, new Point(30, 35), new Point(40, 45), new Scalar(200));

        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // two contours of two rectangles
        assertEquals(contours.size(), 2);
        assertEquals(contours.size(), hierarchy.total());
    }

    public void testFindContoursMatListOfMatMatIntIntPoint() {
        Mat img = new Mat(50, 50, CvType.CV_8UC1, new Scalar(0));
        Mat img2 = img.submat(5, 50, 3, 50);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        List<MatOfPoint> contours2 = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        Core.rectangle(img, new Point(10, 20), new Point(20, 30), new Scalar(100), 3, Core.LINE_AA, 0);
        Core.rectangle(img, new Point(30, 35), new Point(40, 45), new Scalar(200));

        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(img2, contours2, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(3, 5));

        assertEquals(contours.size(), contours2.size());
        assertMatEqual(contours.get(0), contours2.get(0));
        /*
        Log.d("findContours", "hierarchy=" + hierarchy);
        int iBuff[] = new int[ (int) (hierarchy.total() * hierarchy.channels()) ]; // [ Contour0 (next sibling num, previous sibling num, 1st child num, parent num), Contour1(...), ...
        hierarchy.get(0, 0, iBuff);
        Log.d("findContours", Arrays.toString(iBuff));
        */
    }

    public void testFitEllipse() {
        MatOfPoint2f points = new MatOfPoint2f(new Point(0, 0), new Point(-1, 1), new Point(1, 1), new Point(1, -1), new Point(-1, -1));
        RotatedRect rrect = new RotatedRect();

        rrect = Imgproc.fitEllipse(points);

        assertPointEquals(new Point(0, 0), rrect.center, EPS);
        assertEquals(2.828, rrect.size.width, EPS);
        assertEquals(2.828, rrect.size.height, EPS);
    }

    public void testFitLine() {
        Mat points = new Mat(1, 4, CvType.CV_32FC2);
        points.put(0, 0, 0, 0, 2, 3, 3, 4, 5, 8);

        Mat linePoints = new Mat(4, 1, CvType.CV_32FC1);
        linePoints.put(0, 0, 0.53196341, 0.84676737, 2.496531, 3.7467217);

        Imgproc.fitLine(points, dst, Imgproc.CV_DIST_L12, 0, 0.01, 0.01);

        assertMatEqual(linePoints, dst, EPS);
    }

    public void testFloodFillMatMatPointScalar() {
        Mat mask = new Mat(matSize + 2, matSize + 2, CvType.CV_8U, new Scalar(0));
        Mat img = gray0;
        Core.circle(mask, new Point(matSize / 2 + 1, matSize / 2 + 1), 3, new Scalar(2));

        int retval = Imgproc.floodFill(img, mask, new Point(matSize / 2, matSize / 2), new Scalar(1));

        assertEquals(Core.countNonZero(img), retval);
        Core.circle(mask, new Point(matSize / 2 + 1, matSize / 2 + 1), 3, new Scalar(0));
        assertEquals(retval + 4 * (matSize + 1), Core.countNonZero(mask));
        assertMatEqual(mask.submat(1, matSize + 1, 1, matSize + 1), img);
    }

    public void testFloodFillMatMatPointScalar_WithoutMask() {
        Mat img = gray0;
        Core.circle(img, new Point(matSize / 2, matSize / 2), 3, new Scalar(2));

        // TODO: ideally we should pass null instead of "new Mat()"
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
        Imgproc.GaussianBlur(gray0, dst, size, 1);
        assertMatEqual(gray0, dst);

        Imgproc.GaussianBlur(gray2, dst, size, 1);
        assertMatEqual(gray2, dst);
    }

    public void testGaussianBlurMatMatSizeDoubleDouble() {
        Imgproc.GaussianBlur(gray2, dst, size, 0, 0);

        assertMatEqual(gray2, dst);
        // TODO_: write better test
    }

    public void testGaussianBlurMatMatSizeDoubleDoubleInt() {
        Imgproc.GaussianBlur(gray2, dst, size, 1, 3, Core.BORDER_REFLECT);

        assertMatEqual(gray2, dst);
        // TODO_: write better test
    }

    public void testGetAffineTransform() {
        MatOfPoint2f src = new MatOfPoint2f(new Point(2, 3), new Point(3, 1), new Point(1, 4));
        MatOfPoint2f dst = new MatOfPoint2f(new Point(3, 3), new Point(7, 4), new Point(5, 6));

        Mat transform = Imgproc.getAffineTransform(src, dst);

        Mat truth = new Mat(2, 3, CvType.CV_64FC1) {
            {
                put(0, 0, -8, -6, 37);
                put(1, 0, -7, -4, 29);
            }
        };
        assertMatEqual(truth, transform, EPS);
    }

    public void testGetDefaultNewCameraMatrixMat() {
        Mat mtx = Imgproc.getDefaultNewCameraMatrix(gray0);

        assertFalse(mtx.empty());
        assertEquals(0, Core.countNonZero(mtx));
    }

    public void testGetDefaultNewCameraMatrixMatSizeBoolean() {
        Mat mtx = Imgproc.getDefaultNewCameraMatrix(gray0, size, true);

        assertFalse(mtx.empty());
        assertFalse(0 == Core.countNonZero(mtx));
        // TODO_: write better test
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
        // TODO_: write better test
    }

    public void testGetGaussianKernelIntDouble() {
        dst = Imgproc.getGaussianKernel(1, 0.5);

        truth = new Mat(1, 1, CvType.CV_64FC1, new Scalar(1));
        assertMatEqual(truth, dst, EPS);
    }

    public void testGetGaussianKernelIntDoubleInt() {
        dst = Imgproc.getGaussianKernel(3, 0.8, CvType.CV_32F);

        truth = new Mat(3, 1, CvType.CV_32F);
        truth.put(0, 0, 0.23899426, 0.52201146, 0.23899426);
        assertMatEqual(truth, dst, EPS);
    }

    public void testGetPerspectiveTransform() {
        fail("Not yet implemented");
    }

    public void testGetRectSubPixMatSizePointMat() {
        Size size = new Size(3, 3);
        Point center = new Point(gray255.cols() / 2, gray255.rows() / 2);

        Imgproc.getRectSubPix(gray255, size, center, dst);

        truth = new Mat(3, 3, CvType.CV_8U, new Scalar(255));
        assertMatEqual(truth, dst);
    }

    public void testGetRectSubPixMatSizePointMatInt() {
        Mat src = new Mat(10, 10, CvType.CV_32F, new Scalar(2));
        Size patchSize = new Size(5, 5);
        Point center = new Point(src.cols() / 2, src.rows() / 2);

        Imgproc.getRectSubPix(src, patchSize, center, dst);

        truth = new Mat(5, 5, CvType.CV_32F, new Scalar(2));
        assertMatEqual(truth, dst, EPS);
    }

    public void testGetRotationMatrix2D() {
        Point center = new Point(0, 0);

        dst = Imgproc.getRotationMatrix2D(center, 0, 1);

        truth = new Mat(2, 3, CvType.CV_64F) {
            {
                put(0, 0, 1, 0, 0);
                put(1, 0, 0, 1, 0);
            }
        };

        assertMatEqual(truth, dst, EPS);
    }

    public void testGetStructuringElementIntSize() {
        dst = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size);

        truth = new Mat(3, 3, CvType.CV_8UC1, new Scalar(1));
        assertMatEqual(truth, dst);
    }

    public void testGetStructuringElementIntSizePoint() {
        dst = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, size, anchorPoint);

        truth = new Mat(3, 3, CvType.CV_8UC1) {
            {
                put(0, 0, 0, 0, 1);
                put(1, 0, 0, 0, 1);
                put(2, 0, 1, 1, 1);
            }
        };
        assertMatEqual(truth, dst);
    }

    public void testGoodFeaturesToTrackMatListOfPointIntDoubleDouble() {
        Mat src = gray0;
        Core.rectangle(src, new Point(2, 2), new Point(8, 8), new Scalar(100), -1);
        MatOfPoint lp = new MatOfPoint();

        Imgproc.goodFeaturesToTrack(src, lp, 100, 0.01, 3);

        assertEquals(4, lp.total());
    }

    public void testGoodFeaturesToTrackMatListOfPointIntDoubleDoubleMatIntBooleanDouble() {
        Mat src = gray0;
        Core.rectangle(src, new Point(2, 2), new Point(8, 8), new Scalar(100), -1);
        MatOfPoint lp = new MatOfPoint();

        Imgproc.goodFeaturesToTrack(src, lp, 100, 0.01, 3, gray1, 4, true, 0);

        assertEquals(4, lp.total());
    }

    public void testGrabCutMatMatRectMatMatInt() {
        fail("Not yet implemented");
    }

    public void testGrabCutMatMatRectMatMatIntInt() {
        fail("Not yet implemented");
    }

    public void testHoughCirclesMatMatIntDoubleDouble() {
        int sz = 512;
        Mat img = new Mat(sz, sz, CvType.CV_8U, new Scalar(128));
        Mat circles = new Mat();

        Imgproc.HoughCircles(img, circles, Imgproc.CV_HOUGH_GRADIENT, 2, img.rows() / 4);

        assertEquals(0, circles.cols());
    }

    public void testHoughCirclesMatMatIntDoubleDouble1() {
        int sz = 512;
        Mat img = new Mat(sz, sz, CvType.CV_8U, new Scalar(128));
        Mat circles = new Mat();

        Point center = new Point(img.cols() / 2, img.rows() / 2);
        int radius = Math.min(img.cols() / 4, img.rows() / 4);
        Core.circle(img, center, radius, colorBlack, 3);

        Imgproc.HoughCircles(img, circles, Imgproc.CV_HOUGH_GRADIENT, 2, img.rows() / 4);

        assertEquals(1, circles.cols());
    }

    public void testHoughCirclesMatMatIntDoubleDoubleDoubleDoubleIntInt() {
        fail("Not yet implemented");
    }

    public void testHoughLinesMatMatDoubleDoubleInt() {
        int sz = 512;
        Mat img = new Mat(sz, sz, CvType.CV_8U, new Scalar(0));
        Point point1 = new Point(50, 50);
        Point point2 = new Point(img.cols() / 2, img.rows() / 2);
        Core.line(img, point1, point2, colorWhite, 1);
        Mat lines = new Mat();

        Imgproc.HoughLines(img, lines, 1, 3.1415926/180, 100);

        assertEquals(1, lines.cols());

        /*
        Log.d("HoughLines", "lines=" + lines);
        int num = (int)lines.total();
        int buff[] = new int[num*4]; //[ (x1, y1, x2, y2), (...), ...]
        lines.get(0, 0, buff);
        Log.d("HoughLines", "lines=" + Arrays.toString(buff));
        */
    }

    public void testHoughLinesMatMatDoubleDoubleIntDouble() {
        fail("Not yet implemented");
    }

    public void testHoughLinesMatMatDoubleDoubleIntDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testHoughLinesPMatMatDoubleDoubleInt() {
        int sz = 512;
        Mat img = new Mat(sz, sz, CvType.CV_8U, new Scalar(0));
        Point point1 = new Point(0, 0);
        Point point2 = new Point(sz, sz);
        Point point3 = new Point(sz, 0);
        Point point4 = new Point(2*sz/3, sz/3);
        Core.line(img, point1, point2, Scalar.all(255), 1);
        Core.line(img, point3, point4, Scalar.all(255), 1);
        Mat lines = new Mat();

        Imgproc.HoughLinesP(img, lines, 1, 3.1415926/180, 100);

        assertEquals(2, lines.cols());

        /*
        Log.d("HoughLinesP", "lines=" + lines);
        int num = (int)lines.cols();
        int buff[] = new int[num*4]; // CV_32SC4 as [ (x1, y1, x2, y2), (...), ...]
        lines.get(0, 0, buff);
        Log.d("HoughLinesP", "lines=" + Arrays.toString(buff));
        */
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
        fail("Not yet implemented");
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 1);
        cameraMatrix.put(2, 0, 0, 0, 1);

        Mat R = new Mat(3, 3, CvType.CV_32F, new Scalar(2));
        Mat newCameraMatrix = new Mat(3, 3, CvType.CV_32F, new Scalar(3));

        Mat distCoeffs = new Mat();
        Mat map1 = new Mat();
        Mat map2 = new Mat();

        // TODO: complete this test
        Imgproc.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, CvType.CV_32F, map1, map2);
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMat() {
        fail("Not yet implemented");
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F);
        // Size imageSize = new Size(2, 2);

        cameraMatrix.put(0, 0, 1, 0, 1);
        cameraMatrix.put(1, 0, 0, 1, 2);
        cameraMatrix.put(2, 0, 0, 0, 1);

        distCoeffs.put(0, 0, 1, 3, 2, 4);
        truth = new Mat(3, 3, CvType.CV_32F);
        truth.put(0, 0, 0, 0, 0);
        truth.put(1, 0, 0, 0, 0);
        truth.put(2, 0, 0, 3, 0);
        // TODO: No documentation for this function
        // Imgproc.initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize,
        // 5, m1type, truthput1, truthput2);
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMatInt() {
        fail("Not yet implemented");
    }

    public void testInitWideAngleProjMapMatMatSizeIntIntMatMatIntDouble() {
        fail("Not yet implemented");
    }

    public void testIntegral2MatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
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
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
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
        Mat src = new Mat(1, 1, CvType.CV_32F, new Scalar(1));
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
        Mat src = new Mat(1, 1, CvType.CV_32F, new Scalar(1));
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
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2));

        Imgproc.integral(src, dst);

        truth = new Mat(3, 3, CvType.CV_64F) {
            {
                put(0, 0, 0, 0, 0);
                put(1, 0, 0, 2, 4);
                put(2, 0, 0, 4, 8);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testIntegralMatMatInt() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2));

        Imgproc.integral(src, dst, CvType.CV_64F);

        truth = new Mat(3, 3, CvType.CV_64F) {
            {
                put(0, 0, 0, 0, 0);
                put(1, 0, 0, 2, 4);
                put(2, 0, 0, 4, 8);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testInvertAffineTransform() {
        Mat src = new Mat(2, 3, CvType.CV_64F, new Scalar(1));

        Imgproc.invertAffineTransform(src, dst);

        truth = new Mat(2, 3, CvType.CV_64F, new Scalar(0));
        assertMatEqual(truth, dst, EPS);
    }

    public void testIsContourConvex() {
        MatOfPoint contour1 = new MatOfPoint(new Point(0, 0), new Point(10, 0), new Point(10, 10), new Point(5, 4));

        assertFalse(Imgproc.isContourConvex(contour1));

        MatOfPoint contour2 = new MatOfPoint(new Point(0, 0), new Point(10, 0), new Point(10, 10), new Point(5, 6));

        assertTrue(Imgproc.isContourConvex(contour2));
    }

    public void testLaplacianMatMatInt() {
        Imgproc.Laplacian(gray0, dst, CvType.CV_8U);

        assertMatEqual(gray0, dst);
    }

    public void testLaplacianMatMatIntIntDoubleDouble() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        Imgproc.Laplacian(src, dst, CvType.CV_32F, 1, 2, EPS);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F) {
            {
                put(0, 0, -7.9990001, 8.0009995);
                put(1, 0, 8.0009995, -7.9990001);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testLaplacianMatMatIntIntDoubleDoubleInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(2));

        Imgproc.Laplacian(src, dst, CvType.CV_32F, 1, 2, EPS, Core.BORDER_REFLECT);

        truth = new Mat(3, 3, CvType.CV_32F, new Scalar(0.00099945068));
        assertMatEqual(truth, dst, EPS);
    }

    public void testMatchShapes() {
        Mat contour1 = new Mat(1, 4, CvType.CV_32FC2);
        Mat contour2 = new Mat(1, 4, CvType.CV_32FC2);
        contour1.put(0, 0, 1, 1, 5, 1, 4, 3, 6, 2);
        contour2.put(0, 0, 1, 1, 6, 1, 4, 1, 2, 5);

        double distance = Imgproc.matchShapes(contour1, contour2, Imgproc.CV_CONTOURS_MATCH_I1, 1);

        assertEquals(2.81109697365334, distance, EPS);
    }

    public void testMatchTemplate() {
        Mat image = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        Mat templ = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        image.put(0, 0, 1, 2, 3, 4);
        templ.put(0, 0, 5, 6, 7, 8);

        Imgproc.matchTemplate(image, templ, dst, Imgproc.TM_CCORR);

        truth = new Mat(1, 1, CvType.CV_32F, new Scalar(70));
        assertMatEqual(truth, dst, EPS);

        Imgproc.matchTemplate(gray255, gray0, dst, Imgproc.TM_CCORR);

        truth = new Mat(1, 1, CvType.CV_32F, new Scalar(0));
        assertMatEqual(truth, dst, EPS);
    }

    public void testMedianBlur() {
        Imgproc.medianBlur(gray255, dst, 5);
        assertMatEqual(gray255, dst);

        Imgproc.medianBlur(gray2, dst, 3);
        assertMatEqual(gray2, dst);
        // TODO_: write better test
    }

    public void testMinAreaRect() {
        MatOfPoint2f points = new MatOfPoint2f(new Point(1, 1), new Point(5, 1), new Point(4, 3), new Point(6, 2));

        RotatedRect rrect = Imgproc.minAreaRect(points);

        assertEquals(new Size(2, 5), rrect.size);
        assertEquals(-90., rrect.angle);
        assertEquals(new Point(3.5, 2), rrect.center);
    }

    public void testMinEnclosingCircle() {
        MatOfPoint2f points = new MatOfPoint2f(new Point(0, 0), new Point(-1, 0), new Point(0, -1), new Point(1, 0), new Point(0, 1));
        Point actualCenter = new Point();
        float[] radius = new float[1];

        Imgproc.minEnclosingCircle(points, actualCenter, radius);

        assertEquals(new Point(0, 0), actualCenter);
        assertEquals(1.03f, radius[0], EPS);
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
        // TODO_: write better test
    }

    public void testMorphologyExMatMatIntMatPointInt() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_8U);

        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_8U, new Scalar(0));
        Point point = new Point(0, 0);

        Imgproc.morphologyEx(src, dst, Imgproc.MORPH_CLOSE, kernel, point, 10);

        truth = Mat.eye(imgprocSz, imgprocSz, CvType.CV_8U);
        assertMatEqual(truth, dst);
        // TODO_: write better test
    }


    public void testMorphologyExMatMatIntMatPointIntIntScalar() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_8U);
        src.put(0, 0, 2, 1);
        src.put(1, 0, 2, 1);

        Mat kernel = new Mat(imgprocSz, imgprocSz, CvType.CV_8U, new Scalar(1));
        Point point = new Point(1, 1);
        Scalar sc = new Scalar(3, 3);

        Imgproc.morphologyEx(src, dst, Imgproc.MORPH_TOPHAT, kernel, point, 10, Core.BORDER_REFLECT, sc);
        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_8U) {
            {
                put(0, 0, 1, 0);
                put(1, 0, 1, 0);
            }
        };
        assertMatEqual(truth, dst);
        // TODO_: write better test
    }

    public void testPointPolygonTest() {
        MatOfPoint2f contour = new MatOfPoint2f(new Point(0, 0), new Point(1, 3), new Point(3, 4), new Point(4, 3), new Point(2, 1));
        double sign1 = Imgproc.pointPolygonTest(contour, new Point(2, 2), false);
        assertEquals(1.0, sign1);

        double sign2 = Imgproc.pointPolygonTest(contour, new Point(4, 4), true);
        assertEquals(-Math.sqrt(0.5), sign2);
    }

    public void testPreCornerDetectMatMatInt() {
        Mat src = new Mat(4, 4, CvType.CV_32F, new Scalar(1));
        int ksize = 3;

        Imgproc.preCornerDetect(src, dst, ksize);

        truth = new Mat(4, 4, CvType.CV_32F, new Scalar(0));
        assertMatEqual(truth, dst, EPS);
    }

    public void testPreCornerDetectMatMatIntInt() {
        Mat src = new Mat(4, 4, CvType.CV_32F, new Scalar(1));
        int ksize = 3;

        Imgproc.preCornerDetect(src, dst, ksize, Core.BORDER_REFLECT);

        truth = new Mat(4, 4, CvType.CV_32F, new Scalar(0));
        assertMatEqual(truth, dst, EPS);
        // TODO_: write better test
    }

    public void testPyrDownMatMat() {
        Mat src = new Mat(4, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 1, 4, 2);
                put(1, 0, 3, 2, 6, 8);
                put(2, 0, 4, 6, 8, 10);
                put(3, 0, 12, 32, 6, 18);
            }
        };

        Imgproc.pyrDown(src, dst);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F) {
            {
                put(0, 0, 2.78125, 4.609375);
                put(1, 0, 8.546875, 8.8515625);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testPyrDownMatMatSize() {
        Mat src = new Mat(4, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 1, 4, 2);
                put(1, 0, 3, 2, 6, 8);
                put(2, 0, 4, 6, 8, 10);
                put(3, 0, 12, 32, 6, 18);
            }
        };
        Size dstSize = new Size(2, 2);

        Imgproc.pyrDown(src, dst, dstSize);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F) {
            {
                put(0, 0, 2.78125, 4.609375);
                put(1, 0, 8.546875, 8.8515625);
            }
        };
        assertMatEqual(truth, dst, EPS);
        // TODO_: write better test
    }

    public void testPyrMeanShiftFilteringMatMatDoubleDouble() {
        Mat src = new Mat(matSize, matSize, CvType.CV_8UC3, new Scalar(0));

        Imgproc.pyrMeanShiftFiltering(src, dst, 10, 50);

        assertMatEqual(src, dst);
        // TODO_: write better test
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

        Imgproc.pyrUp(src, dst);

        truth = new Mat(4, 4, CvType.CV_32F) {
            {
                put(0, 0, 2,     1.75,  1.375, 1.25);
                put(1, 0, 2.25,  2,     1.625, 1.5);
                put(2, 0, 2.625, 2.375, 2,     1.875);
                put(3, 0, 2.75,  2.5,   2.125, 2);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testPyrUpMatMatSize() {
        fail("Not yet implemented");
    }

    public void testRemapMatMatMatMatInt() {
        fail("Not yet implemented");
        // this test does something weird
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2));
        Mat map1 = new Mat(1, 3, CvType.CV_32FC1);
        Mat map2 = new Mat(1, 3, CvType.CV_32FC1);

        map1.put(0, 0, 3, 6, 5);
        map2.put(0, 0, 4, 8, 12);

        Imgproc.remap(src, dst, map1, map2, Imgproc.INTER_LINEAR);

        truth = new Mat(1, 3, CvType.CV_32F, new Scalar(0));
        assertMatEqual(truth, dst, EPS);
    }

    public void testRemapMatMatMatMatIntIntScalar() {
        fail("Not yet implemented");
        // this test does something weird
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2));
        Mat map1 = new Mat(1, 3, CvType.CV_32FC1);
        Mat map2 = new Mat(1, 3, CvType.CV_32FC1);

        Scalar sc = new Scalar(0);

        map1.put(0, 0, 3, 6, 5, 0);
        map2.put(0, 0, 4, 8, 12);

        truth = new Mat(1, 3, CvType.CV_32F, new Scalar(2));

        Imgproc.remap(src, dst, map1, map2, Imgproc.INTER_LINEAR, Core.BORDER_REFLECT, sc);
        assertMatEqual(truth, dst, EPS);
    }

    public void testResizeMatMatSize() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_8UC1, new Scalar(1));
        Size dsize = new Size(1, 1);

        Imgproc.resize(src, dst, dsize);

        truth = new Mat(1, 1, CvType.CV_8UC1, new Scalar(1));
        assertMatEqual(truth, dst);
    }

    public void testResizeMatMatSizeDoubleDoubleInt() {
        Imgproc.resize(gray255, dst, new Size(2, 2), 0, 0, Imgproc.INTER_AREA);

        truth = new Mat(2, 2, CvType.CV_8UC1, new Scalar(255));
        assertMatEqual(truth, dst);
        // TODO_: write better test
    }

    public void testScharrMatMatIntIntInt() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        Imgproc.Scharr(src, dst, CvType.CV_32F, 1, 0);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(0));
        assertMatEqual(truth, dst, EPS);
    }

    public void testScharrMatMatIntIntIntDoubleDouble() {
        Mat src = Mat.eye(imgprocSz, imgprocSz, CvType.CV_32F);

        Imgproc.Scharr(src, dst, CvType.CV_32F, 1, 0, 1.5, 0.001);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(0.001));
        assertMatEqual(truth, dst, EPS);
    }

    public void testScharrMatMatIntIntIntDoubleDoubleInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32F);

        Imgproc.Scharr(src, dst, CvType.CV_32F, 1, 0, 1.5, 0, Core.BORDER_REFLECT);

        truth = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, -15, -19.5, -4.5);
                put(1, 0, 10.5, 0, -10.5);
                put(2, 0, 4.5, 19.5, 15);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testSepFilter2DMatMatIntMatMat() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(2));
        Mat kernelX = new Mat(1, 3, CvType.CV_32FC1);
        Mat kernelY = new Mat(1, 3, CvType.CV_32FC1);
        kernelX.put(0, 0, 4, 3, 7);
        kernelY.put(0, 0, 9, 4, 2);

        Imgproc.sepFilter2D(src, dst, CvType.CV_32F, kernelX, kernelY);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(420));
        assertMatEqual(truth, dst, EPS);
    }

    public void testSepFilter2DMatMatIntMatMatPointDouble() {
        Mat src = new Mat(imgprocSz, imgprocSz, CvType.CV_32FC1, new Scalar(2));
        Mat kernelX = new Mat(1, 3, CvType.CV_32FC1);
        kernelX.put(0, 0, 2, 2, 2);
        Mat kernelY = new Mat(1, 3, CvType.CV_32FC1);
        kernelY.put(0, 0, 1, 1, 1);

        Imgproc.sepFilter2D(src, dst, CvType.CV_32F, kernelX, kernelY, anchorPoint, weakEPS);

        truth = new Mat(imgprocSz, imgprocSz, CvType.CV_32F, new Scalar(36 + weakEPS));
        assertMatEqual(truth, dst, EPS);
    }

    public void testSepFilter2DMatMatIntMatMatPointDoubleInt() {
        Mat kernelX = new Mat(1, 3, CvType.CV_32FC1);
        kernelX.put(0, 0, 2, 2, 2);

        Mat kernelY = new Mat(1, 3, CvType.CV_32FC1);
        kernelY.put(0, 0, 1, 1, 1);

        Imgproc.sepFilter2D(gray0, dst, CvType.CV_32F, kernelX, kernelY, anchorPoint, weakEPS, Core.BORDER_REFLECT);

        truth = new Mat(10, 10, CvType.CV_32F, new Scalar(weakEPS));
        assertMatEqual(truth, dst, EPS);
        // TODO_: write better test
    }

    public void testSobelMatMatIntIntInt() {
        Imgproc.Sobel(gray255, dst, CvType.CV_8U, 1, 0);

        assertMatEqual(gray0, dst);
    }

    public void testSobelMatMatIntIntIntIntDoubleDouble() {
        Imgproc.Sobel(gray255, dst, CvType.CV_8U, 1, 0, 3, 2, 0.001);
        assertMatEqual(gray0, dst);
        // TODO_: write better test
    }

    public void testSobelMatMatIntIntIntIntDoubleDoubleInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 2, 0, 1);
                put(1, 0, 6, 4, 3);
                put(2, 0, 1, 0, 2);
            }
        };

        Imgproc.Sobel(src, dst, CvType.CV_32F, 1, 0, 3, 2, 0, Core.BORDER_REPLICATE);

        truth = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, -16, -12, 4);
                put(1, 0, -14, -12, 2);
                put(2, 0, -10, 0, 10);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testThreshold() {
        Imgproc.threshold(makeMask(gray0.clone(), 10), dst, 5, 255, Imgproc.THRESH_TRUNC);
        assertMatEqual(makeMask(gray0.clone(), 5), dst);

        Imgproc.threshold(makeMask(gray2.clone(), 10), dst, 1, 255, Imgproc.THRESH_BINARY);
        assertMatEqual(gray255, dst);

        Imgproc.threshold(makeMask(gray2.clone(), 10), dst, 3, 255, Imgproc.THRESH_BINARY_INV);
        assertMatEqual(makeMask(gray255.clone(), 0), dst);
    }

    public void testUndistortMatMatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 0, 1);
                put(1, 0, 0, 1, 2);
                put(2, 0, 0, 0, 1);
            }
        };
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 3, 2, 4);
            }
        };

        Imgproc.undistort(src, dst, cameraMatrix, distCoeffs);

        truth = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 0, 0, 0);
                put(1, 0, 0, 0, 0);
                put(2, 0, 0, 3, 0);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testUndistortMatMatMatMatMat() {
        Mat src = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 0, 1);
                put(1, 0, 0, 1, 2);
                put(2, 0, 0, 0, 1);
            }
        };
        Mat distCoeffs = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 1, 4, 5);
            }
        };
        Mat newCameraMatrix = new Mat(3, 3, CvType.CV_32F, new Scalar(1));

        Imgproc.undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix);

        truth = new Mat(3, 3, CvType.CV_32F, new Scalar(3));
        assertMatEqual(truth, dst, EPS);
    }

    //undistortPoints(List<Point> src, List<Point> dst, Mat cameraMatrix, Mat distCoeffs)
    public void testUndistortPointsListOfPointListOfPointMatMat() {
        MatOfPoint2f src = new MatOfPoint2f(new Point(1, 2), new Point(3, 4), new Point(-1, -1));
        MatOfPoint2f dst = new MatOfPoint2f();
        Mat cameraMatrix = Mat.eye(3, 3, CvType.CV_64FC1);
        Mat distCoeffs = new Mat(8, 1, CvType.CV_64FC1, new Scalar(0));

        Imgproc.undistortPoints(src, dst, cameraMatrix, distCoeffs);

        assertEquals(src.size(), dst.size());
        for(int i=0; i<src.toList().size(); i++) {
            //Log.d("UndistortPoints", "s="+src.get(i)+", d="+dst.get(i));
            assertTrue(src.toList().get(i).equals(dst.toList().get(i)));
        }
    }


    public void testWarpAffineMatMatMatSize() {
        Mat src = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 2, 0, 1);
                put(1, 0, 6, 4, 3);
                put(2, 0, 1, 0, 2);
            }
        };
        Mat M = new Mat(2, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 0, 1);
                put(1, 0, 0, 1, 1);
            }
        };

        Imgproc.warpAffine(src, dst, M, new Size(3, 3));

        truth = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 0, 0, 0);
                put(1, 0, 0, 2, 0);
                put(2, 0, 0, 6, 4);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpAffineMatMatMatSizeInt() {
        Mat src = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 2, 4, 1);
                put(1, 0, 6, 4, 3);
                put(2, 0, 0, 2, 2);
            }
        };
        Mat M = new Mat(2, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 0, 0);
                put(1, 0, 0, 0, 1);
            }
        };

        Imgproc.warpAffine(src, dst, M, new Size(2, 2), Imgproc.WARP_INVERSE_MAP);

        truth = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 6, 4);
                put(1, 0, 6, 4);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpAffineMatMatMatSizeIntInt() {
        fail("Not yet implemented");
    }

    public void testWarpAffineMatMatMatSizeIntIntScalar() {
        fail("Not yet implemented");
    }

    public void testWarpPerspectiveMatMatMatSize() {
        Mat src = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 2, 4, 1);
                put(1, 0, 0, 4, 5);
                put(2, 0, 1, 2, 2);
            }
        };
        Mat M = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 0, 1);
                put(1, 0, 0, 1, 1);
                put(2, 0, 0, 0, 1);
            }
        };

        Imgproc.warpPerspective(src, dst, M, new Size(3, 3));

        truth = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 0, 0, 0);
                put(1, 0, 0, 2, 4);
                put(2, 0, 0, 0, 4);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testWarpPerspectiveMatMatMatSizeInt() {
        fail("Not yet implemented");
    }

    public void testWarpPerspectiveMatMatMatSizeIntInt() {
        fail("Not yet implemented");
    }

    public void testWarpPerspectiveMatMatMatSizeIntIntScalar() {
        fail("Not yet implemented");
    }

    public void testWatershed() {
        Mat image = Mat.eye(4, 4, CvType.CV_8UC(3));
        Mat markers = new Mat(4, 4, CvType.CV_32SC1, new Scalar(0));

        Imgproc.watershed(image, markers);

        truth = new Mat(4, 4, CvType.CV_32SC1) {
            {
                put(0, 0, -1, -1, -1, -1);
                put(1, 0, -1, 0, 0, -1);
                put(2, 0, -1, 0, 0, -1);
                put(3, 0, -1, -1, -1, -1);
            }
        };
        assertMatEqual(truth, markers);
    }

}
