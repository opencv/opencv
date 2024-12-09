package org.opencv.test.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.test.OpenCVTestCase;
import org.opencv.imgproc.Imgproc;

public class CoreTest extends OpenCVTestCase {

    public void testAbsdiff() {
        Core.absdiff(gray128, gray255, dst);

        assertMatEqual(gray127, dst);
    }

    public void testAddMatMatMat() {
        Core.add(gray128, gray128, dst);

        assertMatEqual(gray255, dst);
    }

    public void testAddMatMatMatMatInt() {
        Core.add(gray0, gray1, dst, gray1, CvType.CV_32F);

        assertEquals(CvType.CV_32F, dst.depth());
        assertMatEqual(gray1_32f, dst, EPS);
    }

    public void testAddWeightedMatDoubleMatDoubleDoubleMat() {
        Core.addWeighted(gray1, 120.0, gray127, 1.0, 10.0, dst);

        assertMatEqual(gray255, dst);
    }

    public void testAddWeightedMatDoubleMatDoubleDoubleMatInt() {
        Core.addWeighted(gray1, 126.0, gray127, 1.0, 2.0, dst, CvType.CV_32F);

        assertEquals(CvType.CV_32F, dst.depth());
        assertMatEqual(gray255_32f, dst, EPS);
    }

    public void testBitwise_andMatMatMat() {
        Core.bitwise_and(gray127, gray3, dst);

        assertMatEqual(gray3, dst);
    }

    public void testBitwise_andMatMatMatMat() {
        Core.bitwise_and(gray3, gray1, dst, gray255);

        assertMatEqual(gray1, dst);
    }

    public void testBitwise_notMatMat() {
        Core.bitwise_not(gray255, dst);

        assertMatEqual(gray0, dst);
    }

    public void testBitwise_notMatMatMat() {
        Core.bitwise_not(gray0, dst, gray1);

        assertMatEqual(gray255, dst);
    }

    public void testBitwise_orMatMatMat() {
        Core.bitwise_or(gray1, gray2, dst);

        assertMatEqual(gray3, dst);
    }

    public void testBitwise_orMatMatMatMat() {
        Core.bitwise_or(gray127, gray3, dst, gray255);

        assertMatEqual(gray127, dst);
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
        Mat x = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 3.0, 6.0, 5, 0);
            }
        };
        Mat y = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 4.0, 8.0, 12.0);
            }
        };
        Mat dst_angle = new Mat();

        Core.cartToPolar(x, y, dst, dst_angle);

        Mat expected_magnitude = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 5.0, 10.0, 13.0);
            }
        };
        Mat expected_angle = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, atan2rad(4,3), atan2rad(8,6), atan2rad(12,5));
            }
        };
        assertMatEqual(expected_magnitude, dst, EPS);
        assertMatEqual(expected_angle, dst_angle, EPS);
    }

    public void testCartToPolarMatMatMatMatBoolean() {
        Mat x = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 3.0, 6.0, 5, 0);
            }
        };
        Mat y = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 4.0, 8.0, 12.0);
            }
        };
        Mat dst_angle = new Mat();

        Core.cartToPolar(x, y, dst, dst_angle, true);

        Mat expected_magnitude = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 5.0, 10.0, 13.0);
            }
        };
        Mat expected_angle = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, atan2deg(4,3), atan2deg(8,6), atan2deg(12,5));
            }
        };
        assertMatEqual(expected_magnitude, dst, EPS);
        assertMatEqual(expected_angle, dst_angle, EPS * 180/Math.PI);
    }


    public void testCheckRangeMat() {
        Mat outOfRange = new Mat(2, 2, CvType.CV_64F);
        outOfRange.put(0, 0, Double.NaN, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0);

        assertTrue(Core.checkRange(grayRnd_32f));
        assertTrue(Core.checkRange(new Mat()));
        assertFalse(Core.checkRange(outOfRange));
    }


    public void testCheckRangeMatBooleanPointDoubleDouble() {
        Mat inRange = new Mat(2, 3, CvType.CV_64F) {
            {
                put(0, 0, 14, 48, 76, 33, 5, 99);
            }
        };

        assertTrue(Core.checkRange(inRange, true, 5, 100));

        Mat outOfRange = new Mat(2, 3, CvType.CV_64F) {
            {
                put(0, 0, -4, 0, 6, 33, 4, 109);
            }
        };

        assertFalse(Core.checkRange(outOfRange, true, 5, 100));
    }

    public void testCompare() {
        Core.compare(gray0, gray0, dst, Core.CMP_EQ);

        assertMatEqual(dst, gray255);

        Core.compare(gray0, gray1, dst, Core.CMP_EQ);

        assertMatEqual(dst, gray0);

        grayRnd.put(0, 0, 0, 0);

        Core.compare(gray0, grayRnd, dst, Core.CMP_GE);

        int expected = (int) (grayRnd.total() - Core.countNonZero(grayRnd));
        assertEquals(expected, Core.countNonZero(dst));
    }

    public void testCompleteSymmMat() {
        Core.completeSymm(grayRnd_32f);

        assertMatEqual(grayRnd_32f, grayRnd_32f.t(), EPS);
    }

    public void testCompleteSymmMatBoolean() {
        Mat grayRnd_32f2 = grayRnd_32f.clone();

        Core.completeSymm(grayRnd_32f, true);

        assertMatEqual(grayRnd_32f, grayRnd_32f.t(), EPS);
        Core.completeSymm(grayRnd_32f2, false);
        assertMatNotEqual(grayRnd_32f2, grayRnd_32f, EPS);
    }

    public void testConvertScaleAbsMatMat() {
        Core.convertScaleAbs(gray0, dst);

        assertMatEqual(gray0, dst, EPS);

        Core.convertScaleAbs(gray_16u_256, dst);

        assertMatEqual(gray255, dst, EPS);
    }

    public void testConvertScaleAbsMatMatDoubleDouble() {
        Core.convertScaleAbs(gray_16u_256, dst, 2, -513);

        assertMatEqual(gray1, dst);
    }

    public void testCountNonZero() {
        assertEquals(0, Core.countNonZero(gray0));

        gray0.put(0, 0, 255);
        gray0.put(gray0.rows() - 1, gray0.cols() - 1, 255);

        assertEquals(2, Core.countNonZero(gray0));
    }

    public void testCubeRoot() {
        float res = Core.cubeRoot(-27.0f);

        assertEquals(-3.0f, res);
    }

    public void testDctMatMat() {
        Mat in = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 135.22211, 50.811096, 102.27016, 207.6682);
            }
        };
        Mat dst1 = new Mat();
        Mat dst2 = new Mat();

        Core.dct(gray0_32f_1d, dst1);
        Core.dct(in, dst2);

        truth = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 247.98576, -61.252407, 94.904533, 14.013477);
            }
        };
        assertMatEqual(gray0_32f_1d, dst1, EPS);
        assertMatEqual(truth, dst2, EPS);
    }

    public void testDctMatMatInt() {
        Mat in = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 247.98576, -61.252407, 94.904533, 14.013477);
            }
        };
        Mat dst1 = new Mat();
        Mat dst2 = new Mat();

        Core.dct(gray0_32f_1d, dst1, Core.DCT_INVERSE);
        Core.dct(in, dst2, Core.DCT_INVERSE);

        truth = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 135.22211, 50.811096, 102.27016, 207.6682);
            }
        };
        assertMatEqual(gray0_32f_1d, dst1, EPS);
        assertMatEqual(truth, dst2, EPS);
    }

    public void testDeterminant() {
        Mat mat = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 4.0);
                put(0, 1, 2.0);
                put(1, 0, 4.0);
                put(1, 1, 4.0);
            }
        };

        double det = Core.determinant(mat);

        assertEquals(8.0, det);
    }

    public void testDftMatMat() {
        Core.dft(gray0_32f_1d, dst);

        assertMatEqual(gray0_32f_1d, dst, EPS);
    }

    public void testDftMatMatIntInt() {
        Mat src1 = new Mat(2, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 2, 3, 4);
                put(1, 0, 1, 1, 1, 1);
            }
        };
        Mat src2 = new Mat(2, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 2, 3, 4);
                put(1, 0, 0, 0, 0, 0);
            }
        };
        Mat dst1 = new Mat();
        Mat dst2 = new Mat();

        Core.dft(src1, dst1, Core.DFT_REAL_OUTPUT, 1);
        Core.dft(src2, dst2, Core.DFT_REAL_OUTPUT, 0);

        assertMatEqual(dst2, dst1, EPS);
    }

    public void testDivideDoubleMatMat() {
        Core.divide(4.0, gray2, dst);

        assertMatEqual(gray2, dst);

        Core.divide(4.0, gray0, dst);

        assertMatEqual(gray0, dst);
    }

    public void testDivideDoubleMatMatInt() {
        Core.divide(9.0, gray3, dst, CvType.CV_32F);

        assertMatEqual(gray3_32f, dst, EPS);
    }

    public void testDivideMatMatMat() {
        Core.divide(gray9, gray3, dst);

        assertMatEqual(gray3, dst);
    }

    public void testDivideMatMatMatDouble() {
        Core.divide(gray1, gray2, dst, 6.0);

        assertMatEqual(gray3, dst);
    }

    public void testDivideMatMatMatDoubleInt() {
        Core.divide(gray1, gray2, dst, 6.0, CvType.CV_32F);

        assertMatEqual(gray3_32f, dst, EPS);
    }

    public void testEigen() {
        Mat src = new Mat(3, 3, CvType.CV_32FC1) {
            {
                put(0, 0, 2, 0, 0);
                put(1, 0, 0, 6, 0);
                put(2, 0, 0, 0, 4);
            }
        };
        Mat eigenVals = new Mat();
        Mat eigenVecs = new Mat();

        Core.eigen(src, eigenVals, eigenVecs);

        Mat expectedEigenVals = new Mat(3, 1, CvType.CV_32FC1) {
            {
                put(0, 0, 6, 4, 2);
            }
        };
        assertMatEqual(eigenVals, expectedEigenVals, EPS);

        // check by definition
        double eps = 1e-3;
        for(int i = 0; i < 3; i++)
        {
            Mat vec = eigenVecs.row(i).t();
            Mat lhs = new Mat(3, 1, CvType.CV_32FC1);
            Core.gemm(src, vec, 1.0, new Mat(), 1.0, lhs);
            Mat rhs = new Mat(3, 1, CvType.CV_32FC1);
            Core.gemm(vec, eigenVals.row(i), 1.0, new Mat(), 1.0, rhs);
            assertMatEqual(lhs, rhs, eps);
        }
    }

    public void testExp() {
        Core.exp(gray0_32f, dst);

        assertMatEqual(gray1_32f, dst, EPS);
    }

    public void testExtractChannel() {
        Core.extractChannel(rgba128, dst, 0);

        assertMatEqual(gray128, dst);
    }

    public void testFastAtan2() {
        double eps = 0.3;

        float res = Core.fastAtan2(50, 50);

        assertEquals(45, res, eps);

        float res2 = Core.fastAtan2(80, 20);

        assertEquals(Math.atan2(80, 20) * 180 / Math.PI, res2, eps);
    }

    public void testFillConvexPolyMatListOfPointScalar() {
        MatOfPoint polyline = new MatOfPoint(new Point[]{new Point(1, 1), new Point(5, 0), new Point(6, 8), new Point(0, 9)});

        Imgproc.fillConvexPoly(gray0, polyline, new Scalar(150));

        assertTrue(0 < Core.countNonZero(gray0));
        assertTrue(gray0.total() > Core.countNonZero(gray0));
    }

    public void testFillConvexPolyMatListOfPointScalarIntInt() {
        MatOfPoint polyline1 = new MatOfPoint(new Point(2, 1), new Point(5, 1), new Point(5, 7), new Point(2, 7));
        MatOfPoint polyline2 = new MatOfPoint(new Point(4, 2), new Point(10, 2), new Point(10, 14), new Point(4, 14));

        // current implementation of fixed-point version of fillConvexPoly
        // requires image to be at least 2-pixel wider in each direction than
        // contour
        Imgproc.fillConvexPoly(gray0, polyline1, colorWhite, Imgproc.LINE_8, 0);

        assertTrue(0 < Core.countNonZero(gray0));
        assertTrue(gray0.total() > Core.countNonZero(gray0));

        Imgproc.fillConvexPoly(gray0, polyline2, colorBlack, Imgproc.LINE_8, 1);

        assertEquals("see http://code.opencv.org/issues/1284", 0, Core.countNonZero(gray0));
    }

    public void testFillPolyMatListOfListOfPointScalar() {
        int matSize = 10;
        Mat gray0 = Mat.zeros(matSize, matSize, CvType.CV_8U);
        MatOfPoint polyline = new MatOfPoint(new Point(1, 4), new Point(1, 8), new Point(4, 1), new Point(7, 8), new Point(7, 4));
        List<MatOfPoint> polylines = new ArrayList<MatOfPoint>();
        polylines.add(polyline);

        Imgproc.fillPoly(gray0, polylines, new Scalar(1));

        final byte[] truth = new byte[] {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
                0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

        assertMatEqual(new Mat(gray0.size(), CvType.CV_8U) {
            {
                put(0, 0, truth);
            }
        }, gray0);
    }

    public void testFillPolyMatListOfListOfPointScalarIntIntPoint() {
        MatOfPoint polyline1 = new MatOfPoint(new Point(1, 4), new Point(1, 8), new Point(4, 1), new Point(7, 8), new Point(7, 4));
        MatOfPoint polyline2 = new MatOfPoint(new Point(0, 3), new Point(0, 7), new Point(3, 0), new Point(6, 7), new Point(6, 3));

        List<MatOfPoint> polylines1 = new ArrayList<MatOfPoint>();
        polylines1.add(polyline1);

        List<MatOfPoint> polylines2 = new ArrayList<MatOfPoint>();
        polylines2.add(polyline2);

        Imgproc.fillPoly(gray0, polylines1, new Scalar(1), Imgproc.LINE_8, 0, new Point(0, 0));

        assertTrue(0 < Core.countNonZero(gray0));

        Imgproc.fillPoly(gray0, polylines2, new Scalar(0), Imgproc.LINE_8, 0, new Point(1, 1));

        assertEquals(0, Core.countNonZero(gray0));
    }

    public void testFlip() {
        Mat src = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 1.0);
                put(0, 1, 2.0);
                put(1, 0, 3.0);
                put(1, 1, 4.0);
            }
        };
        Mat dst1 = new Mat();
        Mat dst2 = new Mat();

        Core.flip(src, dst1, 0);
        Core.flip(src, dst2, 1);

        Mat dst_f1 = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 3.0);
                put(0, 1, 4.0);
                put(1, 0, 1.0);
                put(1, 1, 2.0);
            }
        };
        Mat dst_f2 = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 2.0);
                put(0, 1, 1.0);
                put(1, 0, 4.0);
                put(1, 1, 3.0);
            }
        };
        assertMatEqual(dst_f1, dst1, EPS);
        assertMatEqual(dst_f2, dst2, EPS);
    }

    public void testGemmMatMatDoubleMatDoubleMat() {
        Mat m1 = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 1.0, 0.0);
                put(1, 0, 1.0, 0.0);
            }
        };
        Mat m2 = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 1.0, 0.0);
                put(1, 0, 1.0, 0.0);
            }
        };
        Mat dmatrix = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 0.001, 0.001);
                put(1, 0, 0.001, 0.001);
            }
        };

        Core.gemm(m1, m2, 1.0, dmatrix, 1.0, dst);

        Mat expected = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 1.001, 0.001);
                put(1, 0, 1.001, 0.001);
            }
        };
        assertMatEqual(expected, dst, EPS);
    }

    public void testGemmMatMatDoubleMatDoubleMatInt() {
        Mat m1 = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 1.0, 0.0);
                put(1, 0, 1.0, 0.0);
            }
        };
        Mat m2 = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 1.0, 0.0);
                put(1, 0, 1.0, 0.0);
            }
        };
        Mat dmatrix = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 0.001, 0.001);
                put(1, 0, 0.001, 0.001);
            }
        };

        Core.gemm(m1, m2, 1.0, dmatrix, 1.0, dst, Core.GEMM_1_T);

        Mat expected = new Mat(2, 2, CvType.CV_32FC1) {
            {
                put(0, 0, 2.001, 0.001);
                put(1, 0, 0.001, 0.001);
            }
        };
        assertMatEqual(expected, dst, EPS);
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
        int cpus = Core.getNumberOfCPUs();

        assertTrue(Runtime.getRuntime().availableProcessors() <= cpus);
    }

    public void testGetOptimalDFTSize() {
        assertEquals(1, Core.getOptimalDFTSize(0));
        assertEquals(135, Core.getOptimalDFTSize(133));
        assertEquals(15, Core.getOptimalDFTSize(13));
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
        double freq1 = Core.getTickFrequency();
        Core.divide(gray2, gray1, dst);
        double freq2 = Core.getTickFrequency();

        assertTrue(0 < freq1);
        assertEquals(freq1, freq2);
    }

    public void testHconcat() {
        List<Mat> mats = Arrays.asList(Mat.eye(3, 3, CvType.CV_8U), Mat.zeros(3, 2, CvType.CV_8U));

        Core.hconcat(mats, dst);

        assertMatEqual(Mat.eye(3, 5, CvType.CV_8U), dst);
    }

    public void testIdctMatMat() {
        Mat in = new Mat(1, 8, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);
            }
        };

        Core.idct(in, dst);

        truth = new Mat(1, 8, CvType.CV_32F) {
            {
                put(0, 0, 3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testIdctMatMatInt() {
        Mat in = new Mat(2, 8, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);
                put(1, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);
            }
        };

        Core.idct(in, dst, Core.DCT_ROWS);

        truth = new Mat(2, 8, CvType.CV_32F) {
            {
                put(0, 0, 3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115);
                put(1, 0, 3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testIdftMatMat() {
        Mat in = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 3.0, 4.0);
            }
        };

        Core.idft(in, dst);

        truth = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 9, -9, 1, 3);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testIdftMatMatIntInt() {
        Mat in = new Mat(2, 4, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 3.0, 4.0);
                put(1, 0, 1.0, 2.0, 3.0, 4.0);
            }
        };
        Mat dst = new Mat();

        Core.idft(in, dst, Core.DFT_REAL_OUTPUT, 1);

        truth = new Mat(2, 4, CvType.CV_32F) {
            {
                put(0, 0, 18, -18, 2, 6);
                put(1, 0, 0, 0, 0, 0);
            }
        };
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
        Mat src = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 1.0);
                put(0, 1, 2.0);
                put(1, 0, 1.5);
                put(1, 1, 4.0);
            }
        };

        Core.invert(src, dst);

        truth = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 4.0);
                put(0, 1, -2.0);
                put(1, 0, -1.5);
                put(1, 1, 1.0);
            }
        };
        assertMatEqual(truth, dst, EPS);
    }

    public void testInvertMatMatInt() {
        Mat src = Mat.eye(3, 3, CvType.CV_32FC1);
        src.put(0, 2, 1);

        double cond = Core.invert(src, dst, Core.DECOMP_SVD);

        truth = Mat.eye(3, 3, CvType.CV_32FC1);
        truth.put(0, 2, -1);
        assertMatEqual(truth, dst, EPS);
        assertEquals(0.3819660544395447, cond, EPS);
    }

    public void testKmeansMatIntMatTermCriteriaIntInt() {
        Mat data = new Mat(4, 5, CvType.CV_32FC1) {
            {
                put(0, 0, 1, 2, 3, 4, 5);
                put(1, 0, 2, 3, 4, 5, 6);
                put(2, 0, 5, 4, 3, 2, 1);
                put(3, 0, 6, 5, 4, 3, 2);
            }
        };
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS, 0, EPS);
        Mat labels = new Mat();

        Core.kmeans(data, 2, labels, criteria, 1, Core.KMEANS_PP_CENTERS);

        int[] first_center = new int[1];
        labels.get(0, 0, first_center);
        final int c1 = first_center[0];
        Mat expected_labels = new Mat(4, 1, CvType.CV_32S) {
            {
                put(0, 0, c1, c1, 1 - c1, 1 - c1);
            }
        };
        assertMatEqual(expected_labels, labels);
    }

    public void testKmeansMatIntMatTermCriteriaIntIntMat() {
        Mat data = new Mat(4, 5, CvType.CV_32FC1) {
            {
                put(0, 0, 1, 2, 3, 4, 5);
                put(1, 0, 2, 3, 4, 5, 6);
                put(2, 0, 5, 4, 3, 2, 1);
                put(3, 0, 6, 5, 4, 3, 2);
            }
        };
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS, 0, EPS);
        Mat labels = new Mat();
        Mat centers = new Mat();

        Core.kmeans(data, 2, labels, criteria, 6, Core.KMEANS_RANDOM_CENTERS, centers);

        int[] first_center = new int[1];
        labels.get(0, 0, first_center);
        final int c1 = first_center[0];
        Mat expected_labels = new Mat(4, 1, CvType.CV_32S) {
            {
                put(0, 0, c1, c1, 1 - c1, 1 - c1);
            }
        };
        Mat expected_centers = new Mat(2, 5, CvType.CV_32FC1) {
            {
                put(c1, 0, 1.5, 2.5, 3.5, 4.5, 5.5);
                put(1 - c1, 0, 5.5, 4.5, 3.5, 2.5, 1.5);
            }
        };
        assertMatEqual(expected_labels, labels);
        assertMatEqual(expected_centers, centers, EPS);
    }

    public void testLineMatPointPointScalar() {
        int nPoints = Math.min(gray0.cols(), gray0.rows());
        Point point1 = new Point(0, 0);
        Point point2 = new Point(nPoints, nPoints);
        Scalar color = new Scalar(255);

        Imgproc.line(gray0, point1, point2, color);

        assertTrue(nPoints == Core.countNonZero(gray0));
    }

    public void testLineMatPointPointScalarInt() {
        int nPoints = Math.min(gray0.cols(), gray0.rows());
        Point point1 = new Point(0, 0);
        Point point2 = new Point(nPoints, nPoints);

        Imgproc.line(gray0, point1, point2, colorWhite, 1);

        assertTrue(nPoints == Core.countNonZero(gray0));
    }

    public void testLineMatPointPointScalarIntIntInt() {
        int nPoints = Math.min(gray0.cols(), gray0.rows());
        Point point1 = new Point(3, 4);
        Point point2 = new Point(nPoints, nPoints);
        Point point1_4 = new Point(3 * 4, 4 * 4);
        Point point2_4 = new Point(nPoints * 4, nPoints * 4);

        Imgproc.line(gray0, point2, point1, colorWhite, 2, Imgproc.LINE_8, 0);

        assertFalse(0 == Core.countNonZero(gray0));

        Imgproc.line(gray0, point2_4, point1_4, colorBlack, 2, Imgproc.LINE_8, 2);

        assertEquals(0, Core.countNonZero(gray0));
    }

    public void testLog() {
        Mat in = new Mat(1, 4, CvType.CV_32FC1) {
            {
                put(0, 0, 1.0, 10.0, 100.0, 1000.0);
            }
        };

        Core.log(in, dst);

        Mat expected = new Mat(1, 4, CvType.CV_32FC1) {
            {
                put(0, 0, 0, 2.3025851, 4.6051702, 6.9077554);
            }
        };
        assertMatEqual(expected, dst, EPS);
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

    public void testMagnitude() {
        Mat x = new Mat(1, 4, CvType.CV_32F);
        Mat y = new Mat(1, 4, CvType.CV_32F);
        x.put(0, 0, 3.0, 5.0, 9.0, 6.0);
        y.put(0, 0, 4.0, 12.0, 40.0, 8.0);

        Core.magnitude(x, y, dst);

        Mat out = new Mat(1, 4, CvType.CV_32F);
        out.put(0, 0, 5.0, 13.0, 41.0, 10.0);
        assertMatEqual(out, dst, EPS);

        Core.magnitude(gray0_32f, gray255_32f, dst);

        assertMatEqual(gray255_32f, dst, EPS);
    }

    public void testMahalanobis() {
        Mat src = new Mat(matSize + 1, matSize, CvType.CV_32F);
        Core.randu(src, -128, 128);

        Mat covar = new Mat(matSize, matSize, CvType.CV_32F);
        Mat mean = new Mat(1, matSize + 1, CvType.CV_32F);
        Core.calcCovarMatrix(src, covar, mean, Core.COVAR_ROWS | Core.COVAR_NORMAL, CvType.CV_32F);
        covar = covar.inv();

        Mat line1 = src.row(0);
        Mat line2 = src.row(1);

        double d = Core.Mahalanobis(line1, line1, covar);

        assertEquals(0.0, d);

        d = Core.Mahalanobis(line1, line2, covar);
        assertTrue(d > 0.0);
    }

    public void testMax() {
        Core.max(gray0, gray255, dst);

        assertMatEqual(gray255, dst);

        Mat x = new Mat(1, 1, CvType.CV_32F);
        Mat y = new Mat(1, 1, CvType.CV_32F);
        x.put(0, 0, 23.0);
        y.put(0, 0, 4.0);

        Core.max(x, y, dst);

        Mat truth = new Mat(1, 1, CvType.CV_32F);
        truth.put(0, 0, 23.0);
        assertMatEqual(truth, dst, EPS);
    }

    public void testMeanMat() {
        Scalar mean = Core.mean(makeMask(gray128));

        assertScalarEqual(new Scalar(64), mean, EPS);
    }

    public void testMeanMatMat() {
        Mat mask1 = makeMask(gray1.clone());
        Mat mask2 = makeMask(gray0, 1);

        Scalar mean1 = Core.mean(grayRnd, mask1);
        Scalar mean2 = Core.mean(grayRnd, mask2);
        Scalar mean = Core.mean(grayRnd, gray1);

        assertScalarEqual(mean, new Scalar(0.5 * (mean1.val[0] + mean2.val[0])), EPS);
    }

    public void testMeanStdDevMatMatMat() {
        MatOfDouble mean   = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();

        Core.meanStdDev(rgbLena, mean, stddev);

        double expectedMean[] = new double[]
            {105.3989906311035, 99.56269836425781, 179.7303047180176};
        double expectedDev[] = new double[]
            {33.74205485167219, 52.8734582803278, 49.01569488056406};

        assertArrayEquals(expectedMean, mean.toArray(), EPS);
        assertArrayEquals(expectedDev, stddev.toArray(), EPS);
    }

    public void testMeanStdDevMatMatMatMat() {
        Mat submat = grayRnd.submat(0, grayRnd.rows() / 2, 0, grayRnd.cols() / 2);
        submat.setTo(new Scalar(33));
        Mat mask = gray0.clone();
        submat = mask.submat(0, mask.rows() / 2, 0, mask.cols() / 2);
        submat.setTo(new Scalar(1));
        MatOfDouble mean   = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();

        Core.meanStdDev(grayRnd, mean, stddev, mask);

        double expectedMean[] = new double[] {33d};
        double expectedDev[]  = new double[] {0d};

        assertArrayEquals(expectedMean, mean.toArray(), EPS);
        assertArrayEquals(expectedDev, stddev.toArray(), EPS);
    }

    public void testMerge() {
        Mat src1 = new Mat(2, 2, CvType.CV_32FC1, new Scalar(1));
        Mat src2 = new Mat(2, 2, CvType.CV_32FC1, new Scalar(2));
        Mat src3 = new Mat(2, 2, CvType.CV_32FC1, new Scalar(3));
        List<Mat> listMat = Arrays.asList(src1, src2, src3);

        Core.merge(listMat, dst);

        truth = new Mat(2, 2, CvType.CV_32FC3, new Scalar(1, 2, 3));
        assertMatEqual(truth, dst, EPS);
    }

    public void testMin() {
        Core.min(gray0, gray255, dst);

        assertMatEqual(gray0, dst);
    }

    public void testMinMaxLocMat() {
        double minVal = 1;
        double maxVal = 10;
        Point minLoc = new Point(gray3.cols() / 4, gray3.rows() / 2);
        Point maxLoc = new Point(gray3.cols() / 2, gray3.rows() / 4);
        gray3.put((int) minLoc.y, (int) minLoc.x, minVal);
        gray3.put((int) maxLoc.y, (int) maxLoc.x, maxVal);

        Core.MinMaxLocResult mmres = Core.minMaxLoc(gray3);

        assertEquals(minVal, mmres.minVal);
        assertEquals(maxVal, mmres.maxVal);
        assertPointEquals(minLoc, mmres.minLoc, EPS);
        assertPointEquals(maxLoc, mmres.maxLoc, EPS);
    }

    public void testMinMaxLocMatMat() {
        Mat src = new Mat(4, 4, CvType.CV_8U) {
            {
                put(0, 0, 2, 4, 27, 3);
                put(1, 0, 0, 8, 7, 130);
                put(2, 0, 13, 4, 13, 4);
                put(3, 0, 6, 4, 2, 13);
            }
        };
        Mat mask = new Mat(4, 4, CvType.CV_8U, new Scalar(0));
        mask.submat(1, 3, 1, 4).setTo(new Scalar(1));

        MinMaxLocResult res = Core.minMaxLoc(src, mask);

        assertEquals(4.0, res.minVal);
        assertEquals(130.0, res.maxVal);
        assertPointEquals(new Point(1, 2), res.minLoc, EPS);
        assertPointEquals(new Point(3, 1), res.maxLoc, EPS);
    }

    public void testMixChannels() {
        rgba0.setTo(new Scalar(10, 20, 30, 40));
        List<Mat> src = Arrays.asList(rgba0);
        List<Mat> dst = Arrays.asList(gray3, gray2, gray1, gray0, getMat(CvType.CV_8UC3, 0, 0, 0));
        MatOfInt fromTo = new MatOfInt(
                3, 0,
                3, 1,
                2, 2,
                0, 3,
                2, 4,
                1, 5,
                0, 6
        );

        Core.mixChannels(src, dst, fromTo);

        assertMatEqual(getMat(CvType.CV_8U, 40), dst.get(0));
        assertMatEqual(getMat(CvType.CV_8U, 40), dst.get(1));
        assertMatEqual(getMat(CvType.CV_8U, 30), dst.get(2));
        assertMatEqual(getMat(CvType.CV_8U, 10), dst.get(3));
        assertMatEqual(getMat(CvType.CV_8UC3, 30, 20, 10), dst.get(4));
    }

    public void testMulSpectrumsMatMatMatInt() {
        Mat src1 = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 3.0, 4.0);
            }
        };
        Mat src2 = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 3.0, 4.0);
            }
        };

        Core.mulSpectrums(src1, src2, dst, Core.DFT_ROWS);

        Mat expected = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, -5, 12, 16);
            }
        };
        assertMatEqual(expected, dst, EPS);
    }

    public void testMulSpectrumsMatMatMatIntBoolean() {
        Mat src1 = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 3.0, 4.0);
            }
        };
        Mat src2 = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 3.0, 4.0);
            }
        };

        Core.mulSpectrums(src1, src2, dst, Core.DFT_ROWS, true);

        Mat expected = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 13, 0, 16);
            }
        };
        assertMatEqual(expected, dst, EPS);
    }

    public void testMultiplyMatMatMat() {
        Core.multiply(gray0, gray255, dst);

        assertMatEqual(gray0, dst);
    }

    public void testMultiplyMatMatMatDouble() {
        Core.multiply(gray1, gray1, dst, 2.0);

        assertMatEqual(gray2, dst);

    }

    public void testMultiplyMatMatMatDoubleInt() {
        Core.multiply(gray1, gray2, dst, 1.5, CvType.CV_32F);

        assertMatEqual(gray3_32f, dst, EPS);
    }

    public void testMulTransposedMatMatBoolean() {
        Core.mulTransposed(grayE_32f, dst, true);

        assertMatEqual(grayE_32f, dst, EPS);
    }

    public void testMulTransposedMatMatBooleanMatDouble() {
        Core.mulTransposed(grayE_32f, dst, true, gray0_32f, 2);

        truth = gray0_32f;
        truth.diag().setTo(new Scalar(2));
        assertMatEqual(truth, dst, EPS);
    }

    public void testMulTransposedMatMatBooleanMatDoubleInt() {
        Mat a = getMat(CvType.CV_32F, 1);

        Core.mulTransposed(a, dst, true, gray0_32f, 3, CvType.CV_64F);

        assertMatEqual(getMat(CvType.CV_64F, 3 * a.rows()), dst, EPS);
    }

    public void testNormalizeMatMat() {
        Mat m = gray0.clone();
        m.diag().setTo(new Scalar(2));

        Core.normalize(m, dst);

        assertMatEqual(gray0, dst);
    }

    public void testNormalizeMatMatDoubleDoubleInt() {
        Mat src = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1.0, 2.0, 3.0, 4.0);
            }
        };

        Core.normalize(src, dst, 1.0, 2.0, Core.NORM_INF);

        Mat expected = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 0.25, 0.5, 0.75, 1);
            }
        };
        assertMatEqual(expected, dst, EPS);
    }

    public void testNormalizeMatMatDoubleDoubleIntInt() {
        Mat src = new Mat(1, 5, CvType.CV_32F) {
            {
                put(0, 0, 0, 1, 2, 3, 4);
            }
        };

        Core.normalize(src, dst, 1, 2, Core.NORM_MINMAX, CvType.CV_64F);

        Mat expected = new Mat(1, 5, CvType.CV_64F) {
            {
                put(0, 0, 1, 1.25, 1.5, 1.75, 2);
            }
        };
        assertMatEqual(expected, dst, EPS);
    }

    public void testNormalizeMatMatDoubleDoubleIntIntMat() {
        Mat src = new Mat(1, 5, CvType.CV_32F) {
            {
                put(0, 0, 0, 1, 2, 3, 4);
            }
        };
        Mat mask = new Mat(1, 5, CvType.CV_8U) {
            {
                put(0, 0, 1, 0, 0, 0, 1);
            }
        };
        dst = src.clone();

        Core.normalize(src, dst, 1, 2, Core.NORM_MINMAX, CvType.CV_32F, mask);

        Mat expected = new Mat(1, 5, CvType.CV_32F) {
            {
                put(0, 0, 1, 1, 2, 3, 2);
            }
        };
        assertMatEqual(expected, dst, EPS);
    }

    public void testNormMat() {
        double n = Core.norm(gray1);

        assertEquals(10., n);
    }

    public void testNormMatInt() {
        double n = Core.norm(gray127, Core.NORM_INF);

        assertEquals(127., n);
    }

    public void testNormMatIntMat() {
        double n = Core.norm(gray3, Core.NORM_L1, gray0);

        assertEquals(0.0, n);
    }

    public void testNormMatMat() {
        double n = Core.norm(gray0, gray1);

        assertEquals(10.0, n);
    }

    public void testNormMatMatInt() {
        double n = Core.norm(gray127, gray1, Core.NORM_INF);

        assertEquals(126.0, n);
    }

    public void testNormMatMatIntMat() {
        double n = Core.norm(gray3, gray0, Core.NORM_L1, makeMask(gray0.clone(), 1));

        assertEquals(150.0, n);
    }

    public void testPCABackProject() {
        Mat mean = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 4, 4, 8);
            }
        };
        Mat vectors = new Mat(1, 4, CvType.CV_32F, new Scalar(0)) {
            {
                put(0, 0, 0.2, 0.4, 0.4, 0.8);
            }
        };
        Mat data = new Mat(3, 1, CvType.CV_32F) {
            {
                put(0, 0, -5, 0, -10);
            }
        };
        Mat result = new Mat();

        Core.PCABackProject(data, mean, vectors, result);

        Mat truth = new Mat(3, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 2, 2, 4);
                put(1, 0, 2, 4, 4, 8);
                put(2, 0, 0, 0, 0, 0);
            }
        };
        assertMatEqual(truth, result, EPS);
    }

    public void testPCAComputeMatMatMat() {
        Mat data = new Mat(3, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 2, 2, 4);
                put(1, 0, 2, 4, 4, 8);
                put(2, 0, 3, 6, 6, 12);
            }
        };
        Mat mean = new Mat();
        Mat vectors = new Mat();

        Core.PCACompute(data, mean, vectors);
        //System.out.println(mean.dump());
        //System.out.println(vectors.dump());
        Mat mean_truth = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 4, 4, 8);
            }
        };
        Mat vectors_truth = new Mat(3, 4, CvType.CV_32F, new Scalar(0)) {
            {
                put(0, 0, 0.2, 0.4, 0.4, 0.8);
            }
        };
        assertMatEqual(mean_truth, mean, EPS);

        // eigenvectors are normalized (length = 1),
        // but direction is unknown (v and -v are both eigen vectors)
        // so this direct check doesn't work:
        // assertMatEqual(vectors_truth, vectors, EPS);
        for(int i = 0; i < 3; i++)
        {
            Mat vec0 = vectors_truth.row(i);
            Mat vec1 = vectors.row(i);
            Mat vec1_ = new Mat();
            Core.subtract(new Mat(1, 4, CvType.CV_32F, new Scalar(0)), vec1, vec1_);
            double scale1 = Core.norm(vec0, vec1);
            double scale2 = Core.norm(vec0, vec1_);
            assertTrue(Math.min(scale1, scale2) < EPS);
        }
    }

    public void testPCAComputeMatMatMatInt() {
        Mat data = new Mat(3, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 2, 2, 4);
                put(1, 0, 2, 4, 4, 8);
                put(2, 0, 3, 6, 6, 12);
            }
        };
        Mat mean = new Mat();
        Mat vectors = new Mat();

        Core.PCACompute(data, mean, vectors, 1);

        Mat mean_truth = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 4, 4, 8);
            }
        };
        Mat vectors_truth = new Mat(1, 4, CvType.CV_32F, new Scalar(0)) {
            {
                put(0, 0, 0.2, 0.4, 0.4, 0.8);
            }
        };
        assertMatEqual(mean_truth, mean, EPS);
        // eigenvectors are normalized (length = 1),
        // but direction is unknown (v and -v are both eigen vectors)
        // so this direct check doesn't work:
        // assertMatEqual(vectors_truth, vectors, EPS);
        for(int i = 0; i < 1; i++)
        {
            Mat vec0 = vectors_truth.row(i);
            Mat vec1 = vectors.row(i);
            Mat vec1_ = new Mat();
            Core.subtract(new Mat(1, 4, CvType.CV_32F, new Scalar(0)), vec1, vec1_);
            double scale1 = Core.norm(vec0, vec1);
            double scale2 = Core.norm(vec0, vec1_);
            assertTrue(Math.min(scale1, scale2) < EPS);
        }
    }

    public void testPCAProject() {
        Mat mean = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 2, 4, 4, 8);
            }
        };
        Mat vectors = new Mat(1, 4, CvType.CV_32F, new Scalar(0)) {
            {
                put(0, 0, 0.2, 0.4, 0.4, 0.8);
            }
        };
        Mat data = new Mat(3, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 2, 2, 4);
                put(1, 0, 2, 4, 4, 8);
                put(2, 0, 0, 0, 0, 0);
            }
        };
        Mat result = new Mat();

        Core.PCAProject(data, mean, vectors, result);

        Mat truth = new Mat(3, 1, CvType.CV_32F) {
            {
                put(0, 0, -5, 0, -10);
            }
        };
        assertMatEqual(truth, result, EPS);
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

    private static double atan2deg(double y, double x)
    {
        double res = Math.atan2(y, x);
        if (res < 0)
            res = Math.PI * 2 + res;
        return res * 180 / Math.PI;
    }

    private static double atan2rad(double y, double x)
    {
        double res = Math.atan2(y, x);
        if (res < 0)
            res = Math.PI * 2 + res;
        return res;
    }

    public void testPhaseMatMatMat() {
        Mat x = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 10.0, 10.0, 20.0, 5.0);
            }
        };
        Mat y = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 20.0, 15.0, 20.0, 20.0);
            }
        };
        Mat gold = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, atan2rad(20, 10), atan2rad(15, 10), atan2rad(20, 20), atan2rad(20, 5));
            }
        };

        Core.phase(x, y, dst);

        assertMatEqual(gold, dst, EPS);
    }

    public void testPhaseMatMatMatBoolean() {
        Mat x = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 10.0, 10.0, 20.0, 5.0);
            }
        };
        Mat y = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 20.0, 15.0, 20.0, 20.0);
            }
        };
        Mat gold = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, atan2deg(20, 10), atan2deg(15, 10), atan2deg(20, 20), atan2deg(20, 5));
            }
        };

        Core.phase(x, y, dst, true);

        assertMatEqual(gold, dst, EPS * 180 / Math.PI);
    }

    public void testPolarToCartMatMatMatMat() {
        Mat magnitude = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 5.0, 10.0, 13.0);
            }
        };
        Mat angle = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 0.92729962, 0.92729962, 1.1759995);
            }
        };
        Mat xCoordinate = new Mat();
        Mat yCoordinate = new Mat();

        Core.polarToCart(magnitude, angle, xCoordinate, yCoordinate);

        Mat x = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 3.0, 6.0, 5, 0);
            }
        };
        Mat y = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 4.0, 8.0, 12.0);
            }
        };
        assertMatEqual(x, xCoordinate, EPS);
        assertMatEqual(y, yCoordinate, EPS);
    }

    public void testPolarToCartMatMatMatMatBoolean() {
        Mat magnitude = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 5.0, 10.0, 13.0);
            }
        };
        Mat angle = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 0.92729962, 0.92729962, 1.1759995);
            }
        };
        Mat xCoordinate = new Mat();
        Mat yCoordinate = new Mat();

        Core.polarToCart(magnitude, angle, xCoordinate, yCoordinate, true);

        Mat x = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 4.9993458, 9.9986916, 12.997262);
            }
        };
        Mat y = new Mat(1, 3, CvType.CV_32F) {
            {
                put(0, 0, 0.080918625, 0.16183725, 0.26680708);
            }
        };
        assertMatEqual(x, xCoordinate, EPS);
        assertMatEqual(y, yCoordinate, EPS);
    }

    public void testPow() {
        Core.pow(gray2, 7, dst);

        assertMatEqual(gray128, dst);
    }

    public void testRandn() {
        Core.randn(gray0, 100, 23);

        assertEquals(100., Core.mean(gray0).val[0], 23 / 2);
    }

    public void testRandShuffleMat() {
        Mat original = new Mat(1, 10, CvType.CV_32F) {
            {
                put(0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
            }
        };
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
        Mat original = new Mat(1, 10, CvType.CV_32F) {
            {
                put(0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
            }
        };
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
        Core.randu(gray0, 3, 23);
        fail("Not yet implemented");
        //assertTrue(Core.checkRange(gray0, true, null, 3, 23));
    }

    public void testRectangleMatPointPointScalar() {
        Point bottomRight = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Point topLeft = new Point(0, 0);
        Scalar color = new Scalar(128);

        Imgproc.rectangle(gray0, bottomRight, topLeft, color);

        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testRectangleMatPointPointScalarInt() {
        Point bottomRight = new Point(gray0.cols(), gray0.rows());
        Point topLeft = new Point(0, 0);
        Scalar color = new Scalar(128);

        Imgproc.rectangle(gray0, bottomRight, topLeft, color, 2);
        Imgproc.rectangle(gray0, bottomRight, topLeft, colorBlack);

        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testRectangleMatPointPointScalarIntInt() {
        Point bottomRight = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Point topLeft = new Point(0, 0);
        Scalar color = new Scalar(128);

        Imgproc.rectangle(gray0, bottomRight, topLeft, color, 2, Imgproc.LINE_AA, 0);
        Imgproc.rectangle(gray0, bottomRight, topLeft, colorBlack, 2, Imgproc.LINE_4, 0);

        assertTrue(0 != Core.countNonZero(gray0));
    }

    public void testRectangleMatPointPointScalarIntIntInt() {
        Point bottomRight1 = new Point(gray0.cols(), gray0.rows());
        Point bottomRight2 = new Point(gray0.cols() / 2, gray0.rows() / 2);
        Point topLeft = new Point(0, 0);
        Scalar color = new Scalar(128);

        Imgproc.rectangle(gray0, bottomRight1, topLeft, color, 2, Imgproc.LINE_8, 1);

        assertTrue(0 != Core.countNonZero(gray0));

        Imgproc.rectangle(gray0, bottomRight2, topLeft, colorBlack, 2, Imgproc.LINE_8, 0);

        assertEquals(0, Core.countNonZero(gray0));
    }

    public void testReduceMatMatIntInt() {
        Mat src = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 1, 0);
                put(1, 0, 3, 0);
            }
        };

        Core.reduce(src, dst, 0, Core.REDUCE_AVG);

        Mat out = new Mat(1, 2, CvType.CV_32F) {
            {
                put(0, 0, 2, 0);
            }
        };
        assertMatEqual(out, dst, EPS);
    }

    public void testReduceMatMatIntIntInt() {
        Mat src = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 1, 0);
                put(1, 0, 2, 3);
            }
        };

        Core.reduce(src, dst, 1, Core.REDUCE_SUM, CvType.CV_64F);

        Mat out = new Mat(2, 1, CvType.CV_64F) {
            {
                put(0, 0, 1, 5);
            }
        };
        assertMatEqual(out, dst, EPS);
    }

    public void testRepeat() {
        Mat src = new Mat(1, 2, CvType.CV_32F, new Scalar(0));

        Core.repeat(src, matSize, matSize / 2, dst);

        assertMatEqual(gray0_32f, dst, EPS);
    }

    public void testScaleAdd() {
        Core.scaleAdd(gray3, 2.0, gray3, dst);

        assertMatEqual(gray9, dst);
    }

    public void testSetIdentityMat() {
        Core.setIdentity(gray0_32f);

        assertMatEqual(grayE_32f, gray0_32f, EPS);
    }

    public void testSetIdentityMatScalar() {
        Mat m = gray0_32f;

        Core.setIdentity(m, new Scalar(5));

        truth = new Mat(m.size(), m.type(), new Scalar(0));
        truth.diag().setTo(new Scalar(5));
        assertMatEqual(truth, m, EPS);
    }

    public void testSolveCubic() {
        Mat coeffs = new Mat(1, 4, CvType.CV_32F) {
            {
                put(0, 0, 1, 6, 11, 6);
            }
        };

        assertEquals(3, Core.solveCubic(coeffs, dst));

        Mat roots = new Mat(3, 1, CvType.CV_32F) {
            {
                put(0, 0, -3, -1, -2);
            }
        };
        assertMatEqual(roots, dst, EPS);
    }

    public void testSolveMatMatMat() {
        Mat a = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 1, 1);
                put(1, 0, 1, -2, 2);
                put(2, 0, 1, 2, 1);
            }
        };
        Mat b = new Mat(3, 1, CvType.CV_32F) {
            {
                put(0, 0, 0, 4, 2);
            }
        };

        assertTrue(Core.solve(a, b, dst));

        Mat res = new Mat(3, 1, CvType.CV_32F) {
            {
                put(0, 0, -12, 2, 10);
            }
        };
        assertMatEqual(res, dst, EPS);
    }

    public void testSolveMatMatMatInt() {
        Mat a = new Mat(3, 3, CvType.CV_32F) {
            {
                put(0, 0, 1, 1, 1);
                put(1, 0, 1, -2, 2);
                put(2, 0, 1, 2, 1);
            }
        };
        Mat b = new Mat(3, 1, CvType.CV_32F) {
            {
                put(0, 0, 0, 4, 2);
            }
        };

        assertTrue(Core.solve(a, b, dst, Core.DECOMP_QR | Core.DECOMP_NORMAL));

        Mat res = new Mat(3, 1, CvType.CV_32F) {
            {
                put(0, 0, -12, 2, 10);
            }
        };
        assertMatEqual(res, dst, EPS);
    }

    public void testSolvePolyMatMat() {
        Mat coeffs = new Mat(4, 1, CvType.CV_32F) {
            {
                put(0, 0, -6, 11, -6, 1);
            }
        };
        Mat roots = new Mat();

        assertGE(1e-6, Math.abs(Core.solvePoly(coeffs, roots)));

        truth = new Mat(3, 1, CvType.CV_32FC2) {
            {
                put(0, 0, 1, 0, 2, 0, 3, 0);
            }
        };
        assertMatEqual(truth, roots, EPS);
    }

    public void testSolvePolyMatMatInt() {
        Mat coeffs = new Mat(4, 1, CvType.CV_32F) {
            {
                put(0, 0, -6, 11, -6, 1);
            }
        };
        Mat roots = new Mat();

        assertEquals(10.198039027185569, Core.solvePoly(coeffs, roots, 1));

        truth = new Mat(3, 1, CvType.CV_32FC2) {
            {
                put(0, 0, 1, 0, -1, 2, -2, 12);
            }
        };
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

        Core.sortIdx(a, b, Core.SORT_EVERY_ROW | Core.SORT_ASCENDING);

        truth = new Mat(3, 3, CvType.CV_32SC1) {
            {
                put(0, 0, 1, 2, 0);
                put(1, 0, 0, 2, 1);
                put(2, 0, 0, 1, 2);
            }
        };
        assertMatEqual(truth, b);
    }

    public void testSplit() {
        Mat m = getMat(CvType.CV_8UC3, 1, 2, 3);
        ArrayList<Mat> cois = new ArrayList<Mat>();

        Core.split(m, cois);

        assertMatEqual(gray1, cois.get(0));
        assertMatEqual(gray2, cois.get(1));
        assertMatEqual(gray3, cois.get(2));
    }

    public void testSqrt() {
        Core.sqrt(gray9_32f, dst);

        assertMatEqual(gray3_32f, dst, EPS);

        Mat rgba144 = new Mat(matSize, matSize, CvType.CV_32FC4, Scalar.all(144));
        Mat rgba12 = new Mat(matSize, matSize, CvType.CV_32FC4, Scalar.all(12));

        Core.sqrt(rgba144, dst);

        assertMatEqual(rgba12, dst, EPS);
    }

    public void testSubtractMatMatMat() {
        Core.subtract(gray128, gray1, dst);

        assertMatEqual(gray127, dst);
    }

    public void testSubtractMatMatMatMat() {
        Mat mask = makeMask(gray1.clone());
        dst = gray128.clone();

        Core.subtract(gray128, gray1, dst, mask);

        assertMatEqual(makeMask(gray127, 128), dst);
    }

    public void testSubtractMatMatMatMatInt() {
        Core.subtract(gray3, gray2, dst, gray1, CvType.CV_32F);

        assertMatEqual(gray1_32f, dst, EPS);
    }

    public void testSumElems() {
        Mat src = new Mat(4, 4, CvType.CV_8U, new Scalar(10));

        Scalar res1 = Core.sumElems(src);

        assertScalarEqual(new Scalar(160), res1, EPS);
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
        Mat src = new Mat(1, 4, CvType.CV_32FC1) {
            {
                put(0, 0, 1, 4, 8, 6);
            }
        };
        Mat w = new Mat();
        Mat u = new Mat();
        Mat vt = new Mat();

        Core.SVDecomp(src, w, u, vt);

        Mat truthW = new Mat(1, 1, CvType.CV_32FC1, new Scalar(10.816654));
        Mat truthU = new Mat(1, 1, CvType.CV_32FC1, new Scalar(1));
        Mat truthVT = new Mat(1, 4, CvType.CV_32FC1) {
            {
                put(0, 0, 0.09245003, 0.36980012, 0.73960024, 0.5547002);
            }
        };
        assertMatEqual(truthW, w, EPS);
        assertMatEqual(truthU, u, EPS);
        assertMatEqual(truthVT, vt, EPS);
    }

    public void testSVDecompMatMatMatMatInt() {
        Mat src = new Mat(1, 4, CvType.CV_32FC1) {
            {
                put(0, 0, 1, 4, 8, 6);
            }
        };
        Mat w = new Mat();
        Mat u = new Mat();
        Mat vt = new Mat();

        Core.SVDecomp(src, w, u, vt, Core.SVD_NO_UV);

        Mat truthW = new Mat(1, 1, CvType.CV_32FC1, new Scalar(10.816654));
        assertMatEqual(truthW, w, EPS);
        assertTrue(u.empty());
        assertTrue(vt.empty());
    }

    public void testTrace() {
        Scalar s = Core.trace(gray1);

        assertEquals(new Scalar(matSize), s);
    }

    public void testTransform() {
        Mat src = new Mat(2, 2, CvType.CV_32F, new Scalar(55));
        Mat m = Mat.eye(2, 2, CvType.CV_32FC1);

        Core.transform(src, dst, m);

        truth = new Mat(2, 2, CvType.CV_32FC2, new Scalar(55, 1));
        assertMatEqual(truth, dst, EPS);
    }

    public void testTranspose() {
        gray0.submat(0, gray0.rows() / 2, 0, gray0.cols()).setTo(new Scalar(1));
        Mat destination = getMat(CvType.CV_8U, 0);

        Core.transpose(gray0, destination);

        Mat subdst = destination.submat(0, destination.rows(), 0, destination.cols() / 2);
        assertTrue(subdst.total() == Core.countNonZero(subdst));
    }

    public void testVconcat() {
        List<Mat> mats = Arrays.asList(Mat.eye(3, 3, CvType.CV_8U), Mat.zeros(2, 3, CvType.CV_8U));

        Core.vconcat(mats, dst);

        assertMatEqual(Mat.eye(5, 3, CvType.CV_8U), dst);

    }

    public void testCopyMakeBorderMatMatIntIntIntIntInt() {
        Mat src = new Mat(2, 2, CvType.CV_32F, new Scalar(1));
        int border = 2;

        Core.copyMakeBorder(src, dst, border, border, border, border, Core.BORDER_REPLICATE);

        truth = new Mat(6, 6, CvType.CV_32F, new Scalar(1));
        assertMatEqual(truth, dst, EPS);
    }

    public void testCopyMakeBorderMatMatIntIntIntIntIntScalar() {
        Mat src = new Mat(2, 2, CvType.CV_32F, new Scalar(1));

        Scalar value = new Scalar(0);
        int border = 2;

        Core.copyMakeBorder(src, dst, border, border, border, border, Core.BORDER_REPLICATE, value);
        // TODO_: write better test (use Core.BORDER_CONSTANT)

        truth = new Mat(6, 6, CvType.CV_32F, new Scalar(1));
        assertMatEqual(truth, dst, EPS);
    }

    public void testBorderInterpolate() {
        float val1 = Core.borderInterpolate(100, 150, Core.BORDER_REFLECT_101);
        assertEquals(100f, val1);

        float val2 = Core.borderInterpolate(-5, 10, Core.BORDER_WRAP);
        assertEquals(5f, val2);
    }

    public void testVersion() {
        assertEquals(Core.VERSION_MAJOR, Core.getVersionMajor());
        assertEquals(Core.VERSION_MINOR, Core.getVersionMinor());
        assertEquals(Core.VERSION_REVISION, Core.getVersionRevision());
        assertEquals(Core.VERSION, Core.getVersionString());
    }

    public void testHardwareOptions() {
        Core.checkHardwareSupport(0);
        boolean original_status = Core.useOptimized();
        Core.setUseOptimized(!original_status);
        assertEquals(!original_status, Core.useOptimized());
        Core.setUseOptimized(original_status);
        assertEquals(original_status, Core.useOptimized());
    }
}
