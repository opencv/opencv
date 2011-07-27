package org.opencv.test;

import junit.framework.TestCase;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Core;
import org.opencv.highgui.Highgui;

public class OpenCVTestCase extends TestCase {

    protected static int matSize = 10;
    protected static double EPS = 0.001;

    protected static Mat dst;
    protected static Mat truth;

    // Naming notation: <channels info>_[depth]_[dimensions]_value
    // examples: gray0 - single channel 8U 2d Mat filled with 0
    // grayRnd - single channel 8U 2d Mat filled with random numbers
    // gray0_32f_1d

    // TODO: OpenCVTestCase refactorings
    // - rename matrices
    // - create some masks
    // - use truth member everywhere

    protected static Mat gray0;
    protected static Mat gray1;
    protected static Mat gray2;
    protected static Mat gray3;
    protected static Mat gray9;
    protected static Mat gray127;
    protected static Mat gray128;
    protected static Mat gray255;
    protected static Mat grayRnd;

    protected static Mat gray_16u_256;
    protected static Mat gray_16s_1024;

    protected static Mat gray0_32f;
    protected static Mat gray1_32f;
    protected static Mat gray3_32f;
    protected static Mat gray9_32f;
    protected static Mat gray255_32f;
    protected static Mat grayE_32f;
    protected static Mat grayRnd_32f;

    protected static Mat gray0_32f_1d;

    protected static Mat gray0_64f;
    protected static Mat gray0_64f_1d;

    protected static Mat rgba0;
    protected static Mat rgba128;

    protected static Mat rgbLena;
    protected static Mat grayChess;

    protected static Mat v1;
    protected static Mat v2;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        dst = new Mat();
        assertTrue(dst.empty());
        truth = new Mat();
        assertTrue(truth.empty());

        gray0 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(0));
        gray1 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(1));
        gray2 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(2));
        gray3 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(3));
        gray9 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(9));
        gray127 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(127));
        gray128 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(128));
        gray255 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));

        gray_16u_256 = new Mat(matSize, matSize, CvType.CV_16U, new Scalar(256));
        gray_16s_1024 = new Mat(matSize, matSize, CvType.CV_16S, new Scalar(
                1024));

        Mat low = new Mat(1, 1, CvType.CV_16UC1, new Scalar(0));
        Mat high = new Mat(1, 1, CvType.CV_16UC1, new Scalar(256));
        grayRnd = new Mat(matSize, matSize, CvType.CV_8U);
        Core.randu(grayRnd, low, high);

        gray0_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(0.0));
        gray1_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(1.0));
        gray3_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(3.0));
        gray9_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(9.0));
        gray255_32f = new Mat(matSize, matSize, CvType.CV_32F,
                new Scalar(255.0));
        grayE_32f = new Mat(matSize, matSize, CvType.CV_32F);
        grayE_32f = Mat.eye(matSize, matSize, CvType.CV_32FC1);
        grayRnd_32f = new Mat(matSize, matSize, CvType.CV_32F);
        Core.randu(grayRnd_32f, low, high);

        gray0_32f_1d = new Mat(1, matSize, CvType.CV_32F, new Scalar(0.0));

        gray0_64f = new Mat(matSize, matSize, CvType.CV_64F, new Scalar(0.0));
        gray0_64f_1d = new Mat(1, matSize, CvType.CV_64F, new Scalar(0.0));

        rgba0 = new Mat(matSize, matSize, CvType.CV_8UC4, Scalar.all(0));
        rgba128 = new Mat(matSize, matSize, CvType.CV_8UC4, Scalar.all(128));

        rgbLena = Highgui.imread(OpenCVTestRunner.LENA_PATH);
        grayChess = Highgui.imread(OpenCVTestRunner.CHESS_PATH, 0);

        v1 = new Mat(1, 3, CvType.CV_32F);
        v1.put(0, 0, 1.0, 3.0, 2.0);
        v2 = new Mat(1, 3, CvType.CV_32F);
        v2.put(0, 0, 2.0, 1.0, 3.0);
    }

    public static void assertMatEqual(Mat m1, Mat m2) {
        compareMats(m1, m2, true);
    }

    public static void assertMatNotEqual(Mat m1, Mat m2) {
        compareMats(m1, m2, false);
    }
    
    public static void assertMatEqual(Mat expected, Mat actual, double eps){
        compareMats(expected, actual, eps, true);
    }
    
    public static void assertMatNotEqual(Mat expected, Mat actual, double eps){
        compareMats(expected, actual, eps, false);
    }

    static private void compareMats(Mat m1, Mat m2, boolean isEqualityMeasured) {
        // OpenCVTestRunner.Log(m1.toString());
        // OpenCVTestRunner.Log(m2.toString());

        if (m1.type() != m2.type() || m1.cols() != m2.cols()
                || m1.rows() != m2.rows()) {
            throw new UnsupportedOperationException();
        } else if (m1.channels() == 1) {
            if (isEqualityMeasured) {
                assertTrue(CalcPercentageOfDifference(m1, m2) == 0.0);
            } else {
                assertTrue(CalcPercentageOfDifference(m1, m2) != 0.0);
            }
        } else {
            for (int coi = 0; coi < m1.channels(); coi++) {
                Mat m1c = getCOI(m1, coi);
                Mat m2c = getCOI(m2, coi);
                if (isEqualityMeasured) {
                    assertTrue(CalcPercentageOfDifference(m1c, m2c) == 0.0);
                } else {
                    assertTrue(CalcPercentageOfDifference(m1c, m2c) != 0.0);
                }
            }
        }
    }
    
    static private void compareMats(Mat expected, Mat actual, double eps, boolean isEqualityMeasured) {
        if (expected.type() != actual.type() || expected.cols() != actual.cols()
                || expected.rows() != actual.rows()) {
            throw new UnsupportedOperationException();
        }
        Mat diff = new Mat();
        Core.absdiff(expected, actual, diff);
        OpenCVTestRunner.Log(diff + "     \n    " + diff.dump());
        if(isEqualityMeasured)
            assertTrue("Max difference between expected and actiual values is bigger than " + eps,
                    Core.checkRange(diff, true, new Point(), 0.0, eps));
        else
            assertFalse("Max difference between expected and actiual values is less than " + eps,
                    Core.checkRange(diff, true, new Point(), 0.0, eps));
    }

    static private Mat getCOI(Mat m, int coi) {
        Mat ch = new Mat(m.rows(), m.cols(), m.depth());

        for (int i = 0; i < m.rows(); i++)
            for (int j = 0; j < m.cols(); j++) {
                double pixel[] = m.get(i, j);
                ch.put(i, j, pixel[coi]);
            }

        return ch;
    }

    static private double CalcPercentageOfDifference(Mat m1, Mat m2) {
        Mat cmp = new Mat(0, 0, CvType.CV_8U);
        Core.compare(m1, m2, cmp, Core.CMP_EQ);
        double difference = 100.0 * (1.0 - Double.valueOf(Core
                .countNonZero(cmp)) / Double.valueOf(cmp.rows() * cmp.cols()));

        return difference;
    }

    public void test_1(String label) {
        OpenCVTestRunner
                .Log("================================================");
        OpenCVTestRunner.Log("=============== " + label);
    }
}
