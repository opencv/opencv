package org.opencv.test;

import java.util.List;

import junit.framework.TestCase;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Core;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;

public class OpenCVTestCase extends TestCase {

    protected static int matSize = 10;
    protected static double EPS = 0.001;
    protected static double weakEPS = 0.5;

    protected static Mat dst;
    protected static Mat truth;

    protected static Scalar colorBlack;

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

        colorBlack = new Scalar(0);

        gray0 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(0));
        gray1 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(1));
        gray2 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(2));
        gray3 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(3));
        gray9 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(9));
        gray127 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(127));
        gray128 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(128));
        gray255 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));

        gray_16u_256 = new Mat(matSize, matSize, CvType.CV_16U, new Scalar(256));
        gray_16s_1024 = new Mat(matSize, matSize, CvType.CV_16S, new Scalar(1024));

        grayRnd = new Mat(matSize, matSize, CvType.CV_8U);
        Core.randu(grayRnd, 0, 256);

        gray0_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(0.0));
        gray1_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(1.0));
        gray3_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(3.0));
        gray9_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(9.0));
        gray255_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(255.0));
        grayE_32f = new Mat(matSize, matSize, CvType.CV_32F);
        grayE_32f = Mat.eye(matSize, matSize, CvType.CV_32FC1);
        grayRnd_32f = new Mat(matSize, matSize, CvType.CV_32F);
        Core.randu(grayRnd_32f, 0, 256);

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

    @Override
    protected void tearDown() throws Exception {

        gray0.release();
        gray1.release();
        gray2.release();
        gray3.release();
        gray9.release();
        gray127.release();
        gray128.release();
        gray255.release();
        gray_16u_256.release();
        gray_16s_1024.release();
        grayRnd.release();
        gray0_32f.release();
        gray1_32f.release();
        gray3_32f.release();
        gray9_32f.release();
        gray255_32f.release();
        grayE_32f.release();
        grayE_32f.release();
        grayRnd_32f.release();
        gray0_32f_1d.release();
        gray0_64f.release();
        gray0_64f_1d.release();
        rgba0.release();
        rgba128.release();
        rgbLena.release();
        grayChess.release();
        v1.release();
        v2.release();

        super.tearDown();
    }
    
    public static void assertListIntegerEquals(List<Integer> list1, List<Integer> list2) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertEquals(list1.get(i), list2.get(i));
    }

    public static void assertListFloatEquals(List<Float> list1, List<Float> list2, double epsilon) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertTrue(Math.abs(list1.get(i) - list2.get(i)) <= epsilon);
    }

    public static void assertListMatEquals(List<Mat> list1, List<Mat> list2, double epsilon) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertMatEqual(list1.get(i), list2.get(i), epsilon);
    }

    public static void assertListPointEquals(List<Point> list1, List<Point> list2, double epsilon) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertPointEquals(list1.get(i), list2.get(i), epsilon);
    }

    public static void assertListKeyPointEquals(List<KeyPoint> list1, List<KeyPoint> list2, double epsilon) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }
        
        for (int i = 0; i < list1.size(); i++)
            assertKeyPointEqual(list1.get(i), list2.get(i), epsilon);
    }

    public static void assertListRectEquals(List<Rect> list1, List<Rect> list2) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }
        
        for (int i = 0; i < list1.size(); i++)
            assertRectEquals(list1.get(i), list2.get(i));
    }

    public static void assertRectEquals(Rect expected, Rect actual) {
        assertEquals(expected.x, actual.x);
        assertEquals(expected.y, actual.y);
        assertEquals(expected.width, actual.width);
        assertEquals(expected.height, actual.height);
    }

    public static void assertMatEqual(Mat m1, Mat m2) {
        compareMats(m1, m2, true);
    }

    public static void assertMatNotEqual(Mat m1, Mat m2) {
        compareMats(m1, m2, false);
    }

    public static void assertMatEqual(Mat expected, Mat actual, double eps) {
        compareMats(expected, actual, eps, true);
    }

    public static void assertMatNotEqual(Mat expected, Mat actual, double eps) {
        compareMats(expected, actual, eps, false);
    }

    public static void assertKeyPointEqual(KeyPoint expected, KeyPoint actual, double eps) {
        assertTrue(Math.hypot(expected.pt.x - actual.pt.x, expected.pt.y - actual.pt.y) < eps);
        assertTrue(Math.abs(expected.size - actual.size) < eps);
        assertTrue(Math.abs(expected.angle - actual.angle) < eps);
        assertTrue(Math.abs(expected.response - actual.response) < eps);
        assertEquals(expected.octave, actual.octave);
        assertEquals(expected.class_id, actual.class_id);
    }

    public static void assertPointEquals(Point expected, Point actual, double eps) {
        assertEquals(expected.x, actual.x, eps);
        assertEquals(expected.y, actual.y, eps);
    }

    static private void compareMats(Mat expected, Mat actual, boolean isEqualityMeasured) {
        if (expected.type() != actual.type() || expected.cols() != actual.cols() || expected.rows() != actual.rows()) {
            throw new UnsupportedOperationException();
        }

        if (expected.depth() == CvType.CV_32F || expected.depth() == CvType.CV_64F) {
            if (isEqualityMeasured)
                throw new UnsupportedOperationException(
                        "Floating-point Mats must not be checked for exact match. Use assertMatEqual(Mat expected, Mat actual, double eps) instead.");
            else
                throw new UnsupportedOperationException(
                        "Floating-point Mats must not be checked for exact match. Use assertMatNotEqual(Mat expected, Mat actual, double eps) instead.");
        }

        Mat diff = new Mat();
        Core.absdiff(expected, actual, diff);
        Mat reshaped = diff.reshape(1);
        int mistakes = Core.countNonZero(reshaped);

        reshaped.release();
        diff.release();

        if (isEqualityMeasured)
            assertTrue("Mats are different in " + mistakes + " points", 0 == mistakes);
        else
            assertFalse("Mats are equal", 0 == mistakes);
    }

    static private void compareMats(Mat expected, Mat actual, double eps, boolean isEqualityMeasured) {
        if (expected.type() != actual.type() || expected.cols() != actual.cols() || expected.rows() != actual.rows()) {
            throw new UnsupportedOperationException();
        }

        Mat diff = new Mat();
        Core.absdiff(expected, actual, diff);

        if (isEqualityMeasured)
            assertTrue("Max difference between expected and actiual Mats is bigger than " + eps,
                    Core.checkRange(diff, true, new Point(), 0.0, eps));
        else
            assertFalse("Max difference between expected and actiual Mats is less than " + eps,
                    Core.checkRange(diff, true, new Point(), 0.0, eps));
    }

    public void test_1(String label) {
        OpenCVTestRunner.Log("================================================");
        OpenCVTestRunner.Log("=============== " + label);
    }
}
