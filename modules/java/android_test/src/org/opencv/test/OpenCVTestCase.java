package org.opencv.test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.List;

import junit.framework.TestCase;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;

public class OpenCVTestCase extends TestCase {

    protected static final int matSize = 10;
    protected static final double EPS = 0.001;
    protected static final double weakEPS = 0.5;

    protected Mat dst;
    protected Mat truth;

    protected Scalar colorBlack;
    protected Scalar colorWhite;

    // Naming notation: <channels info>_[depth]_[dimensions]_value
    // examples: gray0 - single channel 8U 2d Mat filled with 0
    // grayRnd - single channel 8U 2d Mat filled with random numbers
    // gray0_32f_1d

    // TODO: OpenCVTestCase refactorings
    // - rename matrices
    // - create methods gray0() and create src1 explicitly
    // - create some masks
    // - use truth member everywhere - remove truth from base class - each test
    // fixture should use own truth filed

    protected Mat gray0;
    protected Mat gray1;
    protected Mat gray2;
    protected Mat gray3;
    protected Mat gray9;
    protected Mat gray127;
    protected Mat gray128;
    protected Mat gray255;
    protected Mat grayRnd;

    protected Mat gray_16u_256;
    protected Mat gray_16s_1024;

    protected Mat gray0_32f;
    protected Mat gray1_32f;
    protected Mat gray3_32f;
    protected Mat gray9_32f;
    protected Mat gray255_32f;
    protected Mat grayE_32f;
    protected Mat grayRnd_32f;

    protected Mat gray0_32f_1d;

    protected Mat gray0_64f;
    protected Mat gray0_64f_1d;

    protected Mat rgba0;
    protected Mat rgba128;

    protected Mat rgbLena;
    protected Mat grayChess;

    protected Mat v1;
    protected Mat v2;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        dst = new Mat();
        assertTrue(dst.empty());
        truth = null;

        colorBlack = new Scalar(0);
        colorWhite = new Scalar(255, 255, 255);

        gray0 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(0));
        gray1 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(1));
        gray2 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(2));
        gray3 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(3));
        gray9 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(9));
        gray127 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(127));
        gray128 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(128));
        gray255 = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));

        grayRnd = new Mat(matSize, matSize, CvType.CV_8U);
        Core.randu(grayRnd, 0, 256);

        gray_16u_256 = new Mat(matSize, matSize, CvType.CV_16U, new Scalar(256));
        gray_16s_1024 = new Mat(matSize, matSize, CvType.CV_16S, new Scalar(1024));

        gray0_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(0.0));
        gray1_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(1.0));
        gray3_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(3.0));
        gray9_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(9.0));
        gray255_32f = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(255.0));
        grayE_32f = new Mat(matSize, matSize, CvType.CV_32F);
        grayE_32f = Mat.eye(matSize, matSize, CvType.CV_32FC1);
        grayRnd_32f = new Mat(matSize, matSize, CvType.CV_32F);
        Core.randu(grayRnd_32f, 0, 256);

        gray0_64f = new Mat(matSize, matSize, CvType.CV_64F, new Scalar(0.0));

        gray0_32f_1d = new Mat(1, matSize, CvType.CV_32F, new Scalar(0.0));
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

    protected Mat getMat(int type, double... vals)
    {
        return new Mat(matSize, matSize, type, new Scalar(vals));
    }

    protected Mat makeMask(Mat m, double... vals)
    {
        m.submat(0, m.rows(), 0, m.cols() / 2).setTo(new Scalar(vals));
        return m;
    }

    public static <E extends Number> void assertListEquals(List<E> list1, List<E> list2) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        if (!list1.isEmpty())
        {
            if (list1.get(0) instanceof Float || list1.get(0) instanceof Double)
                throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertEquals(list1.get(i), list2.get(i));
    }

    public static <E extends Number> void assertListEquals(List<E> list1, List<E> list2, double epsilon) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertTrue(Math.abs(list1.get(i).doubleValue() - list2.get(i).doubleValue()) <= epsilon);
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

    public static void assertListPoint3Equals(List<Point3> list1, List<Point3> list2, double epsilon) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertPoint3Equals(list1.get(i), list2.get(i), epsilon);
    }

    public static void assertListRectEquals(List<Rect> list1, List<Rect> list2) {
        if (list1.size() != list2.size()) {
            throw new UnsupportedOperationException();
        }

        for (int i = 0; i < list1.size(); i++)
            assertRectEquals(list1.get(i), list2.get(i));
    }

    public static void assertRectEquals(Rect expected, Rect actual) {
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertEquals(msg, expected.x, actual.x);
        assertEquals(msg, expected.y, actual.y);
        assertEquals(msg, expected.width, actual.width);
        assertEquals(msg, expected.height, actual.height);
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
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertTrue(msg, Math.hypot(expected.pt.x - actual.pt.x, expected.pt.y - actual.pt.y) < eps);
        assertTrue(msg, Math.abs(expected.size - actual.size) < eps);
        assertTrue(msg, Math.abs(expected.angle - actual.angle) < eps);
        assertTrue(msg, Math.abs(expected.response - actual.response) < eps);
        assertEquals(msg, expected.octave, actual.octave);
        assertEquals(msg, expected.class_id, actual.class_id);
    }

    public static void assertListKeyPointEquals(List<KeyPoint> expected, List<KeyPoint> actual, double epsilon) {
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++)
            assertKeyPointEqual(expected.get(i), actual.get(i), epsilon);
    }

    public static void assertDMatchEqual(DMatch expected, DMatch actual, double eps) {
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertEquals(msg, expected.queryIdx, actual.queryIdx);
        assertEquals(msg, expected.trainIdx, actual.trainIdx);
        assertEquals(msg, expected.imgIdx, actual.imgIdx);
        assertTrue(msg, Math.abs(expected.distance - actual.distance) < eps);
    }

    public static void assertScalarEqual(Scalar expected, Scalar actual, double eps) {
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertTrue(msg, Math.abs(expected.val[0] - actual.val[0]) < eps);
        assertTrue(msg, Math.abs(expected.val[1] - actual.val[1]) < eps);
        assertTrue(msg, Math.abs(expected.val[2] - actual.val[2]) < eps);
        assertTrue(msg, Math.abs(expected.val[3] - actual.val[3]) < eps);
    }

    public static void assertListDMatchEquals(List<DMatch> expected, List<DMatch> actual, double epsilon) {
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++)
            assertDMatchEqual(expected.get(i), actual.get(i), epsilon);
    }

    public static void assertPointEquals(Point expected, Point actual, double eps) {
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertEquals(msg, expected.x, actual.x, eps);
        assertEquals(msg, expected.y, actual.y, eps);
    }
    
    public static void assertPoint3Equals(Point3 expected, Point3 actual, double eps) {
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertEquals(msg, expected.x, actual.x, eps);
        assertEquals(msg, expected.y, actual.y, eps);
        assertEquals(msg, expected.z, actual.z, eps);
    }

    static private void compareMats(Mat expected, Mat actual, boolean isEqualityMeasured) {
        if (expected.type() != actual.type() || expected.cols() != actual.cols() || expected.rows() != actual.rows()) {
            throw new UnsupportedOperationException("Can not compare " + expected + " and " + actual);
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
            throw new UnsupportedOperationException("Can not compare " + expected + " and " + actual);
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

    protected static String readFile(String path) {
        FileInputStream stream = null;
        try {
            stream = new FileInputStream(new File(path));
            FileChannel fc = stream.getChannel();
            MappedByteBuffer bb = fc.map(FileChannel.MapMode.READ_ONLY, 0,
                    fc.size());
            return Charset.defaultCharset().decode(bb).toString();
        } catch (IOException e) {
            OpenCVTestRunner.Log("Failed to read file \"" + path
                    + "\". Exception is thrown: " + e);
            return null;
        } finally {
            if (stream != null)
                try {
                    stream.close();
                } catch (IOException e) {
                    OpenCVTestRunner.Log("Exception is thrown: " + e);
                }
        }
    }

    protected static void writeFile(String path, String content) {
        FileOutputStream stream = null;
        try {
            stream = new FileOutputStream(new File(path));
            FileChannel fc = stream.getChannel();
            fc.write(Charset.defaultCharset().encode(content));
        } catch (IOException e) {
            OpenCVTestRunner.Log("Failed to write file \"" + path
                    + "\". Exception is thrown: " + e);
        } finally {
            if (stream != null)
                try {
                    stream.close();
                } catch (IOException e) {
                    OpenCVTestRunner.Log("Exception is thrown: " + e);
                }
        }
    }

}
