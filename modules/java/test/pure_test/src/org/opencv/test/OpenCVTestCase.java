// TODO: This file is largely a duplicate of the one in android_test.

package org.opencv.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Method;
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
import org.opencv.core.Size;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.imgcodecs.Imgcodecs;

public class OpenCVTestCase extends TestCase {

    public static class TestSkipException extends RuntimeException {
        public TestSkipException() {}
    }

    //change to 'true' to unblock fail on fail("Not yet implemented")
    public static final boolean passNYI = true;

    protected static boolean isTestCaseEnabled = true;

    protected static final String XFEATURES2D = "org.opencv.xfeatures2d.";
    protected static final String DEFAULT_FACTORY = "create";

    protected static final int matSize = 10;
    protected static final double EPS = 0.001;
    protected static final double weakEPS = 0.5;

    private static final String TAG = "OpenCVTestCase";

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

        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (SecurityException e) {
            System.out.println(e.toString());
            System.exit(-1);
        } catch (UnsatisfiedLinkError e) {
            System.out.println(e.toString());
            System.exit(-1);
        }

        Core.setErrorVerbosity(false);

        String pwd;
        try {
            pwd = new File(".").getCanonicalPath() + File.separator;
        } catch (IOException e) {
            System.out.println(e);
            return;
        }

        OpenCVTestRunner.LENA_PATH = pwd + "res/drawable/lena.png";
        OpenCVTestRunner.CHESS_PATH = pwd + "res/drawable/chessboard.jpg";
        OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH = pwd + "res/raw/lbpcascade_frontalface.xml";

        assert(new File(OpenCVTestRunner.LENA_PATH).exists());
        assert(new File(OpenCVTestRunner.CHESS_PATH).exists());
        assert(new File(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH).exists());

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

        rgbLena = Imgcodecs.imread(OpenCVTestRunner.LENA_PATH);
        grayChess = Imgcodecs.imread(OpenCVTestRunner.CHESS_PATH, 0);

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

    @Override
    protected void runTest() throws Throwable {
        // Do nothing if the precondition does not hold.
        if (isTestCaseEnabled) {
            try {
                super.runTest();
            } catch (TestSkipException ex) {
                OpenCVTestRunner.Log(TAG + " :: " + "Test case \"" + this.getClass().getName() + "\" skipped!");
                assertTrue(true);
            }
        } else {
            OpenCVTestRunner.Log(TAG + " :: " + "Test case \"" + this.getClass().getName() + "\" disabled!");
        }
    }

    public void runBare() throws Throwable {
        Throwable exception = null;
        try {
            setUp();
        } catch (TestSkipException ex) {
            OpenCVTestRunner.Log(TAG + " :: " + "Test case \"" + this.getClass().getName() + "\" skipped!");
            assertTrue(true);
            return;
        }
        try {
            runTest();
        } catch (Throwable running) {
            exception = running;
        } finally {
            try {
                tearDown();
            } catch (Throwable tearingDown) {
                if (exception == null) exception = tearingDown;
            }
        }
        if (exception != null) throw exception;
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

    public static void fail(String msg) {
        if(msg == "Not yet implemented" && passNYI)
            return;
        TestCase.fail(msg);
    }

    public static void assertGE(double v1, double v2) {
        assertTrue("Failed: " + v1 + " >= " + v2, v1 >= v2);
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

    public static <E extends Number> void assertArrayEquals(E[] ar1, E[] ar2, double epsilon) {
        if (ar1.length != ar2.length) {
            fail("Arrays have different sizes.");
        }

        for (int i = 0; i < ar1.length; i++)
            assertEquals(ar1[i].doubleValue(), ar2[i].doubleValue(), epsilon);
            //assertTrue(Math.abs(ar1[i].doubleValue() - ar2[i].doubleValue()) <= epsilon);
    }

    public static void assertArrayEquals(double[] ar1, double[] ar2, double epsilon) {
        if (ar1.length != ar2.length) {
            fail("Arrays have different sizes.");
        }

        for (int i = 0; i < ar1.length; i++)
            assertEquals(ar1[i], ar2[i], epsilon);
            //assertTrue(Math.abs(ar1[i].doubleValue() - ar2[i].doubleValue()) <= epsilon);
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

    public static void assertArrayPointsEquals(Point[] vp1, Point[] vp2, double epsilon) {
        if (vp1.length != vp2.length) {
            fail("Arrays have different sizes.");
        }

        for (int i = 0; i < vp1.length; i++)
            assertPointEquals(vp1[i], vp2[i], epsilon);
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

    public static void assertArrayDMatchEquals(DMatch[] expected, DMatch[] actual, double epsilon) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++)
            assertDMatchEqual(expected[i], actual[i], epsilon);
    }

    public static void assertListDMatchEquals(List<DMatch> expected, List<DMatch> actual, double epsilon) {
        DMatch expectedArray[] = expected.toArray(new DMatch[0]);
        DMatch actualArray[]   = actual.toArray(new DMatch[0]);
        assertArrayDMatchEquals(expectedArray, actualArray, epsilon);
    }

    public static void assertPointEquals(Point expected, Point actual, double eps) {
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertEquals(msg, expected.x, actual.x, eps);
        assertEquals(msg, expected.y, actual.y, eps);
    }

    public static void assertSizeEquals(Size expected, Size actual, double eps) {
        String msg = "expected:<" + expected + "> but was:<" + actual + ">";
        assertEquals(msg, expected.width, actual.width, eps);
        assertEquals(msg, expected.height, actual.height, eps);
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
        double maxDiff = Core.norm(diff, Core.NORM_INF);

        if (isEqualityMeasured)
            assertTrue("Max difference between expected and actiual Mats is "+ maxDiff + ", that bigger than " + eps,
                    maxDiff <= eps);
        else
            assertFalse("Max difference between expected and actiual Mats is "+ maxDiff + ", that less than " + eps,
                    maxDiff <= eps);
    }

    protected static String readFile(String path) {
        try {
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        StringBuffer result = new StringBuffer();
        while ((line = br.readLine()) != null) {
            result.append(line);
            result.append("\n");
        }
        return result.toString();
        } catch (IOException e) {
            OpenCVTestRunner.Log("Failed to read file \"" + path
                    + "\". Exception is thrown: " + e);
            return null;
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

    protected <T> T createClassInstance(String cname, String factoryName, Class cParams[], Object oValues[]) {
        T instance = null;

        assertFalse("Class name should not be empty", "".equals(cname));

        String message="";
        try {
            Class algClass = getClassForName(cname);
            Method factory = null;

            if(cParams!=null && cParams.length>0) {
                if(!"".equals(factoryName)) {
                    factory = algClass.getDeclaredMethod(factoryName, cParams);
                    instance = (T) factory.invoke(null, oValues);
                }
                else {
                    instance = (T) algClass.getConstructor(cParams).newInstance(oValues);
                }
            }
            else {
                if(!"".equals(factoryName)) {
                    factory = algClass.getDeclaredMethod(factoryName);
                    instance = (T) factory.invoke(null);
                }
                else {
                    instance = (T) algClass.getConstructor().newInstance();
                }
            }
        }
        catch(Exception ex) {
            if (cname.startsWith(XFEATURES2D))
            {
                throw new TestSkipException();
            }
            message = TAG + " :: " + "could not instantiate " + cname + "! Exception: " + ex.getMessage();
        }

        assertTrue(message, instance!=null);

        return instance;
    }

    protected <T> void setProperty(T instance, String propertyName, String propertyType, Object propertyValue) {
        String message = "";
        try {
            String smethod = "set" + propertyName.substring(0,1).toUpperCase() + propertyName.substring(1);
            Method setter = instance.getClass().getMethod(smethod, getClassForName(propertyType));
            setter.invoke(instance, propertyValue);
        }
        catch(Exception ex) {
            message = "Error when setting property [" + propertyName + "]: " + ex.getMessage();
        }

        assertTrue(message, "".equals(message));
    }

    protected Class getClassForName(String sclass) throws ClassNotFoundException{
        if("int".equals(sclass))
            return Integer.TYPE;
        else if("long".equals(sclass))
            return Long.TYPE;
        else if("double".equals(sclass))
            return Double.TYPE;
        else if("float".equals(sclass))
            return Float.TYPE;
        else if("boolean".equals(sclass))
            return Boolean.TYPE;
        else if("char".equals(sclass))
            return Character.TYPE;
        else if("byte".equals(sclass))
            return Byte.TYPE;
        else if("short".equals(sclass))
            return Short.TYPE;
        else
            return Class.forName(sclass);

    }
}
