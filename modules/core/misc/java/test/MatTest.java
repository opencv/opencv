package org.opencv.test.core;

import java.util.Arrays;
import java.nio.ByteBuffer;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;

public class MatTest extends OpenCVTestCase {

    public void testAdjustROI() {
        Mat roi = gray0.submat(3, 5, 7, 10);
        Mat originalroi = roi.clone();

        Mat adjusted = roi.adjustROI(2, 2, 2, 2);

        assertMatEqual(adjusted, roi);
        assertSizeEquals(new Size(5, 6), adjusted.size(), EPS);
        assertEquals(originalroi.type(), adjusted.type());
        assertTrue(adjusted.isSubmatrix());
        assertFalse(adjusted.isContinuous());

        Point offset = new Point();
        Size size = new Size();
        adjusted.locateROI(size, offset);
        assertPointEquals(new Point(5, 1), offset, EPS);
        assertSizeEquals(gray0.size(), size, EPS);
    }

    public void testAssignToMat() {
        gray0.assignTo(dst);

        assertMatEqual(gray0, dst);

        gray255.assignTo(dst);

        assertMatEqual(gray255, dst);
    }

    public void testAssignToMatInt() {
        gray255.assignTo(dst, CvType.CV_32F);

        assertMatEqual(gray255_32f, dst, EPS);
    }

    public void testChannels() {
        assertEquals(1, gray0.channels());
        assertEquals(3, rgbLena.channels());
        assertEquals(4, rgba0.channels());
    }

    public void testCheckVectorInt() {
        // ! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel
        // (1 x N) or (N x 1); negative number otherwise
        assertEquals(2, new Mat(2, 10, CvType.CV_8U).checkVector(10));
        assertEquals(2, new Mat(1, 2, CvType.CV_8UC(10)).checkVector(10));
        assertEquals(2, new Mat(2, 1, CvType.CV_8UC(10)).checkVector(10));
        assertEquals(10, new Mat(1, 10, CvType.CV_8UC2).checkVector(2));

        assertTrue(0 > new Mat().checkVector(0));
        assertTrue(0 > new Mat(10, 1, CvType.CV_8U).checkVector(10));
        assertTrue(0 > new Mat(10, 20, CvType.CV_8U).checkVector(10));
    }

    public void testCheckVectorIntInt() {
        assertEquals(2, new Mat(2, 10, CvType.CV_8U).checkVector(10, CvType.CV_8U));
        assertEquals(2, new Mat(1, 2, CvType.CV_8UC(10)).checkVector(10, CvType.CV_8U));
        assertEquals(2, new Mat(2, 1, CvType.CV_8UC(10)).checkVector(10, CvType.CV_8U));
        assertEquals(10, new Mat(1, 10, CvType.CV_8UC2).checkVector(2, CvType.CV_8U));

        assertTrue(0 > new Mat(2, 10, CvType.CV_8U).checkVector(10, CvType.CV_8S));
        assertTrue(0 > new Mat(1, 2, CvType.CV_8UC(10)).checkVector(10, CvType.CV_8S));
        assertTrue(0 > new Mat(2, 1, CvType.CV_8UC(10)).checkVector(10, CvType.CV_8S));
        assertTrue(0 > new Mat(1, 10, CvType.CV_8UC2).checkVector(10, CvType.CV_8S));
    }

    public void testCheckVectorIntIntBoolean() {
        Mat mm = new Mat(5, 1, CvType.CV_8UC(10));
        Mat roi = new Mat(5, 3, CvType.CV_8UC(10)).submat(1, 3, 2, 3);

        assertEquals(5, mm.checkVector(10, CvType.CV_8U, true));
        assertEquals(5, mm.checkVector(10, CvType.CV_8U, false));
        assertEquals(2, roi.checkVector(10, CvType.CV_8U, false));
        assertTrue(0 > roi.checkVector(10, CvType.CV_8U, true));
    }

    public void testClone() {
        dst = gray0.clone();
        assertMatEqual(gray0, dst);
        assertFalse(gray0.getNativeObjAddr() == dst.getNativeObjAddr());
        assertFalse(gray0.dataAddr() == dst.dataAddr());
    }

    public void testCol() {
        Mat col = gray0.col(0);
        assertEquals(1, col.cols());
        assertEquals(gray0.rows(), col.rows());
    }

    public void testColRangeIntInt() {
        Mat cols = gray0.colRange(0, gray0.cols() / 2);

        assertEquals(gray0.cols() / 2, cols.cols());
        assertEquals(gray0.rows(), cols.rows());
    }

    public void testColRangeRange() {
        Range range = new Range(0, 5);
        dst = gray0.colRange(range);

        truth = new Mat(10, 5, CvType.CV_8UC1, new Scalar(0.0));
        assertMatEqual(truth, dst);
    }

    public void testCols() {
        assertEquals(matSize, gray0.cols());
    }

    public void testConvertToMatInt() {
        gray255.convertTo(dst, CvType.CV_32F);

        truth = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(255));
        assertMatEqual(truth, dst, EPS);
    }

    public void testConvertToMatIntDouble() {
        gray2.convertTo(dst, CvType.CV_16U, 2.0);

        truth = new Mat(matSize, matSize, CvType.CV_16U, new Scalar(4));
        assertMatEqual(truth, dst);
    }

    public void testConvertToMatIntDoubleDouble() {
        gray0_32f.convertTo(dst, CvType.CV_8U, 2.0, 4.0);

        truth = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(4));
        assertMatEqual(truth, dst);
    }

    public void testCopyToMat() {
        rgbLena.copyTo(dst);
        assertMatEqual(rgbLena, dst);
    }

    public void testCopyToMatMat() {
        Mat src = new Mat(4, 4, CvType.CV_8U, new Scalar(5));
        Mat mask = makeMask(src.clone());

        src.copyTo(dst, mask);

        truth = new Mat(4, 4, CvType.CV_8U) {
            {
                put(0, 0, 0, 0, 5, 5);
                put(1, 0, 0, 0, 5, 5);
                put(2, 0, 0, 0, 5, 5);
                put(3, 0, 0, 0, 5, 5);
            }
        };
        assertMatEqual(truth, dst);

    }

    public void testCreateIntIntInt() {
        gray255.create(4, 5, CvType.CV_32F);

        assertEquals(4, gray255.rows());
        assertEquals(5, gray255.cols());
        assertEquals(CvType.CV_32F, gray255.type());
    }

    public void testCreateSizeInt() {
        Size size = new Size(5, 5);
        dst.create(size, CvType.CV_16U);

        assertEquals(5, dst.rows());
        assertEquals(5, dst.cols());
        assertEquals(CvType.CV_16U, dst.type());
    }

    public void testCreateIntArrayInt() {
        int[] dims = new int[] {5, 6, 7};
        dst.create(dims, CvType.CV_16U);

        assertEquals(5, dst.size(0));
        assertEquals(6, dst.size(1));
        assertEquals(7, dst.size(2));
        assertEquals(CvType.CV_16U, dst.type());
    }

    public void testCross() {
        Mat answer = new Mat(1, 3, CvType.CV_32F);
        answer.put(0, 0, 7.0, 1.0, -5.0);

        Mat cross = v1.cross(v2);
        assertMatEqual(answer, cross, EPS);
    }

    public void testDataAddr() {
        assertTrue(0 != gray0.dataAddr());
        assertEquals(0, new Mat().dataAddr());
    }

    public void testDepth() {
        assertEquals(CvType.CV_8U, gray0.depth());
        assertEquals(CvType.CV_32F, gray0_32f.depth());
    }

    public void testDiag() {
        dst = gray0.diag();

        truth = new Mat(10, 1, CvType.CV_8UC1, new Scalar(0));
        assertMatEqual(truth, dst);
    }

    public void testDiagInt() {
        dst = gray255.diag(2);

        truth = new Mat(8, 1, CvType.CV_8UC1, new Scalar(255));
        assertMatEqual(truth, dst);
    }

    public void testDiagMat() {
        Mat diagVector = new Mat(matSize, 1, CvType.CV_32F, new Scalar(1));

        dst = Mat.diag(diagVector);

        assertMatEqual(grayE_32f, dst, EPS);
    }

    public void testDiagMat_sqrMatrix() {
        try {
            dst = Mat.diag(gray255);
        } catch (CvException e) {
            // expected
        }
    }

    public void testDot() {
        double s = v1.dot(v2);
        assertEquals(11.0, s);
    }

    public void testDump() {
        assertEquals("[1, 3, 2]", v1.dump());
    }

    public void testElemSize() {
        assertEquals(Byte.SIZE / 8 * gray0.channels(), gray0.elemSize());
        assertEquals(Float.SIZE / 8 * gray0_32f.channels(), gray0_32f.elemSize());
        assertEquals(Byte.SIZE / 8 * rgbLena.channels(), rgbLena.elemSize());
    }

    public void testElemSize1() {
        assertEquals(Byte.SIZE / 8, gray255.elemSize1());
        assertEquals(Double.SIZE / 8, gray0_64f.elemSize1());
        assertEquals(Byte.SIZE / 8, rgbLena.elemSize1());
    }

    public void testEmpty() {
        assertTrue(dst.empty());
        assertFalse(gray0.empty());
    }

    public void testEyeIntIntInt() {
        Mat eye = Mat.eye(3, 3, CvType.CV_32FC1);

        assertMatEqual(eye, eye.inv(), EPS);
    }

    public void testEyeSizeInt() {
        Size size = new Size(5, 5);

        Mat eye = Mat.eye(size, CvType.CV_32S);

        assertEquals(5, Core.countNonZero(eye));

    }

    public Mat getTestMat(int size, int type) {
        Mat m = new Mat(size, size, type);
        final int ch = CvType.channels(type);
        double buff[] = new double[size*size * ch];
        for(int i=0; i<size; i++)
            for(int j=0; j<size; j++)
                for(int k=0; k<ch; k++) {
                    buff[i*size*ch + j*ch + k] = 100*i + 10*j + k;
                }
        m.put(0, 0, buff);
        return m;
    }

    public void testGetIntInt_8U() {
        Mat m = getTestMat(5, CvType.CV_8UC2);

        // whole Mat
        assertTrue(Arrays.equals(new double[] {0, 1}, m.get(0, 0)));
        assertTrue(Arrays.equals(new double[] {240, 241}, m.get(2, 4)));
        assertTrue(Arrays.equals(new double[] {255, 255}, m.get(4, 4)));

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        assertTrue(Arrays.equals(new double[] {230, 231}, sm.get(0, 0)));
        assertTrue(Arrays.equals(new double[] {255, 255}, sm.get(1, 1)));
    }

    public void testGetIntInt_32S() {
        Mat m = getTestMat(5, CvType.CV_32SC3);

        // whole Mat
        assertTrue(Arrays.equals(new double[] {0, 1, 2}, m.get(0, 0)));
        assertTrue(Arrays.equals(new double[] {240, 241, 242}, m.get(2, 4)));
        assertTrue(Arrays.equals(new double[] {440, 441, 442}, m.get(4, 4)));

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        assertTrue(Arrays.equals(new double[] {230, 231, 232}, sm.get(0, 0)));
        assertTrue(Arrays.equals(new double[] {340, 341, 342}, sm.get(1, 1)));
    }

    public void testGetIntInt_64F() {
        Mat m = getTestMat(5, CvType.CV_64FC1);

        // whole Mat
        assertTrue(Arrays.equals(new double[] {0}, m.get(0, 0)));
        assertTrue(Arrays.equals(new double[] {240}, m.get(2, 4)));
        assertTrue(Arrays.equals(new double[] {440}, m.get(4, 4)));

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        assertTrue(Arrays.equals(new double[] {230}, sm.get(0, 0)));
        assertTrue(Arrays.equals(new double[] {340}, sm.get(1, 1)));
    }

    public void testGetIntIntByteArray() {
        Mat m = getTestMat(5, CvType.CV_8UC3);
        byte[] goodData = new byte[9];
        byte[] badData = new byte[7];

        // whole Mat
        int bytesNum = m.get(1, 1, goodData);

        assertEquals(9, bytesNum);
        assertTrue(Arrays.equals(new byte[] { 110, 111, 112, 120, 121, 122, (byte) 130, (byte) 131, (byte) 132 }, goodData));

        try {
            m.get(2, 2, badData);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        byte buff00[] = new byte[3];
        bytesNum = sm.get(0, 0, buff00);
        assertEquals(3, bytesNum);
        assertTrue(Arrays.equals(new byte[] {(byte) 230, (byte) 231, (byte) 232}, buff00));
        byte buff11[] = new byte[3];
        bytesNum = sm.get(1, 1, buff11);
        assertEquals(3, bytesNum);
        assertTrue(Arrays.equals(new byte[] {(byte) 255, (byte) 255, (byte) 255}, buff11));
    }

    public void testGetIntIntDoubleArray() {
        Mat m = getTestMat(5, CvType.CV_64F);
        double buff[] = new double[4];

        // whole Mat
        int bytesNum = m.get(1, 1, buff);

        assertEquals(32, bytesNum);
        assertTrue(Arrays.equals(new double[] { 110, 120, 130, 140 }, buff));

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        double buff00[] = new double[2];
        bytesNum = sm.get(0, 0, buff00);
        assertEquals(16, bytesNum);
        assertTrue(Arrays.equals(new double[] {230, 240}, buff00));
        double buff11[] = new double[] {0, 0};
        bytesNum = sm.get(1, 1, buff11);
        assertEquals(8, bytesNum);
        assertTrue(Arrays.equals(new double[] {340, 0}, buff11));
}

    public void testGetIntIntFloatArray() {
        Mat m = getTestMat(5, CvType.CV_32F);
        float buff[] = new float[4];

        // whole Mat
        int bytesNum = m.get(1, 1, buff);

        assertEquals(16, bytesNum);
        assertTrue(Arrays.equals(new float[] { 110, 120, 130, 140 }, buff));

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        float buff00[] = new float[2];
        bytesNum = sm.get(0, 0, buff00);
        assertEquals(8, bytesNum);
        assertTrue(Arrays.equals(new float[] {230, 240}, buff00));
        float buff11[] = new float[] {0, 0};
        bytesNum = sm.get(1, 1, buff11);
        assertEquals(4, bytesNum);
        assertTrue(Arrays.equals(new float[] {340, 0}, buff11));
    }

    public void testGetIntIntIntArray() {
        Mat m = getTestMat(5, CvType.CV_32SC2);
        int[] buff = new int[6];

        // whole Mat
        int bytesNum = m.get(1, 1, buff);

        assertEquals(24, bytesNum);
        assertTrue(Arrays.equals(new int[] { 110, 111, 120, 121, 130, 131 }, buff));

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        int buff00[] = new int[4];
        bytesNum = sm.get(0, 0, buff00);
        assertEquals(16, bytesNum);
        assertTrue(Arrays.equals(new int[] {230, 231, 240, 241}, buff00));
        int buff11[] = new int[]{0, 0, 0, 0};
        bytesNum = sm.get(1, 1, buff11);
        assertEquals(8, bytesNum);
        assertTrue(Arrays.equals(new int[] {340, 341, 0, 0}, buff11));
    }

    public void testGetIntIntShortArray() {
        Mat m = getTestMat(5, CvType.CV_16SC2);
        short[] buff = new short[6];

        // whole Mat
        int bytesNum = m.get(1, 1, buff);

        assertEquals(12, bytesNum);
        assertTrue(Arrays.equals(new short[] { 110, 111, 120, 121, 130, 131 }, buff));

        // sub-Mat
        Mat sm = m.submat(2, 4, 3, 5);
        short buff00[] = new short[4];
        bytesNum = sm.get(0, 0, buff00);
        assertEquals(8, bytesNum);
        assertTrue(Arrays.equals(new short[] {230, 231, 240, 241}, buff00));
        short buff11[] = new short[]{0, 0, 0, 0};
        bytesNum = sm.get(1, 1, buff11);
        assertEquals(4, bytesNum);
        assertTrue(Arrays.equals(new short[] {340, 341, 0, 0}, buff11));
    }

    public void testGetNativeObjAddr() {
        assertTrue(0 != gray0.getNativeObjAddr());
    }

    public void testHeight() {
        assertEquals(gray0.rows(), gray0.height());
        assertEquals(rgbLena.rows(), rgbLena.height());
        assertEquals(rgba128.rows(), rgba128.height());
    }

    public void testInv() {
        dst = grayE_32f.inv();
        assertMatEqual(grayE_32f, dst, EPS);
    }

    public void testInvInt() {
        Mat src = new Mat(2, 2, CvType.CV_32F) {
            {
                put(0, 0, 1.0);
                put(0, 1, 2.0);
                put(1, 0, 1.5);
                put(1, 1, 4.0);
            }
        };

        dst = src.inv(Core.DECOMP_CHOLESKY);

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

    public void testIsContinuous() {
        assertTrue(gray0.isContinuous());

        Mat subMat = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols() / 2);
        assertFalse(subMat.isContinuous());
    }

    public void testIsSubmatrix() {
        assertFalse(gray0.isSubmatrix());
        Mat subMat = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols() / 2);
        assertTrue(subMat.isSubmatrix());
    }

    public void testLocateROI() {
        Mat roi = gray0.submat(3, 5, 7, 10);
        Point offset = new Point();
        Size size = new Size();

        roi.locateROI(size, offset);

        assertPointEquals(new Point(7, 3), offset, EPS);
        assertSizeEquals(new Size(10, 10), size, EPS);
    }

    public void testMat() {
        Mat m = new Mat();
        assertNotNull(m);
        assertTrue(m.empty());
    }

    public void testMatIntIntCvType() {
        Mat gray = new Mat(1, 1, CvType.CV_8UC1);
        assertFalse(gray.empty());

        Mat rgb = new Mat(1, 1, CvType.CV_8UC3);
        assertFalse(rgb.empty());
    }

    public void testMatIntIntCvTypeScalar() {
        dst = new Mat(gray127.rows(), gray127.cols(), CvType.CV_8U, new Scalar(127));
        assertFalse(dst.empty());
        assertMatEqual(dst, gray127);

        dst = new Mat(rgba128.rows(), rgba128.cols(), CvType.CV_8UC4, Scalar.all(128));
        assertFalse(dst.empty());
        assertMatEqual(dst, rgba128);
    }

    public void testMatIntIntInt() {
        Mat gray = new Mat(1, 1, CvType.CV_8U);
        assertFalse(gray.empty());

        Mat rgb = new Mat(1, 1, CvType.CV_8U);
        assertFalse(rgb.empty());
    }

    public void testMatIntIntIntScalar() {
        Mat m1 = new Mat(gray127.rows(), gray127.cols(), CvType.CV_8U, new Scalar(127));
        assertFalse(m1.empty());
        assertMatEqual(m1, gray127);

        Mat m2 = new Mat(gray0_32f.rows(), gray0_32f.cols(), CvType.CV_32F, new Scalar(0));
        assertFalse(m2.empty());
        assertMatEqual(m2, gray0_32f, EPS);
    }

    public void testMatMatRange() {
        dst = new Mat(gray0, new Range(0, 5));

        truth = new Mat(5, 10, CvType.CV_8UC1, new Scalar(0));
        assertFalse(dst.empty());
        assertMatEqual(truth, dst);
    }

    public void testMatMatRangeRange() {
        dst = new Mat(gray255_32f, new Range(0, 5), new Range(0, 5));

        truth = new Mat(5, 5, CvType.CV_32FC1, new Scalar(255));

        assertFalse(dst.empty());
        assertMatEqual(truth, dst, EPS);
    }

    public void testMatMatRangeArray() {
        dst = new Mat(gray255_32f_3d, new Range[]{new Range(0, 5), new Range(0, 5), new Range(0, 5)});

        truth = new Mat(new int[] {5, 5, 5}, CvType.CV_32FC1, new Scalar(255));

        assertFalse(dst.empty());
        assertMatEqual(truth, dst, EPS);
    }

    public void testMatMatRect() {
        Mat m = new Mat(7, 6, CvType.CV_32SC1);
        m.put(0,  0,
                 0,  1,  2,  3,  4,  5,
                10, 11, 12, 13, 14, 15,
                20, 21, 22, 23, 24, 25,
                30, 31, 32, 33, 34, 35,
                40, 41, 42, 43, 44, 45,
                50, 51, 52, 53, 54, 55,
                60, 61, 62, 63, 64, 65 );

        dst = new Mat(m, new Rect(1, 2, 3, 4));

        truth = new Mat(4, 3, CvType.CV_32SC1);
        truth.put(0,  0,
                21, 22, 23,
                31, 32, 33,
                41, 42, 43,
                51, 52, 53 );

        assertFalse(dst.empty());
        assertMatEqual(truth, dst);
    }

    public void testMatSizeInt() {
        dst = new Mat(new Size(10, 10), CvType.CV_8U);

        assertFalse(dst.empty());
    }

    public void testMatSizeIntScalar() {
        dst = new Mat(new Size(10, 10), CvType.CV_32F, new Scalar(255));

        assertFalse(dst.empty());
        assertMatEqual(gray255_32f, dst, EPS);
    }

    public void testMatIntArrayIntScalar() {
        dst = new Mat(new int[]{10, 10, 10}, CvType.CV_32F, new Scalar(255));

        assertFalse(dst.empty());
        assertMatEqual(gray255_32f_3d, dst, EPS);
    }

    public void testMulMat() {
        assertMatEqual(gray0, gray0.mul(gray255));

        Mat m1 = new Mat(2, 2, CvType.CV_32F, new Scalar(2));
        Mat m2 = new Mat(2, 2, CvType.CV_32F, new Scalar(3));

        dst = m1.mul(m2);

        truth = new Mat(2, 2, CvType.CV_32F, new Scalar(6));
        assertMatEqual(truth, dst, EPS);

    }

    public void testMulMat3d() {
        Mat m1 = new Mat(new int[] {2, 2, 2}, CvType.CV_32F, new Scalar(2));
        Mat m2 = new Mat(new int[] {2, 2, 2}, CvType.CV_32F, new Scalar(3));

        dst = m1.mul(m2);

        truth = new Mat(new int[] {2, 2, 2}, CvType.CV_32F, new Scalar(6));
        assertMatEqual(truth, dst, EPS);
    }

    public void testMulMatDouble() {
        Mat m1 = new Mat(2, 2, CvType.CV_32F, new Scalar(2));
        Mat m2 = new Mat(2, 2, CvType.CV_32F, new Scalar(3));

        dst = m1.mul(m2, 3.0);

        truth = new Mat(2, 2, CvType.CV_32F, new Scalar(18));
        assertMatEqual(truth, dst, EPS);
    }

    public void testOnesIntIntInt() {
        dst = Mat.ones(matSize, matSize, CvType.CV_32F);

        truth = new Mat(matSize, matSize, CvType.CV_32F, new Scalar(1));
        assertMatEqual(truth, dst, EPS);
    }

    public void testOnesSizeInt() {
        dst = Mat.ones(new Size(2, 2), CvType.CV_16S);
        truth = new Mat(2, 2, CvType.CV_16S, new Scalar(1));
        assertMatEqual(truth, dst);
    }

    public void testOnesIntArrayInt() {
        dst = Mat.ones(new int[]{2, 2, 2}, CvType.CV_16S);
        truth = new Mat(new int[]{2, 2, 2}, CvType.CV_16S, new Scalar(1));
        assertMatEqual(truth, dst);
    }

    public void testPush_back() {
        Mat m1 = new Mat(2, 4, CvType.CV_32F, new Scalar(2));
        Mat m2 = new Mat(3, 4, CvType.CV_32F, new Scalar(3));

        m1.push_back(m2);

        truth = new Mat(5, 4, CvType.CV_32FC1) {
            {
                put(0, 0, 2, 2, 2, 2);
                put(1, 0, 2, 2, 2, 2);
                put(2, 0, 3, 3, 3, 3);
                put(3, 0, 3, 3, 3, 3);
                put(4, 0, 3, 3, 3, 3);
            }
        };

        assertMatEqual(truth, m1, EPS);
    }

    public void testPutIntIntByteArray() {
        Mat m = new Mat(5, 5, CvType.CV_8UC3, new Scalar(1, 2, 3));
        Mat sm = m.submat(2, 4, 3, 5);
        byte[] buff  = new byte[] { 0, 0, 0, 0, 0, 0 };
        byte[] buff0 = new byte[] { 10, 20, 30, 40, 50, 60 };
        byte[] buff1 = new byte[] { -1, -2, -3, -4, -5, -6 };

        int bytesNum = m.put(1, 2, buff0);

        assertEquals(6, bytesNum);
        bytesNum = m.get(1, 2, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff0));

        bytesNum = sm.put(0, 0, buff1);

        assertEquals(6, bytesNum);
        bytesNum = sm.get(0, 0, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff1));
        bytesNum = m.get(2, 3, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff1));

        Mat m1 = m.row(1);
        bytesNum = m1.get(0, 2, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff0));

        try {
            byte[] bytes2 = new byte[] { 10, 20, 30, 40, 50 };
            m.put(2, 2, bytes2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testPutIntArrayByteArray() {
        Mat m = new Mat(new int[]{5, 5, 5}, CvType.CV_8UC3, new Scalar(1, 2, 3));
        Mat sm = m.submat(new Range[]{ new Range(0, 2), new Range(1, 3), new Range(2, 4)});
        byte[] buff  = new byte[] { 0, 0, 0, 0, 0, 0 };
        byte[] buff0 = new byte[] { 10, 20, 30, 40, 50, 60 };
        byte[] buff1 = new byte[] { -1, -2, -3, -4, -5, -6 };

        int bytesNum = m.put(new int[]{1, 2, 0}, buff0);

        assertEquals(6, bytesNum);
        bytesNum = m.get(new int[]{1, 2, 0}, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff0));

        bytesNum = sm.put(new int[]{0, 0, 0}, buff1);

        assertEquals(6, bytesNum);
        bytesNum = sm.get(new int[]{0, 0, 0}, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff1));

        bytesNum = m.get(new int[]{0, 1, 2}, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff1));

        Mat m1 = m.submat(new Range[]{ new Range(1,2), Range.all(), Range.all() });
        bytesNum = m1.get(new int[]{ 0, 2, 0}, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, buff0));

        try {
            byte[] bytes2 = new byte[] { 10, 20, 30, 40, 50 };
            m.put(new int[]{ 2, 2, 0 }, bytes2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }

    }

    public void testPutIntIntDoubleArray() {
        Mat m = new Mat(5, 5, CvType.CV_8UC3, new Scalar(1, 2, 3));
        Mat sm = m.submat(2, 4, 3, 5);
        byte[] buff  = new byte[] { 0, 0, 0, 0, 0, 0 };

        int bytesNum = m.put(1, 2, 10, 20, 30, 40, 50, 60);

        assertEquals(6, bytesNum);
        bytesNum = m.get(1, 2, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, new byte[]{10, 20, 30, 40, 50, 60}));

        bytesNum = sm.put(0, 0, 255, 254, 253, 252, 251, 250);

        assertEquals(6, bytesNum);
        bytesNum = sm.get(0, 0, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, new byte[]{-1, -2, -3, -4, -5, -6}));
        bytesNum = m.get(2, 3, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, new byte[]{-1, -2, -3, -4, -5, -6}));
    }

    public void testPutIntArrayDoubleArray() {
        Mat m = new Mat(new int[]{5, 5, 5}, CvType.CV_8UC3, new Scalar(1, 2, 3));
        Mat sm = m.submat(new Range[]{ new Range(0, 2), new Range(1, 3), new Range(2, 4)});
        byte[] buff  = new byte[] { 0, 0, 0, 0, 0, 0 };

        int bytesNum = m.put(new int[]{1, 2, 0}, 10, 20, 30, 40, 50, 60);

        assertEquals(6, bytesNum);
        bytesNum = m.get(new int[]{1, 2, 0}, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, new byte[]{10, 20, 30, 40, 50, 60}));

        bytesNum = sm.put(new int[]{0, 0, 0}, 255, 254, 253, 252, 251, 250);

        assertEquals(6, bytesNum);
        bytesNum = sm.get(new int[]{0, 0, 0}, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, new byte[]{-1, -2, -3, -4, -5, -6}));
        bytesNum = m.get(new int[]{0, 1, 2}, buff);
        assertEquals(6, bytesNum);
        assertTrue(Arrays.equals(buff, new byte[]{-1, -2, -3, -4, -5, -6}));
    }

    public void testPutIntIntFloatArray() {
        Mat m = new Mat(5, 5, CvType.CV_32FC3, new Scalar(1, 2, 3));
        float[] elements = new float[] { 10, 20, 30, 40, 50, 60 };

        int bytesNum = m.put(4, 3, elements);

        assertEquals(elements.length * 4, bytesNum);
        Mat m1 = m.row(4);
        float buff[] = new float[3];
        bytesNum = m1.get(0, 4, buff);
        assertEquals(buff.length * 4, bytesNum);
        assertTrue(Arrays.equals(new float[]{40, 50, 60}, buff));
        assertArrayEquals(new double[]{10, 20, 30}, m.get(4, 3), EPS);

        try {
            float[] elements2 = new float[] { 10, 20, 30, 40, 50 };
            m.put(2, 2, elements2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testPutIntArrayFloatArray() {
        Mat m = new Mat(new int[]{5, 5, 5}, CvType.CV_32FC3, new Scalar(1, 2, 3));
        float[] elements = new float[] { 10, 20, 30, 40, 50, 60 };

        int bytesNum = m.put(new int[]{0, 4, 3}, elements);

        assertEquals(elements.length * 4, bytesNum);
        Mat m1 = m.submat(new Range[]{ Range.all(), new Range(4, 5), Range.all() });
        float buff[] = new float[3];
        bytesNum = m1.get(new int[]{ 0, 0, 4 }, buff);
        assertEquals(buff.length * 4, bytesNum);
        assertTrue(Arrays.equals(new float[]{40, 50, 60}, buff));
        assertArrayEquals(new double[]{10, 20, 30}, m.get(new int[]{ 0, 4, 3 }), EPS);

        try {
            float[] elements2 = new float[] { 10, 20, 30, 40, 50 };
            m.put(new int[]{4, 2, 2}, elements2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testPutIntIntIntArray() {
        Mat m = new Mat(5, 5, CvType.CV_32SC3, new Scalar(-1, -2, -3));
        int[] elements = new int[] { 10, 20, 30, 40, 50, 60 };

        int bytesNum = m.put(0, 4, elements);

        assertEquals(elements.length * 4, bytesNum);
        Mat m1 = m.col(4);
        int buff[] = new int[3];
        bytesNum = m1.get(0, 0, buff);
        assertEquals(buff.length * 4, bytesNum);
        assertTrue(Arrays.equals(new int[]{10, 20, 30}, buff));
        assertArrayEquals(new double[]{40, 50, 60}, m.get(1, 0), EPS);

        try {
            int[] elements2 = new int[] { 10, 20, 30, 40, 50 };
            m.put(2, 2, elements2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testPutIntArrayIntArray() {
        Mat m = new Mat(new int[]{5, 5, 5}, CvType.CV_32SC3, new Scalar(-1, -2, -3));
        int[] elements = new int[] { 10, 20, 30, 40, 50, 60 };

        int bytesNum = m.put(new int[]{ 0, 0, 4 }, elements);

        assertEquals(elements.length * 4, bytesNum);
        Mat m1 = m.submat(new Range[]{ Range.all(), Range.all(), new Range(4, 5)});
        int buff[] = new int[3];
        bytesNum = m1.get(new int[]{ 0, 0, 0 }, buff);
        assertEquals(buff.length * 4, bytesNum);
        assertTrue(Arrays.equals(new int[]{ 10, 20, 30 }, buff));
        assertArrayEquals(new double[]{ 40, 50, 60 }, m.get(new int[]{ 0, 1, 0 }), EPS);

        try {
            int[] elements2 = new int[] { 10, 20, 30, 40, 50 };
            m.put(new int[] { 2, 2, 0 }, elements2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testPutIntIntShortArray() {
        Mat m = new Mat(5, 5, CvType.CV_16SC3, new Scalar(-1, -2, -3));
        short[] elements = new short[] { 10, 20, 30, 40, 50, 60 };

        int bytesNum = m.put(2, 3, elements);

        assertEquals(elements.length * 2, bytesNum);
        Mat m1 = m.col(3);
        short buff[] = new short[3];
        bytesNum = m1.get(2, 0, buff);
        assertTrue(Arrays.equals(new short[]{10, 20, 30}, buff));
        assertArrayEquals(new double[]{40, 50, 60}, m.get(2, 4), EPS);

        try {
            short[] elements2 = new short[] { 10, 20, 30, 40, 50 };
            m.put(2, 2, elements2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testPutIntArrayShortArray() {
        Mat m = new Mat(new int[]{ 5, 5, 5}, CvType.CV_16SC3, new Scalar(-1, -2, -3));
        short[] elements = new short[] { 10, 20, 30, 40, 50, 60 };

        int bytesNum = m.put(new int[]{ 0, 2, 3 }, elements);

        assertEquals(elements.length * 2, bytesNum);
        Mat m1 = m.submat(new Range[]{ Range.all(), Range.all(), new Range(3, 4)});
        short buff[] = new short[3];
        bytesNum = m1.get(new int[]{ 0, 2, 0 }, buff);
        assertTrue(Arrays.equals(new short[]{10, 20, 30}, buff));
        assertArrayEquals(new double[]{40, 50, 60}, m.get(new int[]{ 0, 2, 4 }), EPS);

        try {
            short[] elements2 = new short[] { 10, 20, 30, 40, 50 };
            m.put(new int[] { 2, 2, 0 }, elements2);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testRelease() {
        assertFalse(gray0.empty());
        assertTrue(gray0.rows() > 0);

        gray0.release();

        assertTrue(gray0.empty());
        assertEquals(0, gray0.rows());
        assertEquals(0, gray0.dataAddr());
    }

    public void testReshapeInt() {
        Mat src = new Mat(4, 4, CvType.CV_8U, new Scalar(0));
        dst = src.reshape(4);

        truth = new Mat(4, 1, CvType.CV_8UC4, new Scalar(0));
        assertMatEqual(truth, dst);
    }

    public void testReshapeIntInt() {
        Mat src = new Mat(5, 7, CvType.CV_8U, new Scalar(0));
        dst = src.reshape(7, 5);

        truth = new Mat(5, 1, CvType.CV_8UC(7), new Scalar(0));
        assertMatEqual(truth, dst);
    }

    public void testReshapeIntIntArray() {
        // 2D -> 4D
        Mat src = new Mat(6, 5, CvType.CV_8UC3, new Scalar(0));
        assertEquals(2, src.dims());
        assertEquals(src.rows(), src.size(0));
        assertEquals(src.cols(), src.size(1));

        int[] newShape = {1, src.channels() * src.cols(), 1, src.rows()};
        dst = src.reshape(1, newShape);
        assertEquals(newShape.length, dst.dims());
        for (int i = 0; i < newShape.length; ++i)
            assertEquals(newShape[i], dst.size(i));

        // 3D -> 2D
        src = new Mat(new int[]{4, 6, 7}, CvType.CV_8UC3, new Scalar(0));
        assertEquals(3, src.dims());
        assertEquals(4, src.size(0));
        assertEquals(6, src.size(1));
        assertEquals(7, src.size(2));

        int[] newShape2 = {src.channels() * src.size(2), src.size(0) * src.size(1)};
        dst = src.reshape(1, newShape2);
        assertEquals(newShape2.length, dst.dims());
        for (int i = 0; i < newShape2.length; ++i)
            assertEquals(newShape2[i], dst.size(i));
    }

    public void testCopySize() {
        Mat src = new Mat(new int[]{1, 1, 10, 10}, CvType.CV_8UC1, new Scalar(1));
        assertEquals(4, src.dims());
        assertEquals(1, src.size(0));
        assertEquals(1, src.size(1));
        assertEquals(10, src.size(2));
        assertEquals(10, src.size(3));
        Mat other = new Mat(new int[]{10, 10}, src.type());

        src.copySize(other);
        assertEquals(other.dims(), src.dims());
        for (int i = 0; i < other.dims(); ++i)
            assertEquals(other.size(i), src.size(i));
    }

    public void testRow() {
        Mat row = gray0.row(0);
        assertEquals(1, row.rows());
        assertEquals(gray0.cols(), row.cols());
    }

    public void testRowRangeIntInt() {
        Mat rows = gray0.rowRange(0, gray0.rows() / 2);
        assertEquals(gray0.rows() / 2, rows.rows());
        assertEquals(gray0.cols(), rows.cols());
    }

    public void testRowRangeRange() {
        Mat rows = gray255.rowRange(new Range(0, 5));

        assertEquals(gray255.rows() / 2, rows.rows());
        assertEquals(gray255.cols(), rows.cols());
    }

    public void testRows() {
        assertEquals(matSize, gray0.rows());
    }

    public void testSetToMat() {
        Mat vals = new Mat(7, 1, CvType.CV_8U) {
            {
                put(0, 0, 1, 2, 3, 4, 5, 6, 7);
            }
        };
        Mat dst = new Mat(1, 1, CvType.CV_8UC(7));

        dst.setTo(vals);

        Mat truth = new Mat(1, 1, CvType.CV_8UC(7)) {
            {
                put(0, 0, 1, 2, 3, 4, 5, 6, 7);
            }
        };
        assertMatEqual(truth, dst);
    }

    public void testSetToMatMat() {
        Mat vals = new Mat(7, 1, CvType.CV_8U) {
            {
                put(0, 0, 1, 2, 3, 4, 5, 6, 7);
            }
        };
        Mat dst = Mat.zeros(2, 1, CvType.CV_8UC(7));
        Mat mask = new Mat(2, 1, CvType.CV_8U) {
            {
                put(0, 0, 0, 1);
            }
        };

        dst.setTo(vals, mask);

        Mat truth = new Mat(2, 1, CvType.CV_8UC(7)) {
            {
                put(0, 0, 0, 0, 0, 0, 0, 0, 0);
                put(1, 0, 1, 2, 3, 4, 5, 6, 7);
            }
        };
        assertMatEqual(truth, dst);
    }

    public void testSetToScalar() {
        gray0.setTo(new Scalar(127));
        assertMatEqual(gray127, gray0);
    }

    public void testSetToScalarMask() {
        Mat mask = gray0.clone();
        mask.put(1, 1, 1, 2, 3);
        gray0.setTo(new Scalar(1), mask);
        assertEquals(3, Core.countNonZero(gray0));
        Core.subtract(gray0, mask, gray0);
        assertEquals(0, Core.countNonZero(gray0));
    }

    public void testSize() {
        assertEquals(new Size(matSize, matSize), gray0.size());

        assertEquals(new Size(3, 1), v1.size());
    }

    public void testStep1() {
        assertEquals(matSize * CvType.channels(CvType.CV_8U), gray0.step1());

        assertEquals(3, v2.step1());
    }

    public void testStep1Int() {
        Mat roi = rgba0.submat(3, 5, 7, 10);
        Mat m = roi.clone();

        assertTrue(rgba0.channels() * rgba0.cols() <= roi.step1(0));
        assertEquals(rgba0.channels(), roi.step1(1));
        assertTrue(m.channels() * (10 - 7) <= m.step1(0));
        assertEquals(m.channels(), m.step1(1));
    }

    public void testSubmatIntIntIntInt() {
        Mat submat = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols() / 2);

        assertTrue(submat.isSubmatrix());
        assertFalse(submat.isContinuous());
        assertEquals(gray0.rows() / 2, submat.rows());
        assertEquals(gray0.cols() / 2, submat.cols());
    }

    public void testSubmatRangeRange() {
        Mat submat = gray255.submat(new Range(2, 4), new Range(2, 4));
        assertTrue(submat.isSubmatrix());
        assertFalse(submat.isContinuous());

        assertEquals(2, submat.rows());
        assertEquals(2, submat.cols());
    }

    public void testSubmatRangeArray() {
        Mat submat = gray255_32f_3d.submat(new Range[]{ new Range(2, 4), new Range(2, 4), new Range(3, 6) });
        assertTrue(submat.isSubmatrix());
        assertFalse(submat.isContinuous());

        assertEquals(2, submat.size(0));
        assertEquals(2, submat.size(1));
        assertEquals(3, submat.size(2));
    }

    public void testSubmatRect() {
        Mat submat = gray255.submat(new Rect(5, 5, gray255.cols() / 2, gray255.rows() / 2));
        assertTrue(submat.isSubmatrix());
        assertFalse(submat.isContinuous());

        assertEquals(gray255.rows() / 2, submat.rows());
        assertEquals(gray255.cols() / 2, submat.cols());
    }

    public void testT() {
        assertMatEqual(gray255, gray255.t());

        Mat src = new Mat(3, 3, CvType.CV_16U) {
            {
                put(0, 0, 1, 2, 4);
                put(1, 0, 7, 5, 0);
                put(2, 0, 3, 4, 6);
            }
        };

        dst = src.t();

        truth = new Mat(3, 3, CvType.CV_16U) {
            {
                put(0, 0, 1, 7, 3);
                put(1, 0, 2, 5, 4);
                put(2, 0, 4, 0, 6);
            }
        };
        assertMatEqual(truth, dst);
    }

    public void testToString() {
        assertNotNull(gray0.toString());
    }

    public void testTotal() {
        int nElements = gray0.rows() * gray0.cols();
        assertEquals(nElements, gray0.total());
    }

    public void testType() {
        assertEquals(CvType.CV_8UC1, gray0.type());
        assertEquals(CvType.CV_32FC1, gray0_32f.type());
        assertEquals(CvType.CV_8UC3, rgbLena.type());
    }

    public void testWidth() {
        assertEquals(gray0.cols(), gray0.width());
        assertEquals(rgbLena.cols(), rgbLena.width());
        assertEquals(rgba128.cols(), rgba128.width());
    }

    public void testZerosIntIntInt() {
        dst = Mat.zeros(matSize, matSize, CvType.CV_32F);

        assertMatEqual(gray0_32f, dst, EPS);
    }

    public void testZerosSizeInt() {
        dst = Mat.zeros(new Size(2, 2), CvType.CV_16S);

        truth = new Mat(2, 2, CvType.CV_16S, new Scalar(0));
        assertMatEqual(truth, dst);
    }

    public void testZerosIntArray() {
        dst = Mat.zeros(new int[]{2, 3, 4}, CvType.CV_16S);

        truth = new Mat(new int[]{2, 3, 4}, CvType.CV_16S, new Scalar(0));
        assertMatEqual(truth, dst);
    }

    public void testMatFromByteBuffer() {
        ByteBuffer bbuf = ByteBuffer.allocateDirect(64*64);
        bbuf.putInt(0x01010101);
        Mat m = new Mat(64,64,CvType.CV_8UC1,bbuf);
        assertEquals(4, Core.countNonZero(m));
        Core.add(m, new Scalar(1), m);
        assertEquals(4096, Core.countNonZero(m));
        m.release();
        assertEquals(2, bbuf.get(0));
        assertEquals(1, bbuf.get(4095));
    }

    public void testMatFromByteBufferWithStep() {
        ByteBuffer bbuf = ByteBuffer.allocateDirect(80*64);
        bbuf.putInt(0x01010101);
        bbuf.putInt(64, 0x02020202);
        bbuf.putInt(80, 0x03030303);
        Mat m = new Mat(64, 64, CvType.CV_8UC1, bbuf, 80);
        assertEquals(8, Core.countNonZero(m));
        Core.add(m, new Scalar(5), m);
        assertEquals(4096, Core.countNonZero(m));
        m.release();
        assertEquals(6, bbuf.get(0));
        assertEquals(5, bbuf.get(63));
        assertEquals(2, bbuf.get(64));
        assertEquals(0, bbuf.get(79));
        assertEquals(8, bbuf.get(80));
        assertEquals(5, bbuf.get(63*80 + 63));
    }

}
