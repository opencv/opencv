package org.opencv.test.core;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;

import java.util.Arrays;

public class MatTest extends OpenCVTestCase {

    public void testAdjustROI() {
        fail("Not yet implemented");
    }

    public void testAssignToMat() {
        gray0.assignTo(dst);
        assertMatEqual(gray0, dst);

        gray255.assignTo(dst);
        assertMatEqual(gray255, dst);
    }

    public void testAssignToMatInt() {
        fail("Not yet implemented");
    }

    public void testChannels() {
        assertEquals(1, gray0.channels());
        assertEquals(3, rgbLena.channels());
        assertEquals(4, rgba0.channels());
    }

    public void testCheckVectorInt() {
        // ! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel
        // (1 x N) or (N x 1); negative number otherwise
        fail("Not yet implemented");
    }

    public void testCheckVectorIntInt() {
        fail("Not yet implemented");
    }

    public void testCheckVectorIntIntBoolean() {
        fail("Not yet implemented");
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
        assertEquals(Double.SIZE / 8 * gray0_32f.channels(), gray0_32f.elemSize());
        assertEquals(Byte.SIZE / 8 * rgbLena.channels(), rgbLena.elemSize());
    }

    public void testElemSize1() {
        assertEquals(Byte.SIZE / 8, gray255.elemSize1());
        assertEquals(Double.SIZE / 8, gray0_64f.elemSize1());
        assertEquals(Byte.SIZE / 8, rgbLena.elemSize1());
    }

    public void testEmpty() {
        assertTrue(dst.empty());
        assertTrue(!gray0.empty());
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

    public void testGetIntInt() {
        Mat src = new Mat(3, 3, CvType.CV_8U, new Scalar(2));
        double[] actualArray = src.get(1, 1);

        assertTrue(Arrays.equals(new double[] { 2 }, actualArray));
    }

    public void testGetIntIntByteArray() {
        Mat m = new Mat(5, 5, CvType.CV_8UC3, new Scalar(1, 2, 3));
        byte[] goodData = new byte[9];
        byte[] badData = new byte[7];
        m.get(1, 1, goodData);

        assertTrue(Arrays.equals(new byte[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 }, goodData));

        try {
            m.get(2, 2, badData);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testGetIntIntDoubleArray() {
        Mat src = new Mat(2, 2, CvType.CV_64F);
        double[] doubleArray = { 1.0, 2.0, 3.0 };

        int numOfBytes = src.get(0, 0, doubleArray);
        assertEquals(24, numOfBytes);
    }

    public void testGetIntIntFloatArray() {
        Mat src = new Mat(2, 2, CvType.CV_32F);
        float[] floatArray = { 3.0f, 1.0f, 4.0f };

        int numOfBytes = src.get(0, 0, floatArray);
        assertEquals(12, numOfBytes);
    }

    public void testGetIntIntIntArray() {
        Mat src = new Mat(2, 2, CvType.CV_32S);
        int[] intArray = { 3, 1, 4, 7 };

        int numOfBytes = src.get(0, 0, intArray);
        assertEquals(16, numOfBytes);
    }

    public void testGetIntIntShortArray() {
        Mat src = new Mat(2, 2, CvType.CV_16U);
        short[] data = { 3, 1, 4, 7 };

        int numOfBytes = src.get(1, 1, data);
        assertEquals(2, numOfBytes);
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
        fail("Not yet implemented");
        dst = gray0_32f.inv(Core.DECOMP_CHOLESKY);
        assertMatEqual(gray0_32f, dst, EPS);
    }

    public void testIsContinuous() {
        assertTrue(gray0.isContinuous());

        Mat subMat = gray0.submat(0, 0, gray0.rows() / 2, gray0.cols() / 2);
        assertFalse(subMat.isContinuous());
    }

    public void testIsSubmatrix() {
        assertFalse(gray0.isSubmatrix());
        Mat subMat = gray0.submat(0, 0, gray0.rows() / 2, gray0.cols() / 2);
        assertTrue(subMat.isSubmatrix());
    }

    public void testLocateROI() {
        fail("Not yet implemented");
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

    public void testMatMatRect() {
        dst = new Mat(gray255_32f, new Rect(2, 2, 7, 7));

        truth = new Mat(7, 7, CvType.CV_32FC1, new Scalar(255));

        assertFalse(dst.empty());
        assertMatEqual(truth, dst, EPS);
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

    public void testMulMat() {
        assertMatEqual(gray0, gray0.mul(gray255));

        Mat m1 = new Mat(2, 2, CvType.CV_32F, new Scalar(2));
        Mat m2 = new Mat(2, 2, CvType.CV_32F, new Scalar(3));

        dst = m1.mul(m2);

        truth = new Mat(2, 2, CvType.CV_32F, new Scalar(6));
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
        fail("Not yet implemented");
    }

    public void testPutIntIntDoubleArray() {
        Mat m = new Mat(5, 5, CvType.CV_8UC3);
        m.put(1, 1, 10, 20, 30, 40, 50, 60);

        try {
            m.put(2, 2, 11, 22, 33, 44, 55);
            fail("Expected UnsupportedOperationException (data.length % CvType.channels(t) != 0)");
        } catch (UnsupportedOperationException e) {
            // expected
        }
    }

    public void testPutIntIntFloatArray() {
        fail("Not yet implemented");
    }

    public void testPutIntIntIntArray() {
        fail("Not yet implemented");
    }

    public void testPutIntIntShortArray() {
        fail("Not yet implemented");
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

    public void testRow() {
        Mat row = gray0.row(0);
        assertEquals(1, row.rows());
        assertEquals(gray0.cols(), row.cols());
    }

    public void testRowRangeIntInt() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testSetToMatMat() {
        fail("Not yet implemented");
    }

    public void testSetToScalar() {
        gray0.setTo(new Scalar(127));
        assertMatEqual(gray127, gray0);
    }

    public void testSize() {
        assertEquals(new Size(matSize, matSize), gray0.size());

        assertEquals(new Size(3, 1), v1.size());
    }

    public void testStep1() {
        assertEquals(10, gray0.step1());

        assertEquals(3, v2.step1());
    }

    public void testStep1Int() {
        fail("Not yet implemented");
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

    public void testSubmatRect() {
        Mat submat = gray255.submat(new Rect(5, gray255.rows() / 2, 5, gray255.cols() / 2));
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
        assertTrue(null != gray0.toString());
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

}
