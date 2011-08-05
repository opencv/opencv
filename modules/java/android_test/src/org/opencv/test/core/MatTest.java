package org.opencv.test.core;

import java.util.Arrays;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.test.OpenCVTestCase;

public class MatTest extends OpenCVTestCase {

    public void testAdjustROI() {
        fail("Not yet implemented");
    }

    public void testAssignToMat() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testCols() {
        assertEquals(matSize, gray0.cols());
    }

    public void testConvertToMatInt() {
        fail("Not yet implemented");
    }

    public void testConvertToMatIntDouble() {
        fail("Not yet implemented");
    }

    public void testConvertToMatIntDoubleDouble() {
        fail("Not yet implemented");
    }

    public void testCopyTo() {
        rgbLena.copyTo(dst);
        assertMatEqual(rgbLena, dst);
    }

    public void testCopyToMat() {
        fail("Not yet implemented");
    }

    public void testCopyToMatMat() {
        fail("Not yet implemented");
    }

    public void testCreateIntIntInt() {
        fail("Not yet implemented");
    }

    public void testCreateSizeInt() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testDiagInt() {
        fail("Not yet implemented");
    }

    public void testDiagMat() {
        fail("Not yet implemented");
    }

    public void testDot() {
        double s = v1.dot(v2);
        assertEquals(11.0, s);
    }

    public void testDump() {
        assertEquals("[1, 3, 2]", v1.dump());
    }

    public void testElemSize() {
        assertEquals(1, gray0.elemSize());
        assertEquals(4, gray0_32f.elemSize());
        assertEquals(3, rgbLena.elemSize());
    }

    public void testElemSize1() {
        fail("Not yet implemented");
    }

    public void testEmpty() {
        assertTrue(dst.empty());
        assertTrue(!gray0.empty());
    }

    public void testEye() {
        Mat eye = Mat.eye(3, 3, CvType.CV_32FC1);
        assertMatEqual(eye, eye.inv(), EPS);
    }

    public void testEyeIntIntInt() {
        fail("Not yet implemented");
    }

    public void testEyeSizeInt() {
        fail("Not yet implemented");
    }

    public void testGetIntInt() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testGetIntIntFloatArray() {
        fail("Not yet implemented");
    }

    public void testGetIntIntIntArray() {
        fail("Not yet implemented");
    }

    public void testGetIntIntShortArray() {
        fail("Not yet implemented");
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

    public void testMatLong() {
        fail("Not yet implemented");
    }

    public void testMatMatRange() {
        fail("Not yet implemented");
    }

    public void testMatMatRangeRange() {
        fail("Not yet implemented");
    }

    public void testMatMatRect() {
        fail("Not yet implemented");
    }

    public void testMatSizeInt() {
        fail("Not yet implemented");
    }

    public void testMatSizeIntScalar() {
        fail("Not yet implemented");
    }

    public void testMulMat() {
        fail("Not yet implemented");
    }

    public void testMulMatDouble() {
        fail("Not yet implemented");
    }

    public void testOnesIntIntInt() {
        fail("Not yet implemented");
    }

    public void testOnesSizeInt() {
        fail("Not yet implemented");
    }

    public void testPush_back() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testReshapeIntInt() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testStep1() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testSubmatRect() {
        fail("Not yet implemented");
    }

    public void testT() {
        fail("Not yet implemented");
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
        fail("Not yet implemented");
    }

    public void testZerosSizeInt() {
        fail("Not yet implemented");
    }

}
