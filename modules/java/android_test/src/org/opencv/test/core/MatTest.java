package org.opencv.test.core;

import org.opencv.CvType;
import org.opencv.Mat;
import org.opencv.Scalar;
import org.opencv.test.OpenCVTestCase;

public class MatTest extends OpenCVTestCase {
	
	public void test_1() {
		super.test_1("Mat");
	}

	public void testChannels() {
		assertEquals(1, gray0.channels());
		assertEquals(3, rgbLena.channels());
		assertEquals(4, rgba0.channels());
	}

	public void testClone() {
		dst = gray0.clone();
		assertMatEqual(gray0, dst);
	}

	public void testCol() {
		Mat col = gray0.col(0);
		assertEquals(1, col.cols());
		assertEquals(gray0.rows(), col.rows());
	}

	public void testColRange() {
		Mat cols = gray0.colRange(0, gray0.cols()/2);
		assertEquals(gray0.cols()/2, cols.cols());
		assertEquals(gray0.rows(), cols.rows());
	}

	public void testCols() {
		assertEquals(matSize, gray0.rows());
	}

	public void testCopyTo() {
		rgbLena.copyTo(dst);
		assertMatEqual(rgbLena, dst);
	}

	public void testCross() {
		Mat answer = new Mat(1, 3, CvType.CV_32F);
		answer.put(0, 0, 7.0, 1.0, -5.0);
		
		Mat cross = v1.cross(v2);		
		assertMatEqual(answer, cross);
	}

	public void testDataAddr() {
		assertTrue(0 != gray0.dataAddr());
	}

	public void testDepth() {
		assertEquals(CvType.CV_8U, gray0.depth());
		assertEquals(CvType.CV_32F, gray0_32f.depth());
	}

	public void testDispose() {
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

	public void testEmpty() {
		assertTrue(dst.empty());
		assertTrue(!gray0.empty());
	}
	
	public void testEye() {
		Mat eye = Mat.eye(3, 3, CvType.CV_32FC1);
		assertMatEqual(eye, eye.inv());
	}

	public void testGetIntInt() {
		fail("Not yet implemented");
	}

	public void testGetIntIntByteArray() {
		fail("Not yet implemented");
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
		assertMatEqual(grayE_32f, dst);
	}

	public void testIsContinuous() {
		assertTrue(gray0.isContinuous());
		
		Mat subMat = gray0.submat(0, 0, gray0.rows()/2, gray0.cols()/2);
		assertFalse(subMat.isContinuous());
	}

	public void testIsSubmatrix() {
		assertFalse(gray0.isSubmatrix());
		Mat subMat = gray0.submat(0, 0, gray0.rows()/2, gray0.cols()/2);
		assertTrue(subMat.isSubmatrix());
	}
	
	public void testMat() {
		Mat m = new Mat();
		assertTrue(null != m);
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
		assertMatEqual(m2, gray0_32f);
	}

	public void testPutIntIntByteArray() {
		fail("Not yet implemented");
	}

	public void testPutIntIntDoubleArray() {
		fail("Not yet implemented");
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

	public void testRow() {
		Mat row = gray0.row(0);
		assertEquals(1, row.rows());
		assertEquals(gray0.cols(), row.cols());
	}

	public void testRowRange() {
		Mat rows = gray0.rowRange(0, gray0.rows()/2);
		assertEquals(gray0.rows()/2, rows.rows());
		assertEquals(gray0.cols(), rows.cols());
	}

	public void testRows() {
		assertEquals(matSize, gray0.rows());
	}

	public void testSetTo() {
		gray0.setTo(new Scalar(127));
		assertMatEqual(gray127, gray0);
	}

	public void testSubmat() {
		Mat submat = gray0.submat(0, gray0.rows()/2, 0, gray0.cols()/2);
		assertEquals(gray0.rows()/2, submat.rows());
		assertEquals(gray0.cols()/2, submat.cols());
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

}
