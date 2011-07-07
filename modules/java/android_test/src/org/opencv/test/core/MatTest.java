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
		fail("Not yet implemented");
	}

	public void testCross() {
		fail("Not yet implemented");
	}

	public void testDataAddr() {
		fail("Not yet implemented");
	}

	public void testDepth() {
		assertEquals(CvType.CV_8U, gray0.depth());
		assertEquals(CvType.CV_32F, gray0_32f.depth());
	}

	public void testDispose() {
		fail("Not yet implemented");
	}

	public void testDot() {
		fail("Not yet implemented");
	}
	
	public void testDump() {
		fail("Not yet implemented");
	}

	public void testElemSize() {
		fail("Not yet implemented");
	}

	public void testEmpty() {
		assertTrue(dst.empty());
		assertTrue(!gray0.empty());
	}
	
	public void testEye() {
		fail("Not yet implemented");
	}

	public void testFinalize() {
		fail("Not yet implemented");
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
		fail("Not yet implemented");
		//dst = grayE_32f.inv();
		//assertMatEqual(grayE_32f, dst);
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
		Mat gray = new Mat(1, 1, CvType.CV_8UC1, new Scalar(127));
		assertFalse(gray.empty());
		assertMatEqual(gray, gray127);
		
		Mat rgb = new Mat(1, 1, CvType.CV_8UC4, new Scalar(128));
		assertFalse(rgb.empty());
		//FIXME: assertMatEqual(rgb, rgba128);
	}

	public void testMatIntIntInt() {
		Mat gray = new Mat(1, 1, CvType.CV_8U);
		assertFalse(gray.empty());
		
		Mat rgb = new Mat(1, 1, CvType.CV_8U);
		assertFalse(rgb.empty());
	}

	public void testMatIntIntIntScalar() {
		Mat m1 = new Mat(1, 1, CvType.CV_8U, new Scalar(127));
		assertFalse(m1.empty());
		assertMatEqual(m1, gray127);
		
		Mat m2 = new Mat(1, 1, CvType.CV_32F, new Scalar(0));
		assertFalse(m2.empty());
		assertMatEqual(m2, gray0_32f);
	}

	public void testMatLong() {
		fail("Not yet implemented");
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
		//TODO: assertEquals(CvType.CV_8U, gray0.type());
		assertEquals(CvType.CV_32FC1, gray0_32f.type());
		assertEquals(CvType.CV_8UC3, rgbLena.type());
	}

	public void testWidth() {
		assertEquals(gray0.cols(), gray0.width());
		assertEquals(rgbLena.cols(), rgbLena.width());
		assertEquals(rgba128.cols(), rgba128.width());
	}

}
