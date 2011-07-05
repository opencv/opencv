package org.opencv.test;

import org.opencv.Mat;

public class MatTest extends OpenCVTestCase {
	
	public void test_1() {
		super.test_1("Mat");
	}

	public void testChannels() {
		//fail("Not yet implemented");
		utils.Log(grayRnd.dump());
	}

	public void testClone() {
		fail("Not yet implemented");
	}

	public void testCol() {
		fail("Not yet implemented");
	}

	public void testColRange() {
		fail("Not yet implemented");
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
		fail("Not yet implemented");
	}

	public void testDispose() {
		fail("Not yet implemented");
	}

	public void testDot() {
		fail("Not yet implemented");
	}

	public void testElemSize() {
		fail("Not yet implemented");
	}

	public void testEmpty() {
		fail("Not yet implemented");
	}

	public void testFinalize() {
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
		fail("Not yet implemented");
	}

	public void testHeight() {
		fail("Not yet implemented");
	}

	public void testInv() {
		fail("Not yet implemented");
	}

	public void testIsContinuous() {
		fail("Not yet implemented");
	}

	public void testIsSubmatrix() {
		fail("Not yet implemented");
	}

	public void testMatIntIntCvType() {
		Mat gray = new Mat(1, 1, Mat.CvType.CV_8UC1);
		assertFalse(gray.empty());
		
		Mat rgb = new Mat(1, 1, Mat.CvType.CV_8UC3);
		assertFalse(rgb.empty());
	}

	public void testMatIntIntCvTypeDouble() {
		fail("Not yet implemented");
	}

	public void testMatIntIntCvTypeDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testMatIntIntCvTypeDoubleDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testMatIntIntCvTypeDoubleDoubleDoubleDouble() {
		fail("Not yet implemented");
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
		fail("Not yet implemented");
	}

	public void testRowRange() {
		fail("Not yet implemented");
	}

	public void testRows() {
		assertEquals(matSize, gray0.rows());
	}

	public void testSetToDouble() {
		fail("Not yet implemented");
	}

	public void testSetToDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testSetToDoubleDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testSetToDoubleDoubleDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testSubmat() {
		fail("Not yet implemented");
	}

	public void testToString() {
		fail("Not yet implemented");
	}

	public void testTotal() {
		fail("Not yet implemented");
	}

	public void testType() {
		fail("Not yet implemented");
	}

	public void testWidth() {
		fail("Not yet implemented");
	}
}
