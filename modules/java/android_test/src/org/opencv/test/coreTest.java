package org.opencv.test;

import org.opencv.Mat;
import org.opencv.core;

public class coreTest extends OpenCVTestCase {
	
	public void test_1() {
		super.test_1("CORE");
	}

	public void testLUTMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testLUTMatMatMat() {
	    Mat lut = new Mat(1, 256, Mat.CvType.CV_8UC1);
	    
	    lut.setTo(0);
	    core.LUT(grayRnd, lut, dst);
	    assertMatEqual(gray0, dst);
	    
	    lut.setTo(255);
	    core.LUT(grayRnd, lut, dst);
	    assertMatEqual(gray255, dst);
	}

	public void testMahalanobis() {	
		Mat covar = new Mat(matSize, matSize, Mat.CvType.CV_32FC1);
		Mat mean = new Mat(1, matSize, Mat.CvType.CV_32FC1);		
		core.calcCovarMatrix(grayRnd_32f, covar, mean, 8|1, Mat.CvType.CV_32F); //FIXME: CV_COVAR_NORMAL
		covar.inv();
		
		Mat line1 = grayRnd_32f.submat(0, 1, 0, grayRnd_32f.cols());
		Mat line2 = grayRnd_32f.submat(1, 2, 0, grayRnd_32f.cols());
		
		double d = 0.0;
		d = core.Mahalanobis(line1, line1, covar);
		assertEquals(0.0, d);
		
		d = core.Mahalanobis(line1, line2, covar);
		assertTrue(d > 0.0);
	}

	public void testAbsdiff() {
		core.absdiff(gray128, gray255, dst);
		assertMatEqual(gray127, dst);
	}

	public void testAddMatMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testAddMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testAddMatMatMat() {
		core.add(gray128, gray128, dst);
		assertMatEqual(gray255, dst);
	}

	public void testAddWeightedMatDoubleMatDoubleDoubleMatInt() {
		fail("Not yet implemented");
	}

	public void testAddWeightedMatDoubleMatDoubleDoubleMat() {
		fail("Not yet implemented");
	}

	public void testBitwise_andMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testBitwise_andMatMatMat() {
		core.bitwise_and(gray3, gray2, dst);
		assertMatEqual(gray2, dst);
	}

	public void testBitwise_notMatMatMat() {
		fail("Not yet implemented");
	}

	public void testBitwise_notMatMat() {
		fail("Not yet implemented");
	}

	public void testBitwise_orMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testBitwise_orMatMatMat() {
		fail("Not yet implemented");
	}

	public void testBitwise_xorMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testBitwise_xorMatMatMat() {
		fail("Not yet implemented");
	}

	public void testCalcCovarMatrixMatMatMatIntInt() {
		Mat covar = new Mat(matSize, matSize, Mat.CvType.CV_32FC1);
		Mat mean = new Mat(1, matSize, Mat.CvType.CV_32FC1);
		
		core.calcCovarMatrix(gray0_32f, covar, mean, 8|1, Mat.CvType.CV_32F); //FIXME: CV_COVAR_NORMAL
		assertMatEqual(gray0_32f, covar);
		assertMatEqual(gray0_32f_1d, mean);
	}

	public void testCalcCovarMatrixMatMatMatInt() {
		Mat covar = new Mat(matSize, matSize, Mat.CvType.CV_64FC1);
		Mat mean = new Mat(1, matSize, Mat.CvType.CV_64FC1);
		
		core.calcCovarMatrix(gray0_32f, covar, mean, 8|1); //FIXME: CV_COVAR_NORMAL
		assertMatEqual(gray0_64f, covar);
		assertMatEqual(gray0_64f_1d, mean);
	}

	public void testCartToPolarMatMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testCartToPolarMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testCheckHardwareSupport() {
		fail("Not yet implemented");
	}

	public void testCircleMatPointIntScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testCircleMatPointIntScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testCircleMatPointIntScalarInt() {
		fail("Not yet implemented");
	}

	public void testCircleMatPointIntScalar() {
		fail("Not yet implemented");
	}

	public void testCompare() {
		fail("Not yet implemented");
	}

	public void testCompleteSymmMatBoolean() {
		fail("Not yet implemented");
	}

	public void testCompleteSymmMat() {
		fail("Not yet implemented");
	}

	public void testConvertScaleAbsMatMatDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testConvertScaleAbsMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testConvertScaleAbsMatMat() {
		fail("Not yet implemented");
	}

	public void testCountNonZero() {
		fail("Not yet implemented");
	}

	public void testCubeRoot() {
		fail("Not yet implemented");
	}

	public void testDctMatMatInt() {
		fail("Not yet implemented");
	}

	public void testDctMatMat() {
		fail("Not yet implemented");
	}

	public void testDeterminant() {
		fail("Not yet implemented");
	}

	public void testDftMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testDftMatMatInt() {
		fail("Not yet implemented");
	}

	public void testDftMatMat() {
		fail("Not yet implemented");
	}

	public void testDivideMatMatMatDoubleInt() {
		fail("Not yet implemented");
	}

	public void testDivideMatMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testDivideMatMatMat() {
		fail("Not yet implemented");
	}

	public void testDivideDoubleMatMatInt() {
		fail("Not yet implemented");
	}

	public void testDivideDoubleMatMat() {
		fail("Not yet implemented");
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalarInt() {
		fail("Not yet implemented");
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalar() {
		fail("Not yet implemented");
	}

	public void testExp() {
		fail("Not yet implemented");
	}

	public void testExtractChannel() {
		fail("Not yet implemented");
	}

	public void testFastAtan2() {
		fail("Not yet implemented");
	}

	public void testFlip() {
		fail("Not yet implemented");
	}

	public void testGemmMatMatDoubleMatDoubleMatInt() {
		fail("Not yet implemented");
	}

	public void testGemmMatMatDoubleMatDoubleMat() {
		fail("Not yet implemented");
	}

	public void testGetOptimalDFTSize() {
		fail("Not yet implemented");
	}

	public void testGetTickFrequency() {
		fail("Not yet implemented");
	}

	public void testHconcat() {
		fail("Not yet implemented");
	}

	public void testIdctMatMatInt() {
		fail("Not yet implemented");
	}

	public void testIdctMatMat() {
		fail("Not yet implemented");
	}

	public void testIdftMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testIdftMatMatInt() {
		fail("Not yet implemented");
	}

	public void testIdftMatMat() {
		fail("Not yet implemented");
	}

	public void testInRange() {
		fail("Not yet implemented");
	}

	public void testInsertChannel() {
		fail("Not yet implemented");
	}

	public void testInvertMatMatInt() {
		fail("Not yet implemented");
	}

	public void testInvertMatMat() {
		fail("Not yet implemented");
	}

	public void testLineMatPointPointScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testLineMatPointPointScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testLineMatPointPointScalarInt() {
		fail("Not yet implemented");
	}

	public void testLineMatPointPointScalar() {
		fail("Not yet implemented");
	}

	public void testLog() {
		fail("Not yet implemented");
	}

	public void testMagnitude() {
		fail("Not yet implemented");
	}

	public void testMax() {
		fail("Not yet implemented");
	}

	public void testMeanStdDevMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testMeanStdDevMatMatMat() {
		fail("Not yet implemented");
	}

	public void testMin() {
		fail("Not yet implemented");
	}

	public void testMulSpectrumsMatMatMatIntBoolean() {
		fail("Not yet implemented");
	}

	public void testMulSpectrumsMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testMulTransposedMatMatBooleanMatDoubleInt() {
		fail("Not yet implemented");
	}

	public void testMulTransposedMatMatBooleanMatDouble() {
		fail("Not yet implemented");
	}

	public void testMulTransposedMatMatBooleanMat() {
		fail("Not yet implemented");
	}

	public void testMulTransposedMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testMultiplyMatMatMatDoubleInt() {
		fail("Not yet implemented");
	}

	public void testMultiplyMatMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testMultiplyMatMatMat() {
		fail("Not yet implemented");
	}

	public void testNormMatIntMat() {
		fail("Not yet implemented");
	}

	public void testNormMatInt() {
		fail("Not yet implemented");
	}

	public void testNormMat() {
		fail("Not yet implemented");
	}

	public void testNormMatMatIntMat() {
		fail("Not yet implemented");
	}

	public void testNormMatMatInt() {
		fail("Not yet implemented");
	}

	public void testNormMatMat() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDoubleIntIntMat() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDoubleIntInt() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMat() {
		fail("Not yet implemented");
	}

	public void testPerspectiveTransform() {
		fail("Not yet implemented");
	}

	public void testPhaseMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testPhaseMatMatMat() {
		fail("Not yet implemented");
	}

	public void testPolarToCartMatMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testPolarToCartMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testPow() {
		fail("Not yet implemented");
	}

	public void testRandn() {
		fail("Not yet implemented");
	}

	public void testRandu() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalarInt() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalar() {
		fail("Not yet implemented");
	}

	public void testReduceMatMatIntIntInt() {
		fail("Not yet implemented");
	}

	public void testReduceMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testRepeat() {
		fail("Not yet implemented");
	}

	public void testScaleAdd() {
		fail("Not yet implemented");
	}

	public void testSetIdentityMatScalar() {
		fail("Not yet implemented");
	}

	public void testSetIdentityMat() {
		fail("Not yet implemented");
	}

	public void testSetUseOptimized() {
		fail("Not yet implemented");
	}

	public void testSolveMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testSolveMatMatMat() {
		fail("Not yet implemented");
	}

	public void testSolveCubic() {
		fail("Not yet implemented");
	}

	public void testSolvePolyMatMatInt() {
		fail("Not yet implemented");
	}

	public void testSolvePolyMatMat() {
		fail("Not yet implemented");
	}

	public void testSort() {
		fail("Not yet implemented");
	}

	public void testSortIdx() {
		fail("Not yet implemented");
	}

	public void testSqrt() {
		fail("Not yet implemented");
	}

	public void testSubtractMatMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testSubtractMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testSubtractMatMatMat() {
		fail("Not yet implemented");
	}

	public void testTransform() {
		fail("Not yet implemented");
	}

	public void testTranspose() {
		fail("Not yet implemented");
	}

	public void testUseOptimized() {
		fail("Not yet implemented");
	}

	public void testVconcat() {
		fail("Not yet implemented");
	}
}