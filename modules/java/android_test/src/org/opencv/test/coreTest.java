package org.opencv.test;

import org.opencv.Mat;
import org.opencv.Point;
import org.opencv.Scalar;
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
	    core.LUT(grayRnd, lut, dst_gray);
	    assertMatEqual(gray0, dst_gray);
	    
	    lut.setTo(255);
	    core.LUT(grayRnd, lut, dst_gray);
	    assertMatEqual(gray255, dst_gray);
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
		core.absdiff(gray128, gray255, dst_gray);
		assertMatEqual(gray127, dst_gray);
	}

	public void testAddMatMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testAddMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testAddMatMatMat() {
		core.add(gray128, gray128, dst_gray);
		assertMatEqual(gray255, dst_gray);
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
		core.bitwise_and(gray3, gray2, dst_gray);
		assertMatEqual(gray2, dst_gray);
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
		boolean hasFeauture = core.checkHardwareSupport(0); //FIXME: do we need this function?
		assertEquals(false, hasFeauture);
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
		Point center = new Point(gray0.cols() / 2, gray0.rows()/2);
		int radius = Math.min(gray0.cols()/4, gray0.rows()/4);
		Scalar color = new Scalar(128);
		
		assertTrue(0 == core.countNonZero(gray0));
		core.circle(gray0, center, radius, color);
		assertTrue(0 != core.countNonZero(gray0));
	}

	public void testCompare() {
        Mat cmp = new Mat(0, 0, Mat.CvType.CV_8UC1);
        
        core.compare(gray0, gray0, cmp, core.CMP_EQ);
        assertMatEqual(cmp, gray255);
		
        core.compare(gray0, gray1, cmp, core.CMP_EQ);
        assertMatEqual(cmp, gray0);
	}

	public void testCompleteSymmMatBoolean() {
		core.completeSymm(grayRnd_32f, true);
		core.transpose(grayRnd_32f, dst_gray_32f);
		assertMatEqual(grayRnd_32f, dst_gray_32f);
	}

	public void testCompleteSymmMat() {
		core.completeSymm(grayRnd_32f);
		core.transpose(grayRnd_32f, dst_gray_32f);
		assertMatEqual(grayRnd_32f, dst_gray_32f);
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
		assertEquals(0, core.countNonZero(gray0));
		gray0.put(0, 0, 255);
		gray0.put(gray0.rows() - 1, gray0.cols() - 1, 255);		
		assertEquals(2, core.countNonZero(gray0));
	}

	public void testCubeRoot() {
		fail("Not yet implemented");
	}

	public void testDctMatMatInt() {
		fail("Not yet implemented");
	}

	public void testDctMatMat() {
		core.dct(gray0_32f_1d, dst_gray);
		assertMatEqual(gray0_32f_1d, dst_gray);
		
		Mat in = new Mat(1, 4, Mat.CvType.CV_32FC1);
		in.put(0, 0, 135.22211, 50.811096, 102.27016, 207.6682);
		Mat out = new Mat(1, 4, Mat.CvType.CV_32FC1);
		out.put(0, 0, 247.98576, -61.252407, 94.904533, 14.013477);
		core.dct(in, dst_gray);
		assertMatEqual(out, dst_gray);
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
		core.extractChannel(rgba128, dst_gray, 0);
		assertMatEqual(gray128, dst_gray);
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
		double freq = core.getTickFrequency();
		assertTrue(0.0 != freq);
	}

	public void testHconcat() {
		Mat e = new Mat(3, 3, Mat.CvType.CV_8UC1);
		Mat eConcat = new Mat(1, 9, Mat.CvType.CV_8UC1);
		e.put(0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1);
		eConcat.put(0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1);
		core.hconcat(e, dst_gray);
		
		assertMatEqual(eConcat, dst_gray);
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
		core.insertChannel(gray0, rgba128, 0);
		core.insertChannel(gray0, rgba128, 1);
		core.insertChannel(gray0, rgba128, 2);
		core.insertChannel(gray0, rgba128, 3);
		//assertMatEqual(rgba0, rgba128);
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
		int nPoints = Math.min(gray0.cols(), gray0.rows());
		
		Point point1 = new Point(0, 0);
		Point point2 = new Point(nPoints, nPoints);
		Scalar color = new Scalar(255);
		
		assertTrue(0 == core.countNonZero(gray0));
		core.line(gray0, point1, point2, color);
		assertTrue(nPoints == core.countNonZero(gray0));
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
		//TODO: complex math. See the dct function test.
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
		core.mulTransposed(grayE_32f, dst_gray_32f, true);
		assertMatEqual(grayE_32f, dst_gray_32f);
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
		double n = core.norm(gray127, core.NORM_INF);
		assertTrue(127 == n);
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
		//XXX: kirill stopped here
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