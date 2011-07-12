package org.opencv.test.calib3d;

import org.opencv.Point;
import org.opencv.Scalar;
import org.opencv.Size;
import org.opencv.calib3d;
import org.opencv.core;
import org.opencv.test.OpenCVTestCase;

public class calib3dTest extends OpenCVTestCase {

	public void testComposeRTMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testComposeRTMatMatMatMatMatMatMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testConvertPointsFromHomogeneous() {
		fail("Not yet implemented");
	}

	public void testConvertPointsToHomogeneous() {
		fail("Not yet implemented");
	}

	public void testDecomposeProjectionMatrixMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testDecomposeProjectionMatrixMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testDecomposeProjectionMatrixMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testDecomposeProjectionMatrixMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testDecomposeProjectionMatrixMatMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testDrawChessboardCorners() {
		fail("Not yet implemented");
	}

	public void testEstimateAffine3DMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testEstimateAffine3DMatMatMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testEstimateAffine3DMatMatMatMatDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testFilterSpecklesMatDoubleIntDouble() {
		gray_16s_1024.copyTo(dst);
	    
		Point center = new Point(gray_16s_1024.rows()/2., gray_16s_1024.cols()/2.);
		core.circle(dst, center, 1, Scalar.all(4096));
		assertMatNotEqual(gray_16s_1024, dst);
	    calib3d.filterSpeckles(dst, 1024.0, 100, 0.);
	    assertMatEqual(gray_16s_1024, dst);
	}

	public void testFilterSpecklesMatDoubleIntDoubleMat() {
		fail("Not yet implemented");
	}

	public void testFindChessboardCornersMatSizeMat() {
		Size patternSize = new Size(9, 6);
		calib3d.findChessboardCorners(grayChess, patternSize, dst);
		assertTrue(!dst.empty());
	}

	public void testFindChessboardCornersMatSizeMatInt() {
		Size patternSize = new Size(9, 6);
		calib3d.findChessboardCorners(grayChess, patternSize, dst, calib3d.CALIB_CB_ADAPTIVE_THRESH 
				+ calib3d.CALIB_CB_NORMALIZE_IMAGE + calib3d.CALIB_CB_FAST_CHECK);
		assertTrue(!dst.empty());
	}

	public void testFindFundamentalMatMatMat() {
		fail("Not yet implemented");
	}

	public void testFindFundamentalMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testFindFundamentalMatMatMatIntDouble() {
		fail("Not yet implemented");
	}

	public void testFindFundamentalMatMatMatIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testFindFundamentalMatMatMatIntDoubleDoubleMat() {
		fail("Not yet implemented");
	}

	public void testFindHomographyMatMat() {
		fail("Not yet implemented");
	}

	public void testFindHomographyMatMatInt() {
		fail("Not yet implemented");
	}

	public void testFindHomographyMatMatIntDouble() {
		fail("Not yet implemented");
	}

	public void testFindHomographyMatMatIntDoubleMat() {
		fail("Not yet implemented");
	}

	public void testMatMulDeriv() {
		fail("Not yet implemented");
	}

	public void testProjectPointsMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testProjectPointsMatMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testProjectPointsMatMatMatMatMatMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testReprojectImageTo3DMatMatMat() {
		fail("Not yet implemented");
	}

	public void testReprojectImageTo3DMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testReprojectImageTo3DMatMatMatBooleanInt() {
		fail("Not yet implemented");
	}

	public void testRodriguesMatMat() {
		fail("Not yet implemented");
	}

	public void testRodriguesMatMatMat() {
		fail("Not yet implemented");
	}

	public void testSolvePnPMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testSolvePnPMatMatMatMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testSolvePnPRansacMatMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testSolvePnPRansacMatMatMatMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testSolvePnPRansacMatMatMatMatMatMatBooleanInt() {
		fail("Not yet implemented");
	}

	public void testSolvePnPRansacMatMatMatMatMatMatBooleanIntFloat() {
		fail("Not yet implemented");
	}

	public void testSolvePnPRansacMatMatMatMatMatMatBooleanIntFloatInt() {
		fail("Not yet implemented");
	}

	public void testSolvePnPRansacMatMatMatMatMatMatBooleanIntFloatIntMat() {
		fail("Not yet implemented");
	}

	public void testStereoRectifyUncalibratedMatMatMatSizeMatMat() {
		fail("Not yet implemented");
	}

	public void testStereoRectifyUncalibratedMatMatMatSizeMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testValidateDisparityMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testValidateDisparityMatMatIntIntInt() {
		fail("Not yet implemented");
	}

}
