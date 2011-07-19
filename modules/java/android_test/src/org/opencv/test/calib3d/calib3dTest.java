package org.opencv.test.calib3d;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.test.OpenCVTestCase;

public class calib3dTest extends OpenCVTestCase {
	
	public void test_1() {
		super.test_1("calib3d");
	}

	public void testComposeRTMatMatMatMatMatMat() {
		Mat rvec1 = new Mat(3, 1, CvType.CV_32F); rvec1.put(0, 0, 0.5302828,  0.19925919, 0.40105945);
		Mat tvec1 = new Mat(3, 1, CvType.CV_32F); tvec1.put(0, 0, 0.81438506, 0.43713298, 0.2487897);
		Mat rvec2 = new Mat(3, 1, CvType.CV_32F); rvec2.put(0, 0, 0.77310503, 0.76209372, 0.30779448);
		Mat tvec2 = new Mat(3, 1, CvType.CV_32F); tvec2.put(0, 0, 0.70243168, 0.4784472,  0.79219002);
			
		Mat rvec3 = new Mat();
		Mat tvec3 = new Mat();
		
		Mat outRvec = new Mat(3, 1, CvType.CV_32F); outRvec.put(0, 0, 1.418641, 0.88665926, 0.56020796);
		Mat outTvec = new Mat(3, 1, CvType.CV_32F); outTvec.put(0, 0, 1.4560841, 1.0680628, 0.81598103);

		Calib3d.composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3);
		
		assertMatEqual(outRvec, rvec3);
		assertMatEqual(outTvec, tvec3);
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
//	    Mat dr3dr1;
//	    Mat dr3dt1;
//	    Mat dr3dr2;
//	    Mat dr3dt2;
//	    Mat dt3dr1;
//	    Mat dt3dt1;
//	    Mat dt3dr2;
//	    Mat dt3dt2;
		//, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2);
//		[0.97031879, -0.091774099, 0.38594806;
//		  0.15181915, 0.98091727, -0.44186208;
//		  -0.39509675, 0.43839464, 0.93872648]
//		[0, 0, 0;
//		  0, 0, 0;
//		  0, 0, 0]
//		[1.0117353, 0.16348237, -0.083180845;
//		  -0.1980398, 1.006078, 0.30299222;
//		  0.075766489, -0.32784501, 1.0163091]
//		[0, 0, 0;
//		  0, 0, 0;
//		  0, 0, 0]
//		[0, 0, 0;
//		  0, 0, 0;
//		  0, 0, 0]
//		[0.69658804, 0.018115902, 0.7172426;
//		  0.51114357, 0.68899536, -0.51382649;
//		  -0.50348526, 0.72453934, 0.47068608]
//		[0.18536358, -0.20515044, -0.48834875;
//		  -0.25120571, 0.29043972, 0.60573936;
//		  0.35370794, -0.69923931, 0.45781645]
//		[1, 0, 0;
//		  0, 1, 0;
//		  0, 0, 1]
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
		Core.circle(dst, center, 1, Scalar.all(4096));
		
		assertMatNotEqual(gray_16s_1024, dst);
	    Calib3d.filterSpeckles(dst, 1024.0, 100, 0.);
	    assertMatEqual(gray_16s_1024, dst);
	}

	public void testFilterSpecklesMatDoubleIntDoubleMat() {
		fail("Not yet implemented");
	}

	public void testFindChessboardCornersMatSizeMat() {
		Size patternSize = new Size(9, 6);
		Calib3d.findChessboardCorners(grayChess, patternSize, dst);
		assertTrue(!dst.empty());
	}

	public void testFindChessboardCornersMatSizeMatInt() {
		Size patternSize = new Size(9, 6);
		Calib3d.findChessboardCorners(grayChess, patternSize, dst, 
			Calib3d.CALIB_CB_ADAPTIVE_THRESH + Calib3d.CALIB_CB_NORMALIZE_IMAGE + Calib3d.CALIB_CB_FAST_CHECK);
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
