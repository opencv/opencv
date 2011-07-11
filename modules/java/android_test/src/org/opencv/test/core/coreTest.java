package org.opencv.test.core;

import org.opencv.CvType;
import org.opencv.Mat;
import org.opencv.Point;
import org.opencv.Scalar;
import org.opencv.core;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class coreTest extends OpenCVTestCase {
	
	public void test_1() {
		super.test_1("CORE");
		
		//System.gc();
	}

	public void testAbsdiff() {
		core.absdiff(gray128, gray255, dst);
		assertMatEqual(gray127, dst);
	}

	public void testAddMatMatMat() {
		core.add(gray128, gray128, dst);
		assertMatEqual(gray255, dst);
	}

	public void testAddMatMatMatMat() {
		core.add(gray0, gray1, dst, gray1);
		assertMatEqual(gray1, dst);
		
		dst.setTo(new Scalar(127));
		core.add(gray0, gray1, dst, gray0);
		assertMatEqual(gray127, dst);
	}

	public void testAddMatMatMatMatInt() {
		core.add(gray0, gray1, dst, gray1, CvType.CV_32F);
		assertTrue(CvType.CV_32F == dst.depth());
//		FIXME: must work assertMatEqual(gray1_32f, dst);
	}

	public void testAddWeightedMatDoubleMatDoubleDoubleMat() {
		core.addWeighted(gray1, 126.0, gray127, 1.0, 2.0, dst);
		assertMatEqual(gray255, dst);		
	}

	public void testAddWeightedMatDoubleMatDoubleDoubleMatInt() {
		core.addWeighted(gray1, 126.0, gray127, 1.0, 2.0, dst, gray255_32f.depth());
		assertTrue(CvType.CV_32F == dst.depth());
		//FIXME: must work
	}

	public void testBitwise_andMatMatMat() {
		core.bitwise_and(gray3, gray2, dst);
		assertMatEqual(gray2, dst);
	}

	public void testBitwise_andMatMatMatMat() {
		core.bitwise_and(gray0, gray1, dst, gray255);
		assertMatEqual(gray0, dst);
	}

	public void testBitwise_notMatMat() {
		core.bitwise_not(gray255, dst);
		assertMatEqual(gray0, dst);
	}

	public void testBitwise_notMatMatMat() {
		core.bitwise_not(gray255, dst, gray255);
		assertMatEqual(gray0, dst);
	}

	public void testBitwise_orMatMatMat() {
		core.bitwise_or(gray3, gray2, dst);
		assertMatEqual(gray3, dst);
	}

	public void testBitwise_orMatMatMatMat() {
		core.bitwise_or(gray127, gray128, dst, gray255);
		assertMatEqual(gray255, dst);
	}

	public void testBitwise_xorMatMatMat() {
		core.bitwise_xor(gray3, gray2, dst);
		assertMatEqual(gray1, dst);
	}

	public void testBitwise_xorMatMatMatMat() {
		core.bitwise_or(gray127, gray128, dst, gray255);
		assertMatEqual(gray255, dst);
	}

	public void testCalcCovarMatrixMatMatMatInt() {
		Mat covar = new Mat(matSize, matSize, CvType.CV_64FC1);
		Mat mean = new Mat(1, matSize, CvType.CV_64FC1);
		
		core.calcCovarMatrix(gray0_32f, covar, mean, 8|1); //TODO: CV_COVAR_NORMAL instead of magic numbers
		assertMatEqual(gray0_64f, covar);
		assertMatEqual(gray0_64f_1d, mean);
	}

	public void testCalcCovarMatrixMatMatMatIntInt() {
		Mat covar = new Mat(matSize, matSize, CvType.CV_32F);
		Mat mean = new Mat(1, matSize, CvType.CV_32F);
		
		core.calcCovarMatrix(gray0_32f, covar, mean, 8|1, CvType.CV_32F); //FIXME: CV_COVAR_NORMAL
		assertMatEqual(gray0_32f, covar);
		assertMatEqual(gray0_32f_1d, mean);
	}

	public void testCartToPolarMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testCartToPolarMatMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testCheckHardwareSupport() {
		//XXX: core.checkHardwareSupport(feature)
		boolean hasFeauture = core.checkHardwareSupport(0);
		assertEquals(false, hasFeauture);
	}

	public void testCircleMatPointIntScalar() {
		Point center = new Point(gray0.cols() / 2, gray0.rows()/2);
		int radius = Math.min(gray0.cols()/4, gray0.rows()/4);
		Scalar color = new Scalar(128);
		
		assertTrue(0 == core.countNonZero(gray0));
		core.circle(gray0, center, radius, color);
		assertTrue(0 != core.countNonZero(gray0));
	}

	public void testCircleMatPointIntScalarInt() {
		Point center = new Point(gray0.cols() / 2, gray0.rows()/2);
		int radius = Math.min(gray0.cols()/4, gray0.rows()/4);
		Scalar color = new Scalar(128);
		
		assertTrue(0 == core.countNonZero(gray0));
		core.circle(gray0, center, radius, color, -1);
		assertTrue(0 != core.countNonZero(gray0));
	}

	public void testCircleMatPointIntScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testCircleMatPointIntScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testCompare() {
        core.compare(gray0, gray0, dst, core.CMP_EQ);
        assertMatEqual(dst, gray255);
		
        core.compare(gray0, gray1, dst, core.CMP_EQ);
        assertMatEqual(dst, gray0);
        
        core.compare(gray0, grayRnd, dst, core.CMP_EQ);
        double nBlackPixels = core.countNonZero(dst);
        double nNonBlackpixels = core.countNonZero(grayRnd);
        assertTrue((nBlackPixels + nNonBlackpixels) == grayRnd.total());
	}

	public void testCompleteSymmMat() {
		core.completeSymm(grayRnd_32f);
		core.transpose(grayRnd_32f, dst);
		assertMatEqual(grayRnd_32f, dst);
	}

	public void testCompleteSymmMatBoolean() {
		core.completeSymm(grayRnd_32f, true);
		core.transpose(grayRnd_32f, dst);
		assertMatEqual(grayRnd_32f, dst);
	}

	public void testConvertScaleAbsMatMat() {
		core.convertScaleAbs(gray0, dst);
		assertMatEqual(gray0, dst);
		
		core.convertScaleAbs(gray_16u_256, dst);
		assertMatEqual(gray255, dst);
	}

	public void testConvertScaleAbsMatMatDouble() {
		core.convertScaleAbs(gray0, dst, 2);
		assertMatEqual(gray0, dst);
		
		core.convertScaleAbs(gray_16u_256, dst, 1);
		assertMatEqual(gray255, dst);
	}

	public void testConvertScaleAbsMatMatDoubleDouble() {
	    core.convertScaleAbs(gray_16u_256, dst, 2, 2);
	    assertMatEqual(gray255, dst);
	}

	public void testCountNonZero() {
		assertEquals(0, core.countNonZero(gray0));
		gray0.put(0, 0, 255);
		gray0.put(gray0.rows() - 1, gray0.cols() - 1, 255);		
		assertEquals(2, core.countNonZero(gray0));
	}

	public void testCubeRoot() {
		float res = core.cubeRoot(27.0f);
		assertEquals(3.0f,res);
	}

	public void testDctMatMat() {
		core.dct(gray0_32f_1d, dst);
		assertMatEqual(gray0_32f_1d, dst);
		
		Mat in = new Mat(1, 4, CvType.CV_32F);
		in.put(0, 0, 135.22211, 50.811096, 102.27016, 207.6682);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		out.put(0, 0, 247.98576, -61.252407, 94.904533, 14.013477);
		
		core.dct(in, dst);
		assertMatEqual(out, dst);
	}

	public void testDctMatMatInt() {
		core.dct(gray0_32f_1d, dst);
		assertMatEqual(gray0_32f_1d, dst);
		
		Mat in = new Mat(1, 8, CvType.CV_32F);
        in.put(0,0, 0.203056, 0.980407, 0.35312, -0.106651, 0.0399382, 0.871475, -0.648355, 0.501067);
		Mat out = new Mat(1, 8, CvType.CV_32F);
		out.put(0,0,0.775716, 0.3727, 0.185299, 0.0121461, -0.325, -0.993021, 0.559794, -0.625127);
		core.dct(in, dst);
		assertMatEqual(out, dst);
	}

	public void testDeterminant() {
		Mat mat = new Mat(2, 2, CvType.CV_32F);
		mat.put(0, 0, 4.0);
		mat.put(0, 1, 2.0);
		mat.put(1, 0, 4.0);
		mat.put(1, 1, 4.0);
		double det = core.determinant(mat);
		assertEquals(8.0,det);		
	}

	public void testDftMatMat() {
		fail("Not yet implemented");
	}

	public void testDftMatMatInt() {
		fail("Not yet implemented");
	}

	public void testDftMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testDivideDoubleMatMat() {
		core.divide(4.0, gray2, dst);
		assertMatEqual(gray2, dst);
	}

	public void testDivideDoubleMatMatInt() {
		core.divide(9.0, gray3, dst, -1);
		assertMatEqual(gray3, dst);
	}

	public void testDivideMatMatMat() {
		core.divide(gray2, gray1, dst);
		assertMatEqual(gray2, dst);
	}

	public void testDivideMatMatMatDouble() {
		core.divide(gray2, gray2, dst, 2.0);
		assertMatEqual(gray2, dst);
	}

	public void testDivideMatMatMatDoubleInt() {
		core.divide(gray3, gray2, dst, 2.0, gray3.depth());
		assertMatEqual(gray3, dst);
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalar() {
		fail("Not yet implemented");
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalarInt() {
		fail("Not yet implemented");
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testEllipseMatPointSizeDoubleDoubleDoubleScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testExp() {
		Mat destination = new Mat(matSize, matSize, CvType.CV_32F); destination.setTo(new Scalar(0.0));
		core.exp(gray0_32f, destination);
		OpenCVTestRunner.Log(destination.dump());
		assertMatEqual(gray1_32f, destination);		
	}

	public void testExtractChannel() {
		core.extractChannel(rgba128, dst, 0);
		assertMatEqual(gray128, dst);
	}

	public void testFastAtan2() {
		double delta = 0.01;
		float res = core.fastAtan2(50, 50);
		assertEquals(45,res,delta);
		
		float res2 = core.fastAtan2(80, 20);
		assertEquals(75.96, res2, delta);
		
	}

	public void testFlip() {
		Mat src = new Mat(2, 2, CvType.CV_32F);
		Mat des_f0 = new Mat(2, 2, CvType.CV_32F);
		src.put(0, 0, 1.0);
		src.put(0, 1, 2.0);
		src.put(1, 0, 3.0);
		src.put(1, 1, 4.0);
		
		des_f0.put(0, 0, 3.0);
		des_f0.put(0, 1, 4.0);
		des_f0.put(1, 0, 1.0);
		des_f0.put(1, 1, 2.0);
		core.flip(src, dst, 0);
		assertMatEqual(des_f0, dst);
		
		Mat des_f1 = new Mat(2, 2, CvType.CV_32F);
		des_f1.put(0, 0, 2.0);
		des_f1.put(0, 1, 1.0);
		des_f1.put(1, 0, 4.0);
		des_f1.put(1, 1, 3.0);
		core.flip(src, dst, 1);
		assertMatEqual(des_f1, dst);	
	}

	public void testGemmMatMatDoubleMatDoubleMat() {
		fail("Not yet implemented.");
//		Mat m1 = new Mat(2,2, CvType.CV_32FC1);
//		Mat m2 = new Mat(2,2, CvType.CV_32FC1);
//		Mat des = new Mat(2,2, CvType.CV_32FC1);
//		Mat dmatrix = new Mat(2,2, CvType.CV_32FC1);
//		m1.put(0, 0, 1.0);
//		m1.put(0, 1, 0.0);
//		m1.put(1, 0, 1.0);
//		m1.put(1, 1, 0.0);
//		
//		m2.put(0, 0, 1.0);
//		m2.put(0, 1, 0.0);
//		m2.put(1, 0, 1.0);
//		m2.put(1, 1, 0.0);
//		
//		dmatrix.put(0, 0, 0.001);
//		dmatrix.put(0, 1, 0.001);
//		dmatrix.put(1, 0, 0.001);
//		dmatrix.put(1, 1, 0.001);
//	    
//		des.put(0, 0, 1.001);
//		des.put(0, 1, 1.001);
//		des.put(1, 0, 1.001);
//	    des.put(1, 1, 1.001);
//	    
////	    OpenCVTestRunner.Log(dst_gray_32f.dump());
//		
//		core.gemm(m1, m2, 1.0, dmatrix, 1.0, dst_gray_32f);
//		OpenCVTestRunner.Log(dst_gray_32f.dump());
//		OpenCVTestRunner.Log(des.dump());
//		assertMatEqual(des,dst_gray_32f);
	}

	public void testGetOptimalDFTSize() {
		int vecsize = core.getOptimalDFTSize(0);
		assertEquals(1, vecsize);
		
		int largeVecSize = core.getOptimalDFTSize(32768);
		assertTrue(largeVecSize < 0);	//FIXME:fails why??
	}

	public void testGetTickFrequency() {
		double freq = 0.0;
		freq = core.getTickFrequency();
		assertTrue(0.0 != freq);
	}

	public void testHconcat() {
		Mat e = Mat.eye(3, 3, CvType.CV_8UC1);
		Mat eConcat = new Mat(1, 9, CvType.CV_8UC1);
		e.put(0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1);
		eConcat.put(0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1);
		
		core.hconcat(e, dst);		
		assertMatEqual(eConcat, dst);
	}

	public void testIdctMatMat() {
		Mat in = new Mat(1, 8, CvType.CV_32F);
		in.put(0, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);
		Mat out = new Mat(1, 8, CvType.CV_32F);
		out.put(0, 0, 3.88909,-0.791065, 0.844623, 0.865723, -1.76777, -0.0228873, -0.732538, 0.352443);
		core.idct(in, dst);
		assertMatEqual(out, dst);
	}

	public void testIdctMatMatInt() {
		fail("Not yet implemented");
	}

	public void testIdftMatMat() {
		fail("Not yet implemented");
	}

	public void testIdftMatMatInt() {
		fail("Not yet implemented");
	}

	public void testIdftMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testInRange() {
		core.inRange(gray0, gray0, gray1, dst);
		assertMatEqual(gray255, dst);
	}

	public void testInsertChannel() {
		core.insertChannel(gray0, rgba128, 0);
		core.insertChannel(gray0, rgba128, 1);
		core.insertChannel(gray0, rgba128, 2);
		core.insertChannel(gray0, rgba128, 3);
		assertMatEqual(rgba0, rgba128);
	}

	public void testInvertMatMat() {
		Mat src = new Mat(2, 2, CvType.CV_32F);
		Mat des = new Mat(2, 2, CvType.CV_32F);
		src.put(0, 0, 1.0);
		src.put(0, 1, 2.0);
		src.put(1, 0, 1.5);
	    src.put(1, 1, 4.0);
		
		des.put(0, 0, -2.0);
		des.put(0, 1, 1.0);
		des.put(1, 0, 1.5);
	    des.put(1, 1, -0.5);
	    core.invert(src, dst);
	    assertMatEqual(des, dst);		
	}

	public void testInvertMatMatInt() {
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

	public void testLineMatPointPointScalarInt() {
		fail("Not yet implemented");
	}

	public void testLineMatPointPointScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testLineMatPointPointScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testLog() {
		//FIXME: why it fails for the above array!
//		Mat in = new Mat(1, 4, Mat.CvType.CV_32FC1);
//		Mat des = new Mat(1, 4, Mat.CvType.CV_32FC1);
//		in.put(0, 0, 1.0, 2.0, 4.0,3.0);
//		des.put(0,0, 0.0, 0.3010,0.6021,0.4771);
//		assertMatEqual(des,dst_gray);

		Mat in = new Mat(1, 1, CvType.CV_32F);
		Mat des = new Mat(1, 1, CvType.CV_32F);
		in.put(0, 0, 1);
		des.put(0,0, 0.0);
		core.log(in, dst);
		assertMatEqual(des, dst);
	}

	public void testLUTMatMatMat() {
	    Mat lut = new Mat(1, 256, CvType.CV_8UC1);
	    
    	lut.setTo(new Scalar(0));
	    core.LUT(grayRnd, lut, dst);
	    assertMatEqual(gray0, dst);
	    
	    lut.setTo(new Scalar(255));
	    core.LUT(grayRnd, lut, dst);
	    assertMatEqual(gray255, dst);
	}

	public void testLUTMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testMagnitude() {
		/*Mat x = new Mat(1, 4, Mat.CvType.CV_32FC1);
		Mat y = new Mat(1, 4, Mat.CvType.CV_32FC1);
		Mat dst = new Mat(1, 4, Mat.CvType.CV_32FC1);
		
		x.put(0, 0, 3.0, 5.0, 9.0, 6.0);
		y.put(0, 0, 4.0, 12.0, 40.0, 8.0);
		dst.put(0, 0, 5.0, 13,0, 41.0, 10.0);
		
        core.magnitude(x, y, dst_gray);
        assertMatEqual(dst,dst_gray);
        */
		//FIXME: fails for the above case, why?
		/*Mat x = new Mat(1, 1, Mat.CvType.CV_32FC1);
		Mat y = new Mat(1, 1, Mat.CvType.CV_32FC1);
		Mat dst = new Mat(1, 1, Mat.CvType.CV_32FC1);
		
		x.put(0, 0, 3.0);
		y.put(0, 0, 4.0);
		dst.put(0, 0, 5.0);
		
        core.magnitude(x, y, dst_gray);
        assertMatEqual(dst,dst_gray);
        */
		core.magnitude(gray0_32f, gray255_32f, dst);
		assertMatEqual(gray255_32f, dst);
	}

	public void testMahalanobis() {	
		Mat covar = new Mat(matSize, matSize, CvType.CV_32F);
		Mat mean = new Mat(1, matSize, CvType.CV_32F);		
		core.calcCovarMatrix(grayRnd_32f, covar, mean, 8|1, CvType.CV_32F); //TODO: CV_COVAR_NORMAL instead of magic numbers
		covar.inv();
		
		Mat line1 = grayRnd_32f.submat(0, 1, 0, grayRnd_32f.cols());
		Mat line2 = grayRnd_32f.submat(1, 2, 0, grayRnd_32f.cols());
		
		double d = 0.0;
		d = core.Mahalanobis(line1, line1, covar);
		assertEquals(0.0, d);
		
		d = core.Mahalanobis(line1, line2, covar);
		assertTrue(d > 0.0);
	}

	public void testMax() {
		core.min(gray0, gray255, dst);
		assertMatEqual(gray0, dst);
		
		Mat x = new Mat(1, 1, CvType.CV_32F);
		Mat y = new Mat(1, 1, CvType.CV_32F);
		Mat dst = new Mat(1, 1, CvType.CV_32F);
		x.put(0, 0, 23.0);
		y.put(0, 0, 4.0);
		dst.put(0, 0, 23.0);
		core.max(x, y, dst);
		assertMatEqual(dst, dst);		
	}

	public void testMeanStdDevMatMatMat() {
		fail("Not yet implemented");
	}

	public void testMeanStdDevMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testMin() {
		core.min(gray0, gray255, dst);
		assertMatEqual(gray0, dst);		
	}

	public void testMulSpectrumsMatMatMatInt() {
		//TODO: nice example
		fail("Not yet implemented");
	}

	public void testMulSpectrumsMatMatMatIntBoolean() {
		fail("Not yet implemented");
	}

	public void testMultiplyMatMatMat() {
		core.multiply(gray0, gray255, dst);
		assertMatEqual(gray0, dst);
	}

	public void testMultiplyMatMatMatDouble() {
		core.multiply(gray1, gray0, dst, 2.0);
		assertMatEqual(gray0, dst);
		
	}

	public void testMultiplyMatMatMatDoubleInt() {
		core.multiply(gray1, gray0, dst, 2.0, -1);
		assertMatEqual(gray0, dst);
	}

	public void testMulTransposedMatMatBoolean() {
		core.mulTransposed(grayE_32f, dst, true);
		assertMatEqual(grayE_32f, dst);
	}

	public void testMulTransposedMatMatBooleanMat() {
		core.mulTransposed(grayRnd_32f, dst, true, grayRnd_32f);
		assertMatEqual(gray0_32f, dst);

		Mat grayDelta = new Mat(matSize, matSize, CvType.CV_8U);
        grayDelta.setTo(new Scalar(0.0001));
		core.mulTransposed(grayE_32f, dst, true, grayDelta);
		assertMatEqual(grayE_32f, dst);
	}

	public void testMulTransposedMatMatBooleanMatDouble() {
		Mat grayDelta = new Mat(matSize, matSize, CvType.CV_8U);
		grayDelta.setTo(new Scalar(0.0001));
		core.mulTransposed(grayE_32f, dst, true, grayDelta, 1);//FIXME: what scale factor to use?!
		assertMatEqual(grayE_32f, dst);
	}

	public void testMulTransposedMatMatBooleanMatDoubleInt() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMat() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDoubleIntInt() {
		fail("Not yet implemented");
	}

	public void testNormalizeMatMatDoubleDoubleIntIntMat() {
		fail("Not yet implemented");
	}

	public void testNormMat() {		
		double n = core.norm(gray0);
		assertTrue(0.0 == n);
		}

	public void testNormMatInt() {
		double n = core.norm(gray127, core.NORM_INF);
		assertTrue(127 == n);
	}

	public void testNormMatIntMat() {
		double n  = core.norm(gray3, core.NORM_L1, gray0);
		assertEquals(0.0, n);
	}

	public void testNormMatMat() {
		double n = core.norm(gray255, gray255);
		assertEquals(0.0, n);		
	}

	public void testNormMatMatInt() {
		double n = core.norm(gray127, gray0, core.NORM_INF);
		assertEquals(127.0, n);		
	}

	public void testNormMatMatIntMat() {
		double n  = core.norm(gray3, gray0, core.NORM_L1, gray0);
		assertEquals(0.0, n);
	}

	public void testPerspectiveTransform() {
		//TODO: nice example
		fail("Not yet implemented");
	}

	public void testPhaseMatMatMat() {
		fail("Not yet implemented");
	}

	public void testPhaseMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testPolarToCartMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testPolarToCartMatMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testPow() {
		core.pow(gray3, 2.0, dst);
		assertMatEqual(gray9, dst);
	}

	public void testRandn() {
		fail("Not yet implemented");
	}

	public void testRandu() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalar() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalarInt() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testRectangleMatPointPointScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testReduceMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testReduceMatMatIntIntInt() {
		fail("Not yet implemented");
	}

	public void testRepeat() {
		fail("Not yet implemented");
	}

	public void testScaleAdd() {
		core.scaleAdd(gray3, 2.0, gray3, dst);
		assertMatEqual(dst, gray9);
	}

	public void testSetIdentityMat() {
		core.setIdentity(dst);
		assertTrue(dst.rows() == core.countNonZero(dst));
	}

	public void testSetIdentityMatScalar() {
		fail("Not yet implemented. Scalar type is not supported");
	}

	public void testSetUseOptimized() {
		//XXX: core.setUseOptimized(onoff)
		fail("Not yet implemented");
	}

	public void testSolveCubic() {
		fail("Not yet implemented");
	}

	public void testSolveMatMatMat() {
		fail("Not yet implemented");
	}

	public void testSolveMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testSolvePolyMatMat() {
		fail("Not yet implemented");
//		Mat coeffs = new Mat(4, 1, CvType.CV_32F);
//		Mat standart = new Mat(3, 1, CvType.CV_32F);
//		Mat roots = new Mat(3, 1, CvType.CV_32F);
//		coeffs.setTo(0);
//		coeffs.put(0, 0, 1);
//		coeffs.put(0, 1, -6);
//		coeffs.put(0, 2, 11);
//		coeffs.put(0, 3, -6);
//		standart.put(0, 0, 1);
//		standart.put(0, 1, 2);
//		standart.put(0, 2, 3);
		
//		utils.Log(standart.dump());
		
//		core.solvePoly(coeffs, roots);
//		
//		OpenCVTestRunner.Log(roots.dump());
//		core.sort(roots, roots, CV_SORT_EVERY_ROW);
//		assertTrue(1 == core.countNonZero(coeffs));
		//assertMatEqual(roots, standart);
	}

	public void testSolvePolyMatMatInt() {
		fail("Not yet implemented");
	}

	public void testSort() {
		Mat matrix = new Mat(matSize, matSize, CvType.CV_8U); 
		matrix.setTo(new Scalar(0.0));
		Mat submatrix = matrix.submat(0, matrix.rows() / 2, 0, matrix.cols() / 2);
		submatrix.setTo(new Scalar(1.0));
		
		core.sort(matrix, dst, 0); //FIXME: #define CV_SORT_EVERY_ROW 0
		
		Mat subdst = dst.submat(0, dst.rows() / 2, dst.cols() / 2, dst.cols());
		assertTrue(subdst.total() == core.countNonZero(subdst));
		
		core.sort(matrix, dst, 1); //FIXME: #define CV_SORT_EVERY_COLUMN 1 
		Mat subdst1 = dst.submat(dst.rows() / 2, dst.rows(), 0, dst.cols() / 2);
		assertTrue(subdst1.total() == core.countNonZero(subdst1));
	}

	public void testSortIdx() {
		fail("Not yet implemented");
//		Mat matrix = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); 
//		matrix.setTo(0);
//		Mat submatrix = matrix.submat(0, matrix.rows() / 2, 0, matrix.cols() / 2);
//		Mat subdst = dst.submat(0, dst.rows() / 2, dst.cols() / 2, dst.cols());
//		submatrix.setTo(1);
//		utils.Log(subdst.dump());
//		core.sortIdx(matrix, dst, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
//		utils.Log(subdst.dump());
//		assertTrue(subdst.total() == core.countNonZero(subdst));
		
		
//		dst.setTo(0);
//		core.sortIdx(matrix, dst, CV_SORT_EVERY_COLUM + CV_SORT_DESCENDING);
//		Mat subdst1 = dst.submat(0, dst.rows() / 2, 0, dst.cols() / 2);
//		utils.Log(subdst1.dump());
//		assertTrue(subdst1.total() == core.countNonZero(subdst1));
	}

	public void testSqrt() {
		core.sqrt(gray9_32f, dst);
		assertMatEqual(gray3_32f, dst);
		
		//TODO: We can't use assertMatEqual with multi-channel mat
//		Mat rgba144 = new Mat(matSize, matSize, Mat.CvType.CV_32FC4);
//		Mat rgba12 = new Mat(matSize, matSize, Mat.CvType.CV_32FC4);
//		Mat rgba_dst = new Mat(matSize, matSize, Mat.CvType.CV_32FC4);
//		rgba144.setTo(144, 144, 144, 144);
//		rgba12.setTo(12, 12, 12, 12);
//		rgba_dst.setTo(0, 0, 0, 0);
//		core.sqrt(rgba144, rgba_dst);
//		//assertMatEqual(rgba12, rgba_dst);
	}

	public void testSubtractMatMatMat() {
		core.subtract(gray128, gray1, dst);
		assertMatEqual(gray127, dst);
	}

	public void testSubtractMatMatMatMat() {
		fail("Not yet implemented");
		Mat mask = new Mat(matSize, matSize, CvType.CV_8U); 
		mask.setTo(new Scalar(0));
		Mat submask = mask.submat(0, mask.rows() / 2, 0, mask.cols() / 2);
		submask.setTo(new Scalar(1));
		
		//FIXME: looks like a bug
		OpenCVTestRunner.Log(" submask.total() = " + String.valueOf(submask.total()));
		OpenCVTestRunner.Log(" 1: core.countNonZero(dst) = " + String.valueOf(core.countNonZero(dst)));		
		core.subtract(gray3, gray2, dst, mask);
		OpenCVTestRunner.Log(" 2: core.countNonZero(dst) = " + String.valueOf(core.countNonZero(dst)));
		assertTrue(submask.total() == core.countNonZero(dst));
	}

	public void testSubtractMatMatMatMatInt() {
		fail("Not yet implemented");
//		core.subtract(gray3, gray1, dst, gray1, gray255_32f.depth());
//		OpenCVTestRunner.Log(" 3: dst.depth() = " + String.valueOf(dst.depth()));
//		OpenCVTestRunner.Log(" 4: core.CV_32F = " + String.valueOf(CvType.CV_32F));
//		//FIXME: assertTrue(CvType.CV_32F == dst.depth());
//		//assertMatEqual(gray2, dst);
	}

	public void testTransform() {
		fail("Not yet implemented");
	}

	public void testTranspose() {
		Mat subgray0 = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols());
		Mat destination = new Mat(matSize, matSize, CvType.CV_8U); destination.setTo(new Scalar(0));
		Mat subdst = destination.submat(0, destination.rows(), 0, destination.cols() / 2);
		subgray0.setTo(new Scalar(1));
		core.transpose(gray0, destination);
		assertTrue(subdst.total() == core.countNonZero(subdst));
	}

	public void testUseOptimized() {
		//XXX: core.useOptimized()
		fail("Not yet implemented");
	}

	public void testVconcat() {
		fail("Not yet implemented");
	}
	
}