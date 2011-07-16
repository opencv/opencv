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
		assertMatEqual(gray1_32f, dst);
	}

	public void testAddWeightedMatDoubleMatDoubleDoubleMat() {
		core.addWeighted(gray1, 126.0, gray127, 1.0, 2.0, dst);
		assertMatEqual(gray255, dst);		
	}

	public void testAddWeightedMatDoubleMatDoubleDoubleMatInt() {
		core.addWeighted(gray1, 126.0, gray127, 1.0, 2.0, dst, CvType.CV_32F);
		assertTrue(CvType.CV_32F == dst.depth());
		assertMatEqual(gray255_32f, dst);
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
		
		core.calcCovarMatrix(gray0_32f, covar, mean, 8|1/*TODO: CV_COVAR_NORMAL*/); 
		assertMatEqual(gray0_64f, covar);
		assertMatEqual(gray0_64f_1d, mean);
	}

	public void testCalcCovarMatrixMatMatMatIntInt() {
		Mat covar = new Mat(matSize, matSize, CvType.CV_32F);
		Mat mean = new Mat(1, matSize, CvType.CV_32F);
		
		core.calcCovarMatrix(gray0_32f, covar, mean, 8|1/*TODO: CV_COVAR_NORMAL*/, CvType.CV_32F);
		assertMatEqual(gray0_32f, covar);
		assertMatEqual(gray0_32f_1d, mean);
	}

	public void testCartToPolarMatMatMatMat() {
		Mat x = new Mat(1, 3, CvType.CV_32F);
		Mat y = new Mat(1, 3, CvType.CV_32F);
		Mat magnitude = new Mat(1, 3, CvType.CV_32F);
		Mat angle = new Mat(1, 3, CvType.CV_32F);
		Mat dst_angle = new Mat();
		
		x.put(0, 0, 3.0, 6.0, 5,0);
		y.put(0, 0, 4.0, 8.0, 12.0);
		magnitude.put(0, 0, 5.0, 10.0, 13.0); 
		angle.put(0, 0, 0.92729962, 0.92729962, 1.1759995);

		core.cartToPolar(x, y, dst, dst_angle);		
		assertMatEqual(magnitude, dst);
		assertMatEqual(angle, dst_angle);		
	}

	public void testCartToPolarMatMatMatMatBoolean() {
		Mat x = new Mat(1, 3, CvType.CV_32F);
		Mat y = new Mat(1, 3, CvType.CV_32F);
		Mat magnitude = new Mat(1, 3, CvType.CV_32F);
		Mat angle = new Mat(1, 3, CvType.CV_32F);
		Mat dst_angle = new Mat();
		
		x.put(0 ,0, 3.0, 6.0, 5,0);
		y.put(0 ,0, 4.0, 8.0, 12.0);
		magnitude.put(0 ,0, 5.0, 10.0, 13.0); 
		angle.put(0 ,0, 0.92729962, 0.92729962, 1.1759995);
		
		core.cartToPolar(x, y, dst, dst_angle,false);
		
		assertMatEqual(magnitude, dst);
		OpenCVTestRunner.Log(dst_angle.dump());
		assertMatEqual(angle, dst_angle);	
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
		Mat out = new Mat(1, 8, CvType.CV_32F);
		
		in.put(0, 0, 0.203056, 0.980407, 0.35312, -0.106651, 0.0399382, 0.871475, -0.648355, 0.501067);
		out.put(0, 0, 0.77571625, 0.37270021, 0.18529896, 0.012146413, -0.32499927, -0.99302113, 0.55979407, -0.6251272);
		
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
		assertEquals(8.0, det);		
	}

	public void testDftMatMat() {
		Mat src = new Mat(1, 4, CvType.CV_32F);
		src.put(0, 0, 0, 0, 0, 0);
		
		Mat out = new Mat(1, 4, CvType.CV_32F);
		out.put(0, 0, 0, 0, 0, 0);
		core.dft(src, dst);
		assertMatEqual(out, dst);		
	}

	public void testDftMatMatInt() {
		Mat src = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		Mat out2 = new Mat(1, 4, CvType.CV_32F);
		
		src.put(0, 0, 1, 2, 3, 4);
		out.put(0 , 0, 10, -2, 2, -2);
		core.dft(src, dst, core.DFT_REAL_OUTPUT);
		assertMatEqual(out, dst);
		
		core.dft(src, dst, core.DFT_INVERSE);
		out2.put(0 , 0, 9, -9, 1, 3);
		assertMatEqual(out2, dst);
	}

	public void testDftMatMatIntInt() {
		Mat src = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
	
		src.put(0, 0, 1, 2, 3, 4);
		out.put(0 , 0, 10, -2, 2, -2);
		core.dft(src, dst, core.DFT_REAL_OUTPUT, 1);
		assertMatEqual(out, dst);
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
		Mat m1 = new Mat(2, 2, CvType.CV_32FC1);
		Mat m2 = new Mat(2, 2, CvType.CV_32FC1);
		Mat desired = new Mat(2, 2, CvType.CV_32FC1);
		Mat dmatrix = new Mat(2, 2, CvType.CV_32FC1);
		
		m1.put(0, 0, 1.0, 0.0);
	    m1.put(1, 0, 1.0, 0.0);
		
		m2.put(0, 0, 1.0, 0.0);
		m2.put(1, 0, 1.0, 0.0);
		
		dmatrix.put(0, 0, 0.001, 0.001);
		dmatrix.put(1, 0, 0.001, 0.001);
	    
		desired.put(0, 0, 1.001, 0.001);
		desired.put(1, 0, 1.001, 0.001);
	    
		core.gemm(m1, m2, 1.0, dmatrix, 1.0, dst);
		assertMatEqual(desired, dst);
	}
	
	public void testGemmMatMatDoubleMatDoubleMatInt() {
		Mat m1 = new Mat(2, 2, CvType.CV_32FC1);
		Mat m2 = new Mat(2, 2, CvType.CV_32FC1);
		Mat desired = new Mat(2, 2, CvType.CV_32FC1);
		Mat dmatrix = new Mat(2, 2, CvType.CV_32FC1);
		
		m1.put(0, 0, 1.0, 0.0);
	    m1.put(1, 0, 1.0, 0.0);
		
		m2.put(0, 0, 1.0, 0.0);
		m2.put(1, 0, 1.0, 0.0);
		
		dmatrix.put(0, 0, 0.001, 0.001);
		dmatrix.put(1, 0, 0.001, 0.001);
	    	
		desired.put(0, 0, 2.001, 0.001);
		desired.put(1, 0, 0.001, 0.001);
	    
		core.gemm(m1, m2, 1.0, dmatrix, 1.0, dst, core.GEMM_1_T);
		assertMatEqual(desired, dst);
	}

	public void testGetOptimalDFTSize() {
		int vecsize = core.getOptimalDFTSize(0);
		assertEquals(1, vecsize);
		
		int largeVecSize = core.getOptimalDFTSize(133);
		assertEquals(135, largeVecSize);
		largeVecSize = core.getOptimalDFTSize(13);
		assertEquals(15, largeVecSize);
	}

	public void testGetTickFrequency() {
		double freq = 0.0;
		freq = core.getTickFrequency();
		assertTrue(0.0 != freq);
	}

	public void testHconcat() {
		Mat e = Mat.eye(3, 3, CvType.CV_8UC1);
		Mat eConcat = new Mat(1, 9, CvType.CV_8UC1);
		eConcat.put(0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1);
		
		core.hconcat(e, dst);		
		assertMatEqual(eConcat, dst);
	}
	

	public void testIdctMatMat() {
		Mat in = new Mat(1, 8, CvType.CV_32F);
		Mat out = new Mat(1, 8, CvType.CV_32F);		
		in.put(0, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);
		out.put(0, 0, 3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115);
		
		core.idct(in, dst);
		assertMatEqual(out, dst);
	}

	public void testIdctMatMatInt() {		
		Mat in = new Mat(1, 8, CvType.CV_32F);
		Mat out = new Mat(1, 8, CvType.CV_32F);
		
		in.put(0, 0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0);
		out.put(0, 0, 3.3769724, -1.6215782, 2.3608727, 0.20730907, -0.86502546, 0.028082132, -0.7673766, 0.10917115);
		
		core.idct(in, dst, core.DCT_ROWS);
		assertMatEqual(out, dst);
	}

	public void testIdftMatMat() {
		Mat in = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);

		in.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 9, -9, 1, 3);
		
		core.idft(in, dst);
		assertMatEqual(out, dst);
	}

	public void testIdftMatMatInt() {
		Mat in = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);

		in.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 9, -9, 1, 3);
		core.idft(in, dst, core.DFT_REAL_OUTPUT);
		assertMatEqual(out, dst);
	}

	public void testIdftMatMatIntInt() {
		Mat in = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);

		in.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 9, -9, 1, 3);
		core.idft(in, dst, core.DFT_REAL_OUTPUT, 1);
		assertMatEqual(out, dst);
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
		src.put(0, 0, 1.0);
		src.put(0, 1, 2.0);
		src.put(1, 0, 1.5);
	    src.put(1, 1, 4.0);

		Mat answer = new Mat(2, 2, CvType.CV_32F);
		answer.put(0, 0, 4.0);
		answer.put(0, 1, -2.0);
		answer.put(1, 0, -1.5);
	    answer.put(1, 1, 1.0);
	    
	    core.invert(src, dst);
	    OpenCVTestRunner.Log(answer.dump());
	    OpenCVTestRunner.Log(dst.dump());
	    assertMatEqual(answer, dst);
	    
	    //TODO: needs epsilon comparison
//	    Mat m = grayRnd_32f.clone();
//	    Mat inv = m.inv();
//	    core.gemm(m, inv, 1.0, new Mat(), 0.0, dst);
//	    assertMatEqual(grayE_32f, dst);
	}

	public void testInvertMatMatInt() {
		Mat src = new Mat(3, 3, CvType.CV_32F);
		Mat out = new Mat(3, 3, CvType.CV_32F);
		src.put(0, 0, 1, 0, 0);
		src.put(1, 0, 0, 1, 0);
		src.put(2, 0, 0, 0, 1);
		
		out.put(0, 0, 1, 0, 0);
		out.put(1, 0, 0, 1, 0);
		out.put(2, 0, 0, 0, 1);
		
		core.invert(src, dst,core.DECOMP_CHOLESKY);
		assertMatEqual(out, dst);	
		
		core.invert(src, dst,core.DECOMP_LU);
		double det = core.determinant(src);
		assertTrue(det > 0.0);
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
       int nPoints = Math.min(gray0.cols(), gray0.rows());
		
		Point point1 = new Point(0, 0);
		Point point2 = new Point(nPoints, nPoints);
		Scalar color = new Scalar(255);
		
		assertTrue(0 == core.countNonZero(gray0));
		core.line(gray0, point1, point2, color, 0);
		assertTrue(nPoints == core.countNonZero(gray0));
	}

	public void testLineMatPointPointScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testLineMatPointPointScalarIntIntInt() {
		fail("Not yet implemented");
	}

	public void testLog() {
		Mat in = new Mat(1, 4, CvType.CV_32FC1);
		Mat desired = new Mat(1, 4, CvType.CV_32FC1);
		in.put(0, 0, 1.0, 10.0, 100.0, 1000.0);
		desired.put(0, 0, 0, 2.3025851, 4.6051702, 6.9077554);
		
		core.log(in, dst);
		assertMatEqual(desired, dst);
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
		Mat lut = new Mat(1, 256, CvType.CV_8UC1);
		lut.setTo(new Scalar(255));
	    core.LUT(grayRnd, lut, dst, 0);
	    assertMatEqual(gray255, dst);
	}

	public void testMagnitude() {
		Mat x = new Mat(1, 4, CvType.CV_32F);
		Mat y = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		
		x.put(0, 0, 3.0, 5.0, 9.0, 6.0);
		y.put(0, 0, 4.0, 12.0, 40.0, 8.0);
		out.put(0, 0, 5.0, 13.0, 41.0, 10.0);
		
        core.magnitude(x, y, dst);
        assertMatEqual(out,dst);
        
       	core.magnitude(gray0_32f, gray255_32f, dst);
		assertMatEqual(gray255_32f, dst);		
	}

	public void testMahalanobis() {	
		Mat covar = new Mat(matSize, matSize, CvType.CV_32F);
		Mat mean = new Mat(1, matSize, CvType.CV_32F);		
		core.calcCovarMatrix(grayRnd_32f, covar, mean, 8|1/*TODO: CV_COVAR_NORMAL*/, CvType.CV_32F);
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
		Mat mean = new Mat();
		Mat stddev = new Mat();
		
		core.meanStdDev(rgba0, mean, stddev);
		assertEquals(0, core.countNonZero(mean));
		assertEquals(0, core.countNonZero(stddev));
	}

	public void testMeanStdDevMatMatMatMat() {
		Mat mean = new Mat();
		Mat stddev = new Mat();
		
		core.meanStdDev(rgba0, mean, stddev, gray255);
		assertEquals(0, core.countNonZero(mean));
		assertEquals(0, core.countNonZero(stddev));
		
		Mat submat = grayRnd.submat(0, grayRnd.rows()/2, 0, grayRnd.cols()/2);
		submat.setTo(new Scalar(33));
		
		Mat mask = gray0.clone();
		submat = mask.submat(0, mask.rows()/2, 0, mask.cols()/2);
		submat.setTo(new Scalar(1));
		
		core.meanStdDev(grayRnd, mean, stddev, mask);
		Mat desiredMean = new Mat(1, 1, CvType.CV_64F, new Scalar(33));
		assertMatEqual(desiredMean, mean);
		assertEquals(0, core.countNonZero(stddev));
		
		core.meanStdDev(grayRnd, mean, stddev, gray1);
		assertTrue(0 != core.countNonZero(mean));
		assertTrue(0 != core.countNonZero(stddev));
	}

	public void testMin() {
		core.min(gray0, gray255, dst);
		assertMatEqual(gray0, dst);		
	}

	public void testMinMaxLoc() {
		double minVal = 1;
		double maxVal = 10;
		Point minLoc = new Point(gray3.cols()/4, gray3.rows()/2);
		Point maxLoc = new Point(gray3.cols()/2, gray3.rows()/4);
		gray3.put((int)minLoc.y, (int)minLoc.x, minVal);
		gray3.put((int)maxLoc.y, (int)maxLoc.x, maxVal);
		
		core.MinMaxLocResult mmres = core.minMaxLoc(gray3);
		
		assertTrue(mmres.minVal == minVal); 
		assertTrue(mmres.maxVal == maxVal); 
		assertTrue(mmres.minLoc.equals(minLoc)); 
		assertTrue(mmres.maxLoc.equals(maxLoc));		
	}

	public void testMulSpectrumsMatMatMatInt() {
		Mat src1 = new Mat(1, 4, CvType.CV_32F);
		Mat src2 = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		src1.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		src2.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 1, -5, 12, 16);
		core.mulSpectrums(src1, src2, dst, core.DFT_ROWS);
		assertMatEqual(out, dst);
	}

	public void testMulSpectrumsMatMatMatIntBoolean() {
		Mat src1 = new Mat(1, 4, CvType.CV_32F);
		Mat src2 = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		src1.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		src2.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 1, 13, 0, 16);
		core.mulSpectrums(src1, src2, dst, core.DFT_ROWS, true);
		assertMatEqual(out, dst);
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

		Mat grayDelta = new Mat(matSize, matSize, CvType.CV_32F);
        grayDelta.setTo(new Scalar(0.0));
		core.mulTransposed(grayE_32f, dst, true, grayDelta);
		assertMatEqual(grayE_32f, dst);
	}

	public void testMulTransposedMatMatBooleanMatDouble() {
		Mat grayDelta = new Mat(matSize, matSize, CvType.CV_32F);
		grayDelta.setTo(new Scalar(0.0));
		core.mulTransposed(grayE_32f, dst, true, grayDelta, 1);
		assertMatEqual(grayE_32f, dst);
	}

	public void testMulTransposedMatMatBooleanMatDoubleInt() {
		Mat a = new Mat(3, 3, CvType.CV_32F);
		Mat grayDelta = new Mat(3, 3, CvType.CV_8U);
		grayDelta.setTo(new Scalar(0.0001));
		Mat res = new Mat(3, 3, CvType.CV_32F);
		a.put(0, 0, 1, 1, 1);
		a.put(1, 0, 1, 1, 1);
		a.put(2, 0, 1, 1, 1);
		res.put(0, 0, 3, 3, 3);
		res.put(1, 0, 3, 3, 3);
		res.put(2, 0, 3, 3, 3);
		
	    core.mulTransposed(a, dst, true, grayDelta, 1.0, 1);
		assertMatEqual(res, dst);        
	}

	public void testNormalizeMatMat() {
		core.normalize(gray0, dst);
		assertMatEqual(gray0, dst);
	}

	public void testNormalizeMatMatDouble() {
		core.normalize(gray0, dst, 0.0);
		assertMatEqual(gray0, dst);
	}

	public void testNormalizeMatMatDoubleDouble() {
		core.normalize(gray0, dst, 0.0, 1.0);
		assertMatEqual(gray0, dst);
	}

	public void testNormalizeMatMatDoubleDoubleInt() {
		Mat src = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		src.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 0.25, 0.5, 0.75, 1);
		core.normalize(src, dst, 1.0, 2.0, core.NORM_INF);
		assertMatEqual(out, dst);
	}

	public void testNormalizeMatMatDoubleDoubleIntInt() {
		Mat src = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		
		src.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 0.25, 0.5, 0.75, 1);
		
		core.normalize(src, dst, 1.0, 2.0, core.NORM_INF, -1);
		assertMatEqual(out, dst);
	}

	public void testNormalizeMatMatDoubleDoubleIntIntMat() {
		Mat src = new Mat(1, 4, CvType.CV_32F);
		Mat out = new Mat(1, 4, CvType.CV_32F);
		Mat mask = new Mat(1, 4, CvType.CV_8U, new Scalar(1));
		
		src.put(0, 0, 1.0, 2.0, 3.0, 4.0);
		out.put(0, 0, 0.25, 0.5, 0.75, 1);
		
		core.normalize(src, dst, 1.0, 2.0, core.NORM_INF, -1, mask);
		assertMatEqual(out, dst);
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
		//nice example
		fail("Not yet implemented");
	}

	public void testPhaseMatMatMat() {
		Mat x = new Mat(1, 4, CvType.CV_32F);
		Mat y = new Mat(1, 4, CvType.CV_32F);
		Mat res = new Mat(1, 4, CvType.CV_32F);
		
		x.put(0, 0, 10.0, 10.0, 20.0, 5.0);
		y.put(0, 0, 20.0, 15.0, 20.0, 20.0);
		res.put(0, 0, 1.1071469, 0.98280007, 0.78539175, 1.3258134);
		
		core.phase(x, y, dst);
		assertMatEqual(res, dst);		
	}

	public void testPhaseMatMatMatBoolean() {
		Mat x = new Mat(1, 4, CvType.CV_32F);
		Mat y = new Mat(1, 4, CvType.CV_32F);
		Mat res = new Mat(1, 4, CvType.CV_32F);
		
		x.put(0, 0, 10.0, 10.0, 20.0, 5.0);
		y.put(0, 0, 20.0, 15.0, 20.0, 20.0);
		res.put(0, 0, 63.434, 56.310, 44.999, 75.963);
		
		core.phase(x, y, dst, true);
		OpenCVTestRunner.Log(res.dump());
		OpenCVTestRunner.Log(dst.dump());
	}

	public void testPolarToCartMatMatMatMat() {
		Mat magnitude = new Mat(1, 3, CvType.CV_32F);
		Mat angle = new Mat(1, 3, CvType.CV_32F);
		Mat x = new Mat(1, 3, CvType.CV_32F);
		Mat y = new Mat(1, 3, CvType.CV_32F);
		Mat xCoordinate = new Mat();
		Mat yCoordinate = new Mat();
		
//		x.put(0, 0, 3.0, 6.0, 5,0);
//		y.put(0, 0, 4.0, 8.0, 12.0);
//		magnitude.put(0, 0, 5.0, 10.0, 13.0); 
//		angle.put(0, 0, 0.92729962, 0.92729962, 1.1759995);
		
		magnitude.put(0, 0, 5.0, 10.0, 13.0);
		angle.put(0, 0, 0.92729962, 0.92729962, 1.1759995);
		
		x.put(0, 0, 3.0, 6.0, 5,0);
		y.put(0, 0, 4.0, 8.0, 12.0);
		
		//TODO: needs epsilon comparison
		core.polarToCart(magnitude, angle, xCoordinate, yCoordinate);
		OpenCVTestRunner.Log(x.dump());
		OpenCVTestRunner.Log(xCoordinate.dump());		
		OpenCVTestRunner.Log(y.dump());
		OpenCVTestRunner.Log(yCoordinate.dump());
		assertMatEqual(x, xCoordinate);		
	}

	public void testPolarToCartMatMatMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testPow() {
		core.pow(gray3, 2.0, dst);
		assertMatEqual(gray9, dst);
	}
	
	public void testRandn() {
        Mat low  = new Mat(1, 1, CvType.CV_16UC1, new Scalar(0));
        Mat high = new Mat(1, 1, CvType.CV_16UC1, new Scalar(256));
		
        assertTrue(0 == core.countNonZero(gray0));
        core.randn(gray0, low, high);
        assertTrue(0 != core.countNonZero(gray0));	
	}

	public void testRandu() {
        Mat low  = new Mat(1, 1, CvType.CV_16UC1, new Scalar(0));
        Mat high = new Mat(1, 1, CvType.CV_16UC1, new Scalar(256));
		
        assertTrue(0 == core.countNonZero(gray0));
        core.randu(gray0, low, high);
        assertTrue(0 != core.countNonZero(gray0));
	}

	public void testRectangleMatPointPointScalar() {
		Point center = new Point(gray0.cols()/2, gray0.rows()/2);
		Point origin = new Point(0,0);
		Scalar color = new Scalar(128);
		
		assertTrue(0 == core.countNonZero(gray0));
		core.rectangle(gray0, center, origin, color);
		assertTrue(0 != core.countNonZero(gray0));
	}

	public void testRectangleMatPointPointScalarInt() {
		 Point center = new Point(gray0.cols(), gray0.rows());
		 Point origin = new Point(0,0);
		 Scalar color = new Scalar(128);
		 
		 assertTrue(0 == core.countNonZero(gray0));
		 core.rectangle(gray0, center, origin, color, 2);
		 assertTrue(0 != core.countNonZero(gray0));
	}

	public void testRectangleMatPointPointScalarIntInt() {
		Point center = new Point(gray0.cols()/2, gray0.rows()/2);
		Point origin = new Point(0,0);
		Scalar color = new Scalar(128);
		
		assertTrue(0 == core.countNonZero(gray0));
		core.rectangle(gray0, center, origin, color, 2, 8);
		assertTrue(0 != core.countNonZero(gray0));
	}

	public void testRectangleMatPointPointScalarIntIntInt() {
		 Point center = new Point(gray0.cols(), gray0.rows());
		 Point origin = new Point(0,0);
		 Scalar color = new Scalar(128);
		 
		 assertTrue(0 == core.countNonZero(gray0));
		 core.rectangle(gray0, center, origin, color, 2, 4, 2);
		 assertTrue(0 != core.countNonZero(gray0));
	}

	public void testReduceMatMatIntInt() {
		Mat src = new Mat(2, 2, CvType.CV_32F);
		Mat out = new Mat(1, 2, CvType.CV_32F);
		src.put(0, 0, 1 , 0);
		src.put(1, 0, 1 , 0);
		
        out.put(0 , 0, 1, 0);
        
        core.reduce(src, dst, 0, 2);
        assertMatEqual(out, dst);
	}

	public void testReduceMatMatIntIntInt() {
		Mat src = new Mat(2, 2, CvType.CV_32F);
		Mat out = new Mat(1, 2, CvType.CV_32F);
		src.put(0, 0, 1 , 0);
		src.put(1, 0, 1 , 0);
		
        out.put(0 , 0, 1, 0);
        
        core.reduce(src, dst, 0, 2, -1);
        assertMatEqual(out, dst);
	}

	public void testRepeat() {
		Mat src = new Mat(1, 3, CvType.CV_32F);
		Mat des1 = new Mat(1, 3, CvType.CV_32F);
		Mat des2 = new Mat(1, 6, CvType.CV_32F);
		src.put(0, 0, 1, 2, 3);
		
		des1.put(0, 0, 1, 2, 3);
		des2.put(0, 0, 1, 2, 3, 1, 2, 3);
		
		core.repeat(src, 1, 1, dst);
		assertMatEqual(des1, dst);
		core.repeat(src, 1, 2, dst);
	    assertMatEqual(des2, dst);
	}

	public void testScaleAdd() {
		core.scaleAdd(gray3, 2.0, gray3, dst);
		assertMatEqual(dst, gray9);
	}

	public void testSetIdentityMat() {
		core.setIdentity(gray0_32f);
		assertMatEqual(grayE_32f, gray0_32f);
	}

	public void testSetIdentityMatScalar() {
		core.gemm(grayE_32f, grayE_32f, 5.0, new Mat(), 0.0, dst);
		core.setIdentity(gray0_32f, new Scalar(5));
		assertMatEqual(dst, gray0_32f);
	}

	public void testSolveCubic() {
		Mat coeffs = new Mat(1, 4, CvType.CV_32F);
		Mat roots = new Mat(3, 1, CvType.CV_32F);
		coeffs.put(0, 0, 1, 6, 11, 6);
		roots.put(0, 0, -3, -1, -2);
		core.solveCubic(coeffs, dst);
		assertMatEqual(roots, dst);
	}

	public void testSolveMatMatMat() {
		Mat a = new Mat(3, 3, CvType.CV_32F);
		Mat b = new Mat(3, 1, CvType.CV_32F);
		Mat res = new Mat(3, 1, CvType.CV_32F);
		a.put(0, 0, 1, 1, 1);
		a.put(1, 0, 1, -2, 2);
		a.put(2, 0, 1, 2, 1);
		
		b.put(0, 0, 0, 4, 2);	
		res.put(0, 0, -12, 2, 10);
		
		core.solve(a, b, dst);
	    assertMatEqual(res, dst);	
	}

	public void testSolveMatMatMatInt() {
		Mat a = new Mat(3, 3, CvType.CV_32F);
		Mat b = new Mat(3, 1, CvType.CV_32F);
		Mat res = new Mat(3, 1, CvType.CV_32F);
		
		a.put(0, 0, 1, 1, 1);
		a.put(1, 0, 1, -2, 2);
		a.put(2, 0, 1, 2, 1);
		
		b.put(0, 0, 0, 4, 2);	
		res.put(0, 0, -12, 2, 10);
		
		core.solve(a, b, dst, 3);
	    assertMatEqual(res, dst);
	}

	public void testSolvePolyMatMat() {
		Mat coeffs = new Mat(4, 1, CvType.CV_32F);
		Mat roots = new Mat(3, 1, CvType.CV_32F);

		coeffs.put(0, 0, -6, 11, -6, 1);
		
		Mat answer = new Mat(3, 1, CvType.CV_32FC2);
		//FIXME: doesn't work answer.put(0, 0, 1, 0, 2, 0, 3, 0);
		answer.put(0, 0, 1, 0);
		answer.put(1, 0, 2, 0);
		answer.put(2, 0, 3, 0);
		
		core.solvePoly(coeffs, roots);		
		assertMatEqual(answer, roots);
	}

	public void testSolvePolyMatMatInt() {
		Mat coeffs = new Mat(4, 1, CvType.CV_32F);
		Mat roots = new Mat(3, 1, CvType.CV_32F);

		coeffs.put(0, 0, -6, 11, -6, 1);
		
		Mat answer = new Mat(3, 1, CvType.CV_32FC2);
		//FIXME: doesn't work answer.put(0, 0, 1, 0, 2, 0, 3, 0);
		answer.put(0, 0, 1, 0);
		answer.put(1, 0, -1, 2);
		answer.put(2, 0, -2, 12);
		
		core.solvePoly(coeffs, roots, 1);		
		assertMatEqual(answer, roots);
	}

	public void testSort() {
		Mat submat = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols() / 2);
		submat.setTo(new Scalar(1.0));
		
		core.sort(gray0, dst, 0/*TODO: CV_SORT_EVERY_ROW*/);		
		submat = dst.submat(0, dst.rows() / 2, dst.cols() / 2, dst.cols());
		assertTrue(submat.total() == core.countNonZero(submat));
		
		core.sort(gray0, dst, 1/*TODO: CV_SORT_EVERY_COLUMN*/); 
		submat = dst.submat(dst.rows() / 2, dst.rows(), 0, dst.cols() / 2);
		assertTrue(submat.total() == core.countNonZero(submat));
	}

	public void testSortIdx() {
		Mat a = Mat.eye(3, 3, CvType.CV_8UC1);
		Mat b = new Mat();
		
		Mat answer = new Mat(3, 3, CvType.CV_32SC1);
		answer.put(0, 0, 1, 2, 0);
		answer.put(1, 0, 0, 2, 1);
		answer.put(2, 0, 0, 1, 2);
		
		core.sortIdx(a, b, 0+0/*TODO: CV_SORT_EVERY_ROW + CV_SORT_ASCENDING*/);
		assertMatEqual(answer, b);
	}

	public void testSqrt() {
		core.sqrt(gray9_32f, dst);
		assertMatEqual(gray3_32f, dst);
		
		Mat rgba144 = new Mat(matSize, matSize, CvType.CV_32FC4);
		Mat rgba12 = new Mat(matSize, matSize, CvType.CV_32FC4);
		rgba144.setTo(Scalar.all(144));
		rgba12.setTo(Scalar.all(12));

		core.sqrt(rgba144, dst);
		assertMatEqual(rgba12, dst);
	}

	public void testSubtractMatMatMat() {
		core.subtract(gray128, gray1, dst);
		assertMatEqual(gray127, dst);
	}

	public void testSubtractMatMatMatMat() {
		Mat mask = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(0)); 
		Mat submask = mask.submat(0, mask.rows() / 2, 0, mask.cols() / 2);
		submask.setTo(new Scalar(1));
		
		dst = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(0));
		core.subtract(gray3, gray2, dst, mask);
		assertTrue(submask.total() == core.countNonZero(dst));
	}

	public void testSubtractMatMatMatMatInt() {
		core.subtract(gray3, gray2, dst, gray1, CvType.CV_32F);
		assertTrue(CvType.CV_32F == dst.depth());
		assertMatEqual(gray1_32f, dst);
	}

	public void testTransform() {	
		Mat src = new Mat(2, 2, CvType.CV_32F, new Scalar(55));
		Mat m = Mat.eye(2, 2, CvType.CV_32FC1);
		
		core.transform(src, dst, m);
		Mat answer = new Mat(2, 2, CvType.CV_32FC2, new Scalar(55, 1));
		assertMatEqual(answer, dst);
	}

	public void testTranspose() {
		Mat subgray0 = gray0.submat(0, gray0.rows() / 2, 0, gray0.cols());
		Mat destination = new Mat(matSize, matSize, CvType.CV_8U); destination.setTo(new Scalar(0));
		Mat subdst = destination.submat(0, destination.rows(), 0, destination.cols() / 2);
		subgray0.setTo(new Scalar(1));
		core.transpose(gray0, destination);
		assertTrue(subdst.total() == core.countNonZero(subdst));
	}

	public void testVconcat() {
		fail("Not yet implemented");
	}
	
}
