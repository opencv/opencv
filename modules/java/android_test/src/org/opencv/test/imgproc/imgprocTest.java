package org.opencv.test.imgproc;

import java.util.ArrayList;
import java.util.List;

import org.opencv.CvType;
import org.opencv.Mat;
import org.opencv.Scalar;
import org.opencv.Size;
import org.opencv.core;
import org.opencv.imgproc;
import org.opencv.test.OpenCVTestCase;


public class imgprocTest extends OpenCVTestCase {
	
	public void test_1() {
		super.test_1("imgproc");
	}
	
	//FIXME: this test crashes
	//public void test_Can_Call_accumulate() {
	//	 dst = new Mat(gray1.rows(), gray1.cols(), Mat.CvType.CV_32FC1);
	//	 imgproc.accumulate(gray1, dst);
	//	 assertMatEqual(gray1, dst);
	//}

	public void testAccumulateMatMat() {
		fail("Not yet implemented");
	}

	public void testAccumulateMatMatMat() {
		fail("Not yet implemented");
	}

	public void testAccumulateProductMatMatMat() {
		fail("Not yet implemented");
	}

	public void testAccumulateProductMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testAccumulateSquareMatMat() {
		fail("Not yet implemented");
	}

	public void testAccumulateSquareMatMatMat() {
		fail("Not yet implemented");
	}

	public void testAccumulateWeightedMatMatDouble() {
		fail("Not yet implemented");
	}

	public void testAccumulateWeightedMatMatDoubleMat() {
		fail("Not yet implemented");
	}

	public void testAdaptiveThreshold() {
		fail("Not yet implemented");
	}

	public void testApproxPolyDP() {
		fail("Not yet implemented");
	}

	public void testArcLength() {
		fail("Not yet implemented");
	}

	public void testBilateralFilterMatMatIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testBilateralFilterMatMatIntDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testBlurMatMatSize() {
		Size sz = new Size(3, 3);

		imgproc.blur(gray0, dst, sz);
		assertMatEqual(gray0, dst);

		imgproc.blur(gray255, dst, sz);
		assertMatEqual(gray255, dst);
	}

	public void testBlurMatMatSizePoint() {
		fail("Not yet implemented");
	}

	public void testBlurMatMatSizePointInt() {
		fail("Not yet implemented");
	}

	public void testBorderInterpolate() {
		fail("Not yet implemented");
	}

	public void testBoundingRect() {
		fail("Not yet implemented");
	}

	public void testBoxFilterMatMatIntSize() {
		Size sz = new Size(3, 3);
		imgproc.boxFilter(gray0, dst, 8, sz);
		assertMatEqual(gray0, dst);
	}

	public void testBoxFilterMatMatIntSizePoint() {
		fail("Not yet implemented");
	}

	public void testBoxFilterMatMatIntSizePointBoolean() {
		fail("Not yet implemented");
	}

	public void testBoxFilterMatMatIntSizePointBooleanInt() {
		fail("Not yet implemented");
	}

	public void testCalcBackProject() {
		ArrayList<Mat> images = new ArrayList<Mat>();
		List<Integer> channels = new ArrayList<Integer>();
		List<Integer> histSize = new ArrayList<Integer>();
		List<Float> ranges = new ArrayList<Float>();		
		
		images.add(grayChess);
		channels.add(0);
		histSize.add(10);
		ranges.add(0.0f); ranges.add(256.0f);

		Mat hist = new Mat();
		imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
		core.normalize(hist, hist);
		
		imgproc.calcBackProject(images, channels, hist, dst, ranges, 255);
		assertTrue(grayChess.size().equals(dst.size()));
		assertEquals(grayChess.depth(), dst.depth());
		assertTrue(0 != core.countNonZero(dst));
	}

	public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat() {
		ArrayList<Mat> images = new ArrayList<Mat>();
		List<Integer> channels = new ArrayList<Integer>();
		List<Integer> histSize = new ArrayList<Integer>();
		List<Float> ranges = new ArrayList<Float>();		
		
		images.add(gray128);
		channels.add(0);
		histSize.add(10);
		ranges.add(0.0f); ranges.add(256.0f);

		truth = new Mat(10, 1, CvType.CV_32F, Scalar.all(0.0));
		truth.put(5, 0, 100.0);
		
		Mat hist = new Mat();
		imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
		assertMatEqual(truth, hist);
	}
	
	public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloat2d() {
		ArrayList<Mat> images = new ArrayList<Mat>();
		List<Integer> channels = new ArrayList<Integer>();
		List<Integer> histSize = new ArrayList<Integer>();
		List<Float> ranges = new ArrayList<Float>();	
		
		images.add(gray255);
		images.add(gray128);		
		
		channels.add(0);
		channels.add(1);
		
		histSize.add(10);
		histSize.add(10);
		
		ranges.add(0.0f); ranges.add(256.0f);
		ranges.add(0.0f); ranges.add(256.0f);
		
		truth = new Mat(10, 10, CvType.CV_32F, Scalar.all(0.0));
		truth.put(9, 5, 100.0);
		
		Mat hist = new Mat();
		imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
		assertMatEqual(truth, hist);
	}

	public void testCalcHistListOfMatListOfIntegerMatMatListOfIntegerListOfFloatBoolean() {
		fail("Not yet implemented");
	}

	public void testCannyMatMatDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testCannyMatMatDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testCannyMatMatDoubleDoubleIntBoolean() {
		fail("Not yet implemented");
	}

	public void testCompareHist() {
		fail("Not yet implemented");
	}

	public void testContourAreaMat() {
		fail("Not yet implemented");
	}

	public void testContourAreaMatBoolean() {
		fail("Not yet implemented");
	}

	public void testConvertMapsMatMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testConvertMapsMatMatMatMatIntBoolean() {
		fail("Not yet implemented");
	}
	
	public void testConvexHullMatMat() {
		fail("Not yet implemented");
	}

	public void testConvexHullMatMatBoolean() {
		fail("Not yet implemented");
	}

	public void testConvexHullMatMatBooleanBoolean() {
		fail("Not yet implemented");
	}

	public void testCopyMakeBorderMatMatIntIntIntIntInt() {
		fail("Not yet implemented");
	}

	public void testCopyMakeBorderMatMatIntIntIntIntIntScalar() {
		fail("Not yet implemented");
	}

	public void testCornerEigenValsAndVecsMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testCornerEigenValsAndVecsMatMatIntIntInt() {
		fail("Not yet implemented");
	}

	public void testCornerHarrisMatMatIntIntDouble() {
		fail("Not yet implemented");
	}

	public void testCornerHarrisMatMatIntIntDoubleInt() {
		fail("Not yet implemented");
	}

	public void testCornerMinEigenValMatMatInt() {
		fail("Not yet implemented");
	}

	public void testCornerMinEigenValMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testCornerMinEigenValMatMatIntIntInt() {
		fail("Not yet implemented");
	}

	public void testCornerSubPix() {
		fail("Not yet implemented");
	}

	public void testCvtColorMatMatInt() {
		fail("Not yet implemented");
	}

	public void testCvtColorMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testDilateMatMatMat() {
		fail("Not yet implemented");
	}

	public void testDilateMatMatMatPoint() {
		fail("Not yet implemented");
	}

	public void testDilateMatMatMatPointInt() {
		fail("Not yet implemented");
	}

	public void testDilateMatMatMatPointIntInt() {
		fail("Not yet implemented");
	}

	public void testDilateMatMatMatPointIntIntScalar() {
		fail("Not yet implemented");
	}

	public void testDistanceTransform() {
		fail("Not yet implemented");
	}

	public void testDrawContoursMatListOfMatIntScalar() {
		fail("Not yet implemented");
	}

	public void testDrawContoursMatListOfMatIntScalarInt() {
		fail("Not yet implemented");
	}

	public void testDrawContoursMatListOfMatIntScalarIntInt() {
		fail("Not yet implemented");
	}

	public void testDrawContoursMatListOfMatIntScalarIntIntMat() {
		fail("Not yet implemented");
	}

	public void testDrawContoursMatListOfMatIntScalarIntIntMatInt() {
		fail("Not yet implemented");
	}

	public void testDrawContoursMatListOfMatIntScalarIntIntMatIntPoint() {
		fail("Not yet implemented");
	}

	public void testEqualizeHist() {
		fail("Not yet implemented");
	}

	public void testErodeMatMatMat() {
		fail("Not yet implemented");
	}

	public void testErodeMatMatMatPoint() {
		fail("Not yet implemented");
	}

	public void testErodeMatMatMatPointInt() {
		fail("Not yet implemented");
	}

	public void testErodeMatMatMatPointIntInt() {
		fail("Not yet implemented");
	}

	public void testErodeMatMatMatPointIntIntScalar() {
		fail("Not yet implemented");
	}

	public void testFilter2DMatMatIntMat() {
		fail("Not yet implemented");
	}

	public void testFilter2DMatMatIntMatPoint() {
		fail("Not yet implemented");
	}

	public void testFilter2DMatMatIntMatPointDouble() {
		fail("Not yet implemented");
	}

	public void testFilter2DMatMatIntMatPointDoubleInt() {
		fail("Not yet implemented");
	}

	public void testFindContoursMatListOfMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testFindContoursMatListOfMatMatIntIntPoint() {
		fail("Not yet implemented");
	}

	public void testFitEllipse() {
		fail("Not yet implemented");
	}

	public void testFitLine() {
		fail("Not yet implemented");
	}

	public void testFloodFillMatMatPointScalar() {
		fail("Not yet implemented");
	}

	public void testFloodFillMatMatPointScalarRect() {
		fail("Not yet implemented");
	}

	public void testFloodFillMatMatPointScalarRectScalar() {
		fail("Not yet implemented");
	}

	public void testFloodFillMatMatPointScalarRectScalarScalar() {
		fail("Not yet implemented");
	}

	public void testFloodFillMatMatPointScalarRectScalarScalarInt() {
		fail("Not yet implemented");
	}

	public void testGaussianBlurMatMatSizeDouble() {
		fail("Not yet implemented");
	}

	public void testGaussianBlurMatMatSizeDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testGaussianBlurMatMatSizeDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testGetAffineTransform() {
		fail("Not yet implemented");
	}

	public void testGetDefaultNewCameraMatrixMat() {
		fail("Not yet implemented");
	}

	public void testGetDefaultNewCameraMatrixMatSize() {
		fail("Not yet implemented");
	}

	public void testGetDefaultNewCameraMatrixMatSizeBoolean() {
		fail("Not yet implemented");
	}

	public void testGetDerivKernelsMatMatIntIntInt() {
		fail("Not yet implemented");
	}

	public void testGetDerivKernelsMatMatIntIntIntBoolean() {
		fail("Not yet implemented");
	}

	public void testGetDerivKernelsMatMatIntIntIntBooleanInt() {
		fail("Not yet implemented");
	}

	public void testGetGaussianKernelIntDouble() {
		fail("Not yet implemented");
	}

	public void testGetGaussianKernelIntDoubleInt() {
		fail("Not yet implemented");
	}

	public void testGetRectSubPixMatSizePointMat() {
		fail("Not yet implemented");
	}

	public void testGetRectSubPixMatSizePointMatInt() {
		fail("Not yet implemented");
	}

	public void testGetRotationMatrix2D() {
		fail("Not yet implemented");
	}

	public void testGetStructuringElementIntSize() {
		fail("Not yet implemented");
	}

	public void testGetStructuringElementIntSizePoint() {
		fail("Not yet implemented");
	}

	public void testGoodFeaturesToTrackMatMatIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMat() {
		fail("Not yet implemented");
	}

	public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMatInt() {
		fail("Not yet implemented");
	}

	public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMatIntBoolean() {
		fail("Not yet implemented");
	}

	public void testGoodFeaturesToTrackMatMatIntDoubleDoubleMatIntBooleanDouble() {
		fail("Not yet implemented");
	}

	public void testGrabCutMatMatRectMatMatInt() {
		fail("Not yet implemented");
	}

	public void testGrabCutMatMatRectMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testHoughCirclesMatMatIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testHoughCirclesMatMatIntDoubleDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testHoughCirclesMatMatIntDoubleDoubleDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testHoughCirclesMatMatIntDoubleDoubleDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testHoughCirclesMatMatIntDoubleDoubleDoubleDoubleIntInt() {
		fail("Not yet implemented");
	}

	public void testHoughLinesMatMatDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testHoughLinesMatMatDoubleDoubleIntDouble() {
		fail("Not yet implemented");
	}

	public void testHoughLinesMatMatDoubleDoubleIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testHoughLinesPMatMatDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testHoughLinesPMatMatDoubleDoubleIntDouble() {
		fail("Not yet implemented");
	}

	public void testHoughLinesPMatMatDoubleDoubleIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testHuMoments() {
		fail("Not yet implemented");
	}

	public void testInitUndistortRectifyMap() {
		fail("Not yet implemented");
	}

	public void testInitWideAngleProjMapMatMatSizeIntIntMatMat() {
		fail("Not yet implemented");
	}

	public void testInitWideAngleProjMapMatMatSizeIntIntMatMatInt() {
		fail("Not yet implemented");
	}

	public void testInitWideAngleProjMapMatMatSizeIntIntMatMatIntDouble() {
		fail("Not yet implemented");
	}

	public void testInpaint() {
		fail("Not yet implemented");
	}

	public void testIntegral2MatMatMat() {
		fail("Not yet implemented");
	}

	public void testIntegral2MatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testIntegral3MatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testIntegral3MatMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testIntegralMatMat() {
		fail("Not yet implemented");
	}

	public void testIntegralMatMatInt() {
		fail("Not yet implemented");
	}

	public void testInvertAffineTransform() {
		fail("Not yet implemented");
	}

	public void testIsContourConvex() {
		fail("Not yet implemented");
	}

	public void testLaplacianMatMatInt() {
		fail("Not yet implemented");
	}

	public void testLaplacianMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testLaplacianMatMatIntIntDouble() {
		fail("Not yet implemented");
	}

	public void testLaplacianMatMatIntIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testLaplacianMatMatIntIntDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testMatchShapes() {
		fail("Not yet implemented");
	}

	public void testMatchTemplate() {
		fail("Not yet implemented");
	}

	public void testMedianBlur() {
		fail("Not yet implemented");
	}

	public void testMinAreaRect() {
		fail("Not yet implemented");
	}

	public void testMinEnclosingCircle() {
		fail("Not yet implemented");
	}

	public void testMomentsMat() {
		fail("Not yet implemented");
	}

	public void testMomentsMatBoolean() {
		fail("Not yet implemented");
	}

	public void testMorphologyExMatMatIntMat() {
		fail("Not yet implemented");
	}

	public void testMorphologyExMatMatIntMatPoint() {
		fail("Not yet implemented");
	}

	public void testMorphologyExMatMatIntMatPointInt() {
		fail("Not yet implemented");
	}

	public void testMorphologyExMatMatIntMatPointIntInt() {
		fail("Not yet implemented");
	}

	public void testMorphologyExMatMatIntMatPointIntIntScalar() {
		fail("Not yet implemented");
	}

	public void testPointPolygonTest() {
		fail("Not yet implemented");
	}

	public void testPreCornerDetectMatMatInt() {
		fail("Not yet implemented");
	}

	public void testPreCornerDetectMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testPyrDownMatMat() {
		fail("Not yet implemented");
	}

	public void testPyrDownMatMatSize() {
		fail("Not yet implemented");
	}

	public void testPyrMeanShiftFilteringMatMatDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testPyrMeanShiftFilteringMatMatDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testPyrMeanShiftFilteringMatMatDoubleDoubleIntTermCriteria() {
		fail("Not yet implemented");
	}

	public void testPyrUpMatMat() {
		fail("Not yet implemented");
	}

	public void testPyrUpMatMatSize() {
		fail("Not yet implemented");
	}

	public void testRemapMatMatMatMatInt() {
		fail("Not yet implemented");
	}

	public void testRemapMatMatMatMatIntInt() {
		fail("Not yet implemented");
	}

	public void testRemapMatMatMatMatIntIntScalar() {
		fail("Not yet implemented");
	}

	public void testResizeMatMatSize() {
		fail("Not yet implemented");
	}

	public void testResizeMatMatSizeDouble() {
		fail("Not yet implemented");
	}

	public void testResizeMatMatSizeDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testResizeMatMatSizeDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testScharrMatMatIntIntInt() {
		fail("Not yet implemented");
	}

	public void testScharrMatMatIntIntIntDouble() {
		fail("Not yet implemented");
	}

	public void testScharrMatMatIntIntIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testScharrMatMatIntIntIntDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testSepFilter2DMatMatIntMatMat() {
		fail("Not yet implemented");
	}

	public void testSepFilter2DMatMatIntMatMatPoint() {
		fail("Not yet implemented");
	}

	public void testSepFilter2DMatMatIntMatMatPointDouble() {
		fail("Not yet implemented");
	}

	public void testSepFilter2DMatMatIntMatMatPointDoubleInt() {
		fail("Not yet implemented");
	}

	public void testSobelMatMatIntIntInt() {
		fail("Not yet implemented");
	}

	public void testSobelMatMatIntIntIntInt() {
		fail("Not yet implemented");
	}

	public void testSobelMatMatIntIntIntIntDouble() {
		fail("Not yet implemented");
	}

	public void testSobelMatMatIntIntIntIntDoubleDouble() {
		fail("Not yet implemented");
	}

	public void testSobelMatMatIntIntIntIntDoubleDoubleInt() {
		fail("Not yet implemented");
	}

	public void testThreshold() {
		fail("Not yet implemented");
	}

	public void testUndistortMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testUndistortMatMatMatMatMat() {
		fail("Not yet implemented");
	}

	public void testWarpAffineMatMatMatSize() {
		fail("Not yet implemented");
	}

	public void testWarpAffineMatMatMatSizeInt() {
		fail("Not yet implemented");
	}

	public void testWarpAffineMatMatMatSizeIntInt() {
		fail("Not yet implemented");
	}

	public void testWarpAffineMatMatMatSizeIntIntScalar() {
		fail("Not yet implemented");
	}

	public void testWarpPerspectiveMatMatMatSize() {
		fail("Not yet implemented");
	}

	public void testWarpPerspectiveMatMatMatSizeInt() {
		fail("Not yet implemented");
	}

	public void testWarpPerspectiveMatMatMatSizeIntInt() {
		fail("Not yet implemented");
	}

	public void testWarpPerspectiveMatMatMatSizeIntIntScalar() {
		fail("Not yet implemented");
	}

	public void testWatershed() {
		fail("Not yet implemented");
	}

}
