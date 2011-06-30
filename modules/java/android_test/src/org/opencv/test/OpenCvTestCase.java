package org.opencv.test;

import org.opencv.Mat;
import org.opencv.core;

import android.test.AndroidTestCase;
import android.util.Log;


public class OpenCvTestCase extends AndroidTestCase {
	
	static String TAG = "OpenCV";
	
	static Mat gray0;
	static Mat gray1;
	static Mat gray2;
	static Mat gray3;
	static Mat gray127;
	static Mat gray128;
	static Mat gray255;
	 
	static Mat dst;
	  	 
	@Override
	protected void setUp() throws Exception {
	    //Log.e(TAG, "setUp");		 	 
	    super.setUp();
		 
		gray0   = new Mat(10, 10, Mat.CvType.CV_8UC1); gray0.setTo(0.0);
		gray1   = new Mat(10, 10, Mat.CvType.CV_8UC1); gray1.setTo(1.0);
		gray2   = new Mat(10, 10, Mat.CvType.CV_8UC1); gray2.setTo(2.0);
		gray3   = new Mat(10, 10, Mat.CvType.CV_8UC1); gray3.setTo(3.0);
		gray127 = new Mat(10, 10, Mat.CvType.CV_8UC1); gray127.setTo(127.0);
		gray128 = new Mat(10, 10, Mat.CvType.CV_8UC1); gray128.setTo(128.0);		 
		gray255 = new Mat(10, 10, Mat.CvType.CV_8UC1); gray255.setTo(256.0);
		 
		dst = new Mat(0, 0, Mat.CvType.CV_8UC1);
		assertTrue(dst.empty());
	}
	
	public static void assertMatEqual(Mat m1, Mat m2) {
		assertTrue(MatDifference(m1, m2) == 0.0);
	}
	 
	static public double MatDifference(Mat m1, Mat m2) {
	    Mat cmp = new Mat(0, 0, Mat.CvType.CV_8UC1);
		core.compare(m1, m2, cmp, core.CMP_EQ);
		double num = 100.0 * (1.0 - Double.valueOf(core.countNonZero(cmp)) / Double.valueOf(cmp.rows() * cmp.cols()));
		 
		return num;
	}
	
	 public void test_1(String label) {
		 Log.e(TAG, "================================================");
		 Log.e(TAG, "=============== " + label);
		 Log.e(TAG, "================================================");
	 }
}
