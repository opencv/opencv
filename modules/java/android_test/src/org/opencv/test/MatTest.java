package org.opencv.test;

import org.opencv.Mat;


public class MatTest extends OpenCvTestCase {	
	 public void test_1() {
		 super.test_1("Mat");
	 }
	 
	 public void test_Can_Create_Gray_Mat() {
		 Mat m = new Mat(1, 1, Mat.CvType.CV_8UC1);
	     assertFalse(m.empty());
	 }
	 
     public void test_Can_Create_RBG_Mat() {
     	 Mat m = new Mat(1, 1, Mat.CvType.CV_8UC3);
     	 assertFalse(m.empty());
     }
     
     public void test_Can_Get_Cols() {
     	 Mat m = new Mat(10, 10, Mat.CvType.CV_8UC1);
         assertEquals(10, m.rows());
     }
     
     public void test_Can_Get_Rows() {
     	 Mat m = new Mat(10, 10, Mat.CvType.CV_8UC1);
         assertEquals(10, m.rows());
     }
}
