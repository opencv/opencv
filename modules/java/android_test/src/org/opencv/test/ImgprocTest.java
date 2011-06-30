package org.opencv.test;

import org.opencv.Size;
import org.opencv.imgproc;


public class ImgprocTest extends OpenCvTestCase {
	 public void test_1() {
		 super.test_1("IMGPROC");
	 }
	 
	 //FIXME: this test crashes
	 //public void test_Can_Call_accumulate() {
	 //	 dst = new Mat(gray1.rows(), gray1.cols(), Mat.CvType.CV_32FC1);
 	 //	 imgproc.accumulate(gray1, dst);
	 //	 assertMatEqual(gray1, dst);
	 //}
	 
	 public void test_blur() {
		 Size sz = new Size(3, 3);
		 
		 imgproc.blur(gray0, dst, sz);
		 assertMatEqual(gray0, dst);
		 
		 imgproc.blur(gray255, dst, sz);
		 assertMatEqual(gray255, dst);
	 }
	 
	 public void test_boxFilter() {
		 Size sz = new Size(3, 3);
		 imgproc.boxFilter(gray0, dst, 8, sz);
		 assertMatEqual(gray0, dst);
	 }
}
