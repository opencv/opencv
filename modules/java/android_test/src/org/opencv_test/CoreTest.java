package org.opencv_test;

import org.opencv.core;


public class CoreTest extends OpenCvTestCase {	 
	 public void test_1() {
		 super.test_1("CORE");
	 }

	 public void test_Can_Call_add() {
		 core.add(gray128, gray128, dst);
		 assertMatEqual(gray255, dst);
	 }
	 
	 public void test_Can_Call_absdiff() {
		 core.absdiff(gray128, gray255, dst);		 
		 assertMatEqual(gray127, dst);
	 }
	 
	 public void test_Can_Call_bitwise_and() {
		 core.bitwise_and(gray3, gray2, dst); 
		 assertMatEqual(gray2, dst);
	 }
}
