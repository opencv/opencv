package org.opencv.test;

import org.opencv.Mat;
import org.opencv.core;

import android.test.AndroidTestCase;
import android.util.Log;

public class OpenCVTestCase extends AndroidTestCase {

    static String TAG = "OpenCV_JavaAPI_Tests";
    static int matSize = 10;

    static Mat gray0;
    static Mat gray1;
    static Mat gray2;
    static Mat gray3;
    static Mat gray127;
    static Mat gray128;
    static Mat gray255;
    
    static Mat grayRnd;    
    static Mat grayRnd_32f;
    
    static Mat grayE_32f;
    
    static Mat gray0_32f;    
    static Mat gray0_32f_1d;
    
    static Mat gray0_64f;    
    static Mat gray0_64f_1d;
    
    static Mat rgba0;
    static Mat rgba128;    

    static Mat dst_gray;
    static Mat dst_gray_32f;

    @Override
    protected void setUp() throws Exception {
        // Log.e(TAG, "setUp");
        super.setUp();

        gray0 = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); gray0.setTo(0.0);
        gray1 = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); gray1.setTo(1.0);
        gray2 = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); gray2.setTo(2.0);
        gray3 = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); gray3.setTo(3.0);
        gray127 = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); gray127.setTo(127.0);
        gray128 = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); gray128.setTo(128.0);
        gray255 = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); gray255.setTo(256.0);
        
        Mat low  = new Mat(1, 1, Mat.CvType.CV_16UC1); low.setTo(0);
        Mat high = new Mat(1, 1, Mat.CvType.CV_16UC1); high.setTo(256);
        grayRnd = new Mat(matSize, matSize, Mat.CvType.CV_8UC1); core.randu(grayRnd, low, high);
        grayRnd_32f = new Mat(matSize, matSize, Mat.CvType.CV_32FC1); core.randu(grayRnd_32f, low, high);
        
        grayE_32f = new Mat(matSize, matSize, Mat.CvType.CV_32FC1); grayE_32f = Mat.eye(matSize, matSize, Mat.CvType.CV_32FC1);
        
        gray0_32f = new Mat(matSize, matSize, Mat.CvType.CV_32FC1); gray0_32f.setTo(0.0);
        gray0_32f_1d = new Mat(1, matSize, Mat.CvType.CV_32FC1); gray0_32f_1d.setTo(0.0);
        
        gray0_64f = new Mat(matSize, matSize, Mat.CvType.CV_64FC1); gray0_64f.setTo(0.0);
        gray0_64f_1d = new Mat(1, matSize, Mat.CvType.CV_64FC1); gray0_64f_1d.setTo(0.0);
        
        rgba0 = new Mat(matSize, matSize, Mat.CvType.CV_8UC4); rgba0.setTo(0, 0, 0, 0);
        rgba128 = new Mat(matSize, matSize, Mat.CvType.CV_8UC4); rgba128.setTo(128, 128, 128, 128);        

        dst_gray = new Mat(0, 0, Mat.CvType.CV_8UC1);
        assertTrue(dst_gray.empty());
        dst_gray_32f = new Mat(0, 0, Mat.CvType.CV_32FC1);
        assertTrue(dst_gray_32f.empty());
    }

    public static void assertMatEqual(Mat m1, Mat m2) {
        assertTrue(CalcPercentageOfDifference(m1, m2) == 0.0);
    }

    static public double CalcPercentageOfDifference(Mat m1, Mat m2) {
        Mat cmp = new Mat(0, 0, Mat.CvType.CV_8UC1);
        core.compare(m1, m2, cmp, core.CMP_EQ);
        double num = 100.0 * 
            (1.0 - Double.valueOf(core.countNonZero(cmp)) / Double.valueOf(cmp.rows() * cmp.cols()));

        return num;
    }

    public void test_1(String label) {
        Log.e(TAG, "================================================");
        Log.e(TAG, "=============== " + label);
        Log.e(TAG, "================================================");
    }
}
