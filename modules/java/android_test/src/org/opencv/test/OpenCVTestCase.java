package org.opencv.test;

import junit.framework.TestCase;

import org.opencv.CvType;
import org.opencv.Mat;
import org.opencv.Scalar;
import org.opencv.core;
import org.opencv.highgui;


public class OpenCVTestCase extends TestCase {
       
    static int matSize = 10;
    
    static Mat dst;
    
    //Naming notation: <channels info>_[depth]_[dimensions]_value
    //examples: gray0   - single channel 8U 2d Mat filled with 0
    //          grayRnd - single channel 8U 2d Mat filled with random numbers
    //          gray0_32f_1d - refactor ;)

    static Mat gray0;
    static Mat gray1;
    static Mat gray2;
    static Mat gray3;
    static Mat gray127;
    static Mat gray128;
    static Mat gray255;    
    static Mat grayRnd;
    
    static Mat gray0_32f;
    static Mat gray255_32f;        
    static Mat grayE_32f;
    static Mat grayRnd_32f;    
      
    static Mat gray0_32f_1d;
    
    static Mat gray0_64f;    
    static Mat gray0_64f_1d;
       
    static Mat rgba0;
    static Mat rgba128;
    
    static Mat rgbLena;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        dst = new Mat();
        assertTrue(dst.empty());

        gray0 = new Mat(matSize, matSize, CvType.CV_8U); gray0.setTo(new Scalar(0.0));
        gray1 = new Mat(matSize, matSize, CvType.CV_8U); gray1.setTo(new Scalar(1.0));
        gray2 = new Mat(matSize, matSize, CvType.CV_8U); gray2.setTo(new Scalar(2.0));
        gray3 = new Mat(matSize, matSize, CvType.CV_8U); gray3.setTo(new Scalar(3.0));
        gray127 = new Mat(matSize, matSize, CvType.CV_8U); gray127.setTo(new Scalar(127.0));
        gray128 = new Mat(matSize, matSize, CvType.CV_8U); gray128.setTo(new Scalar(128.0));
        gray255 = new Mat(matSize, matSize, CvType.CV_8U); gray255.setTo(new Scalar(255.0));
        
        Mat low  = new Mat(1, 1, CvType.CV_16UC1, new Scalar(0));
        Mat high = new Mat(1, 1, CvType.CV_16UC1, new Scalar(256));
        grayRnd = new Mat(matSize, matSize, CvType.CV_8U); core.randu(grayRnd, low, high);
        
        gray0_32f = new Mat(matSize, matSize, CvType.CV_32F); gray0_32f.setTo(new Scalar(0.0));
        gray255_32f = new Mat(matSize, matSize, CvType.CV_32F); gray255_32f.setTo(new Scalar(255.0));
        grayE_32f = new Mat(matSize, matSize, CvType.CV_32F); grayE_32f = Mat.eye(matSize, matSize, CvType.CV_32FC1);
        grayRnd_32f = new Mat(matSize, matSize, CvType.CV_32F); core.randu(grayRnd_32f, low, high);
        
        gray0_32f_1d = new Mat(1, matSize, CvType.CV_32F); gray0_32f_1d.setTo(new Scalar(0.0));
        
        gray0_64f = new Mat(matSize, matSize, CvType.CV_64F); gray0_64f.setTo(new Scalar(0.0));
        gray0_64f_1d = new Mat(1, matSize, CvType.CV_64F); gray0_64f_1d.setTo(new Scalar(0.0));

        rgba0 = new Mat(matSize, matSize, CvType.CV_8UC4); rgba0.setTo(Scalar.all(0));
        rgba128 = new Mat(matSize, matSize, CvType.CV_8UC4); rgba128.setTo(Scalar.all(128));
        
        rgbLena = highgui.imread(OpenCVTestRunner.LENA_PATH);
    }

    public static void assertMatEqual(Mat m1, Mat m2) {
        assertTrue(CalcPercentageOfDifference(m1, m2) == 0.0);
    }

    static public double CalcPercentageOfDifference(Mat m1, Mat m2) {
        Mat cmp = new Mat(0, 0, CvType.CV_8U);
        core.compare(m1, m2, cmp, core.CMP_EQ);
        double num = 100.0 * 
            (1.0 - Double.valueOf(core.countNonZero(cmp)) / Double.valueOf(cmp.rows() * cmp.cols()));

        return num;
    }

    public void test_1(String label) {
    	OpenCVTestRunner.Log("================================================");
    	OpenCVTestRunner.Log("=============== " + label);
    }
}
