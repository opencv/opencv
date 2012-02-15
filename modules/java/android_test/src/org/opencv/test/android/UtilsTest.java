package org.opencv.test.android;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

public class UtilsTest extends OpenCVTestCase {

    public void testBitmapToMat() {
        BitmapFactory.Options opt16 = new BitmapFactory.Options();
        opt16.inPreferredConfig = Bitmap.Config.RGB_565;
        Bitmap bmp16 = BitmapFactory.decodeFile(OpenCVTestRunner.LENA_PATH, opt16);
        Mat m16 = new Mat();
        Utils.bitmapToMat(bmp16, m16);

        BitmapFactory.Options opt32 = new BitmapFactory.Options();
        opt32.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bmp32 = BitmapFactory.decodeFile(OpenCVTestRunner.LENA_PATH, opt32);
        Mat m32 = new Mat();
        Utils.bitmapToMat(bmp32, m32);
        
        assertTrue(m16.rows() == m32.rows() && m16.cols() == m32.cols() && m16.type() == m32.type());
        
        double maxDiff = Core.norm(m16, m32, Core.NORM_INF);
        Log.d("Bmp->Mat", "bmp16->Mat vs bmp32->Mat diff = " + maxDiff);
        assertTrue(maxDiff <= 8 /* 8 == 2^8 / 2^5 */);
    }

    public void testExportResourceContextInt() {
        fail("Not yet implemented");
    }

    public void testExportResourceContextIntString() {
        fail("Not yet implemented");
    }

    public void testLoadResourceContextInt() {
        fail("Not yet implemented");
    }

    public void testLoadResourceContextIntInt() {
        fail("Not yet implemented");
    }

    public void testMatToBitmap() {
    	Mat imgBGR = Highgui.imread( OpenCVTestRunner.LENA_PATH );
    	assertTrue(imgBGR.channels() == 3);
    	
    	Mat m16 = new Mat(imgBGR.rows(), imgBGR.cols(), CvType.CV_8UC4);
    	Mat m32 = new Mat(imgBGR.rows(), imgBGR.cols(), CvType.CV_8UC4);
    	
    	Bitmap bmp16 = Bitmap.createBitmap(imgBGR.cols(), imgBGR.rows(), Bitmap.Config.RGB_565);
    	Bitmap bmp32 = Bitmap.createBitmap(imgBGR.cols(), imgBGR.rows(), Bitmap.Config.ARGB_8888);

    	double maxDiff;
    	Scalar s0 = new Scalar(0);
    	Scalar s255 = Scalar.all(255);
    	

    	// RGBA
    	Mat imgRGBA = new Mat();
        Imgproc.cvtColor(imgBGR, imgRGBA, Imgproc.COLOR_BGR2RGBA);
    	assertTrue(imgRGBA.channels() == 4);
    	
    	bmp16.eraseColor(Color.BLACK); m16.setTo(s0);
    	Utils.matToBitmap(imgRGBA, bmp16); Utils.bitmapToMat(bmp16, m16);
    	maxDiff = Core.norm(imgRGBA, m16, Core.NORM_INF);
    	Log.d("RGBA->bmp16->RGBA", "maxDiff = " + maxDiff);
    	assertTrue(maxDiff <= 8 /* 8 == 2^8 / 2^5 */);
    	
    	bmp32.eraseColor(Color.WHITE); m32.setTo(s255);
    	Utils.matToBitmap(imgRGBA, bmp32); Utils.bitmapToMat(bmp32, m32);
    	maxDiff = Core.norm(imgRGBA, m32, Core.NORM_INF);
    	Log.d("RGBA->bmp32->RGBA", "maxDiff = " + maxDiff);
    	assertTrue(maxDiff == 0);
        

    	// RGB
    	Mat imgRGB = new Mat();
        Imgproc.cvtColor(imgBGR, imgRGB, Imgproc.COLOR_BGR2RGB);
    	assertTrue(imgRGB.channels() == 3);
    	
    	bmp16.eraseColor(Color.BLACK); m16.setTo(s0);
    	Utils.matToBitmap(imgRGB, bmp16); Utils.bitmapToMat(bmp16, m16);
    	maxDiff = Core.norm(imgRGBA, m16, Core.NORM_INF);
    	Log.d("RGB->bmp16->RGBA", "maxDiff = " + maxDiff);
    	assertTrue(maxDiff <= 8 /* 8 == 2^8 / 2^5 */);

    	bmp32.eraseColor(Color.WHITE); m32.setTo(s255);
    	Utils.matToBitmap(imgRGB, bmp32); Utils.bitmapToMat(bmp32, m32);
    	maxDiff = Core.norm(imgRGBA, m32, Core.NORM_INF);
    	Log.d("RGB->bmp32->RGBA", "maxDiff = " + maxDiff);
    	assertTrue(maxDiff == 0);
        

    	// Gray
        Mat imgGray = new Mat();
        Imgproc.cvtColor(imgBGR, imgGray, Imgproc.COLOR_BGR2GRAY);
    	assertTrue(imgGray.channels() == 1);
    	Mat tmp = new Mat();
    	
    	bmp16.eraseColor(Color.BLACK); m16.setTo(s0);
    	Utils.matToBitmap(imgGray, bmp16); Utils.bitmapToMat(bmp16, m16);
    	Core.extractChannel(m16, tmp, 0);
    	maxDiff = Core.norm(imgGray, tmp, Core.NORM_INF);
    	Log.d("Gray->bmp16->RGBA", "maxDiff = " + maxDiff);
    	assertTrue(maxDiff <= 8 /* 8 == 2^8 / 2^5 */);
    	
    	bmp32.eraseColor(Color.WHITE); m32.setTo(s255);
    	Utils.matToBitmap(imgGray, bmp32); Utils.bitmapToMat(bmp32, m32);
    	tmp.setTo(s0);
    	Core.extractChannel(m32, tmp, 0);
    	maxDiff = Core.norm(imgGray, tmp, Core.NORM_INF);
    	Log.d("Gray->bmp32->RGBA", "maxDiff = " + maxDiff);
    	assertTrue(maxDiff == 0);

    }

}
