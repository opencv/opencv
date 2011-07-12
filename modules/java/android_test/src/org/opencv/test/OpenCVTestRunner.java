package org.opencv.test;

import java.io.FileOutputStream;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.graphics.BitmapFactory;
import android.test.AndroidTestRunner;
import android.test.InstrumentationTestRunner;
import android.util.Log;

/**  
 * This only class is Android specific.
 * The original idea about test order randomization is from marek.defecinski blog.
 * 
 * @see <a href="http://opencv.itseez.com">OpenCV</a>
 */  

public class OpenCVTestRunner extends InstrumentationTestRunner {
	
    public static String LENA_PATH = "/data/data/org.opencv.test/files/lena.jpg";
    public static String CHESS_PATH = "/data/data/org.opencv.test/files/chessboard.jpg";
	private static String TAG = "opencv_test_java";
	
	private AndroidTestRunner androidTestRunner;
	
	static public void Log(String message) {
		Log.e(TAG, message);
	}
    	
    @Override  
    public void onStart() {
    	ExportResourceImage("lena.jpg", R.drawable.lena);
    	ExportResourceImage("chessboard.jpg", R.drawable.chessboard);
		
        //List<TestCase> testCases = androidTestRunner.getTestCases();
        //Collections.shuffle(testCases); //shuffle the tests order
    	
        super.onStart();
    }
    
    @Override  
    protected AndroidTestRunner getAndroidTestRunner() {  
         androidTestRunner = super.getAndroidTestRunner();  
         return androidTestRunner;  
    }
    
	private void ExportResourceImage(String image, int rId) {
		try {
			Bitmap mBitmap = BitmapFactory.decodeResource(this.getContext().getResources(), rId);
			FileOutputStream fos = this.getContext().openFileOutput(image, Context.MODE_WORLD_READABLE);
			mBitmap.compress(CompressFormat.JPEG, 100, fos);
			fos.flush();
			fos.close();
		}
		catch (Exception e) {
		   	Log("Tried to write " + image + ", but: " + e.toString());
		}
	}
}
