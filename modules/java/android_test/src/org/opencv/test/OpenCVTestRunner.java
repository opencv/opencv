package org.opencv.test;

import java.io.FileOutputStream;
import java.util.Collections;
import java.util.List;
import junit.framework.TestCase;

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
 */  

public class OpenCVTestRunner extends InstrumentationTestRunner {
	
    public static String LENA_PATH = "/data/data/org.opencv.test/files/lena.jpg";
	
	private AndroidTestRunner androidTestRunner;    
	private static String TAG = "opencv_test_java";
	
	static public void Log(String message) {
		Log.e(TAG, message);
	}
    	
    @Override  
    public void onStart() {
    	ExportLena();
		
        List<TestCase> testCases = androidTestRunner.getTestCases();
        Collections.shuffle(testCases); //shuffle the tests order
        super.onStart();
    }
    
    @Override  
    protected AndroidTestRunner getAndroidTestRunner() {  
         androidTestRunner = super.getAndroidTestRunner();  
         return androidTestRunner;  
    }
    
	private void ExportLena() {
		try {
			Bitmap mBitmap = BitmapFactory.decodeResource(this.getContext().getResources(), R.drawable.lena);
			FileOutputStream fos = this.getContext().openFileOutput("lena.jpg", Context.MODE_WORLD_READABLE);
			mBitmap.compress(CompressFormat.JPEG, 100, fos);
			fos.flush();
			fos.close();
		}
		catch (Exception e) {
		   	Log("Tried to write lena.jpg, but: " + e.toString());
		}
	}
}
