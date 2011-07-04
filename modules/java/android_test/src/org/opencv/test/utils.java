package org.opencv.test;

import java.io.FileOutputStream;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Bitmap.CompressFormat;
import android.test.AndroidTestCase;

import android.util.Log;

public class utils extends AndroidTestCase {
	
	static String TAG = "opencv_test_java";
	
	static public void Log(String message) {
		Log.e(TAG, message);
	}
	
	public void ExportLena() {
        //TODO: can we run this code just once, not for every test case?
		try {
            Bitmap mBitmap = BitmapFactory.decodeResource(this.getContext().getResources(), R.drawable.lena);
            FileOutputStream fos = this.getContext().openFileOutput("lena.jpg", Context.MODE_WORLD_READABLE);
            mBitmap.compress(CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();
		}
		catch (Exception e) {
		   Log.e(TAG, "Tried to write lena.jpg, but: " + e.toString());
		}
	}	

}
