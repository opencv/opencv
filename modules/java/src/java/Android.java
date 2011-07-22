package org.opencv;

import org.opencv.core.Mat;
import android.graphics.Bitmap;

public class Android {

	public static Mat BitmapToMat(Bitmap b) {
		return new Mat( nBitmapToMat(b) );
	}
	
	public static boolean MatToBitmap(Mat m, Bitmap b) {
		return nMatToBitmap(m.nativeObj, b);	
	}

	// native stuff
	static { System.loadLibrary("opencv_java"); }
	private static native long nBitmapToMat(Bitmap b);
	private static native boolean nMatToBitmap(long m, Bitmap b);
}
