package org.opencv.android;

import android.content.Context;
import android.util.Log;

public class OpenCVLoader
{	
	public static final int Success = 0;
	public static final int NoService = 1;
	public static final int RestartRequired = 2;
	public static final int MarketError = 3;
	public static final int InitFailed = 0xff;
	
	public static int initStatic()
	{
		int result;
		
		try
		{
			System.loadLibrary("opencv_java");
			result = Success;
		}
		catch(Exception e)
		{
			Log.e(TAG, "OpenCV error: Cannot load native library");
			e.printStackTrace();
			result = InitFailed;
		}
		
		return result;
	}
	
	public static int initAsync(String Version, Context AppContext, 
			LoaderCallbackInterface Callback)
	{
		return AsyncServiceHelper.initOpenCV(Version, AppContext, Callback);
	}
	
	private static final String TAG = "OpenCV/Helper";
}
