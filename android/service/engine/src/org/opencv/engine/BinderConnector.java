package org.opencv.engine;

import android.os.IBinder;

public class BinderConnector
{
	public BinderConnector(MarketConnector Market)
	{
		Init(Market);
	}
	public native IBinder Connect();
	public boolean Disconnect()
	{
		Final();
		return true;
	}
	
	static
	{	 
		System.loadLibrary("OpenCVEngine");
		System.loadLibrary("OpenCVEngine_jni");
	}	
	
	private native boolean Init(MarketConnector Market);
	public native void Final();	
}
