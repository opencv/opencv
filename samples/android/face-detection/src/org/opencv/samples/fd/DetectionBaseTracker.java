package org.opencv.samples.fd;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

public class DetectionBaseTracker
{	
	public DetectionBaseTracker(String filename, int faceSize)
	{
		mNativeObj = nativeCreateObject(filename, faceSize);
	}
	
	public void start()
	{
		nativeStart(mNativeObj);
	}
	
	public void stop()
	{
		nativeStop(mNativeObj);
	}
	
	public void setMinFaceSize(int faceSize)
	{
		nativeSetFaceSize(mNativeObj, faceSize);
	}
	
	public void detect(Mat imageGray, MatOfRect faces)
	{
		nativeDetect(mNativeObj, imageGray.getNativeObjAddr(), faces.getNativeObjAddr());
	}
	
	public void release()
	{
		nativeDestroyObject(mNativeObj);
		mNativeObj = 0;
	}
	
	protected long mNativeObj = 0;
	
	protected static native long nativeCreateObject(String filename, int faceSize);
	protected static native void nativeDestroyObject(long thiz);
	protected static native void nativeStart(long thiz);
	protected static native void nativeStop(long thiz);
	protected static native void nativeSetFaceSize(long thiz, int faceSize);
	protected static native void nativeDetect(long thiz, long inputImage, long resultMat);
	
	static
	{
		System.loadLibrary("detection_base_tacker");
	}
}
