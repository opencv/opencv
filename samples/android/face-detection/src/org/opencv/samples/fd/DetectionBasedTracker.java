package org.opencv.samples.fd;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

public class DetectionBasedTracker
{	
	public DetectionBasedTracker(String cascadeName, int minFaceSize)
	{
		mMainDetector = nativeCreateDetector(cascadeName, minFaceSize);
		mTrackingDetector = nativeCreateDetector(cascadeName, minFaceSize);
		mNativeObj = nativeCreateTracker(mMainDetector, mTrackingDetector);
	}
	
	public void start()
	{
		nativeStart(mNativeObj);
	}
	
	public void stop()
	{
		nativeStop(mNativeObj);
	}
	
	public void setMinFaceSize(int size)
	{
		nativeSetFaceSize(mMainDetector, size);
		nativeSetFaceSize(mTrackingDetector, size);
	}
	
	public void detect(Mat imageGray, MatOfRect faces)
	{
		nativeDetect(mNativeObj, imageGray.getNativeObjAddr(), faces.getNativeObjAddr());
	}
	
	public void release()
	{
		nativeDestroyTracker(mNativeObj);
		nativeDestroyDetector(mMainDetector);
		nativeDestroyDetector(mTrackingDetector);
		mNativeObj = 0;
		mMainDetector = 0;
		mTrackingDetector = 0;
	}
	
	private long mNativeObj = 0;
	private long mMainDetector = 0;
	private long mTrackingDetector = 0;
	
	private static native long nativeCreateDetector(String cascadeName, int minFaceSize);
	private static native long nativeCreateTracker(long mainDetector, long trackingDetector);
	private static native void nativeDestroyTracker(long tracker);
	private static native void nativeDestroyDetector(long detector);
	private static native void nativeStart(long thiz);
	private static native void nativeStop(long thiz);
	private static native void nativeSetFaceSize(long detector, int size);
	private static native void nativeDetect(long thiz, long inputImage, long faces);
	
	static
	{
		System.loadLibrary("detection_based_tacker");
	}
}
