package org.opencv.engine.manager;

public class OpenCVPackageInfo
{
	public OpenCVPackageInfo(String PackageName, String PackageInstallPath, String RevName)
	{
		
	}
	
	protected long mNativeObject = 0;
	
	protected static native long nativeConstructor(String PackageName, String PackageInstallPath, String RevName);
	protected static native void nativeDestructor(long nativeAddress);
	protected static native String nativeGetVersion(long thiz);
	protected static native String nativeGetPlatfrom(long thiz);
}
