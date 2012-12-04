package org.opencv.engine;

public class OpenCVLibraryInfo {
    public OpenCVLibraryInfo(String packagePath) {
        mNativeObj = open(packagePath + "/libopencv_info.so");
        if (mNativeObj != 0) {
            mPackageName = getPackageName(mNativeObj);
            mLibraryList = getLibraryList(mNativeObj);
            mVersionName = getVersionName(mNativeObj);
            close(mNativeObj);
        }
    }

    public boolean status() {
        return (mNativeObj != 0);
    }

    public String packageName() {
        return mPackageName;
    }

    public String libraryList() {
        return mLibraryList;
    }

    public String versionName() {
        return mVersionName;
    }

    private long mNativeObj;
    private String mPackageName;
    private String mLibraryList;
    private String mVersionName;

    private native long open(String packagePath);
    private native String getPackageName(long obj);
    private native String getLibraryList(long obj);
    private native String getVersionName(long obj);
    private native void close(long obj);
}
