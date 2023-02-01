package org.opencv.samples.tutorial4;

public class NativePart {
    static
    {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("JNIpart");
    }

    public static final int PROCESSING_MODE_NO_PROCESSING = 0;
    public static final int PROCESSING_MODE_CPU = 1;
    public static final int PROCESSING_MODE_OCL_DIRECT = 2;
    public static final int PROCESSING_MODE_OCL_OCV = 3;

    public static native int initCL();
    public static native void closeCL();
    public static native void processFrame(int tex1, int tex2, int w, int h, int mode);
}
