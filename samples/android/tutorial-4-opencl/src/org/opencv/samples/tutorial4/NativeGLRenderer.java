package org.opencv.samples.tutorial4;

public class NativeGLRenderer {
    static
    {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("JNIrender");
    }

    public static final int PROCESSING_MODE_CPU = 1;
    public static final int PROCESSING_MODE_OCL_DIRECT = 2;
    public static final int PROCESSING_MODE_OCL_OCV = 3;

    public static native int initGL();
    public static native void closeGL();
    public static native void drawFrame();
    public static native void changeSize(int width, int height);
    public static native void setProcessingMode(int mode);
}
