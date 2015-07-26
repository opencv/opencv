package org.opencv.samples.tutorial4;

public class NativeGLRenderer {
    static
    {
        System.loadLibrary("JNIrender");
    }
    public static native int initGL();
    public static native void closeGL();
    public static native void drawFrame();
    public static native void changeSize(int width, int height);
}
