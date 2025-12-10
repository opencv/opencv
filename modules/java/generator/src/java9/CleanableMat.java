package org.opencv.core;

import java.lang.ref.Cleaner;

public abstract class CleanableMat {
    // A native memory cleaner for the OpenCV library
    public static Cleaner cleaner = Cleaner.create();

    protected void registerCleaner() {
        // The n_delete action must not refer to the object being registered. So, do not use nativeObj directly.
        long nativeObjCopy = nativeObj;
        cleaner.register(this, () -> n_delete(nativeObjCopy));
    }

    private static native void n_delete(long nativeObj);

    public long nativeObj;
}
