package org.opencv.core;

import java.lang.ref.Cleaner;

public abstract class CleanableMat {
    // A native memory cleaner for the OpenCV library
    public static Cleaner cleaner = Cleaner.create();

    protected CleanableMat(long obj) {
        if (obj == 0)
            throw new UnsupportedOperationException("Native object address is NULL");

        nativeObj = obj;

        // The n_delete action must not refer to the object being registered. So, do not use nativeObj directly.
        long nativeObjCopy = nativeObj;
        cleaner.register(this, () -> n_delete(nativeObjCopy));
    }

    private static native void n_delete(long nativeObj);

    public final long nativeObj;
}
