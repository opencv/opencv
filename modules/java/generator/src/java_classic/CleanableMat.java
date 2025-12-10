package org.opencv.core;

import java.lang.ref.Cleaner;

public abstract class CleanableMat {

    protected void registerCleaner() {
        // Does nothing, finalize() should be called by JVM
    }

    @Override
    protected void finalize() throws Throwable {
        n_delete(nativeObj);
        super.finalize();
    }

    private static native void n_delete(long nativeObj);

    public long nativeObj;
}
