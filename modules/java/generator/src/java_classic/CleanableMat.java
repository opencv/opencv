package org.opencv.core;

public abstract class CleanableMat {

    protected CleanableMat(long obj) {
        if (obj == 0)
            throw new UnsupportedOperationException("Native object address is NULL");

        nativeObj = obj;
    }

    @Override
    protected void finalize() throws Throwable {
        n_delete(nativeObj);
        super.finalize();
    }

    private static native void n_delete(long nativeObj);

    public final long nativeObj;
}
