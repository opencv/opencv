package org.opencv.core;

public abstract class CleanableMat implements AutoCloseable {

    public long nativeObj;

    protected CleanableMat(long obj) {
        if (obj == 0)
            throw new UnsupportedOperationException("Native object address is NULL");

        nativeObj = obj;
    }

    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }

    @Override
    public synchronized void close() {
        if (nativeObj != 0) {
            n_delete(nativeObj);
            nativeObj = 0;
        }
    }

    private static native void n_delete(long nativeObj);
}
