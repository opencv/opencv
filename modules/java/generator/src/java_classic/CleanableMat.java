package org.opencv.core;

import java.lang.ref.Cleaner;

public abstract class CleanableMat implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();
    private Cleaner.Cleanable cleanable;
    
    private static class CleanupTask implements Runnable {
        private final long nativePointer;

        CleanupTask(long pointer) {
            this.nativePointer = pointer;
        }

        @Override
        public void run() {
            if (nativePointer != 0) {
                n_delete(nativePointer);
            }
        }
    }
    
    protected CleanableMat(long obj) {
        if (obj == 0)
            throw new UnsupportedOperationException("Native object address is NULL");

        nativeObj = obj;
        this.cleanable = CLEANER.register(this, new CleanupTask(this.nativeObj));
    }

    @Override
    public void close() {
        if (this.cleanable != null) {
            this.cleanable.clean();
            this.cleanable = null;
        }
    }

    private static native void n_delete(long nativeObj);

    public final long nativeObj;
}
