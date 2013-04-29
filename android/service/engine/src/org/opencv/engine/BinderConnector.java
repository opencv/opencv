package org.opencv.engine;

import android.os.IBinder;

public class BinderConnector
{
    public BinderConnector(MarketConnector Market) {
        mMarket = Market;
    }

    public boolean Init() {
        boolean result = false;
        if (mIsReady)
            result = Init(mMarket);

        return result;
    }

    public native IBinder Connect();

    public boolean Disconnect()
    {
        if (mIsReady)
            Final();

        return mIsReady;
    }

    private native boolean Init(MarketConnector Market);
    private native void Final();
    private static boolean mIsReady = false;
    private MarketConnector mMarket;

    static {
        try {
            System.loadLibrary("OpenCVEngine");
            System.loadLibrary("OpenCVEngine_jni");
            mIsReady = true;
        }
        catch(UnsatisfiedLinkError e) {
            mIsReady = false;
            e.printStackTrace();
        }
    }

}
