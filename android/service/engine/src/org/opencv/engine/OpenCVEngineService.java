package org.opencv.engine;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;


public class OpenCVEngineService extends Service
{
    private static final String TAG = "OpenCVEngine/Service";
    private IBinder mEngineInterface;
    private MarketConnector mMarket;
    private BinderConnector mNativeBinder;
    public void onCreate()
    {
        Log.i(TAG, "Service starting");
        super.onCreate();
        Log.i(TAG, "Engine binder component creating");
        mMarket = new MarketConnector(getBaseContext());
        mNativeBinder = new BinderConnector(mMarket);
        mEngineInterface = mNativeBinder.Connect();
        Log.i(TAG, "Service started successfully");
    }

    public IBinder onBind(Intent intent)
    {
        Log.i(TAG, "Service onBind called for intent " + intent.toString());
        return mEngineInterface;
    }
    public boolean onUnbind(Intent intent)
    {
        Log.i(TAG, "Service onUnbind called for intent " + intent.toString());
        return true;
    }
    public void OnDestroy()
    {
        Log.i(TAG, "OpenCV Engine service destruction");
        mNativeBinder.Disconnect();
    }

}
