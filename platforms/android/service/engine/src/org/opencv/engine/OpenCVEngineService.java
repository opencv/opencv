package org.opencv.engine;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;

public class OpenCVEngineService extends Service
{
    private static final String TAG = "OpenCVEngine/Service";
    private IBinder mEngineInterface = null;
    private MarketConnector mMarket;
    private BinderConnector mNativeBinder;

    public void onCreate() {
        Log.i(TAG, "Service starting");
        super.onCreate();
        Log.i(TAG, "Engine binder component creating");
        mMarket = new MarketConnector(getBaseContext());
        mNativeBinder = new BinderConnector(mMarket);
        if (mNativeBinder.Init()) {
            mEngineInterface = mNativeBinder.Connect();
            Log.i(TAG, "Service started successfully");
        } else {
            Log.e(TAG, "Cannot initialize native part of OpenCV Manager!");
            Log.e(TAG, "Using stub instead");

            mEngineInterface = new OpenCVEngineInterface.Stub() {

                @Override
                public boolean installVersion(String version) throws RemoteException {
                    // TODO Auto-generated method stub
                    return false;
                }

                @Override
                public String getLibraryList(String version) throws RemoteException {
                    // TODO Auto-generated method stub
                    return null;
                }

                @Override
                public String getLibPathByVersion(String version) throws RemoteException {
                    // TODO Auto-generated method stub
                    return null;
                }

                @Override
                public int getEngineVersion() throws RemoteException {
                    return -1;
                }
            };
        }
    }

    public IBinder onBind(Intent intent) {
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
