package org.opencv.engine;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.util.Log;

public class MarketConnector {
    protected static final String OpenCVPackageNamePreffix = "org.opencv.lib";
    private static final String TAG = "OpenCVEngine/MarketConnector";
    protected Context mContext;

    public MarketConnector(Context context) {
        mContext = context;
    }

    public boolean InstallAppFromMarket(String AppID) {
        Log.d(TAG, "Installing app: " + AppID);
        boolean result = true;
        try {
            Uri uri = Uri.parse("market://details?id=" + AppID);
            Intent intent = new Intent(Intent.ACTION_VIEW, uri);
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            mContext.startActivity(intent);
        } catch (Exception e) {
            Log.e(TAG, "Installation failed");
            result = false;
        }
        return result;
    }
}
