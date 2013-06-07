package org.opencv.engine;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.net.Uri;
import android.util.Log;

public class MarketConnector
{
    protected static final String OpenCVPackageNamePreffix = "org.opencv.lib";
    private static final String TAG = "OpenCVEngine/MarketConnector";
    protected Context mContext;

    public MarketConnector(Context context)
    {
        mContext = context;
    }

    public String GetApplicationName(ApplicationInfo info)
    {
        return (String) info.loadLabel(mContext.getPackageManager());
    }

    public boolean InstallAppFromMarket(String AppID)
    {
        boolean result = true;
        try
        {
            Intent intent = new Intent(
                Intent.ACTION_VIEW,
                Uri.parse("market://details?id=" + AppID)
                );
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            mContext.startActivity(intent);
        }
        catch(Exception e)
        {
            result = false;
        }

        return result;
    }

    public boolean RemoveAppFromMarket(String AppID, boolean wait)
    {
        boolean result = true;
    try
        {
            Intent intent = new Intent(
                Intent.ACTION_DELETE,
                Uri.parse("package:" + AppID)
                );
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            if (wait)
            {
                ((Activity)mContext).startActivityForResult(intent, 0);
            }
            else
            {
                mContext.startActivity(intent);
            }
        }
        catch(Exception e)
        {
            e.printStackTrace();
            result = false;
        }

        return result;
    }

    public boolean CheckPackageInstalled(String AppID)
    {
        List<PackageInfo> Packages = mContext.getPackageManager().getInstalledPackages(PackageManager.GET_CONFIGURATIONS);
        Iterator<PackageInfo> it = Packages.iterator();
        while(it.hasNext())
        {
            PackageInfo CurrentPack = it.next();
            if (CurrentPack.packageName == AppID)
            {
                return true;
            }
        }

        return false;
    }

    public PackageInfo[] GetInstalledOpenCVPackages()
    {
        List<PackageInfo> AllPackages = mContext.getPackageManager().getInstalledPackages(PackageManager.GET_CONFIGURATIONS);
        List<PackageInfo> OpenCVPackages = new ArrayList<PackageInfo>();
        try {
            OpenCVPackages.add(mContext.getPackageManager().getPackageInfo("org.opencv.engine", PackageManager.GET_CONFIGURATIONS));
        } catch (NameNotFoundException e) {
            Log.e(TAG, "OpenCV Manager package info was not found!");
            e.printStackTrace();
        }

        Iterator<PackageInfo> it = AllPackages.iterator();
        while(it.hasNext())
        {
            PackageInfo CurrentPack = it.next();
            if (CurrentPack.packageName.contains(OpenCVPackageNamePreffix))
            {
                OpenCVPackages.add(CurrentPack);
            }
        }

        PackageInfo[] OpenCVPackagesArray = new PackageInfo[OpenCVPackages.size()];

        it = OpenCVPackages.iterator();
        int idx = 0;
        while(it.hasNext())
        {
            OpenCVPackagesArray[idx] = it.next();
            idx++;
        }

        return OpenCVPackagesArray;
    }
}
