package org.opencv.engine.manager;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.opencv.engine.HardwareDetector;
import org.opencv.engine.MarketConnector;
import org.opencv.engine.OpenCVEngineInterface;
import org.opencv.engine.OpenCVLibraryInfo;
import org.opencv.engine.R;
import android.annotation.TargetApi;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager.NameNotFoundException;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.Button;
import android.widget.ListView;
import android.widget.SimpleAdapter;
import android.widget.TextView;
import android.widget.Toast;

public class ManagerActivity extends Activity
{
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!HardwareDetector.mIsReady) {
            Log.e(TAG, "Cannot initialize native part of OpenCV Manager!");

            AlertDialog dialog = new AlertDialog.Builder(this).create();

            dialog.setTitle("OpenCV Manager Error");
            dialog.setMessage("OpenCV Manager is incompatible with this device. Please replace it with an appropriate package.");
            dialog.setCancelable(false);
            dialog.setButton("OK", new DialogInterface.OnClickListener() {

                public void onClick(DialogInterface dialog, int which) {
                    finish();
                }
            });

            dialog.show();
            return;
        }

        setContentView(R.layout.main);

        TextView OsVersionView = (TextView)findViewById(R.id.OsVersionValue);
        OsVersionView.setText(Build.VERSION.CODENAME + " (" + Build.VERSION.RELEASE + "), API " + Build.VERSION.SDK_INT);

        try {
            PackageInfo packageInfo = getPackageManager().getPackageInfo(this.getPackageName(), 0);
            ManagerVersion = packageInfo.versionName;
        } catch (NameNotFoundException e) {
            ManagerVersion = "N/A";
            e.printStackTrace();
        }

        mInstalledPackageView = (ListView)findViewById(R.id.InstalledPackageList);

        mMarket = new MarketConnector(this);

        mInstalledPacksAdapter = new PackageListAdapter(
                this,
                mListViewItems,
                R.layout.info,
                new String[] {"Name", "Version", "Hardware", "Activity"},
                new int[] {R.id.InfoName,R.id.InfoVersion, R.id.InfoHardware}
                );

        mInstalledPackageView.setAdapter(mInstalledPacksAdapter);

        TextView HardwarePlatformView = (TextView)findViewById(R.id.HardwareValue);
        int Platform = HardwareDetector.DetectKnownPlatforms();
        int CpuId = HardwareDetector.GetCpuID();

        if (HardwareDetector.PLATFORM_UNKNOWN != Platform)
        {
            if (HardwareDetector.PLATFORM_TEGRA == Platform)
            {
                HardwarePlatformView.setText("Tegra");
            }
            else if (HardwareDetector.PLATFORM_TEGRA2 == Platform)
            {
                HardwarePlatformView.setText("Tegra 2");
            }
            else if (HardwareDetector.PLATFORM_TEGRA3 == Platform)
            {
                HardwarePlatformView.setText("Tegra 3");
            }
            else if (HardwareDetector.PLATFORM_TEGRA4i == Platform)
            {
                HardwarePlatformView.setText("Tegra 4i");
            }
            else if (HardwareDetector.PLATFORM_TEGRA4 == Platform)
            {
                HardwarePlatformView.setText("Tegra 4");
            }
            else
            {
                HardwarePlatformView.setText("Tegra 5");
            }
        }
        else
        {
            if ((CpuId & HardwareDetector.ARCH_X86) == HardwareDetector.ARCH_X86)
            {
                HardwarePlatformView.setText("x86 " + JoinIntelFeatures(CpuId));
            }
            else if ((CpuId & HardwareDetector.ARCH_X64) == HardwareDetector.ARCH_X64)
            {
                HardwarePlatformView.setText("x64 " + JoinIntelFeatures(CpuId));
            }
            else if ((CpuId & HardwareDetector.ARCH_ARMv5) == HardwareDetector.ARCH_ARMv5)
            {
                HardwarePlatformView.setText("ARM v5 " + JoinArmFeatures(CpuId));
            }
            else if ((CpuId & HardwareDetector.ARCH_ARMv6) == HardwareDetector.ARCH_ARMv6)
            {
                HardwarePlatformView.setText("ARM v6 " + JoinArmFeatures(CpuId));
            }
            else if ((CpuId & HardwareDetector.ARCH_ARMv7) == HardwareDetector.ARCH_ARMv7)
            {
                HardwarePlatformView.setText("ARM v7 " + JoinArmFeatures(CpuId));
            }
            else if ((CpuId & HardwareDetector.ARCH_ARMv8) == HardwareDetector.ARCH_ARMv8)
            {
                HardwarePlatformView.setText("ARM v8 " + JoinArmFeatures(CpuId));
            }
            else if ((CpuId & HardwareDetector.ARCH_MIPS) == HardwareDetector.ARCH_MIPS)
            {
                HardwarePlatformView.setText("MIPS");
            }
            else
            {
                HardwarePlatformView.setText("not detected");
            }
        }

        mUpdateEngineButton = (Button)findViewById(R.id.CheckEngineUpdate);
        mUpdateEngineButton.setOnClickListener(new OnClickListener() {

            public void onClick(View v) {
                if (!mMarket.InstallAppFromMarket("org.opencv.engine"))
                {
                    Toast toast = Toast.makeText(getApplicationContext(), "Google Play is not avaliable", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });

        mActionDialog = new AlertDialog.Builder(this).create();

        mActionDialog.setTitle("Choose action");
        mActionDialog.setButton("Update", new DialogInterface.OnClickListener() {

            public void onClick(DialogInterface dialog, int which) {
                int index = (Integer)mInstalledPackageView.getTag();
                if (!mMarket.InstallAppFromMarket(mInstalledPackageInfo[index].packageName))
                {
                    Toast toast = Toast.makeText(getApplicationContext(), "Google Play is not avaliable", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });

        mActionDialog.setButton3("Remove", new DialogInterface.OnClickListener() {

            public void onClick(DialogInterface dialog, int which) {
                int index = (Integer)mInstalledPackageView.getTag();
                if (!mMarket.RemoveAppFromMarket(mInstalledPackageInfo[index].packageName, true))
                {
                    Toast toast = Toast.makeText(getApplicationContext(), "Google Play is not avaliable", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });

        mActionDialog.setButton2("Return", new DialogInterface.OnClickListener() {

            public void onClick(DialogInterface dialog, int which) {
                // nothing
            }
        });

        mInstalledPackageView.setOnItemClickListener(new OnItemClickListener() {

            public void onItemClick(AdapterView<?> adapter, View view, int position, long id) {
                //if (!mListViewItems.get((int) id).get("Name").equals("Built-in OpenCV library"));
                if (!mInstalledPackageInfo[(int) id].packageName.equals("org.opencv.engine"))
                {
                    mInstalledPackageView.setTag(Integer.valueOf((int)id));
                    mActionDialog.show();
                }
            }
        });

        mPackageChangeReciever = new BroadcastReceiver() {

            @Override
            public void onReceive(Context context, Intent intent) {
                Log.d("OpenCVManager/Reciever", "Bradcast message " + intent.getAction() + " reciever");
                Log.d("OpenCVManager/Reciever", "Filling package list on broadcast message");
                if (!bindService(new Intent("org.opencv.engine.BIND"), new OpenCVEngineServiceConnection(), Context.BIND_AUTO_CREATE))
                {
                    TextView EngineVersionView = (TextView)findViewById(R.id.EngineVersionValue);
                    EngineVersionView.setText("not avaliable");
                }
            }
        };

        IntentFilter filter = new IntentFilter();
        filter.addAction(Intent.ACTION_PACKAGE_ADDED);
        filter.addAction(Intent.ACTION_PACKAGE_CHANGED);
        filter.addAction(Intent.ACTION_PACKAGE_INSTALL);
        filter.addAction(Intent.ACTION_PACKAGE_REMOVED);
        filter.addAction(Intent.ACTION_PACKAGE_REPLACED);

        registerReceiver(mPackageChangeReciever, filter);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mPackageChangeReciever != null)
            unregisterReceiver(mPackageChangeReciever);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (HardwareDetector.mIsReady) {
            Log.d(TAG, "Filling package list on resume");
            OpenCVEngineServiceConnection connection = new OpenCVEngineServiceConnection();
            if (!bindService(new Intent("org.opencv.engine.BIND"), connection, Context.BIND_AUTO_CREATE)) {
                Log.e(TAG, "Cannot bind to OpenCV Manager service!");
                TextView EngineVersionView = (TextView)findViewById(R.id.EngineVersionValue);
                if (EngineVersionView != null)
                    EngineVersionView.setText("not avaliable");
                unbindService(connection);
            }
        }
    }

    protected SimpleAdapter mInstalledPacksAdapter;
    protected ListView mInstalledPackageView;
    protected Button mUpdateEngineButton;
    protected PackageInfo[] mInstalledPackageInfo;
    protected final ArrayList<HashMap<String,String>> mListViewItems = new ArrayList<HashMap<String,String>>();
    protected static final String TAG = "OpenCV_Manager/Activity";
    protected MarketConnector mMarket;
    protected AlertDialog mActionDialog;
    protected HashMap<String,String> mActivePackageMap = new HashMap<String, String>();
    protected int ManagerApiLevel = 0;
    protected String ManagerVersion;

    protected BroadcastReceiver mPackageChangeReciever = null;

    protected class OpenCVEngineServiceConnection implements ServiceConnection
    {
        public void onServiceDisconnected(ComponentName name) {
        }

        public void onServiceConnected(ComponentName name, IBinder service) {
            OpenCVEngineInterface EngineService = OpenCVEngineInterface.Stub.asInterface(service);
            if (EngineService == null) {
                Log.e(TAG, "Cannot connect to OpenCV Manager Service!");
                unbindService(this);
                return;
            }

            try {
                ManagerApiLevel = EngineService.getEngineVersion();
            } catch (RemoteException e) {
                e.printStackTrace();
            }

            TextView EngineVersionView = (TextView)findViewById(R.id.EngineVersionValue);
            EngineVersionView.setText(ManagerVersion);

            try {
                String path = EngineService.getLibPathByVersion("2.4");
                Log.d(TAG, "2.4 -> " + path);
                mActivePackageMap.put("24", path);
                path = EngineService.getLibPathByVersion("2.5");
                Log.d(TAG, "2.5 -> " + path);
                mActivePackageMap.put("25", path);
            } catch (RemoteException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }

            Log.d(TAG, "Filling package list on service connection");
            FillPackageList();

            unbindService(this);
        }
    };

    @TargetApi(Build.VERSION_CODES.GINGERBREAD)
    synchronized protected void FillPackageList()
    {
        synchronized (mListViewItems) {
            mInstalledPackageInfo = mMarket.GetInstalledOpenCVPackages();
            mListViewItems.clear();

            int RealPackageCount = mInstalledPackageInfo.length;
            for (int i = 0; i < RealPackageCount; i++)
            {
                if (mInstalledPackageInfo[i] == null)
                    break;

                // Convert to Items for package list view
                HashMap<String,String> temp = new HashMap<String,String>();

                String HardwareName = "";
                String NativeLibDir = "";
                String OpenCVersion = "";

                String PublicName = mMarket.GetApplicationName(mInstalledPackageInfo[i].applicationInfo);
                String PackageName = mInstalledPackageInfo[i].packageName;
                String VersionName = mInstalledPackageInfo[i].versionName;

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD)
                    NativeLibDir = mInstalledPackageInfo[i].applicationInfo.nativeLibraryDir;
                else
                    NativeLibDir = "/data/data/" + mInstalledPackageInfo[i].packageName + "/lib";

                if (PackageName.equals("org.opencv.engine"))
                {
                    OpenCVLibraryInfo NativeInfo = new OpenCVLibraryInfo(NativeLibDir);
                    if (NativeInfo.status())
                    {
                        PublicName = "Built-in OpenCV library";
                        PackageName = NativeInfo.packageName();
                        VersionName = NativeInfo.versionName();
                    }
                    else
                    {
                        mInstalledPackageInfo[i] = mInstalledPackageInfo[RealPackageCount-1];
                        mInstalledPackageInfo[RealPackageCount-1] = null;
                        RealPackageCount--;
                        i--;
                        continue;
                    }
                }

                int idx = 0;
                Log.d(TAG, PackageName);
                StringTokenizer tokenizer = new StringTokenizer(PackageName, "_");
                while (tokenizer.hasMoreTokens())
                {
                    if (idx == 1)
                    {
                        // version of OpenCV
                        OpenCVersion = tokenizer.nextToken().substring(1);
                    }
                    else if (idx >= 2)
                    {
                        // hardware options
                        HardwareName += tokenizer.nextToken() + " ";
                    }
                    else
                    {
                        tokenizer.nextToken();
                    }
                    idx++;
                }

                String ActivePackagePath;
                String Tags = null;
                ActivePackagePath = mActivePackageMap.get(OpenCVersion);
                Log.d(TAG, OpenCVersion + " -> " + ActivePackagePath);

                if (null != ActivePackagePath)
                {
                    if (ActivePackagePath.equals(NativeLibDir))
                    {
                        temp.put("Activity", "y");
                        Tags = "active";
                    }
                    else
                    {
                        temp.put("Activity", "n");
                        if (!PublicName.equals("Built-in OpenCV library"))
                            Tags = "safe to remove";
                    }
                }
                else
                {
                    temp.put("Activity", "n");
                }

                temp.put("Version", NormalizeVersion(OpenCVersion, VersionName));
                // HACK: OpenCV Manager for Armv7-a Neon already has Tegra3 optimizations
                // that is enabled on proper hardware
                if (HardwareDetector.DetectKnownPlatforms() >= HardwareDetector.PLATFORM_TEGRA3 &&
                  HardwareName.equals("armv7a neon ") &&  Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD)
                {
                    temp.put("Hardware", "Tegra");
                    if (Tags == null)
                    {
                        Tags = "optimized";
                    }
                    else
                    {
                        Tags = Tags + ", optimized";
                    }
                }
                else
                {
                    temp.put("Hardware", HardwareName);
                }

                if (Tags != null)
                    PublicName = PublicName + " (" + Tags + ")";

                temp.put("Name", PublicName);

                mListViewItems.add(temp);
            }

            mInstalledPacksAdapter.notifyDataSetChanged();
        }
    }

    protected String NormalizeVersion(String OpenCVersion, String PackageVersion)
    {
        if (OpenCVersion == null || PackageVersion == null)
            return "unknown";

        String[] revisions = PackageVersion.split("\\.");

        if (revisions.length <= 1 || OpenCVersion.length() == 0)
            return "unknown";
        else
            if (revisions.length == 2)
                // the 2nd digit is revision
                return OpenCVersion.substring(0,  OpenCVersion.length()-1) + "." +
                    OpenCVersion.toCharArray()[OpenCVersion.length()-1] + "." +
                    revisions[0] + " rev " + revisions[1];
            else
                // the 2nd digit is part of library version
                // the 3rd digit is revision
                return OpenCVersion.substring(0,  OpenCVersion.length()-1) + "." +
                    OpenCVersion.toCharArray()[OpenCVersion.length()-1] + "." +
                    revisions[0] + "." + revisions[1] + " rev " + revisions[2];
    }

    protected String ConvertPackageName(String Name, String Version)
    {
        return Name + " rev " + Version;
    }

    protected String JoinIntelFeatures(int features)
    {
        // TODO: update if package will be published
        return "";
    }

    protected String JoinArmFeatures(int features)
    {
        // TODO: update if package will be published
        if ((features & HardwareDetector.FEATURES_HAS_NEON) == HardwareDetector.FEATURES_HAS_NEON)
        {
            if ((features & HardwareDetector.FEATURES_HAS_VFPv4) == HardwareDetector.FEATURES_HAS_VFPv4)
            {
                return "with Neon and VFPv4";
            }
            else
            {
                return "with Neon";
            }
        }
        else if ((features & HardwareDetector.FEATURES_HAS_VFPv3) == HardwareDetector.FEATURES_HAS_VFPv3)
        {
            return "with VFP v3";
        }
        else if ((features & HardwareDetector.FEATURES_HAS_VFPv3d16) == HardwareDetector.FEATURES_HAS_VFPv3d16)
        {
            return "with VFP v3d16";
        }
        else
        {
            return "";
        }
    }
}
