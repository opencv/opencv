package org.opencv.engine;

import android.app.Service;
import android.content.Intent;
import android.content.pm.PackageManager.NameNotFoundException;
import android.content.res.XmlResourceParser;
import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;
import android.text.TextUtils;
import java.io.File;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.xmlpull.v1.XmlPullParser;

public class OpenCVEngineService extends Service {
    private static final String TAG = "OpenCVEngine/Service";
    private IBinder mEngineInterface = null;
    private List<LibVariant> variants = new ArrayList<LibVariant>();

    private class LibVariant {
        public String version;
        public List<String> files;

        public void parseFile(XmlResourceParser p) {
            try {
                int eventType = p.getEventType();
                while (eventType != XmlPullParser.END_DOCUMENT) {
                    if (eventType == XmlPullParser.START_TAG) {
                        if (p.getName().equals("library")) {
                            parseLibraryTag(p);
                        } else if (p.getName().equals("file")) {
                            parseFileTag(p);
                        }
                    }
                    eventType = p.next();
                }
            } catch (Exception e) {
                Log.e(TAG, "Failed to parse xml library descriptor");
            }
        }

        private void parseLibraryTag(XmlResourceParser p) {
            version = p.getAttributeValue(null, "version");
            files = new ArrayList<String>();
        }

        private void parseFileTag(XmlResourceParser p) {
            files.add(p.getAttributeValue(null, "name"));
        }

        public boolean hasAllFiles(String path) {
            boolean result = true;
            List<String> actualFiles = Arrays.asList((new File(path)).list());
            for (String f : files)
                result &= actualFiles.contains(f);
            return result;
        }

        public boolean isCompatible(String v) {
            String[] expected = v.split("\\.");
            String[] actual = version.split("\\.");
            int i = 0;
            for (; i < Math.min(expected.length, actual.length); ++i) {
                int diff = Integer.valueOf(expected[i])
                        - Integer.valueOf(actual[i]);
                if (diff > 0 || (diff != 0 && i == 0)) {
                    // requested version is greater than actual OR major version differs
                    return false;
                } else if (diff < 0) {
                    // version is compatible
                    return true;
                }
            }
            if (expected.length > i) {
                // requested version is longer than actual - 2.4.11.2 and 2.4.11
                return false;
            }
            return true;
        }

        public String getFileList() {
            return TextUtils.join(";", files);
        }
    }

    public void onCreate() {
        Log.d(TAG, "Service starting");
        for (Field field : R.xml.class.getDeclaredFields()) {
            Log.d(TAG, "Found config: " + field.getName());
            final LibVariant lib = new LibVariant();
            try {
                final int id = field.getInt(R.xml.class);
                final XmlResourceParser p = getResources().getXml(id);
                lib.parseFile(p);
            } catch (IllegalArgumentException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
            if (lib.version != null
                    && lib.files.size() != 0
                    && lib.hasAllFiles(getApplication().getApplicationInfo().nativeLibraryDir)) {
                variants.add(lib);
            Log.d(TAG, "Added config: " + lib.version);
            }
        }
        super.onCreate();
        mEngineInterface = new OpenCVEngineInterface.Stub() {

            @Override
            public boolean installVersion(String version)
                    throws RemoteException {
                // DO NOTHING
                return false;
            }

            @Override
            public String getLibraryList(String version) throws RemoteException {
                for (LibVariant lib : variants)
                    if (lib.isCompatible(version))
                        return lib.getFileList();
                return null;
            }

            @Override
            public String getLibPathByVersion(String version)
                    throws RemoteException {
                // TODO: support API 8
                return getApplication().getApplicationInfo().nativeLibraryDir;
            }

            @Override
            public int getEngineVersion() throws RemoteException {
                int version = 3200;
                try {
                    version = getPackageManager().getPackageInfo(getPackageName(), 0).versionCode;
                } catch (NameNotFoundException e) {
                    e.printStackTrace();
                }
                return version / 1000;
            }
        };
    }

    public IBinder onBind(Intent intent) {
        Log.i(TAG, "Service onBind called for intent " + intent.toString());
        return mEngineInterface;
    }

    public boolean onUnbind(Intent intent) {
        Log.i(TAG, "Service onUnbind called for intent " + intent.toString());
        return true;
    }

    public void OnDestroy() {
        Log.i(TAG, "OpenCV Engine service destruction");
    }

}
