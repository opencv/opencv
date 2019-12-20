package org.opencv.engine.manager;

import org.opencv.engine.MarketConnector;
import org.opencv.engine.HardwareDetector;
import org.opencv.engine.OpenCVEngineInterface;
import org.opencv.engine.OpenCVEngineService;
import org.opencv.engine.R;
import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.os.RemoteException;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

public class ManagerActivity extends Activity {
    protected static final String TAG = "OpenCVEngine/Activity";
    protected MarketConnector mMarket;
    protected TextView mVersionText;
    protected boolean mExtraInfo = false;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.main);

        final Class<OpenCVEngineService> c = OpenCVEngineService.class;
        final String packageName = c.getPackage().getName();

        mMarket = new MarketConnector(this);

        Button updateButton = (Button) findViewById(R.id.CheckEngineUpdate);
        updateButton.setOnClickListener(new OnClickListener() {
            public void onClick(View v) {
                if (!mMarket.InstallAppFromMarket(packageName)) {
                    Toast toast = Toast.makeText(getApplicationContext(),
                            "Google Play is not available", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });

        TextView aboutText = (TextView) findViewById(R.id.textView4);
        aboutText.setText("About (" + packageName + ")");

        if (mExtraInfo) {
            TextView extraText = (TextView) findViewById(R.id.textView6);
            extraText.setText(
                    "CPU count: "
                    + HardwareDetector.getProcessorCount()
                    + "\nABI: 0x"
                    + Integer.toHexString(HardwareDetector.getAbi())
                    + "\nFlags: "
                    + TextUtils.join(";", HardwareDetector.getFlags())
                    + "\nHardware: "
                    + HardwareDetector.getHardware());
        }

        mVersionText = (TextView) findViewById(R.id.textView5);
        if (!bindService(new Intent(this, c),
                new OpenCVEngineServiceConnection(), Context.BIND_AUTO_CREATE)) {
            Log.e(TAG, "Failed to bind to service:" + c.getName());
            mVersionText.setText("not available");
        } else {
            Log.d(TAG, "Successfully bound to service:" + c.getName());
            mVersionText.setText("available");
        }

    }

    protected class OpenCVEngineServiceConnection implements ServiceConnection {
        public void onServiceDisconnected(ComponentName name) {
            Log.d(TAG, "Handle: service disconnected");
        }

        public void onServiceConnected(ComponentName name, IBinder service) {
            Log.d(TAG, "Handle: service connected");
            OpenCVEngineInterface engine = OpenCVEngineInterface.Stub
                    .asInterface(service);
            if (engine == null) {
                Log.e(TAG, "Cannot connect to OpenCV Manager Service!");
                unbindService(this);
                return;
            }
            Log.d(TAG, "Successful connection");
            try {
                String[] vars = { "2.4", "3.0" };
                String res = new String();
                for (String piece : vars) {
                    res += "\n\t" + piece + " -> "
                            + engine.getLibraryList(piece);
                }
                mVersionText.setText("Path: "
                        + engine.getLibPathByVersion(null) + res);
            } catch (RemoteException e) {
                e.printStackTrace();
                Log.e(TAG, "Call failed");
            }
            unbindService(this);
        }
    };

}
