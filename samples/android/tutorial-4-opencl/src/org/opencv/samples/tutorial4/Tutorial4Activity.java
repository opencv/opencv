package org.opencv.samples.tutorial4;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

public class Tutorial4Activity extends Activity {

    private MyGLSurfaceView mView;
    private TextView mProcMode;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        //mView = new MyGLSurfaceView(this, null);
        //setContentView(mView);
        setContentView(R.layout.activity);
        mView = (MyGLSurfaceView) findViewById(R.id.my_gl_surface_view);
        mView.setCameraTextureListener(mView);
        TextView tv = (TextView)findViewById(R.id.fps_text_view);
        mProcMode = (TextView)findViewById(R.id.proc_mode_text_view);
        runOnUiThread(new Runnable() {
            public void run() {
                mProcMode.setText("Processing mode: No processing");
            }
        });

        mView.setProcessingMode(NativePart.PROCESSING_MODE_NO_PROCESSING);
    }

    @Override
    protected void onPause() {
        mView.onPause();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mView.onResume();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
        case R.id.no_proc:
            runOnUiThread(new Runnable() {
                public void run() {
                    mProcMode.setText("Processing mode: No Processing");
                }
            });
            mView.setProcessingMode(NativePart.PROCESSING_MODE_NO_PROCESSING);
            return true;
        case R.id.cpu:
            runOnUiThread(new Runnable() {
                public void run() {
                    mProcMode.setText("Processing mode: CPU");
                }
            });
            mView.setProcessingMode(NativePart.PROCESSING_MODE_CPU);
            return true;
        case R.id.ocl_direct:
            runOnUiThread(new Runnable() {
                public void run() {
                    mProcMode.setText("Processing mode: OpenCL direct");
                }
            });
            mView.setProcessingMode(NativePart.PROCESSING_MODE_OCL_DIRECT);
            return true;
        case R.id.ocl_ocv:
            runOnUiThread(new Runnable() {
                public void run() {
                    mProcMode.setText("Processing mode: OpenCL via OpenCV (TAPI)");
                }
            });
            mView.setProcessingMode(NativePart.PROCESSING_MODE_OCL_OCV);
            return true;
        default:
            return false;
        }
    }
}