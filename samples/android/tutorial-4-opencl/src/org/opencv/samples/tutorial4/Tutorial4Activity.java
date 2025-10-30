package org.opencv.samples.tutorial4;

import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.CameraActivity;

public class Tutorial4Activity extends CameraActivity {

    private MyGLSurfaceView mView;
    private TextView mProcMode;

    private boolean builtWithOpenCL = false;

    private MenuItem mItemNoProc;
    private MenuItem mItemCpu;
    private MenuItem mItemOclDirect;
    private MenuItem mItemOclOpenCV;

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

        builtWithOpenCL = NativePart.builtWithOpenCL();
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
        mItemNoProc = menu.add("No processing");
        mItemCpu = menu.add("Use CPU code");
        if (builtWithOpenCL) {
            mItemOclOpenCV = menu.add("Use OpenCL via OpenCV");
            mItemOclDirect = menu.add("Use OpenCL direct");
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        String procName = "Not selected";
        int procMode = NativePart.PROCESSING_MODE_NO_PROCESSING;

        if (item == mItemNoProc) {
            procMode = NativePart.PROCESSING_MODE_NO_PROCESSING;
            procName = "Processing mode: No Processing";
        } else if (item == mItemCpu) {
            procMode = NativePart.PROCESSING_MODE_CPU;
            procName = "Processing mode: CPU";
        } else if (item == mItemOclOpenCV && builtWithOpenCL) {
            procMode = NativePart.PROCESSING_MODE_OCL_OCV;
            procName = "Processing mode: OpenCL via OpenCV (TAPI)";
        } else if (item == mItemOclDirect && builtWithOpenCL) {
            procMode = NativePart.PROCESSING_MODE_OCL_DIRECT;
            procName = "Processing mode: OpenCL direct";
        }

        mView.setProcessingMode(procMode);
        mProcMode.setText(procName);

        return true;
    }
}
