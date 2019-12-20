package org.opencv.samples.tutorial4;

import org.opencv.android.CameraGLSurfaceView;

import android.app.Activity;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.widget.TextView;
import android.widget.Toast;

public class MyGLSurfaceView extends CameraGLSurfaceView implements CameraGLSurfaceView.CameraTextureListener {

    static final String LOGTAG = "MyGLSurfaceView";
    protected int procMode = NativePart.PROCESSING_MODE_NO_PROCESSING;
    static final String[] procModeName = new String[] {"No Processing", "CPU", "OpenCL Direct", "OpenCL via OpenCV"};
    protected int  frameCounter;
    protected long lastNanoTime;
    TextView mFpsText = null;

    public MyGLSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if(e.getAction() == MotionEvent.ACTION_DOWN)
            ((Activity)getContext()).openOptionsMenu();
        return true;
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        super.surfaceCreated(holder);
        //NativePart.initCL();
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        //NativePart.closeCL();
        super.surfaceDestroyed(holder);
    }

    public void setProcessingMode(int newMode) {
        if(newMode>=0 && newMode<procModeName.length)
            procMode = newMode;
        else
            Log.e(LOGTAG, "Ignoring invalid processing mode: " + newMode);

        ((Activity) getContext()).runOnUiThread(new Runnable() {
            public void run() {
                Toast.makeText(getContext(), "Selected mode: " + procModeName[procMode], Toast.LENGTH_LONG).show();
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        ((Activity) getContext()).runOnUiThread(new Runnable() {
            public void run() {
                Toast.makeText(getContext(), "onCameraViewStarted", Toast.LENGTH_SHORT).show();
            }
        });
        NativePart.initCL();
        frameCounter = 0;
        lastNanoTime = System.nanoTime();
    }

    @Override
    public void onCameraViewStopped() {
        ((Activity) getContext()).runOnUiThread(new Runnable() {
            public void run() {
                Toast.makeText(getContext(), "onCameraViewStopped", Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    public boolean onCameraTexture(int texIn, int texOut, int width, int height) {
        // FPS
        frameCounter++;
        if(frameCounter >= 30)
        {
            final int fps = (int) (frameCounter * 1e9 / (System.nanoTime() - lastNanoTime));
            Log.i(LOGTAG, "drawFrame() FPS: "+fps);
            if(mFpsText != null) {
                Runnable fpsUpdater = new Runnable() {
                    public void run() {
                        mFpsText.setText("FPS: " + fps);
                    }
                };
                new Handler(Looper.getMainLooper()).post(fpsUpdater);
            } else {
                Log.d(LOGTAG, "mFpsText == null");
                mFpsText = (TextView)((Activity) getContext()).findViewById(R.id.fps_text_view);
            }
            frameCounter = 0;
            lastNanoTime = System.nanoTime();
        }


        if(procMode == NativePart.PROCESSING_MODE_NO_PROCESSING)
            return false;

        NativePart.processFrame(texIn, texOut, width, height, procMode);
        return true;
    }
}
