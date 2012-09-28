package org.opencv.samples.imagemanipulations;

import java.util.List;

import org.opencv.core.Size;
import org.opencv.highgui.VideoCapture;
import org.opencv.highgui.Highgui;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public abstract class SampleCvViewBase extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final String TAG = "OCVSample::BaseView";

    private SurfaceHolder       mHolder;
    private VideoCapture        mCamera;
    private FpsMeter            mFps;

    public SampleCvViewBase(Context context) {
        super(context);
        mHolder = getHolder();
        mHolder.addCallback(this);
        mFps = new FpsMeter();
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    public synchronized boolean openCamera() {
        Log.i(TAG, "Opening Camera");
        mCamera = new VideoCapture(Highgui.CV_CAP_ANDROID);
        if (!mCamera.isOpened()) {
            releaseCamera();
            Log.e(TAG, "Can't open native camera");
            return false;
        }
        return true;
    }

    public synchronized void releaseCamera() {
        Log.i(TAG, "Releasing Camera");
        if (mCamera != null) {
                mCamera.release();
                mCamera = null;
        }
    }

    public synchronized void setupCamera(int width, int height) {
        if (mCamera != null && mCamera.isOpened()) {
            Log.i(TAG, "Setup Camera - " + width + "x" + height);
            List<Size> sizes = mCamera.getSupportedPreviewSizes();
            int mFrameWidth = width;
            int mFrameHeight = height;

            // selecting optimal camera preview size
            {
                double minDiff = Double.MAX_VALUE;
                for (Size size : sizes) {
                    if (Math.abs(size.height - height) < minDiff) {
                        mFrameWidth = (int) size.width;
                        mFrameHeight = (int) size.height;
                        minDiff = Math.abs(size.height - height);
                    }
                }
            }

            mCamera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, mFrameWidth);
            mCamera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, mFrameHeight);
        }
    }

    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        Log.i(TAG, "called surfaceChanged");
        setupCamera(width, height);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        Log.i(TAG, "called surfaceCreated");
        (new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i(TAG, "called surfaceDestroyed");
    }

    protected abstract Bitmap processFrame(VideoCapture capture);

    public void run() {
        Log.i(TAG, "Started processing thread");
        mFps.init();

        while (true) {
            Bitmap bmp = null;

            synchronized (this) {
                if (mCamera == null)
                    break;

                if (!mCamera.grab()) {
                    Log.e(TAG, "mCamera.grab() failed");
                    break;
                }

                bmp = processFrame(mCamera);

                mFps.measure();
            }

            if (bmp != null) {
                Canvas canvas = mHolder.lockCanvas();
                if (canvas != null) {
                    canvas.drawColor(0, android.graphics.PorterDuff.Mode.CLEAR);
                    canvas.drawBitmap(bmp, (canvas.getWidth() - bmp.getWidth()) / 2, (canvas.getHeight() - bmp.getHeight()), null);
                    mFps.draw(canvas, (canvas.getWidth() - bmp.getWidth()) / 2, 0);
                    mHolder.unlockCanvasAndPost(canvas);
                }
                bmp.recycle();
            }
        }
        Log.i(TAG, "Finished processing thread");
    }
}