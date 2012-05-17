package org.opencv.samples.colorblobdetect;

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
    private static final String TAG = "Sample::SurfaceView";

    private SurfaceHolder       mHolder;
    private VideoCapture        mCamera;

    public SampleCvViewBase(Context context) {
        super(context);
        mHolder = getHolder();
        mHolder.addCallback(this);
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    public boolean openCamera() {
        Log.i(TAG, "openCamera");
        synchronized (this) {
	        releaseCamera();
	        mCamera = new VideoCapture(Highgui.CV_CAP_ANDROID);
	        if (!mCamera.isOpened()) {
	            mCamera.release();
	            mCamera = null;
	            Log.e(TAG, "Failed to open native camera");
	            return false;
	        }
	    }
        return true;
    }
    
    public void releaseCamera() {
        Log.i(TAG, "releaseCamera");
        synchronized (this) {
	        if (mCamera != null) {
	                mCamera.release();
	                mCamera = null;
            }
        }
    }
    
    public void setupCamera(int width, int height) {
        Log.i(TAG, "setupCamera("+width+", "+height+")");
        synchronized (this) {
            if (mCamera != null && mCamera.isOpened()) {
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

    }
    
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        Log.i(TAG, "surfaceChanged");
        setupCamera(width, height);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        Log.i(TAG, "surfaceCreated");
        (new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i(TAG, "surfaceDestroyed");
        releaseCamera();
    }

    protected abstract Bitmap processFrame(VideoCapture capture);

    public void run() {
        Log.i(TAG, "Starting processing thread");
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
            }

            if (bmp != null) {
                Canvas canvas = mHolder.lockCanvas();
                if (canvas != null) {
                    canvas.drawBitmap(bmp, (canvas.getWidth() - bmp.getWidth()) / 2, (canvas.getHeight() - bmp.getHeight()) / 2, null);
                    mHolder.unlockCanvasAndPost(canvas);
                }
                bmp.recycle();
            }
        }

        Log.i(TAG, "Finishing processing thread");
    }
}