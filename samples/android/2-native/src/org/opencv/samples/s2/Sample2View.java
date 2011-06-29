package org.opencv.samples.s2;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.util.List;

class Sample2View extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final String TAG = "Sample2Native::View";
    
    private Camera mCamera;
    private SurfaceHolder mHolder;
    private int mFrameWidth;
    private int mFrameHeight;
    private byte[] mFrame;
    private boolean mThreadRun;
    
    public Sample2View(Context context) {
        super(context);
        mHolder = getHolder();
        mHolder.addCallback(this);
    }

    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        if ( mCamera != null) {
            Camera.Parameters params = mCamera.getParameters();
            List<Camera.Size> sizes = params.getSupportedPreviewSizes();
            mFrameWidth = width;
            mFrameHeight = height;
            
            //selecting optimal camera preview size
            {
                double minDiff = Double.MAX_VALUE;
                for (Camera.Size size : sizes) {
                    if (Math.abs(size.height - height) < minDiff) {
                        mFrameWidth = size.width;
                        mFrameHeight = size.height;
                        minDiff = Math.abs(size.height - height);
                    }
                }
            }
            params.setPreviewSize(mFrameWidth, mFrameHeight);
            mCamera.setParameters(params);
            mCamera.startPreview();
        }
    }

    public void surfaceCreated(SurfaceHolder holder) {
        mCamera = Camera.open();
        mCamera.setPreviewCallback(
                new PreviewCallback() {
                    public void onPreviewFrame(byte[] data, Camera camera) {
                        synchronized(Sample2View.this) {
                            mFrame = data;
                            Sample2View.this.notify();
                        }
                    }
                }
        );
        (new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        mThreadRun = false;
        if(mCamera != null) {
            synchronized(Sample2View.this) {
                mCamera.stopPreview();
                mCamera.setPreviewCallback(null);
                mCamera.release();
                mCamera = null;
            }
        }
    }

    public void run() {
        mThreadRun = true;
        Log.i(TAG, "Starting thread");
        while(mThreadRun) {
            byte[] data = null;
            synchronized(this) {
                try {
                    this.wait();
                    data = mFrame;
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            
            int frameSize = mFrameWidth*mFrameHeight;
            int[] rgba = new int[frameSize];
            
            FindFeatures(mFrameWidth, mFrameHeight, data, rgba);
            
            Bitmap bmp = Bitmap.createBitmap(mFrameWidth, mFrameHeight, Bitmap.Config.ARGB_8888);
            bmp.setPixels(rgba, 0/*offset*/, mFrameWidth /*stride*/, 0, 0, mFrameWidth, mFrameHeight);
            
            Canvas canvas = mHolder.lockCanvas();
            canvas.drawBitmap(bmp, (canvas.getWidth()-mFrameWidth)/2, (canvas.getHeight()-mFrameHeight)/2, null);
            mHolder.unlockCanvasAndPost(canvas);
        }
    }

    public native void FindFeatures(int width, int height, byte yuv[], int[] rgba);

    static {
        System.loadLibrary("native_sample");
    }
}
