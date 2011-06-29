package org.opencv.samples;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.util.List;

class Sample0View extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final String TAG = "Sample0Base::View";
    
    private Camera mCamera;
    private SurfaceHolder mHolder;
    private int mFrameWidth;
    private int mFrameHeight;
    private byte[] mFrame;
    private boolean mThreadRun;

    public Sample0View(Context context) {
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
                        synchronized(Sample0View.this) {
                            mFrame = data;
                            Sample0View.this.notify();
                        }
                    }
                }
        );
        (new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        mThreadRun = false;
        if(mCamera != null) {
            mCamera.stopPreview();
            mCamera.setPreviewCallback(null);
            mCamera.release();
            mCamera = null;
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
            
            Sample0Base a = (Sample0Base)getContext();
            int view_mode = a.viewMode;
            if(view_mode == Sample0Base.VIEW_MODE_GRAY) {
                for(int i = 0; i < frameSize; i++) {
                    int y = (0xff & ((int)data[i]));
                    rgba[i] = 0xff000000 + (y << 16) + (y << 8) + y;
                }
            } else if (view_mode == Sample0Base.VIEW_MODE_RGBA) {
                for(int i = 0; i < mFrameHeight; i++)
                    for(int j = 0; j < mFrameWidth; j++) {
                        int y = (0xff & ((int)data[i*mFrameWidth+j]));
                        int u = (0xff & ((int)data[frameSize + (i >> 1) * mFrameWidth + (j & ~1) + 0]));
                        int v = (0xff & ((int)data[frameSize + (i >> 1) * mFrameWidth + (j & ~1) + 1]));
                        if (y < 16) y = 16;
                        
                        int r = Math.round(1.164f * (y - 16) + 1.596f * (v - 128)                     );
                        int g = Math.round(1.164f * (y - 16) - 0.813f * (v - 128) - 0.391f * (u - 128));
                        int b = Math.round(1.164f * (y - 16)                      + 2.018f * (u - 128));
                        
                        if (r < 0) r = 0; if (r > 255) r = 255;
                        if (g < 0) g = 0; if (g > 255) g = 255;
                        if (b < 0) b = 0; if (b > 255) b = 255;
                        
                        rgba[i*mFrameWidth+j] = 0xff000000 + (b << 16) + (g << 8) + r;
                    }
            }
            
            Bitmap bmp = Bitmap.createBitmap(mFrameWidth, mFrameHeight, Bitmap.Config.ARGB_8888);
            bmp.setPixels(rgba, 0/*offset*/, mFrameWidth /*stride*/, 0, 0, mFrameWidth, mFrameHeight);
            
            Canvas canvas = mHolder.lockCanvas();
            canvas.drawBitmap(bmp, (canvas.getWidth()-mFrameWidth)/2, (canvas.getHeight()-mFrameHeight)/2, null);
            mHolder.unlockCanvasAndPost(canvas);
        }
    }
}