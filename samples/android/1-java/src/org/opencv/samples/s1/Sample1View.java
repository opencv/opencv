package org.opencv.samples.s1;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import org.opencv.Mat;
import org.opencv.Point;
import org.opencv.Scalar;
import org.opencv.Size;
import org.opencv.core;
import org.opencv.imgproc;
import org.opencv.utils;

import java.util.List;

class Sample1View extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final String TAG = "Sample1Java::View";
    
    private Camera mCamera;
    private SurfaceHolder mHolder;
    private int mFrameWidth;
    private int mFrameHeight;
    private byte[] mFrame;
    private boolean mThreadRun;
    
    private Mat mYuv;
    private Mat mRgba;
    private Mat mGraySubmat;
    private Mat mIntermediateMat;

    public Sample1View(Context context) {
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
            
            // initialize all required Mats before usage to minimize number of auxiliary jni calls
            if(mYuv != null) mYuv.dispose();
            mYuv = new Mat(mFrameHeight+mFrameHeight/2, mFrameWidth, Mat.CvType.CV_8UC1);
            
            if(mRgba != null) mRgba.dispose();
            mRgba = new Mat(mFrameHeight, mFrameWidth, Mat.CvType.CV_8UC4);
            
            if(mGraySubmat != null) mGraySubmat.dispose();
            mGraySubmat = mYuv.submat(0, mFrameHeight, 0, mFrameWidth); 

            if(mIntermediateMat != null) mIntermediateMat.dispose();
            mIntermediateMat = new Mat(mFrameHeight, mFrameWidth, Mat.CvType.CV_8UC1);
        }
    }

    public void surfaceCreated(SurfaceHolder holder) {
        mCamera = Camera.open();
        mCamera.setPreviewCallback(
                new PreviewCallback() {
                    public void onPreviewFrame(byte[] data, Camera camera) {
                        synchronized(Sample1View.this) {
                            mFrame = data;
                            Sample1View.this.notify();
                        }
                    }
                }
        );
        (new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        mThreadRun = false;
        if(mCamera != null) {
            synchronized(Sample1View.this) {
                mCamera.stopPreview();
                mCamera.setPreviewCallback(null);
                mCamera.release();
                mCamera = null;
            }
        }
        
        // Explicitly dispose Mats 
        if(mYuv != null) {
            mYuv.dispose();
            mYuv = null;
        }
        if(mRgba != null) {
            mRgba.dispose();
            mRgba = null;
        }
        if(mGraySubmat != null) {
            mGraySubmat.dispose();
            mGraySubmat = null;
        }
        if(mIntermediateMat != null) {
            mIntermediateMat.dispose();
            mIntermediateMat = null;
        }
    }

    public void run() {
        mThreadRun = true;
        Log.i(TAG, "Starting thread");
        while(mThreadRun) {
            synchronized(this) {
                try {
                    this.wait();
                    mYuv.put(0, 0, mFrame);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            
            Sample1Java a = (Sample1Java)getContext();
           
            switch(a.viewMode)
            {
            case Sample1Java.VIEW_MODE_GRAY:
                imgproc.cvtColor(mGraySubmat, mRgba, imgproc.CV_GRAY2RGBA, 4);
                break;
            case Sample1Java.VIEW_MODE_RGBA:
                imgproc.cvtColor(mYuv, mRgba, imgproc.CV_YUV420i2RGB, 4);
                core.putText(mRgba, "OpenCV + Android", new Point(10,100), 3/*CV_FONT_HERSHEY_COMPLEX*/, 2, new Scalar(0, 255,0, 255), 3);
                break;
            case Sample1Java.VIEW_MODE_CANNY:
                imgproc.Canny(mGraySubmat, mIntermediateMat, 80, 100);
                imgproc.cvtColor(mIntermediateMat, mRgba, imgproc.CV_GRAY2BGRA, 4);
                break;
            case Sample1Java.VIEW_MODE_SOBEL:
                imgproc.Sobel(mGraySubmat, mIntermediateMat, core.CV_8U, 1, 1);
                core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 8);
                imgproc.cvtColor(mIntermediateMat, mRgba, imgproc.CV_GRAY2BGRA, 4);
                break;
            case Sample1Java.VIEW_MODE_BLUR:
                imgproc.cvtColor(mYuv, mRgba, imgproc.CV_YUV420i2RGB, 4);
                imgproc.blur(mRgba, mRgba, new Size(15, 15));
                break;
            }
            
            Bitmap bmp = Bitmap.createBitmap(mFrameWidth, mFrameHeight, Bitmap.Config.ARGB_8888);
            utils.MatToBitmap(mRgba, bmp);
            
            Canvas canvas = mHolder.lockCanvas();
            canvas.drawBitmap(bmp, (canvas.getWidth()-mFrameWidth)/2, (canvas.getHeight()-mFrameHeight)/2, null);
            mHolder.unlockCanvasAndPost(canvas);
        }
    }
}