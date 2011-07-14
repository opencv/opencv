package org.opencv.samples.s1;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

import org.opencv.CvType;
import org.opencv.Mat;
import org.opencv.Point;
import org.opencv.Scalar;
import org.opencv.Size;
import org.opencv.core;
import org.opencv.imgproc;
import org.opencv.android;


class Sample1View extends SampleViewBase implements SurfaceHolder.Callback {
    Mat mYuv;
    Mat mRgba;
    Mat mGraySubmat;
    Mat mIntermediateMat;

    public Sample1View(Context context) {
        super(context);
    }
    
    @Override
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
    	super.surfaceChanged(_holder, format, width, height);
    	Log.e("SAMP1", "surfaceChanged begin");
    	synchronized (this) {
    		Log.e("SAMP1", "surfaceChanged sync");
            // initialize all required Mats before usage to minimize number of auxiliary jni calls
            if(mYuv != null) mYuv.dispose();
            mYuv = new Mat(mFrameHeight+mFrameHeight/2, mFrameWidth, CvType.CV_8UC1);
            
            if(mRgba != null) mRgba.dispose();
            mRgba = new Mat(mFrameHeight, mFrameWidth, CvType.CV_8UC4);
            
            if(mGraySubmat != null) mGraySubmat.dispose();
            mGraySubmat = mYuv.submat(0, mFrameHeight, 0, mFrameWidth); 

            if(mIntermediateMat != null) mIntermediateMat.dispose();
            mIntermediateMat = new Mat(mFrameHeight, mFrameWidth, CvType.CV_8UC1);
		}
    	Log.e("SAMP1", "surfaceChanged end");
    }

    @Override
    protected Bitmap processFrame(byte[] data)
    {
    	Log.e("SAMP1", "processFrame begin");
    	
    	mYuv.put(0, 0, data);
    	
    	Sample1Java a = (Sample1Java)getContext();
        
        switch(a.viewMode)
        {
        case Sample1Java.VIEW_MODE_GRAY:
            imgproc.cvtColor(mGraySubmat, mRgba, imgproc.CV_GRAY2RGBA, 4);
            break;
        case Sample1Java.VIEW_MODE_RGBA:
            imgproc.cvtColor(mYuv, mRgba, imgproc.CV_YUV420i2RGB, 4);
            core.putText(mRgba, "OpenCV + Android", new Point(10,100), 3/*CV_FONT_HERSHEY_COMPLEX*/, 2, new Scalar(255, 0, 0, 255), 3);
            break;
        case Sample1Java.VIEW_MODE_CANNY:
            imgproc.Canny(mGraySubmat, mIntermediateMat, 80, 100);
            imgproc.cvtColor(mIntermediateMat, mRgba, imgproc.CV_GRAY2BGRA, 4);
            break;
        case Sample1Java.VIEW_MODE_SOBEL:
            imgproc.Sobel(mGraySubmat, mIntermediateMat, CvType.CV_8U, 1, 1);
            core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 8);
            imgproc.cvtColor(mIntermediateMat, mRgba, imgproc.CV_GRAY2BGRA, 4);
            break;
        case Sample1Java.VIEW_MODE_BLUR:
            imgproc.cvtColor(mYuv, mRgba, imgproc.CV_YUV420i2RGB, 4);
            imgproc.blur(mRgba, mRgba, new Size(15, 15));
            break;
        }
        
        Bitmap bmp = Bitmap.createBitmap(mFrameWidth, mFrameHeight, Bitmap.Config.ARGB_8888);
        android.MatToBitmap(mRgba, bmp);
        
        Log.e("SAMP1", "processFrame end");
        return bmp;
    }
    
    @Override
    public void run() {
    	Log.e("SAMP1", "run");
    	super.run();
    	Log.e("SAMP1", "run2");
    	
        // Explicitly release Mats 
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
}