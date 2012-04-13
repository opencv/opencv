package org.opencv.samples.tutorial2;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

class Sample2View extends SampleCvViewBase {
    private Mat mRgba;
    private Mat mGray;
    private Mat mIntermediateMat;
    private Mat mIntermediateMat2;
    private Mat mEmpty;
    private Scalar lo, hi;
    private Scalar bl, wh;

    public Sample2View(Context context) {
        super(context);
    }

    @Override
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        super.surfaceChanged(_holder, format, width, height);

        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
            mRgba = new Mat();
            mIntermediateMat = new Mat();
            mIntermediateMat2 = new Mat();
            mEmpty = new Mat();
            lo = new Scalar(85, 100, 30);
            hi = new Scalar(130, 255, 255);
            bl = new Scalar(0, 0, 0, 255);
            wh = new Scalar(255, 255, 255, 255);
        }
    }

    @Override
    protected Bitmap processFrame(VideoCapture capture) {
    	/**/
        switch (Sample2NativeCamera.viewMode) {
        case Sample2NativeCamera.VIEW_MODE_GRAY:
            capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
            Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            break;
        case Sample2NativeCamera.VIEW_MODE_RGBA:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            Core.putText(mRgba, "OpenCV + Android", new Point(10, 100), 3, 2, new Scalar(255, 0, 0, 255), 3);
            break;
        case Sample2NativeCamera.VIEW_MODE_CANNY:
            /*capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
            Imgproc.Canny(mGray, mIntermediateMat, 80, 100);
            Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2BGRA, 4);
            */
        	capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        	Imgproc.cvtColor(mRgba, mIntermediateMat, Imgproc.COLOR_RGB2HSV_FULL);
        	Core.inRange(mIntermediateMat, lo, hi, mIntermediateMat2); // green
        	Imgproc.dilate(mIntermediateMat2, mIntermediateMat2, mEmpty);
        	//
        	List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        	Mat hierarchy = new Mat();
        	Imgproc.findContours(mIntermediateMat2, contours, hierarchy,Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        	Log.d("processFrame", "contours.size()" + contours.size());
        	double maxArea = 0;
        	int indexMaxArea = -1;
	       for (int i = 0; i < contours.size(); i++) {
	    	   double s = Imgproc.contourArea(contours.get(i));
	             if(s > maxArea){
	    		    indexMaxArea = i;
	    		    maxArea = s;
	    	     }
	       } 
	       
			mRgba.setTo(bl);
			Imgproc.drawContours(mRgba, contours, indexMaxArea, wh);
			//
			//Imgproc.cvtColor(mIntermediateMat2, mRgba, Imgproc.COLOR_GRAY2RGBA);
			break;
        }
    	/**/

        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mRgba, bmp);
            return bmp;
        } catch(Exception e) {
        	Log.e("org.opencv.samples.puzzle15", "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            return null;
        }
    }

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();
            if (mGray != null)
                mGray.release();
            if (mIntermediateMat != null)
                mIntermediateMat.release();

            if (mIntermediateMat2 != null)
                mIntermediateMat2.release();
            
            mRgba = null;
            mGray = null;
            mIntermediateMat = null;
        }
    }
}
