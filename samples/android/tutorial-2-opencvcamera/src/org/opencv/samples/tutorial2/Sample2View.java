package org.opencv.samples.tutorial2;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
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

    public Sample2View(Context context) {
        super(context);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
            mRgba = new Mat();
            mIntermediateMat = new Mat();
        }
        
        super.surfaceCreated(holder);
    }

    @Override
    protected Bitmap processFrame(VideoCapture capture) {
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
            capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
            Imgproc.Canny(mGray, mIntermediateMat, 80, 100);
            Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2BGRA, 4);
			break;
        }

        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mRgba, bmp);
            return bmp;
        } catch(Exception e) {
        	Log.e("org.opencv.samples.tutorial2", "Utils.matToBitmap() throws an exception: " + e.getMessage());
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

            mRgba = null;
            mGray = null;
            mIntermediateMat = null;
        }
    }
}
