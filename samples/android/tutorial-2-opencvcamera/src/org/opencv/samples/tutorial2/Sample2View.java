package org.opencv.samples.tutorial2;

import org.opencv.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.SurfaceHolder;

class Sample2View extends SampleViewBase {
    private Mat mRgba;
    private Mat mGray;
    private Mat mIntermediateMat;

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
        }
    }

    @Override
    protected Bitmap processFrame(VideoCapture capture) {
        switch (Sample2NativeCamera.viewMode) {
        case Sample2NativeCamera.VIEW_MODE_GRAY:
            capture.retrieve(mGray, highgui.CV_CAP_ANDROID_GREY_FRAME);
            imgproc.cvtColor(mGray, mRgba, imgproc.CV_GRAY2RGBA, 4);
            break;
        case Sample2NativeCamera.VIEW_MODE_RGBA:
            capture.retrieve(mRgba, highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            core.putText(mRgba, "OpenCV + Android", new Point(10, 100), 3/* CV_FONT_HERSHEY_COMPLEX */, 2, new Scalar(255, 0, 0, 255), 3);
            break;
        case Sample2NativeCamera.VIEW_MODE_CANNY:
            capture.retrieve(mGray, highgui.CV_CAP_ANDROID_GREY_FRAME);
            imgproc.Canny(mGray, mIntermediateMat, 80, 100);
            imgproc.cvtColor(mIntermediateMat, mRgba, imgproc.CV_GRAY2BGRA, 4);
            break;
        case Sample2NativeCamera.VIEW_MODE_SOBEL:
            capture.retrieve(mGray, highgui.CV_CAP_ANDROID_GREY_FRAME);
            imgproc.Sobel(mGray, mIntermediateMat, CvType.CV_8U, 1, 1);
            core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 8);
            imgproc.cvtColor(mIntermediateMat, mRgba, imgproc.CV_GRAY2BGRA, 4);
            break;
        case Sample2NativeCamera.VIEW_MODE_BLUR:
            capture.retrieve(mRgba, highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            imgproc.blur(mRgba, mRgba, new Size(15, 15));
            break;
        }

        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

        if (android.MatToBitmap(mRgba, bmp))
            return bmp;

        bmp.recycle();
        return null;
    }

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.dispose();
            if (mGray != null)
                mGray.dispose();
            if (mIntermediateMat != null)
                mIntermediateMat.dispose();

            mRgba = null;
            mGray = null;
            mIntermediateMat = null;
        }
    }
}