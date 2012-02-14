package org.opencv.samples.tutorial4;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

class Sample4View extends SampleViewBase {
    private Mat mYuv;
    private Mat mRgba;
    private Mat mGraySubmat;
    private Mat mIntermediateMat;

    public Sample4View(Context context) {
        super(context);
    }

    @Override
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        super.surfaceChanged(_holder, format, width, height);

        synchronized (this) {
            // initialize Mats before usage
            mYuv = new Mat(getFrameHeight() + getFrameHeight() / 2, getFrameWidth(), CvType.CV_8UC1);
            mGraySubmat = mYuv.submat(0, getFrameHeight(), 0, getFrameWidth());

            mRgba = new Mat();
            mIntermediateMat = new Mat();
        }
    }

    @Override
    protected Bitmap processFrame(byte[] data) {
        mYuv.put(0, 0, data);

        switch (Sample4Mixed.viewMode) {
        case Sample4Mixed.VIEW_MODE_GRAY:
            Imgproc.cvtColor(mGraySubmat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            break;
        case Sample4Mixed.VIEW_MODE_RGBA:
            Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV420sp2RGB, 4);
            break;
        case Sample4Mixed.VIEW_MODE_CANNY:
            Imgproc.Canny(mGraySubmat, mIntermediateMat, 80, 100);
            Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2BGRA, 4);
            break;
        case Sample4Mixed.VIEW_MODE_FEATURES:
            Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV420sp2RGB, 4);
            FindFeatures(mGraySubmat.getNativeObjAddr(), mRgba.getNativeObjAddr());
            break;
        }

        Bitmap bmp = Bitmap.createBitmap(getFrameWidth(), getFrameHeight(), Bitmap.Config.ARGB_8888);

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
            if (mYuv != null)
                mYuv.release();
            if (mRgba != null)
                mRgba.release();
            if (mGraySubmat != null)
                mGraySubmat.release();
            if (mIntermediateMat != null)
                mIntermediateMat.release();

            mYuv = null;
            mRgba = null;
            mGraySubmat = null;
            mIntermediateMat = null;
        }
    }

    public native void FindFeatures(long matAddrGr, long matAddrRgba);

    static {
        System.loadLibrary("mixed_sample");
    }
}
