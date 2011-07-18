package org.opencv.samples.tutorial4;

import org.opencv.*;

import android.content.Context;
import android.graphics.Bitmap;
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
            imgproc.cvtColor(mGraySubmat, mRgba, imgproc.CV_GRAY2RGBA, 4);
            break;
        case Sample4Mixed.VIEW_MODE_RGBA:
            imgproc.cvtColor(mYuv, mRgba, imgproc.CV_YUV420i2RGB, 4);
            break;
        case Sample4Mixed.VIEW_MODE_CANNY:
            imgproc.Canny(mGraySubmat, mIntermediateMat, 80, 100);
            imgproc.cvtColor(mIntermediateMat, mRgba, imgproc.CV_GRAY2BGRA, 4);
            break;
        case Sample4Mixed.VIEW_MODE_FEATURES:
            imgproc.cvtColor(mYuv, mRgba, imgproc.CV_YUV420i2RGB, 4);
            FindFeatures(mGraySubmat.getNativeObjAddr(), mRgba.getNativeObjAddr());
            break;
        }

        Bitmap bmp = Bitmap.createBitmap(getFrameWidth(), getFrameHeight(), Bitmap.Config.ARGB_8888);

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
            if (mYuv != null)
                mYuv.dispose();
            if (mRgba != null)
                mRgba.dispose();
            if (mGraySubmat != null)
                mGraySubmat.dispose();
            if (mIntermediateMat != null)
                mIntermediateMat.dispose();

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