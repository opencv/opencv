package org.opencv.samples.imagemanipulations;

import org.opencv.android;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.SurfaceHolder;

class ImageManipulationsView extends SampleCvViewBase {
    private Mat mRgba;
    private Mat mGray;
    private Mat mIntermediateMat;

    private Mat mRgbaInnerWindow;
    private Mat mGrayInnerWindow;
    private Mat mBlurWindow;
    private Mat mZoomWindow;
    private Mat mZoomCorner;

    private Mat mSepiaKernel;

    public ImageManipulationsView(Context context) {
        super(context);

        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
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

    private void CreateAuxiliaryMats() {
        if (mRgba.empty())
            return;

        int rows = mRgba.rows();
        int cols = mRgba.cols();

        int left = cols / 8;
        int top = rows / 8;

        int width = cols * 3 / 4;
        int height = rows * 3 / 4;

        if (mRgbaInnerWindow == null)
            mRgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);

        if (mGrayInnerWindow == null && !mGray.empty())
            mGrayInnerWindow = mGray.submat(top, top + height, left, left + width);

        if (mBlurWindow == null)
            mBlurWindow = mRgba.submat(0, rows, cols / 3, cols * 2 / 3);

        if (mZoomCorner == null)
            mZoomCorner = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);

        if (mZoomWindow == null)
            mZoomWindow = mRgba.submat(rows / 2 - 9 * rows / 100, rows / 2 + 9 * rows / 100, cols / 2 - 9 * cols / 100, cols / 2 + 9 * cols / 100);
    }

    @Override
    protected Bitmap processFrame(VideoCapture capture) {
        switch (ImageManipulationsActivity.viewMode) {

        case ImageManipulationsActivity.VIEW_MODE_RGBA:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            break;

        case ImageManipulationsActivity.VIEW_MODE_CANNY:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);

            if (mRgbaInnerWindow == null || mGrayInnerWindow == null)
                CreateAuxiliaryMats();

            Imgproc.Canny(mGrayInnerWindow, mGrayInnerWindow, 80, 90);
            Imgproc.cvtColor(mGrayInnerWindow, mRgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            break;

        case ImageManipulationsActivity.VIEW_MODE_SOBEL:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);

            if (mRgbaInnerWindow == null || mGrayInnerWindow == null)
                CreateAuxiliaryMats();

            Imgproc.Sobel(mGrayInnerWindow, mIntermediateMat, CvType.CV_8U, 1, 1);
            Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 10);
            Imgproc.cvtColor(mIntermediateMat, mRgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            break;

        case ImageManipulationsActivity.VIEW_MODE_SEPIA:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            Core.transform(mRgba, mRgba, mSepiaKernel);
            break;

        case ImageManipulationsActivity.VIEW_MODE_BLUR:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            if (mBlurWindow == null)
                CreateAuxiliaryMats();
            Imgproc.blur(mBlurWindow, mBlurWindow, new Size(15, 15));
            break;

        case ImageManipulationsActivity.VIEW_MODE_ZOOM:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            if (mZoomCorner == null || mZoomWindow == null)
                CreateAuxiliaryMats();
            Imgproc.resize(mZoomWindow, mZoomCorner, mZoomCorner.size());

            Size wsize = mZoomWindow.size();
            Core.rectangle(mZoomWindow, new Point(1, 1), new Point(wsize.width - 2, wsize.height - 2), new Scalar(255, 0, 0, 255), 2);
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
            if (mZoomWindow != null)
                mZoomWindow.dispose();
            if (mZoomCorner != null)
                mZoomCorner.dispose();
            if (mBlurWindow != null)
                mBlurWindow.dispose();
            if (mGrayInnerWindow != null)
                mGrayInnerWindow.dispose();
            if (mRgbaInnerWindow != null)
                mRgbaInnerWindow.dispose();
            if (mRgba != null)
                mRgba.dispose();
            if (mGray != null)
                mGray.dispose();
            if (mIntermediateMat != null)
                mIntermediateMat.dispose();

            mRgba = null;
            mGray = null;
            mIntermediateMat = null;
            mRgbaInnerWindow = null;
            mGrayInnerWindow = null;
            mBlurWindow = null;
            mZoomCorner = null;
            mZoomWindow = null;
        }
    }
}