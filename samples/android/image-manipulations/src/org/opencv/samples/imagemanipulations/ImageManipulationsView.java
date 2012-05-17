package org.opencv.samples.imagemanipulations;

import java.util.Arrays;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

class ImageManipulationsView extends SampleCvViewBase {
    private Size mSize0;
    private Size mSizeRgba;
    private Size mSizeRgbaInner;
    
    private Mat mRgba;
    private Mat mGray;
    private Mat mIntermediateMat;
    private Mat mHist, mMat0;
    private MatOfInt mChannels[], mHistSize;
    private int mHistSizeNum;
    private MatOfFloat mRanges;
    private Scalar mColorsRGB[], mColorsHue[], mWhilte;
    private Point mP1, mP2;
    float mBuff[];

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
	public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
            mRgba = new Mat();
            mIntermediateMat = new Mat();
            mSize0 = new Size();
            mHist = new Mat();
            mChannels = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };
            mHistSizeNum = 25;
            mBuff = new float[mHistSizeNum];
            mHistSize = new MatOfInt(mHistSizeNum);
            mRanges = new MatOfFloat(0f, 256f);
            mMat0  = new Mat();
            mColorsRGB = new Scalar[] { new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255) };
            mColorsHue = new Scalar[] {
            		new Scalar(255, 0, 0, 255),   new Scalar(255, 60, 0, 255),  new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
            		new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255),  new Scalar(20, 255, 0, 255),  new Scalar(0, 255, 30, 255),
            		new Scalar(0, 255, 85, 255),  new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
            		new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255),  new Scalar(0, 0, 255, 255),   new Scalar(64, 0, 255, 255),  new Scalar(120, 0, 255, 255),
            		new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255),  new Scalar(255, 0, 0, 255)
            };
            mWhilte = Scalar.all(255);
            mP1 = new Point();
            mP2 = new Point();
        }

        super.surfaceCreated(holder);
	}

	private void CreateAuxiliaryMats() {
        if (mRgba.empty())
            return;

        mSizeRgba = mRgba.size(); 

        int rows = (int) mSizeRgba.height;
        int cols = (int) mSizeRgba.width;

        int left = cols / 8;
        int top = rows / 8;

        int width = cols * 3 / 4;
        int height = rows * 3 / 4;

        if (mRgbaInnerWindow == null)
            mRgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
        mSizeRgbaInner = mRgbaInnerWindow.size(); 

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

        case ImageManipulationsActivity.VIEW_MODE_HIST:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            if (mSizeRgba == null)
                CreateAuxiliaryMats();
            int thikness = (int) (mSizeRgba.width / (mHistSizeNum + 10) / 5);
            if(thikness > 5) thikness = 5;
            int offset = (int) ((mSizeRgba.width - (5*mHistSizeNum + 4*10)*thikness)/2);
            // RGB
            for(int c=0; c<3; c++) {
            	Imgproc.calcHist(Arrays.asList(mRgba), mChannels[c], mMat0, mHist, mHistSize, mRanges);
            	Core.normalize(mHist, mHist, mSizeRgba.height/2, 0, Core.NORM_INF);
            	mHist.get(0, 0, mBuff);
            	for(int h=0; h<mHistSizeNum; h++) {
            		mP1.x = mP2.x = offset + (c * (mHistSizeNum + 10) + h) * thikness; 
            		mP1.y = mSizeRgba.height-1;
            		mP2.y = mP1.y - 2 - (int)mBuff[h]; 
            		Core.line(mRgba, mP1, mP2, mColorsRGB[c], thikness);
            	}
            }
            // Value and Hue
            Imgproc.cvtColor(mRgba, mIntermediateMat, Imgproc.COLOR_RGB2HSV_FULL);
            // Value
            Imgproc.calcHist(Arrays.asList(mIntermediateMat), mChannels[2], mMat0, mHist, mHistSize, mRanges);
        	Core.normalize(mHist, mHist, mSizeRgba.height/2, 0, Core.NORM_INF);
        	mHist.get(0, 0, mBuff);
        	for(int h=0; h<mHistSizeNum; h++) {
        		mP1.x = mP2.x = offset + (3 * (mHistSizeNum + 10) + h) * thikness; 
        		mP1.y = mSizeRgba.height-1;
        		mP2.y = mP1.y - 2 - (int)mBuff[h]; 
        		Core.line(mRgba, mP1, mP2, mWhilte, thikness);
        	}
            // Hue
            Imgproc.calcHist(Arrays.asList(mIntermediateMat), mChannels[0], mMat0, mHist, mHistSize, mRanges);
        	Core.normalize(mHist, mHist, mSizeRgba.height/2, 0, Core.NORM_INF);
        	mHist.get(0, 0, mBuff);
        	for(int h=0; h<mHistSizeNum; h++) {
        		mP1.x = mP2.x = offset + (4 * (mHistSizeNum + 10) + h) * thikness; 
        		mP1.y = mSizeRgba.height-1;
        		mP2.y = mP1.y - 2 - (int)mBuff[h]; 
        		Core.line(mRgba, mP1, mP2, mColorsHue[h], thikness);
        	}
            break;

        case ImageManipulationsActivity.VIEW_MODE_CANNY:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);

            if (mRgbaInnerWindow == null || mGrayInnerWindow == null)
                CreateAuxiliaryMats();
            Imgproc.Canny(mRgbaInnerWindow, mIntermediateMat, 80, 90);
            Imgproc.cvtColor(mIntermediateMat, mRgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            break;

        case ImageManipulationsActivity.VIEW_MODE_SOBEL:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);

            if (mRgbaInnerWindow == null || mGrayInnerWindow == null)
                CreateAuxiliaryMats();

            Imgproc.Sobel(mGrayInnerWindow, mIntermediateMat, CvType.CV_8U, 1, 1);
            Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 10, 0);
            Imgproc.cvtColor(mIntermediateMat, mRgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            break;

        case ImageManipulationsActivity.VIEW_MODE_SEPIA:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            Core.transform(mRgba, mRgba, mSepiaKernel);
            break;

        case ImageManipulationsActivity.VIEW_MODE_ZOOM:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            if (mZoomCorner == null || mZoomWindow == null)
                CreateAuxiliaryMats();
            Imgproc.resize(mZoomWindow, mZoomCorner, mZoomCorner.size());

            Size wsize = mZoomWindow.size();
            Core.rectangle(mZoomWindow, new Point(1, 1), new Point(wsize.width - 2, wsize.height - 2), new Scalar(255, 0, 0, 255), 2);
            break;

        case ImageManipulationsActivity.VIEW_MODE_PIXELIZE:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            if (mRgbaInnerWindow == null)
                CreateAuxiliaryMats();
            Imgproc.resize(mRgbaInnerWindow, mIntermediateMat, mSize0, 0.1, 0.1, Imgproc.INTER_NEAREST);
            Imgproc.resize(mIntermediateMat, mRgbaInnerWindow, mSizeRgbaInner, 0., 0., Imgproc.INTER_NEAREST);
            break;

        case ImageManipulationsActivity.VIEW_MODE_POSTERIZE:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            if (mRgbaInnerWindow == null)
                CreateAuxiliaryMats();
            /*
            Imgproc.cvtColor(mRgbaInnerWindow, mIntermediateMat, Imgproc.COLOR_RGBA2RGB);
            Imgproc.pyrMeanShiftFiltering(mIntermediateMat, mIntermediateMat, 5, 50);
            Imgproc.cvtColor(mIntermediateMat, mRgbaInnerWindow, Imgproc.COLOR_RGB2RGBA);
            */
            Imgproc.Canny(mRgbaInnerWindow, mIntermediateMat, 80, 90);
            mRgbaInnerWindow.setTo(new Scalar(0, 0, 0, 255), mIntermediateMat);
            Core.convertScaleAbs(mRgbaInnerWindow, mIntermediateMat, 1./16, 0);
            Core.convertScaleAbs(mIntermediateMat, mRgbaInnerWindow, 16, 0);
            break;
        }

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
            if (mZoomWindow != null)
                mZoomWindow.release();
            if (mZoomCorner != null)
                mZoomCorner.release();
            if (mBlurWindow != null)
                mBlurWindow.release();
            if (mGrayInnerWindow != null)
                mGrayInnerWindow.release();
            if (mRgbaInnerWindow != null)
                mRgbaInnerWindow.release();
            if (mRgba != null)
                mRgba.release();
            if (mGray != null)
                mGray.release();
            if (mIntermediateMat != null)
                mIntermediateMat.release();

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
