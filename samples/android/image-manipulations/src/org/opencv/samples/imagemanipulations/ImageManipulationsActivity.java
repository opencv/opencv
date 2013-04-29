package org.opencv.samples.imagemanipulations;

import java.util.Arrays;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class ImageManipulationsActivity extends Activity implements CvCameraViewListener2 {
    private static final String  TAG                 = "OCVSample::Activity";

    public static final int      VIEW_MODE_RGBA      = 0;
    public static final int      VIEW_MODE_HIST      = 1;
    public static final int      VIEW_MODE_CANNY     = 2;
    public static final int      VIEW_MODE_SEPIA     = 3;
    public static final int      VIEW_MODE_SOBEL     = 4;
    public static final int      VIEW_MODE_ZOOM      = 5;
    public static final int      VIEW_MODE_PIXELIZE  = 6;
    public static final int      VIEW_MODE_POSTERIZE = 7;

    private MenuItem             mItemPreviewRGBA;
    private MenuItem             mItemPreviewHist;
    private MenuItem             mItemPreviewCanny;
    private MenuItem             mItemPreviewSepia;
    private MenuItem             mItemPreviewSobel;
    private MenuItem             mItemPreviewZoom;
    private MenuItem             mItemPreviewPixelize;
    private MenuItem             mItemPreviewPosterize;
    private CameraBridgeViewBase mOpenCvCameraView;

    private Size                 mSize0;
    private Size                 mSizeRgba;
    private Size                 mSizeRgbaInner;

    private Mat                  mRgba;
    private Mat                  mGray;
    private Mat                  mIntermediateMat;
    private Mat                  mHist;
    private Mat                  mMat0;
    private MatOfInt             mChannels[];
    private MatOfInt             mHistSize;
    private int                  mHistSizeNum;
    private MatOfFloat           mRanges;
    private Scalar               mColorsRGB[];
    private Scalar               mColorsHue[];
    private Scalar               mWhilte;
    private Point                mP1;
    private Point                mP2;
    private float                mBuff[];
    private Mat                  mRgbaInnerWindow;
    private Mat                  mGrayInnerWindow;
    private Mat                  mZoomWindow;
    private Mat                  mZoomCorner;
    private Mat                  mSepiaKernel;

    public static int           viewMode = VIEW_MODE_RGBA;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public ImageManipulationsActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.image_manipulations_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.image_manipulations_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA  = menu.add("Preview RGBA");
        mItemPreviewHist  = menu.add("Histograms");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewSepia = menu.add("Sepia");
        mItemPreviewSobel = menu.add("Sobel");
        mItemPreviewZoom  = menu.add("Zoom");
        mItemPreviewPixelize  = menu.add("Pixelize");
        mItemPreviewPosterize = menu.add("Posterize");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        if (item == mItemPreviewHist)
            viewMode = VIEW_MODE_HIST;
        else if (item == mItemPreviewCanny)
            viewMode = VIEW_MODE_CANNY;
        else if (item == mItemPreviewSepia)
            viewMode = VIEW_MODE_SEPIA;
        else if (item == mItemPreviewSobel)
            viewMode = VIEW_MODE_SOBEL;
        else if (item == mItemPreviewZoom)
            viewMode = VIEW_MODE_ZOOM;
        else if (item == mItemPreviewPixelize)
            viewMode = VIEW_MODE_PIXELIZE;
        else if (item == mItemPreviewPosterize)
            viewMode = VIEW_MODE_POSTERIZE;
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
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

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
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

        if (mZoomCorner == null)
            mZoomCorner = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);

        if (mZoomWindow == null)
            mZoomWindow = mRgba.submat(rows / 2 - 9 * rows / 100, rows / 2 + 9 * rows / 100, cols / 2 - 9 * cols / 100, cols / 2 + 9 * cols / 100);
    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mZoomWindow != null)
            mZoomWindow.release();
        if (mZoomCorner != null)
            mZoomCorner.release();
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
        mZoomCorner = null;
        mZoomWindow = null;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        switch (ImageManipulationsActivity.viewMode) {
        case ImageManipulationsActivity.VIEW_MODE_RGBA:
            break;

        case ImageManipulationsActivity.VIEW_MODE_HIST:
            if ((mSizeRgba == null) || (mRgba.cols() != mSizeRgba.width) || (mRgba.height() != mSizeRgba.height))
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
             if ((mRgbaInnerWindow == null) || (mGrayInnerWindow == null) || (mRgba.cols() != mSizeRgba.width) || (mRgba.height() != mSizeRgba.height))
                CreateAuxiliaryMats();
            Imgproc.Canny(mRgbaInnerWindow, mIntermediateMat, 80, 90);
            Imgproc.cvtColor(mIntermediateMat, mRgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            break;

        case ImageManipulationsActivity.VIEW_MODE_SOBEL:
            mGray = inputFrame.gray();

            if ((mRgbaInnerWindow == null) || (mGrayInnerWindow == null) || (mRgba.cols() != mSizeRgba.width) || (mRgba.height() != mSizeRgba.height))
                CreateAuxiliaryMats();

            Imgproc.Sobel(mGrayInnerWindow, mIntermediateMat, CvType.CV_8U, 1, 1);
            Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 10, 0);
            Imgproc.cvtColor(mIntermediateMat, mRgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            break;

        case ImageManipulationsActivity.VIEW_MODE_SEPIA:
            if ((mRgbaInnerWindow == null) || (mRgba.cols() != mSizeRgba.width) || (mRgba.height() != mSizeRgba.height))
                CreateAuxiliaryMats();
            Core.transform(mRgbaInnerWindow, mRgbaInnerWindow, mSepiaKernel);
            break;

        case ImageManipulationsActivity.VIEW_MODE_ZOOM:
            if ((mZoomCorner == null) || (mZoomWindow == null) || (mRgba.cols() != mSizeRgba.width) || (mRgba.height() != mSizeRgba.height))
                CreateAuxiliaryMats();
            Imgproc.resize(mZoomWindow, mZoomCorner, mZoomCorner.size());

            Size wsize = mZoomWindow.size();
            Core.rectangle(mZoomWindow, new Point(1, 1), new Point(wsize.width - 2, wsize.height - 2), new Scalar(255, 0, 0, 255), 2);
            break;

        case ImageManipulationsActivity.VIEW_MODE_PIXELIZE:
            if ((mRgbaInnerWindow == null) || (mRgba.cols() != mSizeRgba.width) || (mRgba.height() != mSizeRgba.height))
                CreateAuxiliaryMats();
            Imgproc.resize(mRgbaInnerWindow, mIntermediateMat, mSize0, 0.1, 0.1, Imgproc.INTER_NEAREST);
            Imgproc.resize(mIntermediateMat, mRgbaInnerWindow, mSizeRgbaInner, 0., 0., Imgproc.INTER_NEAREST);
            break;

        case ImageManipulationsActivity.VIEW_MODE_POSTERIZE:
            if ((mRgbaInnerWindow == null) || (mRgba.cols() != mSizeRgba.width) || (mRgba.height() != mSizeRgba.height))
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

        return mRgba;
    }
}
