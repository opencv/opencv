package org.opencv.samples.tutorial4;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.framework.OpenCvJavaCameraView;
import org.opencv.framework.OpenCvCameraBridgeViewBase.CvCameraViewListener;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;
import android.view.WindowManager;

public class Sample4Mixed extends Activity implements CvCameraViewListener {
    private static final String TAG = "OCVSample::Activity";

    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_CANNY    = 2;
    private static final int       VIEW_MODE_FEATURES = 5;

    private int                    mViewMode;
    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGrayMat;

    private MenuItem               mItemPreviewRGBA;
    private MenuItem               mItemPreviewGray;
    private MenuItem               mItemPreviewCanny;
    private MenuItem               mItemPreviewFeatures;

    private OpenCvJavaCameraView mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("mixed_sample");

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public Sample4Mixed() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial4_surface_view);

        mOpenCvCameraView = (OpenCvJavaCameraView)findViewById(R.id.tutorial4_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewFeatures = menu.add("Find features");
        return true;
    }

    @Override
    public void onPause()
    {
        mOpenCvCameraView.disableView();
        super.onPause();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGrayMat = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGrayMat.release();
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(Mat inputFrame) {
        final int viewMode = mViewMode;

        switch (viewMode) {
        case VIEW_MODE_GRAY:
            // input frame has gray scale format
            Imgproc.cvtColor(inputFrame, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            break;
        case VIEW_MODE_RGBA:
            // input frame has RBGA format
            inputFrame.copyTo(mRgba);
            break;
        case VIEW_MODE_CANNY:
            // input frame has gray scale format
            Imgproc.Canny(inputFrame, mIntermediateMat, 80, 100);
            Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2BGRA, 4);
            break;
        case VIEW_MODE_FEATURES:
            // input frame has RGBA format
            inputFrame.copyTo(mRgba);
            Imgproc.cvtColor(mRgba, mGrayMat, Imgproc.COLOR_RGBA2GRAY);
            FindFeatures(mGrayMat.getNativeObjAddr(), mRgba.getNativeObjAddr());
            break;
        }

        return mRgba;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemPreviewRGBA) {
            mOpenCvCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            mViewMode = VIEW_MODE_RGBA;
        } else if (item == mItemPreviewGray) {
            mOpenCvCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_GREY_FRAME);
            mViewMode = VIEW_MODE_GRAY;
        } else if (item == mItemPreviewCanny) {
            mOpenCvCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_GREY_FRAME);
            mViewMode = VIEW_MODE_CANNY;
        } else if (item == mItemPreviewFeatures) {
            mViewMode = VIEW_MODE_FEATURES;
            mOpenCvCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        }

        return true;
    }

    public native void FindFeatures(long matAddrGr, long matAddrRgba);
}
