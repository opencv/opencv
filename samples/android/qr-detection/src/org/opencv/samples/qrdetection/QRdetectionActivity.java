package org.opencv.samples.qrdetection;

import org.opencv.android.CameraActivity;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.JavaCameraView;

import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import java.util.Collections;
import java.util.List;

public class QRdetectionActivity extends CameraActivity implements CvCameraViewListener {

    private static final String  TAG = "QRdetection::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private QRProcessor    mQRDetector;
    private MenuItem             mItemQRCodeDetectorAruco;
    private MenuItem             mItemQRCodeDetector;
    private MenuItem             mItemTryDecode;
    private MenuItem             mItemMulti;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            return;
        }

        Log.d(TAG, "Creating and setting view");
        mOpenCvCameraView = new JavaCameraView(this, -1);
        setContentView(mOpenCvCameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mQRDetector = new QRProcessor(true);
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
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.enableView();
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemQRCodeDetectorAruco = menu.add("Aruco-based QR code detector");
        mItemQRCodeDetectorAruco.setCheckable(true);
        mItemQRCodeDetectorAruco.setChecked(true);

        mItemQRCodeDetector = menu.add("Legacy QR code detector");
        mItemQRCodeDetector.setCheckable(true);
        mItemQRCodeDetector.setChecked(false);

        mItemTryDecode = menu.add("Try to decode QR codes");
        mItemTryDecode.setCheckable(true);
        mItemTryDecode.setChecked(true);

        mItemMulti = menu.add("Use multi detect/decode");
        mItemMulti.setCheckable(true);
        mItemMulti.setChecked(true);

        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemQRCodeDetector && !mItemQRCodeDetector.isChecked()) {
            mQRDetector = new QRProcessor(false);
            mItemQRCodeDetector.setChecked(true);
            mItemQRCodeDetectorAruco.setChecked(false);
        } else if (item == mItemQRCodeDetectorAruco && !mItemQRCodeDetectorAruco.isChecked()) {
            mQRDetector = new QRProcessor(true);
            mItemQRCodeDetector.setChecked(false);
            mItemQRCodeDetectorAruco.setChecked(true);
        } else if (item == mItemTryDecode) {
            mItemTryDecode.setChecked(!mItemTryDecode.isChecked());
        } else if (item == mItemMulti) {
            mItemMulti.setChecked(!mItemMulti.isChecked());
        }
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(Mat inputFrame) {
        return mQRDetector.handleFrame(inputFrame, mItemTryDecode.isChecked(), mItemMulti.isChecked());
    }
}
