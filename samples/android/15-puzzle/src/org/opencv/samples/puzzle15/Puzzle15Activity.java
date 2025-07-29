package org.opencv.samples.puzzle15;

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
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import java.util.Collections;
import java.util.List;

public class Puzzle15Activity extends CameraActivity implements CvCameraViewListener, View.OnTouchListener {

    private static final String  TAG = "Puzzle15::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private Puzzle15Processor    mPuzzle15;
    private MenuItem             mItemHideNumbers;
    private MenuItem             mItemStartNewGame;

    private int                  mGameWidth;
    private int                  mGameHeight;

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
        mOpenCvCameraView = (CameraBridgeViewBase) new JavaCameraView(this, -1);
        setContentView(mOpenCvCameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mPuzzle15 = new Puzzle15Processor();
        mPuzzle15.prepareNewGame();
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
            mOpenCvCameraView.setOnTouchListener(Puzzle15Activity.this);
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
        mItemHideNumbers = menu.add("Show/hide tile numbers");
        mItemStartNewGame = menu.add("Start new game");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemStartNewGame) {
            /* We need to start new game */
            mPuzzle15.prepareNewGame();
        } else if (item == mItemHideNumbers) {
            /* We need to enable or disable drawing of the tile numbers */
            mPuzzle15.toggleTileNumbers();
        }
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
        mGameWidth = width;
        mGameHeight = height;
        mPuzzle15.prepareGameSize(width, height);
    }

    public void onCameraViewStopped() {
    }

    public boolean onTouch(View view, MotionEvent event) {
        int xpos, ypos;

        xpos = (view.getWidth() - mGameWidth) / 2;
        xpos = (int)event.getX() - xpos;

        ypos = (view.getHeight() - mGameHeight) / 2;
        ypos = (int)event.getY() - ypos;

        if (xpos >=0 && xpos <= mGameWidth && ypos >=0  && ypos <= mGameHeight) {
            /* click is inside the picture. Deliver this event to processor */
            mPuzzle15.deliverTouchEvent(xpos, ypos);
        }

        return false;
    }

    public Mat onCameraFrame(Mat inputFrame) {
        return mPuzzle15.puzzleFrame(inputFrame);
    }
}
