package org.opencv.test.camerawriter;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.test.camerawriter.OpenCvCameraBridgeViewBase.CvCameraViewListener;

import android.os.Bundle;
import android.app.Activity;
import android.util.Log;

public class CameraWriterActivity extends Activity implements CvCameraViewListener {

    protected static final String TAG = "CameraWriterActivity";


    private OpenCvCameraBridgeViewBase mCameraView;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
           switch (status) {
               case LoaderCallbackInterface.SUCCESS:
                  Log.i(TAG, "OpenCV loaded successfully");
                  // Create and set View
                  mCameraView.setMaxFrameSize(640, 480);
                  mCameraView.enableView();
                  break;
               default:
              super.onManagerConnected(status);
                break;
           }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_writer);

        mCameraView = (OpenCvCameraBridgeViewBase)findViewById(R.id.camera_surface_view);
        mCameraView.setCvCameraViewListener(this);
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mLoaderCallback);

    }

    public void onCameraViewStarted(int width, int height) {
        // TODO Auto-generated method stub

    }

    public void onCameraViewStopped() {
        // TODO Auto-generated method stub

    }

    public Mat onCameraFrame(Mat inputFrame) {
        return inputFrame;
    }
}
