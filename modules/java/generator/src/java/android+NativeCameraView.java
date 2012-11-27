package org.opencv.android;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;

/**
 * This class is an implementation of a bridge between SurfaceView and native OpenCV camera.
 * Due to the big amount of work done, by the base class this child is only responsible
 * for creating camera, destroying camera and delivering frames while camera is enabled
 */
public class NativeCameraView extends CameraBridgeViewBase {

    public static final String TAG = "NativeCameraView";
    private boolean mStopThread;
    private Thread mThread;

    protected VideoCapture mCamera;

    public NativeCameraView(Context context, int cameraId) {
        super(context, cameraId);
    }

    public NativeCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    @Override
    protected boolean connectCamera(int width, int height) {

        /* 1. We need to instantiate camera
         * 2. We need to start thread which will be getting frames
         */
        /* First step - initialize camera connection */
        if (!initializeCamera(getWidth(), getHeight()))
            return false;

        /* now we can start update thread */
        mThread = new Thread(new CameraWorker());
        mThread.start();

        return true;
    }

    @Override
    protected void disconnectCamera() {
        /* 1. We need to stop thread which updating the frames
         * 2. Stop camera and release it
         */
        try {
            mStopThread = true;
            mThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            mThread =  null;
            mStopThread = false;
        }

        /* Now release camera */
        releaseCamera();
    }

    public static class OpenCvSizeAccessor implements ListItemAccessor {

        public int getWidth(Object obj) {
            Size size  = (Size)obj;
            return (int)size.width;
        }

        public int getHeight(Object obj) {
            Size size  = (Size)obj;
            return (int)size.height;
        }

    }

    private boolean initializeCamera(int width, int height) {
        synchronized (this) {

            if (mCameraIndex == -1)
                mCamera = new VideoCapture(Highgui.CV_CAP_ANDROID);
            else
                mCamera = new VideoCapture(Highgui.CV_CAP_ANDROID + mCameraIndex);

            if (mCamera == null)
                return false;

            if (mCamera.isOpened() == false)
                return false;

            java.util.List<Size> sizes = mCamera.getSupportedPreviewSizes();

            /* Select the size that fits surface considering maximum size allowed */
            Size frameSize = calculateCameraFrameSize(sizes, new OpenCvSizeAccessor(), width, height);

            mFrameWidth = (int)frameSize.width;
            mFrameHeight = (int)frameSize.height;

            if (mFpsMeter != null) {
                mFpsMeter.setResolution(mFrameWidth, mFrameHeight);
            }

            AllocateCache();

            mCamera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, frameSize.width);
            mCamera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, frameSize.height);
        }

        Log.i(TAG, "Selected camera frame size = (" + mFrameWidth + ", " + mFrameHeight + ")");

        return true;
    }

    private void releaseCamera() {
        synchronized (this) {
            if (mCamera != null) {
                mCamera.release();
            }
        }
    }

    private class CameraWorker implements Runnable {

        private Mat mRgba = new Mat();
        private Mat mGray = new Mat();

        public void run() {
            do {
                if (!mCamera.grab()) {
                    Log.e(TAG, "Camera frame grab failed");
                    break;
                }

                switch (mPreviewFormat) {
                    case Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA:
                    {
                        mCamera.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
                        deliverAndDrawFrame(mRgba);
                    } break;
                    case Highgui.CV_CAP_ANDROID_GREY_FRAME:
                        mCamera.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
                        deliverAndDrawFrame(mGray);
                        break;
                    default:
                        Log.e(TAG, "Invalid frame format! Only RGBA and Gray Scale are supported!");
                }

            } while (!mStopThread);

        }
    }

}
