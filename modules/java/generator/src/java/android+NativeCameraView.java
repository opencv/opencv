package org.opencv.android;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.ViewGroup.LayoutParams;

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
    protected NativeCameraFrame mFrame;

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
        if (!initializeCamera(width, height))
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
        if (mThread != null) {
            try {
                mStopThread = true;
                mThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                mThread =  null;
                mStopThread = false;
            }
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

            mFrame = new NativeCameraFrame(mCamera);

            java.util.List<Size> sizes = mCamera.getSupportedPreviewSizes();

            /* Select the size that fits surface considering maximum size allowed */
            Size frameSize = calculateCameraFrameSize(sizes, new OpenCvSizeAccessor(), width, height);

            mFrameWidth = (int)frameSize.width;
            mFrameHeight = (int)frameSize.height;

            if ((getLayoutParams().width == LayoutParams.MATCH_PARENT) && (getLayoutParams().height == LayoutParams.MATCH_PARENT))
                mScale = Math.min(((float)height)/mFrameHeight, ((float)width)/mFrameWidth);
            else
                mScale = 0;

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
            if (mFrame != null) mFrame.release();
            if (mCamera != null) mCamera.release();
        }
    }

    private static class NativeCameraFrame implements CvCameraViewFrame {

        @Override
        public Mat rgba() {
            mCapture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            return mRgba;
        }

        @Override
        public Mat gray() {
            mCapture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
            return mGray;
        }

        public NativeCameraFrame(VideoCapture capture) {
            mCapture = capture;
            mGray = new Mat();
            mRgba = new Mat();
        }

        public void release() {
            if (mGray != null) mGray.release();
            if (mRgba != null) mRgba.release();
        }

        private VideoCapture mCapture;
        private Mat mRgba;
        private Mat mGray;
    };

    private class CameraWorker implements Runnable {

        public void run() {
            do {
                if (!mCamera.grab()) {
                    Log.e(TAG, "Camera frame grab failed");
                    break;
                }

                deliverAndDrawFrame(mFrame);

            } while (!mStopThread);
        }
    }

}
