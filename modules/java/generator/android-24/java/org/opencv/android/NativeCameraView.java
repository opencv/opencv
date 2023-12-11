package org.opencv.android;

import org.opencv.core.Mat;
import org.opencv.core.Size;

import org.opencv.imgproc.Imgproc;

import org.opencv.videoio.Videoio;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.ViewGroup.LayoutParams;

/**
 * This class is an implementation of a bridge between SurfaceView and OpenCV VideoCapture.
 * The class  is experimental implementation and not recoomended for production usage.
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

            if (mCameraIndex == -1) {
                Log.d(TAG, "Try to open default camera");
                mCamera = new VideoCapture(0, Videoio.CAP_ANDROID);
            } else {
                Log.d(TAG, "Try to open camera with index " + mCameraIndex);
                mCamera = new VideoCapture(mCameraIndex, Videoio.CAP_ANDROID);
            }

            if (mCamera == null)
                return false;

            if (mCamera.isOpened() == false)
                return false;

            mFrame = new NativeCameraFrame(mCamera);

            mCamera.set(Videoio.CAP_PROP_FRAME_WIDTH, width);
            mCamera.set(Videoio.CAP_PROP_FRAME_HEIGHT, height);

            mFrameWidth = (int)mCamera.get(Videoio.CAP_PROP_FRAME_WIDTH);
            mFrameHeight = (int)mCamera.get(Videoio.CAP_PROP_FRAME_HEIGHT);

            if ((getLayoutParams().width == LayoutParams.MATCH_PARENT) && (getLayoutParams().height == LayoutParams.MATCH_PARENT))
                mScale = Math.min(((float)height)/mFrameHeight, ((float)width)/mFrameWidth);
            else
                mScale = 0;

            if (mFpsMeter != null) {
                mFpsMeter.setResolution(mFrameWidth, mFrameHeight);
            }

            AllocateCache();
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
            mCapture.set(Videoio.CAP_PROP_FOURCC, VideoWriter.fourcc('R','G','B','3'));
            mCapture.retrieve(mBgr);
            Log.d(TAG, "Retrived frame with size " + mBgr.cols() + "x" + mBgr.rows() + " and channels: " + mBgr.channels());
            Imgproc.cvtColor(mBgr, mRgba, Imgproc.COLOR_RGB2RGBA);
            return mRgba;
        }

        @Override
        public Mat gray() {
            mCapture.set(Videoio.CAP_PROP_FOURCC, VideoWriter.fourcc('G','R','E','Y'));
            mCapture.retrieve(mGray);
            Log.d(TAG, "Retrived frame with size " + mGray.cols() + "x" + mGray.rows() + " and channels: " + mGray.channels());
            return mGray;
        }

        public NativeCameraFrame(VideoCapture capture) {
            mCapture = capture;
            mGray = new Mat();
            mRgba = new Mat();
            mBgr = new Mat();
        }

        public void release() {
            if (mGray != null) mGray.release();
            if (mRgba != null) mRgba.release();
            if (mBgr != null) mBgr.release();
        }

        private VideoCapture mCapture;
        private Mat mRgba;
        private Mat mGray;
        private Mat mBgr;
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
