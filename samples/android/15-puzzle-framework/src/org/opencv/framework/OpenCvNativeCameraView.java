package org.opencv.framework;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.util.Log;

/**
 * This class is an implementation of a bridge between SurfaceView and native OpenCV camera.
 * Due to the big amount of work done, by the base class this child is only responsible
 * for creating camera, destroying camera and delivering frames while camera is enabled
 */
public class OpenCvNativeCameraView extends OpenCvCameraBridgeViewBase {

        public static final String TAG = "OpenCvNativeCameraView";
        private boolean mStopThread;
        private Thread mThread;
        private VideoCapture mCamera;


    public OpenCvNativeCameraView(Context context, AttributeSet attrs) {
                super(context, attrs);
        }


        @Override
        protected void connectCamera(int width, int height) {

                /* 1. We need to instantiate camera
                 * 2. We need to start thread which will be getting frames
                 */
                /* First step - initialize camera connection */
                initializeCamera(getWidth(), getHeight());

                /* now we can start update thread */
                mThread = new Thread(new CameraWorker(getWidth(), getHeight()));
                mThread.start();
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

                @Override
                public int getWidth(Object obj) {
                        Size size  = (Size)obj;
                        return (int)size.width;
                }

                @Override
                public int getHeight(Object obj) {
                        Size size  = (Size)obj;
                        return (int)size.height;
                }

        }

        private void initializeCamera(int width, int height) {
        mCamera = new VideoCapture(Highgui.CV_CAP_ANDROID);
        //TODO: improve error handling

        java.util.List<Size> sizes = mCamera.getSupportedPreviewSizes();

        /* Select the size that fits surface considering maximum size allowed */
        FrameSize frameSize = calculateCameraFrameSize(sizes, new OpenCvSizeAccessor(), width, height);


        double frameWidth = frameSize.width;
        double frameHeight = frameSize.height;


        mCamera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, frameWidth);
        mCamera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, frameHeight);

        mFrameWidth = (int)frameWidth;
        mFrameHeight = (int)frameHeight;

        Log.i(TAG, "Selected camera frame size = (" + mFrameWidth + ", " + mFrameHeight + ")");
        }

        private void releaseCamera() {
                if (mCamera != null) {
                        mCamera.release();
                }
        }

        private class CameraWorker implements Runnable {

            private Mat mRgba = new Mat();
            private int mWidth;
            private int mHeight;

            CameraWorker(int w, int h) {
		mWidth = w;
		mHeight = h;
            }

                @Override
                public void run() {
                        Mat modified;


                        do {
                                if (!mCamera.grab()) {
                                        Log.e(TAG, "Camera frame grab failed");
                                        break;
                                }
                    mCamera.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);

                    deliverAndDrawFrame(mRgba);

                        } while (!mStopThread);

                }
        }

}
