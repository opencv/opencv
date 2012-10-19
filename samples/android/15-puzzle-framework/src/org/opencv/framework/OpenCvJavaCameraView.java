package org.opencv.framework;

import java.io.IOException;
import java.util.List;

import android.annotation.TargetApi;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * This class is an implementation of the Bridge View between OpenCv and JAVA Camera.
 * This class relays on the functionality available in base class and only implements
 * required functions:
 * connectCamera - opens Java camera and sets the PreviewCallback to be delivered.
 * disconnectCamera - closes the camera and stops preview.
 * When frame is delivered via callback from Camera - it processed via OpenCV to be
 * converted to RGBA32 and then passed to the external callback for modifications if required.
 */
public class OpenCvJavaCameraView extends OpenCvCameraBridgeViewBase implements PreviewCallback {

    private static final int MAGIC_TEXTURE_ID = 10;
    private static final String TAG = "OpenCvJavaCameraView";

    private Mat mBaseMat;
    private byte mBuffer[];

    private Thread mThread;
    private boolean mStopThread;

    public static class JavaCameraSizeAccessor implements ListItemAccessor {

        @Override
        public int getWidth(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.width;
        }

        @Override
        public int getHeight(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.height;
        }
    }

    private Camera mCamera;

    public OpenCvJavaCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    @TargetApi(11)
    protected boolean initializeCamera(int width, int height) {
        synchronized (this) {
            mCamera = null;

            try {
                mCamera = Camera.open();
            }
            catch (Exception e){
                Log.e(TAG, "Camera is not available (in use or does not exist): " + e.getLocalizedMessage());
            }

            if(mCamera == null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
                for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                    try {
                        mCamera = Camera.open(camIdx);
                    }
                    catch (RuntimeException e) {
                        Log.e(TAG, "Camera #" + camIdx + "failed to open: " + e.getLocalizedMessage());
                    }
                }
            }

            if (mCamera == null)
                return false;

            mCamera.setPreviewCallbackWithBuffer(this);

            List<android.hardware.Camera.Size> sizes = mCamera.getParameters().getSupportedPreviewSizes();
            /* Select the size that fits surface considering maximum size allowed */
            Size frameSize = calculateCameraFrameSize(sizes, new JavaCameraSizeAccessor(), width, height);

            mFrameWidth = (int)frameSize.width;
            mFrameHeight = (int)frameSize.height;

            /* Now set camera parameters */
            try {
                Camera.Parameters params = mCamera.getParameters();

                params.setPreviewFormat(ImageFormat.NV21);
                params.setPreviewSize((int)frameSize.width, (int)frameSize.height);

                List<String> FocusModes = params.getSupportedFocusModes();
                if (FocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO))
                {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                }

                mCamera.setParameters(params);
                params = mCamera.getParameters();

                int size = params.getPreviewSize().width * params.getPreviewSize().height;
                size  = size * ImageFormat.getBitsPerPixel(params.getPreviewFormat()) / 8;
                mBuffer = new byte[size];

                mCamera.addCallbackBuffer(mBuffer);

                mBaseMat = new Mat(mFrameHeight + (mFrameHeight/2), mFrameWidth, CvType.CV_8UC1);

                AllocateCache();

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
                        SurfaceTexture tex = new SurfaceTexture(MAGIC_TEXTURE_ID);
                        getHolder().setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
                    mCamera.setPreviewTexture(tex);
                } else
                   mCamera.setPreviewDisplay(null);
            } catch (IOException e) {
                e.printStackTrace();
            }

            /* Finally we are ready to start the preview */
            mCamera.startPreview();
        }

        return true;
    }

    protected void releaseCamera() {
        synchronized (this) {
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
        }
    }

    @Override
    protected boolean connectCamera(int width, int height) {

        /* 1. We need to instantiate camera
         * 2. We need to start thread which will be getting frames
         */
        /* First step - initialize camera connection */
        Log.d(TAG, "Connecting to camera");
        if (!initializeCamera(getWidth(), getHeight()))
            return false;

        /* now we can start update thread */
        Log.d(TAG, "Starting processing thread");
        mStopThread = false;
        mThread = new Thread(new CameraWorker(getWidth(), getHeight()));
        mThread.start();

        return true;
    }

    protected void disconnectCamera() {
        /* 1. We need to stop thread which updating the frames
         * 2. Stop camera and release it
         */
        Log.d(TAG, "Disconnecting from camera");
        try {
            mStopThread = true;
            Log.d(TAG, "Notify thread");
            synchronized (this) {
                this.notify();
                        }
            Log.d(TAG, "Wating for thread");
            mThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            mThread =  null;
        }

        /* Now release camera */
        releaseCamera();
    }

    @Override
    public void onPreviewFrame(byte[] frame, Camera arg1) {
        Log.i(TAG, "Preview Frame received. Need to create MAT and deliver it to clients");
        Log.i(TAG, "Frame size  is " + frame.length);
        synchronized (this)
        {
            mBaseMat.put(0, 0, frame);
            this.notify();
        }
        if (mCamera != null)
            mCamera.addCallbackBuffer(mBuffer);
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
                do {
                        synchronized (OpenCvJavaCameraView.this) {
                        try {
                                        OpenCvJavaCameraView.this.wait();
                                } catch (InterruptedException e) {
                                        // TODO Auto-generated catch block
                                        e.printStackTrace();
                                }
                                }

                        if (!mStopThread) {
                        Mat frameMat = new Mat();
                    Imgproc.cvtColor(mBaseMat, frameMat, Imgproc.COLOR_YUV2RGBA_NV21, 4);
                    deliverAndDrawFrame(frameMat);
                    frameMat.release();
                        }
                } while (!mStopThread);
                Log.d(TAG, "Finish processing thread");
        }
    }
}
