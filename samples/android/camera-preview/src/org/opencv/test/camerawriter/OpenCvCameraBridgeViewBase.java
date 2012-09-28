package org.opencv.test.camerawriter;

import java.util.List;

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
import android.view.SurfaceHolder;
import android.view.SurfaceView;

/**
 * This is a basic class, implementing the interaction with Camera and OpenCV library.
 * The main responsibility of it - is to control when camera can be enabled, process the frame,
 * call external listener to make any adjustments to the frame and then draw the resulting
 * frame to the screen.
 * The clients shall implement CvCameraViewListener
 */
public abstract class OpenCvCameraBridgeViewBase extends SurfaceView implements SurfaceHolder.Callback {

        private static final int MAX_UNSPECIFIED = -1;

        protected int mFrameWidth;
        protected int mFrameHeight;

        protected int mMaxHeight;
        protected int mMaxWidth;

        private Bitmap mCacheBitmap;


        public OpenCvCameraBridgeViewBase(Context context, AttributeSet attrs) {
                super(context,attrs);
                getHolder().addCallback(this);
                mMaxWidth = MAX_UNSPECIFIED;
                mMaxHeight = MAX_UNSPECIFIED;

        }

        public interface CvCameraViewListener {
                /**
                 * This method is invoked when camera preview has started. After this method is invoked
                 * the frames will start to be delivered to client via the onCameraFrame() callback.
                 * @param width -  the width of the frames that will be delivered
                 * @param height - the height of the frames that will be delivered
                 */
                public void onCameraViewStarted(int width, int height);

                /**
                 * This method is invoked when camera preview has been stopped for some reason.
                 * No frames will be delivered via onCameraFrame() callback after this method is called.
                 */
                public void onCameraViewStopped();

                /**
                 * This method is invoked when delivery of the frame needs to be done.
                 * The returned values - is a modified frame which needs to be displayed on the screen.
                 */
                public Mat onCameraFrame(Mat inputFrame);

        }


        private static final int STOPPED = 0;
        private static final int STARTED = 1;
        private  static final String TAG = "SampleCvBase";

        private CvCameraViewListener mListener;
        private int mState = STOPPED;

        private boolean mEnabled;
        private boolean mSurfaceExist;



        private Object mSyncObject = new Object();

        public void surfaceChanged(SurfaceHolder arg0, int arg1, int arg2, int arg3) {
                synchronized(mSyncObject) {
                        if (!mSurfaceExist) {
                                mSurfaceExist = true;
                                checkCurrentState();
                        } else {
                                /** Surface changed. We need to stop camera and restart with new parameters */
                                /* Pretend that old surface has been destroyed */
                                mSurfaceExist = false;
                                checkCurrentState();
                                /* Now use new surface. Say we have it now */
                                mSurfaceExist = true;
                                checkCurrentState();
                        }
                }
        }

        public void surfaceCreated(SurfaceHolder holder) {
                /* Do nothing. Wait until surfaceChanged delivered */
        }

        public void surfaceDestroyed(SurfaceHolder holder) {
                synchronized(mSyncObject) {
                        mSurfaceExist = false;
                        checkCurrentState();
                }
        }


        /**
         * This method is provided for clients, so they can enable the camera connection.
         * The actuall onCameraViewStarted callback will be delivered only after both this method is called and surface is available
         */
        public void enableView() {
                synchronized(mSyncObject) {
                        mEnabled = true;
                        checkCurrentState();
                }
        }

        /**
         * This method is provided for clients, so they can disable camera connection and stop
         * the delivery of frames eventhough the surfaceview itself is not destroyed and still stays on the scren
         */
        public void disableView() {
                synchronized(mSyncObject) {
                        mEnabled = false;
                        checkCurrentState();
                }
        }


        public void setCvCameraViewListener(CvCameraViewListener listener) {
                mListener = listener;
        }

        /**
         * This method sets the maximum size that camera frame is allowed to be. When selecting
         * size - the biggest size which less or equal the size set will be selected.
         * As an example - we set setMaxFrameSize(200,200) and we have 176x152 and 320x240 sizes. The
         * preview frame will be selected with 176x152 size.
         * This method is usefull when need to restrict the size of preview frame for some reason (for example for video recording)
         * @param maxWidth - the maximum width allowed for camera frame.
         * @param maxHeight - the maxumum height allowed for camera frame
         */
        public void setMaxFrameSize(int maxWidth, int maxHeight) {
                mMaxWidth = maxWidth;
                mMaxHeight = maxHeight;
        }

        /**
         * Called when mSyncObject lock is held
         */
        private void checkCurrentState() {
                int targetState;

                if (mEnabled && mSurfaceExist) {
                        targetState = STARTED;
                } else {
                        targetState = STOPPED;
                }

                if (targetState != mState) {
                        /* The state change detected. Need to exit the current state and enter target state */
                        processExitState(mState);
                        mState = targetState;
                        processEnterState(mState);
                }
        }

        private void processEnterState(int state) {
                switch(state) {
                case STARTED:
                        onEnterStartedState();
                        if (mListener != null) {
                                mListener.onCameraViewStarted(mFrameWidth, mFrameHeight);
                        }
                        break;
                case STOPPED:
                        onEnterStoppedState();
                        if (mListener != null) {
                                mListener.onCameraViewStopped();
                        }
                        break;
                };
        }


        private void processExitState(int state) {
                switch(state) {
                case STARTED:
                        onExitStartedState();
                        break;
                case STOPPED:
                        onExitStoppedState();
                        break;
                };
        }

        private void onEnterStoppedState() {
                /* nothing to do */
        }

        private void onExitStoppedState() {
                /* nothing to do */
        }

        private void onEnterStartedState() {

                connectCamera(getWidth(), getHeight());
                /* Now create cahe Bitmap */
                mCacheBitmap = Bitmap.createBitmap(mFrameWidth, mFrameHeight, Bitmap.Config.ARGB_8888);

        }

        private void onExitStartedState() {

                disconnectCamera();
                if (mCacheBitmap != null) {
                        mCacheBitmap.recycle();
                }
        }


        /**
         * This method shall be called by the subclasses when they have valid
         * object and want it to be delivered to external client (via callback) and
         * then displayed on the screen.
         * @param frame - the current frame to be delivered
         */
        protected void deliverAndDrawFrame(Mat frame) {
                Mat modified;

                synchronized(mSyncObject) {
                if (mListener != null) {
			modified = mListener.onCameraFrame(frame);
                } else {
			modified = frame;
                }

                if (modified != null) {
                    Utils.matToBitmap(modified, mCacheBitmap);
                }

                if (mCacheBitmap != null) {
                    Canvas canvas = getHolder().lockCanvas();
                    if (canvas != null) {
                        canvas.drawBitmap(mCacheBitmap, (canvas.getWidth() - mCacheBitmap.getWidth()) / 2, (canvas.getHeight() - mCacheBitmap.getHeight()) / 2, null);
                        getHolder().unlockCanvasAndPost(canvas);
                    }
                }
                }
        }

        /**
         * This method is invoked shall perform concrete operation to initialize the camera.
         * CONTRACT: as a result of this method variables mFrameWidth and mFrameHeight MUST be
         * initialized with the size of the Camera frames that will be delivered to external processor.
         * @param width - the width of this SurfaceView
         * @param height - the height of this SurfaceView
         */
        protected abstract void connectCamera(int width, int height);

        /**
         * Disconnects and release the particular camera object beeing connected to this surface view.
         * Called when syncObject lock is held
         */
        protected abstract void disconnectCamera();


        /**
         * This helper method can be called by subclasses to select camera preview size.
         * It goes over the list of the supported preview sizes and selects the maximum one which
         * fits both values set via setMaxFrameSize() and surface frame allocated for this view
         * @param supportedSizes
         * @param surfaceWidth
         * @param surfaceHeight
         * @return
         */
        protected Size calculateCameraFrameSize(List<Size> supportedSizes, int surfaceWidth, int surfaceHeight) {
                int calcWidth = 0;
                int calcHeight = 0;

                int maxAllowedWidth = (mMaxWidth != MAX_UNSPECIFIED && mMaxWidth < surfaceWidth)? mMaxWidth : surfaceWidth;
                int maxAllowedHeight = (mMaxHeight != MAX_UNSPECIFIED && mMaxHeight < surfaceHeight)? mMaxHeight : surfaceHeight;

        for (Size size : supportedSizes) {
            if (size.width <= maxAllowedWidth && size.height <= maxAllowedHeight) {
		if (size.width >= calcWidth && size.height >= calcHeight) {
                        calcWidth = (int) size.width;
                        calcHeight = (int) size.height;
		}
            }
        }
                return new Size(calcWidth, calcHeight);
        }
}
