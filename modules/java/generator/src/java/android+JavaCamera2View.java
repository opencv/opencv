package org.opencv.android;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import android.annotation.TargetApi;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Surface;
import android.view.ViewGroup.LayoutParams;

import org.opencv.BuildConfig;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * This class is an implementation of the Bridge View between OpenCV and Java Camera.
 * This class relays on the functionality available in base class and only implements
 * required functions:
 * connectCamera - opens Java camera and sets the PreviewCallback to be delivered.
 * disconnectCamera - closes the camera and stops preview.
 * When frame is delivered via callback from Camera - it processed via OpenCV to be
 * converted to RGBA32 and then passed to the external callback for modifications if required.
 */

@TargetApi(21)
public class JavaCamera2View extends CameraBridgeViewBase {

    private static final int MAGIC_TEXTURE_ID = 10;
    private static final String LOGTAG = "JavaCamera2View";

    private ImageReader mImageReader;
    private int mPreviewFormat = ImageFormat.NV21;

    private CameraDevice mCameraDevice;
    private CameraCaptureSession mCaptureSession;
    private CaptureRequest.Builder mPreviewRequestBuilder;
    private String mCameraID;
    private android.util.Size mPreviewSize = new android.util.Size(-1, -1);

    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
    private Semaphore mCameraOpenCloseLock = new Semaphore(1);

    public JavaCamera2View(Context context, int cameraId) {
        super(context, cameraId);
    }

    public JavaCamera2View(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    private void startBackgroundThread() {
        Log.i(LOGTAG, "startBackgroundThread");
        stopBackgroundThread();
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        Log.i(LOGTAG, "stopBackgroundThread");
        if(mBackgroundThread == null)
            return;
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            Log.e(LOGTAG, "stopBackgroundThread");
        }
    }

    protected boolean initializeCamera() {
        Log.i(LOGTAG, "initializeCamera");
        CameraManager manager = (CameraManager) getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            String camList[] = manager.getCameraIdList();
            if(camList.length == 0) {
                Log.e(LOGTAG, "Error: camera isn't detected.");
                return false;
            }
            if(mCameraIndex == CameraBridgeViewBase.CAMERA_ID_ANY) {
                mCameraID = camList[0];
            } else {
                for (String cameraID : camList) {
                    CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraID);
                    if(mCameraIndex == CameraBridgeViewBase.CAMERA_ID_BACK &&
                            characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK ||
                            mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT &&
                                    characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) {
                        mCameraID = cameraID;
                        break;
                    }
                }
            }
            if(mCameraID != null) {
                if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                    throw new RuntimeException(
                            "Time out waiting to lock camera opening.");
                }
                Log.i(LOGTAG, "Opening camera: " + mCameraID);
                manager.openCamera(mCameraID, mStateCallback, mBackgroundHandler);
            }
        } catch (CameraAccessException e) {
            Log.e(LOGTAG, "OpenCamera - Camera Access Exception");
        } catch (IllegalArgumentException e) {
            Log.e(LOGTAG, "OpenCamera - Illegal Argument Exception");
        } catch (SecurityException e) {
            Log.e(LOGTAG, "OpenCamera - Security Exception");
        } catch (InterruptedException e) {
            Log.e(LOGTAG, "OpenCamera - Interrupted Exception");
        }
        return true;
    }

    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

        @Override
        public void onOpened(CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            mCameraOpenCloseLock.release();
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(CameraDevice cameraDevice) {
            cameraDevice.close();
            mCameraDevice = null;
            mCameraOpenCloseLock.release();
        }

        @Override
        public void onError(CameraDevice cameraDevice, int error) {
            cameraDevice.close();
            mCameraDevice = null;
            mCameraOpenCloseLock.release();
        }

    };

    private void createCameraPreviewSession() {
        final int w=mPreviewSize.getWidth(), h=mPreviewSize.getHeight();
        Log.i(LOGTAG, "createCameraPreviewSession("+w+"x"+h+")");
        if(w<0 || h<0)
            return;
        try {
            mCameraOpenCloseLock.acquire();
            if (null == mCameraDevice) {
                mCameraOpenCloseLock.release();
                Log.e(LOGTAG, "createCameraPreviewSession: camera isn't opened");
                return;
            }
            if (null != mCaptureSession) {
                mCameraOpenCloseLock.release();
                Log.e(LOGTAG, "createCameraPreviewSession: mCaptureSession is already started");
                return;
            }

            mImageReader = ImageReader.newInstance(w,h,ImageFormat.YUV_420_888,2);
            mImageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    Image image = reader.acquireLatestImage();
                    if (image == null) return;
                    ByteBuffer plane = image.getPlanes()[0].getBuffer();
                    Mat yuv = new Mat( h * 3/2, w, CvType.CV_8UC1, plane );
                    JavaCamera2Frame tempFrame = new JavaCamera2Frame(yuv,w,h);
                    deliverAndDrawFrame(tempFrame);
                    image.close();
                }
            }, mBackgroundHandler);
            Surface surface = mImageReader.getSurface();

            mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mPreviewRequestBuilder.addTarget(surface);

            mCameraDevice.createCaptureSession(Arrays.asList(surface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured( CameraCaptureSession cameraCaptureSession) {
                            mCaptureSession = cameraCaptureSession;
                            try {
                                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);

                                mCaptureSession.setRepeatingRequest(mPreviewRequestBuilder.build(), null, mBackgroundHandler);
                                Log.i(LOGTAG, "CameraPreviewSession has been started");
                            } catch (CameraAccessException e) {
                                Log.e(LOGTAG, "createCaptureSession failed");
                            }
                            mCameraOpenCloseLock.release();
                        }

                        @Override
                        public void onConfigureFailed(
                                CameraCaptureSession cameraCaptureSession) {
                            Log.e(LOGTAG, "createCameraPreviewSession failed");
                            mCameraOpenCloseLock.release();
                        }
                    }, mBackgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(LOGTAG, "createCameraPreviewSession");
        } catch (InterruptedException e) {
            throw new RuntimeException(
                    "Interrupted while createCameraPreviewSession", e);
        }
        finally {
            //mCameraOpenCloseLock.release();
        }
    }

    @Override
    protected void disconnectCamera() {
        Log.i(LOGTAG, "closeCamera");
        try {
            mCameraOpenCloseLock.acquire();
            if (null != mCaptureSession) {
                mCaptureSession.close();
                mCaptureSession = null;
            }
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            mCameraOpenCloseLock.release();
            stopBackgroundThread();
        }
    }

    boolean cacPreviewSize(final int width, final int height) {
        Log.i(LOGTAG, "cacPreviewSize: "+width+"x"+height);
        if(mCameraID == null) {
            Log.e(LOGTAG, "Camera isn't initialized!");
            return false;
        }
        CameraManager manager = (CameraManager) getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(mCameraID);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            int bestWidth = 0, bestHeight = 0;
            float aspect = (float)width / height;
            for (android.util.Size psize : map.getOutputSizes(ImageReader.class)) {
                int w = psize.getWidth(), h = psize.getHeight();
                Log.d(LOGTAG, "trying size: "+w+"x"+h);
                if ( width >= w && height >= h &&
                        bestWidth <= w && bestHeight <= h &&
                        Math.abs(aspect - (float)w/h) < 0.2 ) {
                    bestWidth = w;
                    bestHeight = h;
                }
            }
            Log.i(LOGTAG, "best size: "+bestWidth+"x"+bestHeight);
            if( bestWidth == 0 || bestHeight == 0 ||
                    mPreviewSize.getWidth() == bestWidth &&
                            mPreviewSize.getHeight() == bestHeight )
                return false;
            else {
                mPreviewSize = new android.util.Size(bestWidth, bestHeight);
                return true;
            }
        } catch (CameraAccessException e) {
            Log.e(LOGTAG, "cacPreviewSize - Camera Access Exception");
        } catch (IllegalArgumentException e) {
            Log.e(LOGTAG, "cacPreviewSize - Illegal Argument Exception");
        } catch (SecurityException e) {
            Log.e(LOGTAG, "cacPreviewSize - Security Exception");
        }
        return false;
    }

    @Override
    protected boolean connectCamera(int width, int height) {
        Log.i(LOGTAG, "setCameraPreviewSize("+width+"x"+height+")");
        startBackgroundThread();
        initializeCamera();
        try {
            mCameraOpenCloseLock.acquire();

            boolean needReconfig = cacPreviewSize(width, height);
            mFrameWidth  = mPreviewSize.getWidth();
            mFrameHeight = mPreviewSize.getHeight();

            AllocateCache();

            if( !needReconfig ) {
                mCameraOpenCloseLock.release();
                return false;
            }
            if (null != mCaptureSession) {
                Log.d(LOGTAG, "closing existing previewSession");
                mCaptureSession.close();
                mCaptureSession = null;
            }
            mCameraOpenCloseLock.release();
            createCameraPreviewSession();
        } catch (InterruptedException e) {
            mCameraOpenCloseLock.release();
            throw new RuntimeException("Interrupted while setCameraPreviewSize.", e);
        }
        return true;
    }

    private class JavaCamera2Frame implements CvCameraViewFrame {
        @Override
        public Mat gray() {
            return mYuvFrameData.submat(0, mHeight, 0, mWidth);
        }

        @Override
        public Mat rgba() {
            if (mPreviewFormat == ImageFormat.NV21)
                Imgproc.cvtColor(mYuvFrameData, mRgba, Imgproc.COLOR_YUV2RGBA_NV21, 4);
            else if (mPreviewFormat == ImageFormat.YV12)
                Imgproc.cvtColor(mYuvFrameData, mRgba, Imgproc.COLOR_YUV2RGB_I420, 4);  // COLOR_YUV2RGBA_YV12 produces inverted colors
            else if (mPreviewFormat == ImageFormat.YUV_420_888)
                Imgproc.cvtColor(mYuvFrameData, mRgba, Imgproc.COLOR_YUV2RGB_YV12, 4);
            else
                throw new IllegalArgumentException("Preview Format can be NV21 or YV12");

            return mRgba;
        }

        public JavaCamera2Frame(Mat Yuv420sp, int width, int height) {
            super();
            mWidth = width;
            mHeight = height;
            mYuvFrameData = Yuv420sp;
            mRgba = new Mat();
        }

        public void release() {
            mRgba.release();
        }

        private Mat mYuvFrameData;
        private Mat mRgba;
        private int mWidth;
        private int mHeight;
    };
}
