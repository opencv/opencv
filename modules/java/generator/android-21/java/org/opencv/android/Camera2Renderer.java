package org.opencv.android;

import java.util.Arrays;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import android.annotation.TargetApi;
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

@TargetApi(21)
public class Camera2Renderer extends CameraGLRendererBase {

    protected final String LOGTAG = "Camera2Renderer";
    private CameraDevice mCameraDevice;
    private CameraCaptureSession mCaptureSession;
    private CaptureRequest.Builder mPreviewRequestBuilder;
    private String mCameraID;
    private Size mPreviewSize = new Size(-1, -1);

    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
    private Semaphore mCameraOpenCloseLock = new Semaphore(1);

    Camera2Renderer(CameraGLSurfaceView view) {
        super(view);
    }

    @Override
    protected void doStart() {
        Log.d(LOGTAG, "doStart");
        startBackgroundThread();
        super.doStart();
    }


    @Override
    protected void doStop() {
        Log.d(LOGTAG, "doStop");
        super.doStop();
        stopBackgroundThread();
    }

    boolean cacPreviewSize(final int width, final int height) {
        Log.i(LOGTAG, "cacPreviewSize: "+width+"x"+height);
        if(mCameraID == null) {
            Log.e(LOGTAG, "Camera isn't initialized!");
            return false;
        }
        CameraManager manager = (CameraManager) mView.getContext()
                .getSystemService(Context.CAMERA_SERVICE);
        try {
            CameraCharacteristics characteristics = manager
                    .getCameraCharacteristics(mCameraID);
            StreamConfigurationMap map = characteristics
                    .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            int bestWidth = 0, bestHeight = 0;
            float aspect = (float)width / height;
            for (Size psize : map.getOutputSizes(SurfaceTexture.class)) {
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
                mPreviewSize = new Size(bestWidth, bestHeight);
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
    protected void openCamera(int id) {
        Log.i(LOGTAG, "openCamera");
        CameraManager manager = (CameraManager) mView.getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            String camList[] = manager.getCameraIdList();
            if(camList.length == 0) {
                Log.e(LOGTAG, "Error: camera isn't detected.");
                return;
            }
            if(id == CameraBridgeViewBase.CAMERA_ID_ANY) {
                mCameraID = camList[0];
            } else {
                for (String cameraID : camList) {
                    CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraID);
                    if( id == CameraBridgeViewBase.CAMERA_ID_BACK &&
                        characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK ||
                        id == CameraBridgeViewBase.CAMERA_ID_FRONT &&
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
    }

    @Override
    protected void closeCamera() {
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
        }
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
        int w=mPreviewSize.getWidth(), h=mPreviewSize.getHeight();
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
            if(null == mSTexture) {
                mCameraOpenCloseLock.release();
                Log.e(LOGTAG, "createCameraPreviewSession: preview SurfaceTexture is null");
                return;
            }
            mSTexture.setDefaultBufferSize(w, h);

            Surface surface = new Surface(mSTexture);

            mPreviewRequestBuilder = mCameraDevice
                    .createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
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

    @Override
    protected void setCameraPreviewSize(int width, int height) {
        Log.i(LOGTAG, "setCameraPreviewSize("+width+"x"+height+")");
        if(mMaxCameraWidth  > 0 && mMaxCameraWidth  < width)  width  = mMaxCameraWidth;
        if(mMaxCameraHeight > 0 && mMaxCameraHeight < height) height = mMaxCameraHeight;
        try {
            mCameraOpenCloseLock.acquire();

            boolean needReconfig = cacPreviewSize(width, height);
            mCameraWidth  = mPreviewSize.getWidth();
            mCameraHeight = mPreviewSize.getHeight();

            if( !needReconfig ) {
                mCameraOpenCloseLock.release();
                return;
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
    }
}
