package org.opencv.android;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import android.annotation.TargetApi;
import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Surface;
import android.view.ViewGroup.LayoutParams;

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

    private static final String LOGTAG = "JavaCamera2View";

    protected ImageReader mImageReader;
    protected int mPreviewFormat = ImageFormat.YUV_420_888;
    protected int mRequestTemplate = CameraDevice.TEMPLATE_PREVIEW;
    private int mFrameRotation;

    protected CameraDevice mCameraDevice;
    protected CameraCaptureSession mCaptureSession;
    protected CaptureRequest.Builder mPreviewRequestBuilder;
    protected String mCameraID;
    protected android.util.Size mPreviewSize = new android.util.Size(-1, -1);

    private HandlerThread mBackgroundThread;
    protected Handler mBackgroundHandler;

    public JavaCamera2View(Context context, int cameraId) {
        super(context, cameraId);
    }

    public JavaCamera2View(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    private void startBackgroundThread() {
        Log.i(LOGTAG, "startBackgroundThread");
        stopBackgroundThread();
        mBackgroundThread = new HandlerThread("OpenCVCameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        Log.i(LOGTAG, "stopBackgroundThread");
        if (mBackgroundThread == null)
            return;
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            Log.e(LOGTAG, "stopBackgroundThread", e);
        }
    }

    protected boolean selectCamera() {
        Log.i(LOGTAG, "selectCamera");
        CameraManager manager = (CameraManager) getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            String camList[] = manager.getCameraIdList();
            if (camList.length == 0) {
                Log.e(LOGTAG, "Error: camera isn't detected.");
                return false;
            }
            if (mCameraIndex == CameraBridgeViewBase.CAMERA_ID_ANY) {
                mCameraID = camList[0];
            } else {
                for (String cameraID : camList) {
                    CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraID);
                    if ((mCameraIndex == CameraBridgeViewBase.CAMERA_ID_BACK &&
                            characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK) ||
                        (mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT &&
                            characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT)
                    ) {
                        mCameraID = cameraID;
                        break;
                    }
                }
            }
            if (mCameraID == null) { // make JavaCamera2View behaves in the same way as JavaCameraView
                Log.i(LOGTAG, "Selecting camera by index (" + mCameraIndex + ")");
                if (mCameraIndex < camList.length) {
                    mCameraID = camList[mCameraIndex];
                } else {
                    // CAMERA_DISCONNECTED is used when the camera id is no longer valid
                    throw new CameraAccessException(CameraAccessException.CAMERA_DISCONNECTED);
                }
            }
            return true;
        } catch (CameraAccessException e) {
            Log.e(LOGTAG, "selectCamera - Camera Access Exception", e);
        } catch (IllegalArgumentException e) {
            Log.e(LOGTAG, "selectCamera - Illegal Argument Exception", e);
        } catch (SecurityException e) {
            Log.e(LOGTAG, "selectCamera - Security Exception", e);
        }
        return false;
    }

    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

        @Override
        public void onOpened(CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(CameraDevice cameraDevice) {
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(CameraDevice cameraDevice, int error) {
            cameraDevice.close();
            mCameraDevice = null;
        }

    };

    protected CameraCaptureSession.StateCallback allocateSessionStateCallback() {
        return new CameraCaptureSession.StateCallback() {
            @Override
            public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                Log.i(LOGTAG, "createCaptureSession::onConfigured");
                if (null == mCameraDevice) {
                    return; // camera is already closed
                }
                mCaptureSession = cameraCaptureSession;
                try {
                    mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                    mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE,
                            CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);

                    mCaptureSession.setRepeatingRequest(mPreviewRequestBuilder.build(), null, mBackgroundHandler);
                    Log.i(LOGTAG, "CameraPreviewSession has been started");
                } catch (Exception e) {
                    Log.e(LOGTAG, "createCaptureSession failed", e);
                }
            }

            @Override
            public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
                Log.e(LOGTAG, "createCameraPreviewSession failed");
            }
        };
    }

    private void createCameraPreviewSession() {
        final int w = mPreviewSize.getWidth(), h = mPreviewSize.getHeight();
        Log.i(LOGTAG, "createCameraPreviewSession(" + w + "x" + h + ")");
        if (w < 0 || h < 0)
            return;
        try {
            if (null == mCameraDevice) {
                Log.e(LOGTAG, "createCameraPreviewSession: camera isn't opened");
                return;
            }
            if (null != mCaptureSession) {
                Log.e(LOGTAG, "createCameraPreviewSession: mCaptureSession is already started");
                return;
            }

            mImageReader = ImageReader.newInstance(w, h, mPreviewFormat, 2);
            mImageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {

                    Image image = reader.acquireLatestImage();
                    if (image == null)
                        return;

                    // sanity checks - 3 planes
                    Image.Plane[] planes = image.getPlanes();
                    assert (planes.length == 3);
                    assert (image.getFormat() == mPreviewFormat);

                    RotatedCameraFrame tempFrame = new RotatedCameraFrame(new JavaCamera2Frame(image), mFrameRotation);
                    deliverAndDrawFrame(tempFrame);
                    tempFrame.mFrame.release();
                    tempFrame.release();
                    image.close();
                }
            }, mBackgroundHandler);
            Surface surface = mImageReader.getSurface();

            mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(mRequestTemplate);
            mPreviewRequestBuilder.addTarget(surface);

            mCameraDevice.createCaptureSession(Arrays.asList(surface),
                                               allocateSessionStateCallback(), null);
        } catch (CameraAccessException e) {
            Log.e(LOGTAG, "createCameraPreviewSession", e);
        }
    }

    @Override
    protected void disconnectCamera() {
        Log.i(LOGTAG, "close camera");
        try {
            CameraDevice c = mCameraDevice;
            mCameraDevice = null;
            if (null != mCaptureSession) {
                mCaptureSession.close();
                mCaptureSession = null;
            }
            if (null != c) {
                c.close();
            }
        } finally {
            stopBackgroundThread();
            if (null != mImageReader) {
                mImageReader.close();
                mImageReader = null;
            }
        }
        Log.i(LOGTAG, "camera closed!");
    }

    public static class JavaCameraSizeAccessor implements ListItemAccessor {
        @Override
        public int getWidth(Object obj) {
            android.util.Size size = (android.util.Size)obj;
            return size.getWidth();
        }

        @Override
        public int getHeight(Object obj) {
            android.util.Size size = (android.util.Size)obj;
            return size.getHeight();
        }
    }

    boolean calcPreviewSize(final int width, final int height) {
        Log.i(LOGTAG, "calcPreviewSize: " + width + "x" + height);
        if (mCameraID == null) {
            Log.e(LOGTAG, "Camera isn't initialized!");
            return false;
        }
        CameraManager manager = (CameraManager) getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(mCameraID);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            android.util.Size[] sizes = map.getOutputSizes(ImageReader.class);
            List<android.util.Size> sizes_list = Arrays.asList(sizes);
            Size frameSize = calculateCameraFrameSize(sizes_list, new JavaCameraSizeAccessor(), width, height);
            Log.i(LOGTAG, "Selected preview size to " + Integer.valueOf((int)frameSize.width) + "x" + Integer.valueOf((int)frameSize.height));
            assert(!(frameSize.width == 0 || frameSize.height == 0));
            if (mPreviewSize.getWidth() == frameSize.width && mPreviewSize.getHeight() == frameSize.height)
                return false;
            else {
                mPreviewSize = new android.util.Size((int)frameSize.width, (int)frameSize.height);
                return true;
            }
        } catch (CameraAccessException e) {
            Log.e(LOGTAG, "calcPreviewSize - Camera Access Exception", e);
        } catch (IllegalArgumentException e) {
            Log.e(LOGTAG, "calcPreviewSize - Illegal Argument Exception", e);
        } catch (SecurityException e) {
            Log.e(LOGTAG, "calcPreviewSize - Security Exception", e);
        }
        return false;
    }

    @Override
    protected boolean connectCamera(int width, int height) {
        Log.i(LOGTAG, "setCameraPreviewSize(" + width + "x" + height + ")");
        startBackgroundThread();
        selectCamera();
        try {
            CameraManager manager = (CameraManager) getContext().getSystemService(Context.CAMERA_SERVICE);
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(mCameraID);
            mFrameRotation = getFrameRotation(
                    characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT,
                    characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION));

            boolean needReconfig = calcPreviewSize(width, height);
            if (mFrameRotation % 180 == 0) {
                mFrameWidth = mPreviewSize.getWidth();
                mFrameHeight = mPreviewSize.getHeight();
            } else {
                mFrameWidth = mPreviewSize.getHeight();
                mFrameHeight = mPreviewSize.getWidth();
            }

            if ((getLayoutParams().width == LayoutParams.MATCH_PARENT) && (getLayoutParams().height == LayoutParams.MATCH_PARENT))
                mScale = Math.min(((float)height)/mFrameHeight, ((float)width)/mFrameWidth);
            else
                mScale = 0;

            AllocateCache();

            if (needReconfig) {
                if (null != mCaptureSession) {
                    Log.d(LOGTAG, "closing existing previewSession");
                    mCaptureSession.close();
                    mCaptureSession = null;
                }
            }

            if (mFpsMeter != null) {
                mFpsMeter.setResolution(mFrameWidth, mFrameHeight);
            }

            Log.i(LOGTAG, "Opening camera: " + mCameraID);
            manager.openCamera(mCameraID, mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(LOGTAG, "OpenCamera - Camera Access Exception", e);
        } catch (RuntimeException e) {
            throw new RuntimeException("Interrupted while setCameraPreviewSize.", e);
        }
        return true;
    }

    private class JavaCamera2Frame implements CvCameraViewFrame {
        @Override
        public Mat gray() {
            Image.Plane[] planes = mImage.getPlanes();
            int w = mImage.getWidth();
            int h = mImage.getHeight();
            assert(planes[0].getPixelStride() == 1);
            ByteBuffer y_plane = planes[0].getBuffer();
            int y_plane_step = planes[0].getRowStride();
            mGray = new Mat(h, w, CvType.CV_8UC1, y_plane, y_plane_step);
            return mGray;
        }

        @Override
        public Mat rgba() {
            Image.Plane[] planes = mImage.getPlanes();
            int w = mImage.getWidth();
            int h = mImage.getHeight();
            int chromaPixelStride = planes[1].getPixelStride();


            if (chromaPixelStride == 2) { // Chroma channels are interleaved
                assert(planes[0].getPixelStride() == 1);
                assert(planes[2].getPixelStride() == 2);
                ByteBuffer y_plane = planes[0].getBuffer();
                int y_plane_step = planes[0].getRowStride();
                ByteBuffer uv_plane1 = planes[1].getBuffer();
                int uv_plane1_step = planes[1].getRowStride();
                ByteBuffer uv_plane2 = planes[2].getBuffer();
                int uv_plane2_step = planes[2].getRowStride();
                Mat y_mat = new Mat(h, w, CvType.CV_8UC1, y_plane, y_plane_step);
                Mat uv_mat1 = new Mat(h / 2, w / 2, CvType.CV_8UC2, uv_plane1, uv_plane1_step);
                Mat uv_mat2 = new Mat(h / 2, w / 2, CvType.CV_8UC2, uv_plane2, uv_plane2_step);
                long addr_diff = uv_mat2.dataAddr() - uv_mat1.dataAddr();
                if (addr_diff > 0) {
                    assert(addr_diff == 1);
                    Imgproc.cvtColorTwoPlane(y_mat, uv_mat1, mRgba, Imgproc.COLOR_YUV2RGBA_NV12);
                } else {
                    assert(addr_diff == -1);
                    Imgproc.cvtColorTwoPlane(y_mat, uv_mat2, mRgba, Imgproc.COLOR_YUV2RGBA_NV21);
                }
                return mRgba;
            } else { // Chroma channels are not interleaved
                byte[] yuv_bytes = new byte[w*(h+h/2)];
                ByteBuffer y_plane = planes[0].getBuffer();
                ByteBuffer u_plane = planes[1].getBuffer();
                ByteBuffer v_plane = planes[2].getBuffer();

                int yuv_bytes_offset = 0;

                int y_plane_step = planes[0].getRowStride();
                if (y_plane_step == w) {
                    y_plane.get(yuv_bytes, 0, w*h);
                    yuv_bytes_offset = w*h;
                } else {
                    int padding = y_plane_step - w;
                    for (int i = 0; i < h; i++){
                        y_plane.get(yuv_bytes, yuv_bytes_offset, w);
                        yuv_bytes_offset += w;
                        if (i < h - 1) {
                            y_plane.position(y_plane.position() + padding);
                        }
                    }
                    assert(yuv_bytes_offset == w * h);
                }

                int chromaRowStride = planes[1].getRowStride();
                int chromaRowPadding = chromaRowStride - w/2;

                if (chromaRowPadding == 0){
                    // When the row stride of the chroma channels equals their width, we can copy
                    // the entire channels in one go
                    u_plane.get(yuv_bytes, yuv_bytes_offset, w*h/4);
                    yuv_bytes_offset += w*h/4;
                    v_plane.get(yuv_bytes, yuv_bytes_offset, w*h/4);
                } else {
                    // When not equal, we need to copy the channels row by row
                    for (int i = 0; i < h/2; i++){
                        u_plane.get(yuv_bytes, yuv_bytes_offset, w/2);
                        yuv_bytes_offset += w/2;
                        if (i < h/2-1){
                            u_plane.position(u_plane.position() + chromaRowPadding);
                        }
                    }
                    for (int i = 0; i < h/2; i++){
                        v_plane.get(yuv_bytes, yuv_bytes_offset, w/2);
                        yuv_bytes_offset += w/2;
                        if (i < h/2-1){
                            v_plane.position(v_plane.position() + chromaRowPadding);
                        }
                    }
                }

                Mat yuv_mat = new Mat(h+h/2, w, CvType.CV_8UC1);
                yuv_mat.put(0, 0, yuv_bytes);
                Imgproc.cvtColor(yuv_mat, mRgba, Imgproc.COLOR_YUV2RGBA_I420, 4);
                return mRgba;
            }
        }


        public JavaCamera2Frame(Image image) {
            super();
            mImage = image;
            mRgba = new Mat();
            mGray = new Mat();
        }

        @Override
        public void release() {
            mRgba.release();
            mGray.release();
        }

        private Image mImage;
        private Mat mRgba;
        private Mat mGray;
    };
}
