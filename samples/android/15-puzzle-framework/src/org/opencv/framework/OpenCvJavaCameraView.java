package org.opencv.framework;

import java.io.IOException;
import java.util.List;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.AttributeSet;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
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



        @Override
        protected void connectCamera(int width, int height) {
                mCamera = Camera.open(0);

                List<android.hardware.Camera.Size> sizes = mCamera.getParameters().getSupportedPreviewSizes();
        /* Select the size that fits surface considering maximum size allowed */
        FrameSize frameSize = calculateCameraFrameSize(sizes, new JavaCameraSizeAccessor(), width, height);

        /* Now set camera parameters */
        try {
		Camera.Parameters params = mCamera.getParameters();

		List<Integer> formats = params.getSupportedPictureFormats();

		params.setPreviewFormat(ImageFormat.NV21);
		params.setPreviewSize(frameSize.width, frameSize.height);

		mCamera.setPreviewCallback(this);
		mCamera.setParameters(params);
                        //mCamera.setPreviewTexture(new SurfaceTexture(MAGIC_TEXTURE_ID));

		SurfaceTexture tex = new SurfaceTexture(MAGIC_TEXTURE_ID);

		mCamera.setPreviewTexture(tex);

                        mFrameWidth = frameSize.width;
                        mFrameHeight = frameSize.height;

                } catch (IOException e) {
                        e.printStackTrace();
                }

                mBaseMat = new Mat(mFrameHeight + (mFrameHeight/2), mFrameWidth, CvType.CV_8UC1);

        /* Finally we are ready to start the preview */
                mCamera.startPreview();
        }

        @Override
        protected void disconnectCamera() {

                mCamera.setPreviewCallback(null);
                mCamera.stopPreview();
                mCamera.release();
        }



        @Override
        public void onPreviewFrame(byte[] frame, Camera arg1) {
                Log.i(TAG, "Preview Frame received. Need to create MAT and deliver it to clients");

                Log.i(TAG, "Frame size  is " + frame.length);

                mBaseMat.put(0, 0, frame);
                Mat frameMat = new Mat();
                Imgproc.cvtColor(mBaseMat, frameMat, Imgproc.COLOR_YUV2RGBA_NV21, 4);
                deliverAndDrawFrame(frameMat);
                frameMat.release();
        }

}
