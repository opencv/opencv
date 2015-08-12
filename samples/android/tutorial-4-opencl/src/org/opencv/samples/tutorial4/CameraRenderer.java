package org.opencv.samples.tutorial4;

import java.io.IOException;
import java.util.List;

import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.util.Log;

@SuppressWarnings("deprecation")
public class CameraRenderer extends MyGLRendererBase {

    protected final String LOGTAG = "CameraRenderer";
    private Camera mCamera;
    boolean mPreviewStarted = false;

    CameraRenderer(MyGLSurfaceView view) {
        super(view);
    }

    protected void closeCamera() {
        Log.i(LOGTAG, "closeCamera");
        if(mCamera != null) {
            mCamera.stopPreview();
            mPreviewStarted = false;
            mCamera.release();
            mCamera = null;
        }
    }

    protected void openCamera() {
        Log.i(LOGTAG, "openCamera");
        closeCamera();
        mCamera = Camera.open();
        try {
            mCamera.setPreviewTexture(mSTex);
        } catch (IOException ioe) {
            Log.e(LOGTAG, "setPreviewTexture() failed: " + ioe.getMessage());
        }
    }

    public void setCameraPreviewSize(int width, int height) {
        Log.i(LOGTAG, "setCameraPreviewSize: "+width+"x"+height);
        if(mCamera == null)
            return;
        if(mPreviewStarted) {
            mCamera.stopPreview();
            mPreviewStarted = false;
        }
        Camera.Parameters param = mCamera.getParameters();
        List<Size> psize = param.getSupportedPreviewSizes();
        int bestWidth = 0, bestHeight = 0;
        if (psize.size() > 0) {
            float aspect = (float)width / height;
            for (Size size : psize) {
                int w = size.width, h = size.height;
                Log.d("Renderer", "checking camera preview size: "+w+"x"+h);
                if ( w <= width && h <= height &&
                     w >= bestWidth && h >= bestHeight &&
                     Math.abs(aspect - (float)w/h) < 0.2 ) {
                    bestWidth = w;
                    bestHeight = h;
                }
            }
            if(bestWidth > 0 && bestHeight > 0) {
                param.setPreviewSize(bestWidth, bestHeight);
                Log.i(LOGTAG, "size: "+bestWidth+" x "+bestHeight);
            }
        }
        param.set("orientation", "landscape");
        mCamera.setParameters(param);
        mCamera.startPreview();
        mPreviewStarted = true;
    }
}
