package org.opencv.android;

import org.opencv.R;

import android.content.Context;
import android.content.res.TypedArray;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;

public class CameraGLSurfaceView extends GLSurfaceView {

    private static final String LOGTAG = "CameraGLSurfaceView";

    public interface CameraTextureListener {
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
         * This method is invoked when a new preview frame from Camera is ready.
         * @param texIn -  the OpenGL texture ID that contains frame in RGBA format
         * @param texOut - the OpenGL texture ID that can be used to store modified frame image t display
         * @param width -  the width of the frame
         * @param height - the height of the frame
         * @return `true` if `texOut` should be displayed, `false` - to show `texIn`
         */
        public boolean onCameraTexture(int texIn, int texOut, int width, int height);
    };

    private CameraTextureListener mTexListener;
    private CameraGLRendererBase mRenderer;

    public CameraGLSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);

        TypedArray styledAttrs = getContext().obtainStyledAttributes(attrs, R.styleable.CameraBridgeViewBase);
        int cameraIndex = styledAttrs.getInt(R.styleable.CameraBridgeViewBase_camera_id, -1);
        styledAttrs.recycle();

        if(android.os.Build.VERSION.SDK_INT >= 21)
            mRenderer = new Camera2Renderer(this);
        else
            mRenderer = new CameraRenderer(this);

        setCameraIndex(cameraIndex);

        setEGLContextClientVersion(2);
        setRenderer(mRenderer);
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
    }

    public void setCameraTextureListener(CameraTextureListener texListener)
    {
        mTexListener = texListener;
    }

    public CameraTextureListener getCameraTextureListener()
    {
        return mTexListener;
    }

    public void setCameraIndex(int cameraIndex) {
        mRenderer.setCameraIndex(cameraIndex);
    }

    public void setMaxCameraPreviewSize(int maxWidth, int maxHeight) {
        mRenderer.setMaxCameraPreviewSize(maxWidth, maxHeight);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        super.surfaceCreated(holder);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        mRenderer.mHaveSurface = false;
        super.surfaceDestroyed(holder);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        super.surfaceChanged(holder, format, w, h);
    }

    @Override
    public void onResume() {
        Log.i(LOGTAG, "onResume");
        super.onResume();
        mRenderer.onResume();
    }

    @Override
    public void onPause() {
        Log.i(LOGTAG, "onPause");
        mRenderer.onPause();
        super.onPause();
    }

    public void enableView() {
        mRenderer.enableView();
    }

    public void disableView() {
        mRenderer.disableView();
    }
}
