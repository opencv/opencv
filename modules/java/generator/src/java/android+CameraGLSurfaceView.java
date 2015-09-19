package org.opencv.android;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.core.Mat;

import android.app.Activity;
import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.widget.TextView;

public class CameraGLSurfaceView extends GLSurfaceView {

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
         * This method is invoked when delivery of the frame needs to be done.
         * The returned values - is a modified frame which needs to be displayed on the screen.
         * TODO: pass the parameters specifying the format of the frame (BPP, YUV or RGB and etc)
         */
        public boolean onCameraFrame(int texIn, int texOut, int width, int height);
    };

    private CameraTextureListener mTexListener;
    private CameraRenderer mRenderer;

    public CameraGLSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);

        /*if(android.os.Build.VERSION.SDK_INT >= 21)
            mRenderer = new Camera2Renderer(this);
        else*/
            mRenderer = new CameraRenderer(this);

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

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        super.surfaceCreated(holder);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        super.surfaceDestroyed(holder);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        super.surfaceChanged(holder, format, w, h);
    }

    @Override
    public void onResume() {
        super.onResume();
        mRenderer.onResume();
    }

    @Override
    public void onPause() {
        mRenderer.onPause();
        super.onPause();
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if(e.getAction() == MotionEvent.ACTION_DOWN)
            ((Activity)getContext()).openOptionsMenu();
        return true;
    }
}
