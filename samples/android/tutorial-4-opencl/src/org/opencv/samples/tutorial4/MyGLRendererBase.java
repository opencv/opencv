package org.opencv.samples.tutorial4;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

public abstract class MyGLRendererBase implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {
    protected final String LOGTAG = "MyGLRendererBase";
    protected int  frameCounter;
    protected long lastNanoTime;

    protected SurfaceTexture mSTex;
    protected MyGLSurfaceView mView;
    protected TextView mFpsText;

    protected boolean mGLInit = false;
    protected boolean mTexUpdate = false;

   MyGLRendererBase(MyGLSurfaceView view) {
        mView = view;
    }

    protected abstract void openCamera();
    protected abstract void closeCamera();
    protected abstract void setCameraPreviewSize(int width, int height);

    public void setFpsTextView(TextView fpsTV)
    {
        mFpsText = fpsTV;
    }

    public void onResume() {
        Log.i(LOGTAG, "onResume");
        frameCounter = 0;
        lastNanoTime = System.nanoTime();
    }

    public void onPause() {
        Log.i(LOGTAG, "onPause");
        mGLInit = false;
        mTexUpdate = false;
        closeCamera();
        if(mSTex != null) {
            mSTex.release();
            mSTex = null;
            NativeGLRenderer.closeGL();
        }
    }

    @Override
    public synchronized void onFrameAvailable(SurfaceTexture surfaceTexture) {
        //Log.i(LOGTAG, "onFrameAvailable");
        mTexUpdate = true;
        mView.requestRender();
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        //Log.i(LOGTAG, "onDrawFrame");
        if (!mGLInit)
            return;

        synchronized (this) {
            if (mTexUpdate) {
                mSTex.updateTexImage();
                mTexUpdate = false;
            }
        }
        NativeGLRenderer.drawFrame();

        // log FPS
        frameCounter++;
        if(frameCounter >= 10)
        {
            final int fps = (int) (frameCounter * 1e9 / (System.nanoTime() - lastNanoTime));
            Log.i(LOGTAG, "drawFrame() FPS: "+fps);
            if(mFpsText != null) {
                Runnable fpsUpdater = new Runnable() {
                    public void run() {
                        mFpsText.setText("FPS: " + fps);
                    }
                };
                new Handler(Looper.getMainLooper()).post(fpsUpdater);
            }
            frameCounter = 0;
            lastNanoTime = System.nanoTime();
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int surfaceWidth, int surfaceHeight) {
        Log.i(LOGTAG, "onSurfaceChanged("+surfaceWidth+"x"+surfaceHeight+")");
        NativeGLRenderer.changeSize(surfaceWidth, surfaceHeight);
        setCameraPreviewSize(surfaceWidth, surfaceHeight);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        Log.i(LOGTAG, "onSurfaceCreated");
        String strGLVersion = GLES20.glGetString(GLES20.GL_VERSION);
        if (strGLVersion != null)
            Log.i(LOGTAG, "OpenGL ES version: " + strGLVersion);

        int hTex = NativeGLRenderer.initGL();
        mSTex = new SurfaceTexture(hTex);
        mSTex.setOnFrameAvailableListener(this);
        openCamera();
        mGLInit = true;
    }
}
