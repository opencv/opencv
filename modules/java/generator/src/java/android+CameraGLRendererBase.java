package org.opencv.android;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import org.opencv.android.CameraGLSurfaceView.CameraTextureListener;

import android.annotation.TargetApi;
import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.View;

@TargetApi(15)
public abstract class CameraGLRendererBase implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

    protected final String LOGTAG = "CameraGLRendererBase";

    // shaders
    private final String vss = ""
            + "attribute vec2 vPosition;\n"
            + "attribute vec2 vTexCoord;\n" + "varying vec2 texCoord;\n"
            + "void main() {\n" + "  texCoord = vTexCoord;\n"
            + "  gl_Position = vec4 ( vPosition.x, vPosition.y, 0.0, 1.0 );\n"
            + "}";

    private final String fssOES = ""
            + "#extension GL_OES_EGL_image_external : require\n"
            + "precision mediump float;\n"
            + "uniform samplerExternalOES sTexture;\n"
            + "varying vec2 texCoord;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(sTexture,texCoord);\n" + "}";

    private final String fss2D = ""
            + "precision mediump float;\n"
            + "uniform sampler2D sTexture;\n"
            + "varying vec2 texCoord;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(sTexture,texCoord);\n" + "}";

    // coord-s
    private final float vertices[] = {
           -1, -1,
           -1,  1,
            1, -1,
            1,  1 };
    private final float texCoordOES[] = {
            0,  1,
            0,  0,
            1,  1,
            1,  0 };
    private final float texCoord2D[] = {
            0,  0,
            0,  1,
            1,  0,
            1,  1 };

    private int[] texCamera = {0}, texFBO = {0}, texDraw = {0};
    private int[] FBO = {0};
    private int progOES = -1, prog2D = -1;
    private int vPosOES, vTCOES, vPos2D, vTC2D;

    private FloatBuffer vert, texOES, tex2D;

    protected int mCameraWidth = -1, mCameraHeight = -1;
    protected int mFBOWidth = -1, mFBOHeight = -1;
    protected int mMaxCameraWidth = -1, mMaxCameraHeight = -1;
    protected int mCameraIndex = CameraBridgeViewBase.CAMERA_ID_ANY;

    protected SurfaceTexture mSTexture;

    protected boolean mHaveSurface = false;
    protected boolean mHaveFBO = false;
    protected boolean mUpdateST = false;
    protected boolean mEnabled = true;
    protected boolean mIsStarted = false;

    protected CameraGLSurfaceView mView;

    protected abstract void openCamera(int id);
    protected abstract void closeCamera();
    protected abstract void setCameraPreviewSize(int width, int height); // updates mCameraWidth & mCameraHeight

    public CameraGLRendererBase(CameraGLSurfaceView view) {
        mView = view;
        int bytes = vertices.length * Float.SIZE / Byte.SIZE;
        vert   = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        texOES = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        tex2D  = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        vert.put(vertices).position(0);
        texOES.put(texCoordOES).position(0);
        tex2D.put(texCoord2D).position(0);
    }

    @Override
    public synchronized void onFrameAvailable(SurfaceTexture surfaceTexture) {
        //Log.i(LOGTAG, "onFrameAvailable");
        mUpdateST = true;
        mView.requestRender();
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        //Log.i(LOGTAG, "onDrawFrame start");

        if (!mHaveFBO)
            return;

        synchronized(this) {
            if (mUpdateST) {
                mSTexture.updateTexImage();
                mUpdateST = false;
            }

            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

            CameraTextureListener texListener = mView.getCameraTextureListener();
            if(texListener != null) {
                //Log.d(LOGTAG, "haveUserCallback");
                // texCamera(OES) -> texFBO
                drawTex(texCamera[0], true, FBO[0]);

                // call user code (texFBO -> texDraw)
                boolean modified = texListener.onCameraTexture(texFBO[0], texDraw[0], mCameraWidth, mCameraHeight);

                if(modified) {
                    // texDraw -> screen
                    drawTex(texDraw[0], false, 0);
                } else {
                    // texFBO -> screen
                    drawTex(texFBO[0], false, 0);
                }
            } else {
                Log.d(LOGTAG, "texCamera(OES) -> screen");
                // texCamera(OES) -> screen
                drawTex(texCamera[0], true, 0);
            }
            //Log.i(LOGTAG, "onDrawFrame end");
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int surfaceWidth, int surfaceHeight) {
        Log.i(LOGTAG, "onSurfaceChanged("+surfaceWidth+"x"+surfaceHeight+")");
        mHaveSurface = true;
        updateState();
        setPreviewSize(surfaceWidth, surfaceHeight);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        Log.i(LOGTAG, "onSurfaceCreated");
        initShaders();
    }

    private void initShaders() {
        String strGLVersion = GLES20.glGetString(GLES20.GL_VERSION);
        if (strGLVersion != null)
            Log.i(LOGTAG, "OpenGL ES version: " + strGLVersion);

        GLES20.glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        progOES = loadShader(vss, fssOES);
        vPosOES = GLES20.glGetAttribLocation(progOES, "vPosition");
        vTCOES  = GLES20.glGetAttribLocation(progOES, "vTexCoord");
        GLES20.glEnableVertexAttribArray(vPosOES);
        GLES20.glEnableVertexAttribArray(vTCOES);

        prog2D  = loadShader(vss, fss2D);
        vPos2D = GLES20.glGetAttribLocation(prog2D, "vPosition");
        vTC2D  = GLES20.glGetAttribLocation(prog2D, "vTexCoord");
        GLES20.glEnableVertexAttribArray(vPos2D);
        GLES20.glEnableVertexAttribArray(vTC2D);
    }

    private void initSurfaceTexture() {
        Log.d(LOGTAG, "initSurfaceTexture");
        deleteSurfaceTexture();
        initTexOES(texCamera);
        mSTexture = new SurfaceTexture(texCamera[0]);
        mSTexture.setOnFrameAvailableListener(this);
    }

    private void deleteSurfaceTexture() {
        Log.d(LOGTAG, "deleteSurfaceTexture");
        if(mSTexture != null) {
            mSTexture.release();
            mSTexture = null;
            deleteTex(texCamera);
        }
    }

    private void initTexOES(int[] tex) {
        if(tex.length == 1) {
            GLES20.glGenTextures(1, tex, 0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex[0]);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        }
    }

    private static void deleteTex(int[] tex) {
        if(tex.length == 1) {
            GLES20.glDeleteTextures(1, tex, 0);
        }
    }

    private static int loadShader(String vss, String fss) {
        Log.d("CameraGLRendererBase", "loadShader");
        int vshader = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
        GLES20.glShaderSource(vshader, vss);
        GLES20.glCompileShader(vshader);
        int[] status = new int[1];
        GLES20.glGetShaderiv(vshader, GLES20.GL_COMPILE_STATUS, status, 0);
        if (status[0] == 0) {
            Log.e("CameraGLRendererBase", "Could not compile vertex shader: "+GLES20.glGetShaderInfoLog(vshader));
            GLES20.glDeleteShader(vshader);
            vshader = 0;
            return 0;
        }

        int fshader = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        GLES20.glShaderSource(fshader, fss);
        GLES20.glCompileShader(fshader);
        GLES20.glGetShaderiv(fshader, GLES20.GL_COMPILE_STATUS, status, 0);
        if (status[0] == 0) {
            Log.e("CameraGLRendererBase", "Could not compile fragment shader:"+GLES20.glGetShaderInfoLog(fshader));
            GLES20.glDeleteShader(vshader);
            GLES20.glDeleteShader(fshader);
            fshader = 0;
            return 0;
        }

        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vshader);
        GLES20.glAttachShader(program, fshader);
        GLES20.glLinkProgram(program);
        GLES20.glDeleteShader(vshader);
        GLES20.glDeleteShader(fshader);
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0);
        if (status[0] == 0) {
            Log.e("CameraGLRendererBase", "Could not link shader program: "+GLES20.glGetProgramInfoLog(program));
            program = 0;
            return 0;
        }
        GLES20.glValidateProgram(program);
        GLES20.glGetProgramiv(program, GLES20.GL_VALIDATE_STATUS, status, 0);
        if (status[0] == 0)
        {
            Log.e("CameraGLRendererBase", "Shader program validation error: "+GLES20.glGetProgramInfoLog(program));
            GLES20.glDeleteProgram(program);
            program = 0;
            return 0;
        }

        Log.d("CameraGLRendererBase", "Shader program is built OK");

        return program;
    }

    private void deleteFBO()
    {
        Log.d(LOGTAG, "deleteFBO("+mFBOWidth+"x"+mFBOHeight+")");
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glDeleteFramebuffers(1, FBO, 0);

        deleteTex(texFBO);
        deleteTex(texDraw);
        mFBOWidth = mFBOHeight = 0;
    }

    private void initFBO(int width, int height)
    {
        Log.d(LOGTAG, "initFBO("+width+"x"+height+")");

        deleteFBO();

        GLES20.glGenTextures(1, texDraw, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texDraw[0]);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

        GLES20.glGenTextures(1, texFBO, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texFBO[0]);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

        //int hFBO;
        GLES20.glGenFramebuffers(1, FBO, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, FBO[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, texFBO[0], 0);
        Log.d(LOGTAG, "initFBO error status: " + GLES20.glGetError());

        int FBOstatus = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (FBOstatus != GLES20.GL_FRAMEBUFFER_COMPLETE)
            Log.e(LOGTAG, "initFBO failed, status: " + FBOstatus);

        mFBOWidth  = width;
        mFBOHeight = height;
    }

    // draw texture to FBO or to screen if fbo == 0
    private void drawTex(int tex, boolean isOES, int fbo)
    {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo);

        if(fbo == 0)
            GLES20.glViewport(0, 0, mView.getWidth(), mView.getHeight());
        else
            GLES20.glViewport(0, 0, mFBOWidth, mFBOHeight);

        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        if(isOES) {
            GLES20.glUseProgram(progOES);
            GLES20.glVertexAttribPointer(vPosOES, 2, GLES20.GL_FLOAT, false, 4*2, vert);
            GLES20.glVertexAttribPointer(vTCOES,  2, GLES20.GL_FLOAT, false, 4*2, texOES);
        } else {
            GLES20.glUseProgram(prog2D);
            GLES20.glVertexAttribPointer(vPos2D, 2, GLES20.GL_FLOAT, false, 4*2, vert);
            GLES20.glVertexAttribPointer(vTC2D,  2, GLES20.GL_FLOAT, false, 4*2, tex2D);
        }

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);

        if(isOES) {
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(progOES, "sTexture"), 0);
        } else {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(prog2D, "sTexture"), 0);
        }

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glFlush();
    }

    public synchronized void enableView() {
        Log.d(LOGTAG, "enableView");
        mEnabled = true;
        updateState();
    }

    public synchronized void disableView() {
        Log.d(LOGTAG, "disableView");
        mEnabled = false;
        updateState();
    }

    protected void updateState() {
        Log.d(LOGTAG, "updateState");
        Log.d(LOGTAG, "mEnabled="+mEnabled+", mHaveSurface="+mHaveSurface);
        boolean willStart = mEnabled && mHaveSurface && mView.getVisibility() == View.VISIBLE;
        if (willStart != mIsStarted) {
            if(willStart) doStart();
            else doStop();
        } else {
            Log.d(LOGTAG, "keeping State unchanged");
        }
        Log.d(LOGTAG, "updateState end");
    }

    protected synchronized void doStart() {
        Log.d(LOGTAG, "doStart");
        initSurfaceTexture();
        openCamera(mCameraIndex);
        mIsStarted = true;
        if(mCameraWidth>0 && mCameraHeight>0)
            setPreviewSize(mCameraWidth, mCameraHeight); // start preview and call listener.onCameraViewStarted()
    }


    protected void doStop() {
        Log.d(LOGTAG, "doStop");
        synchronized(this) {
            mUpdateST = false;
            mIsStarted = false;
            mHaveFBO = false;
            closeCamera();
            deleteSurfaceTexture();
        }
        CameraTextureListener listener = mView.getCameraTextureListener();
        if(listener != null) listener.onCameraViewStopped();

    }

    protected void setPreviewSize(int width, int height) {
        synchronized(this) {
            mHaveFBO = false;
            mCameraWidth  = width;
            mCameraHeight = height;
            setCameraPreviewSize(width, height); // can change mCameraWidth & mCameraHeight
            initFBO(mCameraWidth, mCameraHeight);
            mHaveFBO = true;
        }

        CameraTextureListener listener = mView.getCameraTextureListener();
        if(listener != null) listener.onCameraViewStarted(mCameraWidth, mCameraHeight);
    }

    public void setCameraIndex(int cameraIndex) {
        disableView();
        mCameraIndex = cameraIndex;
        enableView();
    }

    public void setMaxCameraPreviewSize(int maxWidth, int maxHeight) {
        disableView();
        mMaxCameraWidth  = maxWidth;
        mMaxCameraHeight = maxHeight;
        enableView();
    }

    public void onResume() {
        Log.i(LOGTAG, "onResume");
    }

    public void onPause() {
        Log.i(LOGTAG, "onPause");
        mHaveSurface = false;
        updateState();
        mCameraWidth = mCameraHeight = -1;
    }

}
