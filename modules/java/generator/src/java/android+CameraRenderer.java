package org.opencv.android;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import org.opencv.android.CameraGLSurfaceView.CameraTextureListener;

import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.util.Log;
import android.annotation.TargetApi;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.Size;

@TargetApi(15)
public class CameraRenderer implements GLSurfaceView.Renderer,
        SurfaceTexture.OnFrameAvailableListener {

    public static final String LOGTAG = "CameraRenderer";

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
    private int progOES, prog2D;
    private int vPosOES, vTCOES, vPos2D, vTC2D;

    private FloatBuffer vert, texOES, tex2D;

    private Camera mCamera;
    private boolean mPreviewStarted = false;
    private int mPreviewWidth, mPreviewHeight;

    private SurfaceTexture mSTexture;

    private boolean mGLInit = false;
    private boolean mUpdateST = false;

    private CameraGLSurfaceView mView;

    CameraRenderer(CameraGLSurfaceView view) {
        mView = view;
        int bytes = vertices.length * Float.SIZE / Byte.SIZE;
        vert   = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        texOES = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        tex2D  = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        vert.put(vertices).position(0);
        texOES.put(texCoordOES).position(0);
        tex2D.put(texCoord2D).position(0);
    }

    public void onResume() {
        //nothing
        Log.i(LOGTAG, "onResume");
    }

    public void onPause() {
        Log.i(LOGTAG, "onPause");
        mGLInit = false;
        mUpdateST = false;

        if(mCamera != null) {
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
        }

        if(mSTexture != null) {
            mSTexture.release();
            mSTexture = null;
            deleteTex(texCamera);
        }
    }

    public void onSurfaceCreated(GL10 unused, EGLConfig config) {
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

        initTexOES(texCamera);
        mSTexture = new SurfaceTexture(texCamera[0]);
        mSTexture.setOnFrameAvailableListener(this);

        mCamera = Camera.open();
        try {
            mCamera.setPreviewTexture(mSTexture);
        } catch (IOException ioe) {
        }

        mGLInit = true;
    }

    public void onDrawFrame(GL10 unused) {
        if (!mGLInit)
            return;
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        synchronized (this) {
            if (mUpdateST) {
                mSTexture.updateTexImage();
                mUpdateST = false;
            }
        }

        CameraTextureListener texListener = mView.getCameraTextureListener();
        if(texListener != null) {
            Log.d(LOGTAG, "haveUserCallback");
            // texCamera(OES) -> texFBO
            drawTex(texCamera[0], true, FBO[0]);

            // call user code (texFBO -> texDraw)
            boolean modified = texListener.onCameraFrame(texFBO[0], texDraw[0], mPreviewWidth, mPreviewHeight);

            if(modified) {
                // texDraw -> screen
                drawTex(texDraw[0], false, 0);
            } else {
                // texFBO -> screen
                drawTex(texFBO[0], false, 0);
            }
        } else {
            // texCamera(OES) -> screen
            drawTex(texCamera[0], true, 0);
        }
    }

    public void onSurfaceChanged(GL10 unused, int width, int height) {
        Log.i(LOGTAG, "onSurfaceChanged("+width+"x"+height+")");

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
                Log.d(LOGTAG, "checking camera preview size: "+w+"x"+h);
                if ( w <= width && h <= height &&
                     w >= bestWidth && h >= bestHeight &&
                     Math.abs(aspect - (float)w/h) < 0.2 ) {
                    bestWidth = w;
                    bestHeight = h;
                }
            }
            if(bestWidth > 0 && bestHeight > 0) {
                param.setPreviewSize(bestWidth, bestHeight);
                Log.i(LOGTAG, "selected size: "+bestWidth+" x "+bestHeight);

                GLES20.glViewport(0, 0, bestWidth, bestWidth);
                initFBO(bestWidth, bestHeight);
                mPreviewWidth = bestWidth;
                mPreviewHeight = bestHeight;
            }
        }
        //param.set("orientation", "landscape");
        mCamera.setParameters(param);
        mCamera.startPreview();
        mPreviewStarted = true;
    }

    public synchronized void onFrameAvailable(SurfaceTexture st) {
        mUpdateST = true;
        mView.requestRender();
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

    private void deleteTex(int[] tex) {
        if(tex.length == 1) {
            GLES20.glDeleteTextures(1, tex, 0);
        }
    }

    private static int loadShader(String vss, String fss) {
        int vshader = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
        GLES20.glShaderSource(vshader, vss);
        GLES20.glCompileShader(vshader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(vshader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            Log.e(LOGTAG, "Could not compile vertex shader");
            Log.v(LOGTAG, "Could not compile vertex shader:"+GLES20.glGetShaderInfoLog(vshader));
            GLES20.glDeleteShader(vshader);
            vshader = 0;
        }

        int fshader = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        GLES20.glShaderSource(fshader, fss);
        GLES20.glCompileShader(fshader);
        GLES20.glGetShaderiv(fshader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            Log.e("Renderer", "Could not compile fragment shader");
            Log.v("Renderer", "Could not compile fragment shader:"+GLES20.glGetShaderInfoLog(fshader));
            GLES20.glDeleteShader(fshader);
            fshader = 0;
        }

        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vshader);
        GLES20.glAttachShader(program, fshader);
        GLES20.glLinkProgram(program);

        return program;
    }

    private void releaseFBO()
    {

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glDeleteFramebuffers(1, FBO, 0);

        deleteTex(texFBO);
        deleteTex(texDraw);
    }

    private void initFBO(int width, int height)
    {
        Log.d(LOGTAG, "initFBO("+width+"x"+height+")");
        releaseFBO();

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
        Log.d(LOGTAG, "initFBO status: " + GLES20.glGetError());

        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE)
            Log.e(LOGTAG, "initFBO failed: " + GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER));

        //GLES20.glViewport(0, 0, width, height);
    }

    // draw texture to FBO or to screen if fbo == 0
    private void drawTex(int tex, boolean isOES, int fbo)
    {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo);
        GLES20.glViewport(0, 0, mPreviewWidth, mPreviewHeight);
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
}