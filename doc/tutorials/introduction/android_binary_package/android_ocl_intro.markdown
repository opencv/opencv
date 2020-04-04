Use OpenCL in Android camera preview based CV application {#tutorial_android_ocl_intro}
=====================================

This guide was designed to help you in use of [OpenCL &trade;](https://www.khronos.org/opencl/) in Android camera preview based CV application.
It was written for [Eclipse-based ADT tools](http://developer.android.com/tools/help/adt.html)
(deprecated by Google now), but it easily can be reproduced with [Android Studio](http://developer.android.com/tools/studio/index.html).

This tutorial assumes you have the following installed and configured:

-   JDK
-   Android SDK and NDK
-   Eclipse IDE with ADT and CDT plugins

It also assumes that you are familiar with Android Java and JNI programming basics.
If you need help with anything of the above, you may refer to our @ref tutorial_android_dev_intro guide.

This tutorial also assumes you have an Android operated device with OpenCL enabled.

The related source code is located within OpenCV samples at
[opencv/samples/android/tutorial-4-opencl](https://github.com/opencv/opencv/tree/master/samples/android/tutorial-4-opencl/) directory.

Preface
-------

Using [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units)
via OpenCL for applications performance enhancements is quite a modern trend now.
Some CV algo-s (e.g. image filtering) run much faster on a GPU than on a CPU.
Recently it has become possible on Android OS.

The most popular CV application scenario for an Android operated device is starting camera in preview mode, applying some CV algo to every frame
and displaying the preview frames modified by that CV algo.

Let's consider how we can use OpenCL in this scenario. In particular let's try two ways: direct calls to OpenCL API and recently introduced OpenCV T-API
(aka [Transparent API](https://docs.google.com/presentation/d/1qoa29N_B-s297-fp0-b3rBirvpzJQp8dCtllLQ4DVCY/present)) - implicit OpenCL accelerations of some OpenCV algo-s.

Application structure
---------------------

Starting Android API level 11 (Android 3.0) [Camera API](http://developer.android.com/reference/android/hardware/Camera.html)
allows use of OpenGL texture as a target for preview frames.
Android API level 21 brings a new [Camera2 API](http://developer.android.com/reference/android/hardware/camera2/package-summary.html)
that provides much more control over the camera settings and usage modes,
it allows several targets for preview frames and OpenGL texture in particular.

Having a preview frame in an OpenGL texture is a good deal for using OpenCL because there is an
[OpenGL-OpenCL Interoperability API (cl_khr_gl_sharing)](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/cl_khr_gl_sharing.html),
allowing sharing OpenGL texture data with OpenCL functions without copying (with some restrictions of course).

Let's create a base for our application that just configures Android camera to send preview frames to OpenGL texture and displays these frames
on display without any processing.

A minimal `Activity` class for that purposes looks like following:

@code{.java}
public class Tutorial4Activity extends Activity {

    private MyGLSurfaceView mView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        mView = new MyGLSurfaceView(this);
        setContentView(mView);
    }

    @Override
    protected void onPause() {
        mView.onPause();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mView.onResume();
    }
}
@endcode

And a minimal `View` class respectively:

@code{.java}
public class MyGLSurfaceView extends GLSurfaceView {

    MyGLRendererBase mRenderer;

    public MyGLSurfaceView(Context context) {
        super(context);

        if(android.os.Build.VERSION.SDK_INT >= 21)
            mRenderer = new Camera2Renderer(this);
        else
            mRenderer = new CameraRenderer(this);

        setEGLContextClientVersion(2);
        setRenderer(mRenderer);
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
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
}
@endcode

__Note__: we use two renderer classes: one for legacy [Camera](http://developer.android.com/reference/android/hardware/Camera.html) API
and another for modern [Camera2](http://developer.android.com/reference/android/hardware/camera2/package-summary.html).

A minimal `Renderer` class can be implemented in Java (OpenGL ES 2.0 [available](http://developer.android.com/reference/android/opengl/GLES20.html) in Java),
but since we are going to modify the preview texture with OpenCL let's move OpenGL stuff to JNI.
Here is a simple Java wrapper for our JNI stuff:

@code{.java}
public class NativeGLRenderer {
    static
    {
        System.loadLibrary("opencv_java4"); // comment this when using OpenCV Manager
        System.loadLibrary("JNIrender");
    }

    public static native int initGL();
    public static native void closeGL();
    public static native void drawFrame();
    public static native void changeSize(int width, int height);
}
@endcode

Since `Camera` and `Camera2` APIs differ significantly in camera setup and control, let's create a base class for the two corresponding renderers:

@code{.java}
public abstract class MyGLRendererBase implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {
    protected final String LOGTAG = "MyGLRendererBase";

    protected SurfaceTexture mSTex;
    protected MyGLSurfaceView mView;

    protected boolean mGLInit = false;
    protected boolean mTexUpdate = false;

    MyGLRendererBase(MyGLSurfaceView view) {
        mView = view;
    }

    protected abstract void openCamera();
    protected abstract void closeCamera();
    protected abstract void setCameraPreviewSize(int width, int height);

    public void onResume() {
        Log.i(LOGTAG, "onResume");
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
@endcode

As you can see, inheritors for `Camera` and `Camera2` APIs should implement the following abstract methods:
@code{.java}
    protected abstract void openCamera();
    protected abstract void closeCamera();
    protected abstract void setCameraPreviewSize(int width, int height);
@endcode

Let's leave the details of their implementation beyond of this tutorial, please refer the
[source code](https://github.com/opencv/opencv/tree/master/samples/android/tutorial-4-opencl/) to see them.

Preview Frames modification
---------------------------

The details OpenGL ES 2.0 initialization are also quite straightforward and noisy to be quoted here,
but the important point here is that the OpeGL texture to be the target for camera preview should be of type `GL_TEXTURE_EXTERNAL_OES`
(not `GL_TEXTURE_2D`), internally it keeps picture data in _YUV_ format.
That makes unable sharing it via CL-GL interop (`cl_khr_gl_sharing`) and accessing its pixel data via C/C++ code.
To overcome this restriction we have to perform an OpenGL rendering from this texture to another regular `GL_TEXTURE_2D` one
using _FrameBuffer Object_ (aka FBO).

### C/C++ code

After that we can read (_copy_) pixel data from C/C++ via `glReadPixels()` and write them back to texture after modification via `glTexSubImage2D()`.

### Direct OpenCL calls

Also that `GL_TEXTURE_2D` texture can be shared with OpenCL without copying, but we have to create OpenCL context with special way for that:

@code{.cpp}
void initCL()
{
    EGLDisplay mEglDisplay = eglGetCurrentDisplay();
    if (mEglDisplay == EGL_NO_DISPLAY)
        LOGE("initCL: eglGetCurrentDisplay() returned 'EGL_NO_DISPLAY', error = %x", eglGetError());

    EGLContext mEglContext = eglGetCurrentContext();
    if (mEglContext == EGL_NO_CONTEXT)
        LOGE("initCL: eglGetCurrentContext() returned 'EGL_NO_CONTEXT', error = %x", eglGetError());

    cl_context_properties props[] =
    {   CL_GL_CONTEXT_KHR,   (cl_context_properties) mEglContext,
        CL_EGL_DISPLAY_KHR,  (cl_context_properties) mEglDisplay,
        CL_CONTEXT_PLATFORM, 0,
        0 };

    try
    {
        cl::Platform p = cl::Platform::getDefault();
        std::string ext = p.getInfo<CL_PLATFORM_EXTENSIONS>();
        if(ext.find("cl_khr_gl_sharing") == std::string::npos)
            LOGE("Warning: CL-GL sharing isn't supported by PLATFORM");
        props[5] = (cl_context_properties) p();

        theContext = cl::Context(CL_DEVICE_TYPE_GPU, props);
        std::vector<cl::Device> devs = theContext.getInfo<CL_CONTEXT_DEVICES>();
        LOGD("Context returned %d devices, taking the 1st one", devs.size());
        ext = devs[0].getInfo<CL_DEVICE_EXTENSIONS>();
        if(ext.find("cl_khr_gl_sharing") == std::string::npos)
            LOGE("Warning: CL-GL sharing isn't supported by DEVICE");

        theQueue = cl::CommandQueue(theContext, devs[0]);

        // ...
    }
    catch(cl::Error& e)
    {
        LOGE("cl::Error: %s (%d)", e.what(), e.err());
    }
    catch(std::exception& e)
    {
        LOGE("std::exception: %s", e.what());
    }
    catch(...)
    {
        LOGE( "OpenCL info: unknown error while initializing OpenCL stuff" );
    }
    LOGD("initCL completed");
}
@endcode

@note To build this JNI code you need __OpenCL 1.2__ headers from [Khronos web site](https://www.khronos.org/registry/cl/api/1.2/) and
the __libOpenCL.so__ downloaded from the device you'll run the application.

Then the texture can be wrapped by a `cl::ImageGL` object and processed via OpenCL calls:
@code{.cpp}
    cl::ImageGL imgIn (theContext, CL_MEM_READ_ONLY,  GL_TEXTURE_2D, 0, texIn);
    cl::ImageGL imgOut(theContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texOut);

    std::vector < cl::Memory > images;
    images.push_back(imgIn);
    images.push_back(imgOut);
    theQueue.enqueueAcquireGLObjects(&images);
    theQueue.finish();

    cl::Kernel Laplacian = ...
    Laplacian.setArg(0, imgIn);
    Laplacian.setArg(1, imgOut);
    theQueue.finish();

    theQueue.enqueueNDRangeKernel(Laplacian, cl::NullRange, cl::NDRange(w, h), cl::NullRange);
    theQueue.finish();

    theQueue.enqueueReleaseGLObjects(&images);
    theQueue.finish();
@endcode

### OpenCV T-API

But instead of writing OpenCL code by yourselves you may want to use __OpenCV T-API__ that calls OpenCL implicitly.
All that you need is to pass the created OpenCL context to OpenCV (via `cv::ocl::attachContext()`) and somehow wrap OpenGL texture with `cv::UMat`.
Unfortunately `UMat` keeps OpenCL _buffer_ internally, that can't be wrapped over either OpenGL _texture_ or OpenCL _image_ - so we have to copy image data here:
@code{.cpp}
    cl::ImageGL imgIn (theContext, CL_MEM_READ_ONLY,  GL_TEXTURE_2D, 0, tex);
    std::vector < cl::Memory > images(1, imgIn);
    theQueue.enqueueAcquireGLObjects(&images);
    theQueue.finish();

    cv::UMat uIn, uOut, uTmp;
    cv::ocl::convertFromImage(imgIn(), uIn);
    theQueue.enqueueReleaseGLObjects(&images);

    cv::Laplacian(uIn, uTmp, CV_8U);
    cv:multiply(uTmp, 10, uOut);
    cv::ocl::finish();

    cl::ImageGL imgOut(theContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, tex);
    images.clear();
    images.push_back(imgOut);
    theQueue.enqueueAcquireGLObjects(&images);
    cl_mem clBuffer = (cl_mem)uOut.handle(cv::ACCESS_READ);
    cl_command_queue q = (cl_command_queue)cv::ocl::Queue::getDefault().ptr();
    size_t offset = 0;
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { w, h, 1 };
    CV_Assert(clEnqueueCopyBufferToImage (q, clBuffer, imgOut(), offset, origin, region, 0, NULL, NULL) == CL_SUCCESS);
    theQueue.enqueueReleaseGLObjects(&images);
    cv::ocl::finish();
@endcode

- @note We have to make one more image data copy when placing back the modified image to the original OpenGL texture via OpenCL image wrapper.
- @note By default the OpenCL support (T-API) is disabled in OpenCV builds for Android OS (so it's absent in official packages as of version 3.0),
  but it's possible to rebuild locally OpenCV for Android with OpenCL/T-API enabled: use `-DWITH_OPENCL=YES` option for CMake.
  @code{.cmd}
  cd opencv-build-android
  path/to/cmake.exe -GNinja -DCMAKE_MAKE_PROGRAM="path/to/ninja.exe" -DCMAKE_TOOLCHAIN_FILE=path/to/opencv/platforms/android/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a with NEON" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON path/to/opencv
  path/to/ninja.exe install/strip
  @endcode
  To use your own modified `libopencv_java4.so` you have to keep inside your APK, not to use OpenCV Manager and load it manually via `System.loadLibrary("opencv_java4")`.

Performance notes
-----------------

To compare the performance we measured FPS of the same preview frames modification (_Laplacian_) done by C/C++ code (call to `cv::Laplacian` with `cv::Mat`),
by direct OpenCL calls (using OpenCL _images_ for input and output), and by OpenCV _T-API_ (call to `cv::Laplacian` with `cv::UMat`) on _Sony Xperia Z3_ with 720p camera resolution:
* __C/C++ version__ shows __3-4 fps__
* __direct OpenCL calls__ shows __25-27 fps__
* __OpenCV T-API__ shows __11-13 fps__ (due to extra copying from `cl_image` to `cl_buffer` and back)
