
.. _dev_with_OCV_on_Android:

Android Development with OpenCV
*******************************

This tutorial has been created to help you use OpenCV library within your Android project.

This guide was written with Windows 7 in mind, though it should work with any other OS supported by
OpenCV4Android SDK.

This tutorial assumes you have the following installed and configured:

* JDK

* Android SDK and NDK

* Eclipse IDE

* ADT and CDT plugins for Eclipse

     ..

If you need help with anything of the above, you may refer to our :ref:`android_dev_intro` guide.

This tutorial also assumes you have OpenCV4Android SDK already installed on your development
machine and OpenCV Manager on your testing device correspondingly. If you need help with any of
these, you may consult our :ref:`O4A_SDK` tutorial.

If you encounter any error after thoroughly following these steps, feel free to contact us via
`OpenCV4Android <https://groups.google.com/group/android-opencv/>`_ discussion group or OpenCV
`Q&A forum <http://answers.opencv.org>`_ . We'll do our best to help you out.


Using OpenCV Library Within Your Android Project
================================================

In this section we will explain how to make some existing project to use OpenCV.
Starting with 2.4.2 release for Android, *OpenCV Manager* is used to provide apps with the best
available version of OpenCV.
You can get more information here: :ref:`Android_OpenCV_Manager` and in these
`slides <https://docs.google.com/a/itseez.com/presentation/d/1EO_1kijgBg_BsjNp2ymk-aarg-0K279_1VZRcPplSuk/present#slide=id.p>`_.


Java
----

Application Development with Async Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using async initialization is a **recommended** way for application development. It uses the OpenCV
Manager to access OpenCV libraries externally installed in the target system.

#. Add OpenCV library project to your workspace. Use menu
   :guilabel:`File -> Import -> Existing project in your workspace`.

   Press :guilabel:`Browse`  button and locate OpenCV4Android SDK
   (:file:`OpenCV-2.4.3-android-sdk/sdk`).

   .. image:: images/eclipse_opencv_dependency0.png
        :alt: Add dependency from OpenCV library
        :align: center

#. In application project add a reference to the OpenCV Java SDK in
   :guilabel:`Project -> Properties -> Android -> Library -> Add` select ``OpenCV Library - 2.4.3``.

   .. image:: images/eclipse_opencv_dependency1.png
        :alt: Add dependency from OpenCV library
        :align: center

In most cases OpenCV Manager may be installed automatically from Google Play. For the case, when
Google Play is not available, i.e. emulator, developer board, etc, you can install it manually
using adb tool. See :ref:`manager_selection` for details.

There is a very base code snippet implementing the async initialization. It shows basic principles.
See the "15-puzzle" OpenCV sample for details.

.. code-block:: java
    :linenos:

    public class MyActivity extends Activity implements HelperCallbackInterface
    {
    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
       @Override
       public void onManagerConnected(int status) {
         switch (status) {
           case LoaderCallbackInterface.SUCCESS:
           {
              Log.i(TAG, "OpenCV loaded successfully");
              // Create and set View
              mView = new puzzle15View(mAppContext);
              setContentView(mView);
           } break;
           default:
           {
              super.onManagerConnected(status);
           } break;
         }
       }
    };

    /** Call on every application resume **/
    @Override
    protected void onResume()
    {
        Log.i(TAG, "called onResume");
        super.onResume();

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mOpenCVCallBack))
        {
            Log.e(TAG, "Cannot connect to OpenCV Manager");
        }
    }

It this case application works with OpenCV Manager in asynchronous fashion. ``OnManagerConnected``
callback will be called in UI thread, when initialization finishes. Please note, that it is not
allowed to use OpenCV calls or load OpenCV-dependent native libs before invoking this callback.
Load your own native libraries that depend on OpenCV after the successful OpenCV initialization.
Default ``BaseLoaderCallback`` implementation treat application context as Activity and calls
``Activity.finish()`` method to exit in case of initialization failure. To override this behavior
you need to override ``finish()`` method of ``BaseLoaderCallback`` class and implement your own
finalization method.


Application Development with Static Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

According to this approach all OpenCV binaries are included into your application package. It is
designed mostly for development purposes. This approach is deprecated for the production code,
release package is recommended to communicate with OpenCV Manager via the async initialization
described above.

#. Add the OpenCV library project to your workspace the same way as for the async initialization
   above. Use menu :guilabel:`File -> Import -> Existing project in your workspace`,
   press :guilabel:`Browse` button and select OpenCV SDK path
   (:file:`OpenCV-2.4.3-android-sdk/sdk`).

   .. image:: images/eclipse_opencv_dependency0.png
        :alt: Add dependency from OpenCV library
        :align: center

#. In the application project add a reference to the OpenCV4Android SDK in
   :guilabel:`Project -> Properties -> Android -> Library -> Add` select ``OpenCV Library - 2.4.3``;

   .. image:: images/eclipse_opencv_dependency1.png
       :alt: Add dependency from OpenCV library
       :align: center

#. If your application project **doesn't have a JNI part**, just copy the corresponding OpenCV
   native libs from :file:`<OpenCV-2.4.3-android-sdk>/sdk/native/libs/<target_arch>` to your
   project directory to folder :file:`libs/<target_arch>`.

   In case of the application project **with a JNI part**, instead of manual libraries copying you
   need to modify your ``Android.mk`` file:
   add the following two code lines after the ``"include $(CLEAR_VARS)"`` and before
   ``"include path_to_OpenCV-2.4.3-android-sdk/sdk/native/jni/OpenCV.mk"``

   .. code-block:: make
      :linenos:

      OPENCV_CAMERA_MODULES:=on
      OPENCV_INSTALL_MODULES:=on

   The result should look like the following:

   .. code-block:: make
      :linenos:

      include $(CLEAR_VARS)

      # OpenCV
      OPENCV_CAMERA_MODULES:=on
      OPENCV_INSTALL_MODULES:=on
      include ../../sdk/native/jni/OpenCV.mk

   After that the OpenCV libraries will be copied to your application :file:`libs` folder during
   the JNI build.v

   Eclipse will automatically include all the libraries from the :file:`libs` folder to the
   application package (APK).

#. The last step of enabling OpenCV in your application is Java initialization code before calling
   OpenCV API. It can be done, for example, in the static section of the ``Activity`` class:

   .. code-block:: java
      :linenos:

      static {
          if (!OpenCVLoader.initDebug()) {
              // Handle initialization error
          }
      }

   If you application includes other OpenCV-dependent native libraries you should load them
   **after** OpenCV initialization:

   .. code-block:: java
      :linenos:

      static {
          if (!OpenCVLoader.initDebug()) {
              // Handle initialization error
          } else {
              System.loadLibrary("my_jni_lib1");
              System.loadLibrary("my_jni_lib2");
          }
      }


Native/C++
----------

To build your own Android application, using OpenCV as native part, the following steps should be
taken:

#. You can use an environment variable to specify the location of OpenCV package or just hardcode
   absolute or relative path in the :file:`jni/Android.mk` of your projects.

#.  The file :file:`jni/Android.mk` should be written for the current application using the common
    rules for this file.

    For detailed information see the Android NDK documentation from the Android NDK archive, in the
    file :file:`<path_where_NDK_is_placed>/docs/ANDROID-MK.html`.

#. The following line:

   .. code-block:: make

      include C:\Work\OpenCV4Android\OpenCV-2.4.3-android-sdk\sdk\native\jni\OpenCV.mk

   Should be inserted into the :file:`jni/Android.mk` file **after** this line:

   .. code-block:: make

      include $(CLEAR_VARS)

#. Several variables can be used to customize OpenCV stuff, but you **don't need** to use them when
   your application uses the `async initialization` via the `OpenCV Manager` API.

   .. note:: These variables should be set **before**  the ``"include .../OpenCV.mk"`` line:

             .. code-block:: make

                OPENCV_INSTALL_MODULES:=on

   Copies necessary OpenCV dynamic libs to the project ``libs`` folder in order to include them
   into the APK.

   .. code-block:: make

      OPENCV_CAMERA_MODULES:=off

   Skip native OpenCV camera related libs copying to the project ``libs`` folder.

   .. code-block:: make

      OPENCV_LIB_TYPE:=STATIC

   Perform static linking with OpenCV. By default dynamic link is used and the project JNI lib
   depends on ``libopencv_java.so``.

#. The file :file:`Application.mk` should exist and should contain lines:

   .. code-block:: make

      APP_STL := gnustl_static
      APP_CPPFLAGS := -frtti -fexceptions

   Also, the line like this one:

   .. code-block:: make

      APP_ABI := armeabi-v7a

   Should specify the application target platforms.

   In some cases a linkage error (like ``"In function 'cv::toUtf16(std::basic_string<...>...
   undefined reference to 'mbstowcs'"``) happens when building an application JNI library,
   depending on OpenCV. The following line in the :file:`Application.mk` usually fixes it:

   .. code-block:: make

      APP_PLATFORM := android-9


#. Either use :ref:`manual <NDK_build_cli>` ``ndk-build`` invocation or
   :ref:`setup Eclipse CDT Builder <CDT_Builder>` to build native JNI lib before (re)building the Java
   part and creating an APK.


Hello OpenCV Sample
===================

Here are basic steps to guide you trough the process of creating a simple OpenCV-centric
application. It will be capable of accessing camera output, processing it and displaying the
result.

#. Open Eclipse IDE, create a new clean workspace, create a new Android project
   :menuselection:`File --> New --> Android Project`.

#. Set name, target, package and ``minSDKVersion`` accordingly.

#. Create a new class :menuselection:`File -> New -> Class`. Name it for example:
   *HelloOpenCVView*.

   .. image:: images/dev_OCV_new_class.png
        :alt: Add a new class.
        :align: center

   * It should extend ``SurfaceView`` class.
   * It also should implement ``SurfaceHolder.Callback``, ``Runnable``.

#. Edit ``HelloOpenCVView`` class.

   * Add an ``import`` line for ``android.content.context``.

   * Modify autogenerated stubs: ``HelloOpenCVView``, ``surfaceCreated``, ``surfaceDestroyed`` and
     ``surfaceChanged``.

     .. code-block:: java
        :linenos:

        package com.hello.opencv.test;

        import android.content.Context;

        public class HelloOpenCVView extends SurfaceView implements Callback, Runnable {

        public HelloOpenCVView(Context context) {
            super(context);
            getHolder().addCallback(this);
        }

        public void surfaceCreated(SurfaceHolder holder) {
            (new Thread(this)).start();
        }

        public void surfaceDestroyed(SurfaceHolder holder) {
            cameraRelease();
        }

        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
            cameraSetup(width, height);
        }

   * Add ``cameraOpen``, ``cameraRelease`` and ``cameraSetup`` voids as shown below.

   * Also, don't forget to add the public void ``run()`` as follows:

     .. code-block:: java
        :linenos:

        public void run() {
            // TODO: loop { getFrame(), processFrame(), drawFrame() }
        }

        public boolean cameraOpen() {
            return false; //TODO: open camera
        }

        private void cameraRelease() {
            // TODO release camera
        }

        private void cameraSetup(int width, int height) {
            // TODO setup camera
        }

#. Create a new ``Activity`` :menuselection:`New -> Other -> Android -> Android Activity` and name
   it, for example: *HelloOpenCVActivity*. For this activity define ``onCreate``, ``onResume`` and
   ``onPause`` voids.

   .. code-block:: java
      :linenos:

      public void onCreate (Bundle savedInstanceState) {
          super.onCreate(savedInstanceState);
          mView = new HelloOpenCVView(this);
          setContentView (mView);
      }

      protected void onPause() {
          super.onPause();
          mView.cameraRelease();
      }

      protected void onResume() {
          super.onResume();
          if( !mView.cameraOpen() ) {
              // MessageBox and exit app
              AlertDialog ad = new AlertDialog.Builder(this).create();
              ad.setCancelable(false); // This blocks the "BACK" button
              ad.setMessage("Fatal error: can't open camera!");
              ad.setButton("OK", new DialogInterface.OnClickListener() {
                  public void onClick(DialogInterface dialog, int which) {
                      dialog.dismiss();
                      finish();
                  }
              });
              ad.show();
          }
      }

#. Add the following permissions to the :file:`AndroidManifest.xml` file:

   .. code-block:: xml
      :linenos:

      </application>

      <uses-permission android:name="android.permission.CAMERA" />
      <uses-feature android:name="android.hardware.camera" />
      <uses-feature android:name="android.hardware.camera.autofocus" />

#. Reference OpenCV library within your project properties.

   .. image:: images/dev_OCV_reference.png
        :alt: Reference OpenCV library.
        :align: center

#. We now need some code to handle the camera. Update the ``HelloOpenCVView`` class as follows:

   .. code-block:: java
      :linenos:

      private VideoCapture mCamera;

      public boolean cameraOpen() {
          synchronized (this) {
              cameraRelease();
              mCamera = new VideoCapture(Highgui.CV_CAP_ANDROID);
              if (!mCamera.isOpened()) {
                  mCamera.release();
                  mCamera = null;
                  Log.e("HelloOpenCVView", "Failed to open native camera");
                  return false;
              }
          }
          return true;
      }

      public void cameraRelease() {
          synchronized(this) {
              if (mCamera != null) {
                   mCamera.release();
                   mCamera = null;
              }
          }
      }

      private void cameraSetup(int width, int height) {
          synchronized (this) {
              if (mCamera != null && mCamera.isOpened()) {
                  List<Size> sizes = mCamera.getSupportedPreviewSizes();
                  int mFrameWidth = width;
                  int mFrameHeight = height;
                  { // selecting optimal camera preview size
                       double minDiff = Double.MAX_VALUE;
                       for (Size size : sizes) {
                           if (Math.abs(size.height - height) < minDiff) {
                               mFrameWidth = (int) size.width;
                               mFrameHeight = (int) size.height;
                               minDiff = Math.abs(size.height - height);
                           }
                       }
                   }
                   mCamera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, mFrameWidth);
                   mCamera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, mFrameHeight);
              }
          }
      }

#. The last step would be to update the ``run()`` void in ``HelloOpenCVView`` class as follows:

   .. code-block:: java
      :linenos:

      public void run() {
          while (true) {
              Bitmap bmp = null;
              synchronized (this) {
                  if (mCamera == null)
                      break;
                  if (!mCamera.grab())
                      break;

                  bmp = processFrame(mCamera);
              }
              if (bmp != null) {
                  Canvas canvas = getHolder().lockCanvas();
                  if (canvas != null) {
                      canvas.drawBitmap(bmp, (canvas.getWidth()  - bmp.getWidth())  / 2,
                                             (canvas.getHeight() - bmp.getHeight()) / 2, null);
                      getHolder().unlockCanvasAndPost(canvas);

                  }
                  bmp.recycle();
              }
          }
      }

      protected Bitmap processFrame(VideoCapture capture) {
          Mat mRgba = new Mat();
          capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
          //process mRgba
          Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
          try {
              Utils.matToBitmap(mRgba, bmp);
          } catch(Exception e) {
              Log.e("processFrame", "Utils.matToBitmap() throws an exception: " + e.getMessage());
              bmp.recycle();
              bmp = null;
          }
          return bmp;
      }
