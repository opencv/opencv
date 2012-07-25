
.. _Android_Binary_Package:


Using OpenCV4Android SDK with Eclipse
*************************************

This tutorial was tested using Ubuntu 10.04 and Windows 7 SP1 operating systems.
However, it should also work with any other **OS**\ , supported by Android SDK.
If you encounter any error after thoroughly following these steps, feel free to contact us via `OpenCV4Android <https://groups.google.com/group/android-opencv/>`_ discussion group or OpenCV `Q&A forum <http://answers.opencv.org>`_ . We'll do our best to help you out.

Quick environment setup for Android development
===============================================

If you are making a clean environment install, then you can try `Tegra Android Development Pack <http://developer.nvidia.com/tegra-android-development-pack>`_
(**TADP**) released by **NVIDIA**.

When unpacked, TADP will cover all of the environment setup automatically and you can go straight to the section :ref:`Get_the_OpenCV_package_for_Android_development`.

If you are a beginner in Android development then we also recommend you to start with TADP.

.. note:: *NVIDIA*\ 's Tegra Android Development Pack includes some special features for |Nvidia_Tegra_Platform|_ but its use is not limited to *Tegra* devices only.

  * You need at least *1.6 Gb* free disk space for the install.

  * TADP will download Android SDK platforms and Android NDK from Google's server, so Internet connection is required for the installation.

  * TADP may ask you to flash your development kit at the end of installation process. Just skip this step if you have no |Tegra_Ventana_Development_Kit|_\ .

  * (``UNIX``) TADP will ask you for *root* in the middle of installation, so you need to be a member of *sudo* group.

     ..


.. |Nvidia_Tegra_Platform| replace:: *NVIDIA*\ ’s Tegra platform
.. _Nvidia_Tegra_Platform: http://developer.nvidia.com/node/19071
.. |Tegra_Ventana_Development_Kit| replace:: Tegra Ventana Development Kit
.. _Tegra_Ventana_Development_Kit: http://developer.nvidia.com/tegra-ventana-development-kit

.. _Android_Environment_Setup_Lite:

Manual environment setup for Android Development
================================================

You need the following to be installed:

#. **Sun JDK 6**

   Visit `Java SE Downloads page <http://www.oracle.com/technetwork/java/javase/downloads/>`_ and download an installer for your OS.

   Here is a detailed :abbr:`JDK (Java Development Kit)` `installation guide <http://source.android.com/source/initializing.html#installing-the-jdk>`_
   for Ubuntu and Mac OS (only JDK sections are applicable for OpenCV)

   .. note:: OpenJDK is not suitable for Android development, since Android SDK supports only Sun JDK.
        If you use Ubuntu, after installation of Sun JDK you should run the following command to set Sun java environment:

        .. code-block:: bash

           sudo update-java-alternatives --set java-6-sun

#. **Android SDK**

   Get the latest ``Android SDK`` from http://developer.android.com/sdk/index.html

   Here is Google's `install guide <http://developer.android.com/sdk/installing.html>`_ for the SDK.

   .. note:: If you choose SDK packed into a Windows installer, then you should have 32-bit JRE installed. It is not a prerequisite for Android development, but installer is a x86 application and requires 32-bit Java runtime.

   .. note:: If you are running x64 version of Ubuntu Linux, then you need ia32 shared libraries for use on amd64 and ia64 systems to be installed. You can install them with the following command:

      .. code-block:: bash

         sudo apt-get install ia32-libs

      For Red Hat based systems the following command might be helpful:

      .. code-block:: bash

         sudo yum install libXtst.i386

#. **Android SDK components**

   You need the following SDK components to be installed:

   * *Android SDK Tools, revision14* or newer.

     Older revisions should also work, but they are not recommended.

   * *SDK Platform Android 3.0, API 11* and *Android 2.3.1, API 9*.

     The minimal platform supported by OpenCV Java API is **Android 2.2** (API 8). This is also the minimum API Level required for the provided samples to run.
     See the ``<uses-sdk android:minSdkVersion="8"/>`` tag in their **AndroidManifest.xml** files.
     But for successful compilation of some samples the **target** platform should be set to Android 3.0 (API 11) or higher. It will not prevent them from running on  Android 2.2.

     .. image:: images/android_sdk_and_avd_manager.png
        :height: 500px
        :alt: Android SDK Manager
        :align: center

     See `Adding SDK Components  <http://developer.android.com/sdk/adding-components.html>`_ for help with installing/updating SDK components.

#. **Eclipse IDE**

   Check the `Android SDK System Requirements <http://developer.android.com/sdk/requirements.html>`_ document for a list of Eclipse versions that are compatible with the Android SDK.
   For OpenCV 2.4.x we recommend Eclipse 3.7 (Indigo) or later versions. They work well for OpenCV under both Windows and Linux.

   If you have no Eclipse installed, you can get it from the `official site <http://www.eclipse.org/downloads/>`_.

#. **ADT plugin for Eclipse**

   These instructions are copied from `Android Developers site <http://developer.android.com/sdk/eclipse-adt.html#downloading>`_, check it out in case of any ADT-related problem.

   Assuming that you have Eclipse IDE installed, as described above, follow these steps to download and install the ADT plugin:

   #. Start Eclipse, then select :menuselection:`Help --> Install New Software...`
   #. Click :guilabel:`Add` (in the top-right corner).
   #. In the :guilabel:`Add Repository` dialog that appears, enter "ADT Plugin" for the Name and the following URL for the Location:

      https://dl-ssl.google.com/android/eclipse/

   #. Click :guilabel:`OK`

      .. note:: If you have trouble acquiring the plugin, try using "http" in the Location URL, instead of "https" (https is preferred for security reasons).

   #. In the :guilabel:`Available Software` dialog, select the checkbox next to :guilabel:`Developer Tools` and click :guilabel:`Next`.
   #. In the next window, you'll see a list of the tools to be downloaded. Click :guilabel:`Next`.
   #. Read and accept the license agreements, then click :guilabel:`Finish`.

      .. note:: If you get a security warning saying that the authenticity or validity of the software can't be established, click :guilabel:`OK`.

   #. When the installation completes, restart Eclipse.

.. _Get_the_OpenCV_package_for_Android_development:

Get the OpenCV4Android SDK
==========================

#. Go to the `OpenCV dowload page on SourceForge <http://sourceforge.net/projects/opencvlibrary/files/opencv-android/>`_ and download the latest available version. Currently it's |opencv_android_bin_pack_url|_

#. Create a new folder for development for Android with OpenCV. For this tutorial I have unpacked OpenCV to the :file:`C:\\Work\\OpenCV4Android\\` directory.

      .. note:: Better to use a path without spaces in it. Otherwise you will probably have problems with :command:`ndk-build`.

#. Unpack the OpenCV package into the chosen directory.

   You can unpack it using any popular archiver (e.g with |seven_zip|_):

   .. image:: images/android_package_7zip.png
      :alt: Exploring OpenCV package with 7-Zip
      :align: center

   On Unix you can use the following command:

   .. code-block:: bash

      unzip ~/Downloads/OpenCV-2.4.2-android-sdk.zip

.. |opencv_android_bin_pack| replace:: OpenCV-2.4.2-android-sdk.zip
.. _opencv_android_bin_pack_url: http://sourceforge.net/projects/opencvlibrary/files/opencv-android/2.4.2/OpenCV-2.4.2-android-sdk.zip/download
.. |opencv_android_bin_pack_url| replace:: |opencv_android_bin_pack|
.. |seven_zip| replace:: 7-Zip
.. _seven_zip: http://www.7-zip.org/

Open OpenCV library and samples in Eclipse
==========================================

#. Start *Eclipse* and choose your workspace location.

   I recommend to start familiarizing yourself with OpenCV for Android from a new clean workspace. So I have chosen my OpenCV package directory for the new workspace:

      .. image:: images/eclipse_1_choose_workspace.png
         :alt: Choosing C:\Work\android-opencv\ as workspace location
         :align: center

#. Configure your ADT plugin.

   .. important:: In most cases the ADT plugin finds Android SDK automatically, but  sometimes  it  fails and shows the following prompt:

      .. image:: images/eclipse_1a_locate_sdk.png
         :alt: Locating Android SDK
         :align: center

   Select  :guilabel:`Use existing SDKs` option, browse for Android SDK folder and click :guilabel:`Finish`.

   To make sure the SDK folder is set correctly do the following step taken from  `Configuring the ADT Plugin  <http://developer.android.com/sdk/installing/installing-adt.html#Configure>`_ tutorial by *Google*:

   * Select :menuselection:`Window --> Preferences...` to open the Preferences panel (Mac OS X: :menuselection:`Eclipse --> Preferences`):

      .. image:: images/eclipse_2_window_preferences.png
         :alt: Select Window > Preferences...
         :align: center

   * Select :guilabel:`Android` in the left panel.

     You may see a dialog asking whether you want to send usage statistics to *Google*. If so, make your choice and click :guilabel:`Proceed`.

     If the Android SDK folder isn't configured you'll see the following:

      .. image:: images/eclipse_3_preferences_android.png
         :alt: Select Android from the left panel
         :align: center

   * To locate the SDK manually, click :guilabel:`Browse...`.

   * Click :guilabel:`Apply` button at the bottom right corner of main panel.

     If the SDK folder is already configured correctly you'll see something like this:

      .. image:: images/eclipse_4_locate_sdk.png
         :alt: Locate Android SDK
         :align: center

   * Click :guilabel:`OK` to close preferences dialog.

#. Import OpenCV and samples into workspace.

   OpenCV library is packed as a ready-for-use `Android Library Project
   <http://developer.android.com/guide/developing/projects/index.html#LibraryProjects>`_. You can simply reference it in your projects.

   Each sample included into the |opencv_android_bin_pack| is a regular Android project that already references OpenCV library.
   Follow the steps below to import OpenCV and samples into the workspace:

   * Right click on the :guilabel:`Package Explorer` window and choose :guilabel:`Import...` option from the context menu:

      .. image:: images/eclipse_5_import_command.png
         :alt: Select Import... from context menu
         :align: center

   * In the main panel select :menuselection:`General --> Existing Projects into Workspace` and press :guilabel:`Next` button:

      .. image:: images/eclipse_6_import_existing_projects.png
         :alt: General > Existing Projects into Workspace
         :align: center

   * In the :guilabel:`Select root directory` field locate your OpenCV package folder. Eclipse should automatically locate OpenCV library and samples:

      .. image:: images/eclipse_7_select_projects.png
         :alt: Locate OpenCV library and samples
         :align: center

   * Click :guilabel:`Finish` button to complete the import operation.

   After clicking :guilabel:`Finish` button Eclipse will load all selected projects into workspace. Numerous errors will be indicated:

      .. image:: images/eclipse_8_false_alarm.png
         :alt: Confusing Eclipse screen with numerous errors
         :align: center

   However, **all these errors are only false-alarms**!

   To get rid of these misleading error notifications select OpenCV library in :guilabel:`Package Explorer` and press :kbd:`F5`. Then select a sample (except first samples in *Tutorial Base* and *Tutorial Advanced*) and press :kbd:`F5` again.
   
   In some cases these errors disappear after :menuselection:`Project --> Clean... --> Clean all --> OK`.

   Sometimes more advanced manipulations are required:

   * The provided projects are configured for `API 11` target that can be missing platform in your Android SDK. After right click on any project select  :guilabel:`Properties` and then :guilabel:`Android` on the left pane. Click some target with `API Level` 11 or higher:

      .. image:: images/eclipse_8a_target.png
         :alt: Updating target
         :align: center

   Eclipse will rebuild your workspace and error icons will disappear one by one:

      .. image:: images/eclipse_9_errors_dissapearing.png
         :alt: After small help Eclipse removes error icons!
         :align: center

   Once Eclipse completes build you will have the clean workspace without any build errors:

      .. image:: images/eclipse_10_crystal_clean.png
         :alt: OpenCV package imported into Eclipse
         :align: center

Running OpenCV Samples
======================

At this point you should be able to build and run all samples except ``face-detection``, ``Tutorial 3`` and ``Tutorial 4``. These samples include native code and require Android NDK to build working applications, see the next tutorial :ref:`Android_Binary_Package_with_NDK` to learn how to compile them.

Also, please consider that ``Tutorial 0 - Android Camera`` and ``Tutorial 1 - Add OpenCV`` samples are able to run on emulator from the Android SDK. Other samples use OpenCV Native Camera which may not work with emulator.

.. note:: Latest *Android SDK tools, revision 19* can run ARM v7a OS images but *Google* provides such image for Android 4.x only.

Well, running samples from Eclipse is very simple:

* Connect your device with :command:`adb` tool from Android SDK or create emulator with camera support.

   * See `Managing Virtual Devices
     <http://developer.android.com/guide/developing/devices/index.html>`_ document for help with Android Emulator.
   * See `Using Hardware Devices
     <http://developer.android.com/guide/developing/device.html>`_ for help with real devices (not emulators).


* Select project you want to start in :guilabel:`Package Explorer` and just press :kbd:`Ctrl + F11` or select option :menuselection:`Run --> Run` from the main menu, or click :guilabel:`Run` button on the toolbar.

  .. note:: Android Emulator can take several minutes to start. So, please, be patient.

* On the first run Eclipse will ask you about the running mode for your application:

  .. image:: images/eclipse_11_run_as.png
     :alt: Run sample as Android Application
     :align: center

* Select the :guilabel:`Android Application` option and click :guilabel:`OK` button. Eclipse will install and run the sample.

  Chances are that on the first launch you will not have the `OpenCV Manager <https://docs.google.com/a/itseez.com/presentation/d/1EO_1kijgBg_BsjNp2ymk-aarg-0K279_1VZRcPplSuk/present#slide=id.p>`_ package installed. In this case you will see the following message:

  .. image:: images/android_emulator_opencv_manager_fail.png
     :alt: You will see this message if you have no OpenCV Manager installed
     :align: center
	 
  To get rid of the message you will need to install `OpenCV Manager` and the appropriate `OpenCV binary package`. Simply tap :menuselection:`Yes` if you have *Google Play Market* installed on your device/emulator. It will redirect you to the corresponding page on *Google Play Market*.
  
  If you have no access to the *Market*, which is often the case with emulators — you will need to install the packages from OpenCV4Android SDK folder manually. Open the console/terminal and type in the following two commands:
  
  .. code-block:: sh
    :linenos:

    <Android SDK path>/platform-tools/adb install <OpenCV4Android SDK path>/apk/OpenCV_2.4.2_Manager.apk
    <Android SDK path>/platform-tools/adb install <OpenCV4Android SDK path>/apk/OpenCV_2.4.2_binary_pack_armv7a.apk
	
  If you're running Windows, that will probably look like this:
	
  .. image:: images/install_opencv_manager_with_adb.png
     :alt: Run these commands in the console to install OpenCV Manager
     :align: center
	 
  When done, you will be able to run OpenCV samples on your device/emulator seamlessly.
  
* Here is ``Tutorial 2 - Use OpenCV Camera`` sample, running on top of stock camera-preview of the emulator.

  .. image:: images/emulator_canny.png
     :height: 600px
     :alt: Tutorial 1 Basic - 1. Add OpenCV - running Canny
     :align: center

How to use OpenCV library project in your application
=====================================================

In this section we will explain how to make some existing project to use OpenCV.
Starting with 2.4.2 release for Android, *OpenCV Manager* is used to provide apps with the best available version of OpenCV.
You can get more information here: `Intro slides <https://docs.google.com/a/itseez.com/presentation/d/1EO_1kijgBg_BsjNp2ymk-aarg-0K279_1VZRcPplSuk/present#slide=id.p>`_ and :ref:`Android_OpenCV_Manager`.

Application development with async initialization
-------------------------------------------------

Using async initialization is a **recommended** way for application development. It uses the OpenCV Manager to access OpenCV libraries.

#. Add OpenCV library project to your workspace. Use menu :guilabel:`File –> Import –> Existing project in your workspace`,
   press :guilabel:`Browse`  button and locate OpenCV4Android SDK (:file:`OpenCV-2.4.2-android-sdk/sdk`).

   .. image:: images/eclipse_opencv_dependency0.png
        :alt: Add dependency from OpenCV library
        :align: center

#. In application project add a reference to the OpenCV Java SDK in :guilabel:`Project –> Properties –> Android –> Library –> Add` select ``OpenCV Library - 2.4.2``.

   .. image:: images/eclipse_opencv_dependency1.png
        :alt: Add dependency from OpenCV library
        :align: center

To use OpenCV Manager-based approach you need to install packages with the `Manager` and `OpenCV binary pack` for you platform.
You can do it using Google Play Market or manually with ``adb`` tool:

  .. code-block:: sh
    :linenos:

    <Android SDK path>/platform-tools/adb install <OpenCV4Android SDK path>/apk/OpenCV_2.4.2_Manager.apk
    <Android SDK path>/platform-tools/adb install <OpenCV4Android SDK path>/apk/OpenCV_2.4.2_binary_pack_armv7a.apk
	
There is a very base code snippet implementing the async initialization. It shows basic principles. See the "15-puzzle" OpenCV sample for details.

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

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mOpenCVCallBack))
        {
          Log.e(TAG, "Cannot connect to OpenCV Manager");
        }
    }

    // ...
    }

It this case application works with OpenCV Manager in asynchronous fashion. ``OnManagerConnected`` callback will be called in UI thread, when initialization finishes.
Please note, that it is not allowed to use OpenCV calls or load OpenCV-dependent native libs before invoking this callback. 
Load your own native libraries that depend on OpenCV after the successful OpenCV initialization.

Application development with static initialization
--------------------------------------------------

According to this approach all OpenCV binaries are included into your application package. It is designed mostly for development purposes.
This approach is deprecated for the production code, release package is recommended to communicate with OpenCV Manager via the async initialization described above.

#. Add the OpenCV library project to your workspace the same way as for the async initialization above.
   Use menu :guilabel:`File –> Import –> Existing project in your workspace`, push :guilabel:`Browse` button and select OpenCV SDK path (:file:`OpenCV-2.4.2-android-sdk/sdk`).

   .. image:: images/eclipse_opencv_dependency0.png
        :alt: Add dependency from OpenCV library
        :align: center

#. In the application project add a reference to the OpenCV4Android SDK in :guilabel:`Project –> Properties –> Android –> Library –> Add` select ``OpenCV Library - 2.4.2``;

   .. image:: images/eclipse_opencv_dependency1.png
       :alt: Add dependency from OpenCV library
       :align: center

#. If your application project **doesn't have a JNI part**, just copy the OpenCV native libs to your project directory to folder :file:`libs/target_arch/`.
   
   In case of the application project **with a JNI part**, instead of manual libraries copying you need to modify your ``Android.mk`` file: 
   add the following two code lines after the ``"include $(CLEAR_VARS)"`` and before ``"include path_to_OpenCV-2.4.2-android-sdk/sdk/native/jni/OpenCV.mk"``

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

   After that the OpenCV libraries will be copied to your application :file:`libs` folder during the JNI part build.

   Eclipse will automatically include all the libraries from the :file:`libs` folder to the application package (APK).

#. The last step of enabling OpenCV in your application is Java initialization code before call to OpenCV API.
   It can be done, for example, in the static section of the ``Activity`` class:

    .. code-block:: java
       :linenos:

        static {
            if (!OpenCVLoader.initDebug()) {
                // Handle initialization error
            }
        }

    If you application includes other OpenCV-dependent native libraries you should load them **after** OpenCV initialization:

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

What's next?
============

Read the :ref:`Android_Binary_Package_with_NDK` tutorial to learn how to add native OpenCV code to your Android project.
