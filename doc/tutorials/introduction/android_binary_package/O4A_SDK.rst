
.. _O4A_SDK:


OpenCV4Android SDK
******************

This tutorial was designed to help you with installation and configuration of OpenCV4Android SDK.

This guide was written with MS Windows 7 in mind, though it should work with GNU Linux and Apple MacOS as well.

This tutorial assumes you have the following installed and configured:

* JDK

* Android SDK and NDK

* Eclipse IDE

* ADT and CDT plugins for Eclipse

     ..

If you need help with anything of the above, you may refer to our :ref:`android_dev_intro` guide.

If you encounter any error after thoroughly following these steps, feel free to contact us via `OpenCV4Android <https://groups.google.com/group/android-opencv/>`_ discussion group or OpenCV `Q&A forum <http://answers.opencv.org>`_. We'll do our best to help you out.

General info
============

OpenCV4Android SDK package enables development of Android applications with use of OpenCV library.

The structure of package contents looks as follows:

::

    OpenCV-2.4.2-android-sdk
    |_ apk
    |   |_ OpenCV_2.4.2_binary_pack_XXX.apk
    |   |_ OpenCV_2.4.2_Manager.apk
    |
    |_ doc
    |_ samples
    |_ sdk
    |    |_ etc
    |    |_ java
    |    |_ native
    |          |_ 3rdparty
    |          |_ jni
    |          |_ libs
    |               |_ armeabi
    |               |_ armeabi-v7a
    |               |_ x86
    |
    |_ license.txt
    |_ README.android

* :file:`sdk` folder contains OpenCV API and libraries for Android:

* :file:`sdk/java` folder contains an Android library Eclipse project providing OpenCV Java API that can be imported into developer's workspace;

* :file:`sdk/native` folder contains OpenCV C++ headers (for JNI code) and native Android libraries (\*\.so and \*\.a) for ARM-v5, ARM-v7a and x86 architectures;

* :file:`sdk/etc` folder contains Haar and LBP cascades distributed with OpenCV.

* :file:`apk` folder contains Android packages that should be installed on the target Android device to enable OpenCV library access via OpenCV Manager API (see details below).

  On production devices that have access to Google Play Market (and internet) these packages will be installed from Market on the first start of an application using OpenCV Manager API.
  But dev kits without Market or internet require this packages to be installed manually.
  (Install the `Manager.apk` and the corresponding `binary_pack.apk` depending on the device CPU, the Manager GUI provides this info).

  **Note**: installation from internet is the preferable way since we may publish updated versions of this packages on the Market.

* :file:`samples` folder contains sample applications projects and their prebuilt packages (APK).
  Import them into Eclipse workspace (like described below) and browse the code to learn possible ways of OpenCV use on Android.

* :file:`doc` folder contains various OpenCV documentation in PDF format.
  It's also available online at http://docs.opencv.org.

  **Note**: the most recent docs (nightly build) are at http://docs.opencv.org/trunk/.
  Generally, it's more up-to-date, but can refer to not-yet-released functionality.

Starting version 2.4.2 `OpenCV4Android SDK` uses `OpenCV Manager` API for library initialization. `OpenCV Manager` is an Android service based solution providing the following benefits for OpenCV applications developers:

* Compact apk-size, since all applications use the same binaries from Manager and do not store native libs within themselves;

* Hardware specific optimizations are automatically enabled on all supported platforms;

* Automatic updates and bug fixes;

* Trusted OpenCV library source. All packages with OpenCV are published on Google Play;

     ..


For additional information on OpenCV Manager see the:

* |OpenCV4Android_Slides|_

* |OpenCV4Android_Reference|_

     ..

.. |OpenCV4Android_Slides| replace:: Slides
.. _OpenCV4Android_Slides: https://docs.google.com/a/itseez.com/presentation/d/1EO_1kijgBg_BsjNp2ymk-aarg-0K279_1VZRcPplSuk/present#slide=id.p
.. |OpenCV4Android_Reference| replace:: Reference Manual
.. _OpenCV4Android_Reference: http://docs.opencv.org/android/refman.html

Tegra Android Development Pack users
====================================

You may have used `Tegra Android Development Pack <http://developer.nvidia.com/tegra-android-development-pack>`_
(**TADP**) released by **NVIDIA** for Android development environment setup.

Beside Android development tools the TADP 2.0 includes OpenCV4Android SDK 2.4.2, so it can be already installed in your system and you can skip to running the ``face-detection`` sample.

More details regarding TADP can be found in the :ref:`android_dev_intro` guide.

Manual OpenCV4Android SDK setup
===============================

Get the OpenCV4Android SDK
--------------------------

#. Go to the `OpenCV dowload page on SourceForge <http://sourceforge.net/projects/opencvlibrary/files/opencv-android/>`_ and download the latest available version. Currently it's |opencv_android_bin_pack_url|_

#. Create a new folder for Android with OpenCV development. For this tutorial I have unpacked OpenCV to the :file:`C:\\Work\\OpenCV4Android\\` directory.

      .. note:: Better to use a path without spaces in it. Otherwise you may have problems with :command:`ndk-build`.

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
------------------------------------------

#. Start *Eclipse* and choose your workspace location.

   We recommend to start working with OpenCV for Android from a new clean workspace. A new Eclipse workspace can for example be created in the folder where you have unpacked OpenCV4Android SDK package:

      .. image:: images/eclipse_1_choose_workspace.png
         :alt: Choosing C:\Work\android-opencv\ as workspace location
         :align: center

#. Import OpenCV library and samples into workspace.

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

   Just give a minute to Eclipse to complete initialization.

   In some cases these errors disappear after :menuselection:`Project --> Clean... --> Clean all --> OK`
   or after pressing :kbd:`F5` (for Refresh action) when selecting error-label-marked projects in :guilabel:`Package Explorer`.

   Sometimes more advanced manipulations are required:

   The provided projects are configured for ``API 11`` target (and ``API 9`` for the library) that can be missing platform in your Android SDK.
   After right click on any project select  :guilabel:`Properties` and then :guilabel:`Android` on the left pane.
   Click some target with `API Level` 11 or higher:

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

.. _Running_OpenCV_Samples:

Running OpenCV Samples
----------------------

At this point you should be able to build and run the samples. Keep in mind, that ``face-detection``, ``Tutorial 3`` and ``Tutorial 4`` include some native code and require Android NDK and CDT plugin for Eclipse to build working applications.
If you haven't installed these tools see the corresponding section of :ref:`Android_Dev_Intro`.

Also, please consider that ``Tutorial 0`` and ``Tutorial 1`` samples use Java Camera API that definitelly accessible on emulator from the Android SDK.
Other samples use OpenCV Native Camera which may not work with emulator.

.. note:: Recent *Android SDK tools, revision 19+* can run ARM v7a OS images but they available not for all Android versions.

Well, running samples from Eclipse is very simple:

* Connect your device with :command:`adb` tool from Android SDK or create an emulator with camera support.

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

  Chances are that on the first launch you will not have the `OpenCV Manager <https://docs.google.com/a/itseez.com/presentation/d/1EO_1kijgBg_BsjNp2ymk-aarg-0K279_1VZRcPplSuk/present#slide=id.p>`_ package installed.
  In this case you will see the following message:

  .. image:: images/android_emulator_opencv_manager_fail.png
     :alt: You will see this message if you have no OpenCV Manager installed
     :align: center

  To get rid of the message you will need to install `OpenCV Manager` and the appropriate `OpenCV binary pack`.
  Simply tap :menuselection:`Yes` if you have *Google Play Market* installed on your device/emulator. It will redirect you to the corresponding page on *Google Play Market*.

  If you have no access to the *Market*, which is often the case with emulators - you will need to install the packages from OpenCV4Android SDK folder manually. Open the console/terminal and type in the following two commands:

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

What's next
===========

Now, when you have your instance of OpenCV4Adroid SDK set up and configured, you may want to proceed to using OpenCV in your own application. You can learn how to do that in a separate :ref:`dev_with_OCV_on_Android` tutorial.