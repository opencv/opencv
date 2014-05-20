
.. _O4A_SDK:


OpenCV4Android SDK
******************

This tutorial was designed to help you with installation and configuration of OpenCV4Android SDK.

This guide was written with MS Windows 7 in mind, though it should work with GNU Linux and Apple
Mac OS as well.

This tutorial assumes you have the following software installed and configured:

* JDK

* Android SDK and NDK

* Eclipse IDE

* ADT and CDT plugins for Eclipse

     ..

If you need help with anything of the above, you may refer to our :ref:`android_dev_intro` guide.

If you encounter any error after thoroughly following these steps, feel free to contact us via
`OpenCV4Android <https://groups.google.com/group/android-opencv/>`_ discussion group or
OpenCV `Q&A forum <http://answers.opencv.org>`_. We'll do our best to help you out.

Tegra Android Development Pack users
====================================

You may have used `Tegra Android Development Pack <http://developer.nvidia.com/tegra-android-development-pack>`_
(**TADP**) released by **NVIDIA** for Android development environment setup.

Beside Android development tools the TADP 2.0 includes OpenCV4Android SDK, so it can be already
installed in your system and you can skip to :ref:`Running_OpenCV_Samples` section of this tutorial.

More details regarding TADP can be found in the :ref:`android_dev_intro` guide.

General info
============

OpenCV4Android SDK package enables development of Android applications with use of OpenCV library.

The structure of package contents looks as follows:

::

    OpenCV-2.4.9-android-sdk
    |_ apk
    |   |_ OpenCV_2.4.9_binary_pack_armv7a.apk
    |   |_ OpenCV_2.4.9_Manager_2.18_XXX.apk
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
    |_ LICENSE
    |_ README.android

* :file:`sdk` folder contains OpenCV API and libraries for Android:

* :file:`sdk/java` folder contains an Android library Eclipse project providing OpenCV Java API that
  can be imported into developer's workspace;

* :file:`sdk/native` folder contains OpenCV C++ headers (for JNI code) and native Android libraries
  (\*\.so and \*\.a) for ARM-v5, ARM-v7a and x86 architectures;

* :file:`sdk/etc` folder contains Haar and LBP cascades distributed with OpenCV.

* :file:`apk` folder contains Android packages that should be installed on the target Android device
  to enable OpenCV library access via OpenCV Manager API (see details below).

  On production devices that have access to Google Play Market (and Internet) these packages will be
  installed from Market on the first start of an application using OpenCV Manager API.
  But devkits without Market or Internet connection require this packages to be installed manually.
  Install the `Manager.apk` and optional `binary_pack.apk` if it needed.
  See :ref:`manager_selection` for details.

  .. note:: Installation from Internet is the preferable way since OpenCV team may publish updated
            versions of this packages on the Market.

* :file:`samples` folder contains sample applications projects and their prebuilt packages (APK).
  Import them into Eclipse workspace (like described below) and browse the code to learn possible
  ways of OpenCV use on Android.

* :file:`doc` folder contains various OpenCV documentation in PDF format.
  It's also available online at http://docs.opencv.org.

  .. note:: The most recent docs (nightly build) are at http://docs.opencv.org/2.4.
            Generally, it's more up-to-date, but can refer to not-yet-released functionality.

.. TODO: I'm not sure that this is the best place to talk about OpenCV Manager

Starting from version 2.4.3 `OpenCV4Android SDK` uses `OpenCV Manager` API for library
initialization. `OpenCV Manager` is an Android service based solution providing the following
benefits for OpenCV applications developers:

* Compact apk-size, since all applications use the same binaries from Manager and do not store
  native libs within themselves;

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

Manual OpenCV4Android SDK setup
===============================

Get the OpenCV4Android SDK
--------------------------

#. Go to the `OpenCV download page on SourceForge <http://sourceforge.net/projects/opencvlibrary/files/opencv-android/>`_
   and download the latest available version. Currently it's |opencv_android_bin_pack_url|_.

#. Create a new folder for Android with OpenCV development. For this tutorial we have unpacked
   OpenCV SDK to the :file:`C:\\Work\\OpenCV4Android\\` directory.

   .. note:: Better to use a path without spaces in it. Otherwise you may have problems with :command:`ndk-build`.

#. Unpack the SDK archive into the chosen directory.

   You can unpack it using any popular archiver (e.g with |seven_zip|_):

   .. image:: images/android_package_7zip.png
      :alt: Exploring OpenCV package with 7-Zip
      :align: center

   On Unix you can use the following command:

   .. code-block:: bash

      unzip ~/Downloads/OpenCV-2.4.9-android-sdk.zip

.. |opencv_android_bin_pack| replace:: :file:`OpenCV-2.4.9-android-sdk.zip`
.. _opencv_android_bin_pack_url: http://sourceforge.net/projects/opencvlibrary/files/opencv-android/2.4.9/OpenCV-2.4.9-android-sdk.zip/download
.. |opencv_android_bin_pack_url| replace:: |opencv_android_bin_pack|
.. |seven_zip| replace:: 7-Zip
.. _seven_zip: http://www.7-zip.org/

Import OpenCV library and samples to the Eclipse
------------------------------------------------

#. Start Eclipse and choose your workspace location.

   We recommend to start working with OpenCV for Android from a new clean workspace. A new Eclipse
   workspace can for example be created in the folder where you have unpacked OpenCV4Android SDK package:

      .. image:: images/eclipse_1_choose_workspace.png
         :alt: Choosing C:\Work\android-opencv\ as workspace location
         :align: center

#. Import OpenCV library and samples into workspace.

   OpenCV library is packed as a ready-for-use `Android Library Project
   <http://developer.android.com/guide/developing/projects/index.html#LibraryProjects>`_.
   You can simply reference it in your projects.

   Each sample included into the |opencv_android_bin_pack| is a regular Android project that already
   references OpenCV library. Follow the steps below to import OpenCV and samples into the workspace:

   .. note:: OpenCV samples are indeed **dependent** on OpenCV library project so don't forget to import it to your workspace as well.

   * Right click on the :guilabel:`Package Explorer` window and choose :guilabel:`Import...` option
     from the context menu:

      .. image:: images/eclipse_5_import_command.png
         :alt: Select Import... from context menu
         :align: center

   * In the main panel select :menuselection:`General --> Existing Projects into Workspace` and
     press :guilabel:`Next` button:

      .. image:: images/eclipse_6_import_existing_projects.png
         :alt: General > Existing Projects into Workspace
         :align: center

   * In the :guilabel:`Select root directory` field locate your OpenCV package folder. Eclipse
     should automatically locate OpenCV library and samples:

      .. image:: images/eclipse_7_select_projects.png
         :alt: Locate OpenCV library and samples
         :align: center

   * Click :guilabel:`Finish` button to complete the import operation.

   After clicking :guilabel:`Finish` button Eclipse will load all selected projects into workspace,
   and you have to wait some time while it is building OpenCV samples. Just give a minute to
   Eclipse to complete initialization.

   .. warning :: After the initial import, on a non-Windows (Linux and Mac OS) operating system Eclipse
              will still show build errors for applications with native C++ code. To resolve the
              issues, please do the following:

              Open :guilabel:`Project Properties -> C/C++ Build`, and replace "Build command" text
              to ``"${NDKROOT}/ndk-build"`` (remove .cmd at the end).

   .. note :: In some cases the build errors don't disappear, then try the following actions:

              * right click on ``OpenCV Library`` project -> :guilabel:`Android Tools -> Fix Project Properties`,
                then menu :guilabel:`Project -> Clean... -> Clean all`
              * right click on the project with errors -> :guilabel:`Properties -> Android`, make sure the
                ``Target`` is selected and is ``Android 3.0`` or higher
              * check the build errors in the :guilabel:`Problems` view window and try to resolve them by yourselves

   .. image:: images/eclipse_cdt_cfg4.png
      :alt: Configure CDT
      :align: center

   Once Eclipse completes build you will have the clean workspace without any build errors:

      .. image:: images/eclipse_10_crystal_clean.png
         :alt: OpenCV package imported into Eclipse
         :align: center

.. _Running_OpenCV_Samples:

Running OpenCV Samples
----------------------

At this point you should be able to build and run the samples. Keep in mind, that
``face-detection`` and ``Tutorial 2 - Mixed Processing`` include some native code and
require Android NDK and NDK/CDT plugin for Eclipse to build working applications. If you haven't
installed these tools, see the corresponding section of :ref:`Android_Dev_Intro`.

.. warning:: Please consider that some samples use Android Java Camera API, which is accessible
             with an AVD. But most of samples use OpenCV Native Camera which **may not work** with
             an emulator.

.. note:: Recent *Android SDK tools, revision 19+* can run ARM v7a OS images but they available not
          for all Android versions.

Well, running samples from Eclipse is very simple:

* Connect your device with :command:`adb` tool from Android SDK or create an emulator with camera support.

  * See `Managing Virtual Devices
    <http://developer.android.com/guide/developing/devices/index.html>`_ document for help with Android Emulator.
  * See `Using Hardware Devices
    <http://developer.android.com/guide/developing/device.html>`_ for help with real devices (not emulators).


* Select project you want to start in :guilabel:`Package Explorer` and just press :kbd:`Ctrl + F11`
  or select option :menuselection:`Run --> Run` from the main menu, or click :guilabel:`Run` button on the toolbar.

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

  If you have no access to the *Market*, which is often the case with emulators - you will need to install the packages from OpenCV4Android SDK folder manually. See :ref:`manager_selection` for details.

  .. code-block:: sh
    :linenos:

    <Android SDK path>/platform-tools/adb install <OpenCV4Android SDK path>/apk/OpenCV_2.4.9_Manager_2.18_armv7a-neon.apk

  .. note:: ``armeabi``, ``armv7a-neon``, ``arm7a-neon-android8``, ``mips`` and ``x86`` stand for
            platform targets:

            * ``armeabi`` is for ARM v5 and ARM v6 architectures with Android API 8+,

            * ``armv7a-neon`` is for NEON-optimized ARM v7 with Android API 9+,

            * ``arm7a-neon-android8`` is for NEON-optimized ARM v7 with Android API 8,

            * ``mips`` is for MIPS architecture with Android API 9+,

            * ``x86`` is for Intel x86 CPUs with Android API 9+.

            If using hardware device for testing/debugging, run the following command to learn
            its CPU architecture:

            .. code-block:: sh

               adb shell getprop ro.product.cpu.abi

            If you're using an AVD emulator, go :menuselection:`Window > AVD Manager` to see the
            list of availible devices. Click :menuselection:`Edit` in the context menu of the
            selected device. In the window, which then pop-ups, find the CPU field.

            You may also see section :ref:`manager_selection` for details.


  When done, you will be able to run OpenCV samples on your device/emulator seamlessly.

* Here is ``Sample - image-manipulations`` sample, running on top of stock camera-preview of the emulator.

  .. image:: images/emulator_canny.png
     :alt: 'Sample - image-manipulations' running Canny
     :align: center


What's next
===========

Now, when you have your instance of OpenCV4Adroid SDK set up and configured,
you may want to proceed to using OpenCV in your own application.
You can learn how to do that in a separate :ref:`dev_with_OCV_on_Android` tutorial.
