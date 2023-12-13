Android Development with OpenCV {#tutorial_dev_with_OCV_on_Android}
===============================

@prev_tutorial{tutorial_android_dev_intro}
@next_tutorial{tutorial_android_dnn_intro}

|    |    |
| -: | :- |
| Original authors | Alexander Panov, Rostislav Vasilikhin |
| Compatibility | OpenCV >= 4.9.0 |

This tutorial has been created to help you use OpenCV library within your Android project.

This guide was checked on Ubuntu but contains no platform-dependent parts, therefore should be compatible with any OS supported by Android Studio and OpenCV4Android SDK.

This tutorial assumes you have the following installed and configured:

-   Android Studio
-   JDK
-   Android SDK and NDK
-   OpenCV for Android SDK from official [release page on Github](https://github.com/opencv/opencv/releases)
    or [SourceForge](https://sourceforge.net/projects/opencvlibrary/). Advanced: as alternative the SDK may be
    built from source code by [instruction on wiki](https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build).

If you need help with anything of the above, you may refer to our @ref tutorial_android_dev_intro guide.

If you encounter any error after thoroughly following these steps, feel free to contact us via OpenCV [forum](https://forum.opencv.org). We'll do our best to help you out.

Hello OpenCV sample
-------------------

In this section we're gonna create a simple app that does nothing but OpenCV loading. In next section we'll extend it to support camera.

In addition to this instruction you can use some video guide, for example [this one](https://www.youtube.com/watch?v=bR7lL886-uc&ab_channel=ProgrammingHut)

1. Open Android Studio and create empty project by choosing ***Empty Views Activity***

    ![](images/create_empty_project.png)

2. Setup the project:
    - Choose ***Java*** language
    - Choose ***Groovy DSL*** build configuration language
    - Choose ***Minumum SDK*** to the same version number as was used during OpenCV 4 Android build
        - If you don't know it, you can find it in file `OpenCV-android-sdk/sdk/build.gradle` at `android -> defaultConfig -> minSdkVersion`

    ![](images/setup_project.png)


3. Click ***File -> New -> Import module...*** and select OpenCV SDK path

    ![](images/sdk_path.png)

4. Set module name as `OpenCV` and press `Finish`

    ![](images/module_name.png)

5. OpenCV also provides experiemental Kotlin support. Please add Android Kotlin plugin to `MyApplication/OpenCV/build.gradle` file:
    @code{.gradle}
    plugins {
        id 'org.jetbrains.kotlin.android' version '1.7.10' #version may differ for your setup
    }
    @endcode
    Like this:
    ![](images/gradle_ocv_fix.png)
    If you don't do this, you may get an error:
    @code
    Task failed with an exception.
    -----------
    * Where:
    Build file '/home/alexander/AndroidStudioProjects/MyApplication/opencv/build.gradle' line: 4

    * What went wrong:
    A problem occurred evaluating project ':opencv'.
    > Plugin with id 'kotlin-android' not found.
    @endcode
    The fix was found [here](https://stackoverflow.com/questions/73225714/import-opencv-sdk-to-android-studio-chipmunk)

6. OpenCV project uses `aidl` and `buildConfig` features. Please enable them in
   `MyApplication/OpenCV/build.gradle` file to `android` block:
    @code{.gradle}
    buildFeatures{
        aidl true
        buildConfig true
    }

    @endcode
    Like this:
    ![](images/module_gradle_fix.png)
    If you don't do this, you may get an error:
    @code
    JavaCameraView.java:15: error: cannot find symbol import org.opencv.BuildConfig; ^ symbol: class BuildConfig location: package org.opencv
    @endcode
    The fix was found [here](https://stackoverflow.com/questions/76374886/error-cannot-find-symbol-import-org-opencv-buildconfig-android-studio) and [here](https://forum.opencv.org/t/task-compiledebugjavawithjavac-failed/13667/4)

7. Add the module to the project:
    - Click ***File -> Project structure... -> Dependencies -> All modules -> + (Add Dependency button) -> Module dependency***

    ![](images/add_module_1.png)

    - Choose `app`

    ![](images/add_module_2.png)

    - Select `OpenCV`

    ![](images/add_module_3.png)

8. Before using any OpenCV function you have to load the library first. If you application includes other OpenCV-dependent native libraries you should load them ***after*** OpenCV initialization.
    Library is loaded at app start:
    @snippet samples/android/tutorial-1-camerapreview/src/org/opencv/samples/tutorial1/Tutorial1Activity.java ocv_loader_init
    Like this:
    ![](images/sample_code.png)

9. Choose a device to check the sample on and run the code by pressing `run` button

    ![](images/run_app.png)
