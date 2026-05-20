# Introduction to OpenCV

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [OpenCV installation overview](general_install.md)
  - Overview of prebuilt and source-build installation paths across supported platforms.
* - [OpenCV configuration options reference](config_reference.md)
  - Reference for every CMake build flag OpenCV exposes and how to set them.
* - [OpenCV environment variables reference](env_reference.md)
  - Runtime environment variables for debug output, search paths, and algorithm tuning.
```

## Linux

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Installation in Linux](linux_install.md)
  - Step-by-step source build of OpenCV on Linux with CMake and the system compiler.
* - [Building OpenCV with oneAPI](oneapi_install.md)
  - Build instructions for OpenCV linked against Intel oneAPI — DPC++, oneTBB, oneMKL, oneDNN, and IPP.
* - [Using OpenCV with gdb-powered IDEs](linux_gdb_pretty_printer.md)
  - Python pretty-printer install for showing `cv::Mat` element type, flags, and contents in gdb-powered IDEs.
* - [Using OpenCV with gcc and CMake](linux_gcc_cmake.md)
  - Minimal CMake recipe and source for a first OpenCV application on Linux with gcc.
* - [Using OpenCV with Eclipse (plugin CDT)](linux_eclipse.md)
  - Eclipse CDT project setup for OpenCV applications, both directly and through CMake.
```

## Windows

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Installation in Windows](windows_install.md)
  - Prebuilt-library and source-build install paths for OpenCV on Windows.
* - [How to build applications with OpenCV inside the "Microsoft Visual Studio"](windows_visual_studio_opencv.md)
  - Visual Studio project setup for compiling and linking C/C++ applications against OpenCV.
* - [Image Watch: viewing in-memory images in the Visual Studio debugger](windows_visual_studio_image_watch.md)
  - Visual Studio extension for inspecting `cv::Mat` and `IplImage_` images live during a debug session.
* - [Building OpenCV on Windows from Source Code using MSYS2 UCRT64 and VS Code (C++)](windows_msys2_vscode.md)
  - MSYS2 UCRT64 source build with GCC, CMake, and mingw32-make, integrated with Visual Studio Code.
```

## Java & Android

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Introduction to Java Development](java_dev_intro.md)
  - Desktop Java and Scala project setup for OpenCV using Apache Ant or SBT.
* - [Using OpenCV Java with Eclipse](java_eclipse.md)
  - Eclipse user-library configuration for OpenCV Java development on Windows.
* - [Introduction to OpenCV Development with Clojure](clojure_dev_intro.md)
  - Clojure REPL setup for interactive OpenCV exploration on the JVM.
* - [Introduction into Android Development](android_dev_intro.md)
  - Android toolchain setup (Android Studio, JDK, SDK, NDK) and Android development primer for OpenCV work.
* - [Android Development with OpenCV](dev_with_OCV_on_Android.md)
  - Adding the OpenCV library to an Android Studio project and calling it from Java, experimental Kotlin, or via JNI.
* - [How to run deep networks on Android device](android_dnn_intro.md)
  - Pretrained deep-learning model inference on Android via OpenCV's `dnn` module, with a MobileNet object-detection sample.
* - [Use OpenCL in Android camera preview based CV application](android_ocl_intro.md)
  - OpenCL acceleration for an Android camera-preview computer-vision application through OpenCV.
```

## Other platforms

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Installation in MacOS](macos_install.md)
  - Required-package list (CMake, Git, Python, NumPy) and source-build instructions for OpenCV on macOS.
* - [Cross compilation for ARM based Linux systems](arm_crosscompile_with_cmake.md)
  - Cross-compilation of OpenCV for ARM Linux (`gnueabi`/`gnueabihf`) from an Ubuntu host using CMake.
* - [MultiArch cross-compilation with Ubuntu/Debian](crosscompile_with_multiarch.md)
  - Debian/Ubuntu MultiArch setup for resolving foreign-architecture third-party dependencies during cross-compilation.
* - [Building OpenCV for Tegra with CUDA](building_tegra_cuda.md)
  - CUDA-enabled build of OpenCV for Tegra platforms — DRIVE PX 2, Tegra L4T, and desktop Linux variants.
* - [Building OpenCV with FastCV](building_fastcv.md)
  - FastCV-based HAL and contrib-module enablement for accelerated OpenCV on Qualcomm Snapdragon (ARM64) chipsets.
* - [Installation in iOS](https://docs.opencv.org/5.x/d5/da3/tutorial_ios_install.html)
  - Xcode and CMake setup, plus source build of OpenCV for iOS from the cutting-edge git repository.
```

## Usage basics

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Getting Started with Images](display_image.md)
  - Image read with `cv::imread`, display with `cv::imshow`, and write-back with `cv::imwrite`.
```

## Miscellaneous

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Writing documentation for OpenCV](documentation.md)
  - Doxygen-based authoring guide for OpenCV docs — features, install, syntax, and conventions.
* - [Transition guide](transition_guide.md)
  - Migration notes for software developers moving C++ code to OpenCV 5.0.
* - [Cross referencing OpenCV from other Doxygen projects](cross_referencing.md)
  - Tagfile-based cross-references from your own Doxygen project to OpenCV's API documentation.
```

```{toctree}
:hidden:
:maxdepth: 1

general_install
config_reference
env_reference
linux_install
oneapi_install
linux_gdb_pretty_printer
linux_gcc_cmake
linux_eclipse
windows_install
windows_visual_studio_opencv
windows_visual_studio_image_watch
windows_msys2_vscode
java_dev_intro
java_eclipse
clojure_dev_intro
android_dev_intro
dev_with_OCV_on_Android
android_dnn_intro
android_ocl_intro
macos_install
arm_crosscompile_with_cmake
crosscompile_with_multiarch
building_tegra_cuda
building_fastcv
display_image
documentation
transition_guide
cross_referencing
```
