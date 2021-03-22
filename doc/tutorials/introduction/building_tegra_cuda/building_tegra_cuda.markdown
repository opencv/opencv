Building OpenCV for Tegra with CUDA {#tutorial_building_tegra_cuda}
===================================

@prev_tutorial{tutorial_arm_crosscompile_with_cmake}
@next_tutorial{tutorial_display_image}

|    |    |
| -: | :- |
| Original author |  Randy J. Ray |
| Compatibility | OpenCV >= 3.1.0 |

@warning
This tutorial is deprecated.

@tableofcontents

OpenCV with CUDA for Tegra
==========================

This document is a basic guide to building the OpenCV libraries with CUDA support for use in the Tegra environment. It covers the basic elements of building the version 3.1.0 libraries from source code for three (3) different types of platforms:

* NVIDIA DRIVE&trade; PX 2 (V4L)
* NVIDIA<sup>&reg;</sup> Tegra<sup>&reg;</sup> Linux Driver Package (L4T)
* Desktop Linux (Ubuntu 14.04 LTS and 16.04 LTS)

This document is not an exhaustive guide to all of the options available when building OpenCV. Specifically, it covers the basic options used when building each platform but does not cover any options that are not needed (or are unchanged from their default values). Additionally, the installation of the CUDA toolkit is not covered here.

This document is focused on building the 3.1.0 version of OpenCV, but the guidelines here may also work for building from the master branch of the git repository. There are differences in some of the CMake options for builds of the 2.4.13 version of OpenCV, which are summarized below in the @ref tutorial_building_tegra_cuda_opencv_24X section.

Most of the configuration commands are based on the system having CUDA 8.0 installed. In the case of the Jetson TK1, an older CUDA is used because 8.0 is not supported for that platform. These instructions may also work with older versions of CUDA, but are only tested with 8.0.

### A Note on Native Compilation vs. Cross-Compilation

The OpenCV build system supports native compilation for all the supported platforms, as well as cross-compilation for platforms such as ARM and others. The native compilation process is simpler, whereas the cross-compilation is generally faster.

At the present time, this document focuses only on native compilation.

Getting the Source Code {#tutorial_building_tegra_cuda_getting_the_code}
=======================

There are two (2) ways to get the OpenCV source code:

* Direct download from the [OpenCV downloads](http://opencv.org/releases.html) page
* Cloning the git repositories hosted on [GitHub](https://github.com/opencv)

For this guide, the focus is on using the git repositories. This is because the 3.1.0 version of OpenCV will not build with CUDA 8.0 without applying a few small upstream changes from the git repository.

OpenCV
------

Start with the `opencv` repository:

    # Clone the opencv repository locally:
    $ git clone https://github.com/opencv/opencv.git

To build the 3.1.0 version (as opposed to building the most-recent source), you must check out a branch based on the `3.1.0` tag:

    $ cd opencv
    $ git checkout -b v3.1.0 3.1.0

__Note:__ This operation creates a new local branch in your clone's repository.

There are some upstream changes that must be applied via the `git cherry-pick` command. The first of these is to apply a fix for building specifically with the 8.0 version of CUDA that was not part of the 3.1.0 release:

    # While still in the opencv directory:
    $ git cherry-pick 10896

You will see the following output from the command:

    [v3.1.0 d6d69a7] GraphCut deprecated in CUDA 7.5 and removed in 8.0
     Author: Vladislav Vinogradov <vlad.vinogradov@itseez.com>
     1 file changed, 2 insertions(+), 1 deletion(-)

Secondly, there is a fix for a CMake macro call that is problematic on some systems:

    $ git cherry pick cdb9c

You should see output similar to:

    [v3.1.0-28613 e5ac2e4] gpu samples: fix REMOVE_ITEM error
     Author: Alexander Alekhin <alexander.alekhin@itseez.com>
     1 file changed, 1 insertion(+), 1 deletion(-)

The last upstream fix that is needed deals with the `pkg-config` configuration file that is bundled with the developer package (`libopencv-dev`):

    $ git cherry-pick 24dbb

You should see output similar to:

    [v3.1.0 3a6d7ab] pkg-config: modules list contains only OpenCV modules (fixes #5852)
     Author: Alexander Alekhin <alexander.alekhin@itseez.com>
     1 file changed, 7 insertions(+), 4 deletions(-)

At this point, the `opencv` repository is ready for building.

OpenCV Extra
------------

The `opencv_extra` repository contains extra data for the OpenCV library, including the data files used by the tests and demos. It must be cloned separately:

    # In the same base directory from which you cloned OpenCV:
    $ git clone https://github.com/opencv/opencv_extra.git

As with the OpenCV source, you must use the same method as above to set the source tree to the 3.1.0 version. When you are building from a specific tag, both repositories must be checked out at that tag.

    $ cd opencv_extra
    $ git checkout -b v3.1.0 3.1.0

You may opt to not fetch this repository if you do not plan on running the tests or installing the test-data along with the samples and example programs. If it is not referenced in the invocation of CMake, it will not be used.

__Note:__ If you plan to run the tests, some tests expect the data to be present and will fail without it.

Preparation and Prerequisites {#tutorial_building_tegra_cuda_preparation}
=============================

To build OpenCV, you need a directory to create the configuration and build the libraries. You also need a number of 3rd-party libraries upon which OpenCV depends.

Prerequisites for Ubuntu Linux
------------------------------

These are the basic requirements for building OpenCV for Tegra on Linux:

* CMake 2.8.10 or newer
* CUDA toolkit 8.0 (7.0 or 7.5 may also be used)
* Build tools (make, gcc, g++)
* Python 2.6 or greater

These are the same regardless of the platform (DRIVE PX 2, Desktop, etc.).

A number of development packages are required for building on Linux:

* libglew-dev
* libtiff5-dev
* zlib1g-dev
* libjpeg-dev
* libpng12-dev
* libjasper-dev
* libavcodec-dev
* libavformat-dev
* libavutil-dev
* libpostproc-dev
* libswscale-dev
* libeigen3-dev
* libtbb-dev
* libgtk2.0-dev
* pkg-config

Some of the packages above are in the `universe` repository for Ubuntu Linux systems. If you have not already enabled that repository, you need to do the following before trying to install all of the packages listed above:

    $ sudo apt-add-repository universe
    $ sudo apt-get update

The following command can be pasted into a shell in order to install the required packages:

    $ sudo apt-get install \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libpng12-dev \
        libjasper-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config

(Line-breaks and continuation characters are added for readability.)

If you want the Python bindings to be built, you will also need the appropriate packages for either or both of Python 2 and Python 3:

* python-dev / python3-dev
* python-numpy / python3-numpy
* python-py / python3-py
* python-pytest / python3-pytest

The commands that will do this:

    $ sudo apt-get install python-dev python-numpy python-py python-pytest
    # And, optionally:
    $ sudo apt-get install python3-dev python3-numpy python3-py python3-pytest

Once all the necessary packages are installed, you can configure the build.

Preparing the Build Area
------------------------

Software projects that use the CMake system for configuring their builds expect the actual builds to be done outside of the source tree itself. For configuring and building OpenCV, create a directory called "build" in the same base directory into which you cloned the git repositories:

    $ mkdir build
    $ cd build

You are now ready to configure and build OpenCV.

Configuring OpenCV for Building {#tutorial_building_tegra_cuda_configuring}
===============================

The CMake configuration options given below for the different platforms are targeted towards the functionality needed for Tegra. They are based on the original configuration options used for building OpenCV 2.4.13.

The build of OpenCV is configured with CMake. If run with no parameters, it detects what it needs to know about your system. However, it may have difficulty finding the CUDA files if they are not in a standard location, and it may try to build some options that you might otherwise not want included, so the following invocations of CMake are recommended.

In each `cmake` command listed in the following sub-sections, line-breaks and indentation are added for readability. Continuation characters are also added in examples for Linux-based platforms, allowing you to copy and paste the examples directly into a shell. When entering these commands by hand, enter the command and options as a single line. For a detailed explanation of the parameters passed to `cmake`, see the "CMake Parameter Reference" section.

For the Linux-based platforms, the shown value for the `CMAKE_INSTALL_PREFIX` parameter is `/usr`. You can set this to whatever you want, based on the layout of your system.

In each of the `cmake` invocations below, the last parameter, `OPENCV_TEST_DATA_PATH`, tells the build system where to find the test-data that is provided by the `opencv_extra` repository. When this is included, a `make install` installs this test-data alongside the libraries and example code, and a `make test` automatically provides this path to the tests that have to load data from it. If you did not clone the `opencv_extra` repository, do not include this parameter.

Vibrante V4L Configuration
--------------------------

Supported platform: Drive PX 2

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=ON \
        -DBUILD_opencv_python3=OFF \
        -DENABLE_NEON=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
        -DCUDA_ARCH_BIN=6.2 \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

The configuration provided above builds the Python bindings for Python 2 (but not Python 3) as part of the build process. If you want the Python 3 bindings (or do not want the Python 2 bindings), change the values of `BUILD_opencv_python2` and/or `BUILD_opencv_python3` as needed. To enable bindings, set the value to `ON`, to disable them set it to `OFF`:

    -DBUILD_opencv_python2=OFF

Jetson L4T Configuration
------------------------

Supported platforms:

* Jetson TK1
* Jetson TX1

Configuration is slightly different for the Jetson TK1 and the Jetson TX1 systems.

### Jetson TK1

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_CXX_FLAGS=-Wa,-mimplicit-it=thumb \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=ON \
        -DBUILD_opencv_python3=OFF \
        -DENABLE_NEON=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-6.5 \
        -DCUDA_ARCH_BIN=3.2 \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

__Note:__ This uses CUDA 6.5, not 8.0.

### Jetson TX1

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=ON \
        -DBUILD_opencv_python3=OFF \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
        -DCUDA_ARCH_BIN=5.3 \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

__Note:__ This configuration does not set the `ENABLE_NEON` parameter.

Ubuntu Desktop Linux Configuration
----------------------------------

Supported platforms:

* Ubuntu Desktop Linux 14.04 LTS
* Ubuntu Desktop Linux 16.04 LTS

The configuration options given to `cmake` below are targeted towards the functionality needed for Tegra. For a desktop system, you may wish to adjust some options to enable (or disable) certain features. The features enabled below are based on the building of OpenCV 2.4.13.

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=ON \
        -DBUILD_opencv_python3=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
        -DCUDA_ARCH_BIN='3.0 3.5 5.0 6.0 6.2' \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

This configuration is nearly identical to that for V4L and L4T, except that the `CUDA_ARCH_BIN` parameter specifies multiple architectures so as to support a variety of GPU boards. For a desktop, you have the option of omitting this parameter, and CMake will instead run a small test program that probes for the supported architectures. However, the libraries produced might not work on Ubuntu systems with different cards.

As with previous examples, the configuration given above builds the Python bindings for Python 2 (but not Python 3) as part of the build process.

Building OpenCV {#tutorial_building_tegra_cuda_building}
===============

Once `cmake` finishes configuring OpenCV, building is done using the standard `make` utility.

Building with `make`
--------------------

The only parameter that is needed for the invocation of `make` is the `-j` parameter for specifying how many parallel threads to use. This varies depending on the system and how much memory is available, other running processes, etc. The following table offers suggested values for this parameter:

|Platform|Suggested value|Notes|
|--------|---------------|-----|
|DRIVE PX 2|6| |
|Jetson TK1|3|If the build fails due to a compiler-related error, try again with a smaller number of threads. Also consider rebooting the system if it has been running for a long time since the last reboot.|
|Jetson TX1|4| |
|Ubuntu Desktop|7|The actual value will vary with the number of cores you have and the amount of physical memory. Because of the resource requirements of compiling the CUDA code, it is not recommended to go above 7.|

Based on the value you select, build (assuming you selected 6):

    $ make -j6

By default, CMake hides the details of the build steps. If you need to see more detail about each compilation unit, etc., you can enable verbose output:

    $ make -j6 VERBOSE=1

Testing OpenCV {#tutorial_building_tegra_cuda_testing}
==============

Once the build completes successfully, you have the option of running the extensive set of tests that OpenCV provides. If you did not clone the `opencv_extra` repository and specify the path to `testdata` in the `cmake` invocation, then testing is not recommended.

Testing under Linux
-------------------

To run the basic tests under Linux, execute:

    $ make test

This executes `ctest` to carry out the tests, as specified in CTest syntax within the OpenCV repository. The `ctest` harness takes many different parameters (too many to list here, see the manual page for CTest to see the full set), and if you wish to pass any of them, you can do so by specifying them in a `make` command-line parameter called `ARGS`:

    $ make test ARGS="--verbose --parallel 3"

In this example, there are two (2) arguments passed to `ctest`: `--verbose` and `--parallel 3`. The first argument causes the output from `ctest` to be more detailed, and the second causes `ctest` to run as many as three (3) tests in parallel. As with choosing a thread count for building, base any choice for testing on the available number of processor cores, physical memory, etc. Some of the tests do attempt to allocate significant amounts of memory.

### Known Issues with Tests

At present, not all of the tests in the OpenCV test suite pass. There are tests that fail whether or not CUDA is compiled, and there are tests that are only specific to CUDA that also do not currently pass.

__Note:__ There are no tests that pass without CUDA but fail only when CUDA is included.

As the full lists of failing tests vary based on platform, it is impractical to list them here.

Installing OpenCV {#tutorial_building_tegra_cuda_installing}
=================

Installing OpenCV is very straightforward. For the Linux-based platforms, the command is:

    $ make install

Depending on the chosen installation location, you may need root privilege to install.

Building OpenCV 2.4.X {#tutorial_building_tegra_cuda_opencv_24X}
=====================

If you wish to build your own version of the 2.4 version of OpenCV, there are only a few adjustments that must be made. At the time of this writing, the latest version on the 2.4 tree is 2.4.13. These instructions may work for later versions of 2.4, though they have not been tested for any earlier versions.

__Note:__ The 2.4.X OpenCV source does not have the extra modules and code for Tegra that was upstreamed into the 3.X versions of OpenCV. This part of the guide is only for cases where you want to build a vanilla version of OpenCV 2.4.

Selecting the 2.4 Source
------------------------

First you must select the correct source branch or tag. If you want a specific version such as 2.4.13, you want to make a local branch based on the tag, as was done with the 3.1.0 tag above:

    # Within the opencv directory:
    $ git checkout -b v2.4.13 2.4.13

    # Within the opencv_extra directory:
    $ git checkout -b v2.4.13 2.4.13

If you simply want the newest code from the 2.4 line of OpenCV, there is a `2.4` branch already in the repository. You can check that out instead of a specific tag:

    $ git checkout 2.4

There is no need for the `git cherry-pick` commands used with 3.1.0 when building the 2.4.13 source.

Configuring
-----------

Configuring is done with CMake as before. The primary difference is that OpenCV 2.4 only provides Python bindings for Python 2, and thus does not distinguish between Python 2 and Python 3 in the CMake parameters. There is only one parameter, `BUILD_opencv_python`. In addition, there is a build-related parameter that controls features in 2.4 that are not in 3.1.0. This parameter is `BUILD_opencv_nonfree`.

Configuration still takes place in a separate directory that must be a sibling to the `opencv` and `opencv_extra` directories.

### Configuring Vibrante V4L

For DRIVE PX 2:

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_nonfree=OFF \
        -DBUILD_opencv_python=ON \
        -DENABLE_NEON=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
        -DCUDA_ARCH_BIN=6.2 \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=ON \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

### Configuring Jetson L4T

For Jetson TK1:

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_nonfree=OFF \
        -DBUILD_opencv_python=ON \
        -DENABLE_NEON=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-6.5 \
        -DCUDA_ARCH_BIN=3.2 \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=ON \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

For Jetson TX1:

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_nonfree=OFF \
        -DBUILD_opencv_python=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
        -DCUDA_ARCH_BIN=5.3 \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=ON \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

### Configuring Desktop Ubuntu Linux

For both 14.04 LTS and 16.04 LTS:

    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_nonfree=OFF \
        -DBUILD_opencv_python=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
        -DCUDA_ARCH_BIN='3.0 3.5 5.0 6.0 6.2' \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=ON \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
        ../opencv

Building, Testing and Installing
--------------------------------

Once configured, the steps of building, testing, and installing are the same as above for the 3.1.0 source.

CMake Parameter Reference {#tutorial_building_tegra_cuda_parameter_reference}
=========================

The following is a table of all the parameters passed to CMake in the recommended invocations above. Some of these are parameters from CMake itself, while most are specific to OpenCV.

|Parameter|Our Default Value|What It Does|Notes|
|---------|-----------------|------------|-----|
|BUILD_EXAMPLES|ON|Governs whether the C/C++ examples are built| |
|BUILD_JASPER|OFF|Governs whether the Jasper library (`libjasper`) is built from source in the `3rdparty` directory| |
|BUILD_JPEG|OFF|As above, for `libjpeg`| |
|BUILD_PNG|OFF|As above, for `libpng`| |
|BUILD_TBB|OFF|As above, for `tbb`| |
|BUILD_TIFF|OFF|As above, for `libtiff`| |
|BUILD_ZLIB|OFF|As above, for `zlib`| |
|BUILD_JAVA|OFF|Controls the building of the Java bindings for OpenCV|Building the Java bindings requires OpenCV libraries be built for static linking only|
|BUILD_opencv_nonfree|OFF|Controls the building of non-free (non-open-source) elements|Used only for building 2.4.X|
|BUILD_opencv_python|ON|Controls the building of the Python 2 bindings in OpenCV 2.4.X|Used only for building 2.4.X|
|BUILD_opencv_python2|ON|Controls the building of the Python 2 bindings in OpenCV 3.1.0|Not used in 2.4.X|
|BUILD_opencv_python3|OFF|Controls the building of the Python 3 bindings in OpenCV 3.1.0|Not used in 2.4.X|
|CMAKE_BUILD_TYPE|Release|Selects the type of build (release vs. development)|Is generally either `Release` or `Debug`|
|CMAKE_INSTALL_PREFIX|/usr|Sets the root for installation of the libraries and header files| |
|CUDA_ARCH_BIN|varies|Sets the CUDA architecture(s) for which code is compiled|Usually only passed for platforms with known specific cards. OpenCV includes a small program that determines the architectures of the system's installed card if you do not pass this parameter. Here, for Ubuntu desktop, the value is a list to maximize card support.|
|CUDA_ARCH_PTX|""|Builds PTX intermediate code for the specified virtual PTX architectures| |
|CUDA_TOOLKIT_ROOT_DIR|/usr/local/cuda-8.0 (for Linux)|Specifies the location of the CUDA include files and libraries| |
|ENABLE_NEON|ON|Enables the use of NEON SIMD extensions for ARM chips|Only passed for builds on ARM platforms|
|ENABLE_PRECOMPILED_HEADERS|OFF|Enables/disables support for pre-compiled headers|Only specified on some of the ARM platforms|
|INSTALL_C_EXAMPLES|ON|Enables the installation of the C example files as part of `make install`| |
|INSTALL_TESTS|ON|Enables the installation of the tests as part of `make install`| |
|OPENCV_TEST_DATA_PATH|../opencv_extra/testdata|Path to the `testdata` directory in the `opencv_extra` repository| |
|WITH_1394|OFF|Specifies whether to include IEEE-1394 support| |
|WITH_CUDA|ON|Specifies whether to include CUDA support| |
|WITH_FFMPEG|ON|Specifies whether to include FFMPEG support| |
|WITH_GSTREAMER|OFF|Specifies whether to include GStreamer 1.0 support| |
|WITH_GSTREAMER_0_10|OFF|Specifies whether to include GStreamer 0.10 support| |
|WITH_GTK|ON|Specifies whether to include GTK 2.0 support|Only given on Linux platforms, not Microsoft Windows|
|WITH_OPENCL|OFF|Specifies whether to include OpenCL runtime support| |
|WITH_OPENEXR|OFF|Specifies whether to include ILM support via OpenEXR| |
|WITH_OPENMP|OFF|Specifies whether to include OpenMP runtime support| |
|WITH_TBB|ON|Specifies whether to include Intel TBB support| |
|WITH_VTK|OFF|Specifies whether to include VTK support| |

Copyright &copy; 2016, NVIDIA CORPORATION. All rights reserved.
