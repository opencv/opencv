Installation in MacOS {#tutorial_macOS_install}
=====================

The following steps have been tested for MacOX (Mavericks) but should work with other versions as well.

Required Packages
-----------------

-   CMake 2.8.7 or higher
-   Git
-   Python 2.6 or later and Numpy 1.5 or later

This tutorial will assume you have [Python](https://docs.python.org/3/using/mac.html),
[Numpy](https://docs.scipy.org/doc/numpy-1.10.1/user/install.html) and
[Git](https://www.atlassian.com/git/tutorials/install-git) installed on your machine.

@note
OSX comes with Python 2.7 by default, you will need to install Python 3.8 if you want to use it specifically.
If you XCode and XCode Command Line-Tools installed, you already have git installed on your machine.

Installing CMake
----------------
-# Find the version for your system and download CMake from their release's [page](https://cmake.org/download/)

-# Install the dmg package and launch it from Applications. That will give you the UI app of CMake

-# From the CMake app window, choose menu Tools --> Install For Command Line Use.

-# Install folder will be /usr/bin/ by default, submit it by choosing Install command line links.

-# Test that it works by running
    @code{.bash}
    cmake --version
    @endcode

Getting OpenCV Source Code
--------------------------

You can use the latest stable OpenCV version or you can grab the latest snapshot from our [Git
repository](https://github.com/opencv/opencv.git).

### Getting the Latest Stable OpenCV Version

-   Go to our [downloads page](http://opencv.org/releases.html).
-   Download the source archive and unpack it.

### Getting the Cutting-edge OpenCV from the Git Repository

Launch Git client and clone [OpenCV repository](http://github.com/opencv/opencv). If you need
modules from [OpenCV contrib repository](http://github.com/opencv/opencv_contrib) then clone it as well.

For example
@code{.bash}
cd ~/<my_working_directory>
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
@endcode
Building OpenCV from Source Using CMake
---------------------------------------

-#  Create a temporary directory, which we denote as \<cmake_build_dir\>, where you want to put
    the generated Makefiles, project files as well the object files and output binaries and enter
    there.

    For example
    @code{.bash}
    cd ~/opencv
    mkdir build
    cd build
    @endcode
-#  Configuring. Run cmake [\<some optional parameters\>] \<path to the OpenCV source directory\>

    For example
    @code{.bash}
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
    @endcode
    or cmake-gui

    -   set full path to OpenCV source code, e.g. /home/user/opencv
    -   set full path to \<cmake_build_dir\>, e.g. /home/user/opencv/build
    -   set optional parameters
    -   run: "Configure"
    -   run: "Generate"

    @note
    Use `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..` , without spaces after -D if the above example doesn't work.

-#  Description of some parameters
    -   build type: `CMAKE_BUILD_TYPE=Release\Debug`
    -   to build with modules from opencv_contrib set OPENCV_EXTRA_MODULES_PATH to \<path to
        opencv_contrib/modules/\>
    -   set BUILD_DOCS for building documents
    -   set BUILD_EXAMPLES to build all examples

-#  [optional] Building python. Set the following python parameters:
    -   PYTHON3_EXECUTABLE = \<path to python\>
    -   PYTHON_INCLUDE_DIR = /usr/include/python\<version\>
    -   PYTHON3_NUMPY_INCLUDE_DIRS =
        /usr/lib/python\<version\>/dist-packages/numpy/core/include/
    @note
    To specify Python2 versions, you can replace PYTHON3_ with PYTHON2_ in the above parameters.

-#  Build. From build directory execute *make*, it is recommended to do this in several threads

    For example
    @code{.bash}
    make -j7 # runs 7 jobs in parallel
    @endcode

@note
You can also use a package manager like [Homebrew](https://brew.sh/)
or [pip](https://pip.pypa.io/en/stable/) to install releases of OpenCV only (Not the cutting edge).
