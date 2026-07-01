Using OpenCV pre-built binaries in your own projects {#tutorial_using_prebuilt_binaries}
====================================================

|    |    |
| -: | :- |
| Original authors | Abhishek Gola & Kirti Jindal |
| Compatibility | OpenCV >= 5.0, C++17, Python >= 3.6 |

@tableofcontents

Goal
----

The objective of this tutorial is to show how to configure a local build environment to use
pre-built OpenCV 5.0 binaries that are already present on your device.

By the end of this guide you will know how to reference, link, and use an existing OpenCV
installation from your own C++ or Python application, without rebuilding the library from
source.

Detailed Description
--------------------

When OpenCV is installed via an installer, a system package manager (apt, Homebrew, vcpkg),
or built into a local workspace directory, it exports configuration scripts that let build
tools locate and link it automatically.

Because OpenCV 5.0 modernizes its build requirements, keep two things in mind:

-   **C++17 is required.** The library headers use modern language features, so your project
    must be compiled with the C++17 standard or higher.
-   **Modern CMake targets.** The legacy 1.x C API has been removed. Link via the
    `${OpenCV_LIBS}` variable from `find_package(OpenCV)`, or by naming the specific
    libraries on the compiler command line.

C++ Project Configuration
-------------------------

You can link against your pre-built binaries with CMake (recommended for cross-platform
stability) or by invoking g++ directly.

### Method 1: Using CMake (recommended)

#### 1. File layout

@code{.unparsed}
my_opencv_project/
├── CMakeLists.txt
└── main.cpp
@endcode

#### 2. CMakeLists.txt

@code{.cmake}
cmake_minimum_required(VERSION 3.22)
project(OpenCV5_Local_Project)

# OpenCV 5.0 requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the pre-built OpenCV installation. If CMake cannot find it
# automatically, pass the path explicitly:
#   cmake -DOpenCV_DIR=/absolute/path/to/opencv/build/ ..
find_package(OpenCV 5.0 REQUIRED)

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")

add_executable(opencv_test_app main.cpp)
target_link_libraries(opencv_test_app PRIVATE ${OpenCV_LIBS})
@endcode

#### 3. Build and run

@code{.bash}
cd path/to/my_opencv_project
mkdir build && cd build
cmake ..
cmake --build .
./opencv_test_app
@endcode

### Method 2: Direct compilation with g++

On Linux or macOS you can compile with a single command, without generating build files.

#### 1. Locate your paths

-   Standard global location: `/usr/local/include/opencv5` and `/usr/local/lib`
-   Custom build location: `/path/to/opencv/build/include` and `/path/to/opencv/build/lib`

#### 2. Using pkg-config

@code{.bash}
g++ -std=c++17 main.cpp -o opencv_test_app $(pkg-config --cflags --libs opencv5)
@endcode

#### 3. Explicit paths (custom or local builds)

@code{.bash}
g++ -std=c++17 main.cpp -o opencv_test_app \
    -I/usr/local/include/opencv5 \
    -L/usr/local/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
@endcode

@note In OpenCV 5.0 the legacy `calib3d` module was split into four modules: `geometry`,
`calib`, `stereo`, and `ptcloud`. Link the specific one(s) you need, e.g. `-lopencv_geometry`,
`-lopencv_calib`, `-lopencv_stereo`, or `-lopencv_ptcloud`, instead of `-lopencv_calib3d`.

Python Project Configuration
----------------------------

If your device already has the pre-compiled Python bindings (`cv2`), you can use them directly.

#### 1. Verify the binding

If you are using a local or custom build, ensure your system can locate the generated `cv2`
binary by setting the `PYTHONPATH` environment variable:

@code{.bash}
export PYTHONPATH=/path/to/opencv/build/lib:$PYTHONPATH
@endcode

@note Replace `/path/to/opencv/build/lib` with the actual path to the directory containing your
compiled `cv2` module (usually inside your local `build/lib` or `build/lib/python3` folder).

Once the path is set, run the following command to verify that the module imports cleanly and
prints the correct version:

@code{.bash}
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
@endcode

#### 2. Test script

Write a short `app.py` that imports `cv2`, loads an image, and displays it to confirm the bindings work.

Troubleshooting
---------------

-   **CMake: "Could not find a package configuration file ..."** &mdash; CMake cannot locate
    the install. Pass the path explicitly: `cmake -DOpenCV_DIR=/path/to/opencv/build/ ..`
-   **Compiler: "OpenCV 5.0 requires C++17"** &mdash; the toolchain fell back to an older
    standard. Add `-std=c++17` to the g++ command, or `set(CMAKE_CXX_STANDARD 17)` before
    `find_package` in CMake.
