Using OpenCV pre-built binaries in your own projects {#tutorial_using_prebuilt_binaries}
====================================================

|    |    |
| -: | :- |
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

### Source application

Save the following as `main.cpp`:

@code{.cpp}
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    cv::Mat image = cv::imread("lena.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
    cv::Mat processed_image;
    cv::GaussianBlur(image, processed_image, cv::Size(7, 7), 1.5, 1.5);
    cv::imshow("Original", image);
    cv::imshow("Processed", processed_image);
    cv::waitKey(0);
    return 0;
}
@endcode

Python Project Configuration
----------------------------

If your device already has the pre-compiled Python bindings (`cv2`), you can use them directly.

#### 1. Verify the binding

@code{.bash}
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
@endcode

#### 2. Test script

Save the following as `app.py`:

@code{.python}
import cv2
import sys

img = cv2.imread('lena.jpg')
if img is None:
    sys.exit("Error: image file missing or path invalid.")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Color', img)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
@endcode

Troubleshooting
---------------

-   **CMake: "Could not find a package configuration file ..."** &mdash; CMake cannot locate
    the install. Pass the path explicitly: `cmake -DOpenCV_DIR=/path/to/opencv/build/ ..`
-   **Compiler: "OpenCV 5.0 requires C++17"** &mdash; the toolchain fell back to an older
    standard. Add `-std=c++17` to the g++ command, or `set(CMAKE_CXX_STANDARD 17)` before
    `find_package` in CMake.
