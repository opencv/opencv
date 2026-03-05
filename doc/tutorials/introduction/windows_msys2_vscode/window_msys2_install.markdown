Building OpenCV on Windows from Source Code using MSYS2 UCRT64 and VS Code (C++) {#tutorial_windows_msys2_vscode}
=====================================================================

@prev_tutorial{tutorial_windows_install}
@next_tutorial{tutorial_windows_visual_studio_image_watch}

| | |
| -: | :- |
| Original author | Mahadev Kumar |
| Compatibility | OpenCV >= 4.x |

@tableofcontents

@note
This tutorial was tested on **Windows 10/11** using the **MSYS2 UCRT64 environment**.

---

## Introduction {#tutorial_windows_msys2_install_intro}

This tutorial explains how to build OpenCV from source on Windows using the **MSYS2 UCRT64 toolchain** and use it inside **Visual Studio Code with C++**.

The build configuration uses:

- **MSYS2 (UCRT64 shell)**
- **GCC (UCRT toolchain)**
- **CMake**
- **mingw32-make**
- **VS Code**

This method produces **native Windows binaries linked against the Universal C Runtime (UCRT)**.

---

## Prerequisites {#tutorial_windows_install_msys2_prerequisites}

Install the following software before proceeding:

- [MSYS2](https://www.msys2.org/)
- [Git](https://git-scm.com/install/)
- [Python](https://www.python.org/)
- [Visual Studio Code](https://code.visualstudio.com/)

After installing MSYS2, always open the:

**MSYS2 UCRT64 Terminal**

@note
Do not use the **MSYS**, **MinGW64**, or **CLANG64** shells for this build.

---

## Step 1: Update MSYS2 {#tutorial_windows_msys2_install_update}

Open the **MSYS2 UCRT64** terminal and update the system.

@code{.bash}
pacman -Syu
@endcode

Restart the terminal if prompted.

---

## Step 2: Install Required Packages {#tutorial_windows_msys2_install_packages}

Install the required compiler and build tools.

@code{.bash}
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-make \
          mingw-w64-ucrt-x86_64-python
@endcode

Verify installation:

@code{.bash}
gcc --version
cmake --version
mingw32-make --version
@endcode

Add the following directory to your **Windows PATH**:
@code{.bash}
C:\msys64\ucrt64\bin
@endcode

---

## Step 3: Clone OpenCV Source Code {#tutorial_windows_msys2_install_clone}

Clone OpenCV and optional contrib modules.

@code{.bash}
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
@endcode

---

## Step 4: Create Build Directory {#tutorial_windows_msys2_install_build_dir}

@code{.bash}
cd opencv
mkdir build
cd build
@endcode

---

## Step 5: Configure the Build with CMake {#tutorial_windows_msys2_install_configure}

Run CMake using the **MinGW Makefiles generator**.

@code{.bash}
cmake -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=install \
  -DBUILD_opencv_python3=ON \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  ..
@endcode

@note
If you do not want to use **opencv_contrib**, remove the `OPENCV_EXTRA_MODULES_PATH` option.

---

## Step 6: Build OpenCV {#tutorial_windows_msys2_install_build}

Compile OpenCV.

@code{.bash}
mingw32-make -j6
@endcode

@note
If the build fails due to memory limitations, reduce the job count.

Example:

@code{.bash}
mingw32-make -j4
@endcode

---

## Step 7: Install OpenCV {#tutorial_windows_msys2_install_install}

Install the compiled libraries.

@code{.bash}
mingw32-make install
@endcode

After installation, OpenCV will be located in:
`opencv/build/install`

Add the following directory to your **Windows PATH**:
@code{.bash}
C:\path\to\opencv\build\install\x64/mingw/bin
@endcode

---

## Step 8: Verify with a C++ Example {#tutorial_windows_msys2_install_verify}

Create a folder outside the OpenCV source directory.

Example: `first-project`

Inside the folder create **main.cpp**

@code{.cpp}
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3);

    cv::imshow("OpenCV Test", img);
    cv::waitKey(0);

    return 0;
}
@endcode

Create **CMakeLists.txt**

@code{.cmake}
cmake_minimum_required(VERSION 3.10)

project(OpenCVApp)

set(OpenCV_DIR "C:/path/to/opencv/build/install/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

add_executable(app main.cpp)

target_link_libraries(app ${OpenCV_LIBS})
@endcode

Build the project.

@code{.bash}
mkdir build
cd build

cmake -G "MinGW Makefiles" ..
mingw32-make
@endcode

Run the program.

@code{.bash}
./app.exe
@endcode

If everything is configured correctly, a window should appear displaying a blank image.

---

## Using OpenCV in Visual Studio Code {#tutorial_windows_msys2_vscode_section}

Install the following extensions in **VS Code**:

- C/C++
- CMake Tools

Open your project folder in VS Code.

Configure the project using CMake.

@code{.bash}
cmake -G "MinGW Makefiles" ..
@endcode

Build the project.

@code{.bash}
mingw32-make
@endcode

Run the executable.

---

## Troubleshooting {#tutorial_windows_msys2_install_troubleshoot}

### CMake cannot find OpenCV

Ensure that `OpenCV_DIR` is set correctly.

Example:

@code{.cmake}
set(OpenCV_DIR "C:/path/to/opencv/build/install/lib/cmake/opencv4")
@endcode

---

### mingw32-make not found

Make sure the following path is added to the Windows environment variable **PATH**: `C:\msys64\ucrt64\bin`

---

### Build fails due to memory limits

Reduce parallel build jobs.

@code{.bash}
mingw32-make -j2
@endcode

---

## Conclusion

You have successfully built OpenCV from source using **MSYS2 UCRT64** and verified it with a **C++ project in VS Code**.

This setup allows you to develop OpenCV-based C++ applications using a fully open-source toolchain on Windows.
