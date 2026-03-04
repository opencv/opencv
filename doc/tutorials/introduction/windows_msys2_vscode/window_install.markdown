Building OpenCV on Windows using MSYS2 UCRT64 {#tutorial_windows_msys2_vscode}
==================

@prev_tutorial{tutorial_linux_eclipse}
@next_tutorial{tutorial_windows_visual_studio_opencv}

|    |    |
| -: | :- |
| Original author |  |
| Compatibility | OpenCV >= 4.0 |

@tableofcontents

@warning
This tutorial can contain obsolete information.

The description here was tested on Windows 7 SP1. Nevertheless, it should also work on any other
relatively modern version of Windows OS. If you encounter errors after following the steps described
below, feel free to contact us via our [OpenCV Q&A forum](https://forum.opencv.org). We'll do our
best to help you out.

## Introduction {#tutorial_windows_install_intro}

# This tutorial describes how to build OpenCV from source on Windows using the MSYS2 UCRT64 environment.

### The build configuration uses:

- MSYS2 (UCRT64 shell)
- GCC (UCRT toolchain)
- CMake
- mingw32-make

This method produces native Windows binaries linked against the Universal C Runtime (UCRT).

---
## Prerequisites {#tutorial_windows_install_prerequisites}

### Install the following software before proceeding:

- MSYS2
- Git
- Python

### After installing MSYS2, always open:

    MSYS2 UCRT64
[MSYS2](https://www.msys2.org/)

@note  Do not use the MSYS, MinGW64, or CLANG64 shells for this build.

---

## Step 1: Update MSYS2 {#tutorial_windows_install_update}

### Open the MSYS2 UCRT64 shell and update the system:

    @code{.bash}
    pacman -Syu
    @endcode

---

## Step 2: Install Required Packages {#tutorial_windows_install_packages}

### Inside the UCRT64 shell, install required packages:
    @code{.bash}
    pacman -S mingw-w64-ucrt-x86_64-gcc
    pacman -S mingw-w64-ucrt-x86_64-cmake
    pacman -S mingw-w64-ucrt-x86_64-make
    @endcode

### Verify installation in UCRT64 SHELL:

    `gcc --version` or `g++ --version`
    # gcc.exe (Revxx, Built by MSYS2 porject) version

    `cmake --version`
    # cmake version 4.x

    `mingw32-make --version`
    # GNU Mkae 4.x

### Setup System Environment Variables:
    Add this in path: `C:\msys64\ucrt64\bin`

## Verify Installation in Visual Studio Code:

    @code{.bash}
    gcc --version
    g++ --version
    cmake --version
    mingw32-make --version
    @endcode

---

## Step 3: Clone Opencv {#tutorial_windows_install_clone}

## Clone the OpenCV repository:
    @code{.bash}
    git clone https://github.com/opencv/opencv.git
    cd opencv
    @endcode

### (Optional) Clone extra modules:

    git clone https://github.com/opencv/opencv_contrib.git

---

## Step 4: Create Build Directory {#tutorial_windows_install_builddir}
    @code{.bash}
    mkdir build && cd build
    @endcode

---

## Step 5: Configure with `Mingw32-Cmake` {#tutorial_windows_install_configure}

    @code{.cmake}
    cmake -G "MinGW Makefiles" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX=install ^
      -DBUILD_opencv_python3=ON ^
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ^
      ../
    @endcode

@note If not using `opencv_contrib`, remove the `OPENCV_EXTRA_MODULES_PATH` option.

---

## Step 6: Build Opencv with mingw32-make  {#tutorial_windows_install_build}

    @code{.bash}
    mingw32-make -j 6
    @endcode

@note If failed then decrese the code Ex. `mingw32-make -j 4`

---

## Step 7: Intall Compiled lib {#tutorial_windows_install_install}

    mingw32-make install

@note For using Opencv Outside, Add this path in System Variable Path:
    ./opencv/build/install/x64/mingw/bin

Check this PATH in your `opencv/build` and copy the full path and Add to the ENV Variable

---

## Step 8: Use Opencv in cpp project {#tutorial_windows_install_verify}

1. Make a folder `first project`
2. Create a file named test.cpp
3. Copy the code below:

    @code{.cpp}
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace std;
    using namespace cv;
    
    int main() {
    cout << CV_VERSION << endl;
    
    return 0;
    }

    @endcode

4. Create `CmakeLists.txt` and copy this:
 
   @code{.cmake}
    cmake_minimum_required(VERSION 3.10)
    project(OpenCVApp)

    # 🔽 ADD THIS LINE (change path to yours)
    set(OpenCV_DIR "D:/open-source/opencv/build-new/install")

    find_package(OpenCV REQUIRED)
    add_executable(app main.cpp) // app is your project name 
    target_link_libraries(app ${OpenCV_LIBS})

    @endcode

5. Make a folder `build` inside the first-project
6. Inside build Run ` cmake -G "MinGW Makefiles"` ../
6. Run this in build folder `mingw32-make`
@image html images/build-complete.png "build"
7. app.exe

If successful, the installed OpenCV version will be printed.

@note Note :- When make changes in cpp file then run again `mingw32-make` for compilation and run `app.exe`
---


