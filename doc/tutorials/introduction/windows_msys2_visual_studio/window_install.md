@page tutorial_windows_visual_studio Building OpenCV on Windows using MSYS2 UCRT64

@tableofcontents


@section tutorial_windows_install_intro Introduction

# This tutorial describes how to build OpenCV from source on Windows using the MSYS2 UCRT64 environment.

### The build configuration uses:

- MSYS2 (UCRT64 shell)
- GCC (UCRT toolchain)
- CMake
- mingw32-make

This method produces native Windows binaries linked against the Universal C Runtime (UCRT).

---

@section tutorial_windows_install_prerequisites Prerequisites

## Install the following software before proceeding:

- MSYS2
- Git
- Python
@image html images/MSYS2-download.png "MSYS2"

### After installing MSYS2, always open:

    MSYS2 UCRT64
@image html images/UCRT64-shell.png "UCRT64"

### Do not use the MSYS, MinGW64, or CLANG64 shells for this build.

---

@section tutorial_windows_install_update  Step 1: Update MSYS2

## Open the MSYS2 UCRT64 shell and update the system:

    @code{.bash}
    pacman -Syu
    @endcode

---

@section tutorial_windows_install_packages Step 2: Install Required Packages

## Inside the UCRT64 shell, install required packages:
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

### If it is gifing same output as in `UCRT64 SHELL`: OK
 
---

@section tutorial_windows_install_clone Step 3: Clone OpenCV

## Clone the OpenCV repository:
    @code{.bash}
    git clone https://github.com/opencv/opencv.git
    cd opencv
    @endcode

### (Optional) Clone extra modules:

    git clone https://github.com/opencv/opencv_contrib.git

---

@section tutorial_windows_install_builddir Step 4: Create Build Directory

## Create and enter a build directory:
    @code{.bash}
    mkdir build && cd build
    @endcode

---

@section tutorial_windows_install_configure  Step 5: Configure with CMake

## Configure the build using the MinGW Makefiles generator:
    @code{.cmake}
    cmake -G "MinGW Makefiles" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX=install ^
      -DBUILD_opencv_python3=ON ^
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ^
      ../
    @endcode

If not using `opencv_contrib`, remove the `OPENCV_EXTRA_MODULES_PATH` option.

### Ensure configuration completes without errors.

---

@section tutorial_windows_install_build Step 6: Build OpenCV

## Compile OpenCV:
    @code{.bash}
    mingw32-make -j 6
    @endcode

If failed then decrese the code Ex. `mingw32-make -j 4`

---

@section tutorial_windows_install_install ** Step 7: Install**

## Install the compiled libraries:

    mingw32-make install

### For using Opencv Outside, Add this path in System Variable Path:
    `./opencv/build/install/x64/mingw/bin`

Check this PATH in your `opencv/build` and copy the full path and Add to the ENV Variable

---

@section tutorial_windows_install_verify ** Step 8: Use Opencv in CPP **

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

    // 🔽 ADD THIS LINE (change path to yours)
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

### Note :- When make changes in cpp file then run again `mingw32-make` for compilation and run `app.exe`

---

@section tutorial_windows_install_notes Notes

- UCRT64 uses the modern Windows Universal C Runtime.
- Binaries built with UCRT64 are not compatible with MSVC builds.
- Do not mix MSYS2 environments.
- Always build and compile within the same UCRT64 shell.

---


