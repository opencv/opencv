/**
@page tutorial_windows_msys2_vscode Building OpenCV on Windows using MSYS2 UCRT64

@prev_tutorial{tutorial_linux_eclipse}
@next_tutorial{tutorial_windows_visual_studio_opencv}

| | |
| -: | :- |
| Original author | Your Name |
| Compatibility | OpenCV >= 4.0 |

@tableofcontents

@warning
This tutorial can contain obsolete information. The description here was tested on Windows 7 SP1. Nevertheless, it should also work on any other relatively modern version of Windows OS. If you encounter errors, feel free to contact us via the [OpenCV Q&A forum](https://forum.opencv.org).

## Introduction {#tutorial_windows_msys2_install_intro}

This tutorial describes how to build OpenCV from source on Windows using the MSYS2 UCRT64 environment.

**The build configuration uses:**

- MSYS2 (UCRT64 shell)
- GCC (UCRT toolchain)
- CMake
- mingw32-make

This method produces native Windows binaries linked against the Universal C Runtime (UCRT).

---

## Prerequisites {#tutorial_windows_install_msys2_prerequisites}

**Install the following software before proceeding:**

- [MSYS2](https://www.msys2.org/)
- [Git](https://git-scm.com/)
- [Python](https://www.python.org/)

**After installing MSYS2, always open:**
The **MSYS2 UCRT64** terminal.



@note Do not use the MSYS, MinGW64, or CLANG64 shells for this build.

---

## Step 1: Update MSYS2 {#tutorial_windows_msys2_install_update}

Open the MSYS2 UCRT64 shell and update the system:

@code{.bash}
pacman -Syu
@endcode

---

## Step 2: Install Required Packages {#tutorial_windows_msys2_install_packages}

Inside the UCRT64 shell, install the required toolchain:

@code{.bash}
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-make
@endcode

**Verify installation:**

@code{.bash}
gcc --version
cmake --version
mingw32-make --version
@endcode

**Setup System Environment Variables:**
Add the following to your Windows System PATH: `C:\msys64\ucrt64\bin`

---

## Step 3: Clone OpenCV {#tutorial_windows_msys2_install_clone}

@code{.bash}
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
@endcode

---

## Step 4: Create Build Directory {#tutorial_windows_msys2_install_build_dir}

@code{.bash}
cd opencv
mkdir build && cd build
@endcode

---

## Step 5: Configure with CMake {#tutorial_windows_msys2_install_configure}

Use the following command to configure the build. Note the use of `MinGW Makefiles`.

@code{.bash}
cmake -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=install \
  -DBUILD_opencv_python3=ON \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  ../
@endcode

@note If not using `opencv_contrib`, remove the `OPENCV_EXTRA_MODULES_PATH` option.

---

## Step 6: Build OpenCV {#tutorial_windows_msys2_install_build}

@code{.bash}
mingw32-make -j6
@endcode

@note If the build fails due to memory exhaustion, decrease the job count (e.g., `-j4` or `-j2`).

---

## Step 7: Install Compiled Libraries {#tutorial_windows_msys2_install_mingw32}

@code{.bash}
mingw32-make install
@endcode

**Post-Install:** Add the install bin folder to your Windows PATH:
`C:\your_path\opencv\build\install\x64\mingw\bin`

---

## Step 8: Verify with a C++ Project {#tutorial_windows_msys2_install_verify}

1. Create a folder named `first_project`.
2. Create `main.cpp`:

@code{.cpp}
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV Version: " << cv::CV_VERSION << std::endl;
    return 0;
}
@endcode

3. Create `CMakeLists.txt`:

@code{.cmake}
cmake_minimum_required(VERSION 3.10)
project(OpenCVApp)

# Set the path to where OpenCV was installed
set(OpenCV_DIR "C:/path/to/your/opencv/build/install")

find_package(OpenCV REQUIRED)
add_executable(app main.cpp)
target_link_libraries(app ${OpenCV_LIBS})
@endcode

4. Build and run:
@code{.bash}
mkdir build && cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
./app.exe
@endcode

*/
