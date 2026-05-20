# Building OpenCV on Windows from Source Code using MSYS2 UCRT64 and VS Code (C++)

:::{div} opencv-meta-table

| | |
| -: | :- |
| Original author | Mahadev Kumar |
| Compatibility | OpenCV >= 4.x |

:::

:::{note}
This tutorial was tested on Windows >= 7 using the MSYS2 UCRT64 environment.
The OpenCV team does not maintain MSYS/Cygwin configuration and does not regularly test it with continuous integration.
:::
---

(tutorial_windows_msys2_install_intro)=
### Introduction
This tutorial explains how to build OpenCV from source on Windows using the **MSYS2 UCRT64 toolchain** and use it inside **Visual Studio Code with C++**.

The build configuration uses:

- **MSYS2 (UCRT64 shell)**
- **GCC (UCRT toolchain)**
- **CMake**
- **mingw32-make**
- **VS Code**

This method produces **native Windows binaries linked against the Universal C Runtime (UCRT)**.

---

(tutorial_windows_install_msys2_prerequisites)=
### Prerequisites
Install the following software before proceeding:

- [MSYS2](https://www.msys2.org)
- [Git](https://git-scm.com/install)
- [Visual Studio Code](https://code.visualstudio.com)

After installing MSYS2, always open the:

**MSYS2 UCRT64 Terminal**

:::{note}
Do not use the `MSYS`, `MinGW64`, or `CLANG64` shells for this build.
:::
---

(tutorial_windows_msys2_install_update)=
### Step 1: Update MSYS2
Open the **MSYS2 UCRT64** terminal and update the system.

```bash
pacman -Syu
```

Restart the terminal if prompted.

---

(tutorial_windows_msys2_install_packages)=
### Step 2: Install Required Packages
Install the required compiler and build tools.

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-make
```

Verify installation:

```bash
gcc --version
cmake --version
mingw32-make --version
```

:::{note}
**Add the following directory to your `Windows PATH`:**

```bash
C:\msys64\ucrt64\bin
```

:::
---

(tutorial_windows_msys2_install_clone)=
### Step 3: Clone OpenCV Source Code
Clone `OpenCV` and optional `contrib modules`.

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

---

(tutorial_windows_msys2_install_build_dir)=
### Step 4: Create Build Directory

```bash
cd opencv
mkdir build
cd build
```

---

(tutorial_windows_msys2_install_configure)=
### Step 5: Configure the Build with CMake
Run CMake using the **MinGW Makefiles generator**.

```bash
cmake -G "MinGW Makefiles" ../
```

:::{note}
If you do not want to use **opencv_contrib**, remove the `OPENCV_EXTRA_MODULES_PATH` option.
:::
---

(tutorial_windows_msys2_install_build)=
### Step 6: Build OpenCV
Compile OpenCV.

```bash
mingw32-make -j6
```

:::{note}
If the build fails due to memory limitations, reduce the job count.
:::
Example:

```bash
mingw32-make -j4
```

---

(tutorial_windows_msys2_install_install)=
### Step 7: Install OpenCV
Install the compiled libraries.

```bash
mingw32-make install
```

After installation, OpenCV will be located in:
`opencv/build/install`

Add the following directory to your **Windows PATH**:

```bash
C:\path\to\opencv\build\install\x64\mingw\bin
```

---

(tutorial_windows_msys2_install_verify)=
### Step 8: Verify with a C++ Example
- Create a folder outside the OpenCV source directory.
Example: `first-project`
- Inside the folder create **main.cpp**

  ```cpp
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
  ```

- Create **CMakeLists.txt**

  ```cmake
  cmake_minimum_required(VERSION 3.10)

  project(OpenCVApp)

  set(OpenCV_DIR "C:/path/to/opencv/build/install/lib/cmake/opencv4")

  find_package(OpenCV REQUIRED)

  add_executable(app main.cpp)

  target_link_libraries(app ${OpenCV_LIBS})
  ```

- Build the project.

  ```bash
  mkdir build && cd build
  cmake -G "MinGW Makefiles" ..
  mingw32-make
  ```

- Run the program.

```bash
./app.exe
```

:::{note}
If everything is configured correctly, a window should appear displaying a blank image.
:::
---

(tutorial_windows_msys2_vscode_section)=
### Using OpenCV in Visual Studio Code
Install the following extensions in **VS Code**:

- C/C++
- CMake Tools

Open your project folder in VS Code.

Configure the project using CMake.

```bash
cmake -G "MinGW Makefiles" ..
```

Build the project.

```bash
mingw32-make
```

Run the executable.

---

(tutorial_windows_msys2_install_troubleshoot)=
### Troubleshooting
#### CMake cannot find OpenCV

Ensure that `OpenCV_DIR` is set correctly.

Example:

```cmake
set(OpenCV_DIR "C:/path/to/opencv/build/install/lib/cmake/opencv4")
```

---

#### mingw32-make not found

Make sure the following path is added to the Windows environment variable **PATH**: `C:\msys64\ucrt64\bin`

---

#### Build fails due to memory limits

Reduce parallel build jobs.

```bash
mingw32-make -j2
```

---

### Conclusion

You have successfully built OpenCV from source using **MSYS2 UCRT64** and verified it with a **C++ project in VS Code**.

This setup allows you to develop OpenCV-based C++ applications using a fully open-source toolchain on Windows.
