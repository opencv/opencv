Building OpenCV with ARM Performance Libraries (ARMPL) on Windows {#tutorial_windows_armpl}
==================================================================

@prev_tutorial{tutorial_windows_install}
@next_tutorial{tutorial_linux_install}

|    |    |
| -: | :- |

@tableofcontents

Introduction {#tutorial_windows_armpl_intro}
============

This tutorial explains how to build OpenCV on Windows (AArch64) with
[ARM Performance Libraries (ARMPL)](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries)
as a math backend. ARMPL provides optimized BLAS and LAPACK routines for Arm-based hardware
and can significantly accelerate OpenCV operations such as DFT and DCT.

Step 1: Download and Install ARM Performance Libraries {#tutorial_windows_armpl_download}
=====================================================

1. Open a browser and go to the
   [ARM Performance Libraries Downloads page](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries#Downloads).

2. Under **Windows / AArch64**, download the installer for your preferred toolchain:

   | File | Architecture | Size |
   |------|--------------|------|
   | `arm-performance-libraries_26.01_Windows.msi` | AArch64 | ~240 MiB |

3. Run the downloaded `.msi` installer and follow the on-screen instructions.
   The default installation directory is:
   ```
   C:\Program Files\Arm Performance Libraries\armpl_26.01
   ```

Step 2: Configure System Environment Variables {#tutorial_windows_armpl_env}
=============================================

OpenCV's CMake scripts (and the ARMPL runtime itself) need to find the library files at both
build time and run time. Add the following entries to the **System** `PATH` variable:

1. Open **System Properties**, click **Advanced**, then **Environment Variables**.
2. Under **System variables**, select `Path` and click **Edit**.
3. Add the two paths below (adjust the version number if yours differs):

   ```
   C:\Program Files\Arm Performance Libraries\armpl_26.01\lib
   C:\Program Files\Arm Performance Libraries\armpl_26.01\bin
   ```

4. Click **OK** on every dialog to save.

Step 3: Clone OpenCV {#tutorial_windows_armpl_clone}
====================

```bat
git clone https://github.com/opencv/opencv.git
cd opencv
```

If you also need the extra modules:

```bat
git clone https://github.com/opencv/opencv_contrib.git
```

Step 4: Configure with CMake {#tutorial_windows_armpl_cmake}
============================

Create a build directory and run CMake with ARMPL support enabled.

**Without OpenMP (single-threaded ARMPL):**

```bat
mkdir build && cd build

cmake -G "Visual Studio 17 2022" -A ARM64 ^
      -DWITH_ARMPL=ON ^
      -DARMPL_ROOT_DIR="C:\Program Files\Arm Performance Libraries\armpl_26.01" ^
      -DWITH_OPENMP=OFF ^
      ..
```

**With OpenMP (multi-threaded ARMPL):**

ARMPL ships both serial and OpenMP-enabled library variants. To use the multi-threaded variant,
enable OpenMP in CMake:

```bat
mkdir build && cd build

cmake -G "Visual Studio 17 2022" -A ARM64 ^
      -DWITH_ARMPL=ON ^
      -DARMPL_ROOT_DIR="C:\Program Files\Arm Performance Libraries\armpl_26.01" ^
      -DWITH_OPENMP=ON ^
      ..
```

@note Enabling `WITH_OPENMP=ON` causes CMake to link against the `armpl_lp64_mp` (multi-threaded)
variant of ARMPL. Disabling it links against the serial `armpl_lp64` variant. Only one variant
should be enabled at a time to avoid symbol conflicts.

Step 5: Build and Install {#tutorial_windows_armpl_build}
=========================

Open the generated `.sln` file in Visual Studio and build the **Release** configuration, or
build from the command line:

```bat
cmake --build . --config Release --parallel
cmake --install . --config Release
```

Step 6: Verify the Build {#tutorial_windows_armpl_verify}
=========================

After a successful build, confirm that OpenCV detects ARMPL by running:

```bat
opencv_version --verbose 2>&1 | findstr /i armpl
```

You should see a line similar to:

```
  ARMPL:                       YES (armpl_26.01)
```

Alternatively, check the CMake configuration log for the line:

```
--   ARMPL support:             YES
```

Troubleshooting {#tutorial_windows_armpl_troubleshoot}
===============

**CMake cannot find ARMPL:**

Make sure `ARMPL_ROOT_DIR` points to the folder that contains both `include\` and `lib\`
sub-directories:

```
C:\Program Files\Arm Performance Libraries\armpl_26.01
    bin\
    include\
    lib\
```

**Runtime error: DLL not found:**

Ensure that both the `lib\` and `bin\` directories are on the system `PATH` and that
you opened a new Command Prompt after adding them (changes are not picked up by already-open
sessions).

**Linker errors with OpenMP:**

If you see duplicate symbol errors when `WITH_OPENMP=ON`, make sure you are not also linking
against the serial ARMPL library. Pass `-DWITH_OPENMP=ON` consistently and clean the build
directory before re-running CMake.

See also {#tutorial_windows_armpl_seealso}
=========

- @ref tutorial_windows_install - Generic Windows build guide
- [ARM Performance Libraries documentation](https://developer.arm.com/documentation/101004/)
- @ref tutorial_general_install - General installation guide
