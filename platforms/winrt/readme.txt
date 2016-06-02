Building OpenCV from Source, using CMake and Command Line
=========================================================

Requirements
============
CMake 3.1.0 or higher
Windows Phone/Store 8.1 Visual Studio 2013
Windows Phone/Store 8.0 Visual Studio 2012

For example, to be able to build all Windows Phone and Windows Store projects install the following:

Install Visual Studio 2013 Community Edition
    http://go.microsoft.com/?linkid=9863608

Install Visual Studio Express 2012 for Windows Desktop
    http://www.microsoft.com/en-us/download/details.aspx?id=34673



To create and build all Windows Phone (8.0, 8.1) and Windows Store (8.0, 8.1) Visual Studio projects
==========================================================================================
cd opencv/platforms/winrt
setup_winrt.bat "WP,WS" "8.0,8.1" "x86,ARM" -b

If everything's fine, a few minutes later you will get the following output in the opencv/bin directory:

bin
    install
        WP
            8.0
                ARM
                x86
            8.1
                ARM
                x86
        WS
            8.0
                ARM
                x86
            8.1
                ARM
                x86
    WP
        8.0
            ARM
            x86
        8.1
            ARM
            x86
    WS
        8.0
            ARM
            x86
        8.1
            ARM
            x86

"-b" flag in the command above builds each generated solutions in both "Debug" and "Release" configurations. It also builds the predefined "INSTALL" project within generated solutions. Building it creates a separate install location that accumulates binaries and includes for specified platforms. Default location is "<ocv-src>\bin\install\".

WinRT samples reference 'install' binaries and include files via "OPENCV_WINRT_INSTALL_DIR" environment variable. Please declare it and point to "<ocv-src>\bin\install\" directory to resolve references within sample applications.

If you don't want to build all configurations automatically, you can omit "-b" flag and build OpenCV.sln for the particular platform you are targeting manually. Due to the current limitations of CMake, separate x86/x64/ARM projects must be generated for each platform.

You can also target a single specific configuration
    setup_winrt.bat "WP" "8.1" "x86"

Or a subset of configurations
    setup_winrt.bat "WP,WS" "8.1" "x86"

To display the command line options for setup_winrt.bat
    setup_winrt.bat -h

Note that x64 CMake generation support is as follows:
------------------------------
Platform\Version | 8.0 | 8.1 |
-----------------|-----|-----|
Windows Phone    | No  | No  |
Windows Store    | Yes | Yes |

Note: setup_winrt.bat calls the unsigned PowerShell script with the -ExecutionPolicy Unrestricted option.


CMake command line options for Windows Phone and Store
======================================================

cmake [options] <path-to-source>

Windows Phone 8.1 x86
cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>

Windows Phone 8.1 ARM
cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>

Windows Store 8.1 x86
cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>

Windows Store 8.1 ARM
cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>

Note: For Windows 8.0 Phone and Store you can specify either Visual Studio 11 2012 or Visual Studio 12 2013 as the generator

Windows Phone 8.0 x86
cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>

Windows Phone 8.0 ARM
cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>

Windows Store 8.0 x86
cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>

Windows Store 8.0 ARM
cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>

Example
======================================================

To generate Windows Phone 8.1 x86 project files in the opencv/bin dir

mkdir bin
cd bin
cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.1 ../

Running tests for Windows Store
===============================
1. You might need to install this if you haven't already: http://www.microsoft.com/en-US/download/details.aspx?id=40784

2. Set OPENCV_TEST_DATA_PATH environment variable to location of opencv_extra/testdata (cloning of https://github.com/Itseez/opencv_extra repo required) to get tests work correctly. Also, set OPENCV_PERF_VALIDATION_DIR environment variable in case you are planning to have place where to store performance test results and compare them with the future test runs.

3. In case you'd like to adjust some flags that are defaulted by setup_winrt script, go to "Manual build" section. Otherwise go to platforms/winrt and execute

setup_winrt.bat "WS" "8.1" "x64"

This will generate all files needed to build open_cv projects for selected platform in opencv\bin\<Depends on generated configuration>. Open the opencv\bin\<path to required configuration> directory and open the OpenCV.sln.

4. Set OCV solution to Release mode and build it. They should build without errors and generate executables in "bin\WS\8.1\x64\bin\Release\" (or similar path depending on the configuration)

5. Running tests:
 - **Accuracy:** Run opencv_test_{module}.exe via console or as usual by double clicking it. You should see output in the console window
 - **Performance:** Run opencv_perf_{module}.exe via console or as usual by double clicking it. You should see output in the console window. In case you'd like to write test results to file use --perf_write_validation_results=<filename> parameter; To compare current results to previous use --perf_read_validation_results=<filename>. This should read/write files from/to OPENCV_PERF_VALIDATION_DIR

Manual build
============

 CMake interface:
-----------------
  1. Set CMAKE_SYSTEM_NAME to WindowsStore or WindowsPhone and CMAKE_SYSTEM_VERSION to 8.0 or 8.1
  2. Set CMAKE_INSTALL_PREFIX using format "<install dir>\WS\8.1\x64" (this structure is required by samples)
  3. Click "Configure" and choose required generator
  4. Click "Generate"

 Command line:
--------------
  1. md bin
  2. cd bin
  3. Add any required parameters to this command and execute it:

  cmake -G "Visual Studio 12 2013 Win64" -DCMAKE_SYSTEM_NAME:String=WindowsStore -DCMAKE_SYSTEM_VERSION:String=8.1 -DCMAKE_VS_EFFECTIVE_PLATFORMS:String=x64 -DCMAKE_INSTALL_PREFIX:PATH=.\install\WS\8.1\x64\ ..

Return to "Running tests for Windows Store", list item 4.