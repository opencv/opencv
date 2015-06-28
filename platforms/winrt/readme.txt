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
