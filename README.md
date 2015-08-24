Purpose of this branch is to keep samples that can be built in VS2015RC and pregenerated projects as CMake that supports VS2015 was not released on the moment this branch created.

Currently samples are started on Win10/VS201RC but there is not request for accesing camera. And as a result there is no image from camera in the app. Possible VS2015RC issue. Need to be checked on VS2015 release and fixed if reproduced

Needs maintanence: rebasing on the latest [Itseez](https://github.com/Itseez/opencv) master; Regenerating of projects (Look for "Long way" in [README](#test-vs2015-branch)); Checking builds/runs for required samples (mentioned [there](#vs2015-samples-branch)). Adding of CI integration with AppVeyor or other similar tool can be used to reduce maintanence effort.

Support of this branch can be dropped as "no more required" after CMake with VS2015 support released and related WinRT build script (platforms/winrt/setup_winrt.ps1) and code updates applied + related samples fixed for working on Win10/VS2015

## OpenCV: Open Source Computer Vision Library

* [Building for WinRT 8.x](#building-for-winrt-8x)
    * [Requirements](#requirements)
    * [Build](#build)
        * [Overview](#overview)
        * [Generate](#generate)
        * [Build](#build)
        * [Install](#install)
    * [Customize](#customize)
        * [Using CMake commands](#using-cmake-commands)
* [Building for WinRT 10](#building-for-winrt-10)
    * [test-vs2015 branch](#test-vs2015-branch)
        * [Short way (using pregenerated projects)](#short-way-using-pregenerated-projects)
        * [Long way (generating projects on your own)](#long-way-generating-projects-on-your-own)
        * [Common build steps](#common-build-steps)
    * [vs2015-samples branch](#vs2015-samples-branch)
* [Troubleshooting](#troubleshooting)
* [Resources](#resources)
* [Contributing](#contributing)

### Building for WinRT 8x

#### Requirements

- CMake 3.1.0 or higher
- Windows Phone/Store 8.1 Visual Studio 2013
- Windows Phone/Store 8.0 Visual Studio 2012

To be able to build for Windows Phone and Windows Store install the following prerequisites:
- [Visual Studio 2013 Community Edition](http://go.microsoft.com/?linkid=9863608)
- [Visual Studio Express 2012 for Windows Desktop](http://www.microsoft.com/en-us/download/details.aspx?id=34673)


#### Build

##### Overview
Build process includes
 1. **_generating_** solution/projects with CMake
 1. **_building_** generated projects with ```msbuild```
 1. **_installing_** (building generated INSTALL target) to accumulate standalone binary/headers ready for further use
 
You can accomplish all of these steps for OpenCV for Windows Phone (8.0, 8.1) and Windows Store (8.0, 8.1) with a single command

```bash
cd opencv/platforms/winrt
setup_winrt.bat "WP,WS" "8.0,8.1" "x86,ARM" -b
```
Executing this command will perform the following actions.

##### Generate

_**(1) generate**_ the following folder structure in the ```opencv/bin``` directory:

```
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
            ...
    WP
        8.0
            ...
        8.1
           ...
    WS
        ...
```
Note that x64 CMake generation support is as follows:

Platform\Version | 8.0 | 8.1 |
-----------------|-----|-----|
Windows Phone    | No  | No  |
Windows Store    | Yes | Yes |

##### Build

"**-b**" flag in the command above is used to **_(2) build_** each generated solution in both "*Debug*" and "*Release*" configurations. If you don't want to build all configurations automatically, you can omit "**-b**" flag and build OpenCV.sln for the particular platform you are targeting manually. Due to the current limitations of CMake, separate x86/x64/ARM projects must be generated for each platform.

##### Install
Using **-b** flag will also _**(3) install**_ required headers and binaries by means of creating the predefined "*INSTALL*" project within generated solutions and building it. This will create a separate install location that accumulates binaries and includes for specified platforms. Default location is "*<ocv-src>\bin\install\*".

**WinRT samples** reference 'install' binaries and include files via "```OPENCV_WINRT_INSTALL_DIR```" environment variable. Please declare it and point to ```<ocv-src>\bin\install``` directory to resolve references within sample applications.

#### Customize

Use ```setup_winrt.bat -h``` to see all available option

You can omit building and installing steps and only **_generate_** projects for a single specific configuration
    `setup_winrt.bat "WP" "8.1" "x86"`

or a subset of configurations
    `setup_winrt.bat "WP,WS" "8.1" "x86"`

*Note:* setup_winrt.bat calls the unsigned PowerShell script with the -ExecutionPolicy Unrestricted option.

##### Using CMake commands

```cmake [options] <path-to-source>```

- Windows Phone 8.1 x86**

 ```cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>```

- Windows Phone 8.1 ARM**

 ```cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>```

- Windows Store 8.1 x86**

 ```cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>```

- Windows Store 8.1 ARM**

 ```cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.1 <path-to-source>```

*Note:* For Windows 8.0 Phone and Store you can specify either Visual Studio 11 2012 or Visual Studio 12 2013 as the generator

- Windows Phone 8.0 x86

 ```cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>```

- Windows Phone 8.0 ARM

 ```cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>```

- Windows Store 8.0 x86

 ```cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>```

- Windows Store 8.0 ARM

 ```cmake -G "Visual Studio 12 2013 ARM" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=8.0 <path-to-source>```

For example to generate Windows Phone 8.1 x86 project files in the *opencv/bin* dir

 ```bash
 mkdir bin
 cd bin
 cmake -G "Visual Studio 12 2013" -DCMAKE_SYSTEM_NAME=WindowsPhone -DCMAKE_SYSTEM_VERSION=8.1 ../
 ```
</br>
### Building for WinRT 10

There are VS2015 branches in the repo tested with VS2015RC on Windows 10 Insider Preview (10074).

#### Test-vs2015 branch
Branch contains pregenerated solutions for VS2015RC in ```bin_VS2015_RC``` directory (```bin``` directory contains solutions for VS2015 CTP6 however support of these solutions is dropped and correct work is not guaranteed) with accuracy and performance tests enabled.

Follow instructions below to get OpenCV binaries on Windows 10 VS2015RC.

1. Set ```OPENCV_TEST_DATA_PATH``` environment variable to location of ```opencv_extra/testdata``` (cloning of [opencv_extra](https://github.com/Itseez/opencv_extra) repo required) to get tests work correctly. Also, set ```OPENCV_PERF_VALIDATION_DIR``` environment variable in case you are planning to have place for storing and comparing of performance tests result.
2. Install VS2015 RC from [there](https://www.visualstudio.com/).  Select "Universal Windows App Development Tools" using Custom setup option.
3. Jump to **Short way** or **Long way** depending on your aims and preferences

##### Short way (using pregenerated projects)
*Note:* 
Use pregenerated projects from ```bin_VS2015_RC``` folder if you are working with Visual Studio 2015 RC (Version 14.0.22823.1 D14REL) on Windows 10 OS (build 10074 or 10075).

Use pregenerated projects from bin folder if you are still working with Visual Studio 2015 CTP6 (Version 14.0.22609.0 D14REL, support is dropped now).

These projects were tested with specific versions of Visual Studio 2015 and might not work correctly with other pre-release versions.

1. Set environment variable ```OCV2015_ROOT``` to the OpenCV location (e.g. ```Path\to\OCV\opencv```)
2. Check if ```OCV2015_ROOT``` displayed properly in VS2015 macros: open any solution, right click on any project, choose "Properties", find any field that supports macros (say, Configuration Properties -> General -> Output Directory), choose "Edit", click "Macros \>\>" button and look for OCV2015_ROOT there. You may need to delete .sdf file to make VS2015 get updated value of environment variable (it looks like VS caches some vars on solution opening)
3. Jump to **Common build steps**

##### Long way (generating projects on your own)

1. Clone ```https://github.com/Microsoft/CMake.git```
2. Checkout ```feature/Win10``` branch
3. Build CMake for Windows
4. Create ```bin``` directory and cd to it
5. Run command ```Path\To\New CMake\bin\Release\cmake.exe -G "Visual Studio 14 2015 Win64" -DCMAKE_SYSTEM_NAME:String=WindowsStore -DCMAKE_SYSTEM_VERSION:String="10.0" -DCMAKE_VS_EFFECTIVE_PLATFORMS:String=x64 -DCMAKE_INSTALL_PREFIX:PATH=C:\Users\ocv_install ..```

##### Common build steps

1. Open generated solution, change configuration to 'Release' and build it
2. Running tests:
 
 - **Accuracy:** Run ```opencv_test_{module}.exe``` via console or as usual by double clicking it. You should see output in the console window
 - **Performance:** Run ```opencv_perf_{module}.exe``` via console or as usual by double clicking it. You should see output in the console window. In case you'd like to write test results to file use ```--perf_write_validation_results=<filename> parameter```. To compare current results to previous use ```--perf_read_validation_results=<filename>```. This should read/write files from/to ```OPENCV_PERF_VALIDATION_DIR```

#### vs2015-samples branch

This branch contains several samples that were tested on **Windows 10 10074 + Visual Studio 2015 RC**

Before using of samples you need to set environment variable ```OCV2015_ROOT``` to the OpenCV location (e.g. ```Path\to\OCV\opencv```)

The following samples successfully build with VS2015 RC:

- [highgui_xaml](https://github.com/MSOpenTech/opencv/tree/vs2015-samples/samples/winrt_universal/highgui_xaml) (*Release, Win32*)

- [VideoioXAML-arm](https://github.com/MSOpenTech/opencv/tree/vs2015-samples/samples/winrt_universal/VideoioXAML-arm) (*Release, ARM*)

Building configurations other than mentioned in brackets is not guaranteed and may require additional updates of headers/libraries paths.

### Troubleshooting

- In case you're getting errors like (known repro on VS2015RC, test-vs2015 branch)

 ```
 Error APPX1639 File 'Microsoft.Windows.Build.Appx.AppxPackaging.dll.manifest' not found.
 ```
 
 Check your environment set up. Likely some tools are referred from the incorrect SDK (e.g. from 8.1 instead of 10.0). Try setting of `WindowsSdkDir` variable to empty value before build e.g. ```set WindowsSdkDir=```
 Find more details on repro and resolution [there](https://github.com/MSOpenTech/opencv/issues/39)
- In case  you're getting something like (repro with VS2013, master branch)

 ```
 error MSB4057: The target Build doesn't exist in the project
 ```

 Check your environment set up. Likely VS2013 installation cannot find msbuild. Add VS2013 msbuild location to PATH environment variable and run required commands from VS2013 command prompt.
- In case you're trying to build WindowsStore and WindowsPhone 8.0/8.1 projects having only VS2015 installation you'll likely get the following error during setup_winrt.bat run:

 ```
 A Windows Phone component with CMake requires both the Windows Desktop SDK
as well as the Windows Phone '8.0' SDK. Please make sure that you have
both installed
 ```
 
 Resolution is
 1. Install VS2013 Community (w/ Windows 8.0 tools checkbox enabled) and VS2012 Express For Desktop to have all required SDKs
 2. Open ```setup_winrt.ps1``` and change `[ValidateSet("Visual Studio 12 2013","Visual Studio 11 2012")]` to `[ValidateSet("Visual Studio 12 2013","Visual Studio 11 2012","Visual Studio 14 2015")]`
 3. Specify generator explicitly using ```-g``` script option e.g. ```-g "Visual Studio 14 2015"```
 
 More details [here](https://github.com/MSOpenTech/opencv/issues/46)

### Resources

* Homepage: <http://opencv.org>
* Docs: <http://docs.opencv.org/master/>
* Q&A forum: <http://answers.opencv.org>
* Issue tracking: <https://github.com/Itseez/opencv/issues>

### Contributing

Please read before starting work on a pull request: <https://github.com/Itseez/opencv/wiki/How_to_contribute>

Summary of guidelines:

* One pull request per issue;
* Choose the right base branch;
* Include tests and documentation;
* Clean up "oops" commits before submitting;
* Follow the coding style guide.

[![Gittip](http://img.shields.io/gittip/OpenCV.png)](https://www.gittip.com/OpenCV/)
