Installation in iOS {#tutorial_ios_install}
===================

@tableofcontents

@next_tutorial{tutorial_hello}

|    |    |
| -: | :- |
| Original author | Artem Myagkov, Eduard Feicho, Steve Nicholson |
| Compatibility | OpenCV >= 3.0 |

@warning
This tutorial can contain obsolete information.

Required Packages
-----------------

-   CMake 2.8.8 or higher
-   Xcode 4.2 or higher

### Getting the Cutting-edge OpenCV from Git Repository

Launch Git client and clone OpenCV repository from [GitHub](http://github.com/opencv/opencv).

In MacOS it can be done using the following command in Terminal:

@code{.bash}
cd ~/<my_working _directory>
git clone https://github.com/opencv/opencv.git
@endcode

If you want to install OpenCV’s extra modules, clone the opencv_contrib repository as well:

@code{.bash}
cd ~/<my_working _directory>
git clone https://github.com/opencv/opencv_contrib.git
@endcode


Building OpenCV from Source, using CMake and Command Line
---------------------------------------------------------

1.  Make sure the xcode command line tools are installed:
    @code{.bash}
    xcode-select --install
    @endcode

2.  Build OpenCV framework:
    @code{.bash}
    cd ~/<my_working_directory>
    python opencv/platforms/ios/build_framework.py ios
    @endcode

3.  To install OpenCV’s extra modules, append `--contrib opencv_contrib` to the python command above. **Note:** the extra modules are not included in the iOS Pack download at [OpenCV Releases](https://opencv.org/releases/). If you want to use the extra modules (e.g. aruco), you must build OpenCV yourself and include this option:
    @code{.bash}
    cd ~/<my_working_directory>
    python opencv/platforms/ios/build_framework.py ios --contrib opencv_contrib
    @endcode

4.  To exclude a specific module, append `--without <module_name>`. For example, to exclude the "optflow" module from opencv_contrib:
    @code{.bash}
    cd ~/<my_working_directory>
    python opencv/platforms/ios/build_framework.py ios --contrib opencv_contrib --without optflow
    @endcode

5.  The build process can take a significant amount of time. Currently (OpenCV 3.4 and 4.1), five separate architectures are built: armv7, armv7s, and arm64 for iOS plus i386 and x86_64 for the iPhone simulator. If you want to specify the architectures to include in the framework, use the `--iphoneos_archs` and/or `--iphonesimulator_archs` options. For example, to only build arm64 for iOS and x86_64 for the simulator:
    @code{.bash}
    cd ~/<my_working_directory>
    python opencv/platforms/ios/build_framework.py ios --contrib opencv_contrib --iphoneos_archs arm64 --iphonesimulator_archs x86_64
    @endcode

If everything’s fine, the build process will create
`~/<my_working_directory>/ios/opencv2.framework`. You can add this framework to your Xcode projects.

Further Reading
---------------

You can find several OpenCV+iOS tutorials here @ref tutorial_table_of_content_ios.
