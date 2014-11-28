Installation in iOS {#tutorial_ios_install}
===================

Required Packages
-----------------

-   CMake 2.8.8 or higher
-   Xcode 4.2 or higher

### Getting the Cutting-edge OpenCV from Git Repository

Launch GIT client and clone OpenCV repository from [here](http://github.com/itseez/opencv)

In MacOS it can be done using the following command in Terminal:

@code{.bash}
cd ~/<my_working _directory>
git clone https://github.com/Itseez/opencv.git
@endcode

Building OpenCV from Source, using CMake and Command Line
---------------------------------------------------------

-#  Make symbolic link for Xcode to let OpenCV build scripts find the compiler, header files etc.
    @code{.bash}
    cd /
    sudo ln -s /Applications/Xcode.app/Contents/Developer Developer
    @endcode

-#  Build OpenCV framework:
    @code{.bash}
    cd ~/<my_working_directory>
    python opencv/platforms/ios/build_framework.py ios
    @endcode

If everything's fine, a few minutes later you will get
\~/\<my_working_directory\>/ios/opencv2.framework. You can add this framework to your Xcode
projects.

Further Reading
---------------

You can find several OpenCV+iOS tutorials here @ref tutorial_table_of_content_ios.
