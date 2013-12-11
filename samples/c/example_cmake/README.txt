Example for CMake build system.

Compile OpenCV with cmake, preferently in an off-tree build, for example:

 $ mkdir opencv-release
 $ cd opencv-release
 $ cmake <OPENCV_SRC_PATH>
 $ make

And, *only optionally*, install it with.
 $ sudo make install

Then create the binary directory for the example with:
 $ mkdir example-release
 $ cd example-release

Then, if "make install" have been executed, directly running
 $ cmake <OPENCV_SRC_PATH>/samples/c/example_cmake/

will detect the "OpenCVConfig.cmake" file and the project is ready to compile.

If "make install" has not been executed, you'll have to manually pick the opencv
binary directory (Under Windows CMake may remember the correct directory). Open
the CMake gui with:
 $ cmake-gui <OPENCV_SRC_PATH>/samples/c/example_cmake/

And pick the correct value for OpenCV_DIR.
