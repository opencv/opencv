Install OpenCV-JavaScript {#tutorial_js_setup}
===============================


Installing Emscripten
-----------------------------

[Emscripten](https://github.com/kripken/emscripten) is an LLVM-to-JavaScript compiler. We will use Emscripten to build OpenCV-JavaScript.

To Install emscripten, you can follow instructions of [Emscripten SDK](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).

For example
@code{.bash}
./emsdk update
./emsdk install latest
./emsdk activate latest
@endcode

After install, please make sure the `EMSCRIPTEN` environment is setup correctly.

For example
@code{.bash}
source ./emsdk_env.sh
echo ${EMSCRIPTEN}
@endcode

Getting OpenCV Source Code
--------------------------

You can use the latest stable OpenCV version or you can grab the latest snapshot from our [Git
repository](https://github.com/opencv/opencv.git).

### Getting the Latest Stable OpenCV Version

-   Go to our [downloads page](http://opencv.org/downloads.html).
-   Download the source archive and unpack it.

### Getting the Cutting-edge OpenCV from the Git Repository

Launch Git client and clone [OpenCV repository](http://github.com/opencv/opencv). If you need
modules from [OpenCV contrib repository](http://github.com/opencv/opencv_contrib) then clone it as well.

For example
@code{.bash}
cd ~/<my_working_directory>
git clone https://github.com/opencv/opencv.git
@endcode

@note
You may need to install `git` for your development environment.

Building OpenCV-JavaScript from Source Using CMake
---------------------------------------

-#  Create a temporary directory, which we denote as \<cmake_build_dir\>, where you want to put
    the generated Makefiles, project files as well the object files and output files and enter
    there.

    For example
    @code{.bash}
    cd ~/opencv
    mkdir build_js
    cd build_js
    @endcode
-#  Configuring. Run cmake [\<some optional parameters\>] \<path to the OpenCV source directory\>
	To build OpenCV-JavaScript, you need to append `-D CMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake`.

    For example
    @code{.bash}
    cmake -D CMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
    @endcode

    @note
    You may need to install `cmake` for your development environment.

    @note
    Use `cmake -DCMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..` , without spaces after -D if the above example doesn't work.

-#  Build. From build directory execute *make*, it is recommended to do this in several threads

    For example
    @code{.bash}
    make -j7 # runs 7 jobs in parallel
    @endcode

    The `opencv.js` under \<cmake_build_dir\>/bin folder is the final produce which you can include into your web pages.

-#  [optional] Building documents. Run make with target "doxygen"

    For example
    @code{.bash}
    make -j7 doxygen
    @endcode

    The built documents are located at \<cmake_build_dir\>/doc/doxygen/html folder.

    @note
    You may need to install `doxygen` tool for your development environment.

-#  [optional] Running tests

	Run a local web server in \<cmake_build_dir\>/bin folder. For example, node http-server which serves on `localhost:8080`.

	Launch the web browser to URL `http://localhost:8000/tests.html` which runs the unit tests automatically.

    You can also run tests by Node.js
	@code{.sh}
	cd bin
	npm install
	node tests.js
	@endcode

	@node
	You may need to install `node` for your development environment.
