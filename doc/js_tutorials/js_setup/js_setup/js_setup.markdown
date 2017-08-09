Build OpenCV.js {#tutorial_js_setup}
===============================


Installing Emscripten
-----------------------------

[Emscripten](https://github.com/kripken/emscripten) is an LLVM-to-JavaScript compiler. We will use Emscripten to build OpenCV.js.

To Install Emscripten, follow instructions on [Emscripten SDK](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).

For example
@code{.bash}
./emsdk update
./emsdk install latest
./emsdk activate latest
@endcode

@note
To compile to [WebAssembly](http://webassembly.org), you also need to install and activate [Binaryen](https://github.com/WebAssembly/binaryen) with the `emsdk` command. Please refer to [Developer's Guide](http://webassembly.org/getting-started/developers-guide/) for more details.

After install, ensure the `EMSCRIPTEN` environment is setup correctly.

For example:
@code{.bash}
source ./emsdk_env.sh
echo ${EMSCRIPTEN}
@endcode

Obtaining OpenCV Source Code
--------------------------

You can use the latest stable OpenCV version or you can grab the latest snapshot from our [Git
repository](https://github.com/opencv/opencv.git).

### Obtaining the Latest Stable OpenCV Version

-   Go to our [downloads page](http://opencv.org/downloads.html).
-   Download the source archive and unpack it.

### Obtaining the Cutting-edge OpenCV from the Git Repository

Launch Git client and clone [OpenCV repository](http://github.com/opencv/opencv).

For example:
@code{.bash}
cd ~/<my_working_directory>
git clone https://github.com/opencv/opencv.git
@endcode

@note
You may need to install `git` for your development environment.

Building OpenCV.js from Source Using CMake
---------------------------------------

-#  Create and open a temporary directory \<cmake_build_dir\> (`build_js` in this example), put the generated Makefiles, project files, and output files.

    For example:
    @code{.bash}
    cd ~/opencv
    mkdir build_js
    cd build_js
    @endcode

-#  To configure, run cmake [\<some optional parameters\>] \<path to the OpenCV source directory\>
	To build OpenCV.js, you need to append `-D CMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake`.

    For example:
    @code{.bash}
    cmake -D CMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
    @endcode

    @note
    You may need to install `cmake` for your development environment.

    @note
    Use `cmake -DCMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..` , without spaces after -D if the above example doesn't work.

    @note
    Pass `-DCMAKE_C_FLAGS="-s WASM=1"` and `-DCMAKE_CXX_FLAGS="-s WASM=1"` to `cmake` if you are targeting WebAssembly.

-#  To build, execute *make*, from the build directory. it is recommended to do this in several threads.

    For example:
    @code{.bash}
    make -j7 # runs 7 jobs in parallel
    @endcode

    The `opencv.js` found \<cmake_build_dir\>/bin folder is the final product to include into your web pages.

-#  [optional] To build documents, run make with target "doxygen"

    For example:
    @code{.bash}
    make -j7 doxygen
    @endcode

    The built documents are located in the \<cmake_build_dir\>/doc/doxygen/html folder.

    @note
    You may need to install `doxygen` tool for your development environment.

-#  [optional] o run a test, run a local web server in \<cmake_build_dir\>/bin folder. For example, node http-server which serves on `localhost:8080`.

    Navigate the web browser to `http://localhost:8000/tests.html`, which runs the unit tests automatically.

    You can also run tests using Node.js.

    For example:
	@code{.sh}
	cd bin
	npm install
	node tests.js
	@endcode

	@note
	You may need to install `node` for your development environment.
