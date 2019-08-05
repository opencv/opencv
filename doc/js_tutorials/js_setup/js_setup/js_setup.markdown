Build OpenCV.js {#tutorial_js_setup}
===============================


Installing Emscripten
-----------------------------

[Emscripten](https://github.com/kripken/emscripten) is an LLVM-to-JavaScript compiler. We will use Emscripten to build OpenCV.js.

@note
While this describes installation of required tools from scratch, there's a section below also describing an alternative procedure to perform the same build using docker containers which is often easier.

To Install Emscripten, follow instructions of [Emscripten SDK](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).

For example:
@code{.bash}
./emsdk update
./emsdk install latest
./emsdk activate latest
@endcode

@note
To compile to [WebAssembly](http://webassembly.org), you need to install and activate [Binaryen](https://github.com/WebAssembly/binaryen) with the `emsdk` command. Please refer to [Developer's Guide](http://webassembly.org/getting-started/developers-guide/) for more details.

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

-   Go to our [releases page](http://opencv.org/releases.html).
-   Download the source archive and unpack it.

### Obtaining the Cutting-edge OpenCV from the Git Repository

Launch Git client and clone [OpenCV repository](http://github.com/opencv/opencv).

For example:
@code{.bash}
git clone https://github.com/opencv/opencv.git
@endcode

@note
It requires `git` installed in your development environment.

Building OpenCV.js from Source
---------------------------------------

-#  To build `opencv.js`, execute python script `<opencv_src_dir>/platforms/js/build_js.py <build_dir>`.

    For example, to build in `build_js` directory:
    @code{.bash}
    cd opencv
    python ./platforms/js/build_js.py build_js
    @endcode

    @note
    It requires `python` and `cmake` installed in your development environment.

-#  The build script builds asm.js version by default. To build WebAssembly version, append `--build_wasm` switch.

    For example, to build wasm version in `build_wasm` directory:
    @code{.bash}
    python ./platforms/js/build_js.py build_wasm --build_wasm
    @endcode

-#  [optional] To build documents, append `--build_doc` option.

    For example:
    @code{.bash}
    python ./platforms/js/build_js.py build_js --build_doc
    @endcode

    @note
    It requires `doxygen` installed in your development environment.

-#  [optional] To build tests, append `--build_test` option.

    For example:
    @code{.bash}
    python ./platforms/js/build_js.py build_js --build_test
    @endcode

    To run tests, launch a local web server in \<build_dir\>/bin folder. For example, node http-server which serves on `localhost:8080`.

    Navigate the web browser to `http://localhost:8080/tests.html`, which runs the unit tests automatically.

    You can also run tests using Node.js.

    For example:
    @code{.sh}
    cd bin
    npm install
    node tests.js
    @endcode

    @note
    It requires `node` installed in your development environment.

Building OpenCV.js with Docker
---------------------------------------

Alternatively, the same build can be can be accomplished using [docker](https://www.docker.com/) containers which is often easier and more reliable, particularly in non linux systems. You only need to install [docker](https://www.docker.com/) on your system and use a popular container that provides a clean well tested environment for emscripten builds like this, that already has latest versions of all the necessary tools installed.

So, make sure [docker](https://www.docker.com/) is installed in your system and running. The following shell script should work in linux and MacOS:

@code{.bash}
git clone https://github.com/opencv/opencv.git
cd opencv
docker run --rm --workdir /code -v "$PWD":/code "trzeci/emscripten:latest" python ./platforms/js/build_js.py build_js
@endcode

In Windows use the following PowerShell command:

@code{.bash}
docker run --rm --workdir /code -v "$(get-location):/code" "trzeci/emscripten:latest" python ./platforms/js/build_js.py build_js
@endcode

@note
The example uses latest version of [trzeci/emscripten](https://hub.docker.com/r/trzeci/emscripten) docker container. At this time, the latest version works fine and is `trzeci/emscripten:sdk-tag-1.38.32-64bit`
