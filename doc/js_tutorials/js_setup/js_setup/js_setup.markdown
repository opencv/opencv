Build OpenCV.js {#tutorial_js_setup}
===============================

@note
You don't have to build your own copy if you simply want to start using it. Refer the Using Opencv.js tutorial for steps on getting a prebuilt copy from our releases or online documentation.

Installing Emscripten
-----------------------------

[Emscripten](https://github.com/emscripten-core/emscripten) is an LLVM-to-JavaScript compiler. We will use Emscripten to build OpenCV.js.

@note
While this describes installation of required tools from scratch, there's a section below also describing an alternative procedure to perform the same build using docker containers which is often easier.

To Install Emscripten, follow instructions of [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html).

For example:
@code{.bash}
./emsdk update
./emsdk install latest
./emsdk activate latest
@endcode


After install, ensure the `EMSDK` environment is setup correctly.

For example:
@code{.bash}
source ./emsdk_env.sh
echo ${EMSDK}
@endcode

Modern versions of Emscripten requires to use `emcmake` / `emmake` launchers:

@code{.bash}
emcmake sh -c 'echo ${EMSCRIPTEN}'
@endcode


The version 2.0.10 of emscripten is verified for latest WebAssembly. Please check the version of Emscripten to use the newest features of WebAssembly.

For example:
@code{.bash}
./emsdk update
./emsdk install 2.0.10
./emsdk activate 2.0.10
@endcode

Obtaining OpenCV Source Code
--------------------------

You can use the latest stable OpenCV version or you can grab the latest snapshot from our [Git
repository](https://github.com/opencv/opencv.git).

### Obtaining the Latest Stable OpenCV Version

-   Go to our [releases page](https://opencv.org/releases).
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
    emcmake python ./opencv/platforms/js/build_js.py build_js
    @endcode

    @note
    It requires `python` and `cmake` installed in your development environment.

-#  The build script builds asm.js version by default. To build WebAssembly version, append `--build_wasm` switch.
    By default everything is bundled into one JavaScript file by `base64` encoding the WebAssembly code. For production
    builds you can add `--disable_single_file` which will reduce total size by writing the WebAssembly code
    to a dedicated `.wasm` file which the generated JavaScript file will automatically load.

    For example, to build wasm version in `build_wasm` directory:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_wasm --build_wasm
    @endcode

-#  [Optional] To build the OpenCV.js loader, append `--build_loader`.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --build_loader
    @endcode

    @note
    The loader is implemented as a js file in the path `<opencv_js_dir>/bin/loader.js`. The loader utilizes the [WebAssembly Feature Detection](https://github.com/GoogleChromeLabs/wasm-feature-detect) to detect the features of the browser and load corresponding OpenCV.js automatically. To use it, you need to use the UMD version of [WebAssembly Feature Detection](https://github.com/GoogleChromeLabs/wasm-feature-detect) and introduce the `loader.js` in your Web application.

    Example Code:
    @code{.javascript}
    // Set paths configuration
    let pathsConfig = {
        wasm: "../../build_wasm/opencv.js",
        threads: "../../build_mt/opencv.js",
        simd: "../../build_simd/opencv.js",
        threadsSimd: "../../build_mtSIMD/opencv.js",
    }

    // Load OpenCV.js and use the pathsConfiguration and main function as the params.
    loadOpenCV(pathsConfig, main);
    @endcode


-#  [optional] To build documents, append `--build_doc` option.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --build_doc
    @endcode

    @note
    It requires `doxygen` installed in your development environment.

-#  [optional] To build tests, append `--build_test` option.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --build_test
    @endcode

-#  [optional] To enable OpenCV contrib modules append `--cmake_option="-DOPENCV_EXTRA_MODULES_PATH=/path/to/opencv_contrib/modules/"`

    For example:
    @code{.bash}
    emcmake python ./platforms/js/build_js.py build_js --cmake_option="-DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules"
    @endcode

-#  [optional] To enable WebNN backend, append `--webnn` option.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --webnn
    @endcode

Running OpenCV.js Tests
---------------------------------------

Remember to launch the build command passing `--build_test` as mentioned previously. This will generate test source code ready to run together with `opencv.js` file in `build_js/bin`

### Manually in your browser

To run tests, launch a local web server in `\<build_dir\>/bin` folder. For example, node http-server which serves on `localhost:8080`.

Navigate the web browser to `http://localhost:8080/tests.html`, which runs the unit tests automatically. Command example:

@code{.sh}
npx http-server build_js/bin
firefox http://localhost:8080/tests.html
@endcode

@note
This snippet and the following require [Node.js](https://nodejs.org) to be installed.

### Headless with Puppeteer

Alternatively tests can run with [GoogleChrome/puppeteer](https://github.com/GoogleChrome/puppeteer#readme) which is a version of Google Chrome that runs in the terminal (useful for Continuous integration like travis CI, etc)

@code{.sh}
cd build_js/bin
npm install
npm install --no-save puppeteer    # automatically downloads Chromium package
node run_puppeteer.js
@endcode

@note
Checkout `node run_puppeteer --help` for more options to debug and reporting.

@note
The command `npm install` only needs to be executed once, since installs the tools dependencies; after that they are ready to use.

@note
Use `PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=1 npm install --no-save puppeteer` to skip automatic downloading of Chromium.
You may specify own Chromium/Chrome binary through `PUPPETEER_EXECUTABLE_PATH=$(which google-chrome)` environment variable.
**BEWARE**: Puppeteer is only guaranteed to work with the bundled Chromium, use at your own risk.


### Using Node.js.

For example:

@code{.sh}
cd build_js/bin
npm install
node tests.js
@endcode

@note If all tests are failed, then consider using Node.js from 8.x version (`lts/carbon` from `nvm`).


-#  [optional] To build `opencv.js` with threads optimization, append `--threads` option.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --build_wasm --threads
    @endcode

    The default threads number is the logic core number of your device. You can use `cv.parallel_pthreads_set_threads_num(number)` to set threads number by yourself and use `cv.parallel_pthreads_get_threads_num()` to get the current threads number.

    @note
    You should build wasm version of `opencv.js` if you want to enable this optimization. And the threads optimization only works in browser, not in node.js. You need to enable the `WebAssembly threads support` feature first with your browser. For example, if you use Chrome, please enable this flag in chrome://flags.

-#  [optional] To build `opencv.js` with wasm simd optimization, append `--simd` option.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --build_wasm --simd
    @endcode

    The simd optimization is experimental as wasm simd is still in development.

    @note
    Now only emscripten LLVM upstream backend supports wasm simd, referring to https://emscripten.org/docs/porting/simd.html. So you need to setup upstream backend environment with the following command first:
    @code{.bash}
    ./emsdk update
    ./emsdk install latest-upstream
    ./emsdk activate latest-upstream
    source ./emsdk_env.sh
    @endcode

    @note
    You should build wasm version of `opencv.js` if you want to enable this optimization. For browser, you need to enable the `WebAssembly SIMD support` feature first. For example, if you use Chrome, please enable this flag in chrome://flags. For Node.js, you need to run script with flag `--experimental-wasm-simd`.

    @note
    The simd version of `opencv.js` built by latest LLVM upstream may not work with the stable browser or old version of Node.js. Please use the latest version of unstable browser or Node.js to get new features, like `Chrome Dev`.

-#  [optional] To build wasm intrinsics tests, append `--build_wasm_intrin_test` option.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --build_wasm --simd --build_wasm_intrin_test
    @endcode

    For wasm intrinsics tests, you can use the following function to test all the cases:
    @code{.js}
    cv.test_hal_intrin_all()
    @endcode

    And the failed cases will be logged in the JavaScript debug console.

    If you only want to test single data type of wasm intrinsics, you can use the following functions:
    @code{.js}
    cv.test_hal_intrin_uint8()
    cv.test_hal_intrin_int8()
    cv.test_hal_intrin_uint16()
    cv.test_hal_intrin_int16()
    cv.test_hal_intrin_uint32()
    cv.test_hal_intrin_int32()
    cv.test_hal_intrin_uint64()
    cv.test_hal_intrin_int64()
    cv.test_hal_intrin_float32()
    cv.test_hal_intrin_float64()
    @endcode

-#  [optional] To build performance tests, append `--build_perf` option.

    For example:
    @code{.bash}
    emcmake python ./opencv/platforms/js/build_js.py build_js --build_perf
    @endcode

    To run performance tests, launch a local web server in \<build_dir\>/bin folder. For example, node http-server which serves on `localhost:8080`.

    There are some kernels now in the performance test like `cvtColor`, `resize` and `threshold`. For example, if you want to test `threshold`, please navigate the web browser to `http://localhost:8080/perf/perf_imgproc/perf_threshold.html`. You need to input the test parameter like `(1920x1080, CV_8UC1, THRESH_BINARY)`, and then click the `Run` button to run the case. And if you don't input the parameter, it will run all the cases of this kernel.

    You can also run tests using Node.js.

    For example, run `threshold` with parameter `(1920x1080, CV_8UC1, THRESH_BINARY)`:
    @code{.sh}
    cd bin/perf
    npm install
    node perf_threshold.js --test_param_filter="(1920x1080, CV_8UC1, THRESH_BINARY)"
    @endcode

Building OpenCV.js with Docker
---------------------------------------

Alternatively, the same build can be can be accomplished using [docker](https://www.docker.com/) containers which is often easier and more reliable, particularly in non linux systems. You only need to install [docker](https://www.docker.com/) on your system and use a popular container that provides a clean well tested environment for emscripten builds like this, that already has latest versions of all the necessary tools installed.

So, make sure [docker](https://www.docker.com/) is installed in your system and running. The following shell script should work in Linux and MacOS:

@code{.bash}
git clone https://github.com/opencv/opencv.git
cd opencv
docker run --rm -v $(pwd):/src -u $(id -u):$(id -g) emscripten/emsdk emcmake python3 ./platforms/js/build_js.py build_js
@endcode

In Windows use the following PowerShell command:

@code{.bash}
docker run --rm --workdir /src -v "$(get-location):/src" "emscripten/emsdk" emcmake python3 ./platforms/js/build_js.py build_js
@endcode

@warning
The example uses latest version of emscripten. If the build fails you should try a version that is known to work fine which is `2.0.10` using the following command:

@code{.bash}
docker run --rm -v $(pwd):/src -u $(id -u):$(id -g) emscripten/emsdk:2.0.10 emcmake python3 ./platforms/js/build_js.py build_js
@endcode

In Windows use the following PowerShell command:

@code{.bash}
docker run --rm --workdir /src -v "$(get-location):/src" "emscripten/emsdk:2.0.10" emcmake python3 ./platforms/js/build_js.py build_js
@endcode

### Building the documentation with Docker

To build the documentation `doxygen` needs to be installed. Create a file named `Dockerfile` with the following content:

```
FROM emscripten/emsdk:2.0.10

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends doxygen \
  && rm -rf /var/lib/apt/lists/*
```

Then we build the docker image and name it `opencv-js-doc` with the following command (that needs to be run only once):

@code{.bash}
docker build . -t opencv-js-doc
@endcode

Now run the build command again, this time using the new image and passing `--build_doc`:

@code{.bash}
docker run --rm -v $(pwd):/src -u $(id -u):$(id -g) "opencv-js-doc" emcmake python3 ./platforms/js/build_js.py build_js --build_doc
@endcode
