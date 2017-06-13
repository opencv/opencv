Install OpenCV-JavaScript in Ubuntu {#tutorial_js_setup_in_ubuntu}
===============================


Installing Dependencies
-----------------------------

First we will install some dependencies.

### Emscripten
Install emscripten. You can obtain emscripten by using [Emscripten SDK](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).
@code
./emsdk update
./emsdk install sdk-master-64bit --shallow
./emsdk activate sdk-master-64bit
source ./emsdk_env.sh
@endcode
Patch Emscripten & Rebuild. //todo
@code
patch -p1 < PATH/TO/patch_emscripten_master.diff
@endcode
Rebuild emscripten
@code
./emsdk install sdk-master-64bit --shallow
@endcode

### Node.js
//todo

### Downloading OpenCV
You can download the latest release of OpenCV from [sourceforge
site](http://sourceforge.net/projects/opencvlibrary/). Then extract the folder.

Or you can download latest source from OpenCV's github repo. (If you want to contribute to OpenCV,
choose this. It always keeps your OpenCV up-to-date). For that, you need to install **Git** first.
@code{.sh}
apt-get install git
git clone https://github.com/opencv/opencv.git
@endcode
It will create a folder OpenCV in home directory (or the directory you specify). The cloning may
take some time depending upon your internet connection.

Now open a terminal window and navigate to the downloaded OpenCV folder. Create a new build folder
and navigate to it.
@code{.sh}
mkdir build_js
cd build_js
@endcode
### Configuring and Building
Build
@code{.sh}
cd opencv
mkdir build_js
cd build_js
cmake -D CMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j7
@endcode
Test in browser
@code{.sh}
cd bin
python server.py
@endcode

Launch browser to `http://localhost:8000`.

Tests include:
* tests.html
* features_2d.html
* img_proc.html
* face_detect.html

Test in Node.js
@code{.sh}
cd bin
npm install
node tests.js
@endcode
