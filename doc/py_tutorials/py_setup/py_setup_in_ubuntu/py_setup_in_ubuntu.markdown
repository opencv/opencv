Install OpenCV-Python in Ubuntu {#tutorial_py_setup_in_ubuntu}
===============================

Goals
-----

In this tutorial We will learn to setup OpenCV-Python in Ubuntu System. <br>
Below steps are tested for Ubuntu 16.04 (64-bit) and Ubuntu 14.04 (32-bit).<br>

OpenCV-Python can be installed in Ubuntu in two ways:
- Install from pre-built binaries available in Ubuntu repositories<br>
- Compile from the source. In this section, we will see both.

Another important thing is the additional libraries required. OpenCV-Python requires only **Numpy** (in addition to other dependencies, which we will see later). But in this tutorials, we also use **Matplotlib** for some easy and nice plotting purposes (which I feel much better compared to OpenCV). Matplotlib is optional, but highly recommended. Similarly we will also see **IPython**, an Interactive Python Terminal, which is also highly recommended.<br>

Installing OpenCV-Python from Pre-built Binaries<br>
------------------------------------------------

Install all packages with following command in terminal as root.<br>
**This method serves best when using just for programming and developing OpenCV applications.**<br>
```
$ sudo apt-get install python-opencv
```


Open Python IDLE (or IPython) and type following codes in Python terminal.
```
import cv2
print( cv2.__version__ )
```
If the results are printed out without any errors, congratulations !!! You have installed
OpenCV-Python successfully.

It is quite easy. But there is a problem with this. Apt repositories may not contain the latest version of OpenCV always. For example, at the time of writing this tutorial, apt repository contains 2.4.8 while latest OpenCV version is 3.x With respect to Python API, latest version will always contain much better support. Also, there may be chance of problems with camera support, video playback etc depending upon the drivers, ffmpeg, gstreamer packages present etc.<br>

So for getting latest source codes preference is next method, i.e. compiling from source. Also at some point in time,
if you want to contribute to OpenCV, you will need this.

Building OpenCV from source
-----------------------------

Compiling from source may seem a little complicated at first, but once you succeeded in it, there is
nothing complicated.<br>

First we will install some dependencies. Some are compulsory, some are optional. Optional
dependencies, you can leave if you don't want.<br>

### Required Build Dependencies

We need **CMake** to configure the installation, **GCC** for compilation, **Python-devel** and
**Numpy** for creating Python extensions etc.<br>

```
sudo apt-get install cmake
sudo apt-get install python-devel numpy
sudo apt-get install gcc gcc-c++
```

Next we need **GTK** support for GUI features, Camera support (libdc1394, libv4l), Media Support
(ffmpeg, gstreamer) etc.<br>

```
sudo apt-get install gtk2-devel
sudo apt-get install libdc1394-devel
sudo apt-get install libv4l-devel
sudo apt-get install ffmpeg-devel
sudo apt-get install gstreamer-plugins-base-devel
```
### Optional Dependencies

Above dependencies are sufficient to install OpenCV in your ubuntu machine. But depending upon your
requirements, you may need some extra dependencies. A list of such optional dependencies are given
below. You can either leave it or install it, your call :)

OpenCV comes with supporting files for image formats like PNG, JPEG, JPEG2000, TIFF, WebP etc. But
it may be a little old. If you want to get latest libraries, you can install development files for
these formats.
<br>
```
sudo apt-get install libpng-devel
sudo apt-get install libjpeg-turbo-devel
sudo apt-get install jasper-devel
sudo apt-get install openexr-devel
sudo apt-get install libtiff-devel
sudo apt-get install libwebp-devel
```
Several OpenCV functions are parallelized with **Intel(R) Threading Building Blocks** (TBB). But if
you want to enable it, you need to install TBB first. ( Also while configuring installation with
CMake, don't forget to pass -D WITH_TBB=ON. More details below.)
<br>
` sudo apt-get install tbb-devel`<br>

### Downloading OpenCV

To download the latest source from OpenCV's [GitHub Repository](https://github.com/opencv/opencv). (If you want to contribute to OpenCV choose this. For that, you need to install **Git** first.
<br>
```
sudo apt-get install git
git clone https://github.com/opencv/opencv.git
```
<br>
It will create a folder opencv in current directory. The cloning may take some time depending upon your internet connection.<br>
Now open a terminal window and navigate to the downloaded OpenCV folder. Create a new build folder and navigate to it.

```
mkdir build
cd build
```
<br>

### Configuring and Installing
Now we have installed all the required dependencies, let's install OpenCV. Installation has to be configured with CMake. It specifies which modules are to be installed, installation path, which additional libraries to be used, whether documentation and examples to be compiled etc. 
Below command is normally used for configuration with OpenCV's default parameters (executed from build folder):
```
$ cmake ../
```
OpenCV defaults assume "Release" build type and installation path is "/usr/local".
For additional information about CMake options refer to OpenCV [C++ compilation](): 
Refer to the [OpenCV Python tutorials](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
**cmake output**
```
 -- Python 2:
 -- Interpreter: /usr/bin/python2.7 (ver 2.7.6)
 -- Libraries: /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.6)
 -- numpy: /usr/lib/python2.7/dist-packages/numpy/core/include (ver 1.8.2)
 -- packages path: lib/python2.7/dist-packages
 -- 
 -- Python 3:
 -- Interpreter: /usr/bin/python3.4 (ver 3.4.3)
 -- Libraries: /usr/lib/x86_64-linux-gnu/libpython3.4m.so (ver 3.4.3)
 -- numpy: /usr/lib/python3/dist-packages/numpy/core/include (ver 1.8.2)
 -- packages path: lib/python3.4/dist-packages
 ```
- Enable documentation and disable tests and samples
```
cmake -D BUILD_DOCS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF ..
```
- Disable all GPU related modules.

```
cmake -D WITH_OPENCL=OFF -D WITH_CUDA=OFF
```
 
Now you build the files using make command and install it using make install command.

```
sudo make install
```
Installation is over. All files are installed in /usr/local/ folder.Open a terminal and try import cv2.
```
import cv2
print( cv2.__version__ )
```
