Building OpenCV with oneAPI {#tutorial_oneapi_install}
===========================


@prev_tutorial{tutorial_linux_install}
@next_tutorial{tutorial_linux_gcc_cmake}

|    |    |
| -: | :- |
| Original author | Alessandro de Oliveira Faria |
| Compatibility | OpenCV >= 4.11.0 |

@tableofcontents

# Quick start {#tutorial_oneapi_install_quick_start}

**oneAPI** is Intel's open initiative (now also maintained by the UXL Foundation) that combines a specification and a set of toolkits for programming CPUs, GPUs, FPGAs and NPUs with a single code base. The core is the SYCL standard (single-source C++ for parallelism), complemented by high-performance libraries — oneTBB (parallelism), oneMKL (linear algebra), oneDNN (neural networks), oneVPL (video), etc. Thus, when you compile with oneAPI's DPC++ (icpx) compiler, the binary gains optimized execution paths that choose, at runtime, the best vector instructions or the available device, without changing the source code.

## Why compile OpenCV with the oneAPI ecosystem when targeting the CPU:

* Simple, because by enabling the CMake options -DWITH_SYCL=ON -DWITH_TBB=ON -DWITH_ONEDNN=ON -DWITH_IPP=ON and using the icpx compiler, the OpenCV core starts to directly invoke oneAPI libraries.
* oneDNN replaces the generic kernels of the cv::dnn layer with implementations that exploit AVX2, AVX-512, AMX and VNNI, accelerating convolutions, matmul and network post-processing by up to 3-5× on modern CPUs.
* oneTBB takes over the thread pool, scheduling filters like cv::resize, cv::GaussianBlur or the G-API pipeline across all cores without busy-wait.
* IPP (now distributed via oneAPI Base Toolkit) provides optimized intrinsic routines for elementary operations (SAD, DFT, median blur), which OpenCV calls when it encounters the HAVE_IPP macro.
* All this happens transparently: the source code that uses cv::Mat remains the same, but the linked symbols point to vectorized versions, and the internal dispatcher selects the appropriate vector width at runtime.


## CPU Processor Requirements

Systems based on Intel® 64 architectures below are supported both as host and target platforms.

* Intel® Core™ processor family or higher
* Intel® Xeon® processor family
* Intel® Xeon® Scalable processor family


### Requirements for Accelerators

* Integrated GEN9 (and higher) GPUs. See source in Intel® Graphics Compiler for OpenCL™
* FPGA Card: see Intel(R) DPC++ Compiler System Requirements.

### Disk Space Requirements

* 3.3 GB of disk space (minimum) on a standard installation.

@note: During the installation process, the installer may need up to 6 GB of additional temporary disk storage to manage the download and intermediate installation files.


### Memory Requirements

* 8 GB RAM recommended


## How To install oneAPI

Installing oneAPI: To quickly set up the oneAPI ecosystem on openSUSE, simply follow the official guide https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html, which shows you how to enable the distribution’s dedicated repository (zypper ar … oneAPI) and install the metapackages ― for example, intel-basekit (DPC++, TBB, oneDNN, IPP compilers) and, optionally, intel-hpckit or intel-renderkit if you need HPC or graphics tools. The guide also explains post-installation tweaks, such as loading the environment with source /opt/intel/oneapi/setvars.sh , ensuring that the binaries (icpx, dpcpp) and libraries are immediately available in your shell for compiling and running accelerated applications.

## Download, Github Instruction, Build and Install

1. Below are the commands to download last version (latest release on the date of publication of this text):

```
git clone https://github.com/opencv/opencv.git
```

2. and make sure you are using branch 4.*:

```
git status
On branch 4.x
```

3. Navigate to OpenCV repository and prepare the build folder:

```
cd opencv
mkdir build
cd build
```

4. Set up Intel oneAPI environment variables. For default installation:

```
source /opt/intel/oneapi/setvars.sh
```

5. Run CMake * with Intel® oneAPI DPC++/C++ Compiler to configure the project:

```
 cmake -DCMAKE_C_COMPILER=icx \
       -DCMAKE_CXX_COMPILER=icpx
       -DCMAKE_CXX_FLAGS="-march=native -mavx -mfma -msse -msse2" ..
 cmake --build .
```
6. Now Make sure openCV* is compiled with Intel® oneAPI DPC++/C++ Compiler and install:

```
readelf -p .comment bin/opencv_annotation
String dump of section '.comment':
  [     0]  GCC: (SUSE Linux) 13.3.1 20250313 [revision 4ef1d8c84faeebffeb0cc01ee22e891b41e5c4e0]
  [    56]  GCC: (SUSE Linux) 12.3.0
  [    6f]  Intel(R) oneAPI DPC++/C++ Compiler 2025.1.1 (2025.1.1.20250418)
make install
```

Have fun...
