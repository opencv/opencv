# How to enable Halide backend for improve efficiency  {#tutorial_dnn_halide}

## Introduction
This tutorial guidelines how to run your models in OpenCV deep learning module
using Halide language backend. Halide is an open-source project that let us
write image processing algorithms in well-readable format, schedule computations
according to specific device and evaluate it with a quite good efficiency.

An official website of the Halide project: http://halide-lang.org/.

## Efficiency comparison
Measured on Intel&reg; Core&trade; i7-6700K CPU @ 4.00GHz x 8.

Single image forward pass (in milliseconds):

|     Architecture | MKL backend | Halide backend | Speed Up ratio |
|-----------------:|------------:|---------------:|---------------:|
|          AlexNet |       16.55 |          22.38 |          x0.73 |
|        ResNet-50 |       63.69 |          73.91 |          x0.86 |
|  SqueezeNet v1.1 |       10.11 |           8.21 |          x1.23 |
|     Inception-5h |       35.38 |          37.06 |          x0.95 |
| ENet @ 3x512x256 |       82.26 |          41.21 |          x1.99 |

Scheduling directives might be found @ [opencv_extra/testdata/dnn](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn).

## Requirements
### LLVM compiler

@note LLVM compilation might take a long time.

- Download LLVM source code from http://releases.llvm.org/4.0.0/llvm-4.0.0.src.tar.xz.
Unpack it. Let **llvm_root** is a root directory of source code.

- Create directory **llvm_root**/tools/clang

- Download Clang with the same version as LLVM. In our case it will be from
http://releases.llvm.org/4.0.0/cfe-4.0.0.src.tar.xz. Unpack it into
**llvm_root**/tools/clang. Note that it should be a root for Clang source code.

- Build LLVM on Linux
@code
cd llvm_root
mkdir build && cd build
cmake -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j4
@endcode

- Build LLVM on Windows (Developer Command Prompt)
@code
mkdir \\path-to-llvm-build\\ && cd \\path-to-llvm-build\\
cmake.exe -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=\\path-to-llvm-install\\ -G "Visual Studio 14 Win64" \\path-to-llvm-src\\
MSBuild.exe /m:4 /t:Build /p:Configuration=Release .\\INSTALL.vcxproj
@endcode

@note `\\path-to-llvm-build\\` and `\\path-to-llvm-install\\` are different directories.

### Halide language.

- Download source code from GitHub repository, https://github.com/halide/Halide
or using git. The root directory will be a **halide_root**.
@code
git clone https://github.com/halide/Halide.git
@endcode

- Build Halide on Linux
@code
cd halide_root
mkdir build && cd build
cmake -DLLVM_DIR=llvm_root/build/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_VERSION=40 -DWITH_TESTS=OFF -DWITH_APPS=OFF -DWITH_TUTORIALS=OFF ..
make -j4
@endcode

- Build Halide on Windows (Developer Command Prompt)
@code
cd halide_root
mkdir build && cd build
cmake.exe -DLLVM_DIR=\\path-to-llvm-install\\lib\\cmake\\llvm -DLLVM_VERSION=40 -DWITH_TESTS=OFF -DWITH_APPS=OFF -DWITH_TUTORIALS=OFF -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 14 Win64" ..
MSBuild.exe /m:4 /t:Build /p:Configuration=Release .\\ALL_BUILD.vcxproj
@endcode

## Build OpenCV with Halide backend
When you build OpenCV add the following configuration flags:

- `WITH_HALIDE` - enable Halide linkage

- `HALIDE_ROOT_DIR` - path to Halide build directory

## Sample

@include dnn/squeezenet_halide.cpp

## Explanation
Download Caffe model from SqueezeNet repository: [train_val.prototxt](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt) and [squeezenet_v1.1.caffemodel](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel).

Also you need file with names of [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/browse-synsets) classes:
[synset_words.txt](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/synset_words.txt).

Put these files into working dir of this program example.

-# Read and initialize network using path to .prototxt and .caffemodel files
@snippet dnn/squeezenet_halide.cpp Read and initialize network

-# Check that network was read successfully
@snippet dnn/squeezenet_halide.cpp Check that network was read successfully

-# Read input image and convert to the 4-dimensional blob, acceptable by SqueezeNet v1.1
@snippet dnn/squeezenet_halide.cpp Prepare blob

-# Pass the blob to the network
@snippet dnn/squeezenet_halide.cpp Set input blob

-# Enable Halide backend for layers where it is implemented
@snippet dnn/squeezenet_halide.cpp Enable Halide backend

-# Make forward pass
@snippet dnn/squeezenet_halide.cpp Make forward pass
Remember that the first forward pass after initialization require quite more
time that the next ones. It's because of runtime compilation of Halide pipelines
at the first invocation.

-# Determine the best class
@snippet dnn/squeezenet_halide.cpp Determine the best class

-# Print results
@snippet dnn/squeezenet_halide.cpp Print results
For our image we get:

> Best class: #812 'space shuttle'
>
> Probability: 97.9812%
