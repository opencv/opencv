# How to enable Halide backend for improve efficiency  {#tutorial_dnn_halide}

@prev_tutorial{tutorial_dnn_googlenet}
@next_tutorial{tutorial_dnn_halide_scheduling}

## Introduction
This tutorial guidelines how to run your models in OpenCV deep learning module
using Halide language backend. Halide is an open-source project that let us
write image processing algorithms in well-readable format, schedule computations
according to specific device and evaluate it with a quite good efficiency.

An official website of the Halide project: http://halide-lang.org/.

An up to date efficiency comparison: https://github.com/opencv/opencv/wiki/DNN-Efficiency

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

- `ENABLE_CXX11` - enable C++11 standard

- `WITH_HALIDE` - enable Halide linkage

- `HALIDE_ROOT_DIR` - path to Halide build directory

## Set Halide as a preferable backend
@code
net.setPreferableBackend(DNN_BACKEND_HALIDE);
@endcode
