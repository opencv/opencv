# How to enable Halide backend {#tutorial_dnn_halide}

@tableofcontents

@prev_tutorial{tutorial_dnn_googlenet}
@next_tutorial{tutorial_dnn_halide_scheduling}

|    |    |
| -: | :- |
| Original author | Dmitry Kurtaev |
| Compatibility | OpenCV >= 3.3 |

## Introduction
This tutorial guidelines how to run your models in OpenCV deep learning module
using Halide language backend. Halide is an open-source project that let us
write image processing algorithms in well-readable format, schedule computations
according to specific device and evaluate it with a quite good efficiency.

An official website of the Halide project: http://halide-lang.org/.

An up to date efficiency comparison: https://github.com/opencv/opencv/wiki/DNN-Efficiency

## Requirements

Download Halide [binary release](https://github.com/halide/Halide/releases) or build from source.
Actual instructions can be always found at the [official documentation](https://github.com/halide/Halide#building-halide-with-cmake).

### LLVM compiler

@note LLVM compilation might take a long time.

- Download LLVM source code using git
@code
git clone --depth 1 --branch llvmorg-13.0.0 https://github.com/llvm/llvm-project.git
@endcode

- Build LLVM on Linux
@code
cmake -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF \
      -S llvm-project/llvm -B llvm-build

cd llvm-build && make -j$(nproc --all)
cmake --install llvm-build --prefix llvm-install 
@endcode

- Build LLVM on Windows (Developer Command Prompt)
@code
mkdir \\path-to-llvm-build\\ && cd \\path-to-llvm-build\\
cmake.exe -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=\\path-to-llvm-install\\ -G "Visual Studio 14 Win64" \\path-to-llvm-src\\
MSBuild.exe /m:4 /t:Build /p:Configuration=Release .\\INSTALL.vcxproj
@endcode

@note `\\path-to-llvm-build\\` and `\\path-to-llvm-install\\` are different directories.

```bash
```

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

## Set Halide as a preferable backend
@code
net.setPreferableBackend(DNN_BACKEND_HALIDE);
@endcode
