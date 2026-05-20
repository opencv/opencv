# Installation in Linux

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

:::

(tutorial_linux_install_quick_start)=
## Quick start
(tutorial_linux_install_quick_build_core)=
### Build core modules

```{doxysnippet} linux_quick_install.sh
:tag: body
:language: bash
```

(tutorial_linux_install_quick_build_contrib)=
### Build with opencv_contrib

```{doxysnippet} linux_quick_install_contrib.sh
:tag: body
:language: bash
```

(tutorial_linux_install_detailed)=
## Detailed process
This section provides more details of the build process and describes alternative methods and tools. Please refer to the [OpenCV installation overview](general_install.md) tutorial for general installation details and to the [OpenCV configuration options reference](config_reference.md) for configuration options documentation.

(tutorial_linux_install_detailed_basic_compiler)=
### Install compiler and build tools
- To compile OpenCV you will need a C++ compiler. Usually it is G++/GCC or Clang/LLVM:
    - Install GCC...

      ```{doxysnippet} linux_install_a.sh
      :tag: gcc
      :language: bash
      ```

    - ... or Clang:

  ```{doxysnippet} linux_install_b.sh
  :tag: clang
  :language: bash
  ```

- OpenCV uses CMake build configuration tool:

  ```{doxysnippet} linux_install_a.sh
  :tag: cmake
  :language: bash
  ```

- CMake can generate scripts for different build systems, e.g. _make_, _ninja_:

    - Install Make...

      ```{doxysnippet} linux_install_a.sh
      :tag: make
      :language: bash
      ```

    - ... or Ninja:

  ```{doxysnippet} linux_install_b.sh
  :tag: ninja
  :language: bash
  ```

- Install tool for getting and unpacking sources:

    - _wget_ and _unzip_...

      ```{doxysnippet} linux_install_a.sh
      :tag: wget
      :language: bash
      ```

    - ... or _git_:

```{doxysnippet} linux_install_b.sh
:tag: git
:language: bash
```

(tutorial_linux_install_detailed_basic_download)=
### Download sources
There are two methods of getting OpenCV sources:

- Download snapshot of repository using web browser or any download tool (~80-90Mb) and unpack it...

  ```{doxysnippet} linux_install_a.sh
  :tag: download
  :language: bash
  ```

- ... or clone repository to local machine using _git_ to get full change history (>470Mb):

```{doxysnippet} linux_install_b.sh
:tag: download
:language: bash
```

:::{note}
Snapshots of other branches, releases or commits can be found on the [GitHub](https://github.com/opencv/opencv) and the [official download page](https://opencv.org/releases).
:::
(tutorial_linux_install_detailed_basic_build)=
### Configure and build
- Create build directory:

  ```{doxysnippet} linux_install_a.sh
  :tag: prepare
  :language: bash
  ```

- Configure - generate build scripts for the preferred build system:
    - For _make_...

      ```{doxysnippet} linux_install_a.sh
      :tag: configure
      :language: bash
      ```

    - ... or for _ninja_:

  ```{doxysnippet} linux_install_b.sh
  :tag: configure
  :language: bash
  ```

- Build - run actual compilation process:
    - Using _make_...

      ```{doxysnippet} linux_install_a.sh
      :tag: build
      :language: bash
      ```

    - ... or _ninja_:

```{doxysnippet} linux_install_b.sh
:tag: build
:language: bash
```

:::{note}
_Configure_ process can download some files from the internet to satisfy library dependencies, connection failures can cause some of modules or functionalities to be turned off or behave differently. Refer to the [OpenCV installation overview](general_install.md) and [OpenCV configuration options reference](config_reference.md) tutorials for details and full configuration options reference.
If you experience problems with the build process, try to clean or recreate the build directory. Changes in the configuration like disabling a dependency, modifying build scripts or switching sources to another branch are not handled very well and can result in broken workspace.
_Make_ can run multiple compilation processes in parallel, `-j<NUM>` option means "run <NUM> jobs simultaneously". _Ninja_ will automatically detect number of available processor cores and does not need `-j` option.
:::
(tutorial_linux_install_detailed_basic_verify)=
### Check build results
After successful build you will find libraries in the `build/lib` directory and executables (test, samples, apps) in the `build/bin` directory:

```{doxysnippet} linux_install_a.sh
:tag: check
:language: bash
```

CMake package files will be located in the build root:

```{doxysnippet} linux_install_a.sh
:tag: check cmake
:language: bash
```

### Install

:::{warning}
The installation process only copies files to predefined locations and does minor patching. Installing using this method does not integrate opencv into the system package registry and thus, for example, opencv can not be uninstalled automatically. We do not recommend system-wide installation to regular users due to possible conflicts with system packages.
:::
By default OpenCV will be installed to the `/usr/local` directory, all files will be copied to following locations:
* `/usr/local/bin` - executable files
* `/usr/local/lib` - libraries (.so)
* `/usr/local/cmake/opencv5` - cmake package
* `/usr/local/include/opencv5` - headers
* `/usr/local/share/opencv5` - other files (e.g. trained cascades in XML format)

Since `/usr/local` is owned by the root user, the installation should be performed with elevated privileges (`sudo`):

```{doxysnippet} linux_install_a.sh
:tag: install
:language: bash
```

or

```{doxysnippet} linux_install_b.sh
:tag: install
:language: bash
```

Installation root directory can be changed with `CMAKE_INSTALL_PREFIX` configuration parameter, e.g. `-DCMAKE_INSTALL_PREFIX=$HOME/.local` to install to current user's local directory. Installation layout can be changed with `OPENCV_*_INSTALL_PATH` parameters. See [OpenCV configuration options reference](config_reference.md) for details.
