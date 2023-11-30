Cross compilation with Ubuntu/Debian{#tutorial_crosscompile_with_multiarch}
====================================

@prev_tutorial{tutorial_arm_crosscompile_with_cmake}
@next_tutorial{tutorial_building_tegra_cuda}

[TOC]

|    |    |
| -: | :- |
| Original author | Kumataro |
| Compatibility   | Ubuntu >=23.04 |
|^                | OpenCV >=4.8.0 |

@warning
This tutorial can contain obsolete information.

What is "MultiArch"
-------------------
Ubuntu and Debian supports MultiArch.
It allows to install several foreign architecture libraries.

(e.g. aarch64 and/or armv7 binaries are able to installed at x86-64 hosts.)

MultiArch can be used to cross-compile OpenCV library too.

@warning
- Following these steps will make your Linux environment a little dirty.
  If possible, it is better to use VMs.
- This tutorial expects host and target uses same ubuntu version.
   Do not use/mix different versions for external library dependency.
  - OK: Host and Target are 23.04.
  - OK: Host and Target are 23.10.
  - NG: Host is 23.04, and Target is 23.10.
  - NG: Host is 23.10, and Target is 23.04.

Downloading tools
-----------------
Install nessesary tools and toolchains to cross-compile.

- git, cmake, build-essential are required basically.
- ccache is to reduce compilation time(option).
- ninja-build is to reduce compilation time(option).
- libfreetype-dev and libharfbuzz-dev is for example to use external libraries(option).
- crossbuild-essential-armhf is toolchain package for armv7 target.
- crossbuild-essential-arm64 is toolchain package for aarch64 target.

```
sudo apt update -y
sudo apt install -y \
    git \
    cmake \
    build-essential \
    ccache \
    ninja-build \
    libfreetype-dev \
    libharfbuzz-dev \
    crossbuild-essential-armhf \
    crossbuild-essential-arm64
```

Working folder structure
------------------------
In this sample, following working folder structure are used.

```
/home
  + kmtr                    - please replace your account name.
    + work
      + opencv              - source, cloned from github
      + opencv_contrib      - source, cloned from github
      + build4-full         - artifact(for Host), created by cmake
      + build4-full_arm64   - artifact(for aarch64 target), created by cmake
      + build4-full_armhf   - artifact(for armhf target), created by cmake
```

1. Create working folder under your home directory.
2. Clone OpenCV and OpenCV Contrib from repository to work directory.

```
cd ~
mkdir work
cd work
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

Self-compile(for x86-64)
------------------------

Following are to compile for host(x86-64) at host(x86-64).

```
[Host]
cmake -S opencv \
      -B build4-full \
      -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules \
      -GNinja

     cmake --build   build4-full
sudo cmake --install build4-full
sudo ldconfig
```

cmake outputs `Host: Linux X.X.X... x86_64`.

```
-- General configuration for OpenCV 4.8.0-dev =====================================
--   Version control:               4.8.0-xxx-XXXXXXXX
--
--   Extra modules:
--     Location (extra):            /home/kmtr/work/opencv_contrib/modules
--     Version control (extra):     4.8.0-xxx-XXXXXXXX
--
--   Platform:
--     Timestamp:                   yyyy-mm-ddThh:mm:ssZ
--     Host:                        Linux 6.2.0-32-generic x86_64
--     CMake:                       3.25.1
```

Update apt and dpkg settings.
----------------------------
apt and dpkg is package management system using ubuntu and debian.

Following are setup steps to use MultiArch.

### Step1)Add apt source

Execute `sudo apt edit-sources` to add foreign arch libraries at end of file.

ex1) arm64 and armv7 for Ubuntu 23.04

```
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-updates main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-updates universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-updates multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-backports main restricted universe multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-security main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-security universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-security multiverse
```

ex2) arm64 and armv7 for Ubuntu 23.10

```
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-updates main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-updates universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-updates multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-backports main restricted universe multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-security main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-security universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-security multiverse
```

### Step2) Update apt database
Update apt database to apply new apt sources.

```
[Host]
apt update
```

### Step3) Update dpkg setting
Update dpkg settings to support foreign architectures.

With `--print-architecture` and `--print-foreign-architectures`, we can confirm that dpkg works well.

```
[Host]
$ sudo dpkg  --add-architecture arm64
$ sudo dpkg  --add-architecture armhf

$ sudo dpkg  --print-architecture
amd64
$ sudo dpkg  --print-foreign-architectures
arm64
armhf
```

### Option) Confirm working pkg-config

With MultiArch, several shared libraries and pkgconfig information for each architectures are stored into /usr/lib.

```
/usr
  + lib
    + aarch64-linux-gnu   - shared libraries for arm64
      + pkgconfig         - pkg-config files for arm64 libraries
    + arm-linux-gnueabihf - shared libraries for armhf
      + pkgconfig         - pkg-config files for armhf libraries
  + share
    + pkgconfig         - pkg-config files(for header files)
```

Confirm to work pkg-config using `PKG_CONFIG_PATH`, `PKG_CONFIG_LIBDIR` and `PKG_CONFIG_SYSROOT_DIR` options..

```
[Host]
(for host)
  pkg-config  --list-all

(for armv7)
PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig:/usr/share/pkgconfig \
  PKG_CONFIG_LIBDIR=/usr/lib/arm-linux-gnueabihf \
  PKG_CONFIG_SYSROOT_DIR=/ \
      pkg-config --list-all

(for aarch64)
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
      pkg-config --list-all
```

Cross-compile(for aarch64)
--------------------------
Following is to compile for target(aarch64) at host(x86-64).

### Step1) Install external libraries for target into host.

Install libfreetype-dev and libharfbuzz-dev for target(arm64) into host(x86-64).

```
[Host]
sudo apt install libfreetype-dev:arm64 libharfbuzz-dev:arm64

PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
       pkg-config freetype2 harfbuzz --cflags --libs

-I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/harfbuzz -I/usr/include/glib-2.0 -I/usr/lib/aarch64-linux-gnu/glib-2.0/include -L/usr/lib/aarch64-linux-gnu -lfreetype -lharfbuzz
```

### Step2) Build and install OpenCV libraries into target.

Execute `cmake` to cross-compile for aarch64.

@note `-DCMAKE_TOOLCHAIN_FILE` seems to require to set absolute file path(not relative path)..

```
[Host]
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
        cmake -S opencv \
              -B build4-full_arm64 \
              -DCMAKE_TOOLCHAIN_FILE=/home/kmtr/work/opencv/platforms/linux/aarch64-gnu.toolchain.cmake \
              -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules \
              -GNinja

     cmake --build   build4-full_arm64
sudo cmake --install build4-full_arm64
```

cmake outputs `Host: ... x86_64` and `Target: ... aarch64`.

```
-- General configuration for OpenCV 4.8.0-dev =====================================
--   Version control:               4.8.0-xxx-XXXXXXXX
--
--   Extra modules:
--     Location (extra):            /home/kmtr/work/opencv_contrib/modules
--     Version control (extra):     4.8.0-xxx-XXXXXXXX
--
--   Platform:
--     Timestamp:                   yyyy-mm-ddThh:mm:ssZ
--     Host:                        Linux 6.2.0-32-generic x86_64
--     Target:                      Linux 1 aarch64
--     CMake:                       3.25.1
```

And copy OpenCV/OpenCV Contrib libraries from host to target.

But these libraries cannot resolved dependency fot external libraries.

```
[Target]
ldd /usr/local/lib/libopencv_freetype.so
        linux-vdso.so.1 (0xABCDEFG01234567)
        libopencv_imgproc.so.408 => /usr/local/lib/libopencv_imgproc.so.408 (0xABCDEFG01234567)
        libfreetype.so.6 => /lib/aarch64-linux-gnu/libfreetype.so.6 (0xABCDEFG01234567)
        libharfbuzz.so.0 => not found
        libopencv_core.so.408 => /usr/local/lib/libopencv_core.so.408 (0xABCDEFG01234567)
```

### Step3) Install external libraries on target.
(This step is executed at target)

Install external libraries to requires OpenCV/OpenCV contrib libraies at target.
After installing nessesary libraries, dependency is resolved.

```
[Target]
sudo apt install libfreetype-dev libharfbuzz-dev
sudo ldconfig

ldd /usr/local/lib/libopencv_freetype.so
        linux-vdso.so.1 (0xABCDEFG01234567)
        libopencv_imgproc.so.408 => /usr/local/lib/libopencv_imgproc.so.408 (0xABCDEFG01234567)
        libfreetype.so.6 => /lib/aarch64-linux-gnu/libfreetype.so.6 (0xABCDEFG01234567)
        libharfbuzz.so.0 => /lib/aarch64-linux-gnu/libharfbuzz.so.0 (0xABCDEFG01234567)
        libopencv_core.so.408 => /usr/local/lib/libopencv_core.so.408 (0xABCDEFG01234567)
```

Cross-compile(for armv7)
------------------------
Following is to compile for target(armhf) at host(x86-64).

- To resolbe dependencies, `linux-libc-dev:armhf` is required.
- To optimize with neon, `-DENABLE_NEON=ON` is needed.

```
sudo apt install linux-libc-dev:armhf
sudo apt install libfreetype-dev:armhf libharfbuzz-dev:armhf

PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/arm-linux-gnueabihf/ \
    PKG_CONFIG_SYSROOT_DIR=/ \
       pkg-config freetype2 harfbuzz --cflags --libs
(output) -I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/harfbuzz -I/usr/include/glib-2.0 -I/usr/lib/arm-linux-gnueabihf/glib-2.0/include -L/usr/lib/arm-linux-gnueabihf -lfreetype -lharfbuzz

PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/arm-linux-gnueabihf \
    PKG_CONFIG_SYSROOT_DIR=/ \
        cmake -S opencv \
              -B build4-full_armhf \
              -DENABLE_NEON=ON \
              -DCMAKE_TOOLCHAIN_FILE=/home/kmtr/work/opencv/platforms/linux/arm-gnueabi.toolchain.cmake \
              -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules \
              -GNinja

cmake      --build   build4-full_armhf
sudo cmake --install build4-full_armhf
```

