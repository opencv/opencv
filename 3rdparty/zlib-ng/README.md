| CI | Stable | Develop |
|:---|:-------|:--------|
| GitHub Actions | [![Stable CMake](https://github.com/zlib-ng/zlib-ng/actions/workflows/cmake.yml/badge.svg?branch=stable)](https://github.com/zlib-ng/zlib-ng/actions/workflows/cmake.yml?query=branch%3Astable) <br> [![Stable Configure](https://github.com/zlib-ng/zlib-ng/actions/workflows/configure.yml/badge.svg?branch=stable)](https://github.com/zlib-ng/zlib-ng/actions/workflows/configure.yml?query=branch%3Astable) <br> [![Stable NMake](https://github.com/zlib-ng/zlib-ng/actions/workflows/nmake.yml/badge.svg?branch=stable)](https://github.com/zlib-ng/zlib-ng/actions/workflows/nmake.yml?query=branch%3Astable) | [![Develop CMake](https://github.com/zlib-ng/zlib-ng/actions/workflows/cmake.yml/badge.svg?branch=develop)](https://github.com/zlib-ng/zlib-ng/actions/workflows/cmake.yml?query=branch%3Adevelop) <br> [![Develop Configure](https://github.com/zlib-ng/zlib-ng/actions/workflows/configure.yml/badge.svg?branch=develop)](https://github.com/zlib-ng/zlib-ng/actions/workflows/configure.yml?query=branch%3Adevelop) <br> [![Develop NMake](https://github.com/zlib-ng/zlib-ng/actions/workflows/nmake.yml/badge.svg?branch=develop)](https://github.com/zlib-ng/zlib-ng/actions/workflows/nmake.yml?query=branch%3Adevelop) |
| CodeFactor     | [![CodeFactor](https://www.codefactor.io/repository/github/zlib-ng/zlib-ng/badge/stable)](https://www.codefactor.io/repository/github/zlib-ng/zlib-ng/overview/stable) | [![CodeFactor](https://www.codefactor.io/repository/github/zlib-ng/zlib-ng/badge/develop)](https://www.codefactor.io/repository/github/zlib-ng/zlib-ng/overview/develop) |
| OSS-Fuzz       | [![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/zlib-ng.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:zlib-ng) | [![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/zlib-ng.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:zlib-ng) |
| Codecov        | [![codecov](https://codecov.io/github/zlib-ng/zlib-ng/branch/stable/graph/badge.svg?token=uKsgK9LIuC)](https://codecov.io/github/zlib-ng/zlib-ng/tree/stable) | [![codecov](https://codecov.io/github/zlib-ng/zlib-ng/branch/develop/graph/badge.svg?token=uKsgK9LIuC)](https://codecov.io/github/zlib-ng/zlib-ng/tree/develop) |

## zlib-ng
*zlib data compression library for the next generation systems*

Maintained by Hans Kristian Rosbach
          aka Dead2 (zlib-ng àt circlestorm dót org)

Features
--------

* Zlib compatible API with support for dual-linking
* Modernized native API based on zlib API for ease of porting
* Modern C11 syntax and a clean code layout
* Deflate medium and quick algorithms based on Intel’s zlib fork
* Support for CPU intrinsics when available
  * Adler32 implementation using SSSE3, AVX2, AVX512, AVX512-VNNI, Neon, VMX & VSX
  * CRC32-B implementation using PCLMULQDQ, VPCLMULQDQ, ACLE, & IBM Z
  * Slide hash implementations using SSE2, AVX2, ARMv6, Neon, VMX & VSX
  * Compare256 implementations using SSE2, AVX2, Neon, POWER9 & RVV
  * Inflate chunk copying using SSE2, SSSE3, AVX, Neon & VSX
  * Support for hardware-accelerated deflate using IBM Z DFLTCC
* Unaligned memory read/writes and large bit buffer improvements
* Includes improvements from Cloudflare and Intel forks
* Configure, CMake, and NMake build system support
* Comprehensive set of CMake unit tests
* Code sanitizers, fuzzing, and coverage
* GitHub Actions continuous integration on Windows, macOS, and Linux
  * Emulated CI for ARM, AARCH64, PPC, PPC64, RISCV, SPARC64, S390x using qemu


History
-------

The motivation for this fork was seeing several 3rd party contributions with new optimizations not getting
implemented into the official zlib repository.

Mark Adler has been maintaining zlib for a very long time, and he has done a great job and hopefully he will continue
for a long time yet. The idea of zlib-ng is not to replace zlib, but to co-exist as a drop-in replacement with a
lower threshold for code change.

zlib has a long history and is incredibly portable, even supporting many systems that predate the Internet.<br>
That is great, but it can complicate further development and maintainability. The zlib code contains many workarounds
for really old compilers or to accommodate systems with limitations such as operating in a 16-bit environment.

Many of these workarounds are only maintenance burdens, some of them are pretty huge code-wise. With many workarounds
cluttered throughout the code, it makes it harder for new programmers with an idea/interest for zlib to contribute.

I decided to make a fork, merge all the Intel optimizations, some of the Cloudflare optimizations, plus a couple other
smaller patches. Then started cleaning out workarounds, various dead code, all contrib and example code.<br>
The result is a better performing and easier to maintain zlib-ng.

A lot of improvements have gone into zlib-ng since its start, and numerous people and companies have contributed both
small and big improvements, or valuable testing.


Build
-----
<sup>Please read LICENSE.md, it is very simple and very liberal.</sup>

There are two ways to build zlib-ng:

### Cmake

To build zlib-ng using the cross-platform makefile generator cmake.

```
cmake .
cmake --build . --config Release
ctest --verbose -C Release
```

Alternatively, you can use the cmake configuration GUI tool ccmake:

```
ccmake .
```

### Configure

To build zlib-ng using the bash configure script:

```
./configure
make
make test
```

Build Options
-------------

| CMake                      | configure                | Description                                                                         | Default |
|:---------------------------|:-------------------------|:------------------------------------------------------------------------------------|---------|
| ZLIB_COMPAT                | --zlib-compat            | Compile with zlib compatible API                                                    | OFF     |
| ZLIB_ENABLE_TESTS          |                          | Build test binaries                                                                 | ON      |
| WITH_GZFILEOP              | --without-gzfileops      | Compile with support for gzFile related functions                                   | ON      |
| WITH_OPTIM                 | --without-optimizations  | Build with optimisations                                                            | ON      |
| WITH_NEW_STRATEGIES        | --without-new-strategies | Use new strategies                                                                  | ON      |
| WITH_NATIVE_INSTRUCTIONS   |                          | Compiles with full instruction set supported on this host (gcc/clang -march=native) | OFF     |
| WITH_RUNTIME_CPU_DETECTION |                          | Compiles with runtime CPU detection                                                 | ON      |
| WITH_SANITIZER             |                          | Build with sanitizer (memory, address, undefined)                                   | OFF     |
| WITH_GTEST                 |                          | Build gtest_zlib                                                                    | ON      |
| WITH_FUZZERS               |                          | Build test/fuzz                                                                     | OFF     |
| WITH_BENCHMARKS            |                          | Build test/benchmarks                                                               | OFF     |
| WITH_MAINTAINER_WARNINGS   |                          | Build with project maintainer warnings                                              | OFF     |
| WITH_CODE_COVERAGE         |                          | Enable code coverage reporting                                                      | OFF     |


Install
-------

WARNING: We do not recommend manually installing unless you really know what you are doing, because this can
potentially override the system default zlib library, and any incompatibility or wrong configuration of zlib-ng
can make the whole system unusable, requiring recovery or reinstall.
If you still want a manual install, we recommend using the /opt/ path prefix.

For Linux distros, an alternative way to use zlib-ng (if compiled in zlib-compat mode) instead of zlib, is through
the use of the _LD_PRELOAD_ environment variable. If the program is dynamically linked with zlib, then the program
will temporarily attempt to use zlib-ng instead, without risking system-wide instability.

```
LD_PRELOAD=/opt/zlib-ng/libz.so.1.2.13.zlib-ng /usr/bin/program
```

### Cmake

To install zlib-ng system-wide using cmake:

```sh or powershell
cmake --build . --target install
```

### Configure

To install zlib-ng system-wide using the configure script:

```sh
make install
```

### CPack

After building with cmake, an installation package can be created using cpack. By default a tgz package is created,
but you can append `-G <format>` to each command to generate alternative packages types (TGZ, ZIP, RPM, DEB). To easily
create a rpm or deb package, you would use `-G RPM` or `-G DEB` respectively.

```sh or powershell
cd build
cpack --config CPackConfig.cmake
cpack --config CPackSourceConfig.cmake
```

### Vcpkg

Alternatively, you can build and install zlib-ng using the [vcpkg](https://github.com/Microsoft/vcpkg/) dependency manager:

```sh or powershell
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh # "./bootstrap-vcpkg.bat" for powershell
./vcpkg integrate install
./vcpkg install zlib-ng
```

The zlib-ng port in vcpkg is kept up to date by Microsoft team members and community contributors.
If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

Contributing
------------

Zlib-ng is aiming to be open to contributions, and we would be delighted to receive pull requests on github.
Help with testing and reviewing pull requests etc is also very much appreciated.

Please check the Wiki for more info: [Contributing](https://github.com/zlib-ng/zlib-ng/wiki/Contributing)

Acknowledgments
----------------

Thanks go out to all the people and companies who have taken the time to contribute
code reviews, testing and/or patches. Zlib-ng would not have been nearly as good without you.

The deflate format used by zlib was defined by Phil Katz.<br>
The deflate and zlib specifications were written by L. Peter Deutsch.

zlib was originally created by Jean-loup Gailly (compression) and Mark Adler (decompression).


Advanced Build Options
----------------------

| CMake                           | configure             | Description                                                         | Default                |
|:--------------------------------|:----------------------|:--------------------------------------------------------------------|------------------------|
| FORCE_SSE2                      | --force-sse2          | Skip runtime check for SSE2 instructions (Always on for x86_64)     | OFF (x86)              |
| WITH_AVX2                       |                       | Build with AVX2 intrinsics                                          | ON                     |
| WITH_AVX512                     |                       | Build with AVX512 intrinsics                                        | ON                     |
| WITH_AVX512VNNI                 |                       | Build with AVX512VNNI intrinsics                                    | ON                     |
| WITH_SSE2                       |                       | Build with SSE2 intrinsics                                          | ON                     |
| WITH_SSSE3                      |                       | Build with SSSE3 intrinsics                                         | ON                     |
| WITH_SSE42                      |                       | Build with SSE42 intrinsics                                         | ON                     |
| WITH_PCLMULQDQ                  |                       | Build with PCLMULQDQ intrinsics                                     | ON                     |
| WITH_VPCLMULQDQ                 | --without-vpclmulqdq  | Build with VPCLMULQDQ intrinsics                                    | ON                     |
| WITH_ACLE                       | --without-acle        | Build with ACLE intrinsics                                          | ON                     |
| WITH_NEON                       | --without-neon        | Build with NEON intrinsics                                          | ON                     |
| WITH_ARMV6                      | --without-armv6       | Build with ARMv6 intrinsics                                         | ON                     |
| WITH_ALTIVEC                    | --without-altivec     | Build with AltiVec (VMX) intrinsics                                 | ON                     |
| WITH_POWER8                     | --without-power8      | Build with POWER8 optimisations                                     | ON                     |
| WITH_RVV                        |                       | Build with RVV intrinsics                                           | ON                     |
| WITH_CRC32_VX                   | --without-crc32-vx    | Build with vectorized CRC32 on IBM Z                                | ON                     |
| WITH_DFLTCC_DEFLATE             | --with-dfltcc-deflate | Build with DFLTCC intrinsics for compression on IBM Z               | OFF                    |
| WITH_DFLTCC_INFLATE             | --with-dfltcc-inflate | Build with DFLTCC intrinsics for decompression on IBM Z             | OFF                    |
| WITH_UNALIGNED                  | --without-unaligned   | Allow optimizations that use unaligned reads if safe on current arch| ON                     |
| WITH_INFLATE_STRICT             |                       | Build with strict inflate distance checking                         | OFF                    |
| WITH_INFLATE_ALLOW_INVALID_DIST |                       | Build with zero fill for inflate invalid distances                  | OFF                    |
| INSTALL_UTILS                   |                       | Copy minigzip and minideflate during install                        | OFF                    |
| ZLIBNG_ENABLE_TESTS             |                       | Test zlib-ng specific API                                           | ON                     |


Related Projects
----------------

* Fork of the popular minizip                   https://github.com/zlib-ng/minizip-ng
* Python tool to benchmark minigzip/minideflate https://github.com/zlib-ng/deflatebench
* Python tool to benchmark pigz                 https://github.com/zlib-ng/pigzbench
* 3rd party patches for zlib-ng compatibility   https://github.com/zlib-ng/patches
