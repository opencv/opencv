# Imath Release Notes

* [Version 3.1.3](#version-313-september-2-2021) September 2, 2021
* [Version 3.1.2](#version-312-july-31-2021) July 31, 2021
* [Version 3.1.1](#version-311-july-20-2021) July 20, 2021
* [Version 3.1.0](#version-310-july-13-2021) July 13, 2021
* [Version 3.0.5](#version-305-june-29-2021) June 29, 2021
* [Version 3.0.4](#version-304-june-1-2021) June 1, 2021
* [Version 3.0.2](#version-302-may-16-2021) May 16, 2021
* [Version 3.0.1](#version-301-april-1-2021) April 1, 2021
* [Version 3.0.1-beta](#version-301-beta-march-28-2021) March 28, 2021
* [Version 3.0.0-beta](#version-300-beta-march-15-2021) March 15, 2021
* [Inherited History from OpenEXR](#inherited-history-from-openexr)

## Version 3.1.3 (September 2, 2021)

Patch release with miscellaneous fixes

* \[[#204](https://github.com/AcademySoftwareFoundation/Imath/pull/204)\] 
Fix undefined access of a vector when empty
* \[[#203](https://github.com/AcademySoftwareFoundation/Imath/pull/203)\] 
Require sphinx 4.0.3
* \[[#201](https://github.com/AcademySoftwareFoundation/Imath/pull/201)\] 
Build sphinx/doxygen docs with CMake
* \[[#200](https://github.com/AcademySoftwareFoundation/Imath/pull/200)\] 
Use PYIMATH_OVERRIDE_PYTHON_INSTALL_DIR to specify destination python modules
* \[[#199](https://github.com/AcademySoftwareFoundation/Imath/pull/199)\] 
Guard `__has_attribute` for compilers that don't support it
* \[[#198](https://github.com/AcademySoftwareFoundation/Imath/pull/198)\] 
Cuda safety fixes
* \[[#194](https://github.com/AcademySoftwareFoundation/Imath/pull/194)\] 
Replace stray Imath:: with IMATH_INTERNAL_NAMESPACE::

## Version 3.1.2 (July 31, 2021)

Patch release that fixes a Windows header issue.

* \[[#190](https://github.com/AcademySoftwareFoundation/Imath/pull/190)\] 
  Improve handling of ``#include <*intrin.h>``

## Version 3.1.1 (July 20, 2021)

Patch release that fixes a build failure on ARM64 macOS

* \[[#184](https://github.com/AcademySoftwareFoundation/Imath/pull/184)\] Include ``<x86intrin.h>`` only if ``__x86_64__``

## Version 3.1.0 (July 13, 2021)

Minor release with new features:

* Optimized half-to-float and float-to-half conversion, using F16C SSE
  instruction set if available. Non-SSE conversion eliminates the
  float-to-half exponent lookup table, and half-to-float conversion
  provides a compile-time-optional bit shifting that is slower but
  eliminates the need for the lookup table, for applications where
  memory is limited.

  Half-to-float and float-to-half conversion is also available as
  C-language functions ``imath_half_to_float()`` and
  ``imath_float_to_half()``.
  
  All new conversions produced identical results, and new options are
  off by default to ensure backwards compatibility. See
  https://imath.readthedocs.io for more info.
  
* NOEXCEPT specifier can be eliminated at compile-time via the
  ``IMATH_USE_NOEXCEPT`` CMake option.

* Python bindings:
  * FixedArray objects support a "read only" state.
  * FixedArray objects support python buffer protocol.

* Optimized 4x4 matrix multiplication.

### Merged Pull Requests

* [#181](https://github.com/AcademySoftwareFoundation/Imath/pull/181)] Clean up half lookup-table options and related docs
* [#179](https://github.com/AcademySoftwareFoundation/Imath/pull/179)] Remove dead code from half
* [#178](https://github.com/AcademySoftwareFoundation/Imath/pull/178)] Update Imath docs for 3.1
* [#177](https://github.com/AcademySoftwareFoundation/Imath/pull/177)] v3.1.0 release notes
* [#176](https://github.com/AcademySoftwareFoundation/Imath/pull/176)] v3.0.5 notes
* [#175](https://github.com/AcademySoftwareFoundation/Imath/pull/175)] Clean up library VERSION and SOVERSION 
* [#172](https://github.com/AcademySoftwareFoundation/Imath/pull/172)] Use CMAKE_INSTALL_FULL_LIBDIR/INCLUDEDIR for pkgconfig 
* [#173](https://github.com/AcademySoftwareFoundation/Imath/pull/173)] Update README.md and INSTALL.md for 3.1 
* [#169](https://github.com/AcademySoftwareFoundation/Imath/pull/169)] Add testInterop to test list in define_imath_test() 
* [#168](https://github.com/AcademySoftwareFoundation/Imath/pull/168)] Push/pop Windows warning pragma 
* [#167](https://github.com/AcademySoftwareFoundation/Imath/pull/167)] Clean up cmake lib symlink message 
* [#165](https://github.com/AcademySoftwareFoundation/Imath/pull/165)] Use CMAKE_<CONFIG>_POSTFIX for .pc file lib suffix. 
* [#166](https://github.com/AcademySoftwareFoundation/Imath/pull/166)] Fix non-versioned library symlinks in debug build. 
* [#162](https://github.com/AcademySoftwareFoundation/Imath/pull/162)] silence a few warnings noticed with -Weverything 
* [#157](https://github.com/AcademySoftwareFoundation/Imath/pull/157)] IMATH_NOEXCEPT macro to make noexcept a compile-time option 
* [#156](https://github.com/AcademySoftwareFoundation/Imath/pull/156)] PyImath read-only FixedArray state & python buffer protocol support 
* [#158](https://github.com/AcademySoftwareFoundation/Imath/pull/158)] Improve 4x4 matrix multiplication 
* [#160](https://github.com/AcademySoftwareFoundation/Imath/pull/160)] Clean up analysis_workflow.yml 
* [#155](https://github.com/AcademySoftwareFoundation/Imath/pull/155)] Release notes for v3.0.4 
* [#159](https://github.com/AcademySoftwareFoundation/Imath/pull/159)] Add new macros to Doxyfile PREDEFINED 
* [#153](https://github.com/AcademySoftwareFoundation/Imath/pull/153)] Configure ImathTest as optional standalone program 
* [#141](https://github.com/AcademySoftwareFoundation/Imath/pull/141)] Enable C and lighter weight half <-> float conversion 
* [#150](https://github.com/AcademySoftwareFoundation/Imath/pull/150)] Add __version__ attr to imath and imathnumpy python modules 

## Version 3.0.5 (June 29, 2021)

Patch release that fixes problems with library symlinks and
pkg-config. Otherwise, no code changes.

* [#172](https://github.com/AcademySoftwareFoundation/Imath/pull/172)] Use CMAKE_INSTALL_FULL_LIBDIR/INCLUDEDIR for pkgconfig
* [#166](https://github.com/AcademySoftwareFoundation/Imath/pull/166)] Fix non-versioned library symlinks in debug build.
* [#165](https://github.com/AcademySoftwareFoundation/Imath/pull/165)] Use CMAKE_<CONFIG>_POSTFIX for .pc file lib suffix.

## Version 3.0.4 (June 1, 2021)

Patch release that corrects a problem with the release version number
of v3.0.2:

* \[[#147](https://github.com/AcademySoftwareFoundation/Imath/pull/147)\] Add #define for IMATH_VERSION_RELEASE_TYPE
* \[[#145](https://github.com/AcademySoftwareFoundation/Imath/pull/145)\] Set IMATH_VERSION from Imath_VERSION instead of CMAKE_PROJECT_VERSION

## Version 3.0.2 (May 16, 2021)

Patch release with miscellaneous bug/build fixes:

* \[[#142](https://github.com/AcademySoftwareFoundation/Imath/pull/142)\] Fix order of ${IMATH_SOVERSION}.${IMATH_SOREVISION}.${IMATH_SOAGE}
* \[[#140](https://github.com/AcademySoftwareFoundation/Imath/pull/140)\] Fix regression in succf()/predf()          
* \[[#139](https://github.com/AcademySoftwareFoundation/Imath/pull/139)\] Clean up setting of Imath version          
* \[[#137](https://github.com/AcademySoftwareFoundation/Imath/pull/137)\] Don't impose C++14 on downstream projects  
* \[[#135](https://github.com/AcademySoftwareFoundation/Imath/pull/135)\] Add section on python bindings             
* \[[#133](https://github.com/AcademySoftwareFoundation/Imath/pull/133)\] Lib version                                

## Version 3.0.1 (April 1, 2021)

First release of Imath independent of OpenEXR.

See the [porting guide](docs/PortingGuide2-3.md) for details about
differences from previous releases.

Summary:

* Imath includes the half type, formerly in a separate Half library.
* Headers are installed in ``Imath/`` subdirectory.
* All appropriate methods are marked constexpr, noexcept
* Appropriate declaration include CUDA ``__host__`` and ``__device__``
  directives.
* Throwing methods throw std exceptions instead of ``Iex``.
* New Vec and Matrix interoperability constructors for conversion from
  other similar type objects.
* Symbol linkage visibility is limited to specific public symbols.
* python bindings are off by default, available by setting ``PYTHON=ON``.
* Deprecated features:
  - ``std::numeric_limits`` replaces ``Imath::limits``.
  - ``Int64`` and ``SInt64`` are deprecated in favor of ``uint64_t``
    and ``int64_t``.

## Version 3.0.1-beta (March 28, 2021)

Beta patch release:

* \[[#131](https://github.com/AcademySoftwareFoundation/Imath/pull/131)\] #if IMATH_FOREIGN_VECTOR_INTEROP around type detectors

* \[[#130](https://github.com/AcademySoftwareFoundation/Imath/pull/130)\] Forward declarations only if header is not included

## Version 3.0.0-beta (March 15, 2021)

First release of Imath independent of OpenEXR.

See the [porting guide](docs/PortingGuide2-3.md) for details about
differences from previous releases.

Summary and Key Changes:

* Imath includes the half type, formerly in a separate Half library.
* Headers are installed in ``Imath/`` subdirectory.
* Header files have been pruned of extraneous ``#include``'s, which may
  require updates to application source code.
* All appropriate methods are marked constexpr, noexcept
* Appropriate declaration include CUDA ``__host__`` and ``__device__``
  directives.
* Throwing methods throw std exceptions instead of ``Iex``.
* New Vec and Matrix interoperability constructors for conversion from
  other similar type objects.
* Symbol linkage visibility is limited to specific public symbols.
* Python bindings are off by default, available by setting ``PYTHON=ON``.
* Deprecated features:
  - ``std::numeric_limits`` replaces ``Imath::limits``.
  - ``Int64`` and ``SInt64`` are deprecated in favor of ``uint64_t``
    and ``int64_t``.

### Merged Pull Requests

* \[[#119](https://github.com/AcademySoftwareFoundation/Imath/pull/119)\] Enable policy 77 if possible.
* \[[#117](https://github.com/AcademySoftwareFoundation/Imath/pull/117)\] Add/fix rst source for readthedocs
* \[[#116](https://github.com/AcademySoftwareFoundation/Imath/pull/116)\] Add/fix doxygen comments
* \[[#115](https://github.com/AcademySoftwareFoundation/Imath/pull/115)\] Add =delete for the int64_t Vec specializations
* \[[#114](https://github.com/AcademySoftwareFoundation/Imath/pull/114)\] Disable analysis on PR/push
* \[[#113](https://github.com/AcademySoftwareFoundation/Imath/pull/113)\] Clean up cmake/config
* \[[#111](https://github.com/AcademySoftwareFoundation/Imath/pull/111)\] Rename IMATH_IMATH_NAMESPACE option to IMATH_NAMESPACE
* \[[#110](https://github.com/AcademySoftwareFoundation/Imath/pull/110)\] Remove PyImathConfigInternal
* \[[#109](https://github.com/AcademySoftwareFoundation/Imath/pull/109)\] build one python binding only
* \[[#107](https://github.com/AcademySoftwareFoundation/Imath/pull/107)\] Add int64_t specializations of Vec and Box.
* \[[#106](https://github.com/AcademySoftwareFoundation/Imath/pull/106)\] Replace Int64/SInt64 with uint64_t/int64_t
* \[[#103](https://github.com/AcademySoftwareFoundation/Imath/pull/103)\] Drop support for exception-handling in PyImath
* \[[#102](https://github.com/AcademySoftwareFoundation/Imath/pull/102)\] cmake improvements and fixes
* \[[#100](https://github.com/AcademySoftwareFoundation/Imath/pull/100)\] Replace Iex::OverflowExc with std::invalid_argument
* \[[#98](https://github.com/AcademySoftwareFoundation/Imath/pull/98)\] constexpr Vec2, Vec3, Vec4 constructors
* \[[#97](https://github.com/AcademySoftwareFoundation/Imath/pull/97)\] restore original behavior of Matrix33<T>::setScale()
* \[[#95](https://github.com/AcademySoftwareFoundation/Imath/pull/95)\] Build fixups for Visual Studio 2015
* \[[#94](https://github.com/AcademySoftwareFoundation/Imath/pull/94)\] Add background and file/class-specific details to porting guide
* \[[#93](https://github.com/AcademySoftwareFoundation/Imath/pull/93)\] Fix typo in comment in testHalfLimits
* \[[#92](https://github.com/AcademySoftwareFoundation/Imath/pull/92)\] Replace ILMBASE_HAVE_LARGE_STACK with IMATH_HAVE_LARGE_STACK
* \[[#91](https://github.com/AcademySoftwareFoundation/Imath/pull/91)\] Interoperability constructors
* \[[#90](https://github.com/AcademySoftwareFoundation/Imath/pull/90)\] Fix compiler errors from recent changes
* \[[#89](https://github.com/AcademySoftwareFoundation/Imath/pull/89)\] First stab at Imath 2->3 porting guide
* \[[#88](https://github.com/AcademySoftwareFoundation/Imath/pull/88)\] PyImath installs headers into Imath subdirectory
* \[[#87](https://github.com/AcademySoftwareFoundation/Imath/pull/87)\] constexpr as much of half as possible
* \[[#83](https://github.com/AcademySoftwareFoundation/Imath/pull/83)\] Replace NOTICE with STATUS for CMake messages
* \[[#82](https://github.com/AcademySoftwareFoundation/Imath/pull/82)\] Clean up Imath::Limits and numeric_limits issues
* \[[#81](https://github.com/AcademySoftwareFoundation/Imath/pull/81)\] Reformat all Imath header comments to doxygen style
* \[[#77](https://github.com/AcademySoftwareFoundation/Imath/pull/77)\] Change copyright notices to standard SPDX format
* \[[#76](https://github.com/AcademySoftwareFoundation/Imath/pull/76)\] Incorrect constexpr in Imath::limits<half>, and missing test.
* \[[#75](https://github.com/AcademySoftwareFoundation/Imath/pull/75)\] CI: add VFX2021 jobs, enable coverage analysis
* \[[#74](https://github.com/AcademySoftwareFoundation/Imath/pull/74)\] noexcept all the things
* \[[#73](https://github.com/AcademySoftwareFoundation/Imath/pull/73)\] Simplify definition of IMATH_RESTRICT for modern supported compilers
* \[[#72](https://github.com/AcademySoftwareFoundation/Imath/pull/72)\] Eliminate normalize and length methods for Vec<inttype>
* \[[#70](https://github.com/AcademySoftwareFoundation/Imath/pull/70)\] Adding missing header
* \[[#69](https://github.com/AcademySoftwareFoundation/Imath/pull/69)\] [#bugfix] Install error on windows #68
* \[[#67](https://github.com/AcademySoftwareFoundation/Imath/pull/67)\] Fix two typos in m22 tests causing out of bounds references
* \[[#66](https://github.com/AcademySoftwareFoundation/Imath/pull/66)\] Use likely/unlikely to improve certain vector ops
* \[[#65](https://github.com/AcademySoftwareFoundation/Imath/pull/65)\] Deprecate Math<T> in favor of std::
* \[[#60](https://github.com/AcademySoftwareFoundation/Imath/pull/60)\] Make Matrix implementation more SIMD friendly
* \[[#59](https://github.com/AcademySoftwareFoundation/Imath/pull/59)\] Imath::Vec -- omit exception-throwing methods from Cuda side
* \[[#58](https://github.com/AcademySoftwareFoundation/Imath/pull/58)\] Make separate test calls for each test
* \[[#57](https://github.com/AcademySoftwareFoundation/Imath/pull/57)\] Fixes the subproject test
* \[[#56](https://github.com/AcademySoftwareFoundation/Imath/pull/56)\] Initial support for C wrappers
* \[[#55](https://github.com/AcademySoftwareFoundation/Imath/pull/55)\] Fix non-linux platform CI
* \[[#54](https://github.com/AcademySoftwareFoundation/Imath/pull/54)\] Combination of Half/Imath and HalfTest/ImathTest directories.
* \[[#53](https://github.com/AcademySoftwareFoundation/Imath/pull/53)\] Stoped sonar cloud from running on PR
* \[[#52](https://github.com/AcademySoftwareFoundation/Imath/pull/52)\] Fix problems with ImathInterval, and add test
* \[[#51](https://github.com/AcademySoftwareFoundation/Imath/pull/51)\] First pass at sphinx/breathe/doxygen documentation
* \[[#50](https://github.com/AcademySoftwareFoundation/Imath/pull/50)\] Removed all references to PYIMATH_VERSION, as it is redundant.
* \[[#48](https://github.com/AcademySoftwareFoundation/Imath/pull/48)\] Set version to 3.0.0 and SOCURRENT to 26
* \[[#47](https://github.com/AcademySoftwareFoundation/Imath/pull/47)\] Added Exc variants of all methods in frustum that required them.
* \[[#46](https://github.com/AcademySoftwareFoundation/Imath/pull/46)\] Movement of all source directories into one top level src/ 
* \[[#44](https://github.com/AcademySoftwareFoundation/Imath/pull/44)\] Fix copy/paste typos in Doxyfile and conf.py
* \[[#43](https://github.com/AcademySoftwareFoundation/Imath/pull/43)\] Initial Doxygen/sphinx/breathe/readthedocs configuration
* \[[#42](https://github.com/AcademySoftwareFoundation/Imath/pull/42)\] Made various Imath/ header methods inline
* \[[#41](https://github.com/AcademySoftwareFoundation/Imath/pull/41)\] __host__ __device__ CUDA macro added to all header functions under Imath/
* \[[#40](https://github.com/AcademySoftwareFoundation/Imath/pull/40)\] Update INSTALL info on namespaces and cmake options
* \[[#39](https://github.com/AcademySoftwareFoundation/Imath/pull/39)\] Clean up of repo docs.
* \[[#38](https://github.com/AcademySoftwareFoundation/Imath/pull/38)\] Added CUDA __host__ __device__ with macro to Vector, Matrix, Limits, \xe2\x80\xa6
* \[[#37](https://github.com/AcademySoftwareFoundation/Imath/pull/37)\] Add .git-blame-ignore-revs to ignore reformatting
* \[[#36](https://github.com/AcademySoftwareFoundation/Imath/pull/36)\] Disable clang-format for python bindings
* \[[#32](https://github.com/AcademySoftwareFoundation/Imath/pull/32)\] Tune .clang-format to match existing style
* \[[#30](https://github.com/AcademySoftwareFoundation/Imath/pull/30)\] Changed analysis sonarcloud tests, run on pull request.
* \[[#29](https://github.com/AcademySoftwareFoundation/Imath/pull/29)\] Added CI testing and made necessary changes to pass those tests.
* \[[#27](https://github.com/AcademySoftwareFoundation/Imath/pull/27)\] Simplest CUDA half type conflict resolution implementation.
* \[[#25](https://github.com/AcademySoftwareFoundation/Imath/pull/25)\] Used macros to allow compilation with C++11 and constexpr
* \[[#24](https://github.com/AcademySoftwareFoundation/Imath/pull/24)\] b"removed pragma to disable clang's -Wself-assign-overloaded"
* \[[#23](https://github.com/AcademySoftwareFoundation/Imath/pull/23)\] changed assert()\] to throw, which is what the original Iex macro ASSERT()\] macro did
* \[[#21](https://github.com/AcademySoftwareFoundation/Imath/pull/21)\] First pass at adding constexpr where useful
* \[[#20](https://github.com/AcademySoftwareFoundation/Imath/pull/20)\] Speedtest and Inversion python bindings for Arrays
* \[[#19](https://github.com/AcademySoftwareFoundation/Imath/pull/19)\] clean up Imath repo docs
* \[[#18](https://github.com/AcademySoftwareFoundation/Imath/pull/18)\] Transfer of PyImath history and source to Imath
* \[[#17](https://github.com/AcademySoftwareFoundation/Imath/pull/17)\] fixed typo in README.md
* \[[#15](https://github.com/AcademySoftwareFoundation/Imath/pull/15)\] further edits of README.md
* \[[#14](https://github.com/AcademySoftwareFoundation/Imath/pull/14)\] First complete draft of README.md for the Imath project

## Inherited History from OpenEXR

History dated before May 9th, 2020 has been inherited from
https://github.com/AcademySoftwareFoundation/openexr, omitting commits
(via [git-filter-repo](https://github.com/newren/git-filter-repo)) not
pertaining to files now a part of the Imath project.
