# Compilers:
# - CV_GCC - GNU compiler (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
# - CV_CLANG - Clang-compatible compiler (CMAKE_CXX_COMPILER_ID MATCHES "Clang" - Clang or AppleClang, see CMP0025)
# - CV_ICC - Intel compiler
# - MSVC - Microsoft Visual Compiler (CMake variable)
# - MINGW / CYGWIN / CMAKE_COMPILER_IS_MINGW / CMAKE_COMPILER_IS_CYGWIN (CMake original variables)
#
# CPU Platforms:
# - X86 / X86_64
# - ARM - ARM CPU, not defined for AArch64
# - AARCH64 - ARMv8+ (64-bit)
# - PPC64 / PPC64LE - PowerPC
# - MIPS
#
# OS:
# - WIN32 - Windows | MINGW
# - UNIX - Linux | MacOSX | ANDROID
# - ANDROID
# - IOS
# - APPLE - MacOSX | iOS
# ----------------------------------------------------------------------------

ocv_declare_removed_variables(MINGW64 MSVC64)
# do not use (CMake variables): CMAKE_CL_64

if(NOT DEFINED CV_GCC AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CV_GCC 1)
endif()
if(NOT DEFINED CV_CLANG AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")  # Clang or AppleClang (see CMP0025)
  set(CV_CLANG 1)
  set(CMAKE_COMPILER_IS_CLANGCXX 1)  # TODO next release: remove this
  set(CMAKE_COMPILER_IS_CLANGCC 1)   # TODO next release: remove this
endif()

function(access_CMAKE_COMPILER_IS_CLANGCXX)
  if(NOT OPENCV_SUPPRESS_DEPRECATIONS)
    message(WARNING "DEPRECATED: CMAKE_COMPILER_IS_CLANGCXX support is deprecated in OpenCV.
    Consider using:
    - CV_GCC    # GCC
    - CV_CLANG  # Clang or AppleClang (see CMP0025)
")
  endif()
endfunction()
variable_watch(CMAKE_COMPILER_IS_CLANGCXX access_CMAKE_COMPILER_IS_CLANGCXX)
variable_watch(CMAKE_COMPILER_IS_CLANGCC access_CMAKE_COMPILER_IS_CLANGCXX)


# ----------------------------------------------------------------------------
# Detect Intel ICC compiler
# ----------------------------------------------------------------------------
if(UNIX)
  if(__ICL)
    set(CV_ICC   __ICL)
  elseif(__ICC)
    set(CV_ICC   __ICC)
  elseif(__ECL)
    set(CV_ICC   __ECL)
  elseif(__ECC)
    set(CV_ICC   __ECC)
  elseif(__INTEL_COMPILER)
    set(CV_ICC   __INTEL_COMPILER)
  elseif(CMAKE_C_COMPILER MATCHES "icc")
    set(CV_ICC   icc_matches_c_compiler)
  endif()
endif()

if(MSVC AND CMAKE_C_COMPILER MATCHES "icc|icl")
  set(CV_ICC   __INTEL_COMPILER_FOR_WINDOWS)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER_VERSION
    AND NOT OPENCV_SUPPRESS_MESSAGE_MISSING_COMPILER_VERSION)
  message(WARNING "OpenCV: Compiler version is not available: CMAKE_CXX_COMPILER_VERSION is not set")
endif()
if((NOT DEFINED CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR STREQUAL "")
    AND NOT OPENCV_SUPPRESS_MESSAGE_MISSING_CMAKE_SYSTEM_PROCESSOR)
  message(WARNING "OpenCV: CMAKE_SYSTEM_PROCESSOR is not defined. Perhaps CMake toolchain is broken")
endif()
if(NOT DEFINED CMAKE_SIZEOF_VOID_P
    AND NOT OPENCV_SUPPRESS_MESSAGE_MISSING_CMAKE_SIZEOF_VOID_P)
  message(WARNING "OpenCV: CMAKE_SIZEOF_VOID_P is not defined. Perhaps CMake toolchain is broken")
endif()

message(STATUS "Detected processor: ${CMAKE_SYSTEM_PROCESSOR}")
if(OPENCV_SKIP_SYSTEM_PROCESSOR_DETECTION)
  # custom setup: required variables are passed through cache / CMake's command-line
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*")
  set(X86 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(AARCH64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(ARM 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64le")
  set(PPC64LE 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64")
  set(PPC64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(mips.*|MIPS.*)")
  set(MIPS 1)
else()
  if(NOT OPENCV_SUPPRESS_MESSAGE_UNRECOGNIZED_SYSTEM_PROCESSOR)
    message(WARNING "OpenCV: unrecognized target processor configuration")
  endif()
endif()

# Workaround for 32-bit operating systems on x86_64
if(CMAKE_SIZEOF_VOID_P EQUAL 4 AND X86_64
    AND NOT FORCE_X86_64  # deprecated (2019-12)
    AND NOT OPENCV_FORCE_X86_64
)
  message(STATUS "sizeof(void) = 4 on 64 bit processor. Assume 32-bit compilation mode")
  if(X86_64)
    unset(X86_64)
    set(X86 1)
  endif()
endif()
# Workaround for 32-bit operating systems on aarch64 processor
if(CMAKE_SIZEOF_VOID_P EQUAL 4 AND AARCH64
    AND NOT OPENCV_FORCE_AARCH64
)
  message(STATUS "sizeof(void) = 4 on 64 bit processor. Assume 32-bit compilation mode")
  if(AARCH64)
    unset(AARCH64)
    set(ARM 1)
  endif()
endif()


# Similar code exists in OpenCVConfig.cmake
if(NOT DEFINED OpenCV_STATIC)
  # look for global setting
  if(NOT DEFINED BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    set(OpenCV_STATIC OFF)
  else()
    set(OpenCV_STATIC ON)
  endif()
endif()

if(DEFINED OpenCV_ARCH AND DEFINED OpenCV_RUNTIME)
  # custom overridden values
elseif(MSVC)
  # see Modules/CMakeGenericSystem.cmake
  if("${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    set(OpenCV_ARCH "x64")
  elseif("${CMAKE_GENERATOR_PLATFORM}" MATCHES "ARM64")
    set(OpenCV_ARCH "ARM64")
  elseif("${CMAKE_GENERATOR}" MATCHES "ARM")
    set(OpenCV_ARCH "ARM")
  elseif("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
    set(OpenCV_ARCH "x64")
  else()
    set(OpenCV_ARCH x86)
  endif()

  if(MSVC_VERSION EQUAL 1400)
    set(OpenCV_RUNTIME vc8)
  elseif(MSVC_VERSION EQUAL 1500)
    set(OpenCV_RUNTIME vc9)
  elseif(MSVC_VERSION EQUAL 1600)
    set(OpenCV_RUNTIME vc10)
  elseif(MSVC_VERSION EQUAL 1700)
    set(OpenCV_RUNTIME vc11)
  elseif(MSVC_VERSION EQUAL 1800)
    set(OpenCV_RUNTIME vc12)
  elseif(MSVC_VERSION EQUAL 1900)
    set(OpenCV_RUNTIME vc14)
  elseif(MSVC_VERSION MATCHES "^191[0-9]$")
    set(OpenCV_RUNTIME vc15)
  elseif(MSVC_VERSION MATCHES "^192[0-9]$")
    set(OpenCV_RUNTIME vc16)
  elseif(MSVC_VERSION MATCHES "^193[0-9]$")
    set(OpenCV_RUNTIME vc17)
  else()
    message(WARNING "OpenCV does not recognize MSVC_VERSION \"${MSVC_VERSION}\". Cannot set OpenCV_RUNTIME")
  endif()
elseif(MINGW)
  set(OpenCV_RUNTIME mingw)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(OpenCV_ARCH x64)
  else()
    set(OpenCV_ARCH x86)
  endif()
endif()

# Fix handling of duplicated files in the same static library:
# https://public.kitware.com/Bug/view.php?id=14874
if(CMAKE_VERSION VERSION_LESS "3.1")
  foreach(var CMAKE_C_ARCHIVE_APPEND CMAKE_CXX_ARCHIVE_APPEND)
    if(${var} MATCHES "^<CMAKE_AR> r")
      string(REPLACE "<CMAKE_AR> r" "<CMAKE_AR> q" ${var} "${${var}}")
    endif()
  endforeach()
endif()

if(ENABLE_CXX11)
  #cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS OFF) # use -std=c++11 instead of -std=gnu++11
  if(CMAKE_CXX11_COMPILE_FEATURES)
    set(HAVE_CXX11 ON)
  endif()
endif()
if(NOT HAVE_CXX11)
  ocv_check_compiler_flag(CXX "" HAVE_CXX11 "${OpenCV_SOURCE_DIR}/cmake/checks/cxx11.cpp")
  if(NOT HAVE_CXX11 AND ENABLE_CXX11)
    ocv_check_compiler_flag(CXX "-std=c++11" HAVE_STD_CXX11 "${OpenCV_SOURCE_DIR}/cmake/checks/cxx11.cpp")
    if(HAVE_STD_CXX11)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      set(HAVE_CXX11 ON)
    endif()
  endif()
endif()

set(__OPENCV_ENABLE_ATOMIC_LONG_LONG OFF)
if(HAVE_CXX11 AND (X86 OR X86_64))
  set(__OPENCV_ENABLE_ATOMIC_LONG_LONG ON)
endif()
option(OPENCV_ENABLE_ATOMIC_LONG_LONG "Enable C++ compiler support for atomic<long long>" ${__OPENCV_ENABLE_ATOMIC_LONG_LONG})

if((HAVE_CXX11 AND OPENCV_ENABLE_ATOMIC_LONG_LONG
        AND NOT MSVC
        AND NOT (X86 OR X86_64)
    AND NOT OPENCV_SKIP_LIBATOMIC_COMPILER_CHECK)
    OR OPENCV_FORCE_LIBATOMIC_COMPILER_CHECK
)
  ocv_check_compiler_flag(CXX "" HAVE_CXX_ATOMICS_WITHOUT_LIB "${OpenCV_SOURCE_DIR}/cmake/checks/atomic_check.cpp")
  if(NOT HAVE_CXX_ATOMICS_WITHOUT_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES atomic)
    ocv_check_compiler_flag(CXX "" HAVE_CXX_ATOMICS_WITH_LIB "${OpenCV_SOURCE_DIR}/cmake/checks/atomic_check.cpp")
    if(HAVE_CXX_ATOMICS_WITH_LIB)
      set(HAVE_ATOMIC_LONG_LONG ON)
      list(APPEND OPENCV_LINKER_LIBS atomic)
    else()
      message(STATUS "Compiler doesn't support std::atomic<long long>")
    endif()
  else()
    set(HAVE_ATOMIC_LONG_LONG ON)
  endif()
else(HAVE_CXX11 AND OPENCV_ENABLE_ATOMIC_LONG_LONG)
  set(HAVE_ATOMIC_LONG_LONG ${OPENCV_ENABLE_ATOMIC_LONG_LONG})
endif()
