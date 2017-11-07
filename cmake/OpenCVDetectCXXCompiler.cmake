# ----------------------------------------------------------------------------
# Detect Microsoft compiler:
# ----------------------------------------------------------------------------
if(CMAKE_CL_64)
    set(MSVC64 1)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_GNUCXX 1)
  set(CMAKE_COMPILER_IS_CLANGCXX 1)
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_GNUCC 1)
  set(CMAKE_COMPILER_IS_CLANGCC 1)
endif()

# ----------------------------------------------------------------------------
# Detect Intel ICC compiler -- for -fPIC in 3rdparty ( UNIX ONLY ):
#  see  include/opencv/cxtypes.h file for related   ICC & CV_ICC defines.
# NOTE: The system needs to determine if the '-fPIC' option needs to be added
#  for the 3rdparty static libs being compiled.  The CMakeLists.txt files
#  in 3rdparty use the CV_ICC definition being set here to determine if
#  the -fPIC flag should be used.
# ----------------------------------------------------------------------------
if(UNIX)
  if  (__ICL)
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

if(NOT DEFINED CMAKE_CXX_COMPILER_VERSION)
  message(WARNING "Compiler version is not available: CMAKE_CXX_COMPILER_VERSION is not set")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  if(WIN32)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
              OUTPUT_VARIABLE OPENCV_GCC_TARGET_MACHINE
              OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(OPENCV_GCC_TARGET_MACHINE MATCHES "amd64|x86_64|AMD64")
      set(MINGW64 1)
    endif()
  endif()
endif()

if(MSVC64 OR MINGW64)
  set(X86_64 1)
elseif(MINGW OR (MSVC AND NOT CMAKE_CROSSCOMPILING))
  set(X86 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(X86 1)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|ARM.*")
  set(ARM 1)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64.*|AARCH64.*")
  set(AARCH64 1)
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
  # custom overrided values
elseif(MSVC)
  if(CMAKE_CL_64)
    set(OpenCV_ARCH x64)
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
  endif()
elseif(MINGW)
  set(OpenCV_RUNTIME mingw)

  if(MINGW64)
    set(OpenCV_ARCH x64)
  else()
    set(OpenCV_ARCH x86)
  endif()
endif()
