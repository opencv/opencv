if(APPLE)
  set(OPENCL_FOUND YES)
  set(OPENCL_LIBRARY "-framework OpenCL" CACHE STRING "OpenCL library")
  set(OPENCL_INCLUDE_DIR "" CACHE STRING "OpenCL include directory")
  mark_as_advanced(OPENCL_INCLUDE_DIR OPENCL_LIBRARY)
else(APPLE)
  #find_package(OpenCL QUIET)

  if (NOT OPENCL_FOUND)
    find_path(OPENCL_ROOT_DIR
              NAMES OpenCL/cl.h CL/cl.h include/CL/cl.h include/nvidia-current/CL/cl.h
              PATHS ENV OCLROOT ENV AMDAPPSDKROOT ENV CUDA_PATH ENV INTELOCLSDKROOT
              DOC "OpenCL root directory"
              NO_DEFAULT_PATH)

    find_path(OPENCL_INCLUDE_DIR
              NAMES OpenCL/cl.h CL/cl.h
              HINTS ${OPENCL_ROOT_DIR}
              PATH_SUFFIXES include include/nvidia-current
              DOC "OpenCL include directory"
              NO_DEFAULT_PATH)

    if(WIN32)
      if(X86_64)
        set(OPENCL_POSSIBLE_LIB_SUFFIXES lib/Win64 lib/x86_64 lib/x64)
      elseif(X86)
        set(OPENCL_POSSIBLE_LIB_SUFFIXES lib/Win32 lib/x86)
      else()
        set(OPENCL_POSSIBLE_LIB_SUFFIXES lib)
      endif()
    elseif(UNIX)
      if(X86_64)
        set(OPENCL_POSSIBLE_LIB_SUFFIXES lib64 lib)
      elseif(X86)
        set(OPENCL_POSSIBLE_LIB_SUFFIXES lib32 lib)
      else()
        set(OPENCL_POSSIBLE_LIB_SUFFIXES lib)
      endif()
    else()
      set(OPENCL_POSSIBLE_LIB_SUFFIXES lib)
    endif()

    find_library(OPENCL_LIBRARY
              NAMES OpenCL
              HINTS ${OPENCL_ROOT_DIR}
              PATH_SUFFIXES ${OPENCL_POSSIBLE_LIB_SUFFIXES}
              DOC "OpenCL library"
              NO_DEFAULT_PATH)

    mark_as_advanced(OPENCL_INCLUDE_DIR OPENCL_LIBRARY)
    include(FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCL DEFAULT_MSG OPENCL_LIBRARY OPENCL_INCLUDE_DIR )
  endif()
endif(APPLE)

if(OPENCL_FOUND)
  set(HAVE_OPENCL 1)
  set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})
  set(OPENCL_LIBRARIES    ${OPENCL_LIBRARY})

  if(WIN32 AND X86_64)
    set(CLAMD_POSSIBLE_LIB_SUFFIXES lib64/import)
  elseif(WIN32)
    set(CLAMD_POSSIBLE_LIB_SUFFIXES lib32/import)
  endif()

  if(X86_64 AND UNIX)
    set(CLAMD_POSSIBLE_LIB_SUFFIXES lib64)
  elseif(X86 AND UNIX)
    set(CLAMD_POSSIBLE_LIB_SUFFIXES lib32)
  endif()

  if(WITH_OPENCLAMDFFT)
    find_path(CLAMDFFT_ROOT_DIR
              NAMES include/clAmdFft.h
              PATHS ENV CLAMDFFT_PATH ENV ProgramFiles
              PATH_SUFFIXES clAmdFft AMD/clAmdFft
              DOC "AMD FFT root directory"
              NO_DEFAULT_PATH)

    find_path(CLAMDFFT_INCLUDE_DIR
              NAMES clAmdFft.h
              HINTS ${CLAMDFFT_ROOT_DIR}
              PATH_SUFFIXES include
              DOC "clAmdFft include directory")

    find_library(CLAMDFFT_LIBRARY
              NAMES clAmdFft.Runtime
              HINTS ${CLAMDFFT_ROOT_DIR}
              PATH_SUFFIXES ${CLAMD_POSSIBLE_LIB_SUFFIXES}
              DOC "clAmdFft library")

    if(CLAMDFFT_LIBRARY AND CLAMDFFT_INCLUDE_DIR)
      set(HAVE_CLAMDFFT 1)
      list(APPEND OPENCL_INCLUDE_DIRS "${CLAMDFFT_INCLUDE_DIR}")
      list(APPEND OPENCL_LIBRARIES    "${CLAMDFFT_LIBRARY}")
    endif()
  endif()

  if(WITH_OPENCLAMDBLAS)
    find_path(CLAMDBLAS_ROOT_DIR
              NAMES include/clAmdBlas.h
              PATHS ENV CLAMDBLAS_PATH ENV ProgramFiles
              PATH_SUFFIXES clAmdBlas AMD/clAmdBlas
              DOC "AMD FFT root directory"
              NO_DEFAULT_PATH)

    find_path(CLAMDBLAS_INCLUDE_DIR
              NAMES clAmdBlas.h
              HINTS ${CLAMDBLAS_ROOT_DIR}
              PATH_SUFFIXES include
              DOC "clAmdFft include directory")

    find_library(CLAMDBLAS_LIBRARY
              NAMES clAmdBlas
              HINTS ${CLAMDBLAS_ROOT_DIR}
              PATH_SUFFIXES ${CLAMD_POSSIBLE_LIB_SUFFIXES}
              DOC "clAmdBlas library")

    if(CLAMDBLAS_LIBRARY AND CLAMDBLAS_INCLUDE_DIR)
      set(HAVE_CLAMDBLAS 1)
      list(APPEND OPENCL_INCLUDE_DIRS "${CLAMDBLAS_INCLUDE_DIR}")
      list(APPEND OPENCL_LIBRARIES    "${CLAMDBLAS_LIBRARY}")
    endif()
  endif()
endif()
