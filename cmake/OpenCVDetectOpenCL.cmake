set(OPENCL_FOUND ON CACHE BOOL "OpenCL library is found")

option(OPENCV_OPENCL_USE_KHR_HEADERS "Use Khronos OpenCL-Headers instead of bundled 1.2" ON)
set(OPENCV_OPENCL_HEADERS_DIR "" CACHE PATH "Root directory of Khronos OpenCL-Headers (CL/cl.h)")
set(OPENCV_OPENCL_TARGET_VERSION 300 CACHE STRING "OpenCL target version (e.g., 300)")

if(APPLE)
  set(OPENCL_LIBRARY "-framework OpenCL" CACHE STRING "OpenCL library")
  set(OPENCL_INCLUDE_DIR "" CACHE PATH "OpenCL include directory")
else()
  set(OPENCL_LIBRARY "" CACHE STRING "OpenCL library")
  if(OPENCV_OPENCL_USE_KHR_HEADERS)
    if(NOT OPENCV_OPENCL_HEADERS_DIR)
      set(OPENCV_OPENCL_HEADERS_DIR "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/3.0")
    endif()
    if(EXISTS "${OPENCV_OPENCL_HEADERS_DIR}/CL/cl.h")
      set(OPENCL_INCLUDE_DIR "${OPENCV_OPENCL_HEADERS_DIR}" CACHE PATH "OpenCL include directory" FORCE)
      message(STATUS "Using Khronos OpenCL headers from ${OPENCL_INCLUDE_DIR}")

      if(EXISTS "${OPENCV_OPENCL_HEADERS_DIR}/LICENSE")
        ocv_install_3rdparty_licenses(opencl-headers "${OPENCV_OPENCL_HEADERS_DIR}/LICENSE")
      elseif(EXISTS "${OPENCV_OPENCL_HEADERS_DIR}/LICENSE.txt")
        ocv_install_3rdparty_licenses(opencl-headers "${OPENCV_OPENCL_HEADERS_DIR}/LICENSE.txt")
      endif()

    else()
      message(WARNING "Khronos OpenCL headers not found, falling back to bundled 1.2")
      set(OPENCL_INCLUDE_DIR "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/1.2" CACHE PATH "OpenCL include directory" FORCE)
      ocv_install_3rdparty_licenses(opencl-headers "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/LICENSE.txt")
    endif()
  else()
    set(OPENCL_INCLUDE_DIR "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/1.2" CACHE PATH "OpenCL include directory" FORCE)
    ocv_install_3rdparty_licenses(opencl-headers "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/LICENSE.txt")
  endif()
endif()
mark_as_advanced(OPENCL_INCLUDE_DIR OPENCL_LIBRARY)

if(OPENCL_FOUND)

  if(OPENCL_LIBRARY)
    set(HAVE_OPENCL_STATIC ON)
    set(OPENCL_LIBRARIES "${OPENCL_LIBRARY}")
  else()
    set(HAVE_OPENCL_STATIC OFF)
  endif()

  if(NOT HAVE_OPENCL_STATIC)
    try_compile(__VALID_OPENCL
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/opencl.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OPENCL_INCLUDE_DIR}"
      OUTPUT_VARIABLE TRY_OUT
      )
    if(NOT TRY_OUT MATCHES "OpenCL is valid")
      message(WARNING "Can't use OpenCL")
      return()
    endif()
  endif()

  set(HAVE_OPENCL 1)

  if(OPENCV_OPENCL_USE_KHR_HEADERS AND OPENCV_OPENCL_TARGET_VERSION)
    set(OPENCL_TARGET_DEFINITIONS "-DCL_TARGET_OPENCL_VERSION=${OPENCV_OPENCL_TARGET_VERSION}")
    message(STATUS "Setting CL_TARGET_OPENCL_VERSION=${OPENCV_OPENCL_TARGET_VERSION}")
  else()
    set(OPENCL_TARGET_DEFINITIONS "")
  endif()

  if(WITH_OPENCL_SVM)
    set(HAVE_OPENCL_SVM 1)
  endif()

  set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})

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

    if(CLAMDFFT_INCLUDE_DIR)
      set(HAVE_CLAMDFFT 1)
      list(APPEND OPENCL_INCLUDE_DIRS "${CLAMDFFT_INCLUDE_DIR}")
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

    if(CLAMDBLAS_INCLUDE_DIR)
      set(HAVE_CLAMDBLAS 1)
      list(APPEND OPENCL_INCLUDE_DIRS "${CLAMDBLAS_INCLUDE_DIR}")
    endif()
  endif()

  # check WITH_OPENCL_D3D11_NV is located in OpenCVDetectDirectX.cmake file

  if(WITH_VA_INTEL AND HAVE_VA)
    if(HAVE_OPENCL AND EXISTS "${OPENCL_INCLUDE_DIR}/CL/cl_va_api_media_sharing_intel.h")
      set(HAVE_VA_INTEL ON)
    elseif(HAVE_OPENCL AND EXISTS "${OPENCL_INCLUDE_DIR}/CL/va_ext.h")
      set(HAVE_VA_INTEL ON)
      set(HAVE_VA_INTEL_OLD_HEADER ON)
    endif()
  endif()

endif()
