if(APPLE)
  set(OPENCL_FOUND YES)
  set(OPENCL_LIBRARY "-framework OpenCL" CACHE STRING "OpenCL library")
  set(OPENCL_INCLUDE_DIR "" CACHE STRING "OpenCL include directory")
  mark_as_advanced(OPENCL_INCLUDE_DIR OPENCL_LIBRARY)
else(APPLE)
  #find_package(OpenCL QUIET)

  if(NOT OPENCL_FOUND)
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

    set(OPENCL_LIBRARY "OPENCL_DYNAMIC_LOAD")

    mark_as_advanced(OPENCL_INCLUDE_DIR OPENCL_LIBRARY)
    include(FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCL DEFAULT_MSG OPENCL_LIBRARY OPENCL_INCLUDE_DIR )
  endif()
endif(APPLE)

if(OPENCL_FOUND)
  try_compile(HAVE_OPENCL11
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/opencl11.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OPENCL_INCLUDE_DIR}"
    )
  if(NOT HAVE_OPENCL11)
    message(STATUS "OpenCL 1.1 not found, ignore OpenCL SDK")
    return()
  endif()
  try_compile(HAVE_OPENCL12
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/opencl12.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OPENCL_INCLUDE_DIR}"
    )
  if(NOT HAVE_OPENCL12)
    message(STATUS "OpenCL 1.2 not found, will use OpenCL 1.1")
  endif()

  set(HAVE_OPENCL 1)
  set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})
  if(OPENCL_LIBRARY MATCHES "OPENCL_DYNAMIC_LOAD")
    unset(OPENCL_LIBRARIES)
  else()
    set(OPENCL_LIBRARIES "${OPENCL_LIBRARY}")
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
endif()
