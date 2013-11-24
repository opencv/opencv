if(APPLE)
  set(OPENCL_FOUND YES)
  set(OPENCL_LIBRARY "-framework OpenCL" CACHE STRING "OpenCL library")
  set(OPENCL_INCLUDE_DIR "" CACHE STRING "OpenCL include directory")
  mark_as_advanced(OPENCL_INCLUDE_DIR OPENCL_LIBRARY)
  set(HAVE_OPENCL_STATIC ON)
else(APPLE)
  set(OPENCL_FOUND YES)
  set(HAVE_OPENCL_STATIC OFF)
  set(OPENCL_INCLUDE_DIR "${OpenCV_SOURCE_DIR}/3rdparty/include/opencl/1.2")
endif(APPLE)

if(OPENCL_FOUND)
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

  if(HAVE_OPENCL_STATIC)
    set(OPENCL_LIBRARIES "${OPENCL_LIBRARY}")
  else()
    unset(OPENCL_LIBRARIES)
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
endif()
