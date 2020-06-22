# template taken from https://cmake.org/cmake/help/v3.14/manual/cmake-developer.7.html

# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCUDNN
---------

Finds the cuDNN library.

Requires:
^^^^^^^^^

find_cuda_helper_libs from FindCUDA.cmake
i.e. CUDA module should be found using FindCUDA.cmake before attempting to find cuDNN

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``CUDNN_FOUND``
``CUDNN_INCLUDE_DIRS``    location of cudnn.h
``CUDNN_LIBRARIES``       location of cudnn library

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables will be set if cuDNN was found. They may also be set on failure.

``CUDNN_LIBRARY``
``CUDNN_INCLUDE_DIR``
``CUDNN_VERSION``

``CUDNN_VERSION_MAJOR`` INTERNAL
``CUDNN_VERSION_MINOR`` INTERNAL
``CUDNN_VERSION_PATCH`` INTERNAL

#]=======================================================================]

# find the library
if(CUDA_FOUND)
  find_cuda_helper_libs(cudnn)
  set(CUDNN_LIBRARY ${CUDA_cudnn_LIBRARY} CACHE FILEPATH "location of the cuDNN library")
  unset(CUDA_cudnn_LIBRARY CACHE)
endif()

# find the include
if(CUDNN_LIBRARY)
  find_path(CUDNN_INCLUDE_DIR
    cudnn.h
    PATHS ${CUDA_TOOLKIT_INCLUDE}
    DOC "location of cudnn.h"
    NO_DEFAULT_PATH
  )

  if(NOT CUDNN_INCLUDE_DIR)
    find_path(CUDNN_INCLUDE_DIR
      cudnn.h
      DOC "location of cudnn.h"
    )
  endif()
endif()

# extract version from the include
if(CUDNN_INCLUDE_DIR)
  if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_H_CONTENTS)
  else()
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_H_CONTENTS)
  endif()

  string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_MINOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_PATCH_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

  set(CUDNN_VERSION
    "${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCH_VERSION}"
    CACHE
    STRING
    "cuDNN version"
  )

  unset(CUDNN_H_CONTENTS)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS
    CUDNN_LIBRARY
    CUDNN_INCLUDE_DIR
  VERSION_VAR CUDNN_VERSION
)

if(CUDNN_FOUND)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
endif()

mark_as_advanced(
  CUDNN_LIBRARY
  CUDNN_INCLUDE_DIR
  CUDNN_VERSION
)
