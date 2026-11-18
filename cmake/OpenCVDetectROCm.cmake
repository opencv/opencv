# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

# ROCm's version is not HIP's. ROCm 10.0 ships HIP 7.15, and HIP's major is 7 on
# both ROCm 7 and ROCm 10, so hip_VERSION cannot identify the platform.

set(ROCM_FOUND FALSE)
set(ROCM_VERSION "")

set(_rocm_root "")
foreach(_candidate "${ROCM_PATH}" "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}" "/opt/rocm")
  if(_candidate AND IS_DIRECTORY "${_candidate}")
    set(_rocm_root "${_candidate}")
    break()
  endif()
endforeach()

if(NOT _rocm_root)
  message(STATUS "ROCm: no install found.")
  return()
endif()

find_package(rocm-core QUIET
  HINTS "${_rocm_root}"
  PATH_SUFFIXES lib/cmake/rocm-core lib64/cmake/rocm-core
  NO_DEFAULT_PATH)
if(rocm-core_FOUND)
  set(ROCM_VERSION "${rocm-core_VERSION}")
endif()

# ROCm <= 7 writes a version file
if(NOT ROCM_VERSION)
  foreach(_version_file
      "${_rocm_root}/.info/version"
      "${_rocm_root}/.info/version-dev"
      "${_rocm_root}/share/doc/rocm-core/VERSION")
    if(NOT ROCM_VERSION AND EXISTS "${_version_file}")
      file(READ "${_version_file}" _raw)
      string(STRIP "${_raw}" _raw)
      if(_raw MATCHES "^([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
        set(ROCM_VERSION "${CMAKE_MATCH_1}")
      endif()
    endif()
  endforeach()
endif()

# ROCm 10 drops that file and installs core-<major>.<minor> instead
if(NOT ROCM_VERSION)
  file(GLOB _core_dirs "${_rocm_root}/core-[0-9]*.[0-9]*")
  foreach(_dir ${_core_dirs})
    if(IS_DIRECTORY "${_dir}" AND _dir MATCHES "core-([0-9]+\\.[0-9]+)$")
      if(NOT ROCM_VERSION OR CMAKE_MATCH_1 VERSION_GREATER ROCM_VERSION)
        set(ROCM_VERSION "${CMAKE_MATCH_1}")
      endif()
    endif()
  endforeach()
endif()

if(NOT ROCM_VERSION)
  message(STATUS "ROCm: found ${_rocm_root}, version unknown.")
  return()
endif()

string(REGEX MATCH "^[0-9]+" ROCM_VERSION_MAJOR "${ROCM_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" ROCM_VERSION_MINOR "${ROCM_VERSION}")

set(ROCM_FOUND TRUE)
set(ROCM_ROOT_DIR "${_rocm_root}")

unset(_rocm_root)
unset(_raw)
unset(_core_dirs)
