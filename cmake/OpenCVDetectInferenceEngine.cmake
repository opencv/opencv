# The script detects Intel(R) Inference Engine installation
#
# Cache variables:
# INF_ENGINE_RELEASE - a number reflecting IE source interface (linked with OpenVINO release)
#
# Detect parameters:
# 1. Native cmake IE package:
#    - environment variable InferenceEngine_DIR is set to location of cmake module
# 2. Custom location:
#    - INF_ENGINE_INCLUDE_DIRS - headers search location
#    - INF_ENGINE_LIB_DIRS     - library search location
# 3. OpenVINO location:
#    - environment variable INTEL_OPENVINO_DIR is set to location of OpenVINO installation dir
#    - INF_ENGINE_PLATFORM - part of name of library directory representing its platform
#
# Result:
# INF_ENGINE_TARGET - set to name of imported library target representing InferenceEngine
#

if(NOT HAVE_CXX11)
    message(WARNING "DL Inference engine requires C++11. You can turn it on via ENABLE_CXX11=ON CMake flag.")
    return()
endif()

# =======================

function(add_custom_ie_build _inc _lib _lib_rel _lib_dbg _msg)
  if(NOT _inc OR NOT (_lib OR _lib_rel OR _lib_dbg))
    return()
  endif()
  add_library(inference_engine UNKNOWN IMPORTED)
  set_target_properties(inference_engine PROPERTIES
    IMPORTED_LOCATION "${_lib}"
    IMPORTED_IMPLIB_RELEASE "${_lib_rel}"
    IMPORTED_IMPLIB_DEBUG "${_lib_dbg}"
    INTERFACE_INCLUDE_DIRECTORIES "${_inc}"
  )

  set(custom_libraries "")
  file(GLOB libraries "${INF_ENGINE_LIB_DIRS}/${CMAKE_SHARED_LIBRARY_PREFIX}inference_engine_*${CMAKE_SHARED_LIBRARY_SUFFIX}")
  foreach(full_path IN LISTS libraries)
    get_filename_component(library "${full_path}" NAME_WE)
    string(REPLACE "${CMAKE_SHARED_LIBRARY_PREFIX}" "" library "${library}")
    add_library(${library} UNKNOWN IMPORTED)
    set_target_properties(${library} PROPERTIES
      IMPORTED_LOCATION "${full_path}")
    list(APPEND custom_libraries ${library})
  endforeach()

  if(NOT INF_ENGINE_RELEASE VERSION_GREATER "2018050000")
    find_library(INF_ENGINE_OMP_LIBRARY iomp5 PATHS "${INF_ENGINE_OMP_DIR}" NO_DEFAULT_PATH)
    if(NOT INF_ENGINE_OMP_LIBRARY)
      message(WARNING "OpenMP for IE have not been found. Set INF_ENGINE_OMP_DIR variable if you experience build errors.")
    else()
      set_target_properties(inference_engine PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${INF_ENGINE_OMP_LIBRARY}")
    endif()
  endif()
  set(INF_ENGINE_VERSION "Unknown" CACHE STRING "")
  set(INF_ENGINE_TARGET "inference_engine;${custom_libraries}" PARENT_SCOPE)
  message(STATUS "Detected InferenceEngine: ${_msg}")
endfunction()

# ======================

function(download_ie)
  set(ie_src_dir "${OpenCV_BINARY_DIR}/3rdparty/dldt")
  set(ie_subdir "dldt-2019_R2")
  ocv_download(FILENAME "2019_R2.zip"
               HASH "c486380deb75503adca5d7e88763e4a5"
               URL
                 "${OPENCV_IE_URL}"
                 "$ENV{OPENCV_IE_URL}"
                 "https://github.com/opencv/dldt/archive/"
               DESTINATION_DIR ${ie_src_dir}
               ID IE
               STATUS res
               UNPACK RELATIVE_URL)

  if (NOT res)
      return()
  endif()

  # This is a minor patch to IE's cmake files.
  file(READ "${ie_src_dir}/${ie_subdir}/inference-engine/cmake/dependencies.cmake" filedata)
  string(REPLACE "CMAKE_SOURCE_DIR" "PROJECT_SOURCE_DIR" filedata "${filedata}")
  file(WRITE "${ie_src_dir}/${ie_subdir}/inference-engine/cmake/dependencies.cmake" ${filedata})

  file(READ "${ie_src_dir}/${ie_subdir}/inference-engine/src/inference_engine/CMakeLists.txt" filedata)
  string(REPLACE "CMAKE_SOURCE_DIR" "PROJECT_SOURCE_DIR" filedata "${filedata}")
  string(REPLACE "PRIVATE ade)" "PRIVATE ade PUBLIC pugixml)" filedata "${filedata}")
  file(WRITE "${ie_src_dir}/${ie_subdir}/inference-engine/src/inference_engine/CMakeLists.txt" ${filedata})

  file(READ "${ie_src_dir}/${ie_subdir}/inference-engine/CMakeLists.txt" filedata)
  string(REPLACE "add_subdirectory(tests)" "" filedata "${filedata}")  # Disable tests
  file(WRITE "${ie_src_dir}/${ie_subdir}/inference-engine/CMakeLists.txt" ${filedata})

  # Enable extensions library
  file(READ "${ie_src_dir}/${ie_subdir}/inference-engine/src/CMakeLists.txt" filedata)
  string(REPLACE "add_subdirectory(extension EXCLUDE_FROM_ALL)" "add_subdirectory(extension)" filedata "${filedata}")
  file(WRITE "${ie_src_dir}/${ie_subdir}/inference-engine/src/CMakeLists.txt" ${filedata})

  # Download ADE
  set(ade_src_dir "${ie_src_dir}/${ie_subdir}/inference-engine/thirdparty/ade")
  set(ade_filename "v0.1.1d.zip")
  set(ade_subdir "ade-0.1.1d")
  set(ade_md5 "37479d90e3a5d47f132f512b22cbe206")
  ocv_download(FILENAME ${ade_filename}
               HASH ${ade_md5}
               URL
                 "${OPENCV_ADE_URL}"
                 "$ENV{OPENCV_ADE_URL}"
                 "https://github.com/opencv/ade/archive/"
               DESTINATION_DIR ${ade_src_dir}
               ID ADE
               STATUS res
  UNPACK RELATIVE_URL)

  if (NOT res)
      return()
  endif()

  if(EXISTS "${ade_src_dir}/${ade_subdir}")
    file(RENAME "${ade_src_dir}/${ade_subdir}" "${ade_src_dir}_tmp")
    file(RENAME "${ade_src_dir}_tmp" "${ade_src_dir}")
  endif()

  set(ENABLE_GNA OFF)
  set(ENABLE_PROFILING_ITT OFF)
  set(ENABLE_SAMPLES_CORE OFF)
  set(ENABLE_SEGMENTATION_TESTS OFF)
  set(ENABLE_OBJECT_DETECTION_TESTS OFF)
  set(ENABLE_OPENCV OFF)
  set(BUILD_TESTS OFF)
  set(BUILD_SHARED_LIBS OFF)  # pugixml

  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated -Wmissing-prototypes -Wmissing-declarations -Wshadow
                                       -Wunused-parameter -Wsign-compare -Wstrict-prototypes -Wnon-virtual-dtor
                                       -Wundef -Wstrict-aliasing -Wsign-promo -Wreorder -Wunused-variable
                                       -Wunknown-pragmas -Wstrict-overflow -Wextra -Wunused-local-typedefs
                                       -Wunused-function -Wsequence-point -Wunused-but-set-variable -Wparentheses
                                       -Wsuggest-override)
  ocv_warnings_disable(CMAKE_C_FLAGS -Wstrict-prototypes)

  add_subdirectory(${OpenCV_BINARY_DIR}/3rdparty/dldt/${ie_subdir}/inference-engine
                   ${OpenCV_BINARY_DIR}/3rdparty/dldt)
  set_target_properties(MKLDNNPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
  set_target_properties(clDNNPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
  set_target_properties(ie_cpu_extension PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")

  set(INF_ENGINE_TARGET inference_engine_s PARENT_SCOPE)
  set(INF_ENGINE_RELEASE "2018050000" PARENT_SCOPE)
endfunction()

# ======================

find_package(InferenceEngine QUIET)
if(InferenceEngine_FOUND)
  set(INF_ENGINE_TARGET ${InferenceEngine_LIBRARIES})
  set(INF_ENGINE_VERSION "${InferenceEngine_VERSION}" CACHE STRING "")
  message(STATUS "Detected InferenceEngine: cmake package (${InferenceEngine_VERSION})")
endif()

if(NOT INF_ENGINE_TARGET AND INF_ENGINE_LIB_DIRS AND INF_ENGINE_INCLUDE_DIRS)
  find_path(ie_custom_inc "inference_engine.hpp" PATHS "${INF_ENGINE_INCLUDE_DIRS}" NO_DEFAULT_PATH)
  find_library(ie_custom_lib "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}" NO_DEFAULT_PATH)
  find_library(ie_custom_lib_rel "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}/Release" NO_DEFAULT_PATH)
  find_library(ie_custom_lib_dbg "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}/Debug" NO_DEFAULT_PATH)
  add_custom_ie_build("${ie_custom_inc}" "${ie_custom_lib}" "${ie_custom_lib_rel}" "${ie_custom_lib_dbg}" "INF_ENGINE_{INCLUDE,LIB}_DIRS")
endif()

set(_loc "$ENV{INTEL_OPENVINO_DIR}")
if(NOT _loc AND DEFINED ENV{INTEL_CVSDK_DIR})
  set(_loc "$ENV{INTEL_CVSDK_DIR}")  # OpenVINO 2018.x
endif()
if(NOT INF_ENGINE_TARGET AND _loc)
  if(NOT INF_ENGINE_RELEASE VERSION_GREATER "2018050000")
    set(INF_ENGINE_PLATFORM_DEFAULT "ubuntu_16.04")
  else()
    set(INF_ENGINE_PLATFORM_DEFAULT "")
  endif()
  set(INF_ENGINE_PLATFORM "${INF_ENGINE_PLATFORM_DEFAULT}" CACHE STRING "InferenceEngine platform (library dir)")
  find_path(ie_custom_env_inc "inference_engine.hpp" PATHS "${_loc}/deployment_tools/inference_engine/include" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/${INF_ENGINE_PLATFORM}/intel64" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib_rel "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/intel64/Release" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib_dbg "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/intel64/Debug" NO_DEFAULT_PATH)
  add_custom_ie_build("${ie_custom_env_inc}" "${ie_custom_env_lib}" "${ie_custom_env_lib_rel}" "${ie_custom_env_lib_dbg}" "OpenVINO (${_loc})")
endif()

if(NOT INF_ENGINE_TARGET)
  download_ie()
endif()

# Add more features to the target

if(INF_ENGINE_TARGET)
  if(NOT INF_ENGINE_RELEASE)
    message(WARNING "InferenceEngine version has not been set, 2020.1 will be used by default. Set INF_ENGINE_RELEASE variable if you experience build errors.")
  endif()
  set(INF_ENGINE_RELEASE "2020010000" CACHE STRING "Force IE version, should be in form YYYYAABBCC (e.g. 2020.1.0.2 -> 2020010002)")
  set_target_properties(${INF_ENGINE_TARGET} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "HAVE_INF_ENGINE=1;INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}"
  )
endif()

if(WITH_NGRAPH)
  find_package(ngraph QUIET)
  if(ngraph_FOUND)
    ocv_assert(TARGET ngraph::ngraph)
    if(INF_ENGINE_RELEASE VERSION_LESS "2019039999")
      message(WARNING "nGraph is not tested with current InferenceEngine version: INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}")
    endif()
    message(STATUS "Detected ngraph: cmake package (${ngraph_VERSION})")
    set(HAVE_NGRAPH ON)
  endif()
endif()
