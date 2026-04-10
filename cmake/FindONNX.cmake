ocv_clear_vars(HAVE_ONNX ORT_LIB ORT_INCLUDE ONNX_LIBRARIES ONNX_INCLUDE_DIR ONNX_VERSION)

set(ONNXRT_ROOT_DIR "" CACHE PATH "ONNX Runtime install directory")

# For now, check the old name ORT_INSTALL_DIR
if(ORT_INSTALL_DIR AND NOT ONNXRT_ROOT_DIR)
  set(ONNXRT_ROOT_DIR ${ORT_INSTALL_DIR})
endif()

if(NOT ONNXRT_ROOT_DIR AND DEFINED OpenCV_BINARY_DIR)
  if(EXISTS "${OpenCV_BINARY_DIR}/3rdparty/onnxruntime")
    file(GLOB __ort_candidates LIST_DIRECTORIES true
      "${OpenCV_BINARY_DIR}/3rdparty/onnxruntime/onnxruntime-*")
    list(LENGTH __ort_candidates __ort_candidates_len)
    if(__ort_candidates_len GREATER 0)
      list(GET __ort_candidates 0 ONNXRT_ROOT_DIR)
    endif()
    unset(__ort_candidates)
    unset(__ort_candidates_len)
  endif()
endif()

set(__ort_hint_roots "")
if(ONNXRT_ROOT_DIR)
  list(APPEND __ort_hint_roots "${ONNXRT_ROOT_DIR}")
endif()

# Prefer CMake config packages if present i.e system-installed ORT
find_package(onnxruntime CONFIG QUIET)
find_package(ONNXRuntime CONFIG QUIET)

set(__ort_target "")
foreach(t
    onnxruntime::onnxruntime
    ONNXRuntime::onnxruntime
    ONNXRuntime::onnxruntime_shared
)
  if(TARGET ${t})
    set(__ort_target "${t}")
    break()
  endif()
endforeach()

if(__ort_target)
  get_target_property(ORT_INCLUDE ${__ort_target} INTERFACE_INCLUDE_DIRECTORIES)
  if(ORT_INCLUDE AND NOT IS_DIRECTORY "${ORT_INCLUDE}")
    list(GET ORT_INCLUDE 0 ORT_INCLUDE)
  endif()
  get_target_property(ORT_LIB ${__ort_target} IMPORTED_LOCATION)
  if(NOT ORT_LIB)
    get_target_property(ORT_LIB ${__ort_target} IMPORTED_LOCATION_RELEASE)
  endif()
  if(NOT ORT_LIB)
    get_target_property(ORT_LIB ${__ort_target} LOCATION)
  endif()
endif()

# Locate headers and libraries via find_* in system paths and/or ONNXRT_ROOT_DIR.
if(NOT ORT_LIB)
  find_library(ORT_LIB NAMES onnxruntime
    HINTS ${__ort_hint_roots}
    PATH_SUFFIXES lib lib64
    CMAKE_FIND_ROOT_PATH_BOTH)
endif()

if(NOT ORT_INCLUDE)
  find_path(ORT_INCLUDE NAMES onnxruntime_cxx_api.h
    HINTS ${__ort_hint_roots}
    PATH_SUFFIXES
      include
      include/onnxruntime
      include/onnxruntime/core/session
    CMAKE_FIND_ROOT_PATH_BOTH)
endif()

unset(__ort_hint_roots)

macro(detect_onxxrt_ep filename dir have_ep_var)
    find_path(ORT_EP_INCLUDE ${filename} ${dir} CMAKE_FIND_ROOT_PATH_BOTH)
    if(ORT_EP_INCLUDE)
       set(${have_ep_var} TRUE)
    endif()
endmacro()

if(ORT_LIB AND ORT_INCLUDE)
  set(__ort_root_for_ep "${ONNXRT_ROOT_DIR}")
  if(NOT __ort_root_for_ep)
    set(__ort_root_for_ep "${ORT_INCLUDE}")
    string(REGEX REPLACE "(/include/onnxruntime.*)$" "" __ort_root_for_ep "${__ort_root_for_ep}")
  endif()

  # Check DirectML Execution Provider availability
  get_filename_component(dml_dir ${__ort_root_for_ep}/include/onnxruntime/core/providers/dml ABSOLUTE)
  detect_onxxrt_ep(
      dml_provider_factory.h
      ${dml_dir}
      HAVE_ONNX_DML
  )

  # Check CoreML Execution Provider availability
  get_filename_component(coreml_dir ${__ort_root_for_ep}/include/onnxruntime/core/providers/coreml ABSOLUTE)
  detect_onxxrt_ep(
      coreml_provider_factory.h
      ${coreml_dir}
      HAVE_ONNX_COREML
  )

  set(HAVE_ONNX TRUE)
  # Try to report a human-readable ONNX Runtime version in diagnostics/build info.
  if(DEFINED onnxruntime_VERSION AND onnxruntime_VERSION)
    set(ONNX_VERSION "${onnxruntime_VERSION}")
  elseif(DEFINED ONNXRuntime_VERSION AND ONNXRuntime_VERSION)
    set(ONNX_VERSION "${ONNXRuntime_VERSION}")
  elseif(DEFINED ONNXRUNTIME_VERSION AND ONNXRUNTIME_VERSION)
    set(ONNX_VERSION "${ONNXRUNTIME_VERSION}")
  else()
    set(__ort_version_candidates "")
    if(ONNXRT_ROOT_DIR)
      list(APPEND __ort_version_candidates "${ONNXRT_ROOT_DIR}")
    endif()
    if(ORT_LIB)
      list(APPEND __ort_version_candidates "${ORT_LIB}")
    endif()
    foreach(__ort_version_candidate ${__ort_version_candidates})
      string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+([.-][0-9A-Za-z]+)?)" __ort_version_match "${__ort_version_candidate}")
      if(__ort_version_match)
        set(ONNX_VERSION "${__ort_version_match}")
        break()
      endif()
    endforeach()
    unset(__ort_version_candidate)
    unset(__ort_version_candidates)
    unset(__ort_version_match)
  endif()

  # For CMake output only
  set(ONNX_LIBRARIES "${ORT_LIB}" CACHE STRING "ONNX Runtime libraries")
  set(ONNX_INCLUDE_DIR "${ORT_INCLUDE}" CACHE STRING "ONNX Runtime include path")
  if(NOT ONNX_VERSION)
    set(ONNX_VERSION "unknown")
  endif()
  set(ONNX_VERSION "${ONNX_VERSION}" CACHE STRING "ONNX Runtime version")

  # Link target with associated interface headers
  set(ONNX_LIBRARY "onnxruntime" CACHE STRING "ONNX Link Target")
  if(NOT TARGET ${ONNX_LIBRARY})
    ocv_add_library(${ONNX_LIBRARY} SHARED IMPORTED)
  endif()

  if(WIN32)
    # ORT_LIB is typically the import library (.lib). Prefer matching runtime DLL if available.
    set(__ort_dll "")
    if(ONNXRT_ROOT_DIR)
      find_file(__ort_dll NAMES onnxruntime.dll HINTS "${ONNXRT_ROOT_DIR}" PATH_SUFFIXES bin)
    endif()
    if(__ort_dll)
      set_target_properties(${ONNX_LIBRARY} PROPERTIES
                            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}"
                            IMPORTED_LOCATION "${__ort_dll}"
                            IMPORTED_IMPLIB "${ORT_LIB}")
    else()
      set_target_properties(${ONNX_LIBRARY} PROPERTIES
                            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}"
                            IMPORTED_LOCATION "${ORT_LIB}"
                            IMPORTED_IMPLIB "${ORT_LIB}")
    endif()
    unset(__ort_dll)
  else()
    set_target_properties(${ONNX_LIBRARY} PROPERTIES
                          INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}"
                          IMPORTED_LOCATION "${ORT_LIB}")
  endif()
  unset(__ort_root_for_ep)
endif()

if(NOT HAVE_ONNX)
  ocv_clear_vars(HAVE_ONNX ORT_LIB ORT_INCLUDE ONNX_LIBRARIES ONNX_INCLUDE_DIR ONNX_VERSION)
endif()

# ONNX Runtime GenAI
ocv_clear_vars(HAVE_ONNXGENAI ORTGA_LIB ORTGA_INCLUDE ONNXGENAI_LIBRARIES ONNXGENAI_INCLUDE_DIR ONNXGENAI_VERSION)

set(ONNXGENAI_ROOT_DIR "" CACHE PATH "ONNX Runtime GenAI install directory")

if(NOT ONNXGENAI_ROOT_DIR AND DEFINED OpenCV_BINARY_DIR)
  if(EXISTS "${OpenCV_BINARY_DIR}/3rdparty/onnxruntime-genai")
    file(GLOB __ortga_candidates LIST_DIRECTORIES true
      "${OpenCV_BINARY_DIR}/3rdparty/onnxruntime-genai/onnxruntime-genai-*")
    list(LENGTH __ortga_candidates __ortga_candidates_len)
    if(__ortga_candidates_len GREATER 0)
      list(GET __ortga_candidates 0 ONNXGENAI_ROOT_DIR)
    endif()
    unset(__ortga_candidates)
    unset(__ortga_candidates_len)
  endif()
endif()

set(__ortga_hint_roots "")
if(ONNXGENAI_ROOT_DIR)
  list(APPEND __ortga_hint_roots "${ONNXGENAI_ROOT_DIR}")
endif()

if(NOT ORTGA_LIB)
  find_library(ORTGA_LIB NAMES onnxruntime-genai
    HINTS ${__ortga_hint_roots}
    PATH_SUFFIXES lib lib64
    CMAKE_FIND_ROOT_PATH_BOTH)
endif()

if(NOT ORTGA_INCLUDE)
  find_path(ORTGA_INCLUDE NAMES ort_genai.h
    HINTS ${__ortga_hint_roots}
    PATH_SUFFIXES include
    CMAKE_FIND_ROOT_PATH_BOTH)
endif()

unset(__ortga_hint_roots)

if(ORTGA_LIB AND ORTGA_INCLUDE)
  set(HAVE_ONNXGENAI TRUE)

  set(__ortga_version_candidates "")
  if(ONNXGENAI_ROOT_DIR)
    list(APPEND __ortga_version_candidates "${ONNXGENAI_ROOT_DIR}")
  endif()
  if(ORTGA_LIB)
    list(APPEND __ortga_version_candidates "${ORTGA_LIB}")
  endif()
  foreach(__ortga_version_candidate ${__ortga_version_candidates})
    string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+([.-][0-9A-Za-z]+)?)" __ortga_version_match "${__ortga_version_candidate}")
    if(__ortga_version_match)
      set(ONNXGENAI_VERSION "${__ortga_version_match}")
      break()
    endif()
  endforeach()
  unset(__ortga_version_candidate)
  unset(__ortga_version_candidates)
  unset(__ortga_version_match)

  set(ONNXGENAI_LIBRARIES   "${ORTGA_LIB}"    CACHE STRING "ONNX Runtime GenAI libraries")
  set(ONNXGENAI_INCLUDE_DIR "${ORTGA_INCLUDE}" CACHE STRING "ONNX Runtime GenAI include path")
  if(NOT ONNXGENAI_VERSION)
    set(ONNXGENAI_VERSION "unknown")
  endif()
  set(ONNXGENAI_VERSION "${ONNXGENAI_VERSION}" CACHE STRING "ONNX Runtime GenAI version")

  set(ONNXGENAI_LIBRARY "onnxruntime-genai" CACHE STRING "ONNX Runtime GenAI link target")
  if(NOT TARGET ${ONNXGENAI_LIBRARY})
    ocv_add_library(${ONNXGENAI_LIBRARY} SHARED IMPORTED)
  endif()

  if(WIN32)
    set(__ortga_dll "")
    if(ONNXGENAI_ROOT_DIR)
      find_file(__ortga_dll NAMES onnxruntime-genai.dll
        HINTS "${ONNXGENAI_ROOT_DIR}"
        PATH_SUFFIXES bin)
    endif()
    if(__ortga_dll)
      set_target_properties(${ONNXGENAI_LIBRARY} PROPERTIES
                            INTERFACE_INCLUDE_DIRECTORIES "${ORTGA_INCLUDE}"
                            IMPORTED_LOCATION             "${__ortga_dll}"
                            IMPORTED_IMPLIB               "${ORTGA_LIB}")
    else()
      set_target_properties(${ONNXGENAI_LIBRARY} PROPERTIES
                            INTERFACE_INCLUDE_DIRECTORIES "${ORTGA_INCLUDE}"
                            IMPORTED_LOCATION             "${ORTGA_LIB}"
                            IMPORTED_IMPLIB               "${ORTGA_LIB}")
    endif()
    unset(__ortga_dll)
  else()
    set_target_properties(${ONNXGENAI_LIBRARY} PROPERTIES
                          INTERFACE_INCLUDE_DIRECTORIES "${ORTGA_INCLUDE}"
                          IMPORTED_LOCATION             "${ORTGA_LIB}")
  endif()
endif()

if(NOT HAVE_ONNXGENAI)
  ocv_clear_vars(HAVE_ONNXGENAI ORTGA_LIB ORTGA_INCLUDE ONNXGENAI_LIBRARIES ONNXGENAI_INCLUDE_DIR ONNXGENAI_VERSION)
endif()
