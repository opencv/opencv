ocv_clear_vars(HAVE_ONNX ORT_LIB ORT_INCLUDE ONNX_LIBRARIES ONNX_INCLUDE_DIR)

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
  # For CMake output only
  set(ONNX_LIBRARIES "${ORT_LIB}" CACHE STRING "ONNX Runtime libraries")
  set(ONNX_INCLUDE_DIR "${ORT_INCLUDE}" CACHE STRING "ONNX Runtime include path")

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
  ocv_clear_vars(HAVE_ONNX ORT_LIB ORT_INCLUDE ONNX_LIBRARIES ONNX_INCLUDE_DIR)
endif()
