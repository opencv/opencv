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

  set(ONNXGENAI_LIBRARIES   "${ORTGA_LIB}"     CACHE STRING "ONNX Runtime GenAI libraries")
  set(ONNXGENAI_INCLUDE_DIR "${ORTGA_INCLUDE}"  CACHE STRING "ONNX Runtime GenAI include path")
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