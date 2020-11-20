ocv_clear_vars(HAVE_ONNX)

set(ONNXRT_ROOT_DIR "" CACHE PATH "ONNX Runtime install directory")

# For now, check the old name ORT_INSTALL_DIR
if(ORT_INSTALL_DIR AND NOT ONNXRT_ROOT_DIR)
  set(ONNXRT_ROOT_DIR ${ORT_INSTALL_DIR})
endif()

if(ONNXRT_ROOT_DIR)
  find_library(ORT_LIB onnxruntime
    ${ONNXRT_ROOT_DIR}/lib
    CMAKE_FIND_ROOT_PATH_BOTH)
  find_path(ORT_INCLUDE onnxruntime_cxx_api.h
    ${ONNXRT_ROOT_DIR}/include/onnxruntime/core/session
    CMAKE_FIND_ROOT_PATH_BOTH)
endif()

if(ORT_LIB AND ORT_INCLUDE)
  set(HAVE_ONNX TRUE)
  # For CMake output only
  set(ONNX_LIBRARIES "${ORT_LIB}" CACHE STRING "ONNX Runtime libraries")
  set(ONNX_INCLUDE_DIR "${ORT_INCLUDE}" CACHE STRING "ONNX Runtime include path")

  # Link target with associated interface headers
  set(ONNX_LIBRARY "onnxruntime" CACHE STRING "ONNX Link Target")
  ocv_add_library(${ONNX_LIBRARY} SHARED IMPORTED)
  set_target_properties(${ONNX_LIBRARY} PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES ${ORT_INCLUDE}
                        IMPORTED_LOCATION ${ORT_LIB}
                        IMPORTED_IMPLIB ${ORT_LIB})
endif()

if(NOT HAVE_ONNX)
  ocv_clear_vars(HAVE_ONNX ORT_LIB ORT_INCLUDE_DIR)
endif()
