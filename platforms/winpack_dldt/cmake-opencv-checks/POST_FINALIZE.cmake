message(STATUS "Winpack-DLDT: Validating OpenCV build configuration...")

if(NOT INF_ENGINE_TARGET)
  message(SEND_ERROR "Inference engine must be detected")
  set(HAS_ERROR 1)
endif()
if(NOT HAVE_NGRAPH)
  message(SEND_ERROR "Inference engine nGraph must be detected")
  set(HAS_ERROR 1)
endif()

if(HAS_ERROR)
  ocv_cmake_dump_vars("^IE_|INF_|INFERENCE|ngraph")
  message(FATAL_ERROR "Winpack-DLDT: Validating OpenCV build configuration... FAILED")
endif()

message(STATUS "Winpack-DLDT: Validating OpenCV build configuration... DONE")
