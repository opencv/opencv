# Inference Engine CMake config for OpenCV windows package

get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)

set(InferenceEngine_LIBRARIES IE::inference_engine)
add_library(IE::inference_engine SHARED IMPORTED)

set_target_properties(IE::inference_engine PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/deployment_tools/inference_engine/include"
)

# Import target "IE::inference_engine" for configuration "Debug"
set_property(TARGET IE::inference_engine APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(IE::inference_engine PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/deployment_tools/inference_engine/lib/intel64/inference_engined.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG ""
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/inference_engined.dll"
  )

# Import target "IE::inference_engine" for configuration "Release"
set_property(TARGET IE::inference_engine APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(IE::inference_engine PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/deployment_tools/inference_engine/lib/intel64/inference_engine.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE ""
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/inference_engine.dll"
  )

set(InferenceEngine_FOUND ON)
