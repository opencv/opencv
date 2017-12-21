set(MODULE_NAME "python_bindings_generator")
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
ocv_add_module(${MODULE_NAME} INTERNAL)

set(OPENCV_PYTHON_SIGNATURES_FILE "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_signatures.json" CACHE INTERNAL "")
set(OPENCV_PYTHON_BINDINGS_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")

# This file is included from a subdirectory
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

# get list of modules to wrap
set(OPENCV_PYTHON_MODULES)
foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";${MODULE_NAME};" AND HAVE_${m})
    list(APPEND OPENCV_PYTHON_MODULES ${m})
    #message(STATUS "\t${m}")
  endif()
endforeach()

set(opencv_hdrs "")
set(opencv_userdef_hdrs "")
foreach(m ${OPENCV_PYTHON_MODULES})
  list(APPEND opencv_hdrs ${OPENCV_MODULE_${m}_HEADERS})
  file(GLOB userdef_hdrs ${OPENCV_MODULE_${m}_LOCATION}/misc/python/pyopencv*.hpp)
  list(APPEND opencv_userdef_hdrs ${userdef_hdrs})
endforeach(m)

# header blacklist
ocv_list_filterout(opencv_hdrs "modules/.*\\\\.h$")
ocv_list_filterout(opencv_hdrs "modules/core/.*/cuda")
ocv_list_filterout(opencv_hdrs "modules/cuda.*")
ocv_list_filterout(opencv_hdrs "modules/cudev")
ocv_list_filterout(opencv_hdrs "modules/core/.*/hal/")
ocv_list_filterout(opencv_hdrs "modules/.+/utils/.*")
ocv_list_filterout(opencv_hdrs "modules/.*\\\\.inl\\\\.h*")
ocv_list_filterout(opencv_hdrs "modules/.*_inl\\\\.h*")
ocv_list_filterout(opencv_hdrs "modules/.*\\\\.details\\\\.h*")
ocv_list_filterout(opencv_hdrs "modules/.*/detection_based_tracker\\\\.hpp") # Conditional compilation

set(cv2_generated_hdrs
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_include.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_funcs.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_types.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_type_reg.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_generated_ns_reg.h"
)

set(cv2_generated_files ${cv2_generated_hdrs}
    "${OPENCV_PYTHON_SIGNATURES_FILE}"
)

string(REPLACE ";" "\n" opencv_hdrs_ "${opencv_hdrs}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/headers.txt" "${opencv_hdrs_}")
add_custom_command(
    OUTPUT ${cv2_generated_files}
    COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${PYTHON_SOURCE_DIR}/src2/gen2.py" ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/headers.txt"
    DEPENDS ${PYTHON_SOURCE_DIR}/src2/gen2.py
    DEPENDS ${PYTHON_SOURCE_DIR}/src2/hdr_parser.py
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/headers.txt
    DEPENDS ${opencv_hdrs}
    COMMENT "Generate files for Python bindings and documentation"
)

add_custom_target(gen_opencv_python_source DEPENDS ${cv2_generated_files})

set(cv2_custom_hdr "${CMAKE_CURRENT_BINARY_DIR}/pyopencv_custom_headers.h")
file(WRITE ${cv2_custom_hdr} "//user-defined headers\n")
foreach(uh ${opencv_userdef_hdrs})
    file(APPEND ${cv2_custom_hdr} "#include \"${uh}\"\n")
endforeach(uh)
