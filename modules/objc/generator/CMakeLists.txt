set(MODULE_NAME "objc_bindings_generator")
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
ocv_add_module(${MODULE_NAME} INTERNAL)

set(OPENCV_OBJC_SIGNATURES_FILE "${CMAKE_CURRENT_BINARY_DIR}/opencv_objc_signatures.json" CACHE INTERNAL "")
set(OPENCV_OBJC_BINDINGS_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")

file(REMOVE_RECURSE "${OPENCV_OBJC_BINDINGS_DIR}/gen")
file(REMOVE "${OPENCV_DEPHELPER}/gen_opencv_objc_source")  # force re-run after CMake

# This file is included from a subdirectory
set(OBJC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
include(${OBJC_SOURCE_DIR}/common.cmake)

set(__remap_config "") # list of remapped ".in" files (configure_file)
set(__remap_targets "")

macro(ocv_remap_files files_list_var)
  set(target_dir "${OpenCV_BINARY_DIR}/configured")
  foreach(f ${${files_list_var}})
    if(NOT "${f}" MATCHES "^(.*)\\.in$")
      #continue()  # since CMake 3.2+
    else()
    set(f_ "${CMAKE_MATCH_1}")
    file(RELATIVE_PATH rel_path0 "${OpenCV_SOURCE_DIR}" "${f}")
    file(RELATIVE_PATH rel_path1 "${OpenCV_SOURCE_DIR}" "${f_}")
    set(__target_file "${target_dir}/${rel_path1}")
    configure_file("${f}" "${__target_file}" @ONLY)
    if(__remap_config)
      set(__remap_config "${__remap_config},\n")
    endif()
    set(__remap_config "${__remap_config}    { \"src\": \"${rel_path0}\", \"target\": \"${__target_file}\" }")
    list(APPEND __remap_targets "${__target_file}")
    endif()
  endforeach()
endmacro()

# common files
file(GLOB_RECURSE deps "${CMAKE_CURRENT_SOURCE_DIR}/templates/*")
ocv_remap_files(deps)

set(__modules_config "") # list of OpenCV modules
foreach(m ${OPENCV_OBJC_MODULES})
  set(module_objc_dir "${OPENCV_MODULE_${m}_LOCATION}/misc/objc")
  list(APPEND deps ${OPENCV_MODULE_${m}_HEADERS})
  file(GLOB_RECURSE misc_files "${module_objc_dir}/*")
  list(APPEND deps ${misc_files})

  string(REGEX REPLACE "^opencv_" "" m_ "${m}")
  if(__modules_config)
    set(__modules_config "${__modules_config},\n")
  endif()
  file(RELATIVE_PATH rel_path "${OpenCV_SOURCE_DIR}" "${OPENCV_MODULE_${m}_LOCATION}")
  set(__modules_config "${__modules_config}    { \"name\": \"${m_}\", \"location\": \"${rel_path}\" }")

  ocv_remap_files(misc_files)
endforeach(m)

set(CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/gen_objc.json")
set(__config_str
"{
  \"rootdir\": \"${OpenCV_SOURCE_DIR}\",
  \"modules\": [
${__modules_config}
  ],
  \"files_remap\": [
${__remap_config}
  ]
}
")
if(EXISTS "${CONFIG_FILE}")
  file(READ "${CONFIG_FILE}" __content)
else()
  set(__content "")
endif()
if(NOT "${__content}" STREQUAL "${__config_str}")
  file(WRITE "${CONFIG_FILE}" "${__config_str}")
  file(REMOVE "${OPENCV_DEPHELPER}/gen_opencv_objc_source")
endif()
unset(__config_str)

set(objc_generated_files
    # "${OPENCV_OBJC_SIGNATURES_FILE}"
    "${OPENCV_DEPHELPER}/gen_opencv_objc_source"
)

add_custom_command(
    OUTPUT ${objc_generated_files}
    COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${OBJC_SOURCE_DIR}/generator/gen_objc.py" -p "${OBJC_SOURCE_DIR}/../python/src2/gen2.py" -c "${CONFIG_FILE}"
    COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/gen_opencv_objc_source"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${OBJC_SOURCE_DIR}/generator/gen_objc.py"
            "${OBJC_SOURCE_DIR}/../python/src2/gen2.py"
            "${OBJC_SOURCE_DIR}/../python/src2/hdr_parser.py"
            # don't, result of file(WRITE): "${CMAKE_CURRENT_BINARY_DIR}/gen_objc.json"
            ${deps} ${__remap_targets}
            # not allowed (file(WRITE) result): "${CONFIG_FILE}"
    COMMENT "Generate files for Objective-C bindings"
)

add_custom_target(gen_opencv_objc_source DEPENDS ${objc_generated_files}
    SOURCES "${OBJC_SOURCE_DIR}/generator/gen_objc.py"
            "${CMAKE_CURRENT_BINARY_DIR}/gen_objc.json"
)
