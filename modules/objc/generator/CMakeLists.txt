set(MODULE_NAME "objc_bindings_generator")
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
ocv_add_module(${MODULE_NAME} INTERNAL opencv_core opencv_imgproc)

#set(OPENCV_OBJC_SIGNATURES_FILE "${CMAKE_CURRENT_BINARY_DIR}/opencv_objc_signatures.json" CACHE INTERNAL "")
set(OPENCV_OBJC_BINDINGS_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")

file(REMOVE_RECURSE "${OPENCV_OBJC_BINDINGS_DIR}/osx")
file(REMOVE "${OPENCV_DEPHELPER}/gen_opencv_objc_source_osx")  # force re-run after CMake
file(REMOVE_RECURSE "${OPENCV_OBJC_BINDINGS_DIR}/ios")
file(REMOVE "${OPENCV_DEPHELPER}/gen_opencv_objc_source_ios")  # force re-run after CMake
file(REMOVE_RECURSE "${OPENCV_OBJC_BINDINGS_DIR}/visionos")
file(REMOVE "${OPENCV_DEPHELPER}/gen_opencv_objc_source_visionos")  # force re-run after CMake

# This file is included from a subdirectory
set(OBJC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
include(${OBJC_SOURCE_DIR}/common.cmake)  # fill OPENCV_OBJC_MODULES

# common files
file(GLOB_RECURSE deps "${CMAKE_CURRENT_SOURCE_DIR}/templates/*")

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
endforeach(m)

if(HAVE_opencv_objc)
  set(__objc_build_dir "\"objc_build_dir\": \"${CMAKE_CURRENT_BINARY_DIR}/../objc\",")
endif()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVBindingsPreprocessorDefinitions.cmake")

ocv_bindings_generator_populate_preprocessor_definitions(
  OPENCV_MODULES_BUILD
  opencv_preprocessor_defs
)

set(CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/gen_objc.json")
set(__config_str
"{
  \"rootdir\": \"${OpenCV_SOURCE_DIR}\",
  ${__objc_build_dir}
  \"modules\": [
${__modules_config}
  ],
  \"preprocessor_definitions\": {
${opencv_preprocessor_defs}
  }
}
")
#TODO: ocv_update_file("${CONFIG_FILE}" "${__config_str}" ON_CHANGE_REMOVE "${OPENCV_DEPHELPER}/gen_opencv_objc_source")
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
)

string(REPLACE "opencv_" "" MODULES "${OPENCV_OBJC_MODULES}")

if(NOT DEFINED OPENCV_OBJC_TARGET AND APPLE_FRAMEWORK)
  if(IOS)
    set(OPENCV_OBJC_TARGET "ios")
  elseif(XROS)
    set(OPENCV_OBJC_TARGET "visionos")
  else()
    set(OPENCV_OBJC_TARGET "osx")
  endif()
endif()

if(NOT DEFINED OPENCV_OBJC_FRAMEWORK_NAME)
  if(DEFINED FRAMEWORK_NAME)
    set(OPENCV_OBJC_FRAMEWORK_NAME "${FRAMEWORK_NAME}")
  else()
    set(OPENCV_OBJC_FRAMEWORK_NAME "opencv2")
  endif()
endif()

set(objc_generated_targets "")

macro(ocv_add_objc_generated_target TARGET)
  set(objc_${TARGET}_generated_output_dependecy "${OPENCV_DEPHELPER}/gen_opencv_objc_source_${TARGET}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}")
  add_custom_command(
      OUTPUT ${objc_generated_files} "${objc_${TARGET}_generated_output_dependecy}"
      COMMAND ${PYTHON3_EXECUTABLE} "${OBJC_SOURCE_DIR}/generator/gen_objc.py"
              -p "${OBJC_SOURCE_DIR}/../python/src2/gen2.py"
              -c "${CONFIG_FILE}"
              -t "${TARGET}"
              -f "${OPENCV_OBJC_FRAMEWORK_NAME}"
      COMMAND ${CMAKE_COMMAND} -E touch "${objc_${TARGET}_generated_output_dependecy}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}"
      DEPENDS "${OpenCV_SOURCE_DIR}/modules/objc/generator/gen_objc.py"
              "${OpenCV_SOURCE_DIR}/modules/python/src2/gen2.py"
              "${OpenCV_SOURCE_DIR}/modules/python/src2/hdr_parser.py"
              # don't, result of file(WRITE): "${CMAKE_CURRENT_BINARY_DIR}/gen_objc.json"
              ${deps}
              # not allowed (file(WRITE) result): "${CONFIG_FILE}"
      COMMENT "Generate files for Objective-C bindings (${TARGET})"
  )
  add_custom_target(gen_opencv_objc_source_${TARGET}
      # excluded from all: ALL
      DEPENDS ${objc_generated_files} ${objc_${TARGET}_generated_output_dependecy}
      SOURCES "${OBJC_SOURCE_DIR}/generator/gen_objc.py"
              "${OBJC_SOURCE_DIR}/generator/templates/cmakelists.template"
              "${CMAKE_CURRENT_BINARY_DIR}/gen_objc.json"
  )
  list(APPEND objc_generated_targets gen_opencv_objc_source_${TARGET})
endmacro()

if(OPENCV_OBJC_TARGET)
  ocv_add_objc_generated_target(${OPENCV_OBJC_TARGET})
else()
  ocv_add_objc_generated_target(osx)
  ocv_add_objc_generated_target(ios)
  ocv_add_objc_generated_target(visionos)
endif()

add_custom_target(gen_opencv_objc_source
    # excluded from all: ALL
    DEPENDS ${objc_generated_targets}
)
