project(${the_module})

set(OPENCV_MODULE_IS_PART_OF_WORLD TRUE)

# grab files from misc/objc
foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";objc;" AND HAVE_${m})
    set(module_objc_dir "${OPENCV_MODULE_${m}_LOCATION}/misc/objc")
    file(GLOB _result "${module_objc_dir}/src/*\.h" "${module_objc_dir}/src/*\.m" "${module_objc_dir}/src/*\.mm" "${module_objc_dir}/src/*\.swift")
    list(APPEND handwritten_objc_sources ${_result})
  endif()
endforeach()

set(__type ${OPENCV_MODULE_TYPE})
ocv_add_library(${the_module} ${__type}
    ${handwritten_objc_sources}
    ${copied_files}
)
add_dependencies(${the_module} gen_opencv_objc_source)

ocv_target_include_modules(${the_module} ${OPENCV_MODULE_${the_module}_DEPS})
ocv_target_include_directories(${the_module} PRIVATE "${OpenCV_SOURCE_DIR}/include")

set(__deps ${OPENCV_MODULE_${the_module}_DEPS})
list(REMOVE_ITEM __deps opencv_objc_bindings_generator) # don't add dummy module to target_link_libraries list

ocv_target_link_libraries(${the_module} LINK_PRIVATE ${__deps} ${OPENCV_LINKER_LIBS})

enable_language(Swift)

# Additional target properties
set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${OPENCV_OBJC_LIB_NAME_SUFFIX}"
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${OPENCV_MODULE_opencv_core_LOCATION}/misc/objc/src/OpenCV-Bridging-Header.h"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    DEFINE_SYMBOL CVAPI_EXPORTS
    )

if(NOT XCODE_VERSION VERSION_LESS 10.2)
  set_target_properties(${the_module} PROPERTIES
      XCODE_ATTRIBUTE_SWIFT_VERSION 5.0
      )
elseif(NOT XCODE_VERSION VERSION_LESS 9.0)
  set_target_properties(${the_module} PROPERTIES
      XCODE_ATTRIBUTE_SWIFT_VERSION 4.0
      )
elseif(NOT XCODE_VERSION VERSION_LESS 8.0)
  set_target_properties(${the_module} PROPERTIES
      XCODE_ATTRIBUTE_SWIFT_VERSION 3.0
      )
endif()
