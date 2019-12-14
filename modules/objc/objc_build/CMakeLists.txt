project(${the_module})

glob_more_specific_sources(H "${CMAKE_CURRENT_SOURCE_DIR}/../generator/src" handwritten_h_sources)
glob_more_specific_sources(CPP "${CMAKE_CURRENT_SOURCE_DIR}/../generator/src" handwritten_cpp_sources)

# grab C++ files from misc/objc
foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";objc;" AND HAVE_${m})
    set(module_objc_dir "${OPENCV_MODULE_${m}_LOCATION}/misc/objc")
    include_directories("${module_objc_dir}/src/cpp")
    file(GLOB _result "${module_objc_dir}/src/cpp/*.h" "${module_objc_dir}/src/cpp/*.hpp" "${module_objc_dir}/src/cpp/*.cpp")
    list(APPEND handwritten_cpp_sources ${_result})
  endif()
endforeach()

set(__type MODULE)
ocv_add_library(${the_module} ${__type}
    ${handwritten_h_sources} ${handwritten_cpp_sources} ${generated_cpp_sources}
    ${copied_files}
)
add_dependencies(${the_module} gen_opencv_objc_source)

ocv_target_include_directories(${the_module} "${CMAKE_CURRENT_SOURCE_DIR}/../generator/src/cpp")
ocv_target_include_directories(${the_module} "${OPENCV_OBJC_BINDINGS_DIR}/gen/cpp")
ocv_target_include_modules(${the_module} ${OPENCV_MODULE_${the_module}_DEPS})
if(NOT ANDROID)
  ocv_target_include_directories(${the_module} SYSTEM ${JNI_INCLUDE_DIRS})
endif()

set(__deps ${OPENCV_MODULE_${the_module}_DEPS})
list(REMOVE_ITEM __deps opencv_objc_bindings_generator) # don't add dummy module to target_link_libraries list

ocv_target_link_libraries(${the_module} LINK_PRIVATE ${__deps} ${OPENCV_LINKER_LIBS})

# Additional target properties
set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${OPENCV_OBJC_LIB_NAME_SUFFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    DEFINE_SYMBOL CVAPI_EXPORTS
    )

