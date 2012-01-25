if(NOT BUILD_JAVA_SUPPORT OR NOT PYTHON_EXECUTABLE)
    return()
endif()

# ----------------------------------------------------------------------------
#  CMake file for java support
# ----------------------------------------------------------------------------
project(opencv_java)
set(target opencv_java)
include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")


SET(OPENCV_JAVA_MODULES objdetect features2d imgproc video highgui ml core calib3d)
SET(OPENCV_EXTRA_JAVA_MODULES contrib legacy flann)
IF(WITH_ANDROID_CAMERA AND NOT BUILD_SHARED_LIBS)
    LIST(APPEND OPENCV_EXTRA_JAVA_MODULES androidcamera)
ENDIF()


SET(GEN_JAVA "${CMAKE_CURRENT_SOURCE_DIR}/gen_java.py")
SET(HDR_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/../python/src2/hdr_parser.py")
SET(GEN_JAVADOC "${CMAKE_CURRENT_SOURCE_DIR}/gen_javadoc.py")
SET(RST_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/rst_parser.py")
SET(CHECK_TEST_COVERAGE "${CMAKE_CURRENT_SOURCE_DIR}/check-tests.py")

# setup raw java and cpp files generation (without javadoc and at temporary location)
foreach(module ${OPENCV_JAVA_MODULES})
    # get list of module headers
    IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config/${module}.filelist")
        FILE(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/config/${module}.filelist" headers_to_parse)
        SET (module_cheaders "")
        SET (module_cppheaders "")
        foreach(header ${headers_to_parse})
            list(APPEND module_cppheaders "${CMAKE_CURRENT_SOURCE_DIR}/../${module}/${header}")
        endforeach()
    ELSE()
        FILE(GLOB module_cheaders "${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include/opencv2/${module}/*.h")
        FILE(GLOB module_cppheaders "${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include/opencv2/${module}/*.hpp")
        list(SORT module_cheaders)
        list(SORT module_cppheaders)
    ENDIF()

    # add dependencies to cmake (we should rerun cmake if any of these headers are modified)
    foreach(header ${module_cheaders} ${module_cppheaders})
        get_filename_component(header_name "${header}" NAME_WE)
        configure_file("${header}" "${CMAKE_BINARY_DIR}/junk/${header_name}.junk" COPYONLY)
    endforeach()

    # the same about gen_java.py and hdr_parser.py
    configure_file("${GEN_JAVA}" "${CMAKE_BINARY_DIR}/junk/gen_java.junk" COPYONLY)
    configure_file("${HDR_PARSER}" "${CMAKE_BINARY_DIR}/junk/hdr_parser.junk" COPYONLY)


    # first run (to get list of generated files)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out")
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} "${GEN_JAVA}" "${HDR_PARSER}" ${module} ${module_cheaders} ${module_cppheaders}
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out"
                  OUTPUT_QUIET ERROR_QUIET)

    # create list of generated files
    file(GLOB_RECURSE generated_java_sources RELATIVE "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out/" "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out/*.java")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/gen_java_out/")
    
    set(${module}_generated_java_sources)
    foreach(f ${generated_java_sources})
        list(APPEND ${module}_generated_java_sources "${CMAKE_CURRENT_BINARY_DIR}/${f}")
    endforeach()

    # second run
    add_custom_command(
        OUTPUT ${${module}_generated_java_sources}
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp"
        COMMAND ${PYTHON_EXECUTABLE} "${GEN_JAVA}" "${HDR_PARSER}" ${module} ${module_cheaders} ${module_cppheaders}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS "${GEN_JAVA}"
        DEPENDS "${HDR_PARSER}"
        DEPENDS ${module_cheaders}
        DEPENDS ${module_cppheaders}
        )
endforeach()

FILE(GLOB handwrittren_cpp_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")
FILE(GLOB handwrittren_h_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.h*")
FILE(GLOB handwrittren_java_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/java/*.java")
SET (generated_cpp_sources)
SET (generated_java_sources)
SET (documented_java_files)
SET (undocumented_java_files)
SET (dependent_libs)
SET (dependent_extra_libs)

foreach(module ${OPENCV_JAVA_MODULES})
    LIST(APPEND generated_cpp_sources "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp")
    LIST(APPEND generated_java_sources ${${module}_generated_java_sources})
    LIST(APPEND dependent_libs opencv_${module})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include")
endforeach()

# extra includes and dependencies
foreach(module ${OPENCV_EXTRA_JAVA_MODULES})
    LIST(APPEND dependent_extra_libs opencv_${module})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include")
endforeach()

# created list of documented files targets
foreach(java_file ${handwrittren_java_sources} ${generated_java_sources})
    get_filename_component(java_file_name "${java_file}" NAME_WE)
    if (NOT java_file_name MATCHES ".*-jdoc$")
        list(APPEND documented_java_files "${CMAKE_CURRENT_BINARY_DIR}/${java_file_name}-jdoc.java")
        list(APPEND undocumented_java_files "${java_file}")
    endif()
endforeach()

# generate javadoc files
file(GLOB_RECURSE refman_rst_headers "${CMAKE_CURRENT_SOURCE_DIR}/../*.rst")
add_custom_command(
    OUTPUT ${documented_java_files}
    COMMAND ${PYTHON_EXECUTABLE} "${GEN_JAVADOC}" "${CMAKE_CURRENT_SOURCE_DIR}/src/java" "${CMAKE_CURRENT_BINARY_DIR}" 2>"${CMAKE_CURRENT_BINARY_DIR}/get_javadoc_errors.log"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${undocumented_java_files}
    DEPENDS "${GEN_JAVADOC}"
    DEPENDS "${RST_PARSER}"
    DEPENDS ${refman_rst_headers}
)

# copy generated java files to the final location
SET(JAVA_OUTPUT_DIR "src/org/opencv")

set(java_files)
foreach(java_file ${documented_java_files})
    get_filename_component(java_file_name "${java_file}" NAME)
    string(REPLACE "-jdoc.java" ".java" java_file_name "${java_file_name}")
    string(REPLACE "+" "/" java_file_name "${java_file_name}")

    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/${JAVA_OUTPUT_DIR}/${java_file_name}"
        COMMAND ${CMAKE_COMMAND} -E copy "${java_file}" "${CMAKE_BINARY_DIR}/${JAVA_OUTPUT_DIR}/${java_file_name}"
        DEPENDS "${java_file}"
        COMMENT "Generating ${JAVA_OUTPUT_DIR}/${java_file_name}"
        )
    list(APPEND java_files "${CMAKE_BINARY_DIR}/${JAVA_OUTPUT_DIR}/${java_file_name}")
    
    if(ANDROID)
        get_filename_component(install_subdir "${java_file_name}" PATH)
        install(FILES "${CMAKE_BINARY_DIR}/${JAVA_OUTPUT_DIR}/${java_file_name}" DESTINATION ${JAVA_OUTPUT_DIR}/${install_subdir} COMPONENT main)
    endif()
endforeach()

# custom target for java API
set(api_target ${target}_api)
ADD_CUSTOM_TARGET(${api_target} DEPENDS ${java_files})


# add opencv_java library
add_library(${target} SHARED ${handwrittren_h_sources} ${handwrittren_cpp_sources} ${generated_cpp_sources})
target_link_libraries(${target} ${dependent_libs} ${dependent_extra_libs} ${OPENCV_LINKER_LIBS})
add_dependencies(${target} ${dependent_extra_libs} ${dependent_libs} ${api_target})

# Additional target properties
set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${target}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR ${OPENCV_LIB_INSTALL_PATH}
    )

install(TARGETS ${target} LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main)

if(ANDROID)
    target_link_libraries(${target} jnigraphics)

    # force strip library after build command
    # because samples and tests will make a copy of library before install
    ADD_CUSTOM_COMMAND(
        TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_STRIP} "${LIBRARY_OUTPUT_PATH}/lib${target}.so"
        )

    set(lib_proj_files)
    # library project blank
    file(GLOB_RECURSE android_lib_project_files "${CMAKE_CURRENT_SOURCE_DIR}/android/*")
    foreach(file ${android_lib_project_files})
        if(NOT file MATCHES "\\.svn")
            file(RELATIVE_PATH file_rel "${CMAKE_CURRENT_SOURCE_DIR}/android/" "${file}")
            add_custom_command(
                OUTPUT "${CMAKE_BINARY_DIR}/${file_rel}"
                COMMAND ${CMAKE_COMMAND} -E copy "${file}" "${CMAKE_BINARY_DIR}/${file_rel}"
                DEPENDS "${file}"
                COMMENT "Generating ${file_rel}"
                )
            
            list(APPEND lib_proj_files "${CMAKE_BINARY_DIR}/${file_rel}")

            if(NOT file_rel MATCHES "jni/.+")
                install(FILES "${CMAKE_BINARY_DIR}/${file_rel}" DESTINATION . COMPONENT main)
            endif()
        endif()
    endforeach()

    # library project jni sources
    foreach(jni_file ${handwrittren_cpp_sources} ${handwrittren_h_sources} ${generated_cpp_sources})
        get_filename_component(jni_file_name "${jni_file}" NAME)
        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/jni/${jni_file_name}"
            COMMAND ${CMAKE_COMMAND} -E copy "${jni_file}" "${CMAKE_BINARY_DIR}/jni/${jni_file_name}"
            COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/jni/${jni_file_name}"
            DEPENDS "${jni_file}"
            COMMENT "Generating jni/${jni_file_name}"
        )
        list(APPEND lib_proj_files "${CMAKE_BINARY_DIR}/jni/${jni_file_name}")
    endforeach()    
    
    # create Android library project in build folder
    set(lib_target ${target}_android_library)
    ADD_CUSTOM_TARGET(${lib_target}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/res"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/gen"
        DEPENDS ${lib_proj_files}
        )	
    if (ANT_EXECUTABLE AND ANDROID_EXECUTABLE AND ANDROID_TOOLS_Pkg_Revision GREATER 13)
        add_custom_command(
            TARGET ${target}
            COMMAND ${ANDROID_EXECUTABLE} update lib-project --target "${ANDROID_SDK_TARGET}" --path "${CMAKE_BINARY_DIR}"
            COMMAND ${ANT_EXECUTABLE} debug
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            DEPENDS ${lib_target}
            )
    endif()

    add_dependencies(${lib_target} ${api_target})
    add_dependencies(${target} ${lib_target})

endif(ANDROID)

#android test project
if(BUILD_TESTS)
    add_android_project(opencv_test_java "${CMAKE_CURRENT_SOURCE_DIR}/android_test")

    add_custom_command(
        TARGET opencv_test_java_android_project POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} ${CHECK_TEST_COVERAGE} "${CMAKE_CURRENT_SOURCE_DIR}/android_test/src" "${CMAKE_BINARY_DIR}/src" > "${CMAKE_CURRENT_BINARY_DIR}/tests_coverage.log"
        )
endif()
