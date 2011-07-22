# ----------------------------------------------------------------------------
#  CMake file for java support
# ----------------------------------------------------------------------------
project(opencv_java)

SET(OPENCV_JAVA_MODULES objdetect features2d imgproc video highgui ml core calib3d)
SET(OPENCV_EXTRA_JAVA_MODULES contrib legacy flann)
IF(WITH_ANDROID_CAMERA AND NOT BUILD_SHARED_LIBS)
    LIST(APPEND OPENCV_EXTRA_JAVA_MODULES androidcamera)
ENDIF()

SET(GEN_JAVA "${CMAKE_CURRENT_SOURCE_DIR}/gen_java.py")
SET(HDR_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/../python/src2/hdr_parser.py")
SET(GEN_JAVADOC "${CMAKE_CURRENT_SOURCE_DIR}/gen_javadoc.py")
SET(RST_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/rst_parser.py")

#SET_SOURCE_FILES_PROPERTIES("${GEN_JAVA}" PROPERTIES GENERATED TRUE)
#SET_SOURCE_FILES_PROPERTIES("${HDR_PARSER}" PROPERTIES GENERATED TRUE)
#SET_SOURCE_FILES_PROPERTIES("${GEN_JAVADOC}" PROPERTIES GENERATED TRUE)
#SET_SOURCE_FILES_PROPERTIES("${RST_PARSER}" PROPERTIES GENERATED TRUE)

#add_custom_command(
#    OUTPUT "${GEN_JAVA}"
#    COMMAND ${CMAKE_COMMAND} -E echo "!!!!!!"
#)

SET(additional_clean_files "")

macro(cleanup)
    file(GLOB files_to_clean "${CMAKE_CURRENT_BINARY_DIR}/*.java" "${CMAKE_CURRENT_BINARY_DIR}/*.cpp" "${CMAKE_CURRENT_BINARY_DIR}/*.h" "${CMAKE_CURRENT_BINARY_DIR}/*.txt" "${CMAKE_CURRENT_BINARY_DIR}/*.log")
    if(files_to_clean)
        file(REMOVE ${files_to_clean})
    endif()
endmacro()

macro(capitalize name outputvar)
    string(SUBSTRING "${name}" 0 1 first_letter)
    string(SUBSTRING "${name}" 1 -1 tail_letters)
    string(TOUPPER "${first_letter}" first_letter)
    SET(${outputvar} "${first_letter}${tail_letters}")
endmacro()

# cleanup required to correct process situations when some class is removed from exports
cleanup()

# setup raw java and cpp files generation (without javadoc and at temporary location)
foreach(module ${OPENCV_JAVA_MODULES})
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

    # first run
    execute_process(COMMAND ${PYTHON_EXECUTABLE} "${GEN_JAVA}" "${HDR_PARSER}" ${module} ${module_cheaders} ${module_cppheaders}
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                  OUTPUT_QUIET ERROR_QUIET)

    # add dependencies to cmake
    foreach(header ${module_cheaders} ${module_cppheaders})
        get_filename_component(header_name "${header}" NAME_WE)
        configure_file("${header}" "${CMAKE_BINARY_DIR}/junk/${header_name}.junk" COPYONLY)
    endforeach()

    FILE(GLOB ${module}_generated_java_sources "${CMAKE_CURRENT_BINARY_DIR}/${module}+*.java")

    # second run
    add_custom_command(
        OUTPUT ${${module}_generated_java_sources}
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp"
        COMMAND ${PYTHON_EXECUTABLE} "${GEN_JAVA}" "${HDR_PARSER}" ${module} ${module_cheaders} ${module_cppheaders}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS "${GEN_JAVA}"
        DEPENDS "${HDR_PARSER}"
        )
endforeach()

cleanup()

set(target opencv_java)
include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")

SET (generated_cpp_sources "")
SET (generated_java_sources "")
SET (dependent_libs "")
SET (dependent_extra_libs "")

# summarize all sources for Java API
foreach(module ${OPENCV_JAVA_MODULES})
    LIST(APPEND generated_cpp_sources "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp")
    LIST(APPEND generated_java_sources ${${module}_generated_java_sources})
    LIST(APPEND dependent_libs opencv_${module})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include")
endforeach()

# set libopencv_java.so shared library includes and dependencies
foreach(module ${OPENCV_EXTRA_JAVA_MODULES})
    LIST(APPEND dependent_extra_libs opencv_${module})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include")
endforeach()

FILE(GLOB handwrittren_cpp_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")
FILE(GLOB handwrittren_h_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.h*")
FILE(GLOB handwrittren_java_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/java/*.java")

# created list of documented files targets
SET(documented_java_files)
SET(undocumented_java_files)
foreach(java_file ${handwrittren_java_sources} ${generated_java_sources})
    get_filename_component(java_file_name "${java_file}" NAME_WE)
    if (NOT java_file_name MATCHES ".*-jdoc$")
        list(APPEND documented_java_files "${CMAKE_CURRENT_BINARY_DIR}/${java_file_name}-jdoc.java")
        list(APPEND undocumented_java_files "${java_file}")
    endif()
endforeach()

# generate javadoc files
add_custom_command(
    OUTPUT ${documented_java_files}
    COMMAND ${PYTHON_EXECUTABLE} "${GEN_JAVADOC}" "${CMAKE_CURRENT_SOURCE_DIR}/src/java" "${CMAKE_CURRENT_BINARY_DIR}" 2>"${CMAKE_CURRENT_BINARY_DIR}/get_javadoc_errors.log"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${handwrittren_java_sources}
    DEPENDS ${undocumented_java_files}
    DEPENDS "${GEN_JAVADOC}"
    DEPENDS "${RST_PARSER}"
)

# copy generated java files to the final location
SET(JAVA_OUTPUT_DIR "${CMAKE_BINARY_DIR}/src/org/opencv")

set(java_files)
foreach(java_file ${documented_java_files})
    get_filename_component(java_file_name "${java_file}" NAME)
    string(REPLACE "-jdoc.java" ".java" java_file_name "${java_file_name}")
    string(REPLACE "+" "/" java_file_name "${java_file_name}")

    add_custom_command(
        OUTPUT "${JAVA_OUTPUT_DIR}/${java_file_name}"
        COMMAND ${CMAKE_COMMAND} -E copy "${java_file}" "${JAVA_OUTPUT_DIR}/${java_file_name}"
        DEPENDS "${java_file}"
        DEPENDS "${JAVA_OUTPUT_DIR}"
        COMMENT "Generating src/org/opencv/${java_file_name}"
        )
    list(APPEND java_files "${JAVA_OUTPUT_DIR}/${java_file_name}")
    if(ANDROID)
        get_filename_component(install_dir "${java_file_name}" PATH)
        install(FILES "${JAVA_OUTPUT_DIR}/${java_file_name}" DESTINATION src/org/opencv/${install_dir} COMPONENT main)
    endif()
endforeach()

# setup dependencies
SET(dependency_file "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cpp")
add_custom_command(
    OUTPUT "${dependency_file}"
    COMMAND ${CMAKE_COMMAND} -E touch "${dependency_file}"
)
SET_SOURCE_FILES_PROPERTIES(${java_files} PROPERTIES HEADER_FILE_ONLY TRUE)
SET(opencv_java_dependency_files "${java_files}")

# add opencv_java project
add_library(${target} SHARED ${handwrittren_h_sources} ${handwrittren_cpp_sources} ${generated_cpp_sources} ${dependency_file})
target_link_libraries(${target} ${dependent_libs} ${dependent_extra_libs} ${OPENCV_LINKER_LIBS})
add_dependencies(${target} "${GEN_JAVA}" "${HDR_PARSER}")
add_dependencies(${target} ${dependent_extra_libs} ${dependent_libs})

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

    # create Android library project in build folder
    add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/res"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/gen"
        )

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
            SET_SOURCE_FILES_PROPERTIES("${CMAKE_BINARY_DIR}/${file_rel}" PROPERTIES HEADER_FILE_ONLY TRUE)
            list(APPEND opencv_java_dependency_files "${CMAKE_BINARY_DIR}/${file_rel}")
        endif()
    endforeach()

    # library project jni sources
    foreach(jni_file ${handwrittren_cpp_sources} ${handwrittren_h_sources} ${generated_cpp_sources})
        get_filename_component(jni_file_name "${jni_file}" NAME)
        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/jni/${jni_file_name}"
            COMMAND ${CMAKE_COMMAND} -E copy "${jni_file}" "${CMAKE_BINARY_DIR}/jni/${jni_file_name}"
            DEPENDS "${jni_file}"
            COMMENT "Generating jni/${jni_file_name}"
        )
        SET_SOURCE_FILES_PROPERTIES("${CMAKE_BINARY_DIR}/jni/${jni_file_name}" PROPERTIES HEADER_FILE_ONLY TRUE)
        list(APPEND opencv_java_dependency_files "${CMAKE_BINARY_DIR}/jni/${jni_file_name}")
    endforeach()
endif()

# apply dependencies
SET_SOURCE_FILES_PROPERTIES("${dependency_file}" PROPERTIES OBJECT_DEPENDS "${opencv_java_dependency_files}")


#set(api_target opencv_java_api)
#add_custom_target(${api_target} ALL DEPENDS ${target})

IF(OFF)
    # create Android library project in build folder
    add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/res"
        COMMENT "Creating res directory"
        )

    FILE(GLOB android_lib_project_files "${CMAKE_CURRENT_SOURCE_DIR}/android/*")
    foreach(lib_file ${android_lib_project_files})
        if(NOT IS_DIRECTORY ${lib_file})
            get_filename_component(lib_file_name "${lib_file}" NAME)
            add_custom_command(
                TARGET ${api_target}
                COMMAND ${CMAKE_COMMAND} -E copy "${lib_file}" "${CMAKE_BINARY_DIR}"
                DEPENDS "${lib_file}"
                COMMENT "Generating ${lib_file_name}"
                )
            install(FILES "${CMAKE_BINARY_DIR}/${lib_file_name}" DESTINATION . COMPONENT main)
            LIST(APPEND additional_clean_files "${CMAKE_BINARY_DIR}/${lib_file_name}")
        endif()
    endforeach()

    foreach(jni_file ${handwrittren_cpp_sources} ${handwrittren_h_sources} ${generated_cpp_sources})
        get_filename_component(jni_file_name "${jni_file}" NAME)
        add_custom_command(
            TARGET ${api_target}
            COMMAND ${CMAKE_COMMAND} -E copy "${jni_file}" "${CMAKE_BINARY_DIR}/jni/${jni_file_name}"
            DEPENDS "${jni_file}"
            COMMENT "Generating jni/${jni_file_name}"
            )
        LIST(APPEND additional_clean_files "${CMAKE_BINARY_DIR}/jni/${jni_file_name}")
    endforeach()
ENDIF()

# generate javadoc comments
#FILE(GLOB java_project_files "${CMAKE_CURRENT_SOURCE_DIR}/src/java/*.java")

#SET(documented_java_files)
#foreach(java_file ${java_project_files} ${generated_java_sources})
#    get_filename_component(java_file_name "${java_file}" NAME_WE)
#    if (NOT java_file_name MATCHES ".*-jdoc$")
#        list(APPEND documented_java_files "${CMAKE_CURRENT_BINARY_DIR}/${java_file_name}-jdoc.java")
#    endif()
#endforeach()

#add_custom_command(
#    TARGET ${api_target} PRE_BUILD
#    COMMAND ${PYTHON_EXECUTABLE} "${GEN_JAVADOC}" "${CMAKE_CURRENT_SOURCE_DIR}/src/java" "${CMAKE_CURRENT_BINARY_DIR}" 2>"${CMAKE_CURRENT_BINARY_DIR}/get_javadoc_errors.log"
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    DEPENDS ${java_project_files}
#    DEPENDS ${generated_java_sources}
#    DEPENDS "${GEN_JAVA}"
#    DEPENDS "${HDR_PARSER}"
#    DEPENDS "${GEN_JAVADOC}"
#    DEPENDS "${RST_PARSER}"
#)


#SET(JAVA_OUTPUT_DIR "${CMAKE_BINARY_DIR}/src/org/opencv")

#foreach(java_file ${documented_java_files})
#    get_filename_component(java_file_name "${java_file}" NAME)
#    string(REPLACE "-jdoc.java" ".java" java_file_name "${java_file_name}")
#    string(REPLACE "+" "/" java_file_name "${java_file_name}")
#
#    add_custom_command(
#        TARGET ${api_target}
#        COMMAND ${CMAKE_COMMAND} -E copy "${java_file}" "${JAVA_OUTPUT_DIR}/${java_file_name}"
#        DEPENDS "${java_file}"
#        DEPENDS "${JAVA_OUTPUT_DIR}"
#        COMMENT "Generating src/org/opencv/${java_file_name}"
#        )
#    LIST(APPEND additional_clean_files "${JAVA_OUTPUT_DIR}/${java_file_name}")
#    if(ANDROID)
#        get_filename_component(install_dir "${java_file_name}" PATH)
#        install(FILES "${JAVA_OUTPUT_DIR}/${java_file_name}" DESTINATION src/org/opencv/${install_dir} COMPONENT main)
#    endif()
#endforeach()

#android test project
if(ANDROID AND BUILD_TESTS AND CAN_BUILD_ANDROID_PROJECTS)
    file(COPY android_test DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
        PATTERN ".svn" EXCLUDE
        PATTERN "gen" EXCLUDE
        PATTERN "bin" EXCLUDE
        )
    SET(test_dir "${CMAKE_CURRENT_BINARY_DIR}/android_test")
    
    SET(test_name opencv_test_java)
    SET(test_target ${test_name}_android)

    add_custom_target(${test_target} ALL)
    add_dependencies(${test_target} opencv_java)

    file(RELATIVE_PATH OPENCV_REFERENCE_PATH "${test_dir}" "${CMAKE_BINARY_DIR}")

    add_custom_command(
        TARGET ${test_target}
        WORKING_DIRECTORY ${test_dir}
        COMMAND ${CMAKE_COMMAND} -E remove -f "${test_dir}/default.properties"
        COMMAND ${CMAKE_COMMAND} -E touch "${test_dir}/default.properties"
        COMMAND ${ANDROID_EXECUTABLE} update project --name "${test_name}" --target "${ANDROID_SDK_TARGET}" --library "${OPENCV_REFERENCE_PATH}" --path .
        COMMAND ${ANT_EXECUTABLE} debug
        COMMAND ${CMAKE_COMMAND} -E copy "${test_dir}/bin/${test_name}-debug.apk" "${CMAKE_BINARY_DIR}/bin/${test_name}.apk"
        )

    LIST(APPEND additional_clean_files "${CMAKE_BINARY_DIR}/bin/${test_name}.apk" "${test_dir}/build.xml" "${test_dir}/local.properties" "${test_dir}/proguard.cfg")
endif()

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${additional_clean_files}")
