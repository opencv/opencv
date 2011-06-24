# ----------------------------------------------------------------------------
#  CMake file for java support
# ----------------------------------------------------------------------------
project(opencv_java)

SET(OPENCV_JAVA_MODULES objdetect features2d imgproc video highgui ml core)
SET(OPENCV_EXTRA_JAVA_MODULES calib3d contrib legacy flann)
IF(ANDROID AND NOT BUILD_SHARED_LIBS)
    LIST(APPEND OPENCV_EXTRA_JAVA_MODULES androidcamera)
ENDIF()

SET(GEN_JAVA "${CMAKE_CURRENT_SOURCE_DIR}/gen_java.py")
SET(HDR_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/../python/src2/hdr_parser.py")

foreach(module ${OPENCV_JAVA_MODULES})
    IF(${module} STREQUAL core)
        SET (module_cppheaders "${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include/opencv2/${module}/${module}.hpp")
        SET (module_cheaders "")
    ELSE()
        FILE(GLOB module_cheaders "${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include/opencv2/${module}/*.h")
        FILE(GLOB module_cppheaders "${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include/opencv2/${module}/*.hpp")
    ENDIF()
    list(SORT module_cheaders)
    list(SORT module_cppheaders)

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${module}.java"
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp"
        COMMAND "${PYTHON_EXECUTABLE}" -B "${GEN_JAVA}" "${HDR_PARSER}" ${module} ${module_cheaders} ${module_cppheaders}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS "${GEN_JAVA}"
        DEPENDS "${HDR_PARSER}"
        DEPENDS ${module_headers}
        )
endforeach()

set(target "opencv_java")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

FILE(GLOB handwrittren_cpp_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")

SET (generated_cpp_sources "")
SET (generated_java_sources "")
SET (dependent_libs "")
SET (dependent_extra_libs "")

foreach(module ${OPENCV_JAVA_MODULES})
    LIST(APPEND generated_cpp_sources "${CMAKE_CURRENT_BINARY_DIR}/${module}.cpp")
    LIST(APPEND generated_java_sources "${CMAKE_CURRENT_BINARY_DIR}/${module}.java")
    LIST(APPEND dependent_libs opencv_${module})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include")
endforeach()

foreach(module ${OPENCV_EXTRA_JAVA_MODULES})
    LIST(APPEND dependent_extra_libs opencv_${module})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include")
endforeach()

#SET (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/home/andreyk/OpenCV2/trunk/opencv/android/build/libs/armeabi-v7a")

add_library(${target} SHARED ${handwrittren_cpp_sources} ${generated_cpp_sources})

target_link_libraries(${target} ${dependent_libs} ${dependent_extra_libs} ${OPENCV_LINKER_LIBS})
if(ANDROID)
    target_link_libraries(${target} jnigraphics)
endif()

#add_dependencies(${the_target} ${dependent_extra_libs} ${dependent_libs})

# Additional target properties
set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${target}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR lib
    )

set(api_target opencv_java_api)
add_custom_target(${api_target} ALL DEPENDS ${target})

IF(ANDROID)
    # create Android library project in build folder
    add_custom_command(
        TARGET ${api_target}
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
        endif()
    endforeach()
ENDIF()

FILE(GLOB java_project_files "${CMAKE_CURRENT_SOURCE_DIR}/src/java/*.java")
SET(JAVA_OUTPUT_DIR "${CMAKE_BINARY_DIR}/src/org/opencv")

foreach(java_file ${java_project_files} ${generated_java_sources})
    get_filename_component(java_file_name "${java_file}" NAME)
    add_custom_command(
        TARGET ${api_target}
        COMMAND ${CMAKE_COMMAND} -E copy "${java_file}" "${JAVA_OUTPUT_DIR}/${java_file_name}"
        DEPENDS "${java_file}"
        DEPENDS "${JAVA_OUTPUT_DIR}"
        COMMENT "Generating src/org/opencv/${java_file_name}"
        )
endforeach()

