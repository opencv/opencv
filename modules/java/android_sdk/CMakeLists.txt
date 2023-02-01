if(NOT ANDROID_EXECUTABLE)
  message(WARNING "'android' tool required to build Android SDK")
  return()
endif()
ocv_assert(ANDROID_TOOLS_Pkg_Revision GREATER 13)

project(${the_module}_android)

set(OPENCV_JAVA_DIR "${OpenCV_BINARY_DIR}/android_sdk" CACHE INTERNAL "")
set(OPENCV_ANDROID_LIB_DIR "${OPENCV_JAVA_DIR}" CACHE INTERNAL "")  # for OpenCV samples

file(REMOVE_RECURSE "${OPENCV_JAVA_DIR}")
file(MAKE_DIRECTORY "${OPENCV_JAVA_DIR}/bin")
set(java_src_dir "${OPENCV_JAVA_DIR}/src")
file(MAKE_DIRECTORY "${java_src_dir}")

ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/java" "${java_src_dir}")
ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android/java" "${java_src_dir}")

# calc default SDK Target
android_get_compatible_target(ANDROID_SDK_COMPATIBLE_TARGET ${ANDROID_NATIVE_API_LEVEL} ${ANDROID_SDK_TARGET} 14)
if(ANDROID_SDK_COMPATIBLE_TARGET)
  set(ANDROID_SDK_COMPATIBLE_TARGET "${ANDROID_SDK_COMPATIBLE_TARGET}" CACHE INTERNAL "")
endif()
string(REGEX REPLACE "android-" "" android_sdk_target_num ${ANDROID_SDK_COMPATIBLE_TARGET})
if( (ANDROID_SDK_TARGET AND ANDROID_SDK_TARGET LESS 21) OR (android_sdk_target_num LESS 21) )
  message(STATUS "[OpenCV for Android SDK]: A new OpenGL Camera Bridge (CameraGLSurfaceView, CameraGLRendererBase, CameraRenderer, Camera2Renderer) is disabled, because ANDROID_SDK_TARGET (${android_sdk_target_num}) < 21")
else()
  ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android-21/java" "${java_src_dir}")
endif()

# copy boilerplate
file(GLOB_RECURSE seed_project_files_rel RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/" "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/*")
list(REMOVE_ITEM seed_project_files_rel "${ANDROID_MANIFEST_FILE}")
foreach(file ${seed_project_files_rel})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${file}" "${OPENCV_JAVA_DIR}/${file}" @ONLY)
  list(APPEND depends "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${file}")
  get_filename_component(install_subdir "${file}" PATH)
  install(FILES "${OPENCV_JAVA_DIR}/${file}" DESTINATION "${JAVA_INSTALL_ROOT}/${install_subdir}" COMPONENT java)
endforeach()

list(APPEND depends gen_opencv_java_source "${OPENCV_DEPHELPER}/gen_opencv_java_source")
ocv_copyfiles_add_target(${the_module}_android_source_copy JAVA_SRC_COPY "Copy Java(Android SDK) source files" ${depends})
file(REMOVE "${OPENCV_DEPHELPER}/${the_module}_android_source_copy")  # force rebuild after CMake run

set(depends ${the_module}_android_source_copy "${OPENCV_DEPHELPER}/${the_module}_android_source_copy")

# generate Android library project
set(android_sdk_project_files ${ANDROID_LIB_PROJECT_FILES})  # build.xml;local.properties;proguard-project.txt;project.properties
ocv_list_add_prefix(android_sdk_project_files "${OPENCV_JAVA_DIR}/")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${ANDROID_MANIFEST_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" @ONLY)

add_custom_command(
    OUTPUT ${android_sdk_project_files} "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}"
    COMMAND ${CMAKE_COMMAND} -E remove ${android_sdk_project_files}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}"
    COMMAND ${ANDROID_EXECUTABLE} --silent create lib-project --path \"${OPENCV_JAVA_DIR}\" --target \"${ANDROID_SDK_COMPATIBLE_TARGET}\" --name OpenCV --package org.opencv 2>\"${CMAKE_CURRENT_BINARY_DIR}/create_lib_project.log\"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}"
    WORKING_DIRECTORY "${OPENCV_JAVA_DIR}"
    MAIN_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
    DEPENDS ${depends}
    COMMENT "Generating OpenCV Android library project. SDK target: ${lib_target_sdk_target}"
)
list(APPEND depends ${android_sdk_project_files} "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}")

install(DIRECTORY "${OPENCV_JAVA_DIR}/src" DESTINATION "${JAVA_INSTALL_ROOT}" COMPONENT java)
install(FILES "${OPENCV_JAVA_DIR}/${ANDROID_PROJECT_PROPERTIES_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT java)
install(FILES "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT java)

# build jar
set(JAR_FILE "${OpenCV_BINARY_DIR}/bin/classes.jar")
# build the library project
# normally we should do this after a native part, but for a library project we can build the java part first
add_custom_command(
    OUTPUT "${JAR_FILE}" "${OPENCV_DEPHELPER}/${the_module}_android"
    COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug -Djava.target=1.6 -Djava.source=1.6
    COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/${the_module}_android"
    WORKING_DIRECTORY "${OPENCV_JAVA_DIR}"
    DEPENDS ${depends}
    COMMENT "Building OpenCV Android library project"
)

add_custom_target(${the_module}_android DEPENDS "${OPENCV_DEPHELPER}/${the_module}_android" SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${ANDROID_MANIFEST_FILE}")
add_dependencies(${the_module} ${the_module}_android)

# empty 'gen' and 'res' folders
install(CODE "
FILE(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${JAVA_INSTALL_ROOT}/gen\")
FILE(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${JAVA_INSTALL_ROOT}/res\")
" COMPONENT java)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.gradle.in" "${CMAKE_CURRENT_BINARY_DIR}/build.gradle" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build.gradle" DESTINATION ${JAVA_INSTALL_ROOT}/.. COMPONENT java)
