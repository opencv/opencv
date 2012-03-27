#add_android_project(target_name ${path} NATIVE_DEPS opencv_core LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET 11)
macro(add_android_project target path)
  # parse arguments
  set(android_proj_arglist NATIVE_DEPS LIBRARY_DEPS SDK_TARGET)
  set(__varname "android_proj_")
  foreach(v ${android_proj_arglist})
    set(${__varname}${v} "")
  endforeach()
  foreach(arg ${ARGN})
    set(__var "${__varname}")
    foreach(v ${android_proj_arglist})
      if("${v}" STREQUAL "${arg}")
        set(__varname "android_proj_${v}")
        break()
      endif()
    endforeach()
    if(__var STREQUAL __varname)
      list(APPEND ${__var} "${arg}")
    endif()
  endforeach()

  # get compatible SDK target
  android_get_compatible_target(android_proj_sdk_target ${ANDROID_NATIVE_API_LEVEL} ${android_proj_SDK_TARGET})

  if(NOT android_proj_sdk_target)
    message(WARNING "Can not find any SDK target compatible with: ${ANDROID_NATIVE_API_LEVEL} ${android_proj_SDK_TARGET}
                     The project ${target} will not be build")
  endif()

  # check native dependencies
  if(NATIVE_DEPS)
    ocv_check_dependencies(${android_proj_NATIVE_DEPS})
    # warn?
  else()
    set(OCV_DEPENDENCIES_FOUND TRUE)
  endif()

  if(OCV_DEPENDENCIES_FOUND AND android_proj_sdk_target AND ANDROID_EXECUTABLE AND ANT_EXECUTABLE AND ANDROID_TOOLS_Pkg_Revision GREATER 13 AND EXISTS "${path}/${ANDROID_MANIFEST_FILE}")

    project(${target})
    set(android_proj_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")
   
    # get project sources
    file(GLOB_RECURSE android_proj_files RELATIVE "${path}" "${path}/res/*" "${path}/src/*")
    ocv_list_filterout(android_proj_files ".svn")

    # copy sources out from the build tree
    set(android_proj_file_deps "")
    foreach(f ${android_proj_files} ${ANDROID_MANIFEST_FILE})
      add_custom_command(
        OUTPUT "${android_proj_bin_dir}/${f}"
        COMMAND ${CMAKE_COMMAND} -E copy "${path}/${f}" "${android_proj_bin_dir}/${f}"
        MAIN_DEPENDENCY "${path}/${f}"
        COMMENT "Copying ${f}")
      list(APPEND android_proj_file_deps "${path}/${f}" "${android_proj_bin_dir}/${f}")
    endforeach()

    set(android_proj_lib_deps_commands "")
    set(android_proj_target_files ${ANDROID_PROJECT_FILES})
    ocv_list_add_prefix(android_proj_target_files "${android_proj_bin_dir}/")

    # process Android library dependencies
    foreach(dep ${android_proj_LIBRARY_DEPS})
      file(RELATIVE_PATH __dep "${android_proj_bin_dir}" "${dep}")
      list(APPEND android_proj_lib_deps_commands
        COMMAND ${ANDROID_EXECUTABLE} --silent update project --path "${android_proj_bin_dir}" --library "${__dep}")
    endforeach()

    # fix Android project
    add_custom_command(
        OUTPUT ${android_proj_target_files}
        COMMAND ${CMAKE_COMMAND} -E remove ${android_proj_target_files}
        COMMAND ${ANDROID_EXECUTABLE} --silent update project --path "${android_proj_bin_dir}" --target "${android_proj_sdk_target}" --name "${target}"
        ${android_proj_lib_deps_commands}
        MAIN_DEPENDENCY "${android_proj_bin_dir}/${ANDROID_MANIFEST_FILE}"
        DEPENDS "${path}/${ANDROID_MANIFEST_FILE}"
        COMMENT "Updating Android project at ${path}. SDK target: ${android_proj_sdk_target}"
        )

    list(APPEND android_proj_file_deps ${android_proj_target_files})

    # build native part
    file(GLOB_RECURSE android_proj_jni_files "${path}/jni/*.c" "${path}/jni/*.h" "${path}/jni/*.cpp" "${path}/jni/*.hpp")
    ocv_list_filterout(android_proj_jni_files ".svn")

    if(android_proj_jni_files AND EXISTS ${path}/jni/Android.mk)
      file(STRINGS "${path}/jni/Android.mk" JNI_LIB_NAME REGEX "LOCAL_MODULE[ ]*:=[ ]*.*" )
      string(REGEX REPLACE "LOCAL_MODULE[ ]*:=[ ]*([a-zA-Z_][a-zA-Z_0-9]*)[ ]*" "\\1" JNI_LIB_NAME "${JNI_LIB_NAME}")

      if(JNI_LIB_NAME)
        ocv_include_modules_recurse(${android_proj_NATIVE_DEPS})
        ocv_include_directories("${path}/jni")

        add_library(${JNI_LIB_NAME} MODULE ${android_proj_jni_files})
        target_link_libraries(${JNI_LIB_NAME} ${OPENCV_LINKER_LIBS} ${android_proj_NATIVE_DEPS})

        set_target_properties(${JNI_LIB_NAME} PROPERTIES
            OUTPUT_NAME "${JNI_LIB_NAME}"
            LIBRARY_OUTPUT_DIRECTORY "${android_proj_bin_dir}/libs/${ANDROID_NDK_ABI_NAME}"
            )

        get_target_property(android_proj_jni_location "${JNI_LIB_NAME}" LOCATION)
        add_custom_command(TARGET ${JNI_LIB_NAME} POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${android_proj_jni_location}")
      endif()
    else()
      unset(JNI_LIB_NAME)
    endif()

    # build java part
    add_custom_command(
       OUTPUT "${android_proj_bin_dir}/bin/${target}-debug.apk"
       COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug
       COMMAND ${CMAKE_COMMAND} -E touch "${android_proj_bin_dir}/bin/${target}-debug.apk" # needed because ant does not update the timestamp of updated apk
       WORKING_DIRECTORY "${android_proj_bin_dir}"
       MAIN_DEPENDENCY "${android_proj_bin_dir}/${ANDROID_MANIFEST_FILE}"
       DEPENDS "${OpenCV_BINARY_DIR}/bin/classes.jar" opencv_java # as we are part of OpenCV we can just force this dependency
       DEPENDS ${android_proj_file_deps} ${JNI_LIB_NAME})

    add_custom_target(${target} ALL SOURCES "${android_proj_bin_dir}/bin/${target}-debug.apk" )
    add_dependencies(${target} opencv_java ${android_proj_native_deps})

    # put the final .apk to the OpenCV's bin folder
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${android_proj_bin_dir}/bin/${target}-debug.apk" "${OpenCV_BINARY_DIR}/bin/${target}.apk")
    if(INSTALL_ANDROID_EXAMPLES AND target MATCHES "^example-")
      install(FILES "${OpenCV_BINARY_DIR}/bin/${target}.apk" DESTINATION "bin" COMPONENT main)
    endif()
  endif()
endmacro()


