if(EXISTS "${ANDROID_EXECUTABLE}")
  set(ANDROID_SDK_DETECT_QUIET TRUE)
endif()

file(TO_CMAKE_PATH "$ENV{ProgramFiles}" ProgramFiles_ENV_PATH)
file(TO_CMAKE_PATH "$ENV{HOME}" HOME_ENV_PATH)

if(CMAKE_HOST_WIN32)
  set(ANDROID_SDK_OS windows)
elseif(CMAKE_HOST_APPLE)
  set(ANDROID_SDK_OS macosx)
else()
  set(ANDROID_SDK_OS linux)
endif()

#find android SDK: search in ANDROID_SDK first
find_host_program(ANDROID_EXECUTABLE
  NAMES android.bat android
  PATH_SUFFIXES tools
  PATHS
    ENV ANDROID_SDK
  DOC "Android SDK location"
  NO_DEFAULT_PATH
  )

# Now search default paths
find_host_program(ANDROID_EXECUTABLE
  NAMES android.bat android
  PATH_SUFFIXES android-sdk-${ANDROID_SDK_OS}/tools
                android-sdk-${ANDROID_SDK_OS}_x86/tools
                android-sdk-${ANDROID_SDK_OS}_86/tools
                android-sdk/tools
  PATHS /opt
        "${HOME_ENV_PATH}/NVPACK"
        "$ENV{SystemDrive}/NVPACK"
        "${ProgramFiles_ENV_PATH}/Android"
  DOC "Android SDK location"
  )

if(ANDROID_EXECUTABLE)
  if(NOT ANDROID_SDK_DETECT_QUIET)
    message(STATUS "Found android tool: ${ANDROID_EXECUTABLE}")
  endif()

  get_filename_component(ANDROID_SDK_TOOLS_PATH "${ANDROID_EXECUTABLE}" PATH)

  #read source.properties
  if(EXISTS "${ANDROID_SDK_TOOLS_PATH}/source.properties")
    file(STRINGS "${ANDROID_SDK_TOOLS_PATH}/source.properties" ANDROID_SDK_TOOLS_SOURCE_PROPERTIES_LINES REGEX "^[ ]*[^#].*$")
    foreach(line ${ANDROID_SDK_TOOLS_SOURCE_PROPERTIES_LINES})
      string(REPLACE "\\:" ":" line ${line})
      string(REPLACE "=" ";" line ${line})
      list(GET line 0 line_name)
      list(GET line 1 line_value)
      string(REPLACE "." "_" line_name ${line_name})
      SET(ANDROID_TOOLS_${line_name} "${line_value}" CACHE INTERNAL "from ${ANDROID_SDK_TOOLS_PATH}/source.properties")
      MARK_AS_ADVANCED(ANDROID_TOOLS_${line_name})
    endforeach()
  endif()

  #fix missing revision (SDK tools before r9 don't set revision number correctly)
  if(NOT ANDROID_TOOLS_Pkg_Revision)
    SET(ANDROID_TOOLS_Pkg_Revision "Unknown" CACHE INTERNAL "")
    MARK_AS_ADVANCED(ANDROID_TOOLS_Pkg_Revision)
  endif()

  #fix missing description
  if(NOT ANDROID_TOOLS_Pkg_Desc)
    SET(ANDROID_TOOLS_Pkg_Desc "Android SDK Tools, revision ${ANDROID_TOOLS_Pkg_Revision}." CACHE INTERNAL "")
    MARK_AS_ADVANCED(ANDROID_TOOLS_Pkg_Desc)
  endif()

  #warn about outdated SDK
  if(NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
    SET(ANDROID_TOOLS_Pkg_Desc "${ANDROID_TOOLS_Pkg_Desc} It is recommended to update your SDK tools to revision 14 or newer." CACHE INTERNAL "")
  endif()

  if(ANDROID_TOOLS_Pkg_Revision GREATER 13)
    SET(ANDROID_PROJECT_PROPERTIES_FILE project.properties)
    SET(ANDROID_ANT_PROPERTIES_FILE ant.properties)
  else()
    SET(ANDROID_PROJECT_PROPERTIES_FILE default.properties)
    SET(ANDROID_ANT_PROPERTIES_FILE build.properties)
  endif()

  set(ANDROID_MANIFEST_FILE AndroidManifest.xml)
  set(ANDROID_LIB_PROJECT_FILES build.xml local.properties proguard-project.txt ${ANDROID_PROJECT_PROPERTIES_FILE})
  set(ANDROID_PROJECT_FILES ${ANDROID_LIB_PROJECT_FILES})

  #get installed targets
  if(ANDROID_TOOLS_Pkg_Revision GREATER 11)
    execute_process(COMMAND ${ANDROID_EXECUTABLE} list target -c
      RESULT_VARIABLE ANDROID_PROCESS
      OUTPUT_VARIABLE ANDROID_SDK_TARGETS
      ERROR_VARIABLE ANDROID_PROCESS_ERRORS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    string(REGEX MATCHALL "[^\n]+" ANDROID_SDK_TARGETS "${ANDROID_SDK_TARGETS}")
  else()
    #old SDKs (r11 and older) don't provide compact list
    execute_process(COMMAND ${ANDROID_EXECUTABLE} list target
      RESULT_VARIABLE ANDROID_PROCESS
      OUTPUT_VARIABLE ANDROID_SDK_TARGETS_FULL
      ERROR_VARIABLE ANDROID_PROCESS_ERRORS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    string(REGEX MATCHALL "(^|\n)id: [0-9]+ or \"([^\n]+[0-9+])\"(\n|$)" ANDROID_SDK_TARGETS_FULL "${ANDROID_SDK_TARGETS_FULL}")

    SET(ANDROID_SDK_TARGETS "")
    if(ANDROID_PROCESS EQUAL 0)
      foreach(line ${ANDROID_SDK_TARGETS_FULL})
        string(REGEX REPLACE "(^|\n)id: [0-9]+ or \"([^\n]+[0-9+])\"(\n|$)" "\\2" line "${line}")
        list(APPEND ANDROID_SDK_TARGETS "${line}")
      endforeach()
    endif()
  endif()

  if(NOT ANDROID_PROCESS EQUAL 0)
    message(ERROR "Failed to get list of installed Android targets.")
    set(ANDROID_EXECUTABLE "ANDROID_EXECUTABLE-NOTFOUND")
  endif()

  # clear ANDROID_SDK_TARGET if no target is provided by user
  if(NOT ANDROID_SDK_TARGET)
    set(ANDROID_SDK_TARGET "" CACHE STRING "Android SDK target for the OpenCV Java API and samples")
  endif()
  if(ANDROID_SDK_TARGETS)
    set_property( CACHE ANDROID_SDK_TARGET PROPERTY STRINGS ${ANDROID_SDK_TARGETS} )
  endif()
endif(ANDROID_EXECUTABLE)

# finds minimal installed SDK target compatible with provided names or API levels
# usage:
#   get_compatible_android_api_level(VARIABLE [level1] [level2] ...)
macro(android_get_compatible_target VAR)
  set(${VAR} "${VAR}-NOTFOUND")
  if(ANDROID_SDK_TARGETS)
    list(GET ANDROID_SDK_TARGETS 0 __lvl)
    string(REGEX MATCH "[0-9]+$" __lvl "${__lvl}")

    #find minimal level mathing to all provided levels
    foreach(lvl ${ARGN})
      string(REGEX MATCH "[0-9]+$" __level "${lvl}")
      if(__level GREATER __lvl)
        set(__lvl ${__level})
      endif()
    endforeach()

    #search for compatible levels
    foreach(lvl ${ANDROID_SDK_TARGETS})
      string(REGEX MATCH "[0-9]+$" __level "${lvl}")
      if(__level EQUAL __lvl)
        #look for exact match
        foreach(usrlvl ${ARGN})
          if("${usrlvl}" STREQUAL "${lvl}")
            set(${VAR} "${lvl}")
            break()
          endif()
        endforeach()
        if("${${VAR}}" STREQUAL "${lvl}")
          break() #exact match was found
        elseif(NOT ${VAR})
          set(${VAR} "${lvl}")
        endif()
      elseif(__level GREATER __lvl)
        if(NOT ${VAR})
          set(${VAR} "${lvl}")
        endif()
        break()
      endif()
    endforeach()

    unset(__lvl)
    unset(__level)
  endif()
endmacro()

unset(__android_project_chain CACHE)

# add_android_project(target_name ${path} NATIVE_DEPS opencv_core LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET 11)
macro(add_android_project target path)
  # parse arguments
  set(android_proj_arglist NATIVE_DEPS LIBRARY_DEPS SDK_TARGET IGNORE_JAVA IGNORE_MANIFEST)
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
  if(android_proj_IGNORE_JAVA)
    ocv_check_dependencies(${android_proj_NATIVE_DEPS})
  else()
    ocv_check_dependencies(${android_proj_NATIVE_DEPS} opencv_java)
  endif()

  if(EXISTS "${path}/jni/Android.mk" )
    # find if native_app_glue is used
    file(STRINGS "${path}/jni/Android.mk" NATIVE_APP_GLUE REGEX ".*(call import-module,android/native_app_glue)" )
    if(NATIVE_APP_GLUE)
      if(ANDROID_NATIVE_API_LEVEL LESS 9 OR NOT EXISTS "${ANDROID_NDK}/sources/android/native_app_glue")
        set(OCV_DEPENDENCIES_FOUND FALSE)
      endif()
    endif()
  endif()

  if(OCV_DEPENDENCIES_FOUND AND android_proj_sdk_target AND ANDROID_EXECUTABLE AND ANT_EXECUTABLE AND ANDROID_TOOLS_Pkg_Revision GREATER 13 AND EXISTS "${path}/${ANDROID_MANIFEST_FILE}")

    project(${target})
    set(android_proj_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")

    # get project sources
    file(GLOB_RECURSE android_proj_files RELATIVE "${path}" "${path}/res/*" "${path}/src/*")

    if(NOT android_proj_IGNORE_MANIFEST)
      list(APPEND android_proj_files ${ANDROID_MANIFEST_FILE})
    endif()

    # copy sources out from the build tree
    set(android_proj_file_deps "")
    foreach(f ${android_proj_files})
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
    ocv_list_filterout(android_proj_jni_files "\\\\.svn")

    if(android_proj_jni_files AND EXISTS ${path}/jni/Android.mk AND NOT DEFINED JNI_LIB_NAME)
      # find local module name in Android.mk file to build native lib
      file(STRINGS "${path}/jni/Android.mk" JNI_LIB_NAME REGEX "LOCAL_MODULE[ ]*:=[ ]*.*" )
      string(REGEX REPLACE "LOCAL_MODULE[ ]*:=[ ]*([a-zA-Z_][a-zA-Z_0-9]*)[ ]*" "\\1" JNI_LIB_NAME "${JNI_LIB_NAME}")

      if(JNI_LIB_NAME)
        ocv_include_modules_recurse(${android_proj_NATIVE_DEPS})
        ocv_include_directories("${path}/jni")

        if(NATIVE_APP_GLUE)
          include_directories(${ANDROID_NDK}/sources/android/native_app_glue)
          list(APPEND android_proj_jni_files ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
          ocv_warnings_disable(CMAKE_C_FLAGS -Wstrict-prototypes -Wunused-parameter -Wmissing-prototypes)
          set(android_proj_NATIVE_DEPS ${android_proj_NATIVE_DEPS} android)
        endif()

        add_library(${JNI_LIB_NAME} MODULE ${android_proj_jni_files})
        target_link_libraries(${JNI_LIB_NAME} ${OPENCV_LINKER_LIBS} ${android_proj_NATIVE_DEPS})

        set_target_properties(${JNI_LIB_NAME} PROPERTIES
            OUTPUT_NAME "${JNI_LIB_NAME}"
            LIBRARY_OUTPUT_DIRECTORY "${android_proj_bin_dir}/libs/${ANDROID_NDK_ABI_NAME}"
            )

        get_target_property(android_proj_jni_location "${JNI_LIB_NAME}" LOCATION)
        if (NOT (CMAKE_BUILD_TYPE MATCHES "debug"))
            add_custom_command(TARGET ${JNI_LIB_NAME} POST_BUILD COMMAND ${CMAKE_STRIP} --strip-unneeded "${android_proj_jni_location}")
        endif()
      endif()
    endif()

    # build java part
    if(android_proj_IGNORE_JAVA)
      add_custom_command(
         OUTPUT "${android_proj_bin_dir}/bin/${target}-debug.apk"
         COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug
         COMMAND ${CMAKE_COMMAND} -E touch "${android_proj_bin_dir}/bin/${target}-debug.apk" # needed because ant does not update the timestamp of updated apk
         WORKING_DIRECTORY "${android_proj_bin_dir}"
         MAIN_DEPENDENCY "${android_proj_bin_dir}/${ANDROID_MANIFEST_FILE}"
         DEPENDS ${android_proj_file_deps} ${JNI_LIB_NAME})
    else()
      add_custom_command(
         OUTPUT "${android_proj_bin_dir}/bin/${target}-debug.apk"
         COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug
         COMMAND ${CMAKE_COMMAND} -E touch "${android_proj_bin_dir}/bin/${target}-debug.apk" # needed because ant does not update the timestamp of updated apk
         WORKING_DIRECTORY "${android_proj_bin_dir}"
         MAIN_DEPENDENCY "${android_proj_bin_dir}/${ANDROID_MANIFEST_FILE}"
         DEPENDS "${OpenCV_BINARY_DIR}/bin/classes.jar.dephelper" opencv_java # as we are part of OpenCV we can just force this dependency
         DEPENDS ${android_proj_file_deps} ${JNI_LIB_NAME})
    endif()

    unset(JNI_LIB_NAME)

    add_custom_target(${target} ALL SOURCES "${android_proj_bin_dir}/bin/${target}-debug.apk" )
    if(NOT android_proj_IGNORE_JAVA)
      add_dependencies(${target} opencv_java)
    endif()
    if(android_proj_native_deps)
      add_dependencies(${target} ${android_proj_native_deps})
    endif()

    if(__android_project_chain)
      add_dependencies(${target} ${__android_project_chain})
    endif()
    set(__android_project_chain ${target} CACHE INTERNAL "auxiliary variable used for Android progects chaining")

    # put the final .apk to the OpenCV's bin folder
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${android_proj_bin_dir}/bin/${target}-debug.apk" "${OpenCV_BINARY_DIR}/bin/${target}.apk")
    if(INSTALL_ANDROID_EXAMPLES AND "${target}" MATCHES "^example-")
      #apk
      install(FILES "${OpenCV_BINARY_DIR}/bin/${target}.apk" DESTINATION "samples" COMPONENT samples)
      get_filename_component(sample_dir "${path}" NAME)
      #java part
      list(REMOVE_ITEM android_proj_files ${ANDROID_MANIFEST_FILE})
      foreach(f ${android_proj_files} ${ANDROID_MANIFEST_FILE})
        get_filename_component(install_subdir "${f}" PATH)
        install(FILES "${android_proj_bin_dir}/${f}" DESTINATION "samples/${sample_dir}/${install_subdir}" COMPONENT samples)
      endforeach()
      #jni part + eclipse files
      file(GLOB_RECURSE jni_files RELATIVE "${path}" "${path}/jni/*" "${path}/.cproject")
      ocv_list_filterout(jni_files "\\\\.svn")
      foreach(f ${jni_files} ".classpath" ".project" ".settings/org.eclipse.jdt.core.prefs")
        get_filename_component(install_subdir "${f}" PATH)
        install(FILES "${path}/${f}" DESTINATION "samples/${sample_dir}/${install_subdir}" COMPONENT samples)
      endforeach()
      #update proj
      if(android_proj_lib_deps_commands)
        set(inst_lib_opt " --library ../../sdk/java")
      endif()
      install(CODE "EXECUTE_PROCESS(COMMAND ${ANDROID_EXECUTABLE} --silent update project --path . --target \"${android_proj_sdk_target}\" --name \"${target}\" ${inst_lib_opt}
                                    WORKING_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/samples/${sample_dir}\"
                                   )"  COMPONENT samples)
      #empty 'gen'
      install(CODE "MAKE_DIRECTORY(\"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/samples/${sample_dir}/gen\")" COMPONENT samples)
    endif()
  endif()
endmacro()
