if(EXISTS "${ANDROID_EXECUTABLE}")
  set(ANDROID_SDK_DETECT_QUIET TRUE)
endif()

# fixup for https://github.com/android-ndk/ndk/issues/596
if(DEFINED ANDROID_NDK_REVISION AND ANDROID_NDK_REVISION MATCHES "(1[56])([0-9]+)\\.([^\n]+)\n")
  set(ANDROID_NDK_REVISION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  set(ANDROID_NDK_REVISION "${ANDROID_NDK_REVISION}" CACHE INTERNAL "Android NDK revision")
endif()

# fixup -g option: https://github.com/opencv/opencv/issues/8460#issuecomment-434249750
if(INSTALL_CREATE_DISTRIB
  AND (NOT BUILD_WITH_DEBUG_INFO AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
  AND NOT OPENCV_SKIP_ANDROID_G_OPTION_FIX
)
  if(" ${CMAKE_CXX_FLAGS} " MATCHES " -g ")
    message(STATUS "Android: fixup -g compiler option from Android toolchain")
  endif()
  string(REPLACE " -g " " " CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ")
  string(REPLACE " -g " " " CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ")
  string(REPLACE " -g " " " CMAKE_ASM_FLAGS " ${CMAKE_ASM_FLAGS} ")
  if(NOT " ${CMAKE_CXX_FLAGS_DEBUG}" MATCHES " -g")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
  endif()
  if(NOT " ${CMAKE_C_FLAGS_DEBUG}" MATCHES " -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
  endif()
endif()

# https://developer.android.com/studio/command-line/variables.html
ocv_check_environment_variables(ANDROID_SDK_ROOT ANDROID_HOME ANDROID_SDK)

set(__msg_BUILD_ANDROID_PROJECTS "Use BUILD_ANDROID_PROJECTS=OFF to prepare Android project files without building them")

macro(ocv_detect_android_sdk)
  if(NOT DEFINED ANDROID_SDK)
    if(DEFINED ANDROID_SDK AND EXISTS "${ANDROID_SDK}")
      set(ANDROID_SDK "${ANDROID_SDK}" CACHE INTERNAL "Android SDK path")
      elseif(DEFINED ANDROID_HOME AND EXISTS "${ANDROID_HOME}")
      set(ANDROID_SDK "${ANDROID_HOME}" CACHE INTERNAL "Android SDK path")
    elseif(DEFINED ANDROID_SDK_ROOT AND EXISTS "${ANDROID_SDK_ROOT}")
      set(ANDROID_SDK "${ANDROID_SDK_ROOT}" CACHE INTERNAL "Android SDK path")
    endif()
    if(DEFINED ANDROID_SDK)
      message(STATUS "Android SDK: using location: ${ANDROID_SDK}")
    endif()
  endif()
  if(NOT DEFINED ANDROID_SDK)
    message(FATAL_ERROR "Android SDK: specify path to Android SDK via ANDROID_SDK_ROOT / ANDROID_HOME / ANDROID_SDK variables")
  endif()
  if(NOT EXISTS "${ANDROID_SDK}")
    message(FATAL_ERROR "Android SDK: specified path doesn't exist: ${ANDROID_SDK}")
  endif()
endmacro()

macro(ocv_detect_android_sdk_tools)
  # https://developer.android.com/studio/releases/sdk-tools.html
  if(NOT DEFINED ANDROID_SDK_TOOLS)
    if(DEFINED ANDROID_SDK AND EXISTS "${ANDROID_SDK}/tools")
      set(ANDROID_SDK_TOOLS "${ANDROID_SDK}/tools" CACHE INTERNAL "Android SDK Tools path")
    endif()
  endif()
  if(NOT DEFINED ANDROID_SDK_TOOLS)
    message(FATAL_ERROR "Android SDK Tools: can't automatically find Android SDK Tools. Specify path via ANDROID_SDK_TOOLS variable")
  endif()
  if(NOT EXISTS "${ANDROID_SDK_TOOLS}")
    message(FATAL_ERROR "Android SDK Tools: specified path doesn't exist: ${ANDROID_SDK_TOOLS}")
  endif()

  if(NOT DEFINED ANDROID_SDK_TOOLS_VERSION)
    ocv_parse_properties_file("${ANDROID_SDK_TOOLS}/source.properties"
        ANDROID_TOOLS CACHE Pkg_Revision
        MSG_PREFIX "Android SDK Tools: "
    )

    if(NOT DEFINED ANDROID_TOOLS_Pkg_Revision)
      message(FATAL_ERROR "Android SDK Tools: Can't determine package version: ANDROID_SDK_TOOLS=${ANDROID_SDK_TOOLS}\n"
                          "Check specified parameters or force version via 'ANDROID_SDK_TOOLS_VERSION' variable.\n"
                          "${__msg_BUILD_ANDROID_PROJECTS}")
    elseif(NOT ANDROID_SDK_DETECT_QUIET)
      set(__info "")
      if(DEFINED ANDROID_TOOLS_Pkg_Desc)
        set(__info " (description: '${ANDROID_TOOLS_Pkg_Desc}')")
      endif()
      message(STATUS "Android SDK Tools: ver. ${ANDROID_TOOLS_Pkg_Revision}${__info}")
    endif()
    set(ANDROID_SDK_TOOLS_VERSION "${ANDROID_TOOLS_Pkg_Revision}" CACHE INTERNAL "Android SDK Tools version")
  endif()
  if(NOT DEFINED ANDROID_TOOLS_Pkg_Revision)
    set(ANDROID_TOOLS_Pkg_Revision "${ANDROID_SDK_TOOLS_VERSION}" CACHE INTERNAL "Android SDK Tools version (deprecated)")
  endif()
  set(ANDROID_SDK_TOOLS_PATH "${ANDROID_SDK_TOOLS}" CACHE INTERNAL "Android SDK Tools path (deprecated)")
endmacro()  # ocv_detect_android_sdk_tools

macro(ocv_detect_android_sdk_build_tools)
  # https://developer.android.com/studio/releases/build-tools.html
  if(NOT DEFINED ANDROID_SDK_BUILD_TOOLS_VERSION)
    if(NOT DEFINED ANDROID_SDK_BUILD_TOOLS)
      set(__search_dir ${ANDROID_SDK}/build-tools)
      if(NOT EXISTS "${__search_dir}")
        message(FATAL_ERROR "Android SDK Build Tools: directory doesn't exist: ${__search_dir} "
                            "${__msg_BUILD_ANDROID_PROJECTS}")
      endif()

      if(NOT DEFINED ANDROID_SDK_BUILD_TOOLS_SUBDIR)
        file(GLOB __found RELATIVE "${__search_dir}" ${__search_dir}/*)
        set(__dirlist "")
        set(__selected 0)
        set(__versions "")
        foreach(d ${__found})
          if(IS_DIRECTORY "${__search_dir}/${d}")
            list(APPEND __dirlist ${d})
            if(d MATCHES "[0-9]+(\\.[0-9]+)*")
              list(APPEND __versions "${d}")
            endif()
            if(__selected VERSION_LESS d)
              set(__selected "${d}")
            endif()
          endif()
        endforeach()
        if(__selected VERSION_GREATER 0)
          set(ANDROID_SDK_BUILD_TOOLS_SUBDIR "${__selected}")
        elseif(__dirlist)
          set(__versions "")
          foreach(d ${__dirlist})
            if(EXISTS "${__search_dir}/${d}/source.properties")
              ocv_clear_vars(ANDROID_BUILD_TOOLS_Pkg_Revision)
              ocv_parse_properties_file("${__search_dir}/${d}/source.properties"
                  ANDROID_BUILD_TOOLS
                  MSG_PREFIX "Android SDK Tools: "
              )
              if(DEFINED ANDROID_BUILD_TOOLS_Pkg_Revision)
                list(APPEND __versions "${ANDROID_BUILD_TOOLS_Pkg_Revision}")
                if(__selected VERSION_LESS ANDROID_BUILD_TOOLS_Pkg_Revision)
                  set(ANDROID_SDK_BUILD_TOOLS_SUBDIR "${d}")
                  set(__selected "${ANDROID_BUILD_TOOLS_Pkg_Revision}")
                endif()
              endif()
            endif()
          endforeach()
        endif()
        if(DEFINED ANDROID_SDK_BUILD_TOOLS_SUBDIR)
          set(ANDROID_SDK_BUILD_TOOLS_VERSION "${__selected}" CACHE STRING "Android SDK Build Tools version")
          set_property(CACHE ANDROID_SDK_BUILD_TOOLS_VERSION PROPERTY STRINGS ${__versions})
          set(ANDROID_SDK_BUILD_TOOLS "${__search_dir}/${d}" CACHE INTERNAL "Android SDK Build Tools path")
          message(STATUS "Android SDK Build Tools: ver. ${ANDROID_SDK_BUILD_TOOLS_VERSION} (subdir ${ANDROID_SDK_BUILD_TOOLS_SUBDIR} from ${__dirlist})")
        else()
          message(FATAL_ERROR "Android SDK Build Tools: autodetection failed. "
                              "Specify ANDROID_SDK_BUILD_TOOLS_VERSION / ANDROID_SDK_BUILD_TOOLS_SUBDIR / ANDROID_SDK_BUILD_TOOLS variable to bypass autodetection.\n"
                              "${__msg_BUILD_ANDROID_PROJECTS}")
        endif()
      endif()
    else()
      ocv_parse_properties_file("${ANDROID_SDK_BUILD_TOOLS}/source.properties"
          ANDROID_BUILD_TOOLS
          MSG_PREFIX "Android SDK Tools: "
      )
      if(NOT DEFINED ANDROID_BUILD_TOOLS_Pkg_Revision)
        message(FATAL_ERROR "Android SDK Build Tools: Can't detect version: ANDROID_SDK_BUILD_TOOLS=${ANDROID_SDK_BUILD_TOOLS}\n"
                            "Specify ANDROID_SDK_BUILD_TOOLS_VERSION variable to bypass autodetection.\n"
                            "${__msg_BUILD_ANDROID_PROJECTS}")
      else()
        set(ANDROID_SDK_BUILD_TOOLS_VERSION "${ANDROID_BUILD_TOOLS_Pkg_Revision}" CACHE INTERNAL "Android SDK Build Tools version")
        message(STATUS "Android SDK Build Tools: ver. ${ANDROID_SDK_BUILD_TOOLS_VERSION} (ANDROID_SDK_BUILD_TOOLS=${ANDROID_SDK_BUILD_TOOLS})")
      endif()
    endif()  # ANDROID_SDK_BUILD_TOOLS
  endif()  # ANDROID_SDK_BUILD_TOOLS_VERSION
endmacro()  # ocv_detect_android_sdk_build_tools


if(BUILD_ANDROID_PROJECTS)
  ocv_detect_android_sdk()
  ocv_detect_android_sdk_tools()
  ocv_detect_android_sdk_build_tools()

  if(ANDROID_SDK_TOOLS_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "Android SDK Tools: OpenCV requires Android SDK Tools revision 14 or newer.\n"
                        "${__msg_BUILD_ANDROID_PROJECTS}")
  endif()

  if(NOT ANDROID_SDK_TOOLS_VERSION VERSION_LESS 25.3.0)
    message(STATUS "Android SDK Tools: Ant (Eclipse) builds are NOT supported by Android SDK")
    ocv_update(ANDROID_PROJECTS_SUPPORT_ANT OFF)
    if(NOT ANDROID_SDK_BUILD_TOOLS_VERSION VERSION_LESS 26.0.2)
      # https://developer.android.com/studio/releases/gradle-plugin.html
      message(STATUS "Android SDK Build Tools: Gradle 3.0.0+ builds support is available")
      ocv_update(ANDROID_PROJECTS_SUPPORT_GRADLE ON)
    endif()
  else()
    include(${CMAKE_CURRENT_LIST_DIR}/../OpenCVDetectApacheAnt.cmake)
    if(ANT_EXECUTABLE AND NOT ANT_VERSION VERSION_LESS 1.7)
      message(STATUS "Android SDK Tools: Ant (Eclipse) builds are supported")
      ocv_update(ANDROID_PROJECTS_SUPPORT_ANT ON)
    endif()
  endif()

  if(NOT DEFINED ANDROID_PROJECTS_BUILD_TYPE)
    if(ANDROID_PROJECTS_SUPPORT_ANT)
      ocv_update(ANDROID_PROJECTS_BUILD_TYPE "ANT")
    elseif(ANDROID_PROJECTS_SUPPORT_GRADLE)
      ocv_update(ANDROID_PROJECTS_BUILD_TYPE "GRADLE")
    else()
      message(FATAL_ERROR "Android SDK: Can't build Android projects as requested by BUILD_ANDROID_PROJECTS=ON variable.\n"
                          "${__msg_BUILD_ANDROID_PROJECTS}")
    endif()
  endif()

  if(ANDROID_PROJECTS_BUILD_TYPE STREQUAL "ANT")
    message(STATUS "Android SDK Tools: Prepare Android projects for using Ant build scripts (deprecated)")
  elseif(ANDROID_PROJECTS_BUILD_TYPE STREQUAL "GRADLE")
    message(STATUS "Android SDK Tools: Prepare Android projects for using Gradle 3.0.0+ build scripts")
  endif()

else()
  message("Android: Projects builds are DISABLED")
  macro(add_android_project)
  endmacro()
endif()  # BUILD_ANDROID_PROJECTS

if(ANDROID_PROJECTS_BUILD_TYPE STREQUAL "ANT")
  include(${CMAKE_CURRENT_LIST_DIR}/android_ant_projects.cmake)
elseif(ANDROID_PROJECTS_BUILD_TYPE STREQUAL "GRADLE")
  include(${CMAKE_CURRENT_LIST_DIR}/android_gradle_projects.cmake)
elseif(BUILD_ANDROID_PROJECTS)
  message(FATAL_ERROR "Internal error")
else()
  # TODO
  #include(${CMAKE_CURRENT_LIST_DIR}/android_disabled_projects.cmake)
endif()
