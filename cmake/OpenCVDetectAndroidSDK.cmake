if(EXISTS "${ANDROID_EXECUTABLE}")
  set(ANDROID_SDK_DETECT_QUIET TRUE)
endif()

file(TO_CMAKE_PATH "$ENV{ProgramFiles}" ProgramFiles_ENV_PATH)
file(TO_CMAKE_PATH "$ENV{HOME}" HOME_ENV_PATH)

if(CMAKE_HOST_WIN32)
  set(ANDROID_SDK_OS windows)
elseif(CMAKE_HOST_APPLE)
  set(ANDROID_SDK_OS mac)
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
    message(STATUS "    Found android tool: ${ANDROID_EXECUTABLE}")
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
  set(ANDROID_PROJECT_FILES ${ANDROID_ANT_PROPERTIES_FILE} ${ANDROID_LIB_PROJECT_FILES})

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

  # detect ANDROID_SDK_TARGET if no target is provided by user
  #TODO: remove this block
  if(NOT ANDROID_SDK_TARGET)
    set(desired_android_target_level ${ANDROID_NATIVE_API_LEVEL})
    if(desired_android_target_level LESS 11)
      set(desired_android_target_level 11)
    endif()
    if(ANDROID_PROCESS EQUAL 0)
      math(EXPR desired_android_target_level_1 "${desired_android_target_level}-1")

      foreach(target ${ANDROID_SDK_TARGETS})
        string(REGEX MATCH "[0-9]+$" target_level "${target}")
        if(target_level GREATER desired_android_target_level_1)
          set(ANDROID_SDK_TARGET "${target}")
          break()
        endif()
      endforeach()
    else()
      set(ANDROID_SDK_TARGET android-${desired_android_target_level})
      message(WARNING "Could not retrieve list of installed Android targets. Will try to use \"${ANDROID_SDK_TARGET}\" target")
    endif()
  endif(NOT ANDROID_SDK_TARGET)

  SET(ANDROID_SDK_TARGET "${ANDROID_SDK_TARGET}" CACHE STRING "SDK target for Android tests and samples")
  string(REGEX MATCH "[0-9]+$" ANDROID_SDK_TARGET_LEVEL "${ANDROID_SDK_TARGET}")
endif(ANDROID_EXECUTABLE)

# finds minimal installed SDK target compatible with provided names or API levels
# usage:
#   get_compatible_android_api_level(VARIABLE [level1] [level2] ...)
macro(android_get_compatible_target VAR)
  set(${VAR} "${VAR}-NOTFOUND")
  if(ANDROID_SDK_TARGETS)
    list(GET ANDROID_SDK_TARGETS 1 __lvl)
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
