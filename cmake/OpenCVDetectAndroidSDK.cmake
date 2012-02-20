file(TO_CMAKE_PATH "$ENV{ProgramFiles}" ProgramFiles_ENV_PATH)
file(TO_CMAKE_PATH "$ENV{ANDROID_SDK}" ANDROID_SDK_ENV_PATH)

#find android SDK
find_host_program(ANDROID_EXECUTABLE
  NAMES android.bat android
  PATHS "${ANDROID_SDK_ENV_PATH}/tools/"
        "${ProgramFiles_ENV_PATH}/Android/android-sdk/tools/"
        "/opt/android-sdk/tools/"
        "/opt/android-sdk-linux_x86/tools/"
        "/opt/android-sdk-linux_86/tools/"
        "/opt/android-sdk-linux/tools/"
        "/opt/android-sdk-mac_x86/tools/"
        "/opt/android-sdk-mac_86/tools/"
        "/opt/android-sdk-mac/tools/"
        "$ENV{HOME}/NVPACK/android-sdk-linux_x86/tools/"
        "$ENV{HOME}/NVPACK/android-sdk-linux_86/tools/"
        "$ENV{HOME}/NVPACK/android-sdk-linux/tools/"
        "$ENV{HOME}/NVPACK/android-sdk-mac_x86/tools/"
        "$ENV{HOME}/NVPACK/android-sdk-mac_86/tools/"
        "$ENV{HOME}/NVPACK/android-sdk-mac/tools/"
        "$ENV{SystemDrive}/NVPACK/android-sdk-windows/tools/"
  )

if(ANDROID_EXECUTABLE)
  message(STATUS "    Found android tool: ${ANDROID_EXECUTABLE}")
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
      SET(ANDROID_TOOLS_${line_name} "${line_value}")
      MARK_AS_ADVANCED(ANDROID_TOOLS_${line_name})
    endforeach()
  endif()

  if(NOT ANDROID_TOOLS_Pkg_Revision)
    SET(ANDROID_TOOLS_Pkg_Revision "Unknown")
    MARK_AS_ADVANCED(ANDROID_TOOLS_Pkg_Revision)
  endif()

  if(NOT ANDROID_TOOLS_Pkg_Desc)
    SET(ANDROID_TOOLS_Pkg_Desc "Android SDK Tools, revision ${ANDROID_TOOLS_Pkg_Revision}.")
    if(NOT ANDROID_TOOLS_Pkg_Revision GREATER 11)
      SET(ANDROID_TOOLS_Pkg_Desc "${ANDROID_TOOLS_Pkg_Desc} It is recommended to update your SDK tools to revision 12 or newer.")
    endif()
    MARK_AS_ADVANCED(ANDROID_TOOLS_Pkg_Desc)
  endif()

  #get installed targets
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

  # detect ANDROID_SDK_TARGET if no target is provided by user
  if(NOT ANDROID_SDK_TARGET)
    set(desired_android_target_level ${ANDROID_NATIVE_API_LEVEL})
    if(desired_android_target_level LESS 8)
      set(desired_android_target_level 8)
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
  if(ANDROID_PROCESS EQUAL 0 AND CMAKE_VERSION VERSION_GREATER "2.8")
    set_property( CACHE ANDROID_SDK_TARGET PROPERTY STRINGS ${ANDROID_SDK_TARGETS} )
  endif()
  string(REGEX MATCH "[0-9]+$" ANDROID_SDK_TARGET_LEVEL "${ANDROID_SDK_TARGET}")
endif(ANDROID_EXECUTABLE)
