# https://developer.android.com/studio/releases/gradle-plugin
set(ANDROID_GRADLE_PLUGIN_VERSION "3.2.1" CACHE STRING "Android Gradle Plugin version (3.0+)")
message(STATUS "Android Gradle Plugin version: ${ANDROID_GRADLE_PLUGIN_VERSION}")

set(ANDROID_COMPILE_SDK_VERSION "26" CACHE STRING "Android compileSdkVersion")
set(ANDROID_MIN_SDK_VERSION "21" CACHE STRING "Android minSdkVersion")
set(ANDROID_TARGET_SDK_VERSION "26" CACHE STRING "Android minSdkVersion")

set(ANDROID_BUILD_BASE_DIR "${OpenCV_BINARY_DIR}/opencv_android" CACHE INTERNAL "")
set(ANDROID_TMP_INSTALL_BASE_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/opencv_android")

set(ANDROID_INSTALL_SAMPLES_DIR "samples")

set(ANDROID_BUILD_ABI_FILTER "
reset()
include '${ANDROID_ABI}'
")

set(ANDROID_INSTALL_ABI_FILTER "
//reset()
//include 'armeabi-v7a'
//include 'arm64-v8a'
//include 'x86'
//include 'x86_64'
")
if(NOT INSTALL_CREATE_DISTRIB)
  set(ANDROID_INSTALL_ABI_FILTER "${ANDROID_BUILD_ABI_FILTER}")
endif()

# BUG: Ninja generator generates broken targets with ANDROID_ABI_FILTER name (CMake 3.11.2)
#set(__spaces "                        ")
#string(REPLACE "\n" "\n${__spaces}" ANDROID_ABI_FILTER "${__spaces}${ANDROID_BUILD_ABI_FILTER}")
#string(REPLACE REGEX "[ ]+$" "" ANDROID_ABI_FILTER "${ANDROID_ABI_FILTER}")
set(ANDROID_ABI_FILTER "${ANDROID_BUILD_ABI_FILTER}")
configure_file("${OpenCV_SOURCE_DIR}/samples/android/build.gradle.in" "${ANDROID_BUILD_BASE_DIR}/build.gradle" @ONLY)

set(ANDROID_ABI_FILTER "${ANDROID_INSTALL_ABI_FILTER}")
configure_file("${OpenCV_SOURCE_DIR}/samples/android/build.gradle.in" "${ANDROID_TMP_INSTALL_BASE_DIR}/${ANDROID_INSTALL_SAMPLES_DIR}/build.gradle" @ONLY)
install(FILES "${ANDROID_TMP_INSTALL_BASE_DIR}/${ANDROID_INSTALL_SAMPLES_DIR}/build.gradle" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}" COMPONENT samples)

set(GRADLE_WRAPPER_FILES
    "gradle/wrapper/gradle-wrapper.jar"
    "gradle/wrapper/gradle-wrapper.properties"
    "gradlew.bat"
    "gradlew"
    "gradle.properties"
)
foreach(fname ${GRADLE_WRAPPER_FILES})
  get_filename_component(__dir "${fname}" DIRECTORY)
  set(__permissions "")
  set(__permissions_prefix "")
  if(fname STREQUAL "gradlew")
    set(__permissions FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif()
  file(COPY "${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/${fname}" DESTINATION "${ANDROID_BUILD_BASE_DIR}/${__dir}" ${__permissions})
  string(REPLACE "FILE_PERMISSIONS" "PERMISSIONS" __permissions "${__permissions}")
  if("${__dir}" STREQUAL "")
    set(__dir ".")
  endif()
  install(FILES "${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/${fname}" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}/${__dir}" COMPONENT samples ${__permissions})
endforeach()

file(WRITE "${ANDROID_BUILD_BASE_DIR}/settings.gradle" "
include ':opencv'
")

file(WRITE "${ANDROID_TMP_INSTALL_BASE_DIR}/settings.gradle" "
rootProject.name = 'opencv_samples'

def opencvsdk='../'
//def opencvsdk='/<path to OpenCV-android-sdk>'
//println opencvsdk
include ':opencv'
project(':opencv').projectDir = new File(opencvsdk + '/sdk')
")


macro(add_android_project target path)
  get_filename_component(__dir "${path}" NAME)

  set(OPENCV_ANDROID_CMAKE_EXTRA_ARGS "")
  if(DEFINED ANDROID_TOOLCHAIN)
    set(OPENCV_ANDROID_CMAKE_EXTRA_ARGS "${OPENCV_ANDROID_CMAKE_EXTRA_ARGS},\n\"-DANDROID_TOOLCHAIN=${ANDROID_TOOLCHAIN}\"")
  endif()
  if(DEFINED ANDROID_STL)
    set(OPENCV_ANDROID_CMAKE_EXTRA_ARGS "${OPENCV_ANDROID_CMAKE_EXTRA_ARGS},\n\"-DANDROID_STL=${ANDROID_STL}\"")
  endif()

  #
  # Build
  #
  set(ANDROID_SAMPLE_JNI_PATH "${path}/jni")
  set(ANDROID_SAMPLE_JAVA_PATH "${path}/src")
  set(ANDROID_SAMPLE_RES_PATH "${path}/res")
  set(ANDROID_SAMPLE_MANIFEST_PATH "${path}/gradle/AndroidManifest.xml")

  set(ANDROID_ABI_FILTER "${ANDROID_BUILD_ABI_FILTER}")
  set(ANDROID_PROJECT_JNI_PATH "../../")

  string(REPLACE ";" "', '" ANDROID_SAMPLE_JAVA_PATH "['${ANDROID_SAMPLE_JAVA_PATH}']")
  string(REPLACE ";" "', '" ANDROID_SAMPLE_RES_PATH "['${ANDROID_SAMPLE_RES_PATH}']")
  configure_file("${path}/build.gradle.in" "${ANDROID_BUILD_BASE_DIR}/${__dir}/build.gradle" @ONLY)

  file(APPEND "${ANDROID_BUILD_BASE_DIR}/settings.gradle" "
include ':${__dir}'
")

  if (BUILD_ANDROID_EXAMPLES)
    # build apk
    set(APK_FILE "${ANDROID_BUILD_BASE_DIR}/${__dir}/build/outputs/apk/release/${__dir}-${ANDROID_ABI}-release-unsigned.apk")
    ocv_update(OPENCV_GRADLE_VERBOSE_OPTIONS "-i")
    add_custom_command(
        OUTPUT "${APK_FILE}" "${OPENCV_DEPHELPER}/android_sample_${__dir}"
        COMMAND ./gradlew ${OPENCV_GRADLE_VERBOSE_OPTIONS} "${__dir}:assemble"
        COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/android_sample_${__dir}"
        WORKING_DIRECTORY "${ANDROID_BUILD_BASE_DIR}"
        DEPENDS ${depends} opencv_java_android
        COMMENT "Building OpenCV Android sample project: ${__dir}"
    )
  else()  # install only
    # copy samples
    add_custom_command(
        OUTPUT "${OPENCV_DEPHELPER}/android_sample_${__dir}"
        COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/android_sample_${__dir}"
        WORKING_DIRECTORY "${ANDROID_BUILD_BASE_DIR}"
        DEPENDS ${depends} opencv_java_android
        COMMENT "Copying OpenCV Android sample project: ${__dir}"
    )
  endif()

  file(REMOVE "${OPENCV_DEPHELPER}/android_sample_${__dir}")  # force rebuild after CMake run

  add_custom_target(android_sample_${__dir} ALL DEPENDS "${OPENCV_DEPHELPER}/android_sample_${__dir}" SOURCES "${ANDROID_SAMPLE_MANIFEST_PATH}")

  #
  # Install
  #
  set(ANDROID_SAMPLE_JNI_PATH "jni")
  set(ANDROID_SAMPLE_JAVA_PATH "src")
  set(ANDROID_SAMPLE_RES_PATH "res")
  set(ANDROID_SAMPLE_MANIFEST_PATH "AndroidManifest.xml")

  install(DIRECTORY "${path}/res" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}/${__dir}" COMPONENT samples OPTIONAL)
  install(DIRECTORY "${path}/src" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}/${__dir}" COMPONENT samples)
  install(DIRECTORY "${path}/jni" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}/${__dir}" COMPONENT samples OPTIONAL)

  install(FILES "${path}/gradle/AndroidManifest.xml" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}/${__dir}" COMPONENT samples)

  set(ANDROID_ABI_FILTER "${ANDROID_INSTALL_ABI_FILTER}")
  set(ANDROID_PROJECT_JNI_PATH "native/jni")

  string(REPLACE ";" "', '" ANDROID_SAMPLE_JAVA_PATH "['${ANDROID_SAMPLE_JAVA_PATH}']")
  string(REPLACE ";" "', '" ANDROID_SAMPLE_RES_PATH "['${ANDROID_SAMPLE_RES_PATH}']")
  configure_file("${path}/build.gradle.in" "${ANDROID_TMP_INSTALL_BASE_DIR}/${__dir}/build.gradle" @ONLY)
  install(FILES "${ANDROID_TMP_INSTALL_BASE_DIR}/${__dir}/build.gradle" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}/${__dir}" COMPONENT samples)

  file(APPEND "${ANDROID_TMP_INSTALL_BASE_DIR}/settings.gradle" "
include ':${__dir}'
")

endmacro()

install(FILES "${ANDROID_TMP_INSTALL_BASE_DIR}/settings.gradle" DESTINATION "${ANDROID_INSTALL_SAMPLES_DIR}" COMPONENT samples)
