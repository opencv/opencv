# message(STATUS "Initial install layout:")
# ocv_cmake_dump_vars("OPENCV_.*_INSTALL_PATH")

if(ANDROID)

  ocv_update(OPENCV_BIN_INSTALL_PATH            "sdk/native/bin/${ANDROID_NDK_ABI_NAME}")
  ocv_update(OPENCV_TEST_INSTALL_PATH           "${OPENCV_BIN_INSTALL_PATH}")
  ocv_update(OPENCV_SAMPLES_BIN_INSTALL_PATH    "sdk/native/samples/${ANDROID_NDK_ABI_NAME}")
  ocv_update(OPENCV_LIB_INSTALL_PATH            "sdk/native/libs/${ANDROID_NDK_ABI_NAME}")
  ocv_update(OPENCV_LIB_ARCHIVE_INSTALL_PATH    "sdk/native/staticlibs/${ANDROID_NDK_ABI_NAME}")
  ocv_update(OPENCV_3P_LIB_INSTALL_PATH         "sdk/native/3rdparty/libs/${ANDROID_NDK_ABI_NAME}")
  ocv_update(OPENCV_CONFIG_INSTALL_PATH         "sdk/native/jni")
  ocv_update(OPENCV_INCLUDE_INSTALL_PATH        "sdk/native/jni/include")
  ocv_update(OPENCV_OTHER_INSTALL_PATH          "sdk/etc")
  ocv_update(OPENCV_SAMPLES_SRC_INSTALL_PATH    "samples/native")
  ocv_update(OPENCV_LICENSES_INSTALL_PATH       "${OPENCV_OTHER_INSTALL_PATH}/licenses")
  ocv_update(OPENCV_TEST_DATA_INSTALL_PATH      "sdk/etc/testdata")
  ocv_update(OPENCV_DOC_INSTALL_PATH            "doc")
  ocv_update(OPENCV_JAR_INSTALL_PATH            ".")
  ocv_update(OPENCV_JNI_INSTALL_PATH            "${OPENCV_LIB_INSTALL_PATH}")
  ocv_update(OPENCV_JNI_BIN_INSTALL_PATH        "${OPENCV_JNI_INSTALL_PATH}")

elseif(WIN32 AND CMAKE_HOST_SYSTEM_NAME MATCHES Windows)

  if(DEFINED OpenCV_RUNTIME AND DEFINED OpenCV_ARCH)
    ocv_update(OPENCV_INSTALL_BINARIES_PREFIX "${OpenCV_ARCH}/${OpenCV_RUNTIME}/")
  else()
    message(STATUS "Can't detect runtime and/or arch")
    ocv_update(OPENCV_INSTALL_BINARIES_PREFIX "")
  endif()
  if(OpenCV_STATIC)
    ocv_update(OPENCV_INSTALL_BINARIES_SUFFIX "staticlib")
  else()
    ocv_update(OPENCV_INSTALL_BINARIES_SUFFIX "lib")
  endif()
  if(INSTALL_CREATE_DISTRIB)
    set(_jni_suffix "/${OpenCV_ARCH}")
  else()
    set(_jni_suffix "")
  endif()

  ocv_update(OPENCV_BIN_INSTALL_PATH           "${OPENCV_INSTALL_BINARIES_PREFIX}bin")
  ocv_update(OPENCV_TEST_INSTALL_PATH          "${OPENCV_BIN_INSTALL_PATH}")
  ocv_update(OPENCV_SAMPLES_BIN_INSTALL_PATH   "${OPENCV_INSTALL_BINARIES_PREFIX}samples")
  ocv_update(OPENCV_LIB_INSTALL_PATH           "${OPENCV_INSTALL_BINARIES_PREFIX}${OPENCV_INSTALL_BINARIES_SUFFIX}")
  ocv_update(OPENCV_LIB_ARCHIVE_INSTALL_PATH   "${OPENCV_LIB_INSTALL_PATH}")
  ocv_update(OPENCV_3P_LIB_INSTALL_PATH        "${OPENCV_INSTALL_BINARIES_PREFIX}staticlib")
  ocv_update(OPENCV_CONFIG_INSTALL_PATH        ".")
  ocv_update(OPENCV_INCLUDE_INSTALL_PATH       "include")
  ocv_update(OPENCV_OTHER_INSTALL_PATH         "etc")
  ocv_update(OPENCV_SAMPLES_SRC_INSTALL_PATH   "samples")
  ocv_update(OPENCV_LICENSES_INSTALL_PATH      "${OPENCV_OTHER_INSTALL_PATH}/licenses")
  ocv_update(OPENCV_TEST_DATA_INSTALL_PATH     "testdata")
  ocv_update(OPENCV_DOC_INSTALL_PATH           "doc")
  ocv_update(OPENCV_JAR_INSTALL_PATH           "java")
  ocv_update(OPENCV_JNI_INSTALL_PATH           "${OPENCV_JAR_INSTALL_PATH}${_jni_suffix}")
  ocv_update(OPENCV_JNI_BIN_INSTALL_PATH       "${OPENCV_JNI_INSTALL_PATH}")

else() # UNIX

  include(GNUInstallDirs)
  ocv_update(OPENCV_BIN_INSTALL_PATH           "bin")
  ocv_update(OPENCV_TEST_INSTALL_PATH          "${OPENCV_BIN_INSTALL_PATH}")
  ocv_update(OPENCV_SAMPLES_BIN_INSTALL_PATH   "${OPENCV_BIN_INSTALL_PATH}")
  ocv_update(OPENCV_LIB_INSTALL_PATH           "${CMAKE_INSTALL_LIBDIR}")
  ocv_update(OPENCV_LIB_ARCHIVE_INSTALL_PATH   "${OPENCV_LIB_INSTALL_PATH}")
  ocv_update(OPENCV_3P_LIB_INSTALL_PATH        "share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH}")
  ocv_update(OPENCV_CONFIG_INSTALL_PATH        "share/OpenCV")
  ocv_update(OPENCV_INCLUDE_INSTALL_PATH       "include")
  ocv_update(OPENCV_OTHER_INSTALL_PATH         "share/OpenCV")
  ocv_update(OPENCV_SAMPLES_SRC_INSTALL_PATH   "share/OpenCV/samples")
  ocv_update(OPENCV_LICENSES_INSTALL_PATH      "${CMAKE_INSTALL_DATAROOTDIR}/licenses/opencv3")
  ocv_update(OPENCV_TEST_DATA_INSTALL_PATH     "share/OpenCV/testdata")
  ocv_update(OPENCV_DOC_INSTALL_PATH           "share/OpenCV/doc")
  ocv_update(OPENCV_JAR_INSTALL_PATH           "share/OpenCV/java")
  ocv_update(OPENCV_JNI_INSTALL_PATH           "${OPENCV_JAR_INSTALL_PATH}")
  ocv_update(OPENCV_JNI_BIN_INSTALL_PATH       "${OPENCV_JNI_INSTALL_PATH}")

endif()

ocv_update(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_INCLUDE_INSTALL_PATH ${OPENCV_INCLUDE_INSTALL_PATH}/opencv-${OPENCV_VERSION})
  foreach(v
      OPENCV_CONFIG_INSTALL_PATH
      OPENCV_3P_LIB_INSTALL_PATH
      OPENCV_SAMPLES_SRC_INSTALL_PATH
      OPENCV_DOC_INSTALL_PATH
      OPENCV_JAR_INSTALL_PATH
      OPENCV_TEST_DATA_INSTALL_PATH
      OPENCV_OTHER_INSTALL_PATH
    )
    string(REPLACE "OpenCV" "OpenCV-${OPENCV_VERSION}" ${v} "${${v}}")
    string(REPLACE "opencv" "opencv-${OPENCV_VERSION}" ${v} "${${v}}")
  endforeach()
endif()

# message(STATUS "Final install layout:")
# ocv_cmake_dump_vars("OPENCV_.*_INSTALL_PATH")
