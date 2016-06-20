# --------------------------------------------------------------------------------------------
#  Installation for CMake Module:  OpenCVConfig.cmake
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use with "make install"
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# -------------------------------------------------------------------------------------------

if(INSTALL_TO_MANGLED_PATHS)
  set(OpenCV_USE_MANGLED_PATHS_CONFIGCMAKE TRUE)
else()
  set(OpenCV_USE_MANGLED_PATHS_CONFIGCMAKE FALSE)
endif()

if(HAVE_CUDA)
  ocv_cmake_configure("${CMAKE_CURRENT_LIST_DIR}/templates/OpenCVConfig-CUDA.cmake.in" CUDA_CONFIGCMAKE @ONLY)
endif()

if(ANDROID)
  if(NOT ANDROID_NATIVE_API_LEVEL)
    set(OpenCV_ANDROID_NATIVE_API_LEVEL_CONFIGCMAKE 0)
  else()
    set(OpenCV_ANDROID_NATIVE_API_LEVEL_CONFIGCMAKE "${ANDROID_NATIVE_API_LEVEL}")
  endif()
  ocv_cmake_configure("${CMAKE_CURRENT_LIST_DIR}/templates/OpenCVConfig-ANDROID.cmake.in" ANDROID_CONFIGCMAKE @ONLY)
endif()

set(OPENCV_MODULES_CONFIGCMAKE ${OPENCV_MODULES_PUBLIC})

if(BUILD_FAT_JAVA_LIB AND HAVE_opencv_java)
  list(APPEND OPENCV_MODULES_CONFIGCMAKE opencv_java)
endif()

# -------------------------------------------------------------------------------------------
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------
set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${OpenCV_SOURCE_DIR}/include\" \"${OpenCV_SOURCE_DIR}/include/opencv\"")

foreach(m ${OPENCV_MODULES_BUILD})
  if(EXISTS "${OPENCV_MODULE_${m}_LOCATION}/include")
    set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "${OpenCV_INCLUDE_DIRS_CONFIGCMAKE} \"${OPENCV_MODULE_${m}_LOCATION}/include\"")
  endif()
endforeach()

export(TARGETS ${OpenCVModules_TARGETS} FILE "${CMAKE_BINARY_DIR}/OpenCVModules.cmake")

if(TARGET ippicv AND NOT BUILD_SHARED_LIBS)
  set(USE_IPPICV TRUE)
  file(RELATIVE_PATH IPPICV_INSTALL_PATH_RELATIVE_CONFIGCMAKE ${CMAKE_BINARY_DIR} ${IPPICV_LOCATION_PATH})
  ocv_cmake_configure("${CMAKE_CURRENT_LIST_DIR}/templates/OpenCVConfig-IPPICV.cmake.in" IPPICV_CONFIGCMAKE @ONLY)
else()
  set(USE_IPPICV FALSE)
endif()

configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig.cmake" @ONLY)
#support for version checking when finding opencv. find_package(OpenCV 2.3.1 EXACT) should now work.
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig-version.cmake" @ONLY)

# --------------------------------------------------------------------------------------------
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use *with* "make install"
# -------------------------------------------------------------------------------------------
file(RELATIVE_PATH OpenCV_INSTALL_PATH_RELATIVE_CONFIGCMAKE "${CMAKE_INSTALL_PREFIX}/${OPENCV_CONFIG_INSTALL_PATH}/" ${CMAKE_INSTALL_PREFIX})
set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_INSTALL_PATH}\" \"\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_INSTALL_PATH}/opencv\"")

if(USE_IPPICV)
  file(RELATIVE_PATH IPPICV_INSTALL_PATH_RELATIVE_CONFIGCMAKE "${CMAKE_INSTALL_PREFIX}" ${IPPICV_INSTALL_PATH})
  ocv_cmake_configure("${CMAKE_CURRENT_LIST_DIR}/templates/OpenCVConfig-IPPICV.cmake.in" IPPICV_CONFIGCMAKE @ONLY)
endif()

function(ocv_gen_config TMP_DIR NESTED_PATH ROOT_NAME)
  ocv_path_join(__install_nested "${OPENCV_CONFIG_INSTALL_PATH}" "${NESTED_PATH}")
  ocv_path_join(__tmp_nested "${TMP_DIR}" "${NESTED_PATH}")

  file(RELATIVE_PATH OpenCV_INSTALL_PATH_RELATIVE_CONFIGCMAKE "${CMAKE_INSTALL_PREFIX}/${__install_nested}" "${CMAKE_INSTALL_PREFIX}/")

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig-version.cmake.in" "${TMP_DIR}/OpenCVConfig-version.cmake" @ONLY)

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${__tmp_nested}/OpenCVConfig.cmake" @ONLY)
  install(EXPORT OpenCVModules DESTINATION "${__install_nested}" FILE OpenCVModules.cmake COMPONENT dev)
  install(FILES
      "${TMP_DIR}/OpenCVConfig-version.cmake"
      "${__tmp_nested}/OpenCVConfig.cmake"
      DESTINATION "${CMAKE_INSTALL_PREFIX}/${__install_nested}" COMPONENT dev)

  if(ROOT_NAME)
    # Root config file
    configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/${ROOT_NAME}" "${TMP_DIR}/OpenCVConfig.cmake" @ONLY)
    install(FILES
        "${TMP_DIR}/OpenCVConfig-version.cmake"
        "${TMP_DIR}/OpenCVConfig.cmake"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${OPENCV_CONFIG_INSTALL_PATH}" COMPONENT dev)
  endif()
endfunction()

if(UNIX AND NOT ANDROID)
  ocv_gen_config("${CMAKE_BINARY_DIR}/unix-install" "" "")
endif()

if(ANDROID)
  ocv_gen_config("${CMAKE_BINARY_DIR}/unix-install" "abi-${ANDROID_NDK_ABI_NAME}" "OpenCVConfig.root-ANDROID.cmake.in")
  install(FILES "${OpenCV_SOURCE_DIR}/platforms/android/android.toolchain.cmake" DESTINATION "${OPENCV_CONFIG_INSTALL_PATH}" COMPONENT dev)
endif()

# --------------------------------------------------------------------------------------------
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# --------------------------------------------------------------------------------------------
if(WIN32)
  if(CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
    if(BUILD_SHARED_LIBS)
      set(_lib_suffix "lib")
    else()
      set(_lib_suffix "staticlib")
    endif()
    ocv_gen_config("${CMAKE_BINARY_DIR}/win-install" "${OpenCV_INSTALL_BINARIES_PREFIX}${_lib_suffix}" "OpenCVConfig.root-WIN32.cmake.in")
  else()
    ocv_gen_config("${CMAKE_BINARY_DIR}/win-install" "" "")
  endif()
endif()
