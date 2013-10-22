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

if(NOT OpenCV_CUDA_CC)
  set(OpenCV_CUDA_CC_CONFIGCMAKE "\"\"")
  set(OpenCV_CUDA_VERSION "")
else()
  set(OpenCV_CUDA_CC_CONFIGCMAKE "${OpenCV_CUDA_CC}")
  set(OpenCV_CUDA_VERSION ${CUDA_VERSION_STRING})
endif()

if(NOT ANDROID_NATIVE_API_LEVEL)
  set(OpenCV_ANDROID_NATIVE_API_LEVEL_CONFIGCMAKE 0)
else()
  set(OpenCV_ANDROID_NATIVE_API_LEVEL_CONFIGCMAKE "${ANDROID_NATIVE_API_LEVEL}")
endif()

if(CMAKE_GENERATOR MATCHES "Visual" OR CMAKE_GENERATOR MATCHES "Xcode")
  set(OpenCV_ADD_DEBUG_RELEASE_CONFIGCMAKE TRUE)
else()
  set(OpenCV_ADD_DEBUG_RELEASE_CONFIGCMAKE FALSE)
endif()



if(WIN32)
  if(MINGW)
    set(OPENCV_LINK_LIBRARY_SUFFIX ".dll.a")
  else()
    set(OPENCV_LINK_LIBRARY_SUFFIX ".lib")
  endif()
endif()

#build list of modules available for the OpenCV user
set(OpenCV_LIB_COMPONENTS "")
foreach(m ${OPENCV_MODULES_PUBLIC})
  list(INSERT OpenCV_LIB_COMPONENTS 0 ${${m}_MODULE_DEPS_OPT} ${m})
endforeach()
ocv_list_unique(OpenCV_LIB_COMPONENTS)
set(OPENCV_MODULES_CONFIGCMAKE ${OpenCV_LIB_COMPONENTS})
ocv_list_filterout(OpenCV_LIB_COMPONENTS "^opencv_")
if(OpenCV_LIB_COMPONENTS)
  list(REMOVE_ITEM OPENCV_MODULES_CONFIGCMAKE ${OpenCV_LIB_COMPONENTS})
endif()

if(BUILD_FAT_JAVA_LIB AND HAVE_opencv_java)
  list(APPEND OPENCV_MODULES_CONFIGCMAKE opencv_java)
endif()

# -------------------------------------------------------------------------------------------
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------
set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${OpenCV_SOURCE_DIR}/include\" \"${OpenCV_SOURCE_DIR}/include/opencv\"")

set(OpenCV2_INCLUDE_DIRS_CONFIGCMAKE "")
foreach(m ${OPENCV_MODULES_BUILD})
  if(EXISTS "${OPENCV_MODULE_${m}_LOCATION}/include")
    list(APPEND OpenCV2_INCLUDE_DIRS_CONFIGCMAKE "${OPENCV_MODULE_${m}_LOCATION}/include")
  endif()
endforeach()

if(ANDROID AND NOT BUILD_SHARED_LIBS AND HAVE_TBB)
  #export TBB headers location because static linkage of TBB might be troublesome if application wants to use TBB itself
  list(APPEND OpenCV2_INCLUDE_DIRS_CONFIGCMAKE ${TBB_INCLUDE_DIRS})
endif()

export(TARGETS ${OpenCVModules_TARGETS} FILE "${CMAKE_BINARY_DIR}/OpenCVModules.cmake")

configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig.cmake" IMMEDIATE @ONLY)
#support for version checking when finding opencv. find_package(OpenCV 2.3.1 EXACT) should now work.
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig-version.cmake" IMMEDIATE @ONLY)

# --------------------------------------------------------------------------------------------
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use *with* "make install"
# -------------------------------------------------------------------------------------------
set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_INSTALL_PATH}/opencv" "\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_INSTALL_PATH}\"")

set(OpenCV2_INCLUDE_DIRS_CONFIGCMAKE "\"\"")
if(INSTALL_TO_MANGLED_PATHS)
  string(REPLACE "OpenCV" "OpenCV-${OPENCV_VERSION}" OpenCV_3RDPARTY_LIB_DIRS_CONFIGCMAKE "${OPENCV_3P_LIB_INSTALL_PATH}")
  set(OpenCV_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/${OpenCV_3RDPARTY_LIB_DIRS_CONFIGCMAKE}\"")
endif()

configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake" IMMEDIATE @ONLY)

if(UNIX)
  #http://www.vtk.org/Wiki/CMake/Tutorials/Packaging reference
  # For a command "find_package(<name> [major[.minor]] [EXACT] [REQUIRED|QUIET])"
  # cmake will look in the following dir on unix:
  #                <prefix>/(share|lib)/cmake/<name>*/                     (U)
  #                <prefix>/(share|lib)/<name>*/                           (U)
  #                <prefix>/(share|lib)/<name>*/(cmake|CMake)/             (U)
  if(INSTALL_TO_MANGLED_PATHS)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake DESTINATION ${OPENCV_CONFIG_INSTALL_PATH}-${OPENCV_VERSION}/)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake DESTINATION ${OPENCV_CONFIG_INSTALL_PATH}-${OPENCV_VERSION}/)
    install(EXPORT OpenCVModules DESTINATION ${OPENCV_CONFIG_INSTALL_PATH}-${OPENCV_VERSION}/)
  else()
    install(FILES "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake" DESTINATION ${OPENCV_CONFIG_INSTALL_PATH}/)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake DESTINATION ${OPENCV_CONFIG_INSTALL_PATH}/)
    install(EXPORT OpenCVModules DESTINATION ${OPENCV_CONFIG_INSTALL_PATH}/)
  endif()
endif()

if(ANDROID)
  install(FILES "${OpenCV_SOURCE_DIR}/platforms/android/android.toolchain.cmake" DESTINATION ${OPENCV_CONFIG_INSTALL_PATH}/)
endif()

# --------------------------------------------------------------------------------------------
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# --------------------------------------------------------------------------------------------
if(WIN32)
  set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "\"\${OpenCV_CONFIG_PATH}/include\" \"\${OpenCV_CONFIG_PATH}/include/opencv\"")
  set(OpenCV2_INCLUDE_DIRS_CONFIGCMAKE "\"\"")

  exec_program(mkdir ARGS "-p \"${CMAKE_BINARY_DIR}/win-install/\"" OUTPUT_VARIABLE RET_VAL)
  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)
  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig-version.cmake" IMMEDIATE @ONLY)
  if(BUILD_SHARED_LIBS)
    install(FILES "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" DESTINATION "${OpenCV_INSTALL_BINARIES_PREFIX}/lib")
    install(EXPORT OpenCVModules DESTINATION "${OpenCV_INSTALL_BINARIES_PREFIX}/lib")
  else()
    install(FILES "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" DESTINATION "${OpenCV_INSTALL_BINARIES_PREFIX}/staticlib")
    install(EXPORT OpenCVModules DESTINATION "${OpenCV_INSTALL_BINARIES_PREFIX}/staticlib")
  endif()
  install(FILES "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig-version.cmake" DESTINATION "${CMAKE_INSTALL_PREFIX}")
  install(FILES "${OpenCV_SOURCE_DIR}/cmake/OpenCVConfig.cmake" DESTINATION "${CMAKE_INSTALL_PREFIX}/")
endif()
