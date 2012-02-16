# --------------------------------------------------------------------------------------------
#  Installation for CMake Module:  OpenCVConfig.cmake
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use with "make install"
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# -------------------------------------------------------------------------------------------

if(INSTALL_TO_MANGLED_PATHS)
  set(OpenCV_USE_MANGLED_PATHS TRUE)
else()
  set(OpenCV_USE_MANGLED_PATHS FALSE)
endif()

if(NOT OpenCV_CUDA_CC)
  set(OpenCV_CUDA_CC_CONFIGMAKE "\"\"")
else()
  set(OpenCV_CUDA_CC_CONFIGMAKE "${OpenCV_CUDA_CC}")
endif()

if(NOT ANDROID_NATIVE_API_LEVEL)
  set(OpenCV_ANDROID_NATIVE_API_LEVEL_CONFIGMAKE 0)
else()
  set(OpenCV_ANDROID_NATIVE_API_LEVEL_CONFIGMAKE "${ANDROID_NATIVE_API_LEVEL}")
endif()


#build list of modules available for the OpenCV user
set(OpenCV_LIB_COMPONENTS "")
foreach(m ${OPENCV_MODULES_PUBLIC})
  list(INSERT OpenCV_LIB_COMPONENTS 0 ${${m}_MODULE_DEPS_OPT} ${m})
endforeach()
ocv_list_unique(OpenCV_LIB_COMPONENTS)
set(OPENCV_MODULES_CONFIGMAKE ${OpenCV_LIB_COMPONENTS})
ocv_list_filterout(OpenCV_LIB_COMPONENTS "^opencv_")
if(OpenCV_LIB_COMPONENTS)
  list(REMOVE_ITEM OPENCV_MODULES_CONFIGMAKE ${OpenCV_LIB_COMPONENTS})
endif()

macro(ocv_generate_dependencies_map_configmake suffix configuration)
  set(OPENCV_DEPENDENCIES_MAP_${suffix} "")
  set(OPENCV_PROCESSED_LIBS "")
  set(OPENCV_LIBS_TO_PROCESS ${OPENCV_MODULES_CONFIGMAKE})
  while(OPENCV_LIBS_TO_PROCESS)
    list(GET OPENCV_LIBS_TO_PROCESS 0 __ocv_lib)
    get_target_property(__libname ${__ocv_lib} LOCATION_${configuration})
    get_filename_component(__libname "${__libname}" NAME)
    
    if(WIN32)
      string(REGEX REPLACE "${CMAKE_SHARED_LIBRARY_SUFFIX}$" "${CMAKE_LINK_LIBRARY_SUFFIX}" __libname "${__libname}")
    endif()
  
    set(OPENCV_DEPENDENCIES_MAP_${suffix} "${OPENCV_DEPENDENCIES_MAP_${suffix}}set(OpenCV_${__ocv_lib}_LIBNAME_${suffix} \"${__libname}\")\n")
    set(OPENCV_DEPENDENCIES_MAP_${suffix} "${OPENCV_DEPENDENCIES_MAP_${suffix}}set(OpenCV_${__ocv_lib}_DEPS_${suffix} ${${__ocv_lib}_MODULE_DEPS_${suffix}})\n")
    set(OPENCV_DEPENDENCIES_MAP_${suffix} "${OPENCV_DEPENDENCIES_MAP_${suffix}}set(OpenCV_${__ocv_lib}_EXTRA_DEPS_${suffix} ${${__ocv_lib}_EXTRA_DEPS_${suffix}})\n")

    list(APPEND OPENCV_PROCESSED_LIBS ${__ocv_lib})
    list(APPEND OPENCV_LIBS_TO_PROCESS ${${__ocv_lib}_MODULE_DEPS_${suffix}})
    list(REMOVE_ITEM OPENCV_LIBS_TO_PROCESS ${OPENCV_PROCESSED_LIBS})
  endwhile()
  unset(OPENCV_PROCESSED_LIBS)
  unset(OPENCV_LIBS_TO_PROCESS)
  unset(__ocv_lib)
  unset(__libname)
endmacro()

ocv_generate_dependencies_map_configmake(OPT Release)
ocv_generate_dependencies_map_configmake(DBG Debug)

# -------------------------------------------------------------------------------------------
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------
set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${OpenCV_SOURCE_DIR}/include\" \"${OpenCV_SOURCE_DIR}/include/opencv\"")
set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"${LIBRARY_OUTPUT_PATH}\"")
set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"${CMAKE_BINARY_DIR}/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")

#set(CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "${OpenCV_SOURCE_DIR}")
set(CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "")
foreach(m ${OPENCV_MODULES_BUILD})
  if(EXISTS "${OPENCV_MODULE_${m}_LOCATION}/include")
    list(APPEND CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "${OPENCV_MODULE_${m}_LOCATION}/include")
  endif()
endforeach()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig.cmake" IMMEDIATE @ONLY)

#support for version checking when finding opencv. find_package(OpenCV 2.3.1 EXACT) should now work.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig-version.cmake" IMMEDIATE @ONLY)


# --------------------------------------------------------------------------------------------
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use *with* "make install"
# -------------------------------------------------------------------------------------------
set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_PREFIX}/opencv" "\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_PREFIX}\"")

set(CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "\"\"")
if(ANDROID)
    set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/libs/\${ANDROID_NDK_ABI_NAME}\"")
    set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/share/OpenCV/3rdparty/libs/\${ANDROID_NDK_ABI_NAME}\"")
else()
    set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/${OPENCV_LIB_INSTALL_PATH}\"")
    set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")
    if(INSTALL_TO_MANGLED_PATHS)
      set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/share/OpenCV-${OPENCV_VERSION}/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")
    endif()
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake" IMMEDIATE @ONLY)

if(UNIX)
  #http://www.vtk.org/Wiki/CMake/Tutorials/Packaging reference
  # For a command "find_package(<name> [major[.minor]] [EXACT] [REQUIRED|QUIET])"
  # cmake will look in the following dir on unix:
  #                <prefix>/(share|lib)/cmake/<name>*/                     (U)
  #                <prefix>/(share|lib)/<name>*/                           (U)
  #                <prefix>/(share|lib)/<name>*/(cmake|CMake)/             (U)
  if(INSTALL_TO_MANGLED_PATHS)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake DESTINATION share/OpenCV-${OPENCV_VERSION}/)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake DESTINATION share/OpenCV-${OPENCV_VERSION}/)
  else()
    install(FILES "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake" DESTINATION share/OpenCV/)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake DESTINATION share/OpenCV/)
  endif()
endif()

if(ANDROID)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/android/android.toolchain.cmake" DESTINATION share/OpenCV)
endif()

# --------------------------------------------------------------------------------------------
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# -------------------------------------------------------------------------------------------
if(WIN32)
    set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"\${OpenCV_CONFIG_PATH}/include\" \"\${OpenCV_CONFIG_PATH}/include/opencv\"")
    set(CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "\"\"")
    set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_CONFIG_PATH}/${OPENCV_LIB_INSTALL_PATH}\"")
    set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_CONFIG_PATH}/share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")

    exec_program(mkdir ARGS "-p \"${CMAKE_BINARY_DIR}/win-install/\"" OUTPUT_VARIABLE RET_VAL)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)

    # Install the OpenCVConfig.cmake file which has the right paths pointing to the install directory
    install(FILES "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" DESTINATION "${CMAKE_INSTALL_PREFIX}/")
endif()

