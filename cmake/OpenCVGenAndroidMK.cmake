if(ANDROID)
  # --------------------------------------------------------------------------------------------
  #  Installation for Android ndk-build makefile:  OpenCV.mk
  #  Part 1/2: ${BIN_DIR}/OpenCV.mk              -> For use *without* "make install"
  #  Part 2/2: ${BIN_DIR}/unix-install/OpenCV.mk -> For use with "make install"
  # -------------------------------------------------------------------------------------------

  # build type
  if(BUILD_SHARED_LIBS)
    set(OPENCV_LIBTYPE_CONFIGMAKE "SHARED")
  else()
    set(OPENCV_LIBTYPE_CONFIGMAKE "STATIC")
  endif()

  # setup lists of camera libs
  foreach(abi ARMEABI ARMEABI_V7A X86)
    ANDROID_GET_ABI_RAWNAME(${abi} ndkabi)
    if(BUILD_ANDROID_CAMERA_WRAPPER)
      if(ndkabi STREQUAL ANDROID_NDK_ABI_NAME)
        set(OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE "native_camera_r${ANDROID_VERSION}")
      else()
        set(OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE "")
      endif()
    elseif(HAVE_opencv_androidcamera)
      set(OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE "")
      file(GLOB OPENCV_CAMERA_LIBS "${OpenCV_SOURCE_DIR}/3rdparty/lib/${ndkabi}/libnative_camera_r*.so")
      if(OPENCV_CAMERA_LIBS)
        list(SORT OPENCV_CAMERA_LIBS)
      endif()
      foreach(cam_lib ${OPENCV_CAMERA_LIBS})
        get_filename_component(cam_lib "${cam_lib}" NAME)
        string(REGEX REPLACE "lib(native_camera_r[0-9]+\\.[0-9]+\\.[0-9]+)\\.so" "\\1" cam_lib "${cam_lib}")
        set(OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE "${OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE} ${cam_lib}")
      endforeach()
    endif()
  endforeach()

  # build the list of opencv libs and dependencies for all modules
  set(OPENCV_MODULES_CONFIGMAKE "")
  set(OPENCV_EXTRA_COMPONENTS_CONFIGMAKE "")
  set(OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE "")
  foreach(m ${OPENCV_MODULES_PUBLIC})
    list(INSERT OPENCV_MODULES_CONFIGMAKE 0 ${${m}_MODULE_DEPS_${ocv_optkind}} ${m})
    if(${m}_EXTRA_DEPS_${ocv_optkind})
      list(INSERT OPENCV_EXTRA_COMPONENTS_CONFIGMAKE 0 ${${m}_EXTRA_DEPS_${ocv_optkind}})
    endif()
  endforeach()

  # split 3rdparty libs and modules
  foreach(mod ${OPENCV_MODULES_CONFIGMAKE})
    if(NOT mod MATCHES "^opencv_.+$")
      list(INSERT OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE 0 ${mod})
    endif()
  endforeach()
  list(REMOVE_ITEM OPENCV_MODULES_CONFIGMAKE ${OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE})

  # convert CMake lists to makefile literals
  foreach(lst OPENCV_MODULES_CONFIGMAKE OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE OPENCV_EXTRA_COMPONENTS_CONFIGMAKE)
    ocv_list_unique(${lst})
    ocv_list_reverse(${lst})
    string(REPLACE ";" " " ${lst} "${${lst}}")
  endforeach()
  string(REPLACE "opencv_" "" OPENCV_MODULES_CONFIGMAKE "${OPENCV_MODULES_CONFIGMAKE}")

  # -------------------------------------------------------------------------------------------
  #  Part 1/2: ${BIN_DIR}/OpenCV.mk              -> For use *without* "make install"
  # -------------------------------------------------------------------------------------------
  set(OPENCV_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${OpenCV_SOURCE_DIR}/include\" \"${OpenCV_SOURCE_DIR}/include/opencv\"")
  set(OPENCV_BASE_INCLUDE_DIR_CONFIGCMAKE "\"${OpenCV_SOURCE_DIR}\"")
  set(OPENCV_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)")

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCV.mk.in" "${CMAKE_BINARY_DIR}/OpenCV.mk" IMMEDIATE @ONLY)

  # -------------------------------------------------------------------------------------------
  #  Part 2/2: ${BIN_DIR}/unix-install/OpenCV.mk -> For use with "make install"
  # -------------------------------------------------------------------------------------------
  set(OPENCV_INCLUDE_DIRS_CONFIGCMAKE "\"\$(LOCAL_PATH)/\$(OPENCV_THIS_DIR)/../../include/opencv\" \"\$(LOCAL_PATH)/\$(OPENCV_THIS_DIR)/../../include\"")
  set(OPENCV_BASE_INCLUDE_DIR_CONFIGCMAKE "")
  set(OPENCV_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)/../..")

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCV.mk.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCV.mk" IMMEDIATE @ONLY)
  install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCV.mk DESTINATION share/OpenCV/)
endif(ANDROID)
