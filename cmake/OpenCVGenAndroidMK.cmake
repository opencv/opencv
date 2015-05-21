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

  if(BUILD_FAT_JAVA_LIB)
    set(OPENCV_LIBTYPE_CONFIGMAKE "SHARED")
    set(OPENCV_STATIC_LIBTYPE_CONFIGMAKE "STATIC")
  else()
    set(OPENCV_STATIC_LIBTYPE_CONFIGMAKE ${OPENCV_LIBTYPE_CONFIGMAKE})
  endif()

  if (NOT COMMAND ANDROID_GET_ABI_RAWNAME)
    macro( ANDROID_GET_ABI_RAWNAME TOOLCHAIN_FLAG VAR )
      if( " ${TOOLCHAIN_FLAG}" STREQUAL " ARMEABI" )
        set( ${VAR} "armeabi" )
      elseif( " ${TOOLCHAIN_FLAG}" STREQUAL " ARMEABI_V7A" )
        set( ${VAR} "armeabi-v7a" )
      elseif( " ${TOOLCHAIN_FLAG}" STREQUAL " ARM64_V8A" )
        set( ${VAR} "arm64-v8a" )
      elseif( " ${TOOLCHAIN_FLAG}" STREQUAL " X86" )
        set( ${VAR} "x86" )
      elseif( " ${TOOLCHAIN_FLAG}" STREQUAL " MIPS" )
        set( ${VAR} "mips" )
      else()
        set( ${VAR} "unknown" )
      endif()
    endmacro()
  endif()

  # setup lists of camera libs
  foreach(abi ARMEABI ARMEABI_V7A ARM64_V8A X86 MIPS)
    ANDROID_GET_ABI_RAWNAME(${abi} ndkabi)
    if(BUILD_ANDROID_CAMERA_WRAPPER)
      if(ndkabi STREQUAL ANDROID_NDK_ABI_NAME)
        set(OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE "native_camera_r${ANDROID_VERSION}")
      else()
        set(OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE "")
      endif()
    elseif(HAVE_opencv_androidcamera)
      set(OPENCV_CAMERA_LIBS_${abi}_CONFIGCMAKE "")
      # TODO: add prebuild camera libs for arm64-v8a
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
  set(OPENCV_HAVE_GPU_MODULE_CONFIGMAKE "off")
  set(OPENCV_EXTRA_COMPONENTS_CONFIGMAKE "")
  set(OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE "")
  foreach(m ${OPENCV_MODULES_PUBLIC})
    list(INSERT OPENCV_MODULES_CONFIGMAKE 0 ${${m}_MODULE_DEPS_${ocv_optkind}} ${m})
    if(${m}_EXTRA_DEPS_${ocv_optkind})
      list(INSERT OPENCV_EXTRA_COMPONENTS_CONFIGMAKE 0 ${${m}_EXTRA_DEPS_${ocv_optkind}})
    endif()
  endforeach()

  # remove CUDA runtime and NPP from regular deps
  # it can be added separately if needed.
  ocv_list_filterout(OPENCV_EXTRA_COMPONENTS_CONFIGMAKE "cusparse")
  ocv_list_filterout(OPENCV_EXTRA_COMPONENTS_CONFIGMAKE "cufft")
  ocv_list_filterout(OPENCV_EXTRA_COMPONENTS_CONFIGMAKE "cublas")
  ocv_list_filterout(OPENCV_EXTRA_COMPONENTS_CONFIGMAKE "npp")
  ocv_list_filterout(OPENCV_EXTRA_COMPONENTS_CONFIGMAKE "cudart")

  if(HAVE_CUDA)
    # CUDA runtime libraries and are required always
    set(culibs ${CUDA_LIBRARIES})

    # right now NPP is requared always too
    list(INSERT culibs 0 ${CUDA_npp_LIBRARY})

    if(HAVE_CUFFT)
      list(INSERT culibs 0 ${CUDA_cufft_LIBRARY})
    endif()

    if(HAVE_CUBLAS)
      list(INSERT culibs 0 ${CUDA_cublas_LIBRARY})
    endif()
  endif()

  ocv_convert_to_lib_name(CUDA_RUNTIME_LIBS_CONFIGMAKE ${culibs})

  # split 3rdparty libs and modules
  foreach(mod ${OPENCV_MODULES_CONFIGMAKE})
    if(NOT mod MATCHES "^opencv_.+$")
      list(INSERT OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE 0 ${mod})
    endif()
  endforeach()
  if(OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE)
    list(REMOVE_ITEM OPENCV_MODULES_CONFIGMAKE ${OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE})
  endif()

  if(ENABLE_DYNAMIC_CUDA)
    set(OPENCV_DYNAMICUDA_MODULE_CONFIGMAKE "dynamicuda")
  endif()

  # GPU module enabled separately
  list(REMOVE_ITEM OPENCV_MODULES_CONFIGMAKE "opencv_gpu")
  list(REMOVE_ITEM OPENCV_MODULES_CONFIGMAKE "opencv_dynamicuda")

  if(HAVE_opencv_gpu)
    set(OPENCV_HAVE_GPU_MODULE_CONFIGMAKE "on")
  endif()

  # convert CMake lists to makefile literals
  foreach(lst OPENCV_MODULES_CONFIGMAKE OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE OPENCV_EXTRA_COMPONENTS_CONFIGMAKE)
    ocv_list_unique(${lst})
    ocv_list_reverse(${lst})
    string(REPLACE ";" " " ${lst} "${${lst}}")
  endforeach()
  string(REPLACE "opencv_" "" OPENCV_MODULES_CONFIGMAKE "${OPENCV_MODULES_CONFIGMAKE}")
  string(REPLACE ";" " " CUDA_RUNTIME_LIBS_CONFIGMAKE "${CUDA_RUNTIME_LIBS_CONFIGMAKE}")

  # prepare 3rd-party component list without TBB for armeabi and mips platforms. TBB is useless there.
  set(OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE_NO_TBB ${OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE})
  foreach(mod ${OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE_NO_TBB})
     string(REPLACE "tbb" "" OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE_NO_TBB "${OPENCV_3RDPARTY_COMPONENTS_CONFIGMAKE_NO_TBB}")
  endforeach()

  if(BUILD_FAT_JAVA_LIB)
    set(OPENCV_LIBS_CONFIGMAKE java)
  else()
    set(OPENCV_LIBS_CONFIGMAKE "${OPENCV_MODULES_CONFIGMAKE}")
  endif()

  # -------------------------------------------------------------------------------------------
  #  Part 1/2: ${BIN_DIR}/OpenCV.mk              -> For use *without* "make install"
  # -------------------------------------------------------------------------------------------
  set(OPENCV_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${OpenCV_SOURCE_DIR}/include\" \"${OpenCV_SOURCE_DIR}/include/opencv\"")
  set(OPENCV_BASE_INCLUDE_DIR_CONFIGCMAKE "\"${OpenCV_SOURCE_DIR}\"")
  set(OPENCV_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)/lib/\$(OPENCV_TARGET_ARCH_ABI)")
  set(OPENCV_3RDPARTY_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)/3rdparty/lib/\$(OPENCV_TARGET_ARCH_ABI)")

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCV.mk.in" "${CMAKE_BINARY_DIR}/OpenCV.mk" @ONLY)

  # -------------------------------------------------------------------------------------------
  #  Part 2/2: ${BIN_DIR}/unix-install/OpenCV.mk -> For use with "make install"
  # -------------------------------------------------------------------------------------------
  set(OPENCV_INCLUDE_DIRS_CONFIGCMAKE "\"\$(LOCAL_PATH)/\$(OPENCV_THIS_DIR)/include/opencv\" \"\$(LOCAL_PATH)/\$(OPENCV_THIS_DIR)/include\"")
  set(OPENCV_BASE_INCLUDE_DIR_CONFIGCMAKE "")
  set(OPENCV_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)/../libs/\$(OPENCV_TARGET_ARCH_ABI)")
  set(OPENCV_3RDPARTY_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)/../3rdparty/libs/\$(OPENCV_TARGET_ARCH_ABI)")

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/OpenCV.mk.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCV.mk" @ONLY)
  install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCV.mk DESTINATION ${OPENCV_CONFIG_INSTALL_PATH} COMPONENT dev)
endif(ANDROID)
