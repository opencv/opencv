IF(NOT ANDROID OR ANDROID_NATIVE_API_LEVEL LESS 8)
  ocv_module_disable(androidcamera)
ENDIF()

set(the_description "Auxiliary module for Android native camera support")
set(OPENCV_MODULE_TYPE STATIC)

ocv_define_module(androidcamera INTERNAL log dl)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/camera_wrapper")

# Android source tree for native camera
SET (ANDROID_SOURCE_TREE "ANDROID_SOURCE_TREE-NOTFOUND" CACHE PATH
  "Path to Android source tree. Set this variable to path to your Android sources to compile libnative_camera_rx.x.x.so for your Android")
SET(BUILD_ANDROID_CAMERA_WRAPPER OFF)
if(ANDROID_SOURCE_TREE)
  FILE(STRINGS "${ANDROID_SOURCE_TREE}/development/sdk/platform_source.properties" ANDROID_VERSION REGEX "Platform\\.Version=[0-9]+\\.[0-9]+(\\.[0-9]+)?" )
  string(REGEX REPLACE "Platform\\.Version=([0-9]+\\.[0-9]+(\\.[0-9]+)?)" "\\1" ANDROID_VERSION "${ANDROID_VERSION}")
  if(ANDROID_VERSION MATCHES "^[0-9]+\\.[0-9]+$")
    SET(ANDROID_VERSION "${ANDROID_VERSION}.0")
  endif()
  if(NOT "${ANDROID_VERSION}" STREQUAL "")
    SET(BUILD_ANDROID_CAMERA_WRAPPER ON)
  endif()
endif()
MARK_AS_ADVANCED(ANDROID_SOURCE_TREE)

# process wrapper libs
if (BUILD_ANDROID_CAMERA_WRAPPER)
  add_subdirectory(camera_wrapper)
else()
  file(GLOB camera_wrappers "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/lib/${ANDROID_NDK_ABI_NAME}/libnative_camera_r*.so")

  foreach(wrapper ${camera_wrappers})
    ADD_CUSTOM_COMMAND(
      TARGET ${the_module} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${wrapper}" "${LIBRARY_OUTPUT_PATH}"
      )
    get_filename_component(wrapper_name "${wrapper}" NAME)
    install(FILES "${LIBRARY_OUTPUT_PATH}/${wrapper_name}"
            DESTINATION ${OPENCV_LIB_INSTALL_PATH}
            COMPONENT main)
  endforeach()
endif()
