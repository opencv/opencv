IF(NOT ANDROID OR NOT WITH_ANDROID_CAMERA)
  return()
ENDIF()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/camera_wrapper")
set(OPENCV_ANDROIDCAMERA_MODULE_TYPE STATIC)
define_opencv_moduleEx(androidcamera INTERNAL log dl)

if (BUILD_ANDROID_CAMERA_WRAPPER)
  add_subdirectory(camera_wrapper)
else()
  file(GLOB camera_wrappers "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/lib/${ANDROID_NDK_ABI_NAME}/libnative_camera_r*.so")

  foreach(wrapper ${camera_wrappers})
    ADD_CUSTOM_COMMAND(
        TARGET ${the_target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${wrapper}" "${LIBRARY_OUTPUT_PATH}"
    )
    get_filename_component(wrapper_name "${wrapper}" NAME)
    install(FILES "${LIBRARY_OUTPUT_PATH}/${wrapper_name}"
          DESTINATION ${OPENCV_LIB_INSTALL_PATH}
          COMPONENT main)
  endforeach()
endif()
