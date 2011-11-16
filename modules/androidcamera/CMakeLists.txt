IF(NOT ANDROID)
  MESSAGE( FATAL_ERROR "This project is for ANDROID only" )
ENDIF()

if (BUILD_ANDROID_CAMERA_WRAPPER)
  add_subdirectory(camera_wrapper)
endif()

project(opencv_androidcamera)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/camera_wrapper )
SET( LIBRARY_DEPS ${LIBRARY_DEPS} log dl )

SET( the_target opencv_androidcamera )

ADD_LIBRARY( ${the_target} STATIC src/camera_activity.cpp )

if (BUILD_SHARED_LIBS)
  add_definitions(-DCVAPI_EXPORTS) 
endif()

TARGET_LINK_LIBRARIES( ${the_target} ${LIBRARY_DEPS} )

SET_TARGET_PROPERTIES(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      OUTPUT_NAME "${the_target}${OPENCV_DLLVERSION}"
                      DEFINE_SYMBOL "CVAPI_EXPORTS"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      )

IF (NOT BUILD_SHARED_LIBS)
  install(TARGETS ${the_target}
    RUNTIME DESTINATION bin COMPONENT main
    ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
    LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
    )
ENDIF()

if (NOT BUILD_ANDROID_CAMERA_WRAPPER)
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
