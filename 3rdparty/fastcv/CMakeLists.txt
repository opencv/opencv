if(FCV_ENABLE)
  set(OPENCV_3P_FASTCV_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(FASTCV_LIB
    "${FCV_LIB_DIR}/libfastcvopt.so")

  set(FASTCV_HAL_VERSION 0.0.1 CACHE INTERNAL "")
  set(FASTCV_HAL_LIBRARIES "fastcv_hal" CACHE INTERNAL "")
  set(FASTCV_HAL_INCLUDE_DIRS
  "${OPENCV_3P_FASTCV_DIR}/include"
  ${FCV_HEADER_DIR}
  CACHE INTERNAL "")

  file(GLOB FASTCV_HAL_HEADERS  "${OPENCV_3P_FASTCV_DIR}/include/*.hpp")
  file(GLOB FASTCV_HAL_FILES    "${OPENCV_3P_FASTCV_DIR}/src/*.cpp")
  add_library(fastcv_hal STATIC "${FASTCV_HAL_FILES}")

  target_include_directories(fastcv_hal PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/core/include
    ${CMAKE_SOURCE_DIR}/modules/imgproc/include
    ${FASTCV_HAL_INCLUDE_DIRS})

  target_link_libraries(fastcv_hal
    PUBLIC "${FCV_LIB_DIR}/libfastcvopt.so")

  set_target_properties(fastcv_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})

  if(NOT BUILD_SHARED_LIBS)
    ocv_install_target(fastcv_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
  endif()

else()
  message(STATUS "FastCV is not available, disabling related HAL and stuff")
endif()