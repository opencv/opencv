set(HAL_LIB_NAME "fastcv_hal")

set(FASTCV_HAL_FILES
  fastcv_hal_core.cpp
  fastcv_hal_core.hpp
  fastcv_hal_imgproc.cpp
  fastcv_hal_imgproc.hpp
  fastcv_hal_utils.cpp
  fastcv_hal_utils.hpp
)

add_library(fastcv_hal STATIC "${FASTCV_HAL_FILES}")
target_include_directories(fastcv_hal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include
  fastcv_hal PRIVATE ${FASTCV_DIR}/inc)

set(FASTCV_LIB "/libs/Android/lib64/libfastcvopt.so")

target_link_libraries(fastcv_hal PUBLIC ${FASTCV_DIR}/${FASTCV_LIB})

set_target_properties(fastcv_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(fastcv_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

set(FASTCV_HAL_FOUND TRUE CACHE INTERNAL "")
set(FASTCV_HAL_VERSION 0.0.1 CACHE INTERNAL "")
set(FASTCV_HAL_LIBRARIES ${HAL_LIB_NAME} CACHE INTERNAL "")
set(FASTCV_HAL_HEADERS
  fastcv_hal_core.hpp
  fastcv_hal_imgproc.hpp
  fastcv_hal_utils.hpp
  CACHE INTERNAL "")
set(FASTCV_HAL_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${FASTCV_DIR}/inc
  CACHE INTERNAL "")
