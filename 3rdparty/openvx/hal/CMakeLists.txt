add_library(openvx_hal STATIC openvx_hal.cpp openvx_hal.hpp ${OPENCV_3P_OPENVX_DIR}/include/ivx.hpp ${OPENCV_3P_OPENVX_DIR}/include/ivx_lib_debug.hpp)
target_include_directories(openvx_hal PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OPENCV_3P_OPENVX_DIR}/include
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include
  ${CMAKE_SOURCE_DIR}/modules/features2d/include
  ${OPENVX_INCLUDE_DIR})
target_link_libraries(openvx_hal PUBLIC ${OPENVX_LIBRARIES})
set_target_properties(openvx_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})
if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(openvx_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

set(OPENVX_HAL_FOUND TRUE CACHE INTERNAL "")
set(OPENVX_HAL_VERSION 0.0.1 CACHE INTERNAL "")
set(OPENVX_HAL_LIBRARIES "openvx_hal" CACHE INTERNAL "")
set(OPENVX_HAL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/openvx_hal.hpp" CACHE INTERNAL "")
set(OPENVX_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" "${OPENCV_3P_OPENVX_DIR}/include" "${OPENVX_INCLUDE_DIR}" CACHE INTERNAL "")
