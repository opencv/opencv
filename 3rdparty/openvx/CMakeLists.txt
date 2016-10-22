add_library(openvx_hal STATIC include/openvx_hal.hpp src/openvx_hal.cpp)
target_include_directories(openvx_hal PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include
  ${OPENVX_INCLUDE_DIR})
target_link_libraries(openvx_hal LINK_PUBLIC ${OPENVX_LIBRARIES})
set_target_properties(openvx_hal PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(openvx_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})
if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(openvx_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

set(OPENVX_HAL_FOUND TRUE PARENT_SCOPE)
set(OPENVX_HAL_VERSION 0.0.1 PARENT_SCOPE)
set(OPENVX_HAL_LIBRARIES "openvx_hal" PARENT_SCOPE)
set(OPENVX_HAL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/openvx_hal.hpp" PARENT_SCOPE)
set(OPENVX_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" "${OPENVX_INCLUDE_DIR}" PARENT_SCOPE)
