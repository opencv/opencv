if(HAVE_ARMPL)
  set(ARMPL_HAL_VERSION 0.0.1 CACHE INTERNAL "")
  set(ARMPL_HAL_LIBRARIES "armpl_hal" CACHE INTERNAL "")
  set(ARMPL_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
  set(ARMPL_HAL_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/armpl_hal_core.hpp"
    CACHE INTERNAL "")
  file(GLOB ARMPL_HAL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
  add_library(armpl_hal STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${ARMPL_HAL_FILES})
  target_include_directories(armpl_hal PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/core/include
    ${CMAKE_BINARY_DIR}
    ${ARMPL_HAL_INCLUDE_DIRS}
    ${ARMPL_INCLUDE_PATH})
  target_link_libraries(armpl_hal PUBLIC ${ARMPL_LIBRARY})
  set_target_properties(armpl_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})
  if(NOT BUILD_SHARED_LIBS)
    ocv_install_target(armpl_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
  endif()
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(armpl_hal PROPERTIES FOLDER "3rdparty")
  endif()
else()
  message(STATUS "ArmPL is not available, disabling related HAL")
endif(HAVE_ARMPL)
