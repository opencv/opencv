project(ACCELERATE_HAL)

set(ACCELERATE_HAL_VERSION 0.0.1 CACHE INTERNAL "")
set(ACCELERATE_HAL_LIBRARIES "accelerate_hal" CACHE INTERNAL "")
set(ACCELERATE_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
set(ACCELERATE_HAL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/accelerate_hal_core.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/accelerate_hal_imgproc.hpp"
  CACHE INTERNAL "")

add_library(accelerate_hal STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/accelerate_hal_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/accelerate_hal_imgproc.cpp"
)

target_link_libraries(accelerate_hal PRIVATE "-framework Accelerate")

target_include_directories(accelerate_hal PRIVATE
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${ACCELERATE_HAL_INCLUDE_DIRS})

set_target_properties(accelerate_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(accelerate_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(accelerate_hal PROPERTIES FOLDER "3rdparty")
endif()
