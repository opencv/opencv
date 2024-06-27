add_library(fastcv_hal STATIC fastcv_hal.cpp fastcv_hal.hpp)
target_include_directories(fastcv_hal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include
  fastcv_hal PRIVATE ${FASTCV_DIR}/inc)

if(ARM)
  set(FASTCV_LIB "lib/Android/lib32/libfastcv.a")
elseif(AARCH64)
  set(FASTCV_LIB "lib/Android/lib64/libfastcv.a")
else()
  message(FATAL_ERROR "FastCV supports ARM platform only!")
endif()

target_link_libraries(fastcv_hal PUBLIC ${FASTCV_DIR}/${FASTCV_LIB})

set_target_properties(fastcv_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(fastcv_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

set(FASTCV_HAL_FOUND TRUE CACHE INTERNAL "")
set(FASTCV_HAL_VERSION 0.0.1 CACHE INTERNAL "")
set(FASTCV_HAL_LIBRARIES "fastcv_hal" CACHE INTERNAL "")
set(FASTCV_HAL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/fastcv_hal.hpp" CACHE INTERNAL "")
set(FASTCV_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
