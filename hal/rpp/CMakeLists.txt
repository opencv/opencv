project(rpphal)

set(RPP_HAL_VERSION 0.0.1 CACHE INTERNAL "")
set(RPP_HAL_LIBRARIES "rpphal" CACHE INTERNAL "")
set(RPP_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
set(RPP_HAL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/rpp_hal_core.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/rpp_hal_imgproc.hpp"
  CACHE INTERNAL "")

find_package(rpp REQUIRED)

add_library(rpphal STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rpp_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core_rpp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgproc_rpp.cpp"
)

target_include_directories(rpphal PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include
)

target_link_libraries(rpphal PUBLIC rpp::rpp /opt/rocm-7.2.1/lib/libamdhip64.so)

# Define GPU_SUPPORT so RPP exposes GPU APIs.
# Define __HIP_PLATFORM_AMD__ so HIP headers work in plain C++ mode.
target_compile_definitions(rpphal PUBLIC GPU_SUPPORT=1 __HIP_PLATFORM_AMD__=1 RPP_BACKEND_HIP=1)

set_target_properties(rpphal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH} DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}")

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(rpphal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
