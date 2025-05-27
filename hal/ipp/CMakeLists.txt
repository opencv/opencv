project(ipphal)

set(IPP_HAL_VERSION 0.0.1 CACHE INTERNAL "")
set(IPP_HAL_LIBRARIES "ipphal" CACHE INTERNAL "")
set(IPP_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
set(IPP_HAL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/ipp_hal_core.hpp"
  CACHE INTERNAL "")

add_library(ipphal STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mean_ipp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/minmax_ipp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/norm_ipp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cart_polar_ipp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transforms_ipp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sum_ipp.cpp"
)

#TODO: HAVE_IPP_ICV and HAVE_IPP_IW added as private macro till OpenCV itself is
#      source of IPP and public definitions lead to redefinition warning
#      The macro should be redefined as PUBLIC when IPP part is removed from core
#      to make HAL the source of IPP integration
if(HAVE_IPP_ICV)
  target_compile_definitions(ipphal PRIVATE HAVE_IPP_ICV)
endif()

if(HAVE_IPP_IW)
  target_compile_definitions(ipphal PRIVATE HAVE_IPP_IW)
endif()

target_include_directories(ipphal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-suggest-override)

target_include_directories(ipphal PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${IPP_INCLUDE_DIRS}
)

target_link_libraries(ipphal PUBLIC ${IPP_IW_LIBRARY} ${IPP_LIBRARIES})

set_target_properties(ipphal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(ipphal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(ipphal PROPERTIES FOLDER "3rdparty")
endif()
