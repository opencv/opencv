# Install script for directory: /Users/chihiro/Programs/opencv/opencv/modules/core

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY OPTIONAL FILES
    "/Users/chihiro/Programs/opencv/opencv/release/lib/libopencv_core.3.2.0.dylib"
    "/Users/chihiro/Programs/opencv/opencv/release/lib/libopencv_core.3.2.dylib"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.3.2.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.3.2.dylib"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      execute_process(COMMAND /usr/bin/install_name_tool
        -add_rpath "/usr/local/lib"
        "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/chihiro/Programs/opencv/opencv/release/lib/libopencv_core.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "/usr/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.dylib")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/block.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/border_interpolate.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/color.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/common.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/datamov_utils.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/dynamic_smem.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/emulation.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/filters.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/funcattrib.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/functional.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/limits.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/reduce.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/saturate_cast.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/scan.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/simd_functions.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/transform.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/type_traits.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/utility.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/vec_distance.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/vec_math.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/vec_traits.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/warp.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/warp_reduce.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/warp_shuffle.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/detail/color_detail.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/detail/reduce.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/detail/reduce_key_val.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/detail/transform_detail.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/detail/type_traits_detail.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda/detail/vec_distance_detail.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/affine.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/base.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/bufferpool.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/core.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda.inl.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda_stream_accessor.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cuda_types.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cvstd.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cvstd.inl.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/directx.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/eigen.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/fast_math.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/ippasync.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/mat.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/mat.inl.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/matx.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/neon_utils.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/ocl.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/ocl_genbase.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/opengl.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/operations.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/optim.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/ovx.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/persistence.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/private.cuda.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/private.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/ptr.inl.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/saturate.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/sse_utils.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/traits.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/types.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/utility.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/va_intel.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/version.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/wimage.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/core_c.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/cvdef.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/types_c.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/hal/hal.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/hal/intrin.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/hal/intrin_cpp.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/hal/intrin_neon.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/hal/intrin_sse.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/Users/chihiro/Programs/opencv/opencv/modules/core/include/opencv2/core/hal/interface.h")
endif()

