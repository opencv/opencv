set(the_description "The Core Functionality")

macro(ocv_glob_module_sources_no_cuda)
  file(GLOB_RECURSE lib_srcs "src/*.cpp")
  file(GLOB_RECURSE lib_int_hdrs "src/*.hpp" "src/*.h")
  file(GLOB lib_hdrs "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")
  file(GLOB lib_hdrs_detail "include/opencv2/${name}/detail/*.hpp" "include/opencv2/${name}/detail/*.h")

  set(cuda_objs "")
  set(lib_cuda_hdrs "")
  if(HAVE_CUDA)
    ocv_include_directories(${CUDA_INCLUDE_DIRS})
    file(GLOB lib_cuda_hdrs "src/cuda/*.hpp")
  endif()

  source_group("Src" FILES ${lib_srcs} ${lib_int_hdrs})

  file(GLOB cl_kernels "src/opencl/*.cl")
  if(HAVE_opencv_ocl AND cl_kernels)
    ocv_include_directories(${OPENCL_INCLUDE_DIRS})
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/opencl_kernels.cpp" "${CMAKE_CURRENT_BINARY_DIR}/opencl_kernels.hpp"
      COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/opencl" -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/opencl_kernels.cpp" -P "${OpenCV_SOURCE_DIR}/cmake/cl2cpp.cmake"
      DEPENDS ${cl_kernels} "${OpenCV_SOURCE_DIR}/cmake/cl2cpp.cmake")
    source_group("OpenCL" FILES ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/opencl_kernels.cpp" "${CMAKE_CURRENT_BINARY_DIR}/opencl_kernels.hpp")
    list(APPEND lib_srcs ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/opencl_kernels.cpp" "${CMAKE_CURRENT_BINARY_DIR}/opencl_kernels.hpp")
  endif()

  source_group("Include" FILES ${lib_hdrs})
  source_group("Include\\detail" FILES ${lib_hdrs_detail})

  ocv_set_module_sources(${ARGN} HEADERS ${lib_hdrs} ${lib_hdrs_detail}
                                 SOURCES ${lib_srcs} ${lib_int_hdrs} ${cuda_objs} ${lib_cuda_hdrs})
endmacro()

ocv_add_module(core PRIVATE_REQUIRED ${ZLIB_LIBRARIES})
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

if(HAVE_WINRT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW /GS /Gm- /AI\"${WINDOWS_SDK_PATH}/References/CommonConfiguration/Neutral\" /AI\"${VISUAL_STUDIO_PATH}/vcpackages\"")
endif()

if(DYNAMIC_CUDA_SUPPORT)
  add_definitions(-DDYNAMIC_CUDA_SUPPORT)
else()
  add_definitions(-DUSE_CUDA)
endif()

if(HAVE_CUDA)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include")
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

file(GLOB lib_cuda_hdrs        "include/opencv2/${name}/cuda/*.hpp"        "include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail "include/opencv2/${name}/cuda/detail/*.hpp" "include/opencv2/${name}/cuda/detail/*.h")

source_group("Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

if (DYNAMIC_CUDA_SUPPORT)
  ocv_glob_module_sources_no_cuda(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc"
                                  HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})
else()
  ocv_glob_module_sources(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc"
                          HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})
endif()

ocv_create_module()
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()

if (DYNAMIC_CUDA_SUPPORT)
  add_subdirectory(cuda)
endif()
