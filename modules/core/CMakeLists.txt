set(the_description "The Core Functionality")

if (NOT HAVE_CUDA OR ENABLE_DYNAMIC_CUDA)
  ocv_add_module(core PRIVATE_REQUIRED ${ZLIB_LIBRARIES})
else()
  ocv_add_module(core PRIVATE_REQUIRED ${ZLIB_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
endif()

ocv_module_include_directories("${OpenCV_SOURCE_DIR}/modules/dynamicuda/include/" ${ZLIB_INCLUDE_DIR})

if(HAVE_WINRT_CX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()
if(HAVE_WINRT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GS /Gm- /AI\"${WINDOWS_SDK_PATH}/References/CommonConfiguration/Neutral\" /AI\"${VISUAL_STUDIO_PATH}/vcpackages\"")
endif()

if(ENABLE_DYNAMIC_CUDA)
  add_definitions(-DDYNAMIC_CUDA_SUPPORT)
else()
  if (HAVE_CUDA)
    add_definitions(-DUSE_CUDA)
  endif()
endif()

if(HAVE_CUDA)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include")
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

file(GLOB lib_cuda_hdrs        "include/opencv2/${name}/cuda/*.hpp"        "include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail "include/opencv2/${name}/cuda/detail/*.hpp" "include/opencv2/${name}/cuda/detail/*.h")

if(HAVE_CUDA AND NOT ENABLE_DYNAMIC_CUDA)
  file(GLOB lib_cuda           "../dynamicuda/src/cuda/*.cu*")
  ocv_include_directories(${CUDA_INCLUDE_DIRS})
  ocv_cuda_compile(cuda_objs ${lib_cuda})
endif()

source_group("Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

if (HAVE_CUDA AND NOT ENABLE_DYNAMIC_CUDA)
  source_group("Src\\Cuda"      FILES ${lib_cuda} ${lib_cuda_hdrs})
endif()

if (NOT HAVE_CUDA OR ENABLE_DYNAMIC_CUDA)
  ocv_glob_module_sources(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc"
                          HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})
else()
  ocv_glob_module_sources(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc" ${lib_cuda} ${cuda_objs}
                          HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})
endif()

ocv_create_module()
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
