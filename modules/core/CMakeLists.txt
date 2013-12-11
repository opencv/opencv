set(the_description "The Core Functionality")

if(HAVE_WINRT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW /GS /Gm- /AI\"${WINDOWS_SDK_PATH}/References/CommonConfiguration/Neutral\" /AI\"${VISUAL_STUDIO_PATH}/vcpackages\"")
endif()

file(GLOB lib_cuda_hdrs        "include/opencv2/${name}/cuda/*.hpp"        "include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail "include/opencv2/${name}/cuda/detail/*.hpp" "include/opencv2/${name}/cuda/detail/*.h")

source_group("Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

if(DYNAMIC_CUDA_SUPPORT)
  add_definitions(-DDYNAMIC_CUDA_SUPPORT)
endif()

ocv_add_module(core PRIVATE_REQUIRED ${ZLIB_LIBRARIES})
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

if(HAVE_CUDA)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include")
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

ocv_glob_module_sources(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc"
                        HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})

ocv_create_module()
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()

if(DYNAMIC_CUDA_SUPPORT)
  add_subdirectory(cuda)
endif()
