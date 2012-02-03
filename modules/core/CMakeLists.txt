set(the_description "The Core Functionality")
ocv_add_module(core ${ZLIB_LIBRARY})

if(HAVE_CUDA)
  file(GLOB lib_cuda "src/cuda/*.cu")
  source_group("Cuda" FILES "${lib_cuda}")
  
  include_directories(${CUDA_INCLUDE_DIRS} "${OpenCV_SOURCE_DIR}/modules/gpu/src" "${OpenCV_SOURCE_DIR}/modules/gpu/src/cuda")

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_10,code=sm_10
                                         -gencode arch=compute_11,code=sm_11
                                         -gencode arch=compute_12,code=sm_12
                                         -gencode arch=compute_13,code=sm_13
                                         -gencode arch=compute_20,code=sm_20
                                         -gencode arch=compute_20,code=sm_21)

  OCV_CUDA_COMPILE(cuda_objs ${lib_cuda})

  unset(CUDA_npp_LIBRARY CACHE)
  find_cuda_helper_libs(npp)
  set(cuda_link_libs ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
else()
  set(lib_cuda "")
  set(cuda_objs "")
  set(cuda_link_libs "")
endif()

ocv_glob_module_sources(SOURCES ${lib_cuda} ${cuda_objs})

if(NOT ZLIB_FOUND)
    set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/zlib")
endif()
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

ocv_create_module(${cuda_link_libs})
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()




