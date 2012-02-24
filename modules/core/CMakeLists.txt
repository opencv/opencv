set(the_description "The Core Functionality")
ocv_add_module(core ${ZLIB_LIBRARIES})
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

if(HAVE_CUDA)
  file(GLOB lib_cuda "src/cuda/*.cu")
  source_group("Cuda" FILES "${lib_cuda}")
  
  include_directories(${CUDA_INCLUDE_DIRS} "${OpenCV_SOURCE_DIR}/modules/gpu/src" "${OpenCV_SOURCE_DIR}/modules/gpu/src/cuda")  
  OCV_CUDA_COMPILE(cuda_objs ${lib_cuda})
  
  set(cuda_link_libs ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
else()
  set(lib_cuda "")
  set(cuda_objs "")
  set(cuda_link_libs "")
endif()

ocv_glob_module_sources(SOURCES ${lib_cuda} ${cuda_objs})

ocv_create_module(${cuda_link_libs})
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()




