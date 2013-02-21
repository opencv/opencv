set(the_description "Super Resolution")
ocv_add_module(superres opencv_imgproc opencv_video OPTIONAL opencv_gpu opencv_highgui)
ocv_module_include_directories()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef /wd4127)

if(HAVE_CUDA)
  string(REPLACE "-Wsign-promo" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  ocv_source_group("Src\\Cuda" GLOB "src/cuda/*.cu")
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include" ${CUDA_INCLUDE_DIRS})

  file(GLOB lib_cuda "src/cuda/*.cu")
  ocv_cuda_compile(cuda_objs ${lib_cuda})

  set(cuda_link_libs ${CUDA_LIBRARIES})
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
