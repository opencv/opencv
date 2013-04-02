if(BUILD_ANDROID_PACKAGE)
  ocv_module_disable(nonfree)
endif()

set(the_description "Functionality with possible limitations on the use")
ocv_add_module(nonfree opencv_imgproc opencv_features2d opencv_calib3d OPTIONAL opencv_gpu opencv_ocl)
ocv_module_include_directories()

if(HAVE_CUDA AND HAVE_opencv_gpu)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wsign-promo)
  ocv_source_group("Src\\Cuda" GLOB "src/cuda/*.cu")
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include" ${CUDA_INCLUDE_DIRS})
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)

  file(GLOB lib_cuda "src/cuda/*.cu")
  ocv_cuda_compile(cuda_objs ${lib_cuda})

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
