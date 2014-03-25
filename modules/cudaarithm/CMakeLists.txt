if(IOS)
  ocv_module_disable(cudaarithm)
endif()

set(the_description "CUDA-accelerated Operations on Matrices")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_add_module(cudaarithm opencv_core OPTIONAL opencv_cudev)

ocv_module_include_directories()
ocv_glob_module_sources()

set(extra_libs "")

if(HAVE_CUBLAS)
  list(APPEND extra_libs ${CUDA_cublas_LIBRARY})
endif()

if(HAVE_CUFFT)
  list(APPEND extra_libs ${CUDA_cufft_LIBRARY})
endif()

ocv_create_module(${extra_libs})

ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests(DEPENDS_ON opencv_imgproc)
ocv_add_perf_tests(DEPENDS_ON opencv_imgproc)
