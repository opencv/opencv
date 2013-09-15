if(ANDROID OR IOS OR APPLE)
  ocv_module_disable(gpucodec)
endif()

set(the_description "GPU-accelerated Video Encoding/Decoding")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_add_module(gpucodec opencv_highgui)

ocv_module_include_directories()
ocv_glob_module_sources()

set(extra_libs "")

if(HAVE_NVCUVID)
  list(APPEND extra_libs ${CUDA_CUDA_LIBRARY} ${CUDA_nvcuvid_LIBRARY})

  if(WIN32)
    list(APPEND extra_libs ${CUDA_nvcuvenc_LIBRARY})
  endif()
endif()

ocv_create_module(${extra_libs})

ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
