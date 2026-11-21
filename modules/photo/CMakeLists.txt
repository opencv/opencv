set(the_description "Computational Photography")
ocv_add_dispatched_file(mertens_weight AVX512_ICL)
ocv_add_dispatched_file(mertens_fuse AVX512_ICL)

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations -Wshadow)
endif()

ocv_define_module(photo opencv_imgproc OPTIONAL opencv_cudaarithm opencv_cudaimgproc WRAP java objc python js)

if(HAVE_CUDA AND ENABLE_CUDA_FIRST_CLASS_LANGUAGE AND HAVE_opencv_cudaarithm AND HAVE_opencv_cudaimgproc)
  ocv_target_link_libraries(${the_module} PRIVATE CUDA::cudart${CUDA_LIB_EXT})
endif()
