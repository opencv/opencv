if(ANDROID OR IOS)
  ocv_module_disable(gpu)
endif()

set(the_description "GPU-accelerated Computer Vision")

ocv_add_module(gpu opencv_imgproc opencv_calib3d opencv_objdetect opencv_video opencv_photo opencv_legacy opencv_gpuarithm opencv_gpufilters opencv_gpuimgproc opencv_gpufeatures2d OPTIONAL opencv_gpunvidia)

ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")

file(GLOB lib_hdrs               "include/opencv2/*.hpp"                       "include/opencv2/${name}/*.hpp"               "include/opencv2/${name}/*.h")
file(GLOB lib_int_hdrs           "src/*.hpp"      "src/*.h")
file(GLOB lib_cuda_hdrs          "src/cuda/*.hpp" "src/cuda/*.h")
file(GLOB lib_srcs               "src/*.cpp")
file(GLOB lib_cuda               "src/cuda/*.cu*")

source_group("Include"        FILES ${lib_hdrs})
source_group("Src\\Host"      FILES ${lib_srcs} ${lib_int_hdrs})
source_group("Src\\Cuda"      FILES ${lib_cuda} ${lib_cuda_hdrs})

if(HAVE_CUDA)
  ocv_include_directories(${CUDA_INCLUDE_DIRS})
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter /wd4211 /wd4201 /wd4100 /wd4505 /wd4408)

  if(MSVC)
    if(NOT ENABLE_NOISY_WARNINGS)
      foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
        string(REPLACE "/W4" "/W3" ${var} "${${var}}")
      endforeach()

      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler /wd4251)
    endif()
  endif()

  ocv_cuda_compile(cuda_objs ${lib_cuda} ${ncv_cuda})

  set(cuda_link_libs ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
else()
  set(lib_cuda "")
  set(cuda_objs "")
  set(cuda_link_libs "")
endif()

ocv_set_module_sources(
  HEADERS ${lib_hdrs}
  SOURCES ${lib_int_hdrs} ${lib_cuda_hdrs} ${lib_srcs} ${lib_cuda} ${cuda_objs}
  )

ocv_create_module(${cuda_link_libs})

ocv_add_precompiled_headers(${the_module})

################################################################################################################
################################      GPU Module Tests     #####################################################
################################################################################################################
file(GLOB test_srcs "test/*.cpp")
file(GLOB test_hdrs "test/*.hpp" "test/*.h")

ocv_add_accuracy_tests(FILES "Include" ${test_hdrs}
                       FILES "Src" ${test_srcs})
ocv_add_perf_tests()

if(HAVE_CUDA)
  add_subdirectory(perf4au)
endif()
