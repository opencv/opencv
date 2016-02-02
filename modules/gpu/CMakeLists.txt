if(IOS)
  ocv_module_disable(gpu)
endif()

set(the_description "GPU-accelerated Computer Vision")
ocv_add_module(gpu opencv_imgproc opencv_calib3d opencv_objdetect opencv_video opencv_photo opencv_legacy
               OPTIONAL ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cufft_LIBRARY})

ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")

file(GLOB lib_hdrs               "include/opencv2/${name}/*.hpp"               "include/opencv2/${name}/*.h")
file(GLOB lib_device_hdrs        "include/opencv2/${name}/device/*.hpp"        "include/opencv2/${name}/device/*.h")
file(GLOB lib_device_hdrs_detail "include/opencv2/${name}/device/detail/*.hpp" "include/opencv2/${name}/device/detail/*.h")
file(GLOB lib_int_hdrs           "src/*.hpp"      "src/*.h")
file(GLOB lib_cuda_hdrs          "src/cuda/*.hpp" "src/cuda/*.h")
file(GLOB lib_srcs               "src/*.cpp")
file(GLOB lib_cuda               "src/cuda/*.cu*")

source_group("Include"        FILES ${lib_hdrs})
source_group("Src\\Host"      FILES ${lib_srcs} ${lib_int_hdrs})
source_group("Src\\Cuda"      FILES ${lib_cuda} ${lib_cuda_hdrs})
source_group("Device"         FILES ${lib_device_hdrs})
source_group("Device\\Detail" FILES ${lib_device_hdrs_detail})

if(HAVE_CUDA)
  file(GLOB_RECURSE ncv_srcs "src/nvidia/*.cpp" "src/nvidia/*.h*")
  file(GLOB_RECURSE ncv_cuda "src/nvidia/*.cu")
  set(ncv_files ${ncv_srcs} ${ncv_cuda})

  source_group("Src\\NVidia" FILES ${ncv_files})
  ocv_include_directories("src/nvidia" "src/nvidia/core" "src/nvidia/NPP_staging" ${CUDA_INCLUDE_DIRS})
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

  if(HAVE_CUFFT)
      set(cuda_link_libs ${cuda_link_libs} ${CUDA_cufft_LIBRARY})
  endif()

  if(HAVE_CUBLAS)
      set(cuda_link_libs ${cuda_link_libs} ${CUDA_cublas_LIBRARY})
  endif()

  if(WITH_NVCUVID)
    if(HAVE_NVCUVID)
      set(cuda_link_libs ${cuda_link_libs} ${CUDA_CUDA_LIBRARY} ${CUDA_nvcuvid_LIBRARY})
    endif()

    if(WIN32)
      if(HAVE_NVCUVENC)
        set(cuda_link_libs ${cuda_link_libs} ${CUDA_nvcuvenc_LIBRARY})
      endif()
    endif()

    if(WITH_FFMPEG)
      set(cuda_link_libs ${cuda_link_libs} ${HIGHGUI_LIBRARIES})
    endif()
  endif()
else()
  set(lib_cuda "")
  set(cuda_objs "")
  set(cuda_link_libs "")
  set(ncv_files "")
endif()

ocv_set_module_sources(
  HEADERS ${lib_hdrs} ${lib_device_hdrs} ${lib_device_hdrs_detail}
  SOURCES ${lib_int_hdrs} ${lib_cuda_hdrs} ${lib_srcs} ${lib_cuda} ${ncv_files} ${cuda_objs}
  )

ocv_create_module(${cuda_link_libs})

if(HAVE_CUDA)
  install(FILES src/nvidia/NPP_staging/NPP_staging.hpp  src/nvidia/core/NCV.hpp
    DESTINATION ${OPENCV_INCLUDE_INSTALL_PATH}/opencv2/${name}
    COMPONENT dev)
endif()

ocv_add_precompiled_headers(${the_module})

################################################################################################################
################################      GPU Module Tests     #####################################################
################################################################################################################
file(GLOB test_srcs "test/*.cpp")
file(GLOB test_hdrs "test/*.hpp" "test/*.h")

set(nvidia "")
if(HAVE_CUDA)
  file(GLOB nvidia "test/nvidia/*.cpp" "test/nvidia/*.hpp" "test/nvidia/*.h")
  set(nvidia FILES "Src\\\\\\\\NVidia" ${nvidia}) # 8 ugly backslashes :'(
endif()

ocv_add_accuracy_tests(FILES "Include" ${test_hdrs}
                       FILES "Src" ${test_srcs}
                       ${nvidia})
ocv_add_perf_tests()

if(HAVE_CUDA)
  add_subdirectory(perf4au)
endif()
