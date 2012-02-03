if(ANDROID OR IOS)
  ocv_module_disable(gpu)
endif()

set(the_description "GPU-accelerated Computer Vision")
ocv_add_module(gpu opencv_imgproc opencv_calib3d opencv_objdetect)

ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")

file(GLOB lib_hdrs               "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")
file(GLOB lib_int_hdrs           "src/*.hpp" "src/*.h")
file(GLOB lib_cuda_hdrs          "src/cuda/*.hpp" "src/cuda/*.h")
file(GLOB lib_device_hdrs        "src/opencv2/gpu/device/*.hpp" "src/opencv2/gpu/device/*.h")
file(GLOB lib_device_hdrs_detail "src/opencv2/gpu/device/detail/*.hpp" "src/opencv2/gpu/device/detail/*.h")
file(GLOB lib_srcs               "src/*.cpp")
file(GLOB lib_cuda               "src/cuda/*.cu*")

source_group("Include" FILES ${lib_hdrs})
source_group("Src\\Host" FILES ${lib_srcs} ${lib_int_hdrs})
source_group("Src\\Cuda" FILES ${lib_cuda} ${lib_cuda_hdrs})
source_group("Device" FILES ${lib_device_hdrs})
source_group("Device\\Detail" FILES ${lib_device_hdrs_detail})

if (HAVE_CUDA)
  file(GLOB_RECURSE ncv_srcs "src/nvidia/*.cpp")	
  file(GLOB_RECURSE ncv_cuda "src/nvidia/*.cu")
  file(GLOB_RECURSE ncv_hdrs "src/nvidia/*.hpp" "src/nvidia/*.h")
  set(ncv_files ${ncv_srcs} ${ncv_hdrs} ${ncv_cuda})

  source_group("Src\\NVidia" FILES ${ncv_files})
  include_directories("src/nvidia/core" "src/nvidia/NPP_staging" ${CUDA_INCLUDE_DIRS})

  #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-keep")
  #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;/EHsc-;")
    
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    string(REPLACE "/W4" "/W3" ${var} "${${var}}")
  endforeach()

  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4211 /wd4201 /wd4100 /wd4505 /wd4408 /wd4251")

    foreach(var CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      string(REPLACE "/EHsc-" "/EHs" ${var} "${${var}}")
    endforeach()
    
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler /wd4251)
  endif()

  OCV_CUDA_COMPILE(cuda_objs ${lib_cuda} ${ncv_cuda})
  #CUDA_BUILD_CLEAN_TARGET()
  
  unset(CUDA_npp_LIBRARY CACHE)
  find_cuda_helper_libs(npp)
  set(cuda_link_libs ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
else()
  set(lib_cuda "")
  set(cuda_objs "")
  set(cuda_link_libs "")
  set(ncv_files "")
endif()

ocv_set_module_sources(
  HEADERS ${lib_hdrs}
  SOURCES ${lib_int_hdrs} ${lib_cuda_hdrs} ${lib_device_hdrs} ${lib_device_hdrs_detail} ${lib_srcs} ${lib_cuda} ${ncv_files} ${cuda_objs}
  )
  
ocv_create_module(${cuda_link_libs})
  
if(HAVE_CUDA)
  if(HAVE_CUFFT)
    CUDA_ADD_CUFFT_TO_TARGET(${the_module})
  endif()

  if(HAVE_CUBLAS)
    CUDA_ADD_CUBLAS_TO_TARGET(${the_module})
  endif()
  
  install(FILES src/nvidia/NPP_staging/NPP_staging.hpp  src/nvidia/core/NCV.hpp
	DESTINATION ${OPENCV_INCLUDE_PREFIX}/opencv2/${name}
	COMPONENT main)
endif()

ocv_add_precompiled_headers(${the_module})

################################################################################################################
################################      GPU Module Tests     #####################################################
################################################################################################################
file(GLOB test_srcs "test/*.cpp")
file(GLOB test_hdrs "test/*.hpp" "test/*.h")
if(HAVE_CUDA)
  file(GLOB nvidia "test/nvidia/*.cpp" "test/nvidia/*.hpp" "test/nvidia/*.h")            
  set(nvidia FILES "Src\\\\\\\\NVidia" ${nvidia}) # 8 ugly backslashes :'(
else()
  set(nvidia "")
endif()

ocv_add_accuracy_tests(FILES "Include" ${test_hdrs}
                       FILES "Src" ${test_srcs}
                       ${nvidia})
ocv_add_perf_tests()
