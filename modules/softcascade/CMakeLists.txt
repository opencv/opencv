macro(ocv_glob_cuda_powered_module_sources)
  file(GLOB_RECURSE lib_srcs "src/*.cpp")
  file(GLOB_RECURSE lib_int_hdrs "src/*.hpp" "src/*.h")
  file(GLOB lib_hdrs "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")
  file(GLOB lib_hdrs_detail "include/opencv2/${name}/detail/*.hpp" "include/opencv2/${name}/detail/*.h")

  file(GLOB_RECURSE lib_device_srcs "src/*.cu")
  set(device_objs "")
  set(lib_device_hdrs "")

  if (HAVE_CUDA AND lib_device_srcs)
  ocv_include_directories(${CUDA_INCLUDE_DIRS} "${OpenCV_SOURCE_DIR}/modules/gpu/include")
  file(GLOB_RECURSE lib_device_hdrs "src/cuda/*.hpp")

  ocv_cuda_compile(device_objs ${lib_device_srcs})
  source_group("Src\\Cuda" FILES ${lib_device_srcs} ${lib_device_hdrs})
  if (lib_device_hdrs)
  list(REMOVE_ITEM lib_int_hdrs ${lib_device_hdrs})
  endif()
  endif()

  ocv_set_module_sources(${ARGN} HEADERS ${lib_hdrs} ${lib_hdrs_detail}
  SOURCES ${lib_srcs} ${lib_int_hdrs} ${device_objs} ${lib_device_srcs} ${lib_device_hdrs})

  source_group("Src" FILES ${lib_srcs} ${lib_int_hdrs})
  source_group("Include" FILES ${lib_hdrs})
  source_group("Include\\detail" FILES ${lib_hdrs_detail})
endmacro()

set(the_description "Soft Cascade detection and training")
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4310 -Wundef -Wmissing-declarations)
set(cuda_deps "")
set(cuda_include "")

if (NAVE_CUDA)
  set(cuda_deps ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
endif()
ocv_add_module(softcascade opencv_core opencv_imgproc opencv_ml OPTIONAL ${cuda_deps})

if(HAVE_CUDA)
  ocv_module_include_directories(${CUDA_INCLUDE_DIRS})
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

ocv_glob_cuda_powered_module_sources()
ocv_create_module()
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
