set(the_description "The Core Functionality")
ocv_add_module(core ${ZLIB_LIBRARIES})
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

if(HAVE_CUDA)
  file(GLOB lib_cuda "src/cuda/*.cu")
  source_group("Cuda" FILES "${lib_cuda}")
  
  ocv_include_directories(${CUDA_INCLUDE_DIRS} "${OpenCV_SOURCE_DIR}/modules/gpu/src" "${OpenCV_SOURCE_DIR}/modules/gpu/src/cuda")  
  OCV_CUDA_COMPILE(cuda_objs ${lib_cuda})
  
  set(cuda_link_libs ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
else()
  set(lib_cuda "")
  set(cuda_objs "")
  set(cuda_link_libs "")
endif()

set(OPENCV_VERSION_FILE "${opencv_core_BINARY_DIR}/version_string.inc")
add_custom_command(OUTPUT "${OPENCV_VERSION_FILE}"
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_BUILD_INFO_FILE}" "${OPENCV_VERSION_FILE}"
                   MAIN_DEPENDENCY "${OPENCV_BUILD_INFO_FILE}"
                   COMMENT "")

ocv_glob_module_sources(SOURCES ${lib_cuda} ${cuda_objs} "${OPENCV_VERSION_FILE}")

ocv_create_module(${cuda_link_libs})
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()

