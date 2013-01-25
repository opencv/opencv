# Will be modified later
if(NOT HAVE_OPENCL)
  ocv_module_disable(ocl)
endif()

set(the_description "OpenCL-accelerated Computer Vision")
ocv_add_module(ocl opencv_core opencv_imgproc opencv_features2d opencv_objdetect opencv_video)

ocv_module_include_directories()

file(GLOB CL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cl")
set(kernels_cpp "${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp")
set(cl2cpp_script "${CMAKE_CURRENT_SOURCE_DIR}/cl2cpp.cmake")

add_custom_command(
  OUTPUT ${kernels_cpp}
  COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels" -DOUTPUT="${kernels_cpp}" -P ${cl2cpp_script}
  DEPENDS ${CL_FILES} ${cl2cpp_script})

file(GLOB lib_hdrs     "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")
file(GLOB lib_srcs     "src/*.cpp")
file(GLOB lib_int_hdrs "src/*.h*")

source_group("Include"   FILES ${lib_hdrs})
source_group("Src\\Host" FILES ${lib_srcs} ${lib_int_hdrs} ${kernels_cpp})

if (HAVE_OPENCL)
  set(ocl_link_libs ${OPENCL_LIBRARIES})
  if(OPENCL_INCLUDE_DIR)
    ocv_include_directories(${OPENCL_INCLUDE_DIR})
  endif()
  if (HAVE_CLAMDFFT)
    set(ocl_link_libs ${ocl_link_libs} ${CLAMDFFT_LIBRARIES})
    ocv_include_directories(${CLAMDFFT_INCLUDE_DIR})
  endif()
  if (HAVE_CLAMDBLAS)
    set(ocl_link_libs ${ocl_link_libs} ${CLAMDBLAS_LIBRARIES})
    ocv_include_directories(${CLAMDBLAS_INCLUDE_DIR})
  endif()
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)

ocv_set_module_sources(HEADERS ${lib_hdrs} SOURCES ${lib_int_hdrs} ${lib_srcs} ${kernels_cpp})
ocv_create_module(${ocl_link_libs})

install(FILES ${lib_hdrs}
  DESTINATION include/opencv2/${name}
  COMPONENT main)

ocv_add_precompiled_headers(${the_module})

################################################################################################################
################################      OpenCL Module Tests     ##################################################
################################################################################################################
file(GLOB test_srcs "test/*.cpp")
file(GLOB test_hdrs "test/*.hpp" "test/*.h")

ocv_add_accuracy_tests(FILES "Include" ${test_hdrs}
                       FILES "Src" ${test_srcs})

################################################################################################################
################################   OpenCL Module Performance  ##################################################
################################################################################################################
file(GLOB perf_srcs "perf/*.cpp")
file(GLOB perf_hdrs "perf/*.hpp" "perf/*.h")

ocv_add_perf_tests(FILES "Include" ${perf_hdrs}
                   FILES "Src" ${perf_srcs})
