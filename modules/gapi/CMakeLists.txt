# FIXME: Rework standalone build in more generic maner
# (Restructure directories, add common pass, etc)
if(NOT DEFINED OPENCV_INITIAL_PASS)
    cmake_minimum_required(VERSION 3.3)
    project(gapi_standalone)
    include("cmake/standalone.cmake")
    return()
endif()

if(NOT TARGET ade)
  # can't build G-API because of the above reasons
  ocv_module_disable(gapi)
  return()
endif()

set(the_description "OpenCV G-API Core Module")

ocv_add_module(gapi opencv_imgproc)

if(MSVC)
  # Disable obsollete warning C4503 popping up on MSVC <<2017
  # https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4503?view=vs-2019
  ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4503)
endif()

file(GLOB gapi_ext_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cpu/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/fluid/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/gpu/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/infer/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/ocl/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/own/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/render/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/streaming/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/plaidml/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/util/*.hpp"
    )

set(gapi_srcs
    # Front-end part
    src/api/gorigin.cpp
    src/api/gmat.cpp
    src/api/garray.cpp
    src/api/gopaque.cpp
    src/api/gscalar.cpp
    src/api/gkernel.cpp
    src/api/gbackend.cpp
    src/api/gproto.cpp
    src/api/gnode.cpp
    src/api/gcall.cpp
    src/api/gcomputation.cpp
    src/api/operators.cpp
    src/api/kernels_core.cpp
    src/api/kernels_imgproc.cpp
    src/api/render.cpp
    src/api/render_ocv.cpp
    src/api/ginfer.cpp
    src/api/ft_render.cpp

    # Compiler part
    src/compiler/gmodel.cpp
    src/compiler/gmodelbuilder.cpp
    src/compiler/gislandmodel.cpp
    src/compiler/gcompiler.cpp
    src/compiler/gcompiled.cpp
    src/compiler/gstreaming.cpp
    src/compiler/passes/helpers.cpp
    src/compiler/passes/dump_dot.cpp
    src/compiler/passes/islands.cpp
    src/compiler/passes/meta.cpp
    src/compiler/passes/kernels.cpp
    src/compiler/passes/exec.cpp
    src/compiler/passes/transformations.cpp
    src/compiler/passes/pattern_matching.cpp
    src/compiler/passes/perform_substitution.cpp
    src/compiler/passes/streaming.cpp

    # Executor
    src/executor/gexecutor.cpp
    src/executor/gstreamingexecutor.cpp
    src/executor/gasync.cpp

    # CPU Backend (currently built-in)
    src/backends/cpu/gcpubackend.cpp
    src/backends/cpu/gcpukernel.cpp
    src/backends/cpu/gcpuimgproc.cpp
    src/backends/cpu/gcpucore.cpp

    # Fluid Backend (also built-in, FIXME:move away)
    src/backends/fluid/gfluidbuffer.cpp
    src/backends/fluid/gfluidbackend.cpp
    src/backends/fluid/gfluidimgproc.cpp
    src/backends/fluid/gfluidimgproc_func.dispatch.cpp
    src/backends/fluid/gfluidcore.cpp

    # OCL Backend (currently built-in)
    src/backends/ocl/goclbackend.cpp
    src/backends/ocl/goclkernel.cpp
    src/backends/ocl/goclimgproc.cpp
    src/backends/ocl/goclcore.cpp

    # IE Backend. FIXME: should be included by CMake
    # if and only if IE support is enabled
    src/backends/ie/giebackend.cpp

    # Render Backend.
    src/backends/render/grenderocvbackend.cpp
    src/backends/render/grenderocv.cpp

    #PlaidML Backend
    src/backends/plaidml/gplaidmlcore.cpp
    src/backends/plaidml/gplaidmlbackend.cpp

    # Compound
    src/backends/common/gcompoundbackend.cpp
    src/backends/common/gcompoundkernel.cpp
    )

ocv_add_dispatched_file(backends/fluid/gfluidimgproc_func SSE4_1 AVX2)

ocv_list_add_prefix(gapi_srcs "${CMAKE_CURRENT_LIST_DIR}/")

# For IDE users
ocv_source_group("Src"     FILES ${gapi_srcs})
ocv_source_group("Include" FILES ${gapi_ext_hdrs})

ocv_set_module_sources(HEADERS ${gapi_ext_hdrs} SOURCES ${gapi_srcs})
ocv_module_include_directories("${CMAKE_CURRENT_LIST_DIR}/src")

ocv_create_module()

ocv_target_link_libraries(${the_module} PRIVATE ade ${INF_ENGINE_TARGET})
if(HAVE_TBB)
  ocv_target_link_libraries(${the_module} PRIVATE tbb)
endif()

ocv_add_accuracy_tests(${INF_ENGINE_TARGET})
# FIXME: test binary is linked with ADE directly since ADE symbols
# are not exported from libopencv_gapi.so in any form - thus
# there're two copies of ADE code in memory when tests run (!)
# src/ is specified to include dirs for INTERNAL tests only.
if(TARGET opencv_test_gapi)
  target_include_directories(opencv_test_gapi PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
  target_link_libraries(opencv_test_gapi PRIVATE ade)
endif()

if(HAVE_FREETYPE)
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_FREETYPE)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE -DHAVE_FREETYPE)
  endif()
  ocv_target_link_libraries(${the_module} PRIVATE ${FREETYPE_LIBRARIES})
  ocv_target_include_directories(${the_module} PRIVATE ${FREETYPE_INCLUDE_DIRS})
endif()

if(HAVE_PLAIDML)
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_PLAIDML)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE -DHAVE_PLAIDML)
  endif()
  ocv_target_link_libraries(${the_module} PRIVATE ${PLAIDML_LIBRARIES})
  ocv_target_include_directories(${the_module} SYSTEM PRIVATE ${PLAIDML_INCLUDE_DIRS})
endif()

ocv_add_perf_tests()
ocv_add_samples()
