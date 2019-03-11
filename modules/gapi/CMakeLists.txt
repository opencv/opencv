# FIXME: Rework standalone build in more generic maner
# (Restructure directories, add common pass, etc)
if (NOT DEFINED OPENCV_INITIAL_PASS)
    cmake_minimum_required(VERSION 3.3)
    include("cmake/standalone.cmake")
    return()
endif()

# FIXME: Remove CXX11 check after complete switch to OpenCV 4 branch
# (CI, bundle, workloads, etc)
if (NOT HAVE_CXX11 OR NOT TARGET ade)
  # can't build G-API because of the above reasons
  ocv_module_disable(gapi)
  return()
endif()

set(the_description "OpenCV G-API Core Module")
ocv_add_module(gapi opencv_imgproc)

file(GLOB gapi_ext_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/util/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cpu/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/gpu/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/ocl/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/fluid/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/own/*.hpp"
    )

set(gapi_srcs
    # Front-end part
    src/api/gapi_priv.cpp
    src/api/gmat.cpp
    src/api/garray.cpp
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

    # Compiler part
    src/compiler/gmodel.cpp
    src/compiler/gmodelbuilder.cpp
    src/compiler/gislandmodel.cpp
    src/compiler/gcompiler.cpp
    src/compiler/gcompiled.cpp
    src/compiler/passes/helpers.cpp
    src/compiler/passes/dump_dot.cpp
    src/compiler/passes/islands.cpp
    src/compiler/passes/meta.cpp
    src/compiler/passes/kernels.cpp
    src/compiler/passes/exec.cpp

    # Executor
    src/executor/gexecutor.cpp

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

# Note `ade` is not a module name but link dependency for ${the_module}
# (which is opencv_gapi)
ocv_create_module(ade)

ocv_add_accuracy_tests()
# FIXME: test binary is linked with ADE directly since ADE symbols
# are not exported from libopencv_gapi.so in any form - thus
# there're two copies of ADE code in memory when tests run (!)
# src/ is specified to include dirs for INTERNAL tests only.
if(TARGET opencv_test_gapi)
  target_include_directories(opencv_test_gapi PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
  target_link_libraries(opencv_test_gapi PRIVATE ade)
endif()

ocv_add_perf_tests()
ocv_add_samples()
