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

if(TARGET ocv.3rdparty.openvino)
  # TODO: remove OPENCV_GAPI_INF_ENGINE option
  set(initial_value ON)
  if(DEFINED OPENCV_GAPI_INF_ENGINE)
    set(initial_value ${OPENCV_GAPI_INF_ENGINE})
    message(WARNING "OPENCV_GAPI_INF_ENGINE option is deprecated. Use OPENCV_GAPI_WITH_OPENVINO option instead.")
  endif()
  ocv_option(OPENCV_GAPI_WITH_OPENVINO "G-API: Enable OpenVINO Toolkit support" ${initial_value})
endif()

set(the_description "OpenCV G-API Core Module")

ocv_add_module(gapi
    REQUIRED
      opencv_imgproc
    OPTIONAL
      opencv_video opencv_calib3d
    WRAP
      python
)

if(MSVC)
  if(MSVC_VERSION LESS 1910)
    # Disable obsolete warning C4503 popping up on MSVC << 15 2017
    # https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4503?view=vs-2019
    # and IE deprecated code warning C4996
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4503 /wd4996)
  endif()
endif()

file(GLOB gapi_ext_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cpu/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/fluid/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/gpu/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/infer/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/oak/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/ocl/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/own/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/plaidml/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/python/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/render/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/s11n/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/streaming/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/streaming/gstreamer/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/streaming/onevpl/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/plaidml/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/util/*.hpp"
    )

set(gapi_srcs
    # Front-end part
    src/api/grunarg.cpp
    src/api/gorigin.cpp
    src/api/gmat.cpp
    src/api/garray.cpp
    src/api/gopaque.cpp
    src/api/gscalar.cpp
    src/api/gframe.cpp
    src/api/gkernel.cpp
    src/api/gbackend.cpp
    src/api/gproto.cpp
    src/api/gnode.cpp
    src/api/gcall.cpp
    src/api/gcomputation.cpp
    src/api/operators.cpp
    src/api/kernels_core.cpp
    src/api/kernels_imgproc.cpp
    src/api/kernels_video.cpp
    src/api/kernels_nnparsers.cpp
    src/api/kernels_streaming.cpp
    src/api/kernels_stereo.cpp
    src/api/render.cpp
    src/api/render_ocv.cpp
    src/api/ginfer.cpp
    src/api/media.cpp
    src/api/rmat.cpp

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
    src/compiler/passes/intrin.cpp

    # Executor
    src/executor/gexecutor.cpp
    src/executor/gtbbexecutor.cpp
    src/executor/gstreamingexecutor.cpp
    src/executor/gasync.cpp

    # CPU Backend (currently built-in)
    src/backends/cpu/gcpubackend.cpp
    src/backends/cpu/gcpukernel.cpp
    src/backends/cpu/gcpuimgproc.cpp
    src/backends/cpu/gcpustereo.cpp
    src/backends/cpu/gcpuvideo.cpp
    src/backends/cpu/gcpucore.cpp
    src/backends/cpu/gnnparsers.cpp

    # Fluid Backend (also built-in, FIXME:move away)
    src/backends/fluid/gfluidbuffer.cpp
    src/backends/fluid/gfluidbackend.cpp
    src/backends/fluid/gfluidimgproc.cpp
    src/backends/fluid/gfluidimgproc_func.dispatch.cpp
    src/backends/fluid/gfluidcore.cpp
	  src/backends/fluid/gfluidcore_func.dispatch.cpp

    # OAK Backend (optional)
    src/backends/oak/goak.cpp
    src/backends/oak/goakbackend.cpp
    src/backends/oak/goak_media_adapter.cpp

    # OCL Backend (currently built-in)
    src/backends/ocl/goclbackend.cpp
    src/backends/ocl/goclkernel.cpp
    src/backends/ocl/goclimgproc.cpp
    src/backends/ocl/goclcore.cpp

    # IE Backend. FIXME: should be included by CMake
    # if and only if IE support is enabled
    src/backends/ie/giebackend.cpp
    src/backends/ie/giebackend/giewrapper.cpp

    # ONNX backend
    src/backends/onnx/gonnxbackend.cpp

    # Render backend
    src/backends/render/grenderocv.cpp
    src/backends/render/ft_render.cpp

    # PlaidML Backend
    src/backends/plaidml/gplaidmlcore.cpp
    src/backends/plaidml/gplaidmlbackend.cpp

    # Common backend code
    src/backends/common/gmetabackend.cpp
    src/backends/common/gcompoundbackend.cpp
    src/backends/common/gcompoundkernel.cpp

    # Serialization API and routines
    src/api/s11n.cpp
    src/backends/common/serialization.cpp

    # Streaming backend
    src/backends/streaming/gstreamingbackend.cpp

    # Python bridge
    src/backends/ie/bindings_ie.cpp
    src/backends/python/gpythonbackend.cpp

    # OpenVPL Streaming source
    src/streaming/onevpl/source.cpp
    src/streaming/onevpl/source_priv.cpp
    src/streaming/onevpl/file_data_provider.cpp
    src/streaming/onevpl/cfg_params.cpp
    src/streaming/onevpl/cfg_params_parser.cpp
    src/streaming/onevpl/utils.cpp
    src/streaming/onevpl/data_provider_interface_exception.cpp
    src/streaming/onevpl/accelerators/surface/cpu_frame_adapter.cpp
    src/streaming/onevpl/accelerators/surface/dx11_frame_adapter.cpp
    src/streaming/onevpl/accelerators/surface/surface.cpp
    src/streaming/onevpl/accelerators/surface/surface_pool.cpp
    src/streaming/onevpl/accelerators/utils/shared_lock.cpp
    src/streaming/onevpl/accelerators/accel_policy_cpu.cpp
    src/streaming/onevpl/accelerators/accel_policy_dx11.cpp
    src/streaming/onevpl/accelerators/dx11_alloc_resource.cpp
    src/streaming/onevpl/engine/engine_session.cpp
    src/streaming/onevpl/engine/processing_engine_base.cpp
    src/streaming/onevpl/engine/decode/decode_engine_legacy.cpp
    src/streaming/onevpl/engine/decode/decode_session.cpp
    src/streaming/onevpl/engine/transcode/transcode_engine_legacy.cpp
    src/streaming/onevpl/engine/transcode/transcode_session.cpp
    src/streaming/onevpl/demux/async_mfp_demux_data_provider.cpp
    src/streaming/onevpl/data_provider_dispatcher.cpp

    src/streaming/onevpl/cfg_param_device_selector.cpp
    src/streaming/onevpl/device_selector_interface.cpp

    # GStreamer Streaming source
    src/streaming/gstreamer/gstreamer_pipeline_facade.cpp
    src/streaming/gstreamer/gstreamerpipeline.cpp
    src/streaming/gstreamer/gstreamersource.cpp
    src/streaming/gstreamer/gstreamer_buffer_utils.cpp
    src/streaming/gstreamer/gstreamer_media_adapter.cpp
    src/streaming/gstreamer/gstreamerenv.cpp

    # Utils (ITT tracing)
    src/utils/itt.cpp
    )

ocv_add_dispatched_file(backends/fluid/gfluidimgproc_func SSE4_1 AVX2)
ocv_add_dispatched_file(backends/fluid/gfluidcore_func SSE4_1 AVX2)

ocv_list_add_prefix(gapi_srcs "${CMAKE_CURRENT_LIST_DIR}/")

# For IDE users
ocv_source_group("Src"     FILES ${gapi_srcs})
ocv_source_group("Include" FILES ${gapi_ext_hdrs})

ocv_set_module_sources(HEADERS ${gapi_ext_hdrs} SOURCES ${gapi_srcs})
ocv_module_include_directories("${CMAKE_CURRENT_LIST_DIR}/src")

ocv_create_module()

ocv_target_link_libraries(${the_module} PRIVATE ade)

if(TARGET ocv.3rdparty.openvino AND OPENCV_GAPI_WITH_OPENVINO)
  ocv_target_link_libraries(${the_module} PRIVATE ocv.3rdparty.openvino)
endif()

if(HAVE_TBB)
  ocv_target_link_libraries(${the_module} PRIVATE tbb)
endif()

# TODO: Consider support of ITT in G-API standalone mode.
if(CV_TRACE AND HAVE_ITT)
  ocv_target_compile_definitions(${the_module} PRIVATE -DOPENCV_WITH_ITT=1)
  ocv_module_include_directories(${ITT_INCLUDE_DIRS})
  ocv_target_link_libraries(${the_module} PRIVATE ${ITT_LIBRARIES})
endif()

set(__test_extra_deps "")
if(TARGET ocv.3rdparty.openvino AND OPENCV_GAPI_WITH_OPENVINO)
  list(APPEND __test_extra_deps ocv.3rdparty.openvino)
endif()
ocv_add_accuracy_tests(${__test_extra_deps})

# FIXME: test binary is linked with ADE directly since ADE symbols
# are not exported from libopencv_gapi.so in any form - thus
# there're two copies of ADE code in memory when tests run (!)
# src/ is specified to include dirs for INTERNAL tests only.
if(TARGET opencv_test_gapi)
  target_include_directories(opencv_test_gapi PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
  target_link_libraries(opencv_test_gapi PRIVATE ade)
endif()

if(HAVE_TBB AND TARGET opencv_test_gapi)
  ocv_target_link_libraries(opencv_test_gapi PRIVATE tbb)
endif()

if(HAVE_FREETYPE)
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_FREETYPE)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE -DHAVE_FREETYPE)
  endif()
  ocv_target_link_libraries(${the_module} PRIVATE ${FREETYPE_LIBRARIES})
  ocv_target_include_directories(${the_module} PRIVATE ${FREETYPE_INCLUDE_DIRS})
endif()

if(HAVE_OAK)
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_OAK)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE -DHAVE_OAK)
  endif()
  ocv_target_link_libraries(${the_module} PRIVATE depthai::core)
endif()

if(HAVE_PLAIDML)
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_PLAIDML)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE -DHAVE_PLAIDML)
  endif()
  ocv_target_link_libraries(${the_module} PRIVATE ${PLAIDML_LIBRARIES})
  ocv_target_include_directories(${the_module} SYSTEM PRIVATE ${PLAIDML_INCLUDE_DIRS})
endif()

if(HAVE_GAPI_ONEVPL)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE -DHAVE_ONEVPL)
    ocv_target_link_libraries(opencv_test_gapi PRIVATE ${VPL_IMPORTED_TARGETS})
    if(MSVC)
        target_compile_options(opencv_test_gapi PUBLIC "/wd4201")
    endif()
    if(HAVE_D3D11 AND HAVE_OPENCL)
        ocv_target_include_directories(opencv_test_gapi SYSTEM PRIVATE ${OPENCL_INCLUDE_DIRS})
    endif()
  endif()
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_ONEVPL)
  ocv_target_link_libraries(${the_module} PRIVATE ${VPL_IMPORTED_TARGETS})

  if(HAVE_D3D11 AND HAVE_OPENCL)
    ocv_target_include_directories(${the_module} SYSTEM PRIVATE ${OPENCL_INCLUDE_DIRS})
  endif()
endif()

ocv_option(OPENCV_GAPI_GSTREAMER "Build G-API with GStreamer support" HAVE_GSTREAMER)
if(HAVE_GSTREAMER AND OPENCV_GAPI_GSTREAMER)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE -DHAVE_GSTREAMER)
    ocv_target_link_libraries(opencv_test_gapi PRIVATE ocv.3rdparty.gstreamer)
  endif()
  ocv_target_compile_definitions(${the_module} PRIVATE -DHAVE_GSTREAMER)
  ocv_target_link_libraries(${the_module} PRIVATE ocv.3rdparty.gstreamer)
endif()

if(WIN32)
  # Required for htonl/ntohl on Windows
  ocv_target_link_libraries(${the_module} PRIVATE wsock32 ws2_32)
endif()

if(HAVE_ONNX)
  ocv_target_link_libraries(${the_module} PRIVATE ${ONNX_LIBRARY})
  ocv_target_compile_definitions(${the_module} PRIVATE HAVE_ONNX=1)
  if(TARGET opencv_test_gapi)
    ocv_target_compile_definitions(opencv_test_gapi PRIVATE HAVE_ONNX=1)
    ocv_target_link_libraries(opencv_test_gapi PRIVATE ${ONNX_LIBRARY})
  endif()
endif()

ocv_add_perf_tests()
ocv_add_samples()


# Required for sample with inference on host
if(TARGET example_gapi_onevpl_infer_single_roi)
  if(TARGET ocv.3rdparty.openvino AND OPENCV_GAPI_WITH_OPENVINO)
    ocv_target_link_libraries(example_gapi_onevpl_infer_single_roi PRIVATE ocv.3rdparty.openvino)
  endif()
  if(HAVE_D3D11 AND HAVE_OPENCL)
    ocv_target_include_directories(example_gapi_onevpl_infer_single_roi SYSTEM PRIVATE ${OPENCL_INCLUDE_DIRS})
  endif()
endif()

if(TARGET example_gapi_pipeline_modeling_tool)
  if(WIN32)
    ocv_target_link_libraries(example_gapi_pipeline_modeling_tool winmm.lib)
  endif()
endif()

# perf test dependencies postprocessing
if(HAVE_GAPI_ONEVPL)
  # NB: TARGET opencv_perf_gapi doesn't exist before `ocv_add_perf_tests`
  if(TARGET opencv_perf_gapi)
    ocv_target_compile_definitions(opencv_perf_gapi PRIVATE -DHAVE_ONEVPL)
    ocv_target_link_libraries(opencv_perf_gapi PRIVATE ${VPL_IMPORTED_TARGETS})
    if(HAVE_D3D11 AND HAVE_OPENCL)
        ocv_target_include_directories(opencv_perf_gapi SYSTEM PRIVATE ${OPENCL_INCLUDE_DIRS})
    endif()
  endif()
endif()

if(HAVE_OAK)
  # FIXME: consider better solution
  if(TARGET example_gapi_oak_rgb_camera_encoding)
    ocv_target_compile_definitions(example_gapi_oak_rgb_camera_encoding PRIVATE -DHAVE_OAK)
  endif()
  if(TARGET example_gapi_oak_small_hetero_pipeline)
    ocv_target_compile_definitions(example_gapi_oak_small_hetero_pipeline PRIVATE -DHAVE_OAK)
  endif()
endif()
