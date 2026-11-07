# Vendored g2o — graph-based non-linear least-squares optimizer.
# BSD-only subset of g2o tag 20200410_git (see license-bsd.txt). The GPL/LGPL
# parts of upstream g2o (csparse_extension, cholmod, apps/g2o_viewer,
# examples/slam2d_g2o, stuff/opengl_primitives) have been removed; only the
# BSD components the SLAM module uses are kept:
#   core, stuff, types/{sba,sim3}, types/slam3d (headers only: se3quat / se3_ops),
#   solvers/{dense,eigen}.
# Built as a static library and linked into any OpenCV module that needs it.
# Requires Eigen3 (found by the parent OpenCV build via EIGEN_INCLUDE_PATH).

if(NOT HAVE_EIGEN)
  message(STATUS "G2O: Eigen not found — g2o will not be built")
  return()
endif()

set(G2O_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(G2O_SRC  "${G2O_ROOT}/g2o")

# ---------------------------------------------------------------------------
# Generate config.h. All optional features stay OFF (no OpenGL, no OpenMP, no
# CHOLMOD/CSparse, static build, double precision) — the corresponding
# #cmakedefine lines become "#undef". Emitted into the build tree (not the
# source tree) at <build>/g2o/config.h; g2o sources include it as
# "g2o/config.h", so the build dir is added to the include path below.
# ---------------------------------------------------------------------------
set(G2O_CXX_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
set(G2O_SRC_DIR "${G2O_SRC}")
configure_file("${G2O_ROOT}/config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/g2o/config.h")

# ---------------------------------------------------------------------------
# Sources. core + stuff are the engine; types/sba + types/sim3 supply the
# vertices/edges the SLAM optimizer instantiates. slam3d contributes only the
# header-only SE3Quat / se3_ops used by the sba types, so it has no .cpp here.
# ---------------------------------------------------------------------------
file(GLOB g2o_core_srcs  "${G2O_SRC}/core/*.cpp")
# Standalone test program (has its own main()); upstream builds it separately.
list(REMOVE_ITEM g2o_core_srcs "${G2O_SRC}/core/sparse_block_matrix_test.cpp")
file(GLOB g2o_stuff_srcs "${G2O_SRC}/stuff/*.cpp" "${G2O_SRC}/stuff/*.c")
file(GLOB g2o_sba_srcs   "${G2O_SRC}/types/sba/*.cpp")
file(GLOB g2o_sim3_srcs  "${G2O_SRC}/types/sim3/*.cpp")
set(g2o_srcs ${g2o_core_srcs} ${g2o_stuff_srcs} ${g2o_sba_srcs} ${g2o_sim3_srcs})

add_library(g2o STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${g2o_srcs})

target_include_directories(g2o
  PUBLIC
    "${G2O_ROOT}"                  # so <g2o/core/...>, <g2o/types/...> resolve
    "${CMAKE_CURRENT_BINARY_DIR}"  # so generated <g2o/config.h> resolves
)

# g2o 20200410 uses C++14 (std::make_unique, etc.).
target_compile_features(g2o PUBLIC cxx_std_14)

# Eigen3 may be available as an imported target (conda/modern CMake) or as a path.
if(TARGET Eigen3::Eigen)
  target_link_libraries(g2o PUBLIC Eigen3::Eigen)
elseif(EIGEN_INCLUDE_PATH)
  target_include_directories(g2o PUBLIC "${EIGEN_INCLUDE_PATH}")
endif()

if(UNIX)
  target_compile_definitions(g2o PRIVATE UNIX)
endif()

set_target_properties(g2o PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  FOLDER "3rdparty"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

# Suppress all warnings for this 3rdparty target only.
# NOTE: single-value $<CXX_COMPILER_ID:...> checks (not the comma-list form,
# which needs CMake >= 3.15) to stay compatible with MIN_VER_CMAKE (3.13).
target_compile_options(g2o PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-w>
  $<$<CXX_COMPILER_ID:Clang>:-w>
  $<$<CXX_COMPILER_ID:AppleClang>:-w>
  $<$<CXX_COMPILER_ID:MSVC>:/w>
)

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(g2o EXPORT OpenCVModules
      ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()
ocv_install_3rdparty_licenses(g2o license-bsd.txt)

set(G2O_INCLUDE_DIRS
  "${G2O_ROOT}"
  "${CMAKE_CURRENT_BINARY_DIR}"
  PARENT_SCOPE
)
set(G2O_LIBRARIES g2o PARENT_SCOPE)

if(TARGET Eigen3::Eigen)
  message(STATUS "G2O: build enabled (BSD subset, tag 20200410; Eigen: Eigen3::Eigen target)")
else()
  message(STATUS "G2O: build enabled (BSD subset, tag 20200410; Eigen: ${EIGEN_INCLUDE_PATH})")
endif()
