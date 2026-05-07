# Vendored from microsoft/onnxruntime onnxruntime/core/mlas/. License: MIT.
# See LICENSE and README.md in this directory for provenance.
#
# Stripped to MlasGemm SGEMM only. ONLY the source files needed for
# MlasGemm (single-precision GEMM) and its per-arch optimizations are
# compiled here. Non-SGEMM dispatch lines in lib/platform.cpp are
# `#if 0`'d out (search for "MLAS_GEMM_ONLY"). Re-vendoring upstream
# requires re-applying that patch.
#
# Local patches against upstream:
#   - lib/threading.cpp -> threading_opencv.cpp (cv::parallel_for_).
#   - mlasi.h: MlasGetMaximumThreadCount() returns cv::getNumThreads().
#   - lib/core/common/{narrow,common}.h shims for ORT internals MLAS uses.
#   - lib/platform.cpp: non-SGEMM dispatch removed (`#if 0` blocks).

set(MLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(MLAS_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Hardware-agnostic core: just the bare minimum for SGEMM dispatch + math.
set(mlas_common_srcs
  ${MLAS_SRC_DIR}/platform.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/threading_opencv.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
)

# Architecture detection — same as ORT's onnxruntime_mlas.cmake.
set(MLAS_ARM       FALSE)
set(MLAS_ARM64     FALSE)
set(MLAS_POWER     FALSE)
set(MLAS_X86       FALSE)
set(MLAS_X86_64    FALSE)
set(MLAS_RISCV64   FALSE)
set(MLAS_LOONGARCH64 FALSE)
set(MLAS_S390X     FALSE)
set(MLAS_WASM      FALSE)

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(MLAS_WASM TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
  set(MLAS_ARM64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
  set(MLAS_ARM64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
  set(MLAS_ARM TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc.*|ppc.*)")
  set(MLAS_POWER TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
  set(MLAS_X86 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(MLAS_X86_64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64.*")
  set(MLAS_RISCV64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^loongarch64.*")
  set(MLAS_LOONGARCH64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^s390x$")
  set(MLAS_S390X TRUE)
endif()

set(mlas_platform_srcs)

# x86_64 SGEMM kernels: MlasGemmFloatKernelSse / Avx / Fma3 / Avx512F
# plus the M=1 fast paths and B-packing helpers.
if(MLAS_X86_64)
  enable_language(ASM)
  list(APPEND mlas_platform_srcs
    ${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
    ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
    ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
    ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
    ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
    ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
    ${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
    ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx512F.S
  )
  set_source_files_properties(${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
                              ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
                              PROPERTIES COMPILE_FLAGS "-msse2")
  set_source_files_properties(${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
                              ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
                              ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
                              ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
                              PROPERTIES COMPILE_FLAGS "-mavx")
  set_source_files_properties(${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
                              PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
  set_source_files_properties(${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx512F.S
                              PROPERTIES COMPILE_FLAGS "-mavx512f")
endif()

# i386 / 32-bit x86 SGEMM kernels.
if(MLAS_X86)
  enable_language(ASM)
  list(APPEND mlas_platform_srcs
    ${MLAS_SRC_DIR}/x86/SgemmKernelSse2.S
    ${MLAS_SRC_DIR}/x86/SgemmKernelAvx.S
  )
  set_source_files_properties(${MLAS_SRC_DIR}/x86/SgemmKernelSse2.S PROPERTIES COMPILE_FLAGS "-msse2")
  set_source_files_properties(${MLAS_SRC_DIR}/x86/SgemmKernelAvx.S  PROPERTIES COMPILE_FLAGS "-mavx")
  if(ANDROID)
    list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/x86/x86.get_pc_thunk.S)
  endif()
endif()

# ARM 32-bit: pure C++ sgemmc.
if(MLAS_ARM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
  list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/arm/sgemmc.cpp)
endif()

# ARM64 / AArch64.
if(MLAS_ARM64)
  enable_language(ASM)
  list(APPEND mlas_platform_srcs
    ${MLAS_SRC_DIR}/aarch64/SgemmKernelNeon.S
    ${MLAS_SRC_DIR}/aarch64/SgemvKernelNeon.S
  )
endif()

# POWER (ppc64le / AIX). Always compile the base SgemmKernelPower; opt-in
# POWER10 if the compiler supports -mcpu=power10.
if(MLAS_POWER)
  list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/power/SgemmKernelPower.cpp)
  set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPower.cpp
    PROPERTIES COMPILE_FLAGS "-DSINGLE")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=power10" MLAS_HAS_POWER10)
  if(MLAS_HAS_POWER10)
    enable_language(ASM)
    list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/power/SgemmKernelPOWER10.cpp)
    if(NOT AIX)
      list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/power/SgemmKernelPackA.S)
      set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPackA.S
        PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10")
    endif()
    set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPOWER10.cpp
      PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10 -DSINGLE")
  endif()
endif()

# LoongArch 64 (LSX + LASX).
if(MLAS_LOONGARCH64)
  enable_language(ASM)
  list(APPEND mlas_platform_srcs
    ${MLAS_SRC_DIR}/loongarch64/SgemmKernelLsx.S
    ${MLAS_SRC_DIR}/loongarch64/SgemmTransposePackB16x4LSX.S
    ${MLAS_SRC_DIR}/loongarch64/SgemmKernelLasx.S
    ${MLAS_SRC_DIR}/loongarch64/SgemmTransposePackB16x4Lasx.S
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mlsx -mlasx")
endif()

# IBM s390x (z/Architecture, ZVECTOR).
if(MLAS_S390X)
  list(APPEND mlas_platform_srcs
    ${MLAS_SRC_DIR}/s390x/SgemmKernel.cpp
    ${MLAS_SRC_DIR}/s390x/SgemmKernelZVECTOR.cpp
  )
  set_source_files_properties(${MLAS_SRC_DIR}/s390x/SgemmKernel.cpp        PROPERTIES COMPILE_FLAGS "-DSINGLE")
  set_source_files_properties(${MLAS_SRC_DIR}/s390x/SgemmKernelZVECTOR.cpp PROPERTIES COMPILE_FLAGS "-DSINGLE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mvx -mzvector -march=z15")
endif()

# RISC-V 64. RVV vector kernels iff the compiler supports rv64gcv.
if(MLAS_RISCV64)
  include(CheckCXXSourceCompiles)
  set(_old "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "-march=rv64gcv -mabi=lp64d")
  check_cxx_source_compiles("
    #include <stddef.h>
    #include <riscv_vector.h>
    int main() { size_t vl = __riscv_vsetvl_e32m1(4); return static_cast<int>(vl == 0); }"
    MLAS_HAS_RISCV64_RVV)
  set(CMAKE_REQUIRED_FLAGS "${_old}")
  if(MLAS_HAS_RISCV64_RVV)
    list(APPEND mlas_platform_srcs
      ${MLAS_SRC_DIR}/riscv64/sgemm_pack_b_rvv.cpp
      ${MLAS_SRC_DIR}/riscv64/sgemm_kernel_rvv.cpp
    )
    foreach(f
      ${MLAS_SRC_DIR}/riscv64/sgemm_pack_b_rvv.cpp
      ${MLAS_SRC_DIR}/riscv64/sgemm_kernel_rvv.cpp)
      set_source_files_properties(${f} PROPERTIES COMPILE_FLAGS "-march=rv64gcv -mabi=lp64d")
    endforeach()
  endif()
endif()

# WASM and unknown archs fall back to scalar sgemm in sgemm.cpp's generic path.

add_library(opencv_dnn_mlas OBJECT ${mlas_common_srcs} ${mlas_platform_srcs})
set_target_properties(opencv_dnn_mlas PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(opencv_dnn_mlas PUBLIC  ${MLAS_INC_DIR})
target_include_directories(opencv_dnn_mlas PRIVATE ${MLAS_SRC_DIR})
target_include_directories(opencv_dnn_mlas PRIVATE ${CMAKE_SOURCE_DIR}/modules/core/include)

target_compile_definitions(opencv_dnn_mlas PRIVATE
  BUILD_MLAS_NO_ONNXRUNTIME=1
  MLAS_OPENCV_THREADING=1
  MLAS_GEMM_ONLY=1
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(opencv_dnn_mlas PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-w>"
    "$<$<COMPILE_LANGUAGE:CXX>:-include>"
    "$<$<COMPILE_LANGUAGE:CXX>:cstring>"
  )
endif()
if(MLAS_X86_64)
  set_source_files_properties(${MLAS_SRC_DIR}/platform.cpp
    PROPERTIES COMPILE_FLAGS "-include unistd.h")
endif()

set(HAVE_MLAS 1 PARENT_SCOPE)
set(MLAS_OBJECTS $<TARGET_OBJECTS:opencv_dnn_mlas> PARENT_SCOPE)
set(MLAS_INCLUDE_DIRS ${MLAS_INC_DIR} PARENT_SCOPE)

message(STATUS "MLAS [SGEMM-only]: arch — "
  "ARM=${MLAS_ARM} ARM64=${MLAS_ARM64} POWER=${MLAS_POWER} "
  "X86=${MLAS_X86} X86_64=${MLAS_X86_64} RISCV64=${MLAS_RISCV64} "
  "LOONGARCH64=${MLAS_LOONGARCH64} S390X=${MLAS_S390X} WASM=${MLAS_WASM}")
