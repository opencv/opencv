# Vendored from microsoft/onnxruntime onnxruntime/core/mlas/. License: MIT.
# See LICENSE and README.md in this directory for provenance.
#
# Vendored subset: SGEMM (sgemm.cpp + arch kernels) plus MlasFlashAttention
# (flashattn.cpp) and the portable softmax kernels it needs from compute.cpp.
# Non-SGEMM / non-FlashAttention dispatch lines in lib/platform.cpp are
# `#if 0`'d out (search for "MLAS_GEMM_ONLY"). Re-vendoring upstream
# requires re-applying that patch.
#
# Local patches against upstream:
#   - lib/threading.cpp -> modules/dnn/src/layers/cpu_kernels/mlas_threading.cpp
#     (cv::parallel_for_).
#   - mlasi.h: MlasGetMaximumThreadCount() returns cv::getNumThreads().
#   - lib/core/common/{narrow,common}.h shims for ORT internals MLAS uses.
#   - lib/platform.cpp: non-SGEMM dispatch removed (`#if 0` blocks), and
#     ReduceMaximumF32Kernel / ComputeSumExpF32Kernel initialized to the
#     portable compute.cpp fallbacks.

set(MLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(MLAS_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Hardware-agnostic core: SGEMM dispatch + softmax/exp kernels + flash-attention.
set(mlas_common_srcs
  ${MLAS_SRC_DIR}/platform.cpp
  ${CMAKE_SOURCE_DIR}/modules/dnn/src/layers/cpu_kernels/mlas_threading.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
  ${MLAS_SRC_DIR}/compute.cpp
  ${MLAS_SRC_DIR}/flashattn.cpp
)

# Architecture detection — same as ORT's onnxruntime_mlas.cmake.
set(MLAS_ARM         FALSE CACHE INTERNAL "" FORCE)
set(MLAS_ARM64       FALSE CACHE INTERNAL "" FORCE)
set(MLAS_POWER       FALSE CACHE INTERNAL "" FORCE)
set(MLAS_X86         FALSE CACHE INTERNAL "" FORCE)
set(MLAS_X86_64      FALSE CACHE INTERNAL "" FORCE)
set(MLAS_RISCV64     FALSE CACHE INTERNAL "" FORCE)
set(MLAS_LOONGARCH64 FALSE CACHE INTERNAL "" FORCE)
set(MLAS_S390X       FALSE CACHE INTERNAL "" FORCE)
set(MLAS_WASM        FALSE CACHE INTERNAL "" FORCE)

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(MLAS_WASM TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
  set(MLAS_ARM64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
  set(MLAS_ARM64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
  set(MLAS_ARM TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc.*|ppc.*)")
  set(MLAS_POWER TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
  set(MLAS_X86 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(MLAS_X86_64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64.*")
  set(MLAS_RISCV64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^loongarch64.*")
  set(MLAS_LOONGARCH64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^s390x$")
  set(MLAS_S390X TRUE CACHE INTERNAL "" FORCE)
endif()

set(mlas_platform_srcs)

# State flags consumed by the top-level config summary's PNG-style chain:
#   OPENCV_DNN_MLAS_ENABLED      — 1 if MLAS compiled in, 0 if skipped/unavailable
#   OPENCV_DNN_MLAS_SKIP_REASON  — human-readable reason for the NO case
# Cleared in modules/dnn/CMakeLists.txt before add_subdirectory() so stale
# values don't survive a config where MLAS is no longer reached.
set(OPENCV_DNN_MLAS_ENABLED     0  CACHE INTERNAL "" FORCE)
set(OPENCV_DNN_MLAS_SKIP_REASON "" CACHE INTERNAL "" FORCE)

# Probe ASM language once. check_language(ASM) is a no-op when
# CMAKE_ASM_COMPILER is already set in cache — and the Android NDK toolchain
# pre-sets it for every ABI. So on Android the guard falls through to
# enable_language(ASM), which then fails at generate time with
# "CMAKE_ASM_COMPILE_OBJECT not set" on the NDK + CMake 3.22.1 combo CI uses.
# Skip MLAS on Android for the ASM-using arches; the DNN module falls back to
# its built-in SGEMM. Android armv7a stays enabled via the C++-only
# sgemmc.cpp path under MLAS_ARM below.
set(_MLAS_REQUIRES_ASM FALSE)
if(MLAS_X86_64 OR MLAS_X86 OR MLAS_ARM64 OR MLAS_LOONGARCH64)
  set(_MLAS_REQUIRES_ASM TRUE)
endif()

include(CheckLanguage)
set(MLAS_HAS_ASM FALSE CACHE INTERNAL "" FORCE)
check_language(ASM)
if(CMAKE_ASM_COMPILER)
  enable_language(ASM)
  set(MLAS_HAS_ASM TRUE CACHE INTERNAL "" FORCE)
elseif(_MLAS_REQUIRES_ASM)
  set(OPENCV_DNN_MLAS_SKIP_REASON
    "no ASM compiler available for ${CMAKE_SYSTEM_PROCESSOR}"
    CACHE INTERNAL "" FORCE)
  message(WARNING "MLAS: ASM language unavailable on ${CMAKE_SYSTEM_PROCESSOR}; "
                  "MLAS disabled (DNN will use its built-in SGEMM)")
  return()
endif()

# x86_64 SGEMM kernels: MlasGemmFloatKernelSse / Avx / Fma3 / Avx512F
# plus the M=1 fast paths and B-packing helpers.
if(MLAS_X86_64)
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

# ARM 32-bit: pure C++ sgemmc (no .S kernels).
if(MLAS_ARM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
  list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/arm/sgemmc.cpp)
endif()

# ARM64 / AArch64.
if(MLAS_ARM64)
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
  set(MLAS_HAS_POWER10 ${MLAS_HAS_POWER10} CACHE INTERNAL "" FORCE)
  if(MLAS_HAS_POWER10 AND MLAS_HAS_ASM)
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
  set(MLAS_HAS_RISCV64_RVV ${MLAS_HAS_RISCV64_RVV} CACHE INTERNAL "" FORCE)
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
  # Scalar MlasSgemmKernelZero/Add fallback for non-RVV runtime/build.
  list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/scalar/SgemmKernelScalar.cpp)
endif()

# WASM / unknown archs use the scalar sgemm path; link its kernel.
if(MLAS_WASM OR NOT (MLAS_X86 OR MLAS_X86_64 OR MLAS_ARM OR MLAS_ARM64
                     OR MLAS_POWER OR MLAS_LOONGARCH64 OR MLAS_S390X
                     OR MLAS_RISCV64))
  list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/scalar/SgemmKernelScalar.cpp)
endif()

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
  # MLAS .S files lack .note.GNU-stack; tell the assembler the stack is non-exec.
  target_compile_options(opencv_dnn_mlas PRIVATE
    "$<$<COMPILE_LANGUAGE:ASM>:-Wa,--noexecstack>"
  )
endif()
# platform.cpp's MlasInitAMX() calls syscall() but only includes <sys/syscall.h>,
# not <unistd.h> where glibc declares it. It's compiled on both x86 and x86_64
# (shared MLAS_TARGET_AMD64_IX86 block), so force-include unistd.h on both.
# MSVC has no unistd.h and its _WIN32 path doesn't use syscall() anyway.
if((MLAS_X86_64 OR MLAS_X86) AND NOT WIN32)
  set_source_files_properties(${MLAS_SRC_DIR}/platform.cpp
    PROPERTIES COMPILE_FLAGS "-include unistd.h")
endif()

set(HAVE_MLAS 1 PARENT_SCOPE)
set(MLAS_OBJECTS $<TARGET_OBJECTS:opencv_dnn_mlas> PARENT_SCOPE)
set(MLAS_INCLUDE_DIRS ${MLAS_INC_DIR} PARENT_SCOPE)
set(OPENCV_DNN_MLAS_ENABLED 1 CACHE INTERNAL "" FORCE)
