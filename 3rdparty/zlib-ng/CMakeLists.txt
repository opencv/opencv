project(${ZLIB_LIBRARY} LANGUAGES C)

if("c_std_11" IN_LIST CMAKE_C_COMPILE_FEATURES)
  set(CMAKE_C_STANDARD 11)          # The C standard whose features are requested to build this target
else()
  set(CMAKE_C_STANDARD 99)
endif()
set(CMAKE_C_STANDARD_REQUIRED ON) # Boolean describing whether the value of C_STANDARD is a requirement
set(CMAKE_C_EXTENSIONS OFF)       # Boolean specifying whether compiler specific extensions are requested

include(CheckTypeSize)
include(CheckSymbolExists)
include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)
include(CheckCCompilerFlag)
include(CMakeDependentOption)

if(X86_64 OR X86)
  set(BASEARCH_X86_FOUND TRUE)
endif()
if(AARCH64 OR ARM)
  set(BASEARCH_ARM_FOUND TRUE)
endif()
if(PPC64LE OR PPC64)
  set(BASEARCH_PPC_FOUND TRUE)
endif()
if(RISCV)
  set(BASEARCH_RISCV_FOUND TRUE)
endif()

include(cmake/detect-intrinsics.cmake)
include(cmake/fallback-macros.cmake)

set(ZLIB_SYMBOL_PREFIX "")

if(BASEARCH_X86_FOUND)
  set(WITH_AVX2 ON)
  set(WITH_AVX512 ON)
  set(WITH_AVX512VNNI ON)
  set(WITH_SSE2 ON)
  set(WITH_SSSE3 ON)
  set(WITH_SSE42 ON)
  set(WITH_PCLMULQDQ ON)
  set(WITH_VPCLMULQDQ ON)
endif()
if(BASEARCH_ARM_FOUND)
  set(WITH_ACLE ON)
  set(WITH_NEON ON)
  if(ARM)
    set(WITH_ARMV6 ON)
  else()
    set(WITH_ARMV6 OFF)
  endif()
endif()
if(BASEARCH_PPC_FOUND)
  set(WITH_ALTIVEC ON)
  set(WITH_POWER8 ON)
  set(WITH_POWER9 ON)
endif()
if(BASEARCH_RISCV_FOUND)
  set(WITH_RVV ON)
endif()


add_definitions(-DZLIB_COMPAT)

add_definitions(-DWITH_GZFILEOP)

if(CMAKE_C_COMPILER_ID MATCHES "^Intel")
  set(WARNFLAGS_DISABLE)
  check_c_compiler_flag(-diag-disable=10441 HAVE_DIAG_10441)
    if(HAVE_DIAG_10441)
        list(APPEND WARNFLAGS_DISABLE "-diag-disable=10441")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -diag-disable=10441")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -diag-disable=10441")
    endif()
elseif(MSVC)
  # Minimum supported MSVC version is 1800 = Visual Studio 12.0/2013
  # See also https://cmake.org/cmake/help/latest/variable/MSVC_VERSION.html
  if(MSVC_VERSION VERSION_LESS 1800)
    message(SEND_ERROR "Unsupported Visual Studio compiler version (requires 2013 or later).")
  endif()
  # TODO. ICC can be used through MSVC. I'm not sure if we'd ever see that combination
  # (who'd use cmake from an IDE...) but checking for ICC before checking for MSVC should
  # avoid mistakes.
  # /Oi ?
  set(WARNFLAGS_DISABLE)
  if(BASEARCH_ARM_FOUND)
      add_definitions(-D_ARM_WINAPI_PARTITION_DESKTOP_SDK_AVAILABLE)
      if(NOT "${ARCH}" MATCHES "aarch64")
          set(NEONFLAG "/arch:VFPv4")
      endif()
  endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(WARNFLAGS_DISABLE)
  # Check whether -fno-lto is available
  set(CMAKE_REQUIRED_FLAGS "-fno-lto")
  check_c_source_compiles(
    "int main() { return 0; }"
    FNO_LTO_AVAILABLE FAIL_REGEX "not supported")
  set(CMAKE_REQUIRED_FLAGS)
  if(FNO_LTO_AVAILABLE)
    set(ZNOLTOFLAG "-fno-lto")
  endif()
  if(BASEARCH_ARM_FOUND)
    if(ARM AND NOT CMAKE_C_FLAGS MATCHES "-mfloat-abi")
      # Auto-detect support for ARM floating point ABI
      check_include_file(features.h HAVE_FEATURES_H)
      if(HAVE_FEATURES_H)
        set(CMAKE_REQUIRED_FLAGS -mfloat-abi=softfp)
        check_c_source_compiles(
          "#include <features.h>
          int main() { return 0; }"
          HAVE_FLOATABI_SOFTFP)
        if(HAVE_FLOATABI_SOFTFP)
          set(FLOATABI -mfloat-abi=softfp)
        else()
          set(CMAKE_REQUIRED_FLAGS -mfloat-abi=hard)
          check_c_source_compiles(
            "#include <features.h>
            int main() { return 0; }"
            HAVE_FLOATABI_HARD)
          if(HAVE_FLOATABI_HARD)
            set(FLOATABI -mfloat-abi=hard)
          endif()
        endif()
        set(CMAKE_REQUIRED_FLAGS)
      endif()
      if(FLOATABI)
        message(STATUS "${ZLIB_LIBRARY} ARM floating point arch: ${FLOATABI}")
        add_compile_options(${FLOATABI})
      else()
        message(STATUS "${ZLIB_LIBRARY} ARM floating point arch not auto-detected")
      endif()
    endif()
  endif()
  if(FNO_LTO_AVAILABLE)
    set(NOLTOFLAG ${ZNOLTOFLAG})
  endif()
  if(MINGW)
    # Add `-Wno-pedantic-ms-format` only if the toolchain supports it
    check_c_compiler_flag(-Wno-pedantic-ms-format HAVE_NO_PEDANTIC_MS_FORMAT)
    if(HAVE_NO_PEDANTIC_MS_FORMAT)
      list(APPEND WARNFLAGS_DISABLE -Wno-pedantic-ms-format)
    endif()
  endif()
endif()

# Force disable LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Apply warning compiler flags
add_compile_options(${WARNFLAGS_DISABLE})

# Replace optimization level 3 added by default with level 2
if(NOT MSVC AND NOT CMAKE_C_FLAGS MATCHES "([\\/\\-]O)3")
  string(REGEX REPLACE "([\\/\\-]O)3" "\\12"
    CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
endif()

#
# Check for standard/system includes
#
check_include_file(arm_acle.h  HAVE_ARM_ACLE_H)
if(HAVE_ARM_ACLE_H)
  add_definitions(-DHAVE_ARM_ACLE_H)
endif()
check_include_file(sys/auxv.h  HAVE_SYS_AUXV_H)
if(HAVE_SYS_AUXV_H)
  add_definitions(-DHAVE_SYS_AUXV_H)
endif()
check_include_file(sys/sdt.h   HAVE_SYS_SDT_H)
if(HAVE_SYS_SDT_H)
  add_definitions(-DHAVE_SYS_SDT_H)
endif()
check_include_file(unistd.h    HAVE_UNISTD_H)

#
# Check for Linux includes
#
check_include_file(linux/auxvec.h HAVE_LINUX_AUXVEC_H)
if(HAVE_LINUX_AUXVEC_H)
    add_definitions(-DHAVE_LINUX_AUXVEC_H)
endif()

#
# Check to see if we have large file support
#
set(CMAKE_REQUIRED_DEFINITIONS -D_LARGEFILE64_SOURCE=1 -D__USE_LARGEFILE64)
check_type_size(off64_t OFF64_T)
if(HAVE_OFF64_T)
  add_definitions(-D_LARGEFILE64_SOURCE=1 -D__USE_LARGEFILE64)
else()
  check_type_size(_off64_t _OFF64_T)
  if(HAVE__OFF64_T)
    add_definitions(-D_LARGEFILE64_SOURCE=1 -D__USE_LARGEFILE64)
  else()
    check_type_size(__off64_t __OFF64_T)
  endif()
endif()
set(CMAKE_REQUIRED_DEFINITIONS) # clear variable

#
# Check for fseeko and other optional functions
#
check_function_exists(fseeko HAVE_FSEEKO)
if(NOT HAVE_FSEEKO)
  add_definitions(-DNO_FSEEKO)
endif()

check_function_exists(strerror HAVE_STRERROR)
if(NOT HAVE_STRERROR)
  add_definitions(-DNO_STRERROR)
endif()

set(CMAKE_REQUIRED_DEFINITIONS -D_POSIX_C_SOURCE=200112L)
check_symbol_exists(posix_memalign stdlib.h HAVE_POSIX_MEMALIGN)
if(HAVE_POSIX_MEMALIGN)
  add_definitions(-DHAVE_POSIX_MEMALIGN)
endif()
set(CMAKE_REQUIRED_DEFINITIONS)

set(CMAKE_REQUIRED_DEFINITIONS -D_ISOC11_SOURCE=1)
check_symbol_exists(aligned_alloc stdlib.h HAVE_ALIGNED_ALLOC)
if(HAVE_ALIGNED_ALLOC)
  add_definitions(-DHAVE_ALIGNED_ALLOC)
endif()
set(CMAKE_REQUIRED_DEFINITIONS)

#
# Check if we can hide zlib internal symbols that are linked between separate source files using hidden
#
check_c_source_compiles(
  "#define Z_INTERNAL __attribute__((visibility (\"hidden\")))
  int Z_INTERNAL foo;
  int main() {
      return 0;
  }"
  HAVE_ATTRIBUTE_VISIBILITY_HIDDEN FAIL_REGEX "visibility")
if(HAVE_ATTRIBUTE_VISIBILITY_HIDDEN)
  add_definitions(-DHAVE_VISIBILITY_HIDDEN)
endif()

#
# Check if we can hide zlib internal symbols that are linked between separate source files using internal
#
check_c_source_compiles(
  "#define Z_INTERNAL __attribute__((visibility (\"internal\")))
  int Z_INTERNAL foo;
  int main() {
      return 0;
  }"
  HAVE_ATTRIBUTE_VISIBILITY_INTERNAL FAIL_REGEX "visibility")
if(HAVE_ATTRIBUTE_VISIBILITY_INTERNAL)
  add_definitions(-DHAVE_VISIBILITY_INTERNAL)
endif()

#
# Check for __attribute__((aligned(x))) support in the compiler
#
check_c_source_compiles(
  "int main(void) {
      __attribute__((aligned(8))) int test = 0;
      (void)test;
      return 0;
  }"
  HAVE_ATTRIBUTE_ALIGNED FAIL_REGEX "aligned")
if(HAVE_ATTRIBUTE_ALIGNED)
  add_definitions(-DHAVE_ATTRIBUTE_ALIGNED)
endif()

#
# Check for __builtin_assume_aligned(x,n) support in the compiler
#
check_c_source_compiles(
    "char *test(char *buffer) {
        char *abuffer = __builtin_assume_aligned(buffer,64);
        return abuffer;
    }
    int main() {
        return 0;
    }"
    HAVE_BUILTIN_ASSUME_ALIGNED)
if(HAVE_BUILTIN_ASSUME_ALIGNED)
    add_definitions(-DHAVE_BUILTIN_ASSUME_ALIGNED)
endif()

#
# check for __builtin_ctz() support in the compiler
#
check_c_source_compiles(
  "int main(void) {
      unsigned int zero = 0;
      long test = __builtin_ctz(zero);
      (void)test;
      return 0;
  }"
  HAVE_BUILTIN_CTZ
)
if(HAVE_BUILTIN_CTZ)
  add_definitions(-DHAVE_BUILTIN_CTZ)
endif()

#
# check for __builtin_ctzll() support in the compiler
#
check_c_source_compiles(
  "int main(void) {
      unsigned int zero = 0;
      long test = __builtin_ctzll(zero);
      (void)test;
      return 0;
  }"
  HAVE_BUILTIN_CTZLL
)
if(HAVE_BUILTIN_CTZLL)
  add_definitions(-DHAVE_BUILTIN_CTZLL)
endif()

#
# check for ptrdiff_t support
#
check_c_source_compiles(
  "#include <stddef.h>
    int main() {
        ptrdiff_t *a;
        (void)a;
        return 0;
  }"
  HAVE_PTRDIFF_T
)
if(NOT HAVE_PTRDIFF_T)
  set(NEED_PTRDIFF_T 1)

  check_type_size("void *" SIZEOF_DATA_PTR)
  message(STATUS "sizeof(void *) is ${SIZEOF_DATA_PTR} bytes")

  if(${SIZEOF_DATA_PTR} MATCHES "4")
    set(PTRDIFF_TYPE "uint32_t")
  elseif(${SIZEOF_DATA_PTR} MATCHES "8")
    set(PTRDIFF_TYPE "uint64_t")
  else()
    message(FATAL_ERROR "sizeof(void *) is neither 32 nor 64 bit")
  endif()
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

set(GENERIC_ARCHDIR "arch/generic")

set(ZLIB_ARCH_SRCS)
set(ZLIB_ARCH_HDRS ${GENERIC_ARCHDIR}/generic_functions.h)
set(ARCHDIR ${GENERIC_ARCHDIR})
if(BASEARCH_X86_FOUND)
  set(ARCHDIR "arch/x86")
endif()
if(BASEARCH_ARM_FOUND)
  set(ARCHDIR "arch/arm")
endif()
if(BASEARCH_PPC_FOUND)
  set(ARCHDIR "arch/power")
endif()
if(BASEARCH_RISCV_FOUND)
  set(ARCHDIR "arch/riscv")
endif()

if(NOT CV_DISABLE_OPTIMIZATION)
  if(BASEARCH_ARM_FOUND)
    add_definitions(-DARM_FEATURES)
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      if("${ARCH}" MATCHES "aarch64")
        check_c_source_compiles(
          "#include <sys/auxv.h>
          int main() {
              return (getauxval(AT_HWCAP) & HWCAP_CRC32);
          }"
          ARM_AUXV_HAS_CRC32
        )
        if(ARM_AUXV_HAS_CRC32)
          add_definitions(-DARM_AUXV_HAS_CRC32)
        else()
          message(STATUS "HWCAP_CRC32 not present in sys/auxv.h; cannot detect support at runtime.")
        endif()
      else()
        check_c_source_compiles(
          "#include <sys/auxv.h>
          int main() {
              return (getauxval(AT_HWCAP2) & HWCAP2_CRC32);
          }"
          ARM_AUXV_HAS_CRC32
        )
        if(ARM_AUXV_HAS_CRC32)
          add_definitions(-DARM_AUXV_HAS_CRC32)
        else()
          check_c_source_compiles(
            "#include <sys/auxv.h>
            #include <asm/hwcap.h>
            int main() {
                return (getauxval(AT_HWCAP2) & HWCAP2_CRC32);
            }"
            ARM_HWCAP_HAS_CRC32
          )
          if(ARM_HWCAP_HAS_CRC32)
            add_definitions(-DARM_AUXV_HAS_CRC32 -DARM_ASM_HWCAP)
          else()
            message(STATUS "HWCAP2_CRC32 not present in sys/auxv.h; cannot detect support at runtime.")
          endif()
        endif()
        check_c_source_compiles(
          "#include <sys/auxv.h>
          int main() {
            return (getauxval(AT_HWCAP) & HWCAP_ARM_NEON);
          }"
          ARM_AUXV_HAS_NEON
        )
        if(ARM_AUXV_HAS_NEON)
          add_definitions(-DARM_AUXV_HAS_NEON)
        else()
          check_c_source_compiles(
            "#include <sys/auxv.h>
            int main() {
              return (getauxval(AT_HWCAP) & HWCAP_NEON);
            }"
            ARM_AUXV_HAS_NEON
          )
          if (ARM_AUXV_HAS_NEON)
            add_definitions(-DARM_AUXV_HAS_NEON)
          else()
            message(STATUS "Neither HWCAP_ARM_NEON or HWCAP_NEON present in sys/auxv.h; cannot detect support at runtime.")
          endif()
        endif()
      endif()
    endif()
    list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/arm_functions.h)
    list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/arm_features.h)
    list(APPEND ZLIB_ARCH_SRCS ${ARCHDIR}/arm_features.c)
    if(WITH_ACLE)
      check_acle_compiler_flag()
      if(HAVE_ACLE_FLAG)
        add_definitions(-DARM_ACLE)
        set(ACLE_SRCS ${ARCHDIR}/crc32_acle.c)
        set_property(SOURCE ${ACLE_SRCS} PROPERTY COMPILE_FLAGS "${ACLEFLAG} ${NOLTOFLAG}")
        list(APPEND ZLIB_ARCH_SRCS ${ACLE_SRCS})
      else()
        set(WITH_ACLE OFF)
      endif()
    else()
      set(WITH_ACLE OFF)
    endif()
    if(WITH_NEON)
      check_neon_compiler_flag()
      if(NEON_AVAILABLE)
        add_definitions(-DARM_NEON)
        set(NEON_SRCS ${ARCHDIR}/adler32_neon.c ${ARCHDIR}/chunkset_neon.c
          ${ARCHDIR}/compare256_neon.c ${ARCHDIR}/slide_hash_neon.c)
        list(APPEND ZLIB_ARCH_SRCS ${NEON_SRCS})
        set_property(SOURCE ${NEON_SRCS} PROPERTY COMPILE_FLAGS "${NEONFLAG} ${NOLTOFLAG}")
        if(MSVC)
          add_definitions(-D__ARM_NEON__)
        endif()
        check_neon_ld4_intrinsics()
        if(NEON_HAS_LD4)
          add_definitions(-DARM_NEON_HASLD4)
        endif()
      else()
        set(WITH_NEON OFF)
      endif()
    endif()
    if(WITH_ARMV6)
      check_armv6_compiler_flag()
      if(HAVE_ARMV6_INLINE_ASM OR HAVE_ARMV6_INTRIN)
        add_definitions(-DARM_SIMD)
        set(ARMV6_SRCS ${ARCHDIR}/slide_hash_armv6.c)
        set_property(SOURCE ${ARMV6_SRCS} PROPERTY COMPILE_FLAGS "${ARMV6FLAG} ${NOLTOFLAG}")
        list(APPEND ZLIB_ARCH_SRCS ${ARMV6_SRCS})
        if(HAVE_ARMV6_INTRIN)
          add_definitions(-DARM_SIMD_INTRIN)
        endif()
      else()
        set(WITH_ARMV6 OFF)
      endif()
    else()
      set(WITH_ARMV6 OFF)
    endif()
  endif()
  if(BASEARCH_PPC_FOUND)
    # Common arch detection code
    if(WITH_ALTIVEC)
      check_ppc_intrinsics()
    endif()
    if(WITH_POWER8)
      check_power8_intrinsics()
    endif()
    if(WITH_POWER9)
      check_power9_intrinsics()
    endif()
    if(POWER8_NEED_AUXVEC_H OR POWER9_NEED_AUXVEC_H)
      add_definitions(-DPOWER_NEED_AUXVEC_H)
    endif()
    if(HAVE_POWER8_INTRIN OR HAVE_POWER9_INTRIN)
      add_definitions(-DPOWER_FEATURES)
      endif()
    if(HAVE_VMX OR HAVE_POWER8_INTRIN OR HAVE_POWER9_INTRIN)
      list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/power_functions.h)
      list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/power_features.h)
      list(APPEND ZLIB_ARCH_SRCS ${ARCHDIR}/power_features.c)
    endif()
    # VMX specific options and files
    if(WITH_ALTIVEC)
      if(HAVE_VMX)
        add_definitions(-DPPC_FEATURES)
        if(HAVE_ALTIVEC)
          add_definitions(-DPPC_VMX)
          set(PPC_SRCS ${ARCHDIR}/adler32_vmx.c ${ARCHDIR}/slide_hash_vmx.c)
          list(APPEND ZLIB_ARCH_SRCS ${PPC_SRCS})
          set_property(SOURCE ${PPC_SRCS} PROPERTY COMPILE_FLAGS "${PPCFLAGS}")
        else()
          set(WITH_ALTIVEC OFF)
        endif()
      endif()
    endif()
    # Power8 specific options and files
    if(WITH_POWER8)
      if(HAVE_POWER8_INTRIN)
        add_definitions(-DPOWER8_VSX)
        set(POWER8_SRCS ${ARCHDIR}/adler32_power8.c ${ARCHDIR}/chunkset_power8.c ${ARCHDIR}/slide_hash_power8.c)
        if("${ARCH}" MATCHES "powerpc64(le)?")
          add_definitions(-DPOWER8_VSX_CRC32)
          list(APPEND POWER8_SRCS ${ARCHDIR}/crc32_power8.c)
        endif()
        list(APPEND ZLIB_ARCH_SRCS ${POWER8_SRCS})
        set_property(SOURCE ${POWER8_SRCS} PROPERTY COMPILE_FLAGS "${POWER8FLAG} ${NOLTOFLAG}")
      else()
        set(WITH_POWER8 OFF)
      endif()
    endif()
    # Power9 specific options and files
    if(WITH_POWER9)
      if(HAVE_POWER9_INTRIN)
        add_definitions(-DPOWER9)
        set(POWER9_SRCS ${ARCHDIR}/compare256_power9.c)
        list(APPEND ZLIB_ARCH_SRCS ${POWER9_SRCS})
        set_property(SOURCE ${POWER9_SRCS} PROPERTY COMPILE_FLAGS "${POWER9FLAG} ${NOLTOFLAG}")
      else()
        set(WITH_POWER9 OFF)
      endif()
    endif()
  endif()
  if(BASEARCH_RISCV_FOUND)
    if(WITH_RVV)
      check_rvv_intrinsics()
      if(HAVE_RVV_INTRIN)
        add_definitions(-DRISCV_FEATURES)
        add_definitions(-DRISCV_RVV)
        list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/riscv_functions.h)
        list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/riscv_features.h)
        list(APPEND ZLIB_ARCH_SRCS ${ARCHDIR}/riscv_features.c)
        # FIXME: we will not set compile flags for riscv_features.c when
        # the kernels update hwcap or hwprobe for riscv
        set(RVV_SRCS ${ARCHDIR}/adler32_rvv.c ${ARCHDIR}/chunkset_rvv.c ${ARCHDIR}/compare256_rvv.c ${ARCHDIR}/slide_hash_rvv.c)
        list(APPEND RVV_SRCS ${ARCHDIR}/riscv_features.c)
        list(APPEND ZLIB_ARCH_SRCS ${RVV_SRCS})
        set_property(SOURCE ${RVV_SRCS} PROPERTY COMPILE_FLAGS "${RISCVFLAG} ${NOLTOFLAG}")
      else()
        set(WITH_RVV OFF)
      endif()
    endif()
  endif()
  if(BASEARCH_X86_FOUND)
    add_definitions(-DX86_FEATURES)
    list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/x86_functions.h)
    list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/x86_features.h)
    list(APPEND ZLIB_ARCH_SRCS ${ARCHDIR}/x86_features.c)
    if(MSVC)
      list(APPEND ZLIB_ARCH_HDRS fallback_builtins.h)
    endif()
    check_xsave_intrinsics()
    if(HAVE_XSAVE_INTRIN)
      set_property(SOURCE ${ARCHDIR}/x86_features.c PROPERTY COMPILE_FLAGS "${XSAVEFLAG}")
      if(NOT (CMAKE_C_COMPILER_ID MATCHES "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 8.2))
        add_definitions(-DX86_HAVE_XSAVE_INTRIN)
      endif()
    endif()
    if(WITH_SSE2)
      check_sse2_intrinsics()
      if(HAVE_SSE2_INTRIN)
        add_definitions(-DX86_SSE2)
        set(SSE2_SRCS ${ARCHDIR}/chunkset_sse2.c ${ARCHDIR}/compare256_sse2.c ${ARCHDIR}/slide_hash_sse2.c)
        list(APPEND ZLIB_ARCH_SRCS ${SSE2_SRCS})
        if(NOT ${ARCH} MATCHES "x86_64")
          set_property(SOURCE ${SSE2_SRCS} PROPERTY COMPILE_FLAGS "${SSE2FLAG} ${NOLTOFLAG}")
        endif()
      else()
        set(WITH_SSE2 OFF)
      endif()
    endif()
    if(WITH_SSSE3)
      check_ssse3_intrinsics()
      if(HAVE_SSSE3_INTRIN AND WITH_SSE2)
        add_definitions(-DX86_SSSE3)
        set(SSSE3_SRCS ${ARCHDIR}/adler32_ssse3.c ${ARCHDIR}/chunkset_ssse3.c)
        list(APPEND ZLIB_ARCH_SRCS ${SSSE3_SRCS})
        set_property(SOURCE ${SSSE3_SRCS} PROPERTY COMPILE_FLAGS "${SSSE3FLAG} ${NOLTOFLAG}")
      else()
        set(WITH_SSSE3 OFF)
      endif()
    endif()
    if(WITH_SSE42)
      check_sse42_intrinsics()
      if(HAVE_SSE42_INTRIN AND WITH_SSSE3)
        add_definitions(-DX86_SSE42)
        set(SSE42_SRCS ${ARCHDIR}/adler32_sse42.c)
        list(APPEND ZLIB_ARCH_SRCS ${SSE42_SRCS})
        set_property(SOURCE ${SSE42_SRCS} PROPERTY COMPILE_FLAGS "${SSE42FLAG} ${NOLTOFLAG}")
      else()
        set(WITH_SSE42 OFF)
      endif()
    endif()
    if(WITH_PCLMULQDQ)
      check_pclmulqdq_intrinsics()
      if(HAVE_PCLMULQDQ_INTRIN AND WITH_SSE42)
        add_definitions(-DX86_PCLMULQDQ_CRC)
        set(PCLMULQDQ_SRCS ${ARCHDIR}/crc32_pclmulqdq.c)
        list(APPEND ZLIB_ARCH_SRCS ${PCLMULQDQ_SRCS})
        set_property(SOURCE ${PCLMULQDQ_SRCS} PROPERTY COMPILE_FLAGS "${SSE42FLAG} ${PCLMULFLAG} ${NOLTOFLAG}")
      else()
        set(WITH_PCLMULQDQ OFF)
      endif()
    endif()
    if(WITH_AVX2)
      check_avx2_intrinsics()
      if(HAVE_AVX2_INTRIN AND WITH_SSE42)
        add_definitions(-DX86_AVX2)
        set(AVX2_SRCS ${ARCHDIR}/slide_hash_avx2.c)
        list(APPEND AVX2_SRCS ${ARCHDIR}/chunkset_avx2.c)
        list(APPEND AVX2_SRCS ${ARCHDIR}/compare256_avx2.c)
        list(APPEND AVX2_SRCS ${ARCHDIR}/adler32_avx2.c)
        list(APPEND ZLIB_ARCH_SRCS ${AVX2_SRCS})
        set_property(SOURCE ${AVX2_SRCS} PROPERTY COMPILE_FLAGS "${AVX2FLAG} ${NOLTOFLAG}")
      else()
        set(WITH_AVX2 OFF)
      endif()
    endif()
    if(WITH_AVX512)
      check_avx512_intrinsics()
      if(HAVE_AVX512_INTRIN AND WITH_AVX2)
        add_definitions(-DX86_AVX512)
        list(APPEND AVX512_SRCS ${ARCHDIR}/adler32_avx512.c)
        list(APPEND ZLIB_ARCH_SRCS ${AVX512_SRCS})
        list(APPEND ZLIB_ARCH_HDRS ${ARCHDIR}/adler32_avx512_p.h)
        set_property(SOURCE ${AVX512_SRCS} PROPERTY COMPILE_FLAGS "${AVX512FLAG} ${NOLTOFLAG}")
      else()
        set(WITH_AVX512 OFF)
      endif()
    endif()
    if(WITH_AVX512VNNI)
      check_avx512vnni_intrinsics()
      if(HAVE_AVX512VNNI_INTRIN AND WITH_AVX2)
        add_definitions(-DX86_AVX512VNNI)
        list(APPEND AVX512VNNI_SRCS ${ARCHDIR}/adler32_avx512_vnni.c)
        list(APPEND ZLIB_ARCH_SRCS ${AVX512VNNI_SRCS})
        set_property(SOURCE ${AVX512VNNI_SRCS} PROPERTY COMPILE_FLAGS "${AVX512VNNIFLAG} ${NOLTOFLAG}")
      else()
        set(WITH_AVX512VNNI OFF)
      endif()
    endif()
    if(WITH_VPCLMULQDQ)
      check_vpclmulqdq_intrinsics()
      if(HAVE_VPCLMULQDQ_INTRIN AND WITH_PCLMULQDQ AND WITH_AVX512)
        add_definitions(-DX86_VPCLMULQDQ_CRC)
        set(VPCLMULQDQ_SRCS ${ARCHDIR}/crc32_vpclmulqdq.c)
        list(APPEND ZLIB_ARCH_SRCS ${VPCLMULQDQ_SRCS})
        set_property(SOURCE ${VPCLMULQDQ_SRCS} PROPERTY COMPILE_FLAGS "${PCLMULFLAG} ${VPCLMULFLAG} ${AVX512FLAG} ${NOLTOFLAG}")
      else()
        set(WITH_VPCLMULQDQ OFF)
      endif()
    endif()
  endif()
endif()

#============================================================================
# zconf.h
#============================================================================

macro(generate_cmakein input output)
  file(REMOVE ${output})
  file(STRINGS ${input} _lines)
  foreach(_line IN LISTS _lines)
    string(REGEX REPLACE "#ifdef HAVE_UNISTD_H.*" "@ZCONF_UNISTD_LINE@" _line "${_line}")
    string(REGEX REPLACE "#ifdef NEED_PTRDIFF_T.*" "@ZCONF_PTRDIFF_LINE@" _line "${_line}")
    if(NEED_PTRDIFF_T)
      string(REGEX REPLACE "typedef PTRDIFF_TYPE" "typedef @PTRDIFF_TYPE@" _line "${_line}")
    endif()
    file(APPEND ${output} "${_line}\n")
  endforeach()
endmacro(generate_cmakein)

generate_cmakein( ${CMAKE_CURRENT_SOURCE_DIR}/zconf.h.in ${CMAKE_CURRENT_BINARY_DIR}/zconf.h.cmakein )

#============================================================================
# zlib
#============================================================================

set(ZLIB_PUBLIC_HDRS
    ${CMAKE_CURRENT_BINARY_DIR}/zconf.h
    ${CMAKE_CURRENT_BINARY_DIR}/zlib_name_mangling.h
    ${CMAKE_CURRENT_BINARY_DIR}/zlib.h
)
set(ZLIB_PRIVATE_HDRS
  adler32_p.h
  chunkset_tpl.h
  compare256_rle.h
  arch_functions.h
  crc32_braid_p.h
  crc32_braid_comb_p.h
  crc32_braid_tbl.h
  deflate.h
  deflate_p.h
  functable.h
  inffast_tpl.h
  inffixed_tbl.h
  inflate.h
  inflate_p.h
  inftrees.h
  insert_string_tpl.h
  match_tpl.h
  trees.h
  trees_emit.h
  trees_tbl.h
  zbuild.h
  zendian.h
  zutil.h
)
set(ZLIB_SRCS
  arch/generic/adler32_c.c
  arch/generic/adler32_fold_c.c
  arch/generic/chunkset_c.c
  arch/generic/compare256_c.c
  arch/generic/crc32_braid_c.c
  arch/generic/crc32_fold_c.c
  arch/generic/slide_hash_c.c
  adler32.c
  compress.c
  crc32.c
  crc32_braid_comb.c
  deflate.c
  deflate_fast.c
  deflate_huff.c
  deflate_medium.c
  deflate_quick.c
  deflate_rle.c
  deflate_slow.c
  deflate_stored.c
  functable.c
  infback.c
  inflate.c
  inftrees.c
  insert_string.c
  insert_string_roll.c
  trees.c
  uncompr.c
  zutil.c
)

list(APPEND ZLIB_PRIVATE_HDRS cpu_features.h)
list(APPEND ZLIB_SRCS cpu_features.c)

set(ZLIB_GZFILE_PRIVATE_HDRS
    gzguts.h
)
set(ZLIB_GZFILE_SRCS
    gzlib.c
    ${CMAKE_CURRENT_BINARY_DIR}/gzread.c
    gzwrite.c
)

set(ZLIB_ALL_SRCS ${ZLIB_SRCS} ${ZLIB_ARCH_HDRS} ${ZLIB_ARCH_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS})
list(APPEND ZLIB_ALL_SRCS ${ZLIB_GZFILE_PRIVATE_HDRS} ${ZLIB_GZFILE_SRCS})

add_library(zlib STATIC ${ZLIB_ALL_SRCS})

target_include_directories(zlib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}$<SEMICOLON>${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

if(HAVE_UNISTD_H)
  SET(ZCONF_UNISTD_LINE "#if 1    /* was set to #if 1 by configure/cmake/etc */")
else()
  SET(ZCONF_UNISTD_LINE "#if 0    /* was set to #if 0 by configure/cmake/etc */")
endif()
if(NEED_PTRDIFF_T)
  SET(ZCONF_PTRDIFF_LINE "#if 1    /* was set to #if 1 by configure/cmake/etc */")
else()
  SET(ZCONF_PTRDIFF_LINE "#ifdef NEED_PTRDIFF_T    /* may be set to #if 1 by configure/cmake/etc */")
endif()

configure_file(${CMAKE_CURRENT_BINARY_DIR}/zconf.h.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/zconf.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/zlib.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/zlib.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gzread.c.in
  ${CMAKE_CURRENT_BINARY_DIR}/gzread.c @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/zlib_name_mangling.h.empty
  ${CMAKE_CURRENT_BINARY_DIR}/zlib_name_mangling${SUFFIX}.h COPYONLY)

ocv_warnings_disable(CMAKE_C_FLAGS -Wmissing-prototypes
  -Wundef
  -Wmissing-declarations
  /wd4244
  /wd4334
  /wd4819
)

set_target_properties(${ZLIB_LIBRARY} PROPERTIES
  OUTPUT_NAME ${ZLIB_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${ZLIB_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${ZLIB_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${ZLIB_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${ZLIB_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(${ZLIB_LIBRARY} LICENSE.md)
