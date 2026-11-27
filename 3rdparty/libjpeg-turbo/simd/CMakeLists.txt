macro(simd_fail message)
  if(REQUIRE_SIMD)
    message(FATAL_ERROR "${message}.")
  else()
    message(STATUS "${message}.  Performance will suffer.")
    set(WITH_SIMD 0)
    set(WITH_SIMD 0 PARENT_SCOPE)
    set(SIMD_ARCHITECTURE NONE PARENT_SCOPE)
  endif()
endmacro()


option(WITH_SIMDE "Use SIMD Everywhere or Emscripten to translate Arm/Neon intrinsics into SIMD intrinsics for other platforms (Experimental)"
  FALSE)


###############################################################################
# x86[-64] (NASM)
###############################################################################

if((CPU_TYPE STREQUAL "x86_64" OR CPU_TYPE STREQUAL "i386") AND NOT WITH_SIMDE)

string(TOUPPER ${CPU_TYPE} SIMD_ARCHITECTURE)
set(SIMD_ARCHITECTURE ${SIMD_ARCHITECTURE} PARENT_SCOPE)

set(CMAKE_ASM_NASM_FLAGS_DEBUG_INIT "-g")
set(CMAKE_ASM_NASM_FLAGS_RELWITHDEBINFO_INIT "-g")

# Allow the location of the NASM executable to be specified using the ASM_NASM
# environment variable.  This should happen automatically, but unfortunately
# enable_language(ASM_NASM) doesn't parse the ASM_NASM environment variable
# until after CMAKE_ASM_NASM_COMPILER has been populated with the results of
# searching for NASM or Yasm in the PATH.
if(NOT DEFINED CMAKE_ASM_NASM_COMPILER AND DEFINED ENV{ASM_NASM})
  set(CMAKE_ASM_NASM_COMPILER $ENV{ASM_NASM})
endif()

if(CPU_TYPE STREQUAL "x86_64")
  if(CYGWIN)
    set(CMAKE_ASM_NASM_OBJECT_FORMAT win64)
  endif()
  if(CMAKE_C_COMPILER_ABI MATCHES "ELF X32")
    set(CMAKE_ASM_NASM_OBJECT_FORMAT elfx32)
  endif()
elseif(CPU_TYPE STREQUAL "i386")
  if(BORLAND)
    set(CMAKE_ASM_NASM_OBJECT_FORMAT obj)
  elseif(CYGWIN)
    set(CMAKE_ASM_NASM_OBJECT_FORMAT win32)
  endif()
endif()

if(NOT REQUIRE_SIMD)
  include(CheckLanguage)
  check_language(ASM_NASM)
  if(NOT CMAKE_ASM_NASM_COMPILER)
    simd_fail("SIMD extensions disabled: could not find NASM compiler")
    return()
  endif()
endif()
enable_language(ASM_NASM)
message(STATUS "CMAKE_ASM_NASM_COMPILER = ${CMAKE_ASM_NASM_COMPILER}")

if(CMAKE_ASM_NASM_OBJECT_FORMAT MATCHES "^macho")
  set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -DMACHO")
elseif(CMAKE_ASM_NASM_OBJECT_FORMAT MATCHES "^elf")
  set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -DELF")
  set(CMAKE_ASM_NASM_DEBUG_FORMAT "dwarf2")
endif()
if(CPU_TYPE STREQUAL "x86_64")
  if(WIN32 OR CYGWIN)
    set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -DWIN64")
  endif()
  set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -D__x86_64__")
elseif(CPU_TYPE STREQUAL "i386")
  if(BORLAND)
    set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -DOBJ32")
  elseif(WIN32 OR CYGWIN)
    set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -DWIN32")
  endif()
endif()

message(STATUS "CMAKE_ASM_NASM_OBJECT_FORMAT = ${CMAKE_ASM_NASM_OBJECT_FORMAT}")

if(NOT CMAKE_ASM_NASM_OBJECT_FORMAT)
  simd_fail("SIMD extensions disabled: could not determine NASM object format")
  return()
endif()

get_filename_component(CMAKE_ASM_NASM_COMPILER_TYPE
  "${CMAKE_ASM_NASM_COMPILER}" NAME_WE)
if(CMAKE_ASM_NASM_COMPILER_TYPE MATCHES "yasm")
  foreach(var CMAKE_ASM_NASM_FLAGS_DEBUG CMAKE_ASM_NASM_FLAGS_RELWITHDEBINFO)
    if(${var} STREQUAL "-g")
      if(CMAKE_ASM_NASM_DEBUG_FORMAT)
        set_property(CACHE ${var} PROPERTY VALUE "-g ${CMAKE_ASM_NASM_DEBUG_FORMAT}")
      else()
        set_property(CACHE ${var} PROPERTY VALUE "")
      endif()
    endif()
  endforeach()
endif()

if(NOT WIN32 AND (CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED))
  set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -DPIC")
endif()

if(CPU_TYPE STREQUAL "x86_64" AND CMAKE_ASM_NASM_OBJECT_FORMAT MATCHES "^elf")
  check_c_source_compiles("
    #if (__CET__ & 3) == 0
    #error \"CET not enabled\"
    #endif
    int main(void) { return 0; }" HAVE_CET)

  if(HAVE_CET)
    set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -D__CET__")
  endif()
endif()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UC)
set(EFFECTIVE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} ${CMAKE_ASM_NASM_FLAGS_${CMAKE_BUILD_TYPE_UC}}")
message(STATUS "CMAKE_ASM_NASM_FLAGS = ${EFFECTIVE_ASM_NASM_FLAGS}")

set(CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS} -I\"${CMAKE_CURRENT_SOURCE_DIR}/nasm/\" -I\"${CMAKE_CURRENT_SOURCE_DIR}/${CPU_TYPE}/\"")

set(GREP grep)
if(CMAKE_SYSTEM_NAME STREQUAL "SunOS")
  set(GREP ggrep)
endif()
add_custom_target(jsimdcfg COMMAND
  ${CMAKE_C_COMPILER} -Wall -Werror -E -I${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/nasm/jsimdcfg.inc.h |
  ${GREP} -E '^[\;%]|^\ %' | sed 's%_cpp_protection_%%' |
  sed 's@% define@%define@g' >${CMAKE_CURRENT_SOURCE_DIR}/nasm/jsimdcfg.inc)

if(CPU_TYPE STREQUAL "x86_64")
  set(SIMD_SOURCES x86_64/jsimdcpu.asm x86_64/jfdctflt-sse.asm
    x86_64/jccolor-sse2.asm x86_64/jcgray-sse2.asm x86_64/jchuff-sse2.asm
    x86_64/jcphuff-sse2.asm x86_64/jcsample-sse2.asm x86_64/jdcolor-sse2.asm
    x86_64/jdmerge-sse2.asm x86_64/jdsample-sse2.asm x86_64/jfdctfst-sse2.asm
    x86_64/jfdctint-sse2.asm x86_64/jidctflt-sse2.asm x86_64/jidctfst-sse2.asm
    x86_64/jidctint-sse2.asm x86_64/jidctred-sse2.asm x86_64/jquantf-sse2.asm
    x86_64/jquanti-sse2.asm
    x86_64/jccolor-avx2.asm x86_64/jcgray-avx2.asm x86_64/jcsample-avx2.asm
    x86_64/jdcolor-avx2.asm x86_64/jdmerge-avx2.asm x86_64/jdsample-avx2.asm
    x86_64/jfdctint-avx2.asm x86_64/jidctint-avx2.asm x86_64/jquanti-avx2.asm)
else()
  set(SIMD_SOURCES i386/jsimdcpu.asm i386/jfdctflt-3dn.asm
    i386/jidctflt-3dn.asm i386/jquantf-3dn.asm
    i386/jccolor-mmx.asm i386/jcgray-mmx.asm i386/jcsample-mmx.asm
    i386/jdcolor-mmx.asm i386/jdmerge-mmx.asm i386/jdsample-mmx.asm
    i386/jfdctfst-mmx.asm i386/jfdctint-mmx.asm i386/jidctfst-mmx.asm
    i386/jidctint-mmx.asm i386/jidctred-mmx.asm i386/jquanti-mmx.asm
    i386/jfdctflt-sse.asm i386/jidctflt-sse.asm i386/jquantf-sse.asm
    i386/jccolor-sse2.asm i386/jcgray-sse2.asm i386/jchuff-sse2.asm
    i386/jcphuff-sse2.asm i386/jcsample-sse2.asm i386/jdcolor-sse2.asm
    i386/jdmerge-sse2.asm i386/jdsample-sse2.asm i386/jfdctfst-sse2.asm
    i386/jfdctint-sse2.asm i386/jidctflt-sse2.asm i386/jidctfst-sse2.asm
    i386/jidctint-sse2.asm i386/jidctred-sse2.asm i386/jquantf-sse2.asm
    i386/jquanti-sse2.asm
    i386/jccolor-avx2.asm i386/jcgray-avx2.asm i386/jcsample-avx2.asm
    i386/jdcolor-avx2.asm i386/jdmerge-avx2.asm i386/jdsample-avx2.asm
    i386/jfdctint-avx2.asm i386/jidctint-avx2.asm i386/jquanti-avx2.asm)
endif()

if(MSVC_IDE)
  set(OBJDIR "${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)")
  string(REGEX REPLACE " " ";" CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS}")
elseif(XCODE)
  set(OBJDIR "${CMAKE_CURRENT_BINARY_DIR}")
  string(REGEX REPLACE " " ";" CMAKE_ASM_NASM_FLAGS "${CMAKE_ASM_NASM_FLAGS}")
endif()

file(GLOB INC_FILES nasm/*.inc)

foreach(file ${SIMD_SOURCES})
  set(OBJECT_DEPENDS "")
  if(${file} MATCHES jccolor)
    string(REGEX REPLACE "jccolor" "jccolext" DEPFILE ${file})
    set(OBJECT_DEPENDS ${OBJECT_DEPENDS}
      ${CMAKE_CURRENT_SOURCE_DIR}/${DEPFILE})
  endif()
  if(${file} MATCHES jcgray)
    string(REGEX REPLACE "jcgray" "jcgryext" DEPFILE ${file})
    set(OBJECT_DEPENDS ${OBJECT_DEPENDS}
      ${CMAKE_CURRENT_SOURCE_DIR}/${DEPFILE})
  endif()
  if(${file} MATCHES jdcolor)
    string(REGEX REPLACE "jdcolor" "jdcolext" DEPFILE ${file})
    set(OBJECT_DEPENDS ${OBJECT_DEPENDS}
      ${CMAKE_CURRENT_SOURCE_DIR}/${DEPFILE})
  endif()
  if(${file} MATCHES jdmerge)
    string(REGEX REPLACE "jdmerge" "jdmrgext" DEPFILE ${file})
    set(OBJECT_DEPENDS ${OBJECT_DEPENDS}
      ${CMAKE_CURRENT_SOURCE_DIR}/${DEPFILE})
  endif()
  set(OBJECT_DEPENDS ${OBJECT_DEPENDS} ${INC_FILES})
  if(MSVC_IDE OR XCODE)
    # The CMake Visual Studio generators do not work properly with the ASM_NASM
    # language, so we have to go rogue here and use a custom command like we
    # did in prior versions of libjpeg-turbo.  (This is why we can't have nice
    # things.)
    string(REGEX REPLACE "${CPU_TYPE}/" "" filename ${file})
    set(SIMD_OBJ ${OBJDIR}/${filename}${CMAKE_C_OUTPUT_EXTENSION})
    add_custom_command(OUTPUT ${SIMD_OBJ} DEPENDS ${file} ${OBJECT_DEPENDS}
      COMMAND ${CMAKE_ASM_NASM_COMPILER} -f${CMAKE_ASM_NASM_OBJECT_FORMAT}
        ${CMAKE_ASM_NASM_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${file}
        -o${SIMD_OBJ})
    set(SIMD_OBJS ${SIMD_OBJS} ${SIMD_OBJ})
  else()
    set_source_files_properties(${file} PROPERTIES OBJECT_DEPENDS
      "${OBJECT_DEPENDS}")
  endif()
endforeach()

if(MSVC_IDE OR XCODE)
  set(SIMD_OBJS ${SIMD_OBJS} PARENT_SCOPE)
  add_library(simd OBJECT jsimd.c)
  add_custom_target(simd-objs DEPENDS ${SIMD_OBJS})
  add_dependencies(simd simd-objs)
  set_target_properties(simd PROPERTIES FOLDER "3rdparty/jpeg")
  set_target_properties(simd-objs PROPERTIES FOLDER "3rdparty/jpeg")
  set_target_properties(jsimdcfg PROPERTIES FOLDER "3rdparty/jpeg")
else()
  add_library(simd OBJECT ${SIMD_SOURCES} jsimd.c)
endif()
if(NOT WIN32 AND (CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED))
  set_target_properties(simd PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()


###############################################################################
# Arm (Intrinsics)
###############################################################################

elseif(CPU_TYPE STREQUAL "arm64" OR CPU_TYPE STREQUAL "arm" OR WITH_SIMDE)

string(TOUPPER ${CPU_TYPE} SIMD_ARCHITECTURE)
set(SIMD_ARCHITECTURE ${SIMD_ARCHITECTURE} PARENT_SCOPE)

if(WITH_SIMDE)

if(NOT DEFINED HAVE_VLD1_S16_X3)
  set(HAVE_VLD1_S16_X3 1)
endif()
if(NOT DEFINED HAVE_VLD1_U16_X2)
  set(HAVE_VLD1_U16_X2 1)
endif()
if(NOT DEFINED HAVE_VLD1Q_U8_X4)
  set(HAVE_VLD1Q_U8_X4 0)
endif()

if(BITS EQUAL 64)
  set(SIMD_ARCHITECTURE ARM64 PARENT_SCOPE)
else()
  set(SIMD_ARCHITECTURE ARM PARENT_SCOPE)
endif()

else() # WITH_SIMDE

# If Neon instructions are not explicitly enabled at compile time (e.g. using
# -mfpu=neon) with an AArch32 Linux or Android build, then the AArch32 SIMD
# dispatcher will parse /proc/cpuinfo to determine whether the Neon SIMD
# extensions can be enabled at run time.  In order to support all AArch32 CPUs
# using the same code base, i.e. to support run-time FPU and Neon
# auto-detection, it is necessary to compile the scalar C source code using
# -mfloat-abi=soft (which is usually the default) but compile the intrinsics
# implementation of the Neon SIMD extensions using -mfloat-abi=softfp.  The
# following test determines whether -mfloat-abi=softfp should be explicitly
# added to the compile flags for the intrinsics implementation of the Neon SIMD
# extensions.
if(BITS EQUAL 32)
  check_c_source_compiles("
    #if defined(__ARM_NEON__) || (!defined(__linux__) && !defined(ANDROID) && !defined(__ANDROID__))
    #error \"Neon run-time auto-detection will not be used\"
    #endif
    #if __ARM_PCS_VFP == 1
    #error \"float ABI = hard\"
    #endif
    #if __SOFTFP__ != 1
    #error \"float ABI = softfp\"
    #endif
    int main(void) { return 0; }" NEED_SOFTFP_FOR_INTRINSICS)
  if(NEED_SOFTFP_FOR_INTRINSICS)
    set(SOFTFP_FLAG -mfloat-abi=softfp)
  endif()
endif()

if(BITS EQUAL 32)
  set(CMAKE_REQUIRED_FLAGS "-mfpu=neon ${SOFTFP_FLAG}")
  check_c_source_compiles("
    #include <arm_neon.h>
    int main(int argc, char **argv) {
      uint16x8_t input = vdupq_n_u16((uint16_t)argc);
      uint8x8_t output = vmovn_u16(input);
      return (int)output[0];
    }" HAVE_NEON)
  if(NOT HAVE_NEON)
    simd_fail("SIMD extensions not available for this architecture")
    return()
  endif()
endif()
check_c_source_compiles("
  #include <arm_neon.h>
  int main(int argc, char **argv) {
    int16_t input[12];
    int16x4x3_t output;
    int i;
    for (i = 0; i < 12; i++) input[i] = (int16_t)argc;
    output = vld1_s16_x3(input);
    vst3_s16(input, output);
    return (int)input[0];
  }" HAVE_VLD1_S16_X3)
check_c_source_compiles("
  #include <arm_neon.h>
  int main(int argc, char **argv) {
    uint16_t input[8];
    uint16x4x2_t output;
    int i;
    for (i = 0; i < 8; i++) input[i] = (uint16_t)argc;
    output = vld1_u16_x2(input);
    vst2_u16(input, output);
    return (int)input[0];
  }" HAVE_VLD1_U16_X2)
check_c_source_compiles("
  #include <arm_neon.h>
  int main(int argc, char **argv) {
    uint8_t input[64];
    uint8x16x4_t output;
    int i;
    for (i = 0; i < 64; i++) input[i] = (uint8_t)argc;
    output = vld1q_u8_x4(input);
    vst4q_u8(input, output);
    return (int)input[0];
  }" HAVE_VLD1Q_U8_X4)
if(BITS EQUAL 32)
  unset(CMAKE_REQUIRED_FLAGS)
endif()

# GCC 11 and earlier and some older versions of Clang do not have a full or
# optimal set of Neon intrinsics.  The presence or absence of the three
# intrinsics we tested above is a reasonable proxy for this, except with GCC 10
# and 11.
if(NOT HAVE_VLD1_S16_X3 OR NOT HAVE_VLD1_U16_X2 OR NOT HAVE_VLD1Q_U8_X4 OR
  (CMAKE_COMPILER_IS_GNUCC AND CMAKE_C_COMPILER_VERSION VERSION_LESS 12.0.0))
  message(WARNING "The compiler's Arm Neon intrinsics implementation is incomplete or suboptimal.  Performance will suffer.  Use GCC 12 or later or Clang for full performance.")
endif()

endif() # WITH_SIMDE

configure_file(arm/neon-compat.h.in arm/neon-compat.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/arm)

set(SIMD_SOURCES arm/jccolor-neon.c arm/jcgray-neon.c
  arm/aarch${BITS}/jchuff-neon.c arm/jcphuff-neon.c arm/jcsample-neon.c
  arm/jdcolor-neon.c arm/jdmerge-neon.c arm/jdsample-neon.c arm/jfdctfst-neon.c
  arm/jfdctint-neon.c arm/jidctfst-neon.c arm/jidctint-neon.c
  arm/jidctred-neon.c arm/jquanti-neon.c)
if(BITS EQUAL 32)
  set_source_files_properties(${SIMD_SOURCES} COMPILE_FLAGS "-mfpu=neon ${SOFTFP_FLAG}")
endif()

if(UNIX AND BITS EQUAL 32)
  include(CheckSymbolExists)
  check_symbol_exists(getauxval sys/auxv.h HAVE_GETAUXVAL)
  if(HAVE_GETAUXVAL)
    set_source_files_properties(arm/aarch${BITS}/jsimdcpu.c PROPERTIES
      COMPILE_DEFINITIONS HAVE_GETAUXVAL)
  endif()
  check_symbol_exists(elf_aux_info sys/auxv.h HAVE_ELF_AUX_INFO)
  if(HAVE_ELF_AUX_INFO)
    set_source_files_properties(arm/aarch${BITS}/jsimdcpu.c PROPERTIES
      COMPILE_DEFINITIONS HAVE_ELF_AUX_INFO)
  endif()
endif()

add_library(simd OBJECT ${SIMD_SOURCES} arm/aarch${BITS}/jsimdcpu.c jsimd.c)

if(CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED)
  set_target_properties(simd PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()


###############################################################################
# PowerPC (Intrinsics)
###############################################################################

elseif(CPU_TYPE STREQUAL "powerpc")

string(TOUPPER ${CPU_TYPE} SIMD_ARCHITECTURE)
set(SIMD_ARCHITECTURE ${SIMD_ARCHITECTURE} PARENT_SCOPE)

set(CMAKE_REQUIRED_FLAGS -maltivec)

check_c_source_compiles("
  #include <altivec.h>
  int main(void) {
    __vector int vi = { 0, 0, 0, 0 };
    int i[4];
    vec_st(vi, 0, i);
    return i[0];
  }" HAVE_ALTIVEC)

unset(CMAKE_REQUIRED_FLAGS)

if(NOT HAVE_ALTIVEC)
  simd_fail("SIMD extensions not available for this CPU (PowerPC SPE)")
  return()
endif()

set(SIMD_SOURCES powerpc/jccolor-altivec.c powerpc/jcgray-altivec.c
  powerpc/jcsample-altivec.c powerpc/jdcolor-altivec.c
  powerpc/jdmerge-altivec.c powerpc/jdsample-altivec.c
  powerpc/jfdctfst-altivec.c powerpc/jfdctint-altivec.c
  powerpc/jidctfst-altivec.c powerpc/jidctint-altivec.c
  powerpc/jquanti-altivec.c)

set_source_files_properties(${SIMD_SOURCES} PROPERTIES
  COMPILE_FLAGS -maltivec)

if(UNIX)
  include(CheckSymbolExists)
  check_symbol_exists(getauxval sys/auxv.h HAVE_GETAUXVAL)
  if(HAVE_GETAUXVAL)
    set_source_files_properties(powerpc/jsimdcpu.c PROPERTIES
      COMPILE_DEFINITIONS HAVE_GETAUXVAL)
  endif()
  check_symbol_exists(elf_aux_info sys/auxv.h HAVE_ELF_AUX_INFO)
  if(HAVE_ELF_AUX_INFO)
    set_source_files_properties(powerpc/jsimdcpu.c PROPERTIES
      COMPILE_DEFINITIONS HAVE_ELF_AUX_INFO)
  endif()
endif()

add_library(simd OBJECT ${SIMD_SOURCES} powerpc/jsimdcpu.c jsimd.c)

if(CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED)
  set_target_properties(simd PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()


###############################################################################
# RISC-V 64-bit (Intrinsics)
###############################################################################

elseif(CPU_TYPE STREQUAL "riscv64")

set(SIMD_ARCHITECTURE RISCV64 PARENT_SCOPE)

set(CMAKE_REQUIRED_FLAGS -march=rv64gcv)

check_c_source_compiles("
  #include <riscv_vector.h>
  int main(int argc, char **argv) {
    size_t vl = __riscv_vsetvl_e32m8(32);
    vuint32m8_t tmp0 = __riscv_vmv_v_x_u32m8(argc, vl);
    vuint16m4_t tmp1 = __riscv_vnclipu_wx_u16m4(tmp0, 1, __RISCV_VXRM_RNU, vl);
    vuint16m4_t tmp2 = __riscv_vnclipu_wx_u16m4(tmp0, 1, __RISCV_VXRM_RDN, vl);
    vuint8m2_t tmp3 = __riscv_vmv_v_x_u8m2(argc, vl);
    vuint8m2_t tmp4 = __riscv_vmv_v_x_u8m2(argc, vl);
    vuint8m2x2_t tmp5 = __riscv_vcreate_v_u8m2x2(tmp3, tmp4);
    return __riscv_vmv_x_s_u16m4_u16(tmp1) + __riscv_vmv_x_s_u16m4_u16(tmp2) +
           __riscv_vmv_x_s_u8m2_u8(__riscv_vget_v_u8m2x2_u8m2(tmp5, 0)) +
           __riscv_vmv_x_s_u8m2_u8(__riscv_vget_v_u8m2x2_u8m2(tmp5, 1));
  }" HAVE_RVV)

unset(CMAKE_REQUIRED_FLAGS)

if(NOT HAVE_RVV)
  simd_fail("SIMD extensions not available for this CPU")
  return()
endif()

set(SIMD_SOURCES riscv64/jccolor-rvv.c riscv64/jcgray-rvv.c
  riscv64/jcsample-rvv.c riscv64/jdcolor-rvv.c riscv64/jdmerge-rvv.c
  riscv64/jdsample-rvv.c riscv64/jfdctfst-rvv.c riscv64/jfdctint-rvv.c
  riscv64/jidctfst-rvv.c riscv64/jidctint-rvv.c riscv64/jquanti-rvv.c)

enable_language(ASM)

set_source_files_properties(${SIMD_SOURCES} PROPERTIES
  COMPILE_FLAGS -march=rv64gcv)

if(UNIX)
  include(CheckSymbolExists)
  check_symbol_exists(getauxval sys/auxv.h HAVE_GETAUXVAL)
  if(HAVE_GETAUXVAL)
    set_source_files_properties(riscv64/jsimdcpu.c PROPERTIES
      COMPILE_DEFINITIONS HAVE_GETAUXVAL)
  endif()
  check_symbol_exists(elf_aux_info sys/auxv.h HAVE_ELF_AUX_INFO)
  if(HAVE_ELF_AUX_INFO)
    set_source_files_properties(riscv64/jsimdcpu.c PROPERTIES
      COMPILE_DEFINITIONS HAVE_ELF_AUX_INFO)
  endif()
endif()

add_library(simd OBJECT ${SIMD_SOURCES} jsimd.c riscv64/jsimdcpu.c
  riscv64/jsimdcpu.S)

if(CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED)
  set_target_properties(simd PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()


###############################################################################
# MIPS64 (Intrinsics)
###############################################################################

elseif(CPU_TYPE STREQUAL "loongson" OR CPU_TYPE MATCHES "^mips64")

set(SIMD_ARCHITECTURE MIPS64 PARENT_SCOPE)

set(CMAKE_REQUIRED_FLAGS -Wa,-mloongson-mmi,-mloongson-ext)

check_c_source_compiles("
  #if !(defined(__mips__) && __mips_isa_rev < 6)
  #error \"Loongson MMI can't work with MIPS Release 6+\"
  #endif
  int main(void) {
    int c = 0, a = 0, b = 0;
    asm (
      \"paddb %0, %1, %2\"
      : \"=f\" (c)
      : \"f\" (a), \"f\" (b)
    );
    return c;
  }" HAVE_MMI)

unset(CMAKE_REQUIRED_FLAGS)

if(NOT HAVE_MMI)
  simd_fail("SIMD extensions not available for this CPU")
  return()
endif()

set(SIMD_SOURCES mips64/jccolor-mmi.c mips64/jcgray-mmi.c mips64/jcsample-mmi.c
  mips64/jdcolor-mmi.c mips64/jdmerge-mmi.c mips64/jdsample-mmi.c
  mips64/jfdctfst-mmi.c mips64/jfdctint-mmi.c mips64/jidctfst-mmi.c
  mips64/jidctint-mmi.c mips64/jquanti-mmi.c)

if(CMAKE_COMPILER_IS_GNUCC)
  foreach(file ${SIMD_SOURCES})
    set_property(SOURCE ${file} APPEND_STRING PROPERTY COMPILE_FLAGS
      " -fno-strict-aliasing")
  endforeach()
endif()
foreach(file ${SIMD_SOURCES})
  set_property(SOURCE ${file} APPEND_STRING PROPERTY COMPILE_FLAGS
    " -Wa,-mloongson-mmi,-mloongson-ext")
endforeach()

add_library(simd OBJECT ${SIMD_SOURCES} mips64/jsimdcpu.c jsimd.c)

if(CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED)
  set_target_properties(simd PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()


###############################################################################
# None
###############################################################################

else()

simd_fail("SIMD extensions not available for this CPU (${CMAKE_SYSTEM_PROCESSOR})")

endif() # CPU_TYPE
