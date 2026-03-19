# ----------------------------------------------------------------------------
#  CMake file for libpng. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

project(${PNG_LIBRARY} C)

if(UNIX AND NOT APPLE AND NOT BEOS AND NOT HAIKU AND NOT EMSCRIPTEN)
  find_library(M_LIBRARY m)
  if(M_LIBRARY)
    set(M_LIBRARY m)
  else()
    set(M_LIBRARY "")
  endif()
else()
  # libm is not available or not needed.
endif()

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" ${ZLIB_INCLUDE_DIRS})

file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)

# CMake currently sets CMAKE_SYSTEM_PROCESSOR to one of x86_64 or arm64 on macOS,
# based upon the OS architecture, not the target architecture. As such, we need
# to check CMAKE_OSX_ARCHITECTURES to identify which hardware-specific flags to
# enable. Note that this will fail if you attempt to build a universal binary in
# a single CMake invocation.
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  set(TARGET_ARCH ${CMAKE_OSX_ARCHITECTURES})
else()
  set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

OCV_OPTION(PNG_HARDWARE_OPTIMIZATIONS "Enable Hardware Optimizations, if available for this platform" (NOT CV_DISABLE_OPTIMIZATION))

if(PNG_HARDWARE_OPTIMIZATIONS)

# Set definitions and sources for ARM.
if(TARGET_ARCH MATCHES "^(ARM|arm|aarch)")
  if(TARGET_ARCH MATCHES "^(ARM64|arm64|aarch64)")
    set(PNG_ARM_NEON_POSSIBLE_VALUES on off)
    set(PNG_ARM_NEON "on"
        CACHE STRING "Enable ARM NEON optimizations: on|off; on is default")
  else()
    set(PNG_ARM_NEON_POSSIBLE_VALUES check on off)
    set(PNG_ARM_NEON "off"
        CACHE STRING "Enable ARM NEON optimizations: check|on|off; off is default")
  endif()
  set_property(CACHE PNG_ARM_NEON
               PROPERTY STRINGS ${PNG_ARM_NEON_POSSIBLE_VALUES})
  list(FIND PNG_ARM_NEON_POSSIBLE_VALUES ${PNG_ARM_NEON} index)
  if(index EQUAL -1)
    message(FATAL_ERROR "PNG_ARM_NEON must be one of [${PNG_ARM_NEON_POSSIBLE_VALUES}]")
  elseif(NOT PNG_ARM_NEON STREQUAL "off")
    list(APPEND lib_srcs arm/arm_init.c arm/filter_neon_intrinsics.c arm/palette_neon_intrinsics.c)
    if(NOT MSVC)
      if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
        message(STATUS "Skipping arm/filter_neon.S on QNX")
      else()
       enable_language(ASM)
       list(APPEND lib_srcs arm/filter_neon.S)
      endif()
    endif()
    if(PNG_ARM_NEON STREQUAL "on")
      add_definitions(-DPNG_ARM_NEON_OPT=2)
    elseif(PNG_ARM_NEON STREQUAL "check")
      add_definitions(-DPNG_ARM_NEON_CHECK_SUPPORTED)
    endif()
  else()
    add_definitions(-DPNG_ARM_NEON_OPT=0) # NEON assembler is not supported
  endif()
endif()

# Set definitions and sources for PowerPC.
if(TARGET_ARCH MATCHES "^(powerpc|ppc64)")
  if (CPU_VSX_SUPPORTED)
    set(CPU_VSX_SUPPORTED_STR "on")
  else()
    set(CPU_VSX_SUPPORTED_STR "off")
  endif()
  set(PNG_POWERPC_VSX_POSSIBLE_VALUES on off)
  set(PNG_POWERPC_VSX ${CPU_VSX_SUPPORTED_STR}
      CACHE STRING "Enable POWERPC VSX optimizations: on|off; on is default")
  set_property(CACHE PNG_POWERPC_VSX
               PROPERTY STRINGS ${PNG_POWERPC_VSX_POSSIBLE_VALUES})
  list(FIND PNG_POWERPC_VSX_POSSIBLE_VALUES ${PNG_POWERPC_VSX} index)
  if(index EQUAL -1)
    message(FATAL_ERROR "PNG_POWERPC_VSX must be one of [${PNG_POWERPC_VSX_POSSIBLE_VALUES}]")
  elseif(NOT PNG_POWERPC_VSX STREQUAL "off")
    list(APPEND lib_srcs powerpc/powerpc_init.c powerpc/filter_vsx_intrinsics.c)
    if(PNG_POWERPC_VSX STREQUAL "on")
      add_definitions(-DPNG_POWERPC_VSX_OPT=2)
    endif()
  else()
    add_definitions(-DPNG_POWERPC_VSX_OPT=0)
  endif()
endif()

# Set definitions and sources for Intel.
if(TARGET_ARCH MATCHES "^(i[3-6]86|x86|AMD64)")
  set(PNG_INTEL_SSE_POSSIBLE_VALUES on off)
  set(PNG_INTEL_SSE "on"
      CACHE STRING "Enable INTEL_SSE optimizations: on|off; on is default")
  set_property(CACHE PNG_INTEL_SSE
               PROPERTY STRINGS ${PNG_INTEL_SSE_POSSIBLE_VALUES})
  list(FIND PNG_INTEL_SSE_POSSIBLE_VALUES ${PNG_INTEL_SSE} index)
  if(index EQUAL -1)
    message(FATAL_ERROR "PNG_INTEL_SSE must be one of [${PNG_INTEL_SSE_POSSIBLE_VALUES}]")
  elseif(NOT PNG_INTEL_SSE STREQUAL "off")
    list(APPEND lib_srcs intel/intel_init.c intel/filter_sse2_intrinsics.c)
    if(PNG_INTEL_SSE STREQUAL "on")
      add_definitions(-DPNG_INTEL_SSE_OPT=1)
    endif()
  else()
    add_definitions(-DPNG_INTEL_SSE_OPT=0)
  endif()
endif()

# Set definitions and sources for MIPS.
if(TARGET_ARCH MATCHES "^(mipsel|mips64el)")
  if (CPU_MSA_SUPPORTED)
    set(CPU_MSA_SUPPORTED_STR "on")
  else()
    set(CPU_MSA_SUPPORTED_STR "off")
  endif()
  set(PNG_MIPS_MSA_POSSIBLE_VALUES on off)
  set(PNG_MIPS_MSA ${CPU_MSA_SUPPORTED_STR}
      CACHE STRING "Enable MIPS_MSA optimizations: on|off; on is default")
  set_property(CACHE PNG_MIPS_MSA
               PROPERTY STRINGS ${PNG_MIPS_MSA_POSSIBLE_VALUES})
  list(FIND PNG_MIPS_MSA_POSSIBLE_VALUES ${PNG_MIPS_MSA} index_msa)
  if(index_msa EQUAL -1)
    message(FATAL_ERROR "PNG_MIPS_MSA must be one of [${PNG_MIPS_MSA_POSSIBLE_VALUES}]")
  endif()

  set(PNG_MIPS_MMI_POSSIBLE_VALUES on off)
  set(PNG_MIPS_MMI "on"
      CACHE STRING "Enable MIPS_MMI optimizations: on|off; on is default")
  set_property(CACHE PNG_MIPS_MMI
               PROPERTY STRINGS ${PNG_MIPS_MMI_POSSIBLE_VALUES})
  list(FIND PNG_MIPS_MMI_POSSIBLE_VALUES ${PNG_MIPS_MMI} index_mmi)
  if(index_mmi EQUAL -1)
    message(FATAL_ERROR "PNG_MIPS_MMI must be one of [${PNG_MIPS_MMI_POSSIBLE_VALUES}]")
  endif()

  if(PNG_MIPS_MSA STREQUAL "on" AND PNG_MIPS_MMI STREQUAL "on")
    list(APPEND lib_srcs mips/mips_init.c mips/filter_msa_intrinsics.c mips/filter_mmi_inline_assembly.c)
    add_definitions(-DPNG_MIPS_MSA_OPT=2)
    add_definitions(-DPNG_MIPS_MMI_OPT=1)
  elseif(PNG_MIPS_MSA STREQUAL "on")
    list(APPEND lib_srcs mips/mips_init.c mips/filter_msa_intrinsics.c)
    add_definitions(-DPNG_MIPS_MSA_OPT=2)
    add_definitions(-DPNG_MIPS_MMI_OPT=0)
  elseif(PNG_MIPS_MMI STREQUAL "on")
    list(APPEND lib_srcs mips/mips_init.c mips/filter_mmi_inline_assembly.c)
    add_definitions(-DPNG_MIPS_MSA_OPT=0)
    add_definitions(-DPNG_MIPS_MMI_OPT=1)
    else()
    add_definitions(-DPNG_MIPS_MSA_OPT=0)
    add_definitions(-DPNG_MIPS_MMI_OPT=0)
    endif()
endif()

# Set definitions and sources for LoongArch.
if(TARGET_ARCH MATCHES "^(loongarch)")
  if (CPU_LSX_SUPPORTED)
    set(CPU_LSX_SUPPORTED_STR "on")
  else()
    set(CPU_LSX_SUPPORTED_STR "off")
  endif()
  include(CheckCCompilerFlag)
  set(PNG_LOONGARCH_LSX_POSSIBLE_VALUES on off)
  set(PNG_LOONGARCH_LSX ${CPU_LSX_SUPPORTED_STR}
      CACHE STRING "Enable LOONGARCH_LSX optimizations: on|off; on is default")
  set_property(CACHE PNG_LOONGARCH_LSX
               PROPERTY STRINGS ${PNG_LOONGARCH_LSX_POSSIBLE_VALUES})
  list(FIND PNG_LOONGARCH_LSX_POSSIBLE_VALUES ${PNG_LOONGARCH_LSX} index)
  if(index EQUAL -1)
    message(FATAL_ERROR "PNG_LOONGARCH_LSX must be one of [${PNG_LOONGARCH_LSX_POSSIBLE_VALUES}]")
  elseif(NOT PNG_LOONGARCH_LSX STREQUAL "off")
    CHECK_C_COMPILER_FLAG("-mlsx" COMPILER_SUPPORTS_LSX)
    if(COMPILER_SUPPORTS_LSX)
      set(libpng_loongarch_sources
          loongarch/loongarch_lsx_init.c
          loongarch/filter_lsx_intrinsics.c)
      set_source_files_properties(${libpng_loongarch_sources}
                                  PROPERTIES
                                  COMPILE_FLAGS "-mlsx")
      list(APPEND lib_srcs ${libpng_loongarch_sources})
      add_definitions(-DPNG_LOONGARCH_LSX_OPT=1)
    else()
      message(FATAL_ERROR "Compiler does not support -mlsx option")
    endif()
  else()
    add_definitions(-DPNG_LOONGARCH_LSX_OPT=0)
  endif()
endif()

# Set definitions and sources for RISC-V.
if(TARGET_ARCH MATCHES "^(riscv)")
  if (CPU_RVV_SUPPORTED)
    set(CPU_RVV_SUPPORTED_STR "on")
  else()
    set(CPU_RVV_SUPPORTED_STR "off")
  endif()
  include(CheckCCompilerFlag)
  set(PNG_RISCV_RVV_POSSIBLE_VALUES on off)
  set(PNG_RISCV_RVV ${CPU_RVV_SUPPORTED_STR}
    CACHE STRING "Enable RISC-V Vector optimizations: on|off; off is default")
  set_property(CACHE PNG_RISCV_RVV
              PROPERTY STRINGS ${PNG_RISCV_RVV_POSSIBLE_VALUES})
  list(FIND PNG_RISCV_RVV_POSSIBLE_VALUES ${PNG_RISCV_RVV} index)
  if(index EQUAL -1)
    message(FATAL_ERROR "PNG_RISCV_RVV must be one of [${PNG_RISCV_RVV_POSSIBLE_VALUES}]")
  elseif(NOT PNG_RISCV_RVV STREQUAL "off")
    check_c_source_compiles("
      #include <riscv_vector.h>
      int main() {
        const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        uint64_t ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
        vuint8m1_t a = __riscv_vreinterpret_v_u64m1_u8m1(__riscv_vle64_v_u64m1(ptr, 2));
        vfloat32m1_t val = __riscv_vle32_v_f32m1((const float*)(src), 4);
        return (int)__riscv_vfmv_f_s_f32m1_f32(val);
    }" COMPILER_SUPPORTS_RVV)

    if(NOT COMPILER_SUPPORTS_RVV)
      message(FATAL_ERROR "Compiler does not support RISC-V Vector extension or its unable to detect it")
    endif()
    list(APPEND lib_srcs riscv/filter_rvv_intrinsics.c riscv/riscv_init.c)
    if(PNG_RISCV_RVV STREQUAL "on")
      add_definitions(-DPNG_RISCV_RVV_OPT=2)
    else()
      add_definitions(-DPNG_RISCV_RVV_OPT=0)
    endif()
  else()
    add_definitions(-DPNG_RISCV_RVV_OPT=0)
  endif()
endif()

else(PNG_HARDWARE_OPTIMIZATIONS)

# Set definitions and sources for ARM.
if(TARGET_ARCH MATCHES "^(ARM|arm|aarch)")
  add_definitions(-DPNG_ARM_NEON_OPT=0)
endif()

# Set definitions and sources for PowerPC.
if(TARGET_ARCH MATCHES "^(powerpc|ppc64)")
  add_definitions(-DPNG_POWERPC_VSX_OPT=0)
endif()

# Set definitions and sources for Intel.
if(TARGET_ARCH MATCHES "^(i[3-6]86|x86|AMD64)")
  add_definitions(-DPNG_INTEL_SSE_OPT=0)
endif()

# Set definitions and sources for MIPS.
if(TARGET_ARCH MATCHES "^(mipsel|mips64el)")
  add_definitions(-DPNG_MIPS_MSA_OPT=0)
endif()

# Set definitions and sources for LoongArch.
if(TARGET_ARCH MATCHES "^(loongarch)")
  add_definitions(-DPNG_LOONGARCH_LSX_OPT=0)
endif()

# Set definitions and sources for RISC-V.
if(TARGET_ARCH MATCHES "^(riscv)")
  add_definitions(-DPNG_RISCV_RVV_OPT=0)
endif()

endif(PNG_HARDWARE_OPTIMIZATIONS)

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

add_library(${PNG_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})
target_link_libraries(${PNG_LIBRARY} ${ZLIB_LIBRARIES})

ocv_warnings_disable(CMAKE_C_FLAGS -Wundef -Wcast-align -Wimplicit-fallthrough -Wunused-parameter -Wsign-compare
    -Wmaybe-uninitialized
    -Wnull-pointer-subtraction                   # clang15
    -Wunused-but-set-variable                    # clang15
)

set_target_properties(${PNG_LIBRARY}
  PROPERTIES OUTPUT_NAME ${PNG_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${PNG_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${PNG_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PNG_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${PNG_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(libpng LICENSE README)
