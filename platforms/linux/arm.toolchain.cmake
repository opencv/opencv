if(COMMAND toolchain_save_config)
  return() # prevent recursive call
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_SYSTEM_PROCESSOR arm)
else()
  #message("CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/gnu.toolchain.cmake")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL arm AND NOT ARM_IGNORE_FP)
  set(FLOAT_ABI_SUFFIX "")
  if(NOT SOFTFP)
    set(FLOAT_ABI_SUFFIX "hf")
  endif()
endif()

if(NOT "x${GCC_COMPILER_VERSION}" STREQUAL "x")
  set(__GCC_VER_SUFFIX "-${GCC_COMPILER_VERSION}")
endif()

if(NOT DEFINED CMAKE_C_COMPILER)
  find_program(CMAKE_C_COMPILER NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-gcc${__GCC_VER_SUFFIX})
else()
  #message(WARNING "CMAKE_C_COMPILER=${CMAKE_C_COMPILER} is defined")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(CMAKE_CXX_COMPILER NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-g++${__GCC_VER_SUFFIX})
else()
  #message(WARNING "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} is defined")
endif()
if(NOT DEFINED CMAKE_LINKER)
  find_program(CMAKE_LINKER NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ld${__GCC_VER_SUFFIX} ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ld)
else()
  #message(WARNING "CMAKE_LINKER=${CMAKE_LINKER} is defined")
endif()
if(NOT DEFINED CMAKE_AR)
  find_program(CMAKE_AR NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ar${__GCC_VER_SUFFIX} ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ar)
else()
  #message(WARNING "CMAKE_AR=${CMAKE_AR} is defined")
endif()

if(NOT DEFINED ARM_LINUX_SYSROOT AND DEFINED GNU_MACHINE)
  set(ARM_LINUX_SYSROOT /usr/${GNU_MACHINE}${FLOAT_ABI_SUFFIX})
endif()

# == Compiler flags
set(common_cc_opt "-fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
  set(CMAKE_CXX_FLAGS_INIT "-mthumb ${common_cc_opt}")
  set(CMAKE_C_FLAGS_INIT   "-mthumb ${common_cc_opt}")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  message(STATUS "!!! ENABLE_BF16=${ENABLE_BF16}")
  message(STATUS "!!! ENABLE_FP16=${ENABLE_FP16}")
  message(STATUS "!!! ENABLE_DOTPROD=${ENABLE_DOTPROD}")
  # see https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html
  if(ENABLE_BF16)
    set(CMAKE_CXX_FLAGS_INIT "-march=armv8.6-a ${common_cc_opt}")
    set(CMAKE_C_FLAGS_INIT "-march=armv8.6-a ${common_cc_opt}")
  elseif(ENABLE_FP16 OR ENABLE_DOTPROD)
    set(CMAKE_CXX_FLAGS_INIT "-march=armv8.4-a ${common_cc_opt}")
    set(CMAKE_C_FLAGS_INIT "-march=armv8.4-a ${common_cc_opt}")
  else()
    set(CMAKE_CXX_FLAGS_INIT "${common_cc_opt}")
    set(CMAKE_C_FLAGS_INIT "${common_cc_opt}")
  endif()
endif()

# == Linker flags
if(CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
  set(common_ld_opt "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  set(CMAKE_EXE_LINKER_FLAGS_INIT    "${common_ld_opt} -Wl,-z,nocopyreloc")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(common_ld_opt "-Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  set(CMAKE_EXE_LINKER_FLAGS_INIT    "${common_ld_opt}")
endif()
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${common_ld_opt}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${common_ld_opt}")


if(USE_NEON)
  message(WARNING "You use obsolete variable USE_NEON to enable NEON instruction set. Use -DENABLE_NEON=ON instead." )
  set(ENABLE_NEON TRUE)
elseif(USE_VFPV3)
  message(WARNING "You use obsolete variable USE_VFPV3 to enable VFPV3 instruction set. Use -DENABLE_VFPV3=ON instead." )
  set(ENABLE_VFPV3 TRUE)
endif()

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${ARM_LINUX_SYSROOT})

if(EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
  set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${CUDA_TOOLKIT_ROOT_DIR})
endif()

set(TOOLCHAIN_CONFIG_VARS ${TOOLCHAIN_CONFIG_VARS}
    ARM_LINUX_SYSROOT
    ENABLE_NEON
    ENABLE_VFPV3
    CUDA_TOOLKIT_ROOT_DIR
)
toolchain_save_config()
