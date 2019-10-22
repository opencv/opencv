# ----------------------------------------------------------------------------------------------
#  MIPS ToolChanin can be downloaded from https://www.mips.com/develop/tools/codescape-mips-sdk/ .
#  Toolchains with 'mti' in the name (and install directory) are for MIPS R2-R5 instruction sets.
#  Toolchains with 'img' in the name are for MIPS R6 instruction sets.
#  It is recommended to use cmake-gui application for build scripts configuration and generation:
#  1. Run cmake-gui
#  2. Specifiy toolchain file for cross-compiling, mips32r5el-gnu.toolchian.cmake or mips64r6el-gnu.toolchain.cmake
#     can be selected.
#  3. Configure and Generate makefiles.
#  4. make -j4 & make install
# ----------------------------------------------------------------------------------------------

if(COMMAND toolchain_save_config)
  return() # prevent recursive call
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_SYSTEM_PROCESSOR mips)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/gnu.toolchain.cmake")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL mips AND NOT MIPS_IGNORE_FP)
  set(FLOAT_ABI_SUFFIX "")
endif()

if(NOT "x${GCC_COMPILER_VERSION}" STREQUAL "x")
  set(__GCC_VER_SUFFIX "-${GCC_COMPILER_VERSION}")
endif()

if(NOT DEFINED CMAKE_C_COMPILER)
  find_program(CMAKE_C_COMPILER NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-gcc${__GCC_VER_SUFFIX})
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(CMAKE_CXX_COMPILER NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-g++${__GCC_VER_SUFFIX})
endif()
if(NOT DEFINED CMAKE_LINKER)
  find_program(CMAKE_LINKER NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ld${__GCC_VER_SUFFIX} ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ld)
endif()
if(NOT DEFINED CMAKE_AR)
  find_program(CMAKE_AR NAMES ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ar${__GCC_VER_SUFFIX} ${GNU_MACHINE}${FLOAT_ABI_SUFFIX}-ar)
endif()

if(NOT DEFINED MIPS_LINUX_SYSROOT AND DEFINED GNU_MACHINE)
  set(MIPS_LINUX_SYSROOT /usr/bin)
endif()

if(NOT DEFINED CMAKE_CXX_FLAGS)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "mips32r5el")
    set(CMAKE_C_FLAGS             "-march=mips32r5 -EL -mmsa -mhard-float -mfp64 -mnan=2008 -mabs=2008 -O3 -ffp-contract=off -mtune=p5600" CACHE INTERNAL "")
    set(CMAKE_SHARED_LINKER_FLAGS "" CACHE INTERNAL "")
    set(CMAKE_CXX_FLAGS           "-march=mips32r5 -EL -mmsa -mhard-float -mfp64 -mnan=2008 -mabs=2008 -O3 -ffp-contract=off -mtune=p5600" CACHE INTERNAL "")
    set(CMAKE_MODULE_LINKER_FLAGS "" CACHE INTERNAL "")
    set(CMAKE_EXE_LINKER_FLAGS    "-lpthread -lrt -ldl -latomic" CACHE INTERNAL "Added for mips cross build error")

    set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} -fdata-sections  -Wa,--noexecstack -fsigned-char -Wno-psabi")
    set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS} -fdata-sections  -Wa,--noexecstack -fsigned-char -Wno-psabi")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips64r6el")
    set(CMAKE_C_FLAGS             "-O3 -march=i6500 -EL -mmsa -mabi=64 -mhard-float -mfp64 -mnan=2008" CACHE INTERNAL "")
    set(CMAKE_SHARED_LINKER_FLAGS "" CACHE INTERNAL "")
    set(CMAKE_CXX_FLAGS           "-O3 -march=i6500 -EL -mmsa -mabi=64 -mhard-float -mfp64 -mnan=2008" CACHE INTERNAL "")
    set(CMAKE_MODULE_LINKER_FLAGS "" CACHE INTERNAL "")
    set(CMAKE_EXE_LINKER_FLAGS    "-lpthread -lrt -ldl" CACHE INTERNAL "Added for mips cross build error")

    set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
    set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
  endif()
  set(CMAKE_SHARED_LINKER_FLAGS "${MIPS_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS "${MIPS_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS    "${MIPS_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${MIPS_LINUX_SYSROOT})

set(TOOLCHAIN_CONFIG_VARS ${TOOLCHAIN_CONFIG_VARS}
    MIPS_LINUX_SYSROOT
)
toolchain_save_config()
