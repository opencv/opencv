# ----------------------------------------------------------------------------------------------
#  MIPS ToolChanin can be downloaded from https://www.mips.com/develop/tools/codescape-mips-sdk/ .
#  Toolchains with 'mti' in the name (and install directory) are for MIPS R2-R5 instruction sets.
#  Toolchains with 'img' in the name are for MIPS R6 instruction sets.
#  It is recommended to use cmake-gui for build scripts configuration and generation:
#  1. Run cmake-gui
#  2. Specifiy toolchain file mips64r6el-gnu.toolchain.cmake for cross-compiling.
#  3. Configure and Generate makefiles.
#  4. make -j4 & make install
# ----------------------------------------------------------------------------------------------
set(CMAKE_SYSTEM_PROCESSOR mips64r6el)
set(GCC_COMPILER_VERSION "" CACHE STRING "GCC Compiler version")
set(GNU_MACHINE "mips-img-linux-gnu" CACHE STRING "GNU compiler triple")
include("${CMAKE_CURRENT_LIST_DIR}/mips.toolchain.cmake")
