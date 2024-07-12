set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

message(STATUS "RISCV: $ENV{RISCV}")
message(STATUS "RISCV_GCC_INSTALL_ROOT: $ENV{RISCV_GCC_INSTALL_ROOT}")

set(RISCV_GCC_INSTALL_ROOT $ENV{RISCV} CACHE PATH "Path to GCC for RISC-V cross compiler installation directory")

set(CMAKE_C_COMPILER  ${RISCV_GCC_INSTALL_ROOT}/bin/riscv64-linux-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_GCC_INSTALL_ROOT}/bin/riscv64-linux-g++)

# fix toolchain macro
# enable rvp

set(CMAKE_C_FLAGS_INIT "-march=rv64gc -mext-dsp -D__ANDES=1")
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gc -mext-dsp -D__ANDES=1")

# fix segment address

set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,-Ttext-segment=0x50000")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-Wl,-Ttext-segment=0x50000")
