set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

message(STATUS "RISCV: $ENV{RISCV}")
message(STATUS "RISCV_GCC_INSTALL_ROOT: $ENV{RISCV_GCC_INSTALL_ROOT}")

set(RISCV_GCC_INSTALL_ROOT $ENV{RISCV} CACHE PATH "Path to GCC for RISC-V cross compiler installation directory")

set(CMAKE_C_COMPILER  ${RISCV_GCC_INSTALL_ROOT}/bin/riscv64-linux-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_GCC_INSTALL_ROOT}/bin/riscv64-linux-g++)

# fix toolchain macro

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ANDES=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ANDES=1")

# enable rvp

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc -mext-dsp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gc -mext-dsp")

# fix segment address

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Ttext-segment=0x50000")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Ttext-segment=0x50000")
