set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV_GCC_INSTALL_ROOT $ENV{RISCV} CACHE PATH "Path to GCC for RISC-V cross compiler installation directory")

set(CMAKE_C_COMPILER  ${RISCV_GCC_INSTALL_ROOT}/bin/riscv64-linux-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_GCC_INSTALL_ROOT}/bin/riscv64-linux-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc -mext-dsp -g -ggdb -print-file-name=libstdc++.so.6")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gc -mext-dsp -g -ggdb -print-file-name=libstdc++.so.6")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=/home/junyan/lib-andes/usr/lib,-dynamic-linker=/home/junyan/lib-andes/usr/lib/libstdc++.so.6")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=/home/junyan/lib-andes/usr/lib,-dynamic-linker=/home/junyan/lib-andes/usr/lib/libstdc++.so.6")
