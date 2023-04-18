set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(CMAKE_C_COMPILER  riscv64-unknown-linux-gnu-gcc)

set(CMAKE_CXX_FLAGS_INIT "-march=rv64gcv -mabi=lp64d -D__riscv_vector_071")
set(CMAKE_C_FLAGS_INIT "-march=rv64gcv -mabi=lp64d -D__riscv_vector_071")
