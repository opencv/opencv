set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_C_COMPILER  riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)

set(CMAKE_CXX_FLAGS ""    CACHE STRING "")
set(CMAKE_C_FLAGS ""    CACHE STRING "")

set(CMAKE_CXX_FLAGS "-static -march=rv64gcvxthead -mabi=lp64v -pthread -D__riscv_vector_071")
set(CMAKE_C_FLAGS "-static -march=rv64gcvxthead -mabi=lp64v -pthread -D__riscv_vector_071")
