set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_C_COMPILER  riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)

set(CMAKE_CXX_FLAGS ""    CACHE STRING "")
set(CMAKE_C_FLAGS ""    CACHE STRING "")

set(CMAKE_CXX_FLAGS "-static -march=rv64gcv -mabi=lp64dv -pthread")
set(CMAKE_C_FLAGS "-static -march=rv64gcv -mabi=lp64dv -pthread")
