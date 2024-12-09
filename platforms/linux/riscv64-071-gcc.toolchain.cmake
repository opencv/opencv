set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(CMAKE_C_COMPILER  riscv64-unknown-linux-gnu-gcc)

# MangoPi MQ Pro - C906FD, C906FDV
# Lichee Pi 4A - C910, C910V (?)
# CanMV K230 - C908, C908V

# See https://github.com/T-head-Semi/gcc/blob/xuantie-gcc-10.4.0/gcc/config/riscv/riscv-cores.def

set(_enable_vector OFF)
if(CORE STREQUAL "C906FD")
  set(CMAKE_C_FLAGS_INIT "-mcpu=c906fd -mabi=lp64d  -mtune=c906fd")
  set(CMAKE_CXX_FLAGS_INIT "-mcpu=c906fd -mabi=lp64d  -mtune=c906fd")
elseif(CORE STREQUAL "C906FDV")
  set(CMAKE_C_FLAGS_INIT "-mcpu=c906fd -mabi=lp64d  -mtune=c906fd")
  set(CMAKE_CXX_FLAGS_INIT "-mcpu=c906fd -mabi=lp64d  -mtune=c906fd")
  # Disabled due to limited 64-bit SEW support
  # set(_enable_vector ON)
elseif(CORE STREQUAL "C908")
  set(CMAKE_C_FLAGS_INIT "-mcpu=c908 -mabi=lp64d  -mtune=c908")
  set(CMAKE_CXX_FLAGS_INIT "-mcpu=c908 -mabi=lp64d  -mtune=c908")
elseif(CORE STREQUAL "C908V")
  set(CMAKE_C_FLAGS_INIT "-mcpu=c908v -mabi=lp64d  -mtune=c908")
  set(CMAKE_CXX_FLAGS_INIT "-mcpu=c908v -mabi=lp64d  -mtune=c908")
  set(_enable_vector ON) # RVV 1.0
elseif(CORE STREQUAL "C910")
  set(CMAKE_C_FLAGS_INIT "-mcpu=c910 -mabi=lp64d -mtune=c910")
  set(CMAKE_CXX_FLAGS_INIT "-mcpu=c910 -mabi=lp64d -mtune=c910")
elseif(CORE STREQUAL "C910V")
  set(CMAKE_C_FLAGS_INIT "-march=rv64imafdcv0p7xthead -mabi=lp64d")
  set(CMAKE_CXX_FLAGS_INIT "-march=rv64imafdcv0p7xthead -mabi=lp64d")
  set(_enable_vector ON) # RVV 0.7.1
elseif(CORE STREQUAL "C920")
  set(CMAKE_C_FLAGS_INIT "-mcpu=c920 -mabi=lp64d  -mtune=c920")
  set(CMAKE_CXX_FLAGS_INIT "-mcpu=c920 -mabi=lp64d  -mtune=c920")
  set(_enable_vector ON) # RVV 0.7.1
elseif(CORE STREQUAL "C920V2")
  set(CMAKE_C_FLAGS_INIT "-mcpu=c920v2 -mabi=lp64d  -mtune=c920v2")
  set(CMAKE_CXX_FLAGS_INIT "-mcpu=c920v2 -mabi=lp64d  -mtune=c920v2")
  set(_enable_vector ON) # RVV 1.0
else()
  set(CMAKE_C_FLAGS_INIT "-march=rv64imafdc_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d")
  set(CMAKE_CXX_FLAGS_INIT "-march=rv64imafdc_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d")
endif()

if(_enable_vector)
  set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -D__riscv_vector_071 -mrvv-vector-bits=128")
  set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -D__riscv_vector_071 -mrvv-vector-bits=128")
endif()

if(ENABLE_GCOV)
  set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -fprofile-arcs -ftest-coverage")
endif()
