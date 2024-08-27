# see https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html
if(ENABLE_BF16)
  set(CMAKE_CXX_FLAGS_INIT "-march=armv8.6-a")
  set(CMAKE_C_FLAGS_INIT "-march=armv8.6-a")
elseif(ENABLE_FP16 OR ENABLE_DOTPROD)
  set(CMAKE_CXX_FLAGS_INIT "-march=armv8.4-a")
  set(CMAKE_C_FLAGS_INIT "-march=armv8.4-a")
endif()
