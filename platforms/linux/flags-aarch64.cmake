# see https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html
if(ENABLE_BF16)
  set(OCV_FLAGS "-march=armv8.6-a")
elseif(ENABLE_FP16 OR ENABLE_DOTPROD)
  set(OCV_FLAGS "-march=armv8.4-a")
endif()
