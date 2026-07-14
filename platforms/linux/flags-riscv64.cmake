# see https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html#index-march-14
function(ocv_set_platform_flags VAR)
  if(ENABLE_RVV OR RISCV_RVV_SCALABLE)
    set(flags "-march=rv64gcv")
  else()
    set(flags "-march=rv64gc")
  endif()
  set(${VAR} "${flags}" PARENT_SCOPE)
endfunction()
