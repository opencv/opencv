if(ENABLE_RVV OR RISCV_RVV_SCALABLE)
  set(OCV_FLAGS "-march=rv64gcv")
else()
  set(OCV_FLAGS "-march=rv64gc")
endif()

