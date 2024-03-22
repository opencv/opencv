if(NOT DEFINED MIN_VER_CMAKE)
  # Driven by install(FILES) support of generator expressions
  # Ubuntu Focal comes with 3.16
  set(MIN_VER_CMAKE 3.14)
endif()
set(MIN_VER_CUDA 6.5)
set(MIN_VER_CUDNN 7.5)
set(MIN_VER_PYTHON3 3.2)
set(MIN_VER_ZLIB 1.2.3)
