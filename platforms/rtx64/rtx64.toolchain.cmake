cmake_minimum_required( VERSION 2.6.3 )

# Empty linker for RTX64
SET(CMAKE_CXX_STANDARD_LIBRARIES "" CACHE STRING "Force an empty linker for RTX64" FORCE)
SET(CMAKE_C_STANDARD_LIBRARIES "" CACHE STRING "Force an empty linker for RTX64" FORCE)