# Populate a C array with random data.
set(RAW_PIXELS_SIZE 102400)
set(RAW_PIXELS_HEADER ${CMAKE_CURRENT_BINARY_DIR}/raw_pixels.hpp)
set(RAW_PIXELS_HEADER_IN ${CMAKE_CURRENT_SOURCE_DIR}/raw_pixels.hpp.in)

set(RAW_PIXEL_VALUES "")
# Seed the random number generator.
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdf RANDOM_SEED 314 number)
math(EXPR LOOP_RANGE "${RAW_PIXELS_SIZE} - 1")

foreach(i RANGE ${LOOP_RANGE})
  string(RANDOM LENGTH 8 ALPHABET 0123456789abcdf number)
  string(CONCAT RAW_PIXEL_VALUES ${RAW_PIXEL_VALUES} "0x${number}, \\\n")
endforeach()

configure_file(${RAW_PIXELS_HEADER_IN} ${RAW_PIXELS_HEADER})
