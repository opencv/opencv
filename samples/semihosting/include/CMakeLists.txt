# Generate an array with random data.
set(RAW_PIXELS_SIZE 102400)
set(RAW_PIXELS_HEADER_PATH ${CMAKE_CURRENT_BINARY_DIR}/raw_pixels.hpp)
file(WRITE ${RAW_PIXELS_HEADER_PATH})
file(APPEND ${RAW_PIXELS_HEADER_PATH} "#ifndef RAW_PIXELS_HPP\n")
file(APPEND ${RAW_PIXELS_HEADER_PATH} "#define RAW_PIXELS_HPP\n")
file(APPEND ${RAW_PIXELS_HEADER_PATH} "#include <cstdint>\n")
file(APPEND ${RAW_PIXELS_HEADER_PATH} "#define RAW_PIXELS_SIZE ${RAW_PIXELS_SIZE}\n")

# Seed the random number generator.
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdf RANDOM_SEED 314 number)

file(APPEND ${RAW_PIXELS_HEADER_PATH} "static std::uint32_t raw_pixels[RAW_PIXELS_SIZE + 1] = {\n")
foreach(i RANGE ${RAW_PIXELS_SIZE})
  string(RANDOM LENGTH 8 ALPHABET 0123456789abcdf number)
  file(APPEND ${RAW_PIXELS_HEADER_PATH} "0x${number},\n")
endforeach()
file(APPEND ${RAW_PIXELS_HEADER_PATH} "};\n")
file(APPEND ${RAW_PIXELS_HEADER_PATH} "#endif //RAW_PIXELS_HPP\n")
