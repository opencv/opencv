add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/raw_pixels.hpp
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/fill_random_image.sh 102400 > raw_pixels.hpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fill_random_image.sh
  VERBATIM)
add_custom_target(raw_pixels_for_baremetal DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/raw_pixels.hpp)
