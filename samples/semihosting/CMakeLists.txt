# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html

set(SEMIHOSTING_SUFFIX semihosting)

add_subdirectory(include)
set(RAW_PIXEL_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(histogram)
add_subdirectory(norm)
