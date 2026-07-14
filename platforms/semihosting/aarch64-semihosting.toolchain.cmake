# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html

set(CMAKE_SYSTEM_NAME               Generic)
set(CMAKE_SYSTEM_PROCESSOR          AArch64)

set(CMAKE_TRY_COMPILE_TARGET_TYPE   STATIC_LIBRARY)

set(PORT_FILE ${CMAKE_SOURCE_DIR}/platforms/semihosting/include/aarch64_semihosting_port.hpp)

set(COMMON_FLAGS "--specs=rdimon.specs -DOPENCV_INCLUDE_PORT_FILE=\\\"${PORT_FILE}\\\"")

set(CMAKE_AR                        ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-ar${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_ASM_COMPILER              ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-gcc${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_C_COMPILER                ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-gcc${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_CXX_COMPILER              ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-g++${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_LINKER                    ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-ld${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_OBJCOPY                   ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-objcopy${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_RANLIB                    ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-ranlib${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_SIZE                      ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-size${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_STRIP                     ${SEMIHOSTING_TOOLCHAIN_PATH}aarch64-none-elf-strip${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_C_FLAGS                   ${COMMON_FLAGS} CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS                 ${COMMON_FLAGS} CACHE INTERNAL "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(OPENCV_SEMIHOSTING ON)
set(OPENCV_DISABLE_THREAD_SUPPORT ON)
set(OPENCV_DISABLE_FILESYSTEM_SUPPORT ON)
set(BUILD_SHARED_LIBS OFF)
set(OPENCV_FORCE_3RDPARTY_BUILD OFF)


# Enable newlib.
add_definitions(-D_GNU_SOURCE)

add_definitions(-D_POSIX_PATH_MAX=0)
