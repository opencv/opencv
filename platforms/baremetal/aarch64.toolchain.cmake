# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html

set(CMAKE_SYSTEM_NAME               Generic)
set(CMAKE_SYSTEM_PROCESSOR          AArch64)

set(CMAKE_TRY_COMPILE_TARGET_TYPE   STATIC_LIBRARY)

set(AARCH64_PORT_FILE ${CMAKE_SOURCE_DIR}/platforms/baremetal/include/aarch64_baremetal_port.hpp)

set(CMAKE_AR                        ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-ar${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_ASM_COMPILER              ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-gcc${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_C_COMPILER                ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-gcc${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_CXX_COMPILER              ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-g++${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_LINKER                    ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-ld${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_OBJCOPY                   ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-objcopy${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_RANLIB                    ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-ranlib${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_SIZE                      ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-size${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_STRIP                     ${BAREMETAL_TOOLCHAIN_PATH}aarch64-none-elf-strip${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_C_FLAGS                   "--specs=rdimon.specs -DOPENCV_INCLUDE_PORT_FILE=\\\"${AARCH64_PORT_FILE}\\\"" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS                 "--specs=rdimon.specs -DOPENCV_INCLUDE_PORT_FILE=\\\"${AARCH64_PORT_FILE}\\\"" CACHE INTERNAL "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CV_BAREMETAL ON)
set(OPENCV_DISABLE_THREAD_SUPPORT ON)
set(OPENCV_DISABLE_FILESYSTEM_SUPPORT ON)
set(BUILD_SHARED_LIBS OFF)
set(OPENCV_FORCE_3RDPARTY_BUILD ON)

# These third parties libraries are incompatible with the baremetal
# toolchain.
set(WITH_JPEG OFF)
set(WITH_OPENEXR OFF)
set(WITH_TIFF OFF)
