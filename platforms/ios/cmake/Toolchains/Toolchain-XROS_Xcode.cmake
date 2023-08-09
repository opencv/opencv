message(STATUS "Setting up visionOS toolchain for VISIONOS_ARCH='${VISIONOS_ARCH}'")
set(VISIONOS TRUE)
include(${CMAKE_CURRENT_LIST_DIR}/common-ios-toolchain.cmake)
message(STATUS "visionOS toolchain loaded")
