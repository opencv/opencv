message(STATUS "Setting up iPhoneOS toolchain for IOS_ARCH='${IOS_ARCH}'")
set(IPHONEOS TRUE)
include(${CMAKE_CURRENT_LIST_DIR}/common-ios-toolchain.cmake)
message(STATUS "iPhoneOS toolchain loaded")
