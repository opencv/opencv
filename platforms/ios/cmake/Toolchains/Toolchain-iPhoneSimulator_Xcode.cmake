message(STATUS "Setting up iPhoneSimulator toolchain for IOS_ARCH='${IOS_ARCH}'")
set(IPHONESIMULATOR TRUE)
include(${CMAKE_CURRENT_LIST_DIR}/common-ios-toolchain.cmake)
message(STATUS "iPhoneSimulator toolchain loaded")
