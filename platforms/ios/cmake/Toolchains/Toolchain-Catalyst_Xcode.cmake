set(MAC_CATALYST TRUE)
message(STATUS "Setting up Catalyst toolchain for IOS_ARCH='${IOS_ARCH}'")
include(${CMAKE_CURRENT_LIST_DIR}/common-ios-toolchain.cmake)
message(STATUS "Catalyst toolchain loaded")
