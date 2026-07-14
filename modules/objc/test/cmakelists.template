cmake_minimum_required(VERSION 3.15)

project(OpenCVTest)

enable_testing()

find_package(XCTest REQUIRED)

# Enable C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED TRUE)

set (OBJC_COMPILE_FLAGS "-fobjc-arc -fobjc-weak -fvisibility=hidden")
set (SUPPRESS_WARNINGS_FLAGS "-Wno-incomplete-umbrella")
set (CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OBJC_COMPILE_FLAGS} ${SUPPRESS_WARNINGS_FLAGS}")
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OBJC_ARC_COMPILE_FLAGS} ${SUPPRESS_WARNINGS_FLAGS}")

# grab the files
file(GLOB_RECURSE test_sources "test/*\.h" "test/*\.m" "test/*\.mm" "test/*\.swift")

add_library(OpenCVTest STATIC dummy/dummy.mm)

enable_language(Swift)

# XCTest for Framework
xctest_add_bundle(OpenCVTestTests OpenCVTest ${test_sources} test/resources ${FRAMEWORK_DIR}/${FRAMEWORK_NAME}.framework)
set_target_properties(OpenCVTestTests PROPERTIES
    XCODE_ATTRIBUTE_FRAMEWORK_SEARCH_PATHS "${FRAMEWORK_DIR}"
    XCODE_ATTRIBUTE_OTHER_SWIFT_FLAGS "-Xcc ${SUPPRESS_WARNINGS_FLAGS}"
    XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "${FRAMEWORK_DIR}"
    )

# link necessary Frameworks
target_link_libraries(OpenCVTestTests PRIVATE "-framework Accelerate")
target_link_libraries(OpenCVTestTests PRIVATE "-framework AVFoundation")
target_link_libraries(OpenCVTestTests PRIVATE "-framework CoreMedia")
if (NOT DEFINED IOS_ARCH)
  target_link_libraries(OpenCVTestTests PRIVATE "-framework OpenCL")
endif()

# its OK to ignore stuff we don't know about since there will be no tests relying on it
set_target_properties(OpenCVTestTests PROPERTIES LINK_FLAGS -Wl,-undefined,dynamic_lookup)

set_source_files_properties(test/resources PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

xctest_add_test(XCTest.OpenCVTest OpenCVTestTests)
