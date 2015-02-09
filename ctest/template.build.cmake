#
# Mandatory settings (must be set)
#

set(CTEST_TARGET_SYSTEM                      "@CTEST_TARGET_SYSTEM@")
set(CTEST_MODEL                              "@MODEL@")

set(CTEST_SITE                               "@CTEST_SITE@")
set(CTEST_BUILD_NAME                         "@CTEST_BUILD_NAME@")

set(CTEST_DASHBOARD_ROOT                     "@CMAKE_CURRENT_BINARY_DIR@")
set(CTEST_SOURCE_DIRECTORY                   "@CMAKE_SOURCE_DIR@")
set(CTEST_BINARY_DIRECTORY                   "@CMAKE_BINARY_DIR@")

#
# Repository settings (mandatory, if the script should support checkout/update steps)
#

set(CTEST_WITH_UPDATE                        FALSE)
set(CTEST_GIT_COMMAND                        "@GIT_EXECUTABLE@")

#
# Project settings (optional)
#

set(OPENCV_TEST_DATA_PATH                    "@OPENCV_TEST_DATA_PATH@")

set(OPENCV_EXTRA_MODULES                     "")
set(OPENCV_EXTRA_MODULES_PATH                @OPENCV_EXTRA_MODULES_PATH@)

#
# Testing settings (optional)
#

set(CTEST_UPDATE_CMAKE_CACHE                 FALSE)
set(CTEST_EMPTY_BINARY_DIRECTORY             FALSE)
set(CTEST_WITH_TESTS                         @WITH_TESTS@)
set(CTEST_WITH_MEMCHECK                      @WITH_MEMCHECK@)
set(CTEST_WITH_COVERAGE                      @WITH_COVERAGE@)
set(CTEST_WITH_SUBMIT                        @CTEST_WITH_SUBMIT@)

set(CTEST_CMAKE_GENERATOR                    "@CMAKE_GENERATOR@")
set(CTEST_CONFIGURATION_TYPE                 "@CMAKE_BUILD_TYPE@")
set(CTEST_BUILD_FLAGS                        @CTEST_BUILD_FLAGS@)

set(CTEST_MEMORYCHECK_COMMAND                "@CTEST_MEMORYCHECK_COMMAND@")
set(CTEST_MEMORYCHECK_COMMAND_OPTIONS        @CTEST_MEMORYCHECK_COMMAND_OPTIONS@)

set(CTEST_COVERAGE_COMMAND                   "@CTEST_COVERAGE_COMMAND@")
set(CTEST_COVERAGE_EXTRA_FLAGS               "@CTEST_COVERAGE_EXTRA_FLAGS@")

#
# Include common part of client scripts
#

include("${CTEST_SCRIPT_DIRECTORY}/opencv_test.cmake")
