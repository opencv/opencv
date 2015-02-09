# CTEST_TARGET_SYSTEM

if(ANDROID)
    set(TARGET_SYSTEM_INIT "Android")
else()
    set(TARGET_SYSTEM_INIT "${CMAKE_SYSTEM}")
endif()
if(CMAKE_CROSSCOMPILING AND NOT ANDROID) # Android is always cross-compiled
    set(TARGET_SYSTEM_INIT "${TARGET_SYSTEM_INIT}-cross")
endif()
set(CTEST_TARGET_SYSTEM "${TARGET_SYSTEM_INIT}" CACHE STRING "Target system for CTest submission")

# CTEST_SITE

site_name(CTEST_SITE)

# CTEST_BUILD_NAME

set(CTEST_BUILD_NAME "${CTEST_TARGET_SYSTEM}" CACHE STRING "Build name for CTest submission")

# CTEST_BUILD_FLAGS

set(CTEST_BUILD_FLAGS "" CACHE STRING "Build flags (like -j5) for CTest submission")

# CTEST_MEMORYCHECK_*

find_program(CTEST_MEMORYCHECK_COMMAND
    NAMES valgrind
    DOC "Path to the memory checking command, used for memory error detection.")
set(CTEST_MEMORYCHECK_COMMAND_OPTIONS "" CACHE STRING "Memory check tool options")

# CTEST_COVERAGE_*

find_program(CTEST_COVERAGE_COMMAND
    NAMES gcov
    DOC "Path to the coverage program that CTest uses for performing coverage inspection")
set(CTEST_COVERAGE_EXTRA_FLAGS "-l" CACHE STRING
    "Extra command line flags to pass to the coverage tool")

# CTEST_WITH_SUBMIT

option(CTEST_WITH_SUBMIT "Submit test results to remote server" OFF)

# CTEST_EXTRA_ARGS

set(CTEST_EXTRA_ARGS "" CACHE STRING "Extra flags for CTest tool")

# Targets

configure_file("CTestCustom.cmake" "${CMAKE_BINARY_DIR}/CTestCustom.cmake" COPYONLY)
configure_file("opencv_test.cmake" "${CMAKE_CURRENT_BINARY_DIR}/opencv_test.cmake" COPYONLY)
configure_file("ctest_ext.cmake" "${CMAKE_CURRENT_BINARY_DIR}/ctest_ext.cmake" COPYONLY)

function(add_ctest_target NAME MODEL WITH_TESTS WITH_MEMCHECK WITH_COVERAGE)
    set(script_file "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.cmake")
    configure_file("template.build.cmake" "${script_file}" @ONLY)

    file(GLOB scripts "*.cmake")

    add_custom_target("${NAME}"
        COMMAND "${CMAKE_CTEST_COMMAND}" -VV -S "${script_file}" ${CTEST_EXTRA_ARGS}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        SOURCES ${scripts} "${CMAKE_SOURCE_DIR}/CTestConfig.cmake")

    set_property(TARGET "${NAME}" PROPERTY FOLDER "CTestDashboardTargets")
endfunction()

add_ctest_target(Experimental "Experimental" TRUE FALSE FALSE)
add_ctest_target(ExperimentalMemCheck "Experimental" FALSE TRUE FALSE)
if(ENABLE_COVERAGE)
    add_ctest_target(ExperimentalCoverage "Experimental" TRUE FALSE TRUE)
endif()

add_ctest_target(Performance "Performance" TRUE FALSE FALSE)
