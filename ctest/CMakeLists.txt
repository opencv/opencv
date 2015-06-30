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

# CTEST_EXTRA_ARGS

set(CTEST_EXTRA_ARGS "" CACHE STRING "Extra flags for CTest tool")

# Targets

file(COPY "CTestCustom.cmake" DESTINATION "${CMAKE_BINARY_DIR}")
file(COPY "opencv_test.cmake" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "ctest-ext" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

function(add_ctest_target NAME MODEL)
    set(script_file "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.cmake")
    configure_file("template.build.cmake" "${script_file}" @ONLY)

    file(GLOB scripts "*.cmake")

    add_custom_target("${NAME}"
        COMMAND "${CMAKE_CTEST_COMMAND}" -VV -S "${script_file}" ${CTEST_EXTRA_ARGS}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        SOURCES ${scripts})

    set_property(TARGET "${NAME}" PROPERTY FOLDER "CTestDashboardTargets")
endfunction()

add_ctest_target(Experimental "Experimental")
add_ctest_target(Performance "Performance")
