#
# Check supported targets and models
#

check_if_matches(CTEST_TARGET_SYSTEM    "^Linux" "^Windows" "^Android" "^MacOS")
check_if_matches(CTEST_MODEL            "^Nightly$" "^Experimental$" "^Continuous$" "^Release$" "^Performance$" "^MemCheck$" "^Documentation$")

#
# Configure the testing model
#

set_ifndef(CTEST_UPDATE_CMAKE_CACHE TRUE)
set_ifndef(CTEST_WITH_SUBMIT        FALSE)

if(CTEST_MODEL MATCHES "(Performance|MemCheck)")
    # 4 hours
    set_ifndef(CTEST_TEST_TIMEOUT   14400)
else()
    # 1 hour
    set_ifndef(CTEST_TEST_TIMEOUT   3600)
endif()

if(CTEST_MODEL MATCHES "Continuous")
    set_ifndef(CTEST_EMPTY_BINARY_DIRECTORY FALSE)
else()
    set_ifndef(CTEST_EMPTY_BINARY_DIRECTORY TRUE)
endif()

if(CTEST_TARGET_SYSTEM MATCHES "(Android|cross)" OR CTEST_MODEL MATCHES "(MemCheck|Documentation)")
    set_ifndef(CTEST_WITH_TESTS FALSE)
else()
    set_ifndef(CTEST_WITH_TESTS TRUE)
endif()

set_ifndef(CTEST_WITH_GCOVR             FALSE)
set_ifndef(CTEST_WITH_LCOV              FALSE)
if(NOT CTEST_MODEL MATCHES "Nightly")
    set_ifndef(CTEST_WITH_COVERAGE      FALSE)
else()
    if(CTEST_COVERAGE_COMMAND)
        set_ifndef(CTEST_WITH_COVERAGE  TRUE)
    else()
        set_ifndef(CTEST_WITH_COVERAGE  FALSE)
    endif()
endif()

if(CTEST_MODEL MATCHES "MemCheck")
    set_ifndef(CTEST_WITH_MEMCHECK  TRUE)
else()
    set_ifndef(CTEST_WITH_MEMCHECK  FALSE)
endif()

if(CTEST_WITH_COVERAGE AND CTEST_COVERAGE_COMMAND MATCHES "gcov")
    set_ifndef(CTEST_COVERAGE_EXTRA_FLAGS "-l")
endif()

#
# Configure extra/contrib repositories
#

if(NOT OPENCV_TEST_DATA_PATH)
    set_ifndef(OPENCV_EXTRA_SOURCE_DIRECTORY    "${CTEST_DASHBOARD_ROOT}/extra")
    set_ifndef(OPENCV_EXTRA_GIT_URL             "https://github.com/Itseez/opencv_extra.git")
    set_ifndef(OPENCV_EXTRA_GIT_BRANCH          "master")
    set_ifndef(OPENCV_TEST_DATA_PATH            "${OPENCV_EXTRA_SOURCE_DIRECTORY}/testdata")
endif()

set_ifndef(OPENCV_contrib_GIT_URL               "https://github.com/Itseez/opencv_contrib.git")
set_ifndef(OPENCV_contrib_GIT_BRANCH            "master")

#
# Configure OpenCV options
#

if(CTEST_UPDATE_CMAKE_CACHE)
    if(CTEST_TARGET_SYSTEM MATCHES "Android")
        set_ifndef(OPENCV_BUILD_SHARED_LIBS FALSE)
    else()
        set_ifndef(OPENCV_BUILD_SHARED_LIBS TRUE)
    endif()

    if(CTEST_MODEL MATCHES "Nightly")
        set_ifndef(OPENCV_BUILD_EXAMPLES TRUE)
    else()
        set_ifndef(OPENCV_BUILD_EXAMPLES FALSE)
    endif()

    if(OPENCV_FEATURES_ONLY)
        list(REMOVE_DUPLICATES OPENCV_FEATURES_ONLY)

        set(OPENCV_FEATURES_ENABLE "")
        set(OPENCV_FEATURES_DISABLE "")

        set(ALL_WITH_OPTIONS "")
        file(STRINGS "${CTEST_SOURCE_DIRECTORY}/CMakeLists.txt" cmake_lists)
        foreach(line ${cmake_lists})
            string(REGEX MATCH "OCV_OPTION\\(WITH_([A-Z0-9_]+) " output ${line})
            if(output)
                list(APPEND ALL_WITH_OPTIONS "${CMAKE_MATCH_1}")
            endif()
        endforeach()

        foreach(item ${ALL_WITH_OPTIONS})
            if(item MATCHES "${OPENCV_FEATURES_ONLY}")
                list(APPEND OPENCV_FEATURES_ENABLE "${item}")
            else()
                list(APPEND OPENCV_FEATURES_DISABLE "${item}")
            endif()
        endforeach()
    else()
        if(CTEST_TARGET_SYSTEM MATCHES "cross")
            list(APPEND OPENCV_FEATURES_DISABLE "VTK")
        endif()

        if(OPENCV_FEATURES_ENABLE)
            list(REMOVE_DUPLICATES OPENCV_FEATURES_ENABLE)
        endif()
        if(OPENCV_FEATURES_DISABLE)
            list(REMOVE_DUPLICATES OPENCV_FEATURES_DISABLE)
        endif()
        foreach(item ${OPENCV_FEATURES_DISABLE})
            if(OPENCV_FEATURES_ENABLE MATCHES "${item}")
                list(REMOVE_ITEM OPENCV_FEATURES_ENABLE "${item}")
            endif()
        endforeach()
    endif()
endif()

#
# Checkout/update opencv_extra and opencv_contrib if needed
#

if(OPENCV_EXTRA_SOURCE_DIRECTORY)
    ctext_ext_log_stage("Checkout/update testdata")

    if(NOT EXISTS "${OPENCV_EXTRA_SOURCE_DIRECTORY}")
        check_vars_def(OPENCV_EXTRA_GIT_URL OPENCV_EXTRA_GIT_BRANCH)

        clone_git_repo("${OPENCV_EXTRA_GIT_URL}" "${OPENCV_EXTRA_SOURCE_DIRECTORY}"
            BRANCH "${OPENCV_EXTRA_GIT_BRANCH}")

        set(HAVE_UPDATES TRUE)
    elseif(CTEST_WITH_UPDATE AND CTEST_STAGE MATCHES "Start")
        check_vars_def(OPENCV_EXTRA_GIT_BRANCH)

        update_git_repo("${OPENCV_EXTRA_SOURCE_DIRECTORY}"
            BRANCH "${OPENCV_EXTRA_GIT_BRANCH}"
            UPDATE_COUNT_OUTPUT update_count)

        if(update_count GREATER "0")
            set(HAVE_UPDATES TRUE)
        endif()
    endif()
endif()

if(OPENCV_EXTRA_MODULES)
    ctext_ext_log_stage("Checkout/update extra modules")

    foreach(module ${OPENCV_EXTRA_MODULES})
        set_ifndef(OPENCV_${module}_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${module}")
        set_ifndef(OPENCV_${module}_MODULES_DIRECTORY "${OPENCV_${module}_SOURCE_DIRECTORY}/modules")

        list(APPEND OPENCV_EXTRA_MODULES_PATH "${OPENCV_${module}_MODULES_DIRECTORY}")

        if(NOT EXISTS "${OPENCV_${module}_SOURCE_DIRECTORY}")
            check_vars_def(OPENCV_${module}_GIT_URL OPENCV_${module}_GIT_BRANCH)

            clone_git_repo("${OPENCV_${module}_GIT_URL}" "${OPENCV_${module}_SOURCE_DIRECTORY}"
                BRANCH "${OPENCV_${module}_GIT_BRANCH}")

            set(HAVE_UPDATES TRUE)
        elseif(CTEST_WITH_UPDATE AND CTEST_STAGE MATCHES "Start")
            check_vars_def(OPENCV_${module}_GIT_BRANCH)

            update_git_repo("${OPENCV_${module}_SOURCE_DIRECTORY}"
                BRANCH "${OPENCV_${module}_GIT_BRANCH}"
                UPDATE_COUNT_OUTPUT update_count)

            if(update_count GREATER "0")
                set(HAVE_UPDATES TRUE)
            endif()
        endif()
    endforeach()
endif()

#
# Checks for Continuous model
#

set(IS_CONTINUOUS FALSE)
if(CTEST_MODEL MATCHES "Continuous")
    set(IS_CONTINUOUS TRUE)
endif()

set(IS_BINARY_EMPTY FALSE)
if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")
    set(IS_BINARY_EMPTY TRUE)
endif()

if(IS_CONTINUOUS AND NOT IS_BINARY_EMPTY AND NOT HAVE_UPDATES)
    ctest_ext_info("Continuous model : no updates")
    return()
endif()

#
# Set CMake options
#

if(CTEST_UPDATE_CMAKE_CACHE)
    if(CTEST_TARGET_SYSTEM MATCHES "Windows")
        if(CTEST_TARGET_SYSTEM MATCHES "64")
            set_ifndef(CTEST_CMAKE_GENERATOR "Visual Studio 12 Win64")
        else()
            set_ifndef(CTEST_CMAKE_GENERATOR "Visual Studio 12")
        endif()
    else()
        set_ifndef(CTEST_CMAKE_GENERATOR "Unix Makefiles")
    endif()

    if(CTEST_MODEL MATCHES "(Release|Performance)")
        set_ifndef(CTEST_CONFIGURATION_TYPE "Release")
    else()
        set_ifndef(CTEST_CONFIGURATION_TYPE "Debug")
    endif()

    if(CTEST_TARGET_SYSTEM MATCHES "Android")
        add_cmake_cache_entry("CMAKE_TOOLCHAIN_FILE" TYPE "FILEPATH" "${CTEST_SOURCE_DIRECTORY}/platforms/android/android.toolchain.cmake")
    elseif(CTEST_TARGET_SYSTEM MATCHES "Linux.*ARMHF-cross")
        add_cmake_cache_entry("CMAKE_TOOLCHAIN_FILE" TYPE "FILEPATH" "${CTEST_SOURCE_DIRECTORY}/platforms/linux/arm-gnueabi.toolchain.cmake")
    endif()

    if(OPENCV_TEST_DATA_PATH AND EXISTS "${OPENCV_TEST_DATA_PATH}")
        add_cmake_cache_entry("OPENCV_TEST_DATA_PATH" TYPE "PATH" "${OPENCV_TEST_DATA_PATH}")
    endif()

    if(OPENCV_EXTRA_MODULES_PATH)
        add_cmake_cache_entry("OPENCV_EXTRA_MODULES_PATH" TYPE "STRING" "${OPENCV_EXTRA_MODULES_PATH}")
    endif()

    add_cmake_cache_entry("ENABLE_CTEST" TYPE "BOOL" "ON")

    if(CTEST_WITH_COVERAGE OR CTEST_WITH_GCOVR OR CTEST_WITH_LCOV)
        add_cmake_cache_entry("ENABLE_COVERAGE" TYPE "BOOL" "ON")
    else()
        add_cmake_cache_entry("ENABLE_COVERAGE" TYPE "BOOL" "OFF")
    endif()

    add_cmake_cache_entry("BUILD_SHARED_LIBS" TYPE "BOOL" "${OPENCV_BUILD_SHARED_LIBS}")
    add_cmake_cache_entry("BUILD_EXAMPLES"    TYPE "BOOL" "${OPENCV_BUILD_EXAMPLES}")
    add_cmake_cache_entry("BUILD_TESTS"       TYPE "BOOL" "ON")
    add_cmake_cache_entry("BUILD_PERF_TESTS"  TYPE "BOOL" "ON")

    if(CTEST_MODEL MATCHES "Documentation")
        add_cmake_cache_entry("BUILD_DOCS" TYPE "BOOL" "ON")
    endif()

    foreach(item ${OPENCV_FEATURES_ENABLE})
        add_cmake_cache_entry("WITH_${item}" TYPE "BOOL" "ON")
    endforeach()

    foreach(item ${OPENCV_FEATURES_DISABLE})
        add_cmake_cache_entry("WITH_${item}" TYPE "BOOL" "OFF")
    endforeach()

    if(CTEST_MODEL MATCHES "Release")
        add_cmake_cache_entry("INSTALL_TESTS" TYPE "BOOL" "ON")

        if(CTEST_TARGET_SYSTEM MATCHES "Windows")
            add_cmake_cache_entry("CPACK_GENERATOR" TYPE "STRING" "ZIP")
        else()
            add_cmake_cache_entry("CPACK_GENERATOR" TYPE "STRING" "TGZ")
        endif()
    endif()
endif()

#
# Start testing
#

ctest_ext_start()

if(CTEST_STAGE MATCHES "Start")
    ctest_ext_note("OPENCV_EXTRA_SOURCE_DIRECTORY         : ${OPENCV_EXTRA_SOURCE_DIRECTORY}")
    ctest_ext_note("OPENCV_EXTRA_GIT_URL                  : ${OPENCV_EXTRA_GIT_URL}")
    ctest_ext_note("OPENCV_EXTRA_GIT_BRANCH               : ${OPENCV_EXTRA_GIT_BRANCH}")
    ctest_ext_note("OPENCV_TEST_DATA_PATH                 : ${OPENCV_TEST_DATA_PATH}")
    ctest_ext_note("")

    ctest_ext_note("OPENCV_EXTRA_MODULES                  : ${OPENCV_EXTRA_MODULES}")
    foreach(module ${OPENCV_EXTRA_MODULES})
        ctest_ext_note("OPENCV_${module}_SOURCE_DIRECTORY     : ${OPENCV_${module}_SOURCE_DIRECTORY}")
        ctest_ext_note("OPENCV_${module}_MODULES_DIRECTORY    : ${OPENCV_${module}_MODULES_DIRECTORY}")
        ctest_ext_note("OPENCV_${module}_GIT_URL              : ${OPENCV_${module}_GIT_URL}")
        ctest_ext_note("OPENCV_${module}_GIT_BRANCH           : ${OPENCV_${module}_GIT_BRANCH}")
    endforeach()
    ctest_ext_note("OPENCV_EXTRA_MODULES_PATH             : ${OPENCV_EXTRA_MODULES_PATH}")
    ctest_ext_note("")

    ctest_ext_note("OPENCV_BUILD_SHARED_LIBS              : ${OPENCV_BUILD_SHARED_LIBS}")
    ctest_ext_note("OPENCV_BUILD_EXAMPLES                 : ${OPENCV_BUILD_EXAMPLES}")
    ctest_ext_note("")

    ctest_ext_note("OPENCV_FEATURES_ONLY                  : ${OPENCV_FEATURES_ONLY}")
    ctest_ext_note("OPENCV_FEATURES_ENABLE                : ${OPENCV_FEATURES_ENABLE}")
    ctest_ext_note("OPENCV_FEATURES_DISABLE               : ${OPENCV_FEATURES_DISABLE}")
    ctest_ext_note("")
endif()

#
# Remove previous test reports
#

set(TEST_REPORTS_DIR "${CTEST_BINARY_DIRECTORY}/test-reports")
set(ACCURACY_REPORTS_DIR "${TEST_REPORTS_DIR}/accuracy")
set(PERF_REPORTS_DIR "${TEST_REPORTS_DIR}/performance")
set(SANITY_REPORTS_DIR "${TEST_REPORTS_DIR}/sanity")

if(CTEST_STAGE MATCHES "Configure")
    if(NOT CTEST_EMPTY_BINARY_DIRECTORY AND EXISTS "${TEST_REPORTS_DIR}")
        file(REMOVE_RECURSE "${TEST_REPORTS_DIR}")
    endif()
endif()

#
# Configure
#

ctest_ext_configure()

#
# Build
#

if(CTEST_MODEL MATCHES "Release")
    ctest_ext_build(TARGETS "ALL" "package")
elseif(CTEST_MODEL MATCHES "Documentation")
    ctest_ext_build(TARGET "doxygen")
else()
    ctest_ext_build()
endif()

#
# Test
#

if(CTEST_MODEL MATCHES "Performance")
    ctest_ext_test(
        INCLUDE_LABEL "Performance")
else()
    ctest_ext_test(
        EXCLUDE_LABEL "Performance"
        EXCLUDE "^(opencv_test_viz|opencv_test_highgui|opencv_test_shape|opencv_sanity_videoio)$")
endif()

#
# Coverage
#

ctest_ext_coverage(CTEST LABELS "Module")

#
# MemCheck
#

ctest_ext_memcheck(
    INCLUDE_LABEL "Accuracy"
    EXCLUDE "^(opencv_test_viz|opencv_test_highgui|opencv_test_shape)$")

#
# Submit
#

if(CTEST_MODEL MATCHES "Performance")
    file(GLOB perf_xmls "${PERF_REPORTS_DIR}/*.xml")
    list(APPEND CTEST_UPLOAD_FILES ${perf_xmls})
endif()

ctest_ext_submit()
