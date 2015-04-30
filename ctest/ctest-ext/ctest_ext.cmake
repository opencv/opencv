#
# The MIT License (MIT)
#
# Copyright (c) 2014-2015 Vladislav Vinogradov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

#
# The CTest Extension module is a set of additional functions for CTest scripts.
# The main goal of the CTest Extension module is to provide uniform testing approach
# for CMake projects.
#
# The CTest Extension module supports the following functionality:
#
#   - clone/update git repository;
#   - configure CMake project;
#   - build CMake project;
#   - run project's tests;
#   - build coverage report (in CTest format and in gcovr format);
#   - run dynamic analysis (like valgrind);
#   - upload testing results to remote server (eg. CDash web server).
#

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(DEFINED CTEST_EXT_INCLUDED)
    return()
endif()
set(CTEST_EXT_INCLUDED TRUE)
set(CTEST_EXT_VERSION  0.6.1)

#
# Auxiliary modules
#

include(CMakeParseArguments)

set(CTEST_EXT_MODULES_PATH "${CMAKE_CURRENT_LIST_DIR}/modules")

include("${CTEST_EXT_MODULES_PATH}/vars.cmake")
include("${CTEST_EXT_MODULES_PATH}/logging.cmake")
include("${CTEST_EXT_MODULES_PATH}/system.cmake")
include("${CTEST_EXT_MODULES_PATH}/git_repo.cmake")
include("${CTEST_EXT_MODULES_PATH}/cmake_config.cmake")
include("${CTEST_EXT_MODULES_PATH}/gcovr.cmake")
include("${CTEST_EXT_MODULES_PATH}/lcov.cmake")

#
# ctest_ext_init()
#
#   Initializes CTest Ext module for dashboard testing.
#
#   The function sets dashboard settings to default values (if they were not defined prior the call)
#   and performs project repository checkout/update if needed.
#

macro(ctest_ext_init)
    check_vars_def(CTEST_PROJECT_NAME)

    #
    # Initialize settings from environment variables
    #

    set_from_env(
        CTEST_TARGET_SYSTEM
        CTEST_MODEL)

    set_from_env(
        CTEST_SITE
        CTEST_BUILD_NAME)

    set_from_env(
        CTEST_DASHBOARD_ROOT
        CTEST_SOURCE_DIRECTORY
        CTEST_BINARY_DIRECTORY
        CTEST_NOTES_LOG_FILE)

    set_from_env(
        CTEST_WITH_UPDATE
        CTEST_SCM_TOOL
        CTEST_GIT_COMMAND
        CTEST_PROJECT_GIT_URL
        CTEST_PROJECT_GIT_BRANCH)

    set_from_env(
        CTEST_UPDATE_CMAKE_CACHE
        CTEST_EMPTY_BINARY_DIRECTORY)

    set_from_env(
        CTEST_CMAKE_GENERATOR
        CTEST_CONFIGURATION_TYPE
        CTEST_INITIAL_CACHE
        CTEST_CMAKE_EXTRA_OPTIONS
        CTEST_BUILD_FLAGS)

    set_from_env(
        CTEST_WITH_TESTS
        CTEST_TEST_TIMEOUT)

    set_from_env(
        CTEST_WITH_COVERAGE
        CTEST_COVERAGE_TOOL)

    set_from_env(
        CTEST_GCOVR_EXECUTABLE
        CTEST_GCOVR_XML
        CTEST_GCOVR_HTML
        CTEST_GCOVR_FILTER
        CTEST_GCOVR_OUTPUT_BASE_NAME
        CTEST_GCOVR_XML_DIR
        CTEST_GCOVR_HTML_DIR
        CTEST_GCOVR_EXTRA_OPTIONS)

    set_from_env(
        CTEST_LCOV_EXECUTABLE
        CTEST_GENHTML_EXECUTABLE
        CTEST_LCOV_BRANCH_COVERAGE
        CTEST_LCOV_FUNCTION_COVERAGE
        CTEST_LCOV_SKIP_HTML
        CTEST_LCOV_OUTPUT_LCOV_DIR
        CTEST_LCOV_OUTPUT_REPORT_NAME
        CTEST_LCOV_OUTPUT_HTML_DIR
        CTEST_LCOV_EXTRACT
        CTEST_LCOV_REMOVE
        CTEST_LCOV_EXTRA_LCOV_OPTIONS
        CTEST_LCOV_EXTRA_GENTHML_OPTIONS)

    set_from_env(
        CTEST_COVERAGE_COMMAND
        CTEST_COVERAGE_EXTRA_FLAGS)

    set_from_env(
        CTEST_WITH_DYNAMIC_ANALYSIS
        CTEST_DYNAMIC_ANALYSIS_TOOL
        CTEST_MEMORYCHECK_COMMAND
        CTEST_MEMORYCHECK_SUPPRESSIONS_FILE
        CTEST_MEMORYCHECK_COMMAND_OPTIONS)

    set_from_env(
        CTEST_WITH_SUBMIT
        CTEST_NOTES_FILES
        CTEST_UPLOAD_FILES)

    set_from_env(
        CTEST_TRACK
        CTEST_TMP_BASE_DIR)

    #
    # Set dashboard setting to default values
    #

    set_ifndef(CTEST_TARGET_SYSTEM      "${CMAKE_SYSTEM}-${CMAKE_SYSTEM_PROCESSOR}")
    set_ifndef(CTEST_MODEL              "Experimental")

    if(NOT DEFINED CTEST_SITE)
        site_name(CTEST_SITE)
    endif()
    set_ifndef(CTEST_BUILD_NAME         "${CTEST_TARGET_SYSTEM}-${CTEST_MODEL}")

    set_ifndef(CTEST_DASHBOARD_ROOT     "${CTEST_SCRIPT_DIRECTORY}/${CTEST_TARGET_SYSTEM}/${CTEST_MODEL}")
    set_ifndef(CTEST_SOURCE_DIRECTORY   "${CTEST_DASHBOARD_ROOT}/source")
    set_ifndef(CTEST_BINARY_DIRECTORY   "${CTEST_DASHBOARD_ROOT}/build")
    set_ifndef(CTEST_NOTES_LOG_FILE     "${CTEST_DASHBOARD_ROOT}/ctest_ext_notes_log.txt")

    set_ifndef(CTEST_WITH_UPDATE        FALSE)
    if(EXISTS "${CTEST_SOURCE_DIRECTORY}/.git" OR DEFINED CTEST_PROJECT_GIT_URL)
        set_ifndef(CTEST_SCM_TOOL       "GIT")
    endif()

    #
    # Locate tools
    #

    if(NOT DEFINED CTEST_GIT_COMMAND)
        find_package(Git QUIET)
        if(GIT_FOUND)
            ctest_ext_info("Found git : ${GIT_EXECUTABLE} (version ${GIT_VERSION_STRING})")
            set_ifndef(CTEST_GIT_COMMAND "${GIT_EXECUTABLE}")
        endif()
    endif()

    if(NOT DEFINED CTEST_GCOVR_EXECUTABLE)
        find_program(CTEST_GCOVR_EXECUTABLE NAMES gcovr)
        if(CTEST_GCOVR_EXECUTABLE)
            ctest_ext_info("Found gcovr : ${CTEST_GCOVR_EXECUTABLE}")
        endif()
    endif()

    if(NOT DEFINED CTEST_LCOV_EXECUTABLE)
        find_program(CTEST_LCOV_EXECUTABLE NAMES lcov)
        if(CTEST_LCOV_EXECUTABLE)
            ctest_ext_info("Found lcov : ${CTEST_LCOV_EXECUTABLE}")
        endif()
    endif()

    if(NOT DEFINED CTEST_GENHTML_EXECUTABLE)
        find_program(CTEST_GENHTML_EXECUTABLE NAMES genhtml)
        if(CTEST_GENHTML_EXECUTABLE)
            ctest_ext_info("Found genhtml : ${CTEST_GENHTML_EXECUTABLE}")
        endif()
    endif()

    if(NOT DEFINED CTEST_COVERAGE_COMMAND)
        find_program(CTEST_COVERAGE_COMMAND NAMES gcov)
        if(CTEST_COVERAGE_COMMAND)
            ctest_ext_info("Found gcov : ${CTEST_COVERAGE_COMMAND}")
        endif()
    endif()

    if(NOT DEFINED CTEST_MEMORYCHECK_COMMAND)
        find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
        if(CTEST_MEMORYCHECK_COMMAND)
            ctest_ext_info("Found valgrind : ${CTEST_MEMORYCHECK_COMMAND}")
        endif()
    endif()

    #
    # Determine current stage
    #

    set_ifndef(CTEST_STAGE "${CTEST_SCRIPT_ARG}")
    if(NOT CTEST_STAGE)
        set(CTEST_STAGE "Start;Configure;Build;Test;Coverage;DynamicAnalysis;Submit;Extra")
    endif()

    #
    # Initialize sources
    #

    set(HAVE_UPDATES TRUE)

    if(CTEST_STAGE MATCHES "Start")
        ctext_ext_log_stage("Initialize testing for MODEL ${CTEST_MODEL}")

        if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
            if(NOT DEFINED CTEST_CHECKOUT_COMMAND)
                if(CTEST_SCM_TOOL MATCHES "GIT")
                    check_vars_exist(CTEST_GIT_COMMAND)
                    check_vars_def(CTEST_PROJECT_GIT_URL)

                    if(CTEST_PROJECT_GIT_BRANCH)
                        set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone -b ${CTEST_PROJECT_GIT_BRANCH} -- ${CTEST_PROJECT_GIT_URL} ${CTEST_SOURCE_DIRECTORY}")
                    else()
                        set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone ${CTEST_PROJECT_GIT_URL} ${CTEST_SOURCE_DIRECTORY}")
                    endif()
                else()
                    message(FATAL_ERROR "CTEST_SOURCE_DIRECTORY = ${CTEST_SOURCE_DIRECTORY} is not exist and no SCM configuration is provided")
                endif()
            endif()

            ctest_ext_info("Source directory (${CTEST_SOURCE_DIRECTORY}) is not exist, checkout it with [${CTEST_CHECKOUT_COMMAND}] command")
        endif()

        ctest_ext_info("Initial start and checkout")
        ctest_start("${CTEST_MODEL}")

        if(CTEST_WITH_UPDATE)
            if(NOT DEFINED CTEST_UPDATE_COMMAND)
                if(CTEST_SCM_TOOL MATCHES "GIT")
                    check_vars_exist(CTEST_GIT_COMMAND)

                    set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
                else()
                    message(FATAL_ERROR "Unsupported SCM tool : ${CTEST_SCM_TOOL}")
                endif()
            endif()

            ctest_ext_info("Repository update")
            ctest_update(RETURN_VALUE count)

            set(HAVE_UPDATES FALSE)
            if(count GREATER 0)
                set(HAVE_UPDATES TRUE)
            endif()
        endif()
    endif()
endmacro()

#
# ctest_ext_start()
#
#   Starts dashboard testing.
#
#   The function sets testing settings to default values (if they were not defined prior the call)
#   and initializes logging mechanism.
#

macro(ctest_ext_start)
    #
    # Set unspecified options to default values
    #

    set_ifndef(CTEST_UPDATE_CMAKE_CACHE         TRUE)
    set_ifndef(CTEST_EMPTY_BINARY_DIRECTORY     TRUE)
    set_ifndef(CTEST_CMAKE_GENERATOR            "Unix Makefiles")
    set_ifndef(CTEST_CONFIGURATION_TYPE         "Debug")

    set_ifndef(CTEST_WITH_TESTS                 TRUE)
    set_ifndef(CTEST_TEST_TIMEOUT               600)

    set_ifndef(CTEST_WITH_COVERAGE              FALSE)

    set_ifndef(CTEST_WITH_DYNAMIC_ANALYSIS      FALSE)

    set_ifndef(CTEST_WITH_SUBMIT                FALSE)
    list(APPEND CTEST_NOTES_FILES   "${CTEST_NOTES_LOG_FILE}")
    list(APPEND CTEST_UPLOAD_FILES  "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")

    set_ifndef(CTEST_TRACK "${CTEST_MODEL}")

    #
    # Start
    #

    ctext_ext_log_stage("Start testing MODEL ${CTEST_MODEL} TRACK ${CTEST_TRACK}")

    ctest_start("${CTEST_MODEL}" TRACK "${CTEST_TRACK}" APPEND)

    if(CTEST_STAGE MATCHES "Start")
        file(REMOVE "${CTEST_NOTES_LOG_FILE}")
        ctest_ext_dump_notes()
    endif()
endmacro()

#
# ctest_ext_configure()
#
#   Configures CMake project.
#
#   To configure CMake cache variables use `add_cmake_cache_entry` command.
#

macro(ctest_ext_configure)
    if(CTEST_STAGE MATCHES "Configure")
        ctext_ext_log_stage("Configure")

        if(CTEST_EMPTY_BINARY_DIRECTORY)
            ctest_ext_info("Clean binary directory : ${CTEST_BINARY_DIRECTORY}")

            if(EXISTS "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")
                ctest_empty_binary_directory("${CTEST_BINARY_DIRECTORY}")
            else()
                file(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}")
            endif()
        endif()

        if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
            ctest_ext_info("Create binary directory : ${CTEST_BINARY_DIRECTORY}")
            file(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
        endif()

        if(CTEST_UPDATE_CMAKE_CACHE)
            ctest_ext_info("Rewrite CMake cache : ${CTEST_INITIAL_CACHE}")
            file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" ${CTEST_INITIAL_CACHE})
        endif()

        ctest_configure(OPTIONS "${CTEST_CMAKE_EXTRA_OPTIONS}")
    endif()

    ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")
endmacro()

#
# ctest_ext_build([TARGET <target>] [TARGETS <target1> <target2> ...])
#
#   Builds CMake project.
#

function(ctest_ext_build)
    set(options "")
    set(oneValueArgs "TARGET")
    set(multiValueArgs "TARGETS")
    cmake_parse_arguments(BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(CTEST_STAGE MATCHES "Build")
        ctext_ext_log_stage("Build")

        if(BUILD_TARGET)
            ctest_ext_info("Build target : " ${BUILD_TARGET})
            ctest_build(TARGET "${BUILD_TARGET}")
        elseif(BUILD_TARGETS)
            ctest_ext_info("Build targets : " ${BUILD_TARGETS})

            # ctest_build doesn't support multiple target, emulate them with CMake script
            set(BUILD_SCRIPT "${CTEST_BINARY_DIRECTORY}/ctest_ext_build.cmake")
            file(REMOVE "${BUILD_SCRIPT}")

            foreach(target ${BUILD_TARGETS})
                file(APPEND "${BUILD_SCRIPT}" "message(STATUS \"Build target : ${target}\") \n")

                set(BUILD_COMMAND "execute_process(COMMAND \"${CMAKE_COMMAND}\"")
                set(BUILD_COMMAND "${BUILD_COMMAND} --build \"${CTEST_BINARY_DIRECTORY}\"")
                if(NOT target MATCHES "^(all|ALL)$")
                    set(BUILD_COMMAND "${BUILD_COMMAND} --target \"${target}\"")
                endif()
                set(BUILD_COMMAND "${BUILD_COMMAND} --config \"${CTEST_CONFIGURATION_TYPE}\"")
                if(CTEST_BUILD_FLAGS)
                    set(BUILD_COMMAND "${BUILD_COMMAND} -- ${CTEST_BUILD_FLAGS}")
                endif()

                set(BUILD_COMMAND "${BUILD_COMMAND} WORKING_DIRECTORY \"${CTEST_BINARY_DIRECTORY}\")")

                file(APPEND "${BUILD_SCRIPT}" "${BUILD_COMMAND} \n")
            endforeach()

            set(CTEST_BUILD_COMMAND "${CMAKE_COMMAND} -P ${BUILD_SCRIPT}")
            ctest_build()
        else()
            ctest_ext_info("Build target : ALL")
            ctest_build()
        endif()
    endif()
endfunction()

#
# ctest_ext_test(<arguments>)
#
#   Runs tests.
#
#   The function will pass its arguments to `ctest_test` as is.
#

function(ctest_ext_test)
    if(CTEST_WITH_TESTS AND CTEST_STAGE MATCHES "Test")
        ctext_ext_log_stage("Test")

        ctest_ext_info("ctest_test parameters : " ${ARGN})
        ctest_test(${ARGN})
    endif()
endfunction()

#
# ctest_ext_coverage(
#       [GCOVR <options for run_gcovr>]
#       [LCOV <options for run_lcov>]
#       [CDASH <options for ctest_coverage>])
#
#   Collects coverage reports.
#
#   The function passes own arguments to `run_gcovr`, `run_lcov` and `ctest_coverage` as is.
#

function(ctest_ext_coverage)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs "GCOVR" "LCOV" "CDASH")
    cmake_parse_arguments(COVERAGE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(CTEST_WITH_COVERAGE AND CTEST_STAGE MATCHES "Coverage")
        ctext_ext_log_stage("Coverage")

        if(CTEST_COVERAGE_TOOL MATCHES "GCOVR")
            ctest_ext_info("Generate GCOVR coverage report")

            ctest_ext_info("run_gcovr parameters : " ${COVERAGE_GCOVR})
            run_gcovr(${COVERAGE_GCOVR})
        endif()

        if(CTEST_COVERAGE_TOOL MATCHES "LCOV")
            ctest_ext_info("Generate LCOV coverage report")

            ctest_ext_info("run_lcov parameters : " ${COVERAGE_LCOV})
            run_lcov(${COVERAGE_LCOV})
        endif()

        if(CTEST_COVERAGE_TOOL MATCHES "CDASH")
            check_vars_def(CTEST_COVERAGE_COMMAND)

            ctest_ext_info("Generate CDASH coverage report")

            ctest_ext_info("ctest_coverage parameters : " ${COVERAGE_CDASH})
            ctest_coverage(${COVERAGE_CDASH})
        endif()
    endif()
endfunction()

#
# ctest_ext_dynamic_analysis(
#       [CDASH <options for ctest_memcheck>])
#
#   Runs dynamic analysis testing.
#
#   The function will pass its arguments to `ctest_memcheck` as is.
#

function(ctest_ext_dynamic_analysis)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs "CDASH")
    cmake_parse_arguments(DYNAMIC_ANALYSIS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(CTEST_WITH_DYNAMIC_ANALYSIS AND CTEST_STAGE MATCHES "DynamicAnalysis")
        ctext_ext_log_stage("Dynamic analysis")

        if(CTEST_DYNAMIC_ANALYSIS_TOOL MATCHES "CDASH")
            check_vars_def(CTEST_MEMORYCHECK_COMMAND)

            ctest_ext_info("Generate CDASH dynamic analysis report")

            ctest_ext_info("ctest_memcheck parameters : " ${ARGN})
            ctest_memcheck(${DYNAMIC_ANALYSIS_CDASH})
        endif()
    endif()
endfunction()

#
# ctest_ext_submit()
#
#   Submits testing results to remote server.
#

function(ctest_ext_submit)
    if(CTEST_WITH_SUBMIT AND CTEST_STAGE MATCHES "Submit")
        ctext_ext_log_stage("Submit")

        if(CTEST_UPLOAD_FILES)
            ctest_ext_info("Upload files : " ${CTEST_UPLOAD_FILES})
            ctest_upload(FILES ${CTEST_UPLOAD_FILES})
        endif()

        ctest_submit()
    endif()
endfunction()
