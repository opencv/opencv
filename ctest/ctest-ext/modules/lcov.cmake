#
# The MIT License (MIT)
#
# Copyright (c) 2015 Vladislav Vinogradov
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
# run_lcov([BRANCH_COVERAGE] [FUNCTION_COVERAGE]
#          [SKIP_HTML]
#          [OUTPUT_LCOV_DIR <output_lcov_dir>]
#          [OUTPUT_HTML_DIR <output_html_dir>]
#          [EXTRACT] <extract patterns>
#          [REMOVE] <remove patterns>
#          [EXTRA_LCOV_OPTIONS <lcov extra options>]
#          [EXTRA_GENTHML_OPTIONS <genhtml extra options>])
#
#   Runs `lcov` and `genthml` commands to generate coverage report.
#
#   This is an internal function, which is used in `ctest_ext_coverage`.
#
#   `BRANCH_COVERAGE` and `FUNCTION_COVERAGE` options turn on branch and function coverage analysis.
#
#   `SKIP_HTML` disables html report generation.
#
#   The `lcov` command is run in `CTEST_BINARY_DIRECTORY` directory relatively to `CTEST_SOURCE_DIRECTORY` directory.
#   The binaries must be built with `gcov` coverage support.
#   The `lcov` command must be run after all tests.
#
#   If `CTEST_LCOV_<option_name>` variable if defined, it will override the value of
#   `<option_name>` option.
#
#   `CTEST_LCOV_EXECUTABLE` variable must be defined and must point to `lcov` command.
#   `CTEST_GENHTML_EXECUTABLE` variable must be defined and must point to `genhtml` command.
#

function(run_lcov)
    set(options "BRANCH_COVERAGE" "FUNCTION_COVERAGE" "SKIP_HTML")
    set(oneValueArgs "OUTPUT_LCOV_DIR" "OUTPUT_REPORT_NAME" "OUTPUT_HTML_DIR")
    set(multiValueArgs "EXTRACT" "REMOVE" "EXTRA_LCOV_OPTIONS" "EXTRA_GENTHML_OPTIONS")
    cmake_parse_arguments(LCOV "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    override_from_ctest_vars(
        LCOV_BRANCH_COVERAGE
        LCOV_FUNCTION_COVERAGE
        LCOV_SKIP_HTML
        LCOV_OUTPUT_LCOV_DIR
        LCOV_OUTPUT_REPORT_NAME
        LCOV_OUTPUT_HTML_DIR
        LCOV_EXTRACT
        LCOV_REMOVE
        LCOV_EXTRA_LCOV_OPTIONS
        LCOV_EXTRA_GENTHML_OPTIONS)

    check_vars_exist(CTEST_LCOV_EXECUTABLE)
    if(NOT LCOV_SKIP_HTML)
        check_vars_exist(CTEST_GENHTML_EXECUTABLE)
    endif()

    set_ifndef(LCOV_OUTPUT_LCOV_DIR "${CTEST_BINARY_DIRECTORY}/coverage-lcov/plain")
    set_ifndef(LCOV_OUTPUT_REPORT_NAME "coverage.info")
    set_ifndef(LCOV_OUTPUT_HTML_DIR "${CTEST_BINARY_DIRECTORY}/coverage-lcov/html")

    set(LCOV_OPTIONS "--quiet")
    set(LCOV_GENTHML_OPTIONS "--demangle-cpp")
    if(LCOV_BRANCH_COVERAGE)
        list(APPEND LCOV_OPTIONS "--rc" "lcov_branch_coverage=1")
        list(APPEND LCOV_GENTHML_OPTIONS "--branch-coverage")
    else()
        list(APPEND LCOV_OPTIONS "--rc" "lcov_branch_coverage=0")
        list(APPEND LCOV_GENTHML_OPTIONS "--no-branch-coverage")
    endif()
    if(LCOV_FUNCTION_COVERAGE)
        list(APPEND LCOV_OPTIONS "--rc" "lcov_function_coverage=1")
        list(APPEND LCOV_GENTHML_OPTIONS "--function-coverage" "--demangle-cpp")
    else()
        list(APPEND LCOV_OPTIONS "--rc" "lcov_function_coverage=0")
        list(APPEND LCOV_GENTHML_OPTIONS "--no-function-coverage")
    endif()

    if(EXISTS "${LCOV_OUTPUT_LCOV_DIR}")
        file(REMOVE_RECURSE "${LCOV_OUTPUT_LCOV_DIR}")
    endif()
    if(EXISTS "${LCOV_OUTPUT_HTML_DIR}")
        file(REMOVE_RECURSE "${LCOV_OUTPUT_HTML_DIR}")
    endif()

    set(LCOV_COMMAND_LINE
        "${CTEST_LCOV_EXECUTABLE}"
        "--capture" "--no-external"
        "--directory" "${CTEST_BINARY_DIRECTORY}"
        "--base-directory" "${CTEST_SOURCE_DIRECTORY}"
        "--output-file" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}"
        ${LCOV_OPTIONS}
        ${LCOV_EXTRA_LCOV_OPTIONS})
    ctest_ext_info("Generate LCOV trace file : ${LCOV_COMMAND_LINE}")
    execute_process(COMMAND ${LCOV_COMMAND_LINE} WORKING_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

    if(LCOV_EXTRACT)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" "-E" "copy" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}.tmp"
            WORKING_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

        set(LCOV_COMMAND_LINE
            "${CTEST_LCOV_EXECUTABLE}"
            "--extract" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}.tmp"
            ${LCOV_EXTRACT}
            "--output-file" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}"
            ${LCOV_OPTIONS}
            ${LCOV_EXTRA_LCOV_OPTIONS})
        ctest_ext_info("Extract pattern from LCOV report : ${LCOV_COMMAND_LINE}")
        execute_process(COMMAND ${LCOV_COMMAND_LINE} WORKING_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

        file(REMOVE "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}.tmp")
    endforeach()

    if(LCOV_REMOVE)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" "-E" "copy" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}.tmp"
            WORKING_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

        set(LCOV_COMMAND_LINE
            "${CTEST_LCOV_EXECUTABLE}"
            "--remove" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}.tmp"
            ${LCOV_REMOVE}
            "--output-file" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}"
            ${LCOV_OPTIONS}
            ${LCOV_EXTRA_LCOV_OPTIONS})
        ctest_ext_info("Remove pattern from LCOV report : ${LCOV_COMMAND_LINE}")
        execute_process(COMMAND ${LCOV_COMMAND_LINE} WORKING_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

        file(REMOVE "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}.tmp")
    endforeach()

    if(NOT LCOV_SKIP_HTML)
        set(GENHTML_COMMAND_LINE
            "${CTEST_GENHTML_EXECUTABLE}" "${LCOV_OUTPUT_LCOV_DIR}/${LCOV_OUTPUT_REPORT_NAME}"
            "--prefix" "${CTEST_SOURCE_DIRECTORY}"
            "--output-directory" "${LCOV_OUTPUT_HTML_DIR}"
            ${LCOV_OPTIONS}
            ${LCOV_EXTRA_LCOV_OPTIONS}
            ${LCOV_GENTHML_OPTIONS}
            ${LCOV_EXTRA_GENTHML_OPTIONS})
        ctest_ext_info("Convert LCOV report to HTML : ${GENHTML_COMMAND_LINE}")
        execute_process(COMMAND ${GENHTML_COMMAND_LINE} WORKING_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
    endif()
endfunction()
