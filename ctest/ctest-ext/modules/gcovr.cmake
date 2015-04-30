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
# gcovr coverage report commands
#

#
# run_gcovr([XML] [HTML]
#           [FILTER <filter>]
#           [OUTPUT_BASE_NAME <output_dir>]
#           [XML_DIR <xml output dir>]
#           [HTML_DIR <html output dir>]
#           [EXTRA_OPTIONS <option1> <option2> ...])
#
#   Runs gcovr command to generate coverage report.
#
#   This is an internal function, which is used in `ctest_ext_coverage`.
#
#   The gcovr command is run in `CTEST_BINARY_DIRECTORY` directory relatively to `CTEST_SOURCE_DIRECTORY` directory.
#   The binaries must be built with gcov coverage support.
#   The gcovr command must be run after all tests.
#
#   Coverage reports will be generated in:
#
#     - <XML_DIR>/<OUTPUT_BASE_NAME>.xml
#     - <HTML_DIR>/<OUTPUT_BASE_NAME>.html
#
#   `XML` and `HTML` options choose coverage report format (both can be specified).
#
#   `FILTER` options is used to specify file filter for report.
#   If not specified `${CTEST_SOURCE_DIRECTORY}/*` will be used.
#
#   `OUTPUT_BASE_NAME` specifies base name for output reports.
#   If not specified `coverage` will be used.
#
#   `XML_DIR` specifies base directory for XML reports.
#   If not specified `${CTEST_BINARY_DIRECTORY}/coverage-gcovr/xml` will be used.
#
#   `HTML_DIR` specifies base directory for HTML reports.
#   If not specified `${CTEST_BINARY_DIRECTORY}/coverage-gcovr/html` will be used.
#
#   `EXTRA_OPTIONS` specifies additional options for gcovr command line.
#
#   If `CTEST_GCOVR_<option_name>` variable if defined, it will override the value of
#   `<option_name>` option.
#
#   `CTEST_GCOVR_EXECUTABLE` variable must be defined and must point to gcovr command.
#

function(run_gcovr)
    set(options "XML" "HTML")
    set(oneValueArgs "FILTER" "OUTPUT_BASE_NAME" "XML_DIR" "HTML_DIR")
    set(multiValueArgs "EXTRA_OPTIONS")
    cmake_parse_arguments(GCOVR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    override_from_ctest_vars(
        GCOVR_XML
        GCOVR_HTML
        GCOVR_FILTER
        GCOVR_OUTPUT_BASE_NAME
        GCOVR_XML_DIR
        GCOVR_HTML_DIR
        GCOVR_EXTRA_OPTIONS)

    check_vars_exist(CTEST_GCOVR_EXECUTABLE)

    set_ifndef(GCOVR_FILTER "${CTEST_SOURCE_DIRECTORY}/*")
    set_ifndef(GCOVR_OUTPUT_BASE_NAME "coverage")
    set_ifndef(GCOVR_XML_DIR "${CTEST_BINARY_DIRECTORY}/coverage-gcovr/xml")
    set_ifndef(GCOVR_HTML_DIR "${CTEST_BINARY_DIRECTORY}/coverage-gcovr/html")

    set(GCOVR_BASE_COMMAND_LINE
        "${CTEST_GCOVR_EXECUTABLE}"
        "${CTEST_BINARY_DIRECTORY}"
        "-r" "${CTEST_SOURCE_DIRECTORY}"
        "-f" "${GCOVR_FILTER}"
        ${GCOVR_EXTRA_OPTIONS})

    if(GCOVR_XML)
        if(EXISTS "${GCOVR_XML_DIR}")
            file(REMOVE_RECURSE "${GCOVR_XML_DIR}")
        endif()
        file(MAKE_DIRECTORY "${GCOVR_XML_DIR}")

        set(GCOVR_XML_COMMAND_LINE
            ${GCOVR_BASE_COMMAND_LINE}
            "--xml" "--xml-pretty"
            "-o" "${GCOVR_OUTPUT_BASE_NAME}.xml")

        ctest_ext_info("Generate XML gcovr report : ${GCOVR_XML_COMMAND_LINE}")
        execute_process(COMMAND ${GCOVR_XML_COMMAND_LINE} WORKING_DIRECTORY "${GCOVR_XML_DIR}")
    endif()

    if(GCOVR_HTML)
        if(EXISTS "${GCOVR_HTML_DIR}")
            file(REMOVE_RECURSE "${GCOVR_HTML_DIR}")
        endif()
        file(MAKE_DIRECTORY "${GCOVR_HTML_DIR}")

        set(GCOVR_HTML_COMMAND_LINE
            ${GCOVR_BASE_COMMAND_LINE}
            "--html" "--html-details"
            "-o" "${GCOVR_OUTPUT_BASE_NAME}.html")

        ctest_ext_info("Generate HTML gcovr report : ${GCOVR_HTML_COMMAND_LINE}")
        execute_process(COMMAND ${GCOVR_HTML_COMMAND_LINE} WORKING_DIRECTORY "${GCOVR_HTML_DIR}")
    endif()
endfunction()
