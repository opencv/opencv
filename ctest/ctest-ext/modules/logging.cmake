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
# Logging commands
#

set_from_env(
    CTEST_EXT_COLOR_OUTPUT
)

if(CTEST_EXT_COLOR_OUTPUT)
    string(ASCII 27 CTEST_EXT_TEXT_STYLE_ESC)

    set_ifndef(CTEST_EXT_TEXT_STYLE_RESET          "${CTEST_EXT_TEXT_STYLE_ESC}[m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLD           "${CTEST_EXT_TEXT_STYLE_ESC}[1m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_RED            "${CTEST_EXT_TEXT_STYLE_ESC}[31m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_GREEN          "${CTEST_EXT_TEXT_STYLE_ESC}[32m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_YELLOW         "${CTEST_EXT_TEXT_STYLE_ESC}[33m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BLUE           "${CTEST_EXT_TEXT_STYLE_ESC}[34m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_MAGENTA        "${CTEST_EXT_TEXT_STYLE_ESC}[35m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_CYAN           "${CTEST_EXT_TEXT_STYLE_ESC}[36m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_WHITE          "${CTEST_EXT_TEXT_STYLE_ESC}[37m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLDRED        "${CTEST_EXT_TEXT_STYLE_ESC}[1;31m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLD_GREEN     "${CTEST_EXT_TEXT_STYLE_ESC}[1;32m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLD_YELLOW    "${CTEST_EXT_TEXT_STYLE_ESC}[1;33m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLD_BLUE      "${CTEST_EXT_TEXT_STYLE_ESC}[1;34m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLD_MAGENTA   "${CTEST_EXT_TEXT_STYLE_ESC}[1;35m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLD_CYAN      "${CTEST_EXT_TEXT_STYLE_ESC}[1;36m")
    set_ifndef(CTEST_EXT_TEXT_STYLE_BOLD_WHITE     "${CTEST_EXT_TEXT_STYLE_ESC}[1;37m")
endif()

#
# ctest_ext_info(<message>)
#
#   Prints `<message>` to standard output with `[CTEST EXT INFO]` prefix for better visibility.
#

function(ctest_ext_info MSG)
    message("${CTEST_EXT_TEXT_STYLE_BOLD_BLUE}[CTEST EXT INFO] ${MSG}${CTEST_EXT_TEXT_STYLE_RESET}")
endfunction()

#
# ctext_ext_log_stage(<message>)
#
#   Log new stage start.
#

function(ctext_ext_log_stage MSG)
    message("${CTEST_EXT_TEXT_STYLE_BOLD_CYAN}[CTEST EXT STAGE] ==========================================================================${CTEST_EXT_TEXT_STYLE_RESET}")
    message("${CTEST_EXT_TEXT_STYLE_BOLD_CYAN}[CTEST EXT STAGE] ${MSG}")
    message("${CTEST_EXT_TEXT_STYLE_BOLD_CYAN}[CTEST EXT STAGE] ==========================================================================${CTEST_EXT_TEXT_STYLE_RESET}")
endfunction()

#
# ctest_ext_note(<message>)
#
#   Writes `<message>` both to console and to note file.
#   The function appends `[CTEST EXT NOTE]` prefix to console output for better visibility.
#   The note file is used in submit command.
#
#   The command will be available after `ctest_ext_start` call.
#
#   `CTEST_NOTES_LOG_FILE` variable must be defined.
#

function(ctest_ext_note MSG)
    check_vars_def(CTEST_NOTES_LOG_FILE)

    message("${CTEST_EXT_TEXT_STYLE_BOLD_MAGENTA}[CTEST EXT NOTE] ${MSG}${CTEST_EXT_TEXT_STYLE_RESET}")
    file(APPEND "${CTEST_NOTES_LOG_FILE}" "${MSG}\n")
endfunction()

#
# ctest_ext_dump_notes()
#
#   Dumps all CTest Ext launch options to note file.
#   This is an internal function, which is used by `ctest_ext_start`.
#

function(ctest_ext_dump_notes)
    check_vars_exist(CTEST_SOURCE_DIRECTORY)

    if(CTEST_SCM_TOOL MATCHES "GIT")
        get_git_repo_info("${CTEST_SOURCE_DIRECTORY}" GIT_CURRENT_BRANCH GIT_CURRENT_REVISION)
    endif()

    ctest_ext_note("CTest Ext configuration information:")
    ctest_ext_note("")

    ctest_ext_note("CTEST_EXT_VERSION                       : ${CTEST_EXT_VERSION}")
    ctest_ext_note("")

    ctest_ext_note("CTEST_PROJECT_NAME                      : ${CTEST_PROJECT_NAME}")
    ctest_ext_note("")

    ctest_ext_note("CTEST_TARGET_SYSTEM                     : ${CTEST_TARGET_SYSTEM}")
    ctest_ext_note("CTEST_MODEL                             : ${CTEST_MODEL}")
    ctest_ext_note("")

    ctest_ext_note("CTEST_SITE                              : ${CTEST_SITE}")
    ctest_ext_note("CTEST_BUILD_NAME                        : ${CTEST_BUILD_NAME}")
    ctest_ext_note("")

    ctest_ext_note("CTEST_DASHBOARD_ROOT                    : ${CTEST_DASHBOARD_ROOT}")
    ctest_ext_note("CTEST_SOURCE_DIRECTORY                  : ${CTEST_SOURCE_DIRECTORY}")
    ctest_ext_note("CTEST_BINARY_DIRECTORY                  : ${CTEST_BINARY_DIRECTORY}")
    ctest_ext_note("CTEST_NOTES_LOG_FILE                    : ${CTEST_NOTES_LOG_FILE}")
    ctest_ext_note("")

    ctest_ext_note("CTEST_WITH_UPDATE                       : ${CTEST_WITH_UPDATE}")
    if(CTEST_SCM_TOOL MATCHES "GIT")
        if(CTEST_WITH_UPDATE)
            ctest_ext_note("CTEST_GIT_COMMAND                       : ${CTEST_GIT_COMMAND}")
            ctest_ext_note("CTEST_PROJECT_GIT_URL                   : ${CTEST_PROJECT_GIT_URL}")
            ctest_ext_note("CTEST_PROJECT_GIT_BRANCH                : ${CTEST_PROJECT_GIT_BRANCH}")
        endif()
        ctest_ext_note("GIT_CURRENT_BRANCH                      : ${GIT_CURRENT_BRANCH}")
        ctest_ext_note("GIT_CURRENT_REVISION                    : ${GIT_CURRENT_REVISION}")
    endif()
    ctest_ext_note("")

    ctest_ext_note("CTEST_UPDATE_CMAKE_CACHE                : ${CTEST_UPDATE_CMAKE_CACHE}")
    ctest_ext_note("CTEST_EMPTY_BINARY_DIRECTORY            : ${CTEST_EMPTY_BINARY_DIRECTORY}")
    ctest_ext_note("")

    if(CTEST_UPDATE_CMAKE_CACHE)
        ctest_ext_note("CTEST_CMAKE_GENERATOR                   : ${CTEST_CMAKE_GENERATOR}")
        ctest_ext_note("CTEST_CONFIGURATION_TYPE                : ${CTEST_CONFIGURATION_TYPE}")
        ctest_ext_note("CTEST_INITIAL_CACHE                     : ${CTEST_INITIAL_CACHE}")
        ctest_ext_note("CTEST_CMAKE_EXTRA_OPTIONS               : ${CTEST_CMAKE_EXTRA_OPTIONS}")
    endif()
    ctest_ext_note("CTEST_BUILD_FLAGS                       : ${CTEST_BUILD_FLAGS}")
    ctest_ext_note("")

    ctest_ext_note("CTEST_WITH_TESTS                        : ${CTEST_WITH_TESTS}")
    if(CTEST_WITH_TESTS)
        ctest_ext_note("CTEST_TEST_TIMEOUT                      : ${CTEST_TEST_TIMEOUT}")
    endif()
    ctest_ext_note("")

    ctest_ext_note("CTEST_WITH_COVERAGE                     : ${CTEST_WITH_COVERAGE}")
    if(CTEST_WITH_COVERAGE)
        ctest_ext_note("CTEST_COVERAGE_TOOL                     : ${CTEST_COVERAGE_TOOL}")
        if(CTEST_COVERAGE_TOOL MATCHES "GCOVR")
            ctest_ext_note("CTEST_GCOVR_EXECUTABLE                  : ${CTEST_GCOVR_EXECUTABLE}")
            ctest_ext_note("CTEST_GCOVR_XML                         : ${CTEST_GCOVR_XML}")
            ctest_ext_note("CTEST_GCOVR_HTML                        : ${CTEST_GCOVR_HTML}")
            ctest_ext_note("CTEST_GCOVR_FILTER                      : ${CTEST_GCOVR_FILTER}")
            ctest_ext_note("CTEST_GCOVR_OUTPUT_BASE_NAME            : ${CTEST_GCOVR_OUTPUT_BASE_NAME}")
            ctest_ext_note("CTEST_GCOVR_XML_DIR                     : ${CTEST_GCOVR_XML_DIR}")
            ctest_ext_note("CTEST_GCOVR_HTML_DIR                    : ${CTEST_GCOVR_HTML_DIR}")
            ctest_ext_note("CTEST_GCOVR_EXTRA_OPTIONS               : ${CTEST_GCOVR_EXTRA_OPTIONS}")
        endif()
        if(CTEST_COVERAGE_TOOL MATCHES "LCOV")
            ctest_ext_note("CTEST_LCOV_EXECUTABLE                   : ${CTEST_LCOV_EXECUTABLE}")
            ctest_ext_note("CTEST_GENHTML_EXECUTABLE                : ${CTEST_GENHTML_EXECUTABLE}")
            ctest_ext_note("CTEST_LCOV_BRANCH_COVERAGE              : ${CTEST_LCOV_BRANCH_COVERAGE}")
            ctest_ext_note("CTEST_LCOV_FUNCTION_COVERAGE            : ${CTEST_LCOV_FUNCTION_COVERAGE}")
            ctest_ext_note("CTEST_LCOV_SKIP_HTML                    : ${CTEST_LCOV_SKIP_HTML}")
            ctest_ext_note("CTEST_LCOV_OUTPUT_LCOV_DIR              : ${CTEST_LCOV_OUTPUT_LCOV_DIR}")
            ctest_ext_note("CTEST_LCOV_OUTPUT_REPORT_NAME           : ${CTEST_LCOV_OUTPUT_REPORT_NAME}")
            ctest_ext_note("CTEST_LCOV_OUTPUT_HTML_DIR              : ${CTEST_LCOV_OUTPUT_HTML_DIR}")
            ctest_ext_note("CTEST_LCOV_EXTRACT                      : ${CTEST_LCOV_EXTRACT}")
            ctest_ext_note("CTEST_LCOV_REMOVE                       : ${CTEST_LCOV_REMOVE}")
            ctest_ext_note("CTEST_LCOV_EXTRA_LCOV_OPTIONS           : ${CTEST_LCOV_EXTRA_LCOV_OPTIONS}")
            ctest_ext_note("CTEST_LCOV_EXTRA_GENTHML_OPTIONS        : ${CTEST_LCOV_EXTRA_GENTHML_OPTIONS}")
        endif()
        if(CTEST_COVERAGE_TOOL MATCHES "CDASH")
            ctest_ext_note("CTEST_COVERAGE_COMMAND                  : ${CTEST_COVERAGE_COMMAND}")
            ctest_ext_note("CTEST_COVERAGE_EXTRA_FLAGS              : ${CTEST_COVERAGE_EXTRA_FLAGS}")
        endif()
    endif()
    ctest_ext_note("")

    ctest_ext_note("CTEST_WITH_DYNAMIC_ANALYSIS             : ${CTEST_WITH_DYNAMIC_ANALYSIS}")
    if(CTEST_WITH_DYNAMIC_ANALYSIS)
        ctest_ext_note("CTEST_DYNAMIC_ANALYSIS_TOOL             : ${CTEST_DYNAMIC_ANALYSIS_TOOL}")
        if(CTEST_DYNAMIC_ANALYSIS_TOOL MATCHES "CDASH")
            ctest_ext_note("CTEST_MEMORYCHECK_COMMAND               : ${CTEST_MEMORYCHECK_COMMAND}")
            ctest_ext_note("CTEST_MEMORYCHECK_SUPPRESSIONS_FILE     : ${CTEST_MEMORYCHECK_SUPPRESSIONS_FILE}")
            ctest_ext_note("CTEST_MEMORYCHECK_COMMAND_OPTIONS       : ${CTEST_MEMORYCHECK_COMMAND_OPTIONS}")
        endif()
    endif()
    ctest_ext_note("")

    ctest_ext_note("CTEST_WITH_SUBMIT                       : ${CTEST_WITH_SUBMIT}")
    if(CTEST_WITH_SUBMIT)
        ctest_ext_note("CTEST_NOTES_FILES                       : ${CTEST_NOTES_FILES}")
        ctest_ext_note("CTEST_UPLOAD_FILES                      : ${CTEST_UPLOAD_FILES}")
    endif()
    ctest_ext_note("")

    ctest_ext_note("CTEST_TRACK                             : ${CTEST_TRACK}")
    ctest_ext_note("CTEST_TMP_BASE_DIR                      : ${CTEST_TMP_BASE_DIR}")
    ctest_ext_note("CTEST_EXT_COLOR_OUTPUT                  : ${CTEST_EXT_COLOR_OUTPUT}")
    ctest_ext_note("")
endfunction()
