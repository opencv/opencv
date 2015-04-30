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
# CMake configuration commands
#

#
# add_cmake_cache_entry(<name> <value> [TYPE <type>] [FORCE])
#
#   Adds new CMake cache entry.
#

function(add_cmake_cache_entry OPTION_NAME)
    set(options "FORCE")
    set(oneValueArgs "TYPE")
    set(multiValueArgs "")
    cmake_parse_arguments(OPTION "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set_ifndef(OPTION_TYPE "STRING")

    if(CTEST_INITIAL_CACHE MATCHES "${OPTION_NAME}.*\n")
        if(NOT OPTION_FORCE)
            return()
        else()
            string(REGEX REPLACE "${OPTION_NAME}.*\n" "" CTEST_INITIAL_CACHE "${CTEST_INITIAL_CACHE}")
        endif()
    endif()

    set(CTEST_INITIAL_CACHE "${CTEST_INITIAL_CACHE}${OPTION_NAME}:${OPTION_TYPE}=${OPTION_UNPARSED_ARGUMENTS}\n" PARENT_SCOPE)
endfunction()
