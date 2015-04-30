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
# System commands
#

#
# create_tmp_dir(<output_variable> [BASE_DIR <path to base temp directory>])
#
#   Creates temporary directory and returns path to it via `<output_variable>`.
#
#   `BASE_DIR` can be used to specify location for base temporary path,
#   if it is not defined `TEMP`, `TMP` or `TMPDIR` environment variables will be used.
#

function(create_tmp_dir OUT_VAR)
    set(options "")
    set(oneValueArgs "BASE_DIR")
    set(multiValueArgs "")
    cmake_parse_arguments(TMP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    override_from_ctest_vars(TMP_BASE_DIR)

    if(NOT DEFINED TMP_BASE_DIR)
        foreach(dir "$ENV{TEMP}" "$ENV{TMP}" "$ENV{TMPDIR}" "/tmp")
            if (EXISTS "${dir}")
                set(TMP_BASE_DIR "${dir}")
            endif()
        endforeach()
    endif()

    check_vars_exist(TMP_BASE_DIR)

    # TODO : find better way to avoid collisions.
    string(RANDOM rand_name)
    while(EXISTS "${TMP_BASE_DIR}/${rand_name}")
        string(RANDOM rand_name)
    endwhile()

    set(tmp_dir "${TMP_BASE_DIR}/${rand_name}")

    ctest_ext_info("Creating temporary directory : ${tmp_dir}")
    file(MAKE_DIRECTORY "${tmp_dir}")

    set(${OUT_VAR} "${tmp_dir}" PARENT_SCOPE)
endfunction()
