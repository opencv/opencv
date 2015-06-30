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
# Variables management commands
#

#
# set_ifndef(<variable> <value>)
#
#   Sets `<variable>` to the `<value>`, only if the `<variable>` is not defined.
#

function(set_ifndef VAR)
    if(NOT DEFINED ${VAR})
        set(${VAR} "${ARGN}" PARENT_SCOPE)
    endif()
endfunction()

#
# set_from_env(<variable1> <variable2> ...)
#
#   Sets `<variableN>` to the value of environment variable with the same name,
#   only if the `<variableN>` is not defined and the environment variable is defined.
#

function(set_from_env)
    foreach(var ${ARGN})
        if(NOT DEFINED ${var} AND DEFINED ENV{${var}})
            set(${var} "$ENV{${var}}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()

#
# override_from_ctest_vars(<variable1> <variable2> ...)
#
#   Overrides all variables from `CTEST_<var_name>` values, if they are defined.
#

function(override_from_ctest_vars)
    foreach(var ${ARGN})
        if(DEFINED CTEST_${var})
            set(${var} "${CTEST_${var}}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()

#
# check_vars_def(<variable1> <variable2> ...)
#
#   Checks that all variables are defined.
#

function(check_vars_def)
    foreach(var ${ARGN})
        if(NOT DEFINED ${var})
            message(FATAL_ERROR "${var} is not defined")
        endif()
    endforeach()
endfunction()

#
# check_vars_exist(<variable1> <variable2> ...)
#
#   Checks that all variables are defined and point to existed file/directory.
#

function(check_vars_exist)
    check_vars_def(${ARGN})

    foreach(var ${ARGN})
        if(NOT EXISTS "${${var}}")
            message(FATAL_ERROR "${var} = ${${var}} is not exist")
        endif()
    endforeach()
endfunction()

#
# check_if_matches(<variable> <regexp1> <regexp2> ...)
#
#   Checks that `<variable>` matches one of the regular expression from the input list.
#

function(check_if_matches VAR)
    check_vars_def(${VAR})

    set(found FALSE)
    foreach(regexp ${ARGN})
        if(${VAR} MATCHES "${regexp}")
            set(found TRUE)
            break()
        endif()
    endforeach()

    if(NOT found)
        message(FATAL_ERROR "${VAR}=${${VAR}} must match one from ${ARGN} list")
    endif()
endfunction()

#
# list_filter_out(<list> <regexp1> <regexp2> ...)
#
#   Filter out all items in the `<list>`, which match one of the regular expression from the input list.
#

function(list_filter_out LST_VAR)
    set(tmp_lst ${${LST_VAR}})

    foreach(item ${tmp_lst})
        foreach(regex ${ARGN})
            if(item MATCHES "${regex}")
                list(REMOVE_ITEM tmp_lst "${item}")
            endif()
        endforeach()
    endforeach()

    set(${LST_VAR} "${tmp_lst}" PARENT_SCOPE)
endfunction()
