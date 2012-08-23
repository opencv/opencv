# CMAKE_PARSE_ARGUMENTS(<prefix> <options> <one_value_keywords> <multi_value_keywords> args...)
#
# CMAKE_PARSE_ARGUMENTS() is intended to be used in macros or functions for
# parsing the arguments given to that macro or function.
# It processes the arguments and defines a set of variables which hold the
# values of the respective options.
#
# The <options> argument contains all options for the respective macro,
# i.e. keywords which can be used when calling the macro without any value
# following, like e.g. the OPTIONAL keyword of the install() command.
#
# The <one_value_keywords> argument contains all keywords for this macro
# which are followed by one value, like e.g. DESTINATION keyword of the
# install() command.
#
# The <multi_value_keywords> argument contains all keywords for this macro
# which can be followed by more than one value, like e.g. the TARGETS or
# FILES keywords of the install() command.
#
# When done, CMAKE_PARSE_ARGUMENTS() will have defined for each of the
# keywords listed in <options>, <one_value_keywords> and
# <multi_value_keywords> a variable composed of the given <prefix>
# followed by "_" and the name of the respective keyword.
# These variables will then hold the respective value from the argument list.
# For the <options> keywords this will be TRUE or FALSE.
#
# All remaining arguments are collected in a variable
# <prefix>_UNPARSED_ARGUMENTS, this can be checked afterwards to see whether
# your macro was called with unrecognized parameters.
#
# As an example here a my_install() macro, which takes similar arguments as the
# real install() command:
#
#   function(MY_INSTALL)
#     set(options OPTIONAL FAST)
#     set(oneValueArgs DESTINATION RENAME)
#     set(multiValueArgs TARGETS CONFIGURATIONS)
#     cmake_parse_arguments(MY_INSTALL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
#     ...
#
# Assume my_install() has been called like this:
#   my_install(TARGETS foo bar DESTINATION bin OPTIONAL blub)
#
# After the cmake_parse_arguments() call the macro will have set the following
# variables:
#   MY_INSTALL_OPTIONAL = TRUE
#   MY_INSTALL_FAST = FALSE (this option was not used when calling my_install()
#   MY_INSTALL_DESTINATION = "bin"
#   MY_INSTALL_RENAME = "" (was not used)
#   MY_INSTALL_TARGETS = "foo;bar"
#   MY_INSTALL_CONFIGURATIONS = "" (was not used)
#   MY_INSTALL_UNPARSED_ARGUMENTS = "blub" (no value expected after "OPTIONAL"
#
# You can the continue and process these variables.
#
# Keywords terminate lists of values, e.g. if directly after a one_value_keyword
# another recognized keyword follows, this is interpreted as the beginning of
# the new option.
# E.g. my_install(TARGETS foo DESTINATION OPTIONAL) would result in
# MY_INSTALL_DESTINATION set to "OPTIONAL", but MY_INSTALL_DESTINATION would
# be empty and MY_INSTALL_OPTIONAL would be set to TRUE therefor.

#=============================================================================
# Copyright 2010 Alexander Neundorf <neundorf@kde.org>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)


if(__CMAKE_PARSE_ARGUMENTS_INCLUDED)
  return()
endif()
set(__CMAKE_PARSE_ARGUMENTS_INCLUDED TRUE)


function(CMAKE_PARSE_ARGUMENTS prefix _optionNames _singleArgNames _multiArgNames)
  # first set all result variables to empty/FALSE
  foreach(arg_name ${_singleArgNames} ${_multiArgNames})
    set(${prefix}_${arg_name})
  endforeach(arg_name)

  foreach(option ${_optionNames})
    set(${prefix}_${option} FALSE)
  endforeach(option)

  set(${prefix}_UNPARSED_ARGUMENTS)

  set(insideValues FALSE)
  set(currentArgName)

  # now iterate over all arguments and fill the result variables
  foreach(currentArg ${ARGN})
    list(FIND _optionNames "${currentArg}" optionIndex)  # ... then this marks the end of the arguments belonging to this keyword
    list(FIND _singleArgNames "${currentArg}" singleArgIndex)  # ... then this marks the end of the arguments belonging to this keyword
    list(FIND _multiArgNames "${currentArg}" multiArgIndex)  # ... then this marks the end of the arguments belonging to this keyword

    if(${optionIndex} EQUAL -1  AND  ${singleArgIndex} EQUAL -1  AND  ${multiArgIndex} EQUAL -1)
      if(insideValues)
        if("${insideValues}" STREQUAL "SINGLE")
          set(${prefix}_${currentArgName} ${currentArg})
          set(insideValues FALSE)
        elseif("${insideValues}" STREQUAL "MULTI")
          list(APPEND ${prefix}_${currentArgName} ${currentArg})
        endif()
      else(insideValues)
        list(APPEND ${prefix}_UNPARSED_ARGUMENTS ${currentArg})
      endif(insideValues)
    else()
      if(NOT ${optionIndex} EQUAL -1)
        set(${prefix}_${currentArg} TRUE)
        set(insideValues FALSE)
      elseif(NOT ${singleArgIndex} EQUAL -1)
        set(currentArgName ${currentArg})
        set(${prefix}_${currentArgName})
        set(insideValues "SINGLE")
      elseif(NOT ${multiArgIndex} EQUAL -1)
        set(currentArgName ${currentArg})
        set(${prefix}_${currentArgName})
        set(insideValues "MULTI")
      endif()
    endif()

  endforeach(currentArg)

  # propagate the result variables to the caller:
  foreach(arg_name ${_singleArgNames} ${_multiArgNames} ${_optionNames})
    set(${prefix}_${arg_name}  ${${prefix}_${arg_name}} PARENT_SCOPE)
  endforeach(arg_name)
  set(${prefix}_UNPARSED_ARGUMENTS ${${prefix}_UNPARSED_ARGUMENTS} PARENT_SCOPE)

endfunction(CMAKE_PARSE_ARGUMENTS _options _singleArgs _multiArgs)
