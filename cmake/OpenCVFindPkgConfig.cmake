#
# OpenCV note: the file has been extracted from CMake 2.6.2 distribution.
# It is used to build OpenCV with CMake 2.4.x
#

# - a pkg-config module for CMake
#
# Usage:
#   pkg_check_modules(<PREFIX> [REQUIRED] <MODULE> [<MODULE>]*)
#     checks for all the given modules
#
#   pkg_search_module(<PREFIX> [REQUIRED] <MODULE> [<MODULE>]*)
#     checks for given modules and uses the first working one
#
# When the 'REQUIRED' argument was set, macros will fail with an error
# when module(s) could not be found
#
# It sets the following variables:
#   PKG_CONFIG_FOUND         ... true if pkg-config works on the system
#   PKG_CONFIG_EXECUTABLE    ... pathname of the pkg-config program
#   <PREFIX>_FOUND           ... set to 1 if module(s) exist
#
# For the following variables two sets of values exist; first one is the
# common one and has the given PREFIX. The second set contains flags
# which are given out when pkgconfig was called with the '--static'
# option.
#   <XPREFIX>_LIBRARIES      ... only the libraries (w/o the '-l')
#   <XPREFIX>_LIBRARY_DIRS   ... the paths of the libraries (w/o the '-L')
#   <XPREFIX>_LDFLAGS        ... all required linker flags
#   <XPREFIX>_LDFLAGS_OTHER  ... all other linker flags
#   <XPREFIX>_INCLUDE_DIRS   ... the '-I' preprocessor flags (w/o the '-I')
#   <XPREFIX>_CFLAGS         ... all required cflags
#   <XPREFIX>_CFLAGS_OTHER   ... the other compiler flags
#
#   <XPREFIX> = <PREFIX>        for common case
#   <XPREFIX> = <PREFIX>_STATIC for static linking
#
# There are some special variables whose prefix depends on the count
# of given modules. When there is only one module, <PREFIX> stays
# unchanged. When there are multiple modules, the prefix will be
# changed to <PREFIX>_<MODNAME>:
#   <XPREFIX>_VERSION    ... version of the module
#   <XPREFIX>_PREFIX     ... prefix-directory of the module
#   <XPREFIX>_INCLUDEDIR ... include-dir of the module
#   <XPREFIX>_LIBDIR     ... lib-dir of the module
#
#   <XPREFIX> = <PREFIX>  when |MODULES| == 1, else
#   <XPREFIX> = <PREFIX>_<MODNAME>
#
# A <MODULE> parameter can have the following formats:
#   {MODNAME}            ... matches any version
#   {MODNAME}>={VERSION} ... at least version <VERSION> is required
#   {MODNAME}={VERSION}  ... exactly version <VERSION> is required
#   {MODNAME}<={VERSION} ... modules must not be newer than <VERSION>
#
# Examples
#   pkg_check_modules (GLIB2   glib-2.0)
#
#   pkg_check_modules (GLIB2   glib-2.0>=2.10)
#     requires at least version 2.10 of glib2 and defines e.g.
#       GLIB2_VERSION=2.10.3
#
#   pkg_check_modules (FOO     glib-2.0>=2.10 gtk+-2.0)
#     requires both glib2 and gtk2, and defines e.g.
#       FOO_glib-2.0_VERSION=2.10.3
#       FOO_gtk+-2.0_VERSION=2.8.20
#
#   pkg_check_modules (XRENDER REQUIRED xrender)
#     defines e.g.:
#       XRENDER_LIBRARIES=Xrender;X11
#       XRENDER_STATIC_LIBRARIES=Xrender;X11;pthread;Xau;Xdmcp
#
#   pkg_search_module (BAR     libxml-2.0 libxml2 libxml>=2)


# Copyright (C) 2006 Enrico Scholz <enrico.scholz@informatik.tu-chemnitz.de>
#
# Redistribution and use, with or without modification, are permitted
# provided that the following conditions are met:
#
#    1. Redistributions must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#    2. The name of the author may not be used to endorse or promote
#       products derived from this software without specific prior
#       written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


### Common stuff ####
set(PKG_CONFIG_VERSION 1)
set(PKG_CONFIG_FOUND   0)

find_program(PKG_CONFIG_EXECUTABLE NAMES pkg-config DOC "pkg-config executable")
mark_as_advanced(PKG_CONFIG_EXECUTABLE)

if(PKG_CONFIG_EXECUTABLE)
  set(PKG_CONFIG_FOUND 1)
endif(PKG_CONFIG_EXECUTABLE)


# Unsets the given variables
macro(_pkgconfig_unset var)
  set(${var} "" CACHE INTERNAL "")
endmacro(_pkgconfig_unset)

macro(_pkgconfig_set var value)
  set(${var} ${value} CACHE INTERNAL "")
endmacro(_pkgconfig_set)

# Invokes pkgconfig, cleans up the result and sets variables
macro(_pkgconfig_invoke _pkglist _prefix _varname _regexp)
  set(_pkgconfig_invoke_result)

  execute_process(
    COMMAND ${PKG_CONFIG_EXECUTABLE} ${ARGN} ${_pkglist}
    OUTPUT_VARIABLE _pkgconfig_invoke_result
    RESULT_VARIABLE _pkgconfig_failed)

  if (_pkgconfig_failed)
    set(_pkgconfig_${_varname} "")
    _pkgconfig_unset(${_prefix}_${_varname})
  else(_pkgconfig_failed)
    string(REGEX REPLACE "[\r\n]"                  " " _pkgconfig_invoke_result "${_pkgconfig_invoke_result}")
    string(REGEX REPLACE " +$"                     ""  _pkgconfig_invoke_result "${_pkgconfig_invoke_result}")

    if (NOT ${_regexp} STREQUAL "")
      string(REGEX REPLACE "${_regexp}" " " _pkgconfig_invoke_result "${_pkgconfig_invoke_result}")
    endif(NOT ${_regexp} STREQUAL "")

    separate_arguments(_pkgconfig_invoke_result)

    #message(STATUS "  ${_varname} ... ${_pkgconfig_invoke_result}")
    set(_pkgconfig_${_varname} ${_pkgconfig_invoke_result})
    _pkgconfig_set(${_prefix}_${_varname} "${_pkgconfig_invoke_result}")
  endif(_pkgconfig_failed)
endmacro(_pkgconfig_invoke)

# Invokes pkgconfig two times; once without '--static' and once with
# '--static'
macro(_pkgconfig_invoke_dyn _pkglist _prefix _varname cleanup_regexp)
  _pkgconfig_invoke("${_pkglist}" ${_prefix}        ${_varname} "${cleanup_regexp}" ${ARGN})
  _pkgconfig_invoke("${_pkglist}" ${_prefix} STATIC_${_varname} "${cleanup_regexp}" --static  ${ARGN})
endmacro(_pkgconfig_invoke_dyn)

# Splits given arguments into options and a package list
macro(_pkgconfig_parse_options _result _is_req)
  set(${_is_req} 0)

  foreach(_pkg ${ARGN})
    if (_pkg STREQUAL "REQUIRED")
      set(${_is_req} 1)
    endif (_pkg STREQUAL "REQUIRED")
  endforeach(_pkg ${ARGN})

  set(${_result} ${ARGN})
  list(REMOVE_ITEM ${_result} "REQUIRED")
endmacro(_pkgconfig_parse_options)

###
macro(_pkg_check_modules_internal _is_required _is_silent _prefix)
  _pkgconfig_unset(${_prefix}_FOUND)
  _pkgconfig_unset(${_prefix}_VERSION)
  _pkgconfig_unset(${_prefix}_PREFIX)
  _pkgconfig_unset(${_prefix}_INCLUDEDIR)
  _pkgconfig_unset(${_prefix}_LIBDIR)
  _pkgconfig_unset(${_prefix}_LIBS)
  _pkgconfig_unset(${_prefix}_LIBS_L)
  _pkgconfig_unset(${_prefix}_LIBS_PATHS)
  _pkgconfig_unset(${_prefix}_LIBS_OTHER)
  _pkgconfig_unset(${_prefix}_CFLAGS)
  _pkgconfig_unset(${_prefix}_CFLAGS_I)
  _pkgconfig_unset(${_prefix}_CFLAGS_OTHER)
  _pkgconfig_unset(${_prefix}_STATIC_LIBDIR)
  _pkgconfig_unset(${_prefix}_STATIC_LIBS)
  _pkgconfig_unset(${_prefix}_STATIC_LIBS_L)
  _pkgconfig_unset(${_prefix}_STATIC_LIBS_PATHS)
  _pkgconfig_unset(${_prefix}_STATIC_LIBS_OTHER)
  _pkgconfig_unset(${_prefix}_STATIC_CFLAGS)
  _pkgconfig_unset(${_prefix}_STATIC_CFLAGS_I)
  _pkgconfig_unset(${_prefix}_STATIC_CFLAGS_OTHER)

  # create a better addressable variable of the modules and calculate its size
  set(_pkg_check_modules_list ${ARGN})
  list(LENGTH _pkg_check_modules_list _pkg_check_modules_cnt)

  if(PKG_CONFIG_EXECUTABLE)
    # give out status message telling checked module
    if (NOT ${_is_silent})
      if (_pkg_check_modules_cnt EQUAL 1)
        message(STATUS "checking for module '${_pkg_check_modules_list}'")
      else(_pkg_check_modules_cnt EQUAL 1)
        message(STATUS "checking for modules '${_pkg_check_modules_list}'")
      endif(_pkg_check_modules_cnt EQUAL 1)
    endif(NOT ${_is_silent})

    set(_pkg_check_modules_packages)
    set(_pkg_check_modules_failed)

    # iterate through module list and check whether they exist and match the required version
    foreach (_pkg_check_modules_pkg ${_pkg_check_modules_list})
      set(_pkg_check_modules_exist_query)

      # check whether version is given
      if (_pkg_check_modules_pkg MATCHES ".*(>=|=|<=).*")
        string(REGEX REPLACE "(.*[^><])(>=|=|<=)(.*)" "\\1" _pkg_check_modules_pkg_name "${_pkg_check_modules_pkg}")
        string(REGEX REPLACE "(.*[^><])(>=|=|<=)(.*)" "\\2" _pkg_check_modules_pkg_op   "${_pkg_check_modules_pkg}")
        string(REGEX REPLACE "(.*[^><])(>=|=|<=)(.*)" "\\3" _pkg_check_modules_pkg_ver  "${_pkg_check_modules_pkg}")
      else(_pkg_check_modules_pkg MATCHES ".*(>=|=|<=).*")
        set(_pkg_check_modules_pkg_name "${_pkg_check_modules_pkg}")
        set(_pkg_check_modules_pkg_op)
        set(_pkg_check_modules_pkg_ver)
      endif(_pkg_check_modules_pkg MATCHES ".*(>=|=|<=).*")

      # handle the operands
      if (_pkg_check_modules_pkg_op STREQUAL ">=")
        list(APPEND _pkg_check_modules_exist_query --atleast-version)
      endif(_pkg_check_modules_pkg_op STREQUAL ">=")

      if (_pkg_check_modules_pkg_op STREQUAL "=")
        list(APPEND _pkg_check_modules_exist_query --exact-version)
      endif(_pkg_check_modules_pkg_op STREQUAL "=")

      if (_pkg_check_modules_pkg_op STREQUAL "<=")
        list(APPEND _pkg_check_modules_exist_query --max-version)
      endif(_pkg_check_modules_pkg_op STREQUAL "<=")

      # create the final query which is of the format:
      # * --atleast-version <version> <pkg-name>
      # * --exact-version <version> <pkg-name>
      # * --max-version <version> <pkg-name>
      # * --exists <pkg-name>
      if (_pkg_check_modules_pkg_op)
        list(APPEND _pkg_check_modules_exist_query "${_pkg_check_modules_pkg_ver}")
      else(_pkg_check_modules_pkg_op)
        list(APPEND _pkg_check_modules_exist_query --exists)
      endif(_pkg_check_modules_pkg_op)

      _pkgconfig_unset(${_prefix}_${_pkg_check_modules_pkg_name}_VERSION)
      _pkgconfig_unset(${_prefix}_${_pkg_check_modules_pkg_name}_PREFIX)
      _pkgconfig_unset(${_prefix}_${_pkg_check_modules_pkg_name}_INCLUDEDIR)
      _pkgconfig_unset(${_prefix}_${_pkg_check_modules_pkg_name}_LIBDIR)

      list(APPEND _pkg_check_modules_exist_query "${_pkg_check_modules_pkg_name}")
      list(APPEND _pkg_check_modules_packages    "${_pkg_check_modules_pkg_name}")

      # execute the query
      execute_process(
        COMMAND ${PKG_CONFIG_EXECUTABLE} ${_pkg_check_modules_exist_query}
        RESULT_VARIABLE _pkgconfig_retval)

      # evaluate result and tell failures
      if (_pkgconfig_retval)
        if(NOT ${_is_silent})
          message(STATUS "  package '${_pkg_check_modules_pkg}' not found")
        endif(NOT ${_is_silent})

        set(_pkg_check_modules_failed 1)
      endif(_pkgconfig_retval)
    endforeach(_pkg_check_modules_pkg)

    if(_pkg_check_modules_failed)
      # fail when requested
      if (${_is_required})
        message(SEND_ERROR "A required package was not found")
      endif (${_is_required})
    else(_pkg_check_modules_failed)
      # when we are here, we checked whether requested modules
      # exist. Now, go through them and set variables

      _pkgconfig_set(${_prefix}_FOUND 1)
      list(LENGTH _pkg_check_modules_packages pkg_count)

      # iterate through all modules again and set individual variables
      foreach (_pkg_check_modules_pkg ${_pkg_check_modules_packages})
        # handle case when there is only one package required
        if (pkg_count EQUAL 1)
          set(_pkg_check_prefix "${_prefix}")
        else(pkg_count EQUAL 1)
          set(_pkg_check_prefix "${_prefix}_${_pkg_check_modules_pkg}")
        endif(pkg_count EQUAL 1)

        _pkgconfig_invoke(${_pkg_check_modules_pkg} "${_pkg_check_prefix}" VERSION    ""   --modversion )
        _pkgconfig_invoke(${_pkg_check_modules_pkg} "${_pkg_check_prefix}" PREFIX     ""   --variable=prefix )
        _pkgconfig_invoke(${_pkg_check_modules_pkg} "${_pkg_check_prefix}" INCLUDEDIR ""   --variable=includedir )
        _pkgconfig_invoke(${_pkg_check_modules_pkg} "${_pkg_check_prefix}" LIBDIR     ""   --variable=libdir )

        message(STATUS "  found ${_pkg_check_modules_pkg}, version ${_pkgconfig_VERSION}")
      endforeach(_pkg_check_modules_pkg)

      # set variables which are combined for multiple modules
      _pkgconfig_invoke_dyn("${_pkg_check_modules_packages}" "${_prefix}" LIBRARIES           "(^| )-l" --libs-only-l )
      _pkgconfig_invoke_dyn("${_pkg_check_modules_packages}" "${_prefix}" LIBRARY_DIRS        "(^| )-L" --libs-only-L )
      _pkgconfig_invoke_dyn("${_pkg_check_modules_packages}" "${_prefix}" LDFLAGS             ""        --libs )
      _pkgconfig_invoke_dyn("${_pkg_check_modules_packages}" "${_prefix}" LDFLAGS_OTHER       ""        --libs-only-other )

      _pkgconfig_invoke_dyn("${_pkg_check_modules_packages}" "${_prefix}" INCLUDE_DIRS        "(^| )-I" --cflags-only-I )
      _pkgconfig_invoke_dyn("${_pkg_check_modules_packages}" "${_prefix}" CFLAGS              ""        --cflags )
      _pkgconfig_invoke_dyn("${_pkg_check_modules_packages}" "${_prefix}" CFLAGS_OTHER        ""        --cflags-only-other )
    endif(_pkg_check_modules_failed)
  else(PKG_CONFIG_EXECUTABLE)
    if (${_is_required})
      message(SEND_ERROR "pkg-config tool not found")
    endif (${_is_required})
  endif(PKG_CONFIG_EXECUTABLE)
endmacro(_pkg_check_modules_internal)

###
### User visible macros start here
###

###
macro(pkg_check_modules _prefix _module0)
  # check cached value
  if (NOT DEFINED __pkg_config_checked_${_prefix} OR __pkg_config_checked_${_prefix} LESS ${PKG_CONFIG_VERSION} OR NOT ${_prefix}_FOUND)
    _pkgconfig_parse_options   (_pkg_modules _pkg_is_required "${_module0}" ${ARGN})
    _pkg_check_modules_internal("${_pkg_is_required}" 0 "${_prefix}" ${_pkg_modules})

    _pkgconfig_set(__pkg_config_checked_${_prefix} ${PKG_CONFIG_VERSION})
  endif(NOT DEFINED __pkg_config_checked_${_prefix} OR __pkg_config_checked_${_prefix} LESS ${PKG_CONFIG_VERSION} OR NOT ${_prefix}_FOUND)
endmacro(pkg_check_modules)

###
macro(pkg_search_module _prefix _module0)
  # check cached value
  if (NOT DEFINED __pkg_config_checked_${_prefix} OR __pkg_config_checked_${_prefix} LESS ${PKG_CONFIG_VERSION} OR NOT ${_prefix}_FOUND)
    set(_pkg_modules_found 0)
    _pkgconfig_parse_options(_pkg_modules_alt _pkg_is_required "${_module0}" ${ARGN})

    message(STATUS "checking for one of the modules '${_pkg_modules_alt}'")

    # iterate through all modules and stop at the first working one.
    foreach(_pkg_alt ${_pkg_modules_alt})
      if(NOT _pkg_modules_found)
        _pkg_check_modules_internal(0 1 "${_prefix}" "${_pkg_alt}")
      endif(NOT _pkg_modules_found)

      if (${_prefix}_FOUND)
        set(_pkg_modules_found 1)
      endif(${_prefix}_FOUND)
    endforeach(_pkg_alt)

    if (NOT ${_prefix}_FOUND)
      if(${_pkg_is_required})
        message(SEND_ERROR "None of the required '${_pkg_modules_alt}' found")
      endif(${_pkg_is_required})
    endif(NOT ${_prefix}_FOUND)

    _pkgconfig_set(__pkg_config_checked_${_prefix} ${PKG_CONFIG_VERSION})
  endif(NOT DEFINED __pkg_config_checked_${_prefix} OR __pkg_config_checked_${_prefix} LESS ${PKG_CONFIG_VERSION} OR NOT ${_prefix}_FOUND)
endmacro(pkg_search_module)

### Local Variables:
### mode: cmake
### End:
